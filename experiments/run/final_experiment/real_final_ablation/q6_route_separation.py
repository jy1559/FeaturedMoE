#!/usr/bin/env python3
"""Q6: route_consistency + route_separation joint hyperopt + diag rerun (Phase-1 + Phase-2).

Phase-1  Train 6 datasets × top-6 base configs, 8 GPUs.
         Jointly tunes three variables per trial:
           - learning_rate          loguniform [base×0.8, base×1.25]  (tighter than default)
           - route_consistency_lambda  loguniform_zero [0, 1e-4..1e-2]
           - route_separation_lambda   loguniform_zero [0, 1e-4..1e-2]
         z_loss_lambda inherited from base config (fixed).
         group_top_k=3, expert_top_k=2 fixed. epoch=100, patience=10.

Phase-2  After Phase-1, reads q6 summary, picks top-K best results per dataset,
         reruns with full diagnostics (fmoe_diag_logging, special_logging, checkpoint
         export) then runs case-eval. Results available for the notebook
         writing/260419_real_final_exp/05_q5_behavior_semantics.ipynb.
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    REPO_ROOT,
    QUESTION_AXIS,
    append_csv_row,
    build_route_command,
    build_route_row,
    build_summary_row,
    build_train_rows,
    canonical_stage_maps,
    common_arg_parser,
    extract_error_tail,
    find_completed_case_eval_row,
    has_run_status_end_normal,
    index_path,
    load_base_candidates,
    load_result_payload,
    log_path_for_row,
    parse_csv_ints,
    parse_csv_list,
    parse_result_path_from_log,
    python_bin,
    read_json,
    read_summary_rows,
    result_has_successful_trials,
    run_case_eval_pipeline,
    run_jobs,
    selected_candidates_from_args,
    validate_session_fixed_files,
    write_index_rows,
    write_manifest,
    DEFAULT_BASE_CSV,
    DEFAULT_DATASETS,
    ROUTE_MODEL,
    SUMMARY_FIELDS,
    BaseCandidate,
    build_search_entries,
)

QUESTION_AXIS["q6"] = "q6_route_separation"
QUESTION_AXIS["q6_diag"] = "q6_route_separation_diag"

# Lambda search range shared by both consistency and separation.
# loguniform_zero means hyperopt samples 0 OR loguniform in [LO, HI].
LAMBDA_LO = 1e-4
LAMBDA_HI = 1e-2

ALL_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "beauty",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
]


# ---------------------------------------------------------------------------
# Settings — single variant; lambda values tuned by hyperopt per trial
# ---------------------------------------------------------------------------

def q6_settings() -> list[dict]:
    full = canonical_stage_maps()
    return [{
        "setting_key": "consistency_separation_joint",
        "setting_label": "Consistency+Separation Joint Tuning",
        "variant_label": "joint_lambda",
        "variant_group": "route_separation_sweep",
        "panel_family": "route_separation",
        "variant_order": 0,
        "overrides": {
            **deepcopy(full),
            # Both lambdas removed from fixed context — added to search_space below.
            "topk_scope_mode": "per_group",
            "group_top_k": 3,
            "expert_top_k": 2,
            "moe_top_k": 0,
        },
    }]


def _tighter_lr_spec(base_lr: float) -> tuple[list[float], str]:
    """LR range ×0.8 ~ ×1.25 (tighter than the default ×0.67~×1.5)."""
    lr = float(base_lr if base_lr and base_lr > 0 else 1e-3)
    lo = max(lr * 0.80, 1e-6)
    hi = max(lr * 1.25, lo * 1.02)
    return [lo, hi], "loguniform"


def _build_q6_rows(
    *,
    question: str,
    candidates: list[BaseCandidate],
    seeds: list[int],
    max_evals: int,
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
    smoke_test: bool,
    smoke_max_runs: int,
) -> list[dict]:
    """Build rows like build_train_rows but injects lambda search spaces."""
    from common import sanitize_token, QUESTION_AXIS as _QA, now_utc

    settings = q6_settings()
    rows: list[dict] = []
    cursor = 0
    for candidate in candidates:
        for setting in settings:
            for seed in seeds:
                cursor += 1
                row = build_route_row(
                    question=question,
                    candidate=candidate,
                    setting=setting,
                    seed=seed,
                    runtime_seed=960000 + cursor,
                    max_evals=max_evals,
                    max_run_hours=max_run_hours,
                    tune_epochs=tune_epochs,
                    tune_patience=tune_patience,
                    lr_mode="fixed",  # we set LR manually below
                )
                # Replace LR search with tighter range
                base_lr = float(candidate.base_config.get("learning_rate") or 1e-3)
                lr_vals, lr_type = _tighter_lr_spec(base_lr)
                row["search_space"]["learning_rate"] = lr_vals
                row["search_space_types"]["learning_rate"] = lr_type

                # Remove lambdas from fixed_context so they're not silently locked
                row["fixed_context"].pop("route_consistency_lambda", None)
                row["fixed_context"].pop("route_separation_lambda", None)

                # Add both lambdas to search space as loguniform_zero
                # [LO, HI] with type loguniform_zero → hyperopt samples 0 or loguniform(LO,HI)
                row["search_space"]["route_consistency_lambda"] = [LAMBDA_LO, LAMBDA_HI]
                row["search_space_types"]["route_consistency_lambda"] = "loguniform_zero"
                row["search_space"]["route_separation_lambda"] = [LAMBDA_LO, LAMBDA_HI]
                row["search_space_types"]["route_separation_lambda"] = "loguniform_zero"

                rows.append(row)
    if smoke_test:
        rows = rows[: max(1, int(smoke_max_runs))]
    return rows


# ---------------------------------------------------------------------------
# Phase-2 helpers
# ---------------------------------------------------------------------------

def _pick_best_per_dataset(summary_rows: list[dict], top_k: int) -> list[dict]:
    by_dataset: dict[str, list[dict]] = {}
    for row in summary_rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        ds = str(row.get("dataset", "")).strip()
        if ds:
            by_dataset.setdefault(ds, []).append(row)
    picked: list[dict] = []
    for rows_ds in by_dataset.values():
        rows_sorted = sorted(rows_ds, key=lambda r: float(r.get("test_score") or 0.0), reverse=True)
        picked.extend(rows_sorted[:top_k])
    return picked


def _load_phase1_best_params(source_summary: dict) -> dict[str, float]:
    src_path = str(source_summary.get("result_path", "") or "").strip()
    if not src_path:
        raise ValueError("missing result_path")
    payload = read_json(Path(src_path))
    best_params = payload.get("best_params") or {}
    if not isinstance(best_params, dict) or not best_params:
        raise ValueError(f"missing best_params in {src_path}")

    out: dict[str, float] = {}
    for key in ("learning_rate", "route_consistency_lambda", "route_separation_lambda"):
        value = best_params.get(key)
        if value is None:
            continue
        out[key] = float(value)
    if "route_separation_lambda" not in out:
        raise ValueError(f"missing route_separation_lambda in {src_path}")
    return out


def _build_diag_row(
    *,
    source_summary: dict,
    candidate,
    tuned_params: dict[str, float],
    seed: int,
    runtime_seed: int,
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
) -> dict:
    full = canonical_stage_maps()
    sep_lambda = float(tuned_params.get("route_separation_lambda") or 0.0)
    if sep_lambda == 0.0:
        sk = "sep_0_diag"
    else:
        sk = f"sep_{sep_lambda:.0e}_diag".replace("e-0", "e-").replace("e+0", "e+")
    setting = {
        "setting_key": sk,
        "setting_label": f"separation={sep_lambda:.0e} [diag]",
        "variant_label": "sep_diag",
        "variant_group": "route_separation_diag",
        "panel_family": "route_separation",
        "variant_order": 99,
        "overrides": {
            **deepcopy(full),
            "route_separation_lambda": sep_lambda,
            "topk_scope_mode": "per_group",
            "group_top_k": 3,
            "expert_top_k": 2,
            "moe_top_k": 0,
        },
    }
    row = build_route_row(
        question="q6_diag",
        candidate=candidate,
        setting=setting,
        seed=seed,
        runtime_seed=runtime_seed,
        max_evals=1,
        max_run_hours=max_run_hours,
        tune_epochs=tune_epochs,
        tune_patience=tune_patience,
        lr_mode="fixed",
    )
    row["fixed_context"]["route_separation_lambda"] = sep_lambda
    if "route_consistency_lambda" in tuned_params:
        row["fixed_context"]["route_consistency_lambda"] = float(tuned_params["route_consistency_lambda"])
    if "learning_rate" in tuned_params:
        row["fixed_context"]["learning_rate"] = float(tuned_params["learning_rate"])
    row["search_space"] = {}
    row["search_space_types"] = {}
    row["job_id"] += "_DIAG"
    row["run_phase"] = row["job_id"]
    return row


def _diag_extra_args() -> list[str]:
    return [
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_family_ablation_logging=true",
        "fmoe_best_only_logging=true",
        "++artifact_export_final_checkpoint=true",
        "fmoe_eval_logging_timing=final_only",
    ]


def _run_diag_job(row: dict, gpu_id: str) -> dict:
    log_path = log_path_for_row("q6_diag", row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_route_command(row, gpu_id, search_algo="tpe")
    cmd.extend(_diag_extra_args())
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    print(f"[q6_diag][gpu={gpu_id}] START {row['job_id']} dataset={row['dataset']}", flush=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        import subprocess
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT / "experiments"), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
        proc.wait()
        rc = proc.returncode
    elapsed = time.time() - start
    result_path_obj = parse_result_path_from_log(log_path)
    payload = load_result_payload(result_path_obj) if result_path_obj else {}
    success = result_has_successful_trials(payload)
    normal_end = has_run_status_end_normal(log_path)
    status = "ok" if (rc == 0 and success and (normal_end or result_path_obj)) else "fail"
    error = "" if status == "ok" else f"rc={rc} tail={extract_error_tail(log_path)}"
    summary = build_summary_row(
        row,
        gpu_id=gpu_id,
        status=status,
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=elapsed,
        error=error,
    )
    print(f"[q6_diag][gpu={gpu_id}] END {row['job_id']} status={status}", flush=True)
    return summary


def _run_phase2(
    *,
    datasets: list[str],
    gpus: list[str],
    top_k_diag: int,
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
    skip_case_eval: bool,
    base_csv: Path,
    dry_run: bool,
) -> int:
    q6_summary = read_summary_rows("q6")
    q6_summary = [r for r in q6_summary if str(r.get("dataset", "")).strip() in set(datasets)]
    best_rows = _pick_best_per_dataset(q6_summary, top_k=top_k_diag)

    if not best_rows:
        print("[q6_diag] No completed q6 rows — skipping Phase-2.", flush=True)
        return 0

    print(f"[q6_diag] Phase-2: {len(best_rows)} configs selected for diag rerun:", flush=True)
    for r in best_rows:
        score_str = f"{float(r.get('test_score') or 0.0):.4f}"
        print(f"  dataset={r['dataset']} setting={r.get('setting_key','')} test={score_str}", flush=True)

    if dry_run:
        print("[q6_diag] dry-run: skipping actual rerun.", flush=True)
        return 0

    cursor = itertools.count(7_000_000)
    diag_jobs: list[tuple[dict, dict]] = []

    for src in best_rows:
        ds = str(src.get("dataset", "")).strip()
        base_rank = int(src.get("base_rank", 1) or 1)
        seed = int(src.get("seed_id", 1) or 1)
        try:
            candidates = load_base_candidates(base_csv, datasets=[ds], models=[ROUTE_MODEL], top_k_configs=base_rank)
            candidate = next(c for c in candidates if c.rank == base_rank)
        except Exception as exc:
            print(f"[q6_diag] WARN: cannot load candidate ds={ds} rank={base_rank}: {exc}", flush=True)
            continue
        try:
            tuned_params = _load_phase1_best_params(src)
        except Exception as exc:
            print(f"[q6_diag] WARN: cannot load tuned params ds={ds} rank={base_rank}: {exc}", flush=True)
            continue
        row = _build_diag_row(
            source_summary=src,
            candidate=candidate,
            tuned_params=tuned_params,
            seed=seed,
            runtime_seed=next(cursor),
            max_run_hours=max_run_hours,
            tune_epochs=tune_epochs,
            tune_patience=tune_patience,
        )
        diag_jobs.append((row, src))

    diag_summary_path = LOG_ROOT / "q6_diag" / "summary.csv"
    diag_summary_path.parent.mkdir(parents=True, exist_ok=True)

    pending: Queue[tuple[dict, dict]] = Queue()
    for item in diag_jobs:
        pending.put(item)
    gpu_queue: Queue[str] = Queue()
    for g in gpus:
        gpu_queue.put(g)

    completed: list[tuple[dict, dict]] = []
    lock = threading.Lock()

    def _worker() -> None:
        while True:
            try:
                row, src = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = _run_diag_job(row, gpu_id)
                with lock:
                    completed.append((summary, src))
                    append_csv_row(diag_summary_path, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not skip_case_eval:
        case_rows: list[dict] = []
        for summary, src in completed:
            if str(summary.get("status", "")).lower() != "ok":
                print(f"[q6_diag] SKIP case-eval (failed): {summary.get('job_id','')}", flush=True)
                continue
            existing = find_completed_case_eval_row("q6_diag", summary)
            if existing is not None:
                case_rows.append(existing)
                continue
            try:
                bundle = run_case_eval_pipeline(
                    question="q6_diag",
                    source_summary_row=summary,
                    output_root=LOG_ROOT / "q6_diag" / "case_eval" / str(summary.get("job_id", "")),
                    skip_by_group=False,
                )
                case_rows.append(bundle)
                print(f"[q6_diag] case-eval OK: {summary.get('dataset')} {summary.get('setting_key')}", flush=True)
            except Exception as exc:
                print(f"[q6_diag] WARN case-eval failed: {exc}", flush=True)
                case_rows.append({
                    "question": "q6_diag",
                    "dataset": summary.get("dataset", ""),
                    "setting_key": summary.get("setting_key", ""),
                    "base_rank": summary.get("base_rank", ""),
                    "seed_id": summary.get("seed_id", ""),
                    "result_path": summary.get("result_path", ""),
                    "checkpoint_file": summary.get("checkpoint_file", ""),
                    "status": "error",
                    "error": str(exc),
                })
        if case_rows:
            write_index_rows(index_path("q6_diag", "q6_diag_case_eval_index.csv"), case_rows)

    ok = sum(1 for s, _ in completed if str(s.get("status", "")).lower() == "ok")
    print(f"[q6_diag] Phase-2 done: {ok}/{len(diag_jobs)} succeeded.", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = common_arg_parser("Q6 route_separation sweep + diag rerun", question="q6")
    parser.add_argument(
        "--skip-phase2", action="store_true",
        help="Only run Phase-1 (sweep), skip Phase-2 diag rerun.",
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Skip Phase-1 and run only Phase-2 diag rerun (Phase-1 must have completed).",
    )
    parser.add_argument(
        "--skip-case-eval", action="store_true",
        help="In Phase-2, skip case-eval (only rerun training with diag logging).",
    )
    parser.add_argument(
        "--top-k-diag", type=int, default=2,
        help="How many top results per dataset to rerun with full diagnostics in Phase-2.",
    )
    args = parser.parse_args()

    datasets = parse_csv_list(args.datasets) or list(ALL_DATASETS)
    gpus = [g for g in str(args.gpus).split(",") if g.strip()]
    seeds = parse_csv_ints(args.seeds) or [1]
    base_csv = Path(args.base_csv).expanduser().resolve()

    phase2_kwargs = dict(
        datasets=datasets,
        gpus=gpus,
        top_k_diag=args.top_k_diag,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        skip_case_eval=bool(args.skip_case_eval),
        base_csv=base_csv,
        dry_run=bool(args.dry_run),
    )

    # ── Phase-2 only ────────────────────────────────────────────────────────
    if args.phase2_only:
        print("[q6] --phase2-only: skipping Phase-1.", flush=True)
        return _run_phase2(**phase2_kwargs)

    # ── Phase-1: sweep ──────────────────────────────────────────────────────
    candidates = selected_candidates_from_args(args)
    rows = _build_q6_rows(
        question="q6",
        candidates=candidates,
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("q6", rows)
    print(f"[q6] Phase-1 manifest -> {manifest}  ({len(rows)} jobs)", flush=True)

    rc = run_jobs(
        rows,
        question="q6",
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run or args.skip_phase2:
        return rc

    # ── Phase-2: diag rerun ─────────────────────────────────────────────────
    print("\n[q6] Phase-1 complete. Starting Phase-2 (diag rerun)...", flush=True)
    return _run_phase2(**phase2_kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
