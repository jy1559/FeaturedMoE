#!/usr/bin/env python3
"""Separation-centric RouteRec tuning for KuaiRec and Foursquare.

Phase-1 searches only consistency / separation on 4 anchors per dataset while
keeping the rest of the hyperparameters fixed. Phase-2 automatically reruns two
selected configs per dataset with full diagnostics, runs case-eval, and exports
q5-style weight figures.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue

CODE_DIR = Path(__file__).resolve().parent
REAL_FINAL_DIR = CODE_DIR / "real_final_ablation"
if str(REAL_FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_FINAL_DIR))

from common import (  # noqa: E402
    BaseCandidate,
    LOG_ROOT,
    QUESTION_AXIS,
    REPO_ROOT,
    SUMMARY_FIELDS,
    append_csv_row,
    build_route_command,
    build_route_row,
    build_summary_row,
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
    validate_session_fixed_files,
    write_index_rows,
    write_manifest,
)


QUESTION = "sep_main_quick_p5_fix"
DIAG_QUESTION = "sep_main_quick_p5_fix_diag"
QUESTION_AXIS[QUESTION] = "separation_main_tuning_quick_p5_fix"
QUESTION_AXIS[DIAG_QUESTION] = "separation_main_diag_quick_p5_fix"

DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2", "foursquare"]

LAMBDA_LO = 1e-4
LAMBDA_HI = 1e-2
HIGH_SEP_MIN = 1.0e-2
HIGH_SEP_SCORE_RATIO = 0.985

ZERO_AUX_KEYS = (
    "balance_loss_lambda",
    "z_loss_lambda",
    "gate_entropy_lambda",
    "gate_entropy_until",
    "rule_agreement_lambda",
    "group_coverage_lambda",
    "group_prior_align_lambda",
    "feature_group_bias_lambda",
    "factored_group_balance_lambda",
    "route_smoothness_lambda",
    "route_sharpness_lambda",
    "route_monopoly_lambda",
    "route_monopoly_tau",
    "route_prior_lambda",
    "fmoe_v2_stage_merge_aux_lambda_scale",
)

ANCHOR_SOURCE_RANKS = (1, 2)

MANUAL_ANCHOR_OVERRIDES = {
    "KuaiRecLargeStrictPosV2_0.2": {
        "general": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 256,
            "embedding_size": 256,
            "inner_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 128,
            "d_expert_hidden": 256,
            "expert_scale": 4,
            "hidden_dropout_prob": 0.10,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.0,
            "mid_router_feature_dropout": 0.0,
            "micro_router_feature_dropout": 0.0,
        },
        "challenge": {
            "MAX_ITEM_LIST_LENGTH": 30,
            "hidden_size": 320,
            "embedding_size": 320,
            "inner_size": 384,
            "num_layers": 3,
            "num_heads": 4,
            "d_feat_emb": 24,
            "d_router_hidden": 160,
            "d_expert_hidden": 320,
            "expert_scale": 6,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "stage_feature_dropout_prob": 0.05,
            "mid_router_feature_dropout": 0.05,
            "micro_router_feature_dropout": 0.05,
        },
    },
    "foursquare": {
        "general": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 256,
            "embedding_size": 256,
            "inner_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 128,
            "d_expert_hidden": 256,
            "expert_scale": 4,
            "hidden_dropout_prob": 0.10,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.0,
            "mid_router_feature_dropout": 0.0,
            "micro_router_feature_dropout": 0.0,
        },
        "challenge": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 288,
            "embedding_size": 288,
            "inner_size": 384,
            "num_layers": 3,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 160,
            "d_expert_hidden": 320,
            "expert_scale": 5,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.05,
            "mid_router_feature_dropout": 0.05,
            "micro_router_feature_dropout": 0.05,
        },
    },
}


def separation_settings() -> list[dict]:
    full = canonical_stage_maps()
    return [{
        "setting_key": "separation_main",
        "setting_label": "Separation Main Tuning",
        "variant_label": "sep_main",
        "variant_group": "separation_main",
        "panel_family": "route_separation",
        "variant_order": 0,
        "overrides": {
            **deepcopy(full),
            "topk_scope_mode": "per_group",
            "group_top_k": 3,
            "expert_top_k": 2,
            "moe_top_k": 0,
            "fmoe_v2_stage_merge_aux_enable": False,
        },
    }]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clone_candidate(
    source: BaseCandidate,
    *,
    rank: int,
    tag_suffix: str,
    notes_suffix: str,
    overrides: dict,
) -> BaseCandidate:
    base_config = deepcopy(source.base_config)
    base_config.update(deepcopy(overrides))
    return BaseCandidate(
        dataset=source.dataset,
        model=source.model,
        rank=rank,
        tag=f"{source.tag}_{tag_suffix}",
        notes=f"{source.notes}; {notes_suffix}",
        result_json=source.result_json,
        payload=source.payload,
        base_config=base_config,
        checkpoint_file=source.checkpoint_file,
    )


def _build_anchor_candidates(args) -> list[BaseCandidate]:
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) or ["featured_moe_n3"]
    base_csv = Path(args.base_csv).expanduser().resolve()
    for dataset in datasets:
        validate_session_fixed_files(dataset)

    loaded = load_base_candidates(
        base_csv,
        datasets=datasets,
        models=models,
        top_k_configs=8,
    )
    grouped: dict[str, dict[int, BaseCandidate]] = {}
    for candidate in loaded:
        grouped.setdefault(candidate.dataset, {})[candidate.rank] = candidate

    selected: list[BaseCandidate] = []
    for dataset in datasets:
        dataset_candidates = grouped.get(dataset, {})
        missing = [rank for rank in ANCHOR_SOURCE_RANKS if rank not in dataset_candidates]
        if missing:
            raise RuntimeError(f"Missing anchor source ranks {missing} for dataset={dataset}")

        best_anchor = dataset_candidates[ANCHOR_SOURCE_RANKS[0]]
        selected.extend(dataset_candidates[rank] for rank in ANCHOR_SOURCE_RANKS)

        manual = MANUAL_ANCHOR_OVERRIDES.get(dataset)
        if not manual:
            raise RuntimeError(f"Missing manual anchor overrides for dataset={dataset}")
        selected.append(
            _clone_candidate(
                best_anchor,
                rank=91,
                tag_suffix="general_anchor",
                notes_suffix="manual general anchor",
                overrides=manual["general"],
            )
        )
        selected.append(
            _clone_candidate(
                best_anchor,
                rank=92,
                tag_suffix="challenge_anchor",
                notes_suffix="manual challenge anchor",
                overrides=manual["challenge"],
            )
        )
    return selected


def build_rows(args) -> list[dict]:
    candidates = _build_anchor_candidates(args)
    settings = separation_settings()
    seeds = parse_csv_ints(args.seeds) or [1]
    rows: list[dict] = []
    cursor = 0

    for candidate in candidates:
        for setting in settings:
            for seed in seeds:
                cursor += 1
                row = build_route_row(
                    question=QUESTION,
                    candidate=candidate,
                    setting=setting,
                    seed=seed,
                    runtime_seed=980000 + cursor,
                    max_evals=args.max_evals,
                    max_run_hours=args.max_run_hours,
                    tune_epochs=args.tune_epochs,
                    tune_patience=args.tune_patience,
                    lr_mode="fixed",
                )

                fixed_context = row["fixed_context"]
                for key in ZERO_AUX_KEYS:
                    fixed_context[key] = 0.0
                fixed_context["fmoe_v2_stage_merge_aux_enable"] = False

                fixed_context.pop("route_consistency_lambda", None)
                fixed_context.pop("route_separation_lambda", None)

                row["search_space"] = {
                    "route_consistency_lambda": [LAMBDA_LO, LAMBDA_HI],
                    "route_separation_lambda": [LAMBDA_LO, LAMBDA_HI],
                }
                row["search_space_types"] = {
                    "route_consistency_lambda": "loguniform",
                    "route_separation_lambda": "loguniform",
                }
                rows.append(row)

    if bool(args.smoke_test):
        rows = rows[: max(1, int(args.smoke_max_runs))]
    return rows


def _load_manifest_rows(question: str) -> dict[str, dict]:
    manifest_path = LOG_ROOT / question / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    rows = list(payload.get("rows") or [])
    return {
        str(row.get("job_id", "")).strip(): row
        for row in rows
        if str(row.get("job_id", "")).strip()
    }


def _load_best_params(source_summary: dict) -> dict[str, float]:
    result_path = str(source_summary.get("result_path", "") or "").strip()
    if not result_path:
        raise ValueError("missing result_path")
    payload = read_json(Path(result_path))
    best_params = payload.get("best_params") or {}
    if not isinstance(best_params, dict) or not best_params:
        trials = list(payload.get("trials") or [])
        if not trials:
            raise ValueError(f"missing best_params in {result_path}")

        def _trial_key(trial: dict) -> tuple[float, float]:
            valid_result = trial.get("valid_result") or {}
            test_result = trial.get("test_result") or {}
            valid_mrr20 = _safe_float(valid_result.get("mrr@20", trial.get("mrr@20", 0.0)))
            test_mrr20 = _safe_float(test_result.get("mrr@20", trial.get("test_mrr@20", 0.0)))
            return valid_mrr20, test_mrr20

        best_trial = max(trials, key=_trial_key)
        best_params = best_trial.get("params") or {}
        if not isinstance(best_params, dict) or not best_params:
            raise ValueError(f"missing best_params in {result_path}")

    out: dict[str, float] = {}
    for key in (
        "learning_rate",
        "weight_decay",
        "route_consistency_lambda",
        "route_separation_lambda",
    ):
        value = best_params.get(key)
        if value is not None:
            out[key] = float(value)
    if "route_separation_lambda" not in out:
        raise ValueError(f"missing route_separation_lambda in {result_path}")
    return out


def _select_phase2_sources(summary_rows: list[dict]) -> list[tuple[str, dict]]:
    by_dataset: dict[str, list[dict]] = {}
    for row in summary_rows:
        if str(row.get("status", "")).lower() != "ok":
            continue
        dataset = str(row.get("dataset", "")).strip()
        if dataset:
            by_dataset.setdefault(dataset, []).append(row)

    selected: list[tuple[str, dict]] = []
    for dataset, rows_ds in by_dataset.items():
        rows_sorted = sorted(rows_ds, key=lambda row: _safe_float(row.get("test_score")), reverse=True)
        if not rows_sorted:
            continue
        best_row = rows_sorted[0]
        selected.append(("best", best_row))

        best_score = _safe_float(best_row.get("test_score"))
        high_sep_candidates: list[tuple[float, float, dict]] = []
        fallback_candidates: list[tuple[float, float, dict]] = []
        for row in rows_sorted:
            if str(row.get("job_id", "")) == str(best_row.get("job_id", "")):
                continue
            try:
                params = _load_best_params(row)
            except Exception:
                continue
            sep = _safe_float(params.get("route_separation_lambda"))
            score = _safe_float(row.get("test_score"))
            fallback_candidates.append((sep, score, row))
            if sep >= HIGH_SEP_MIN and score >= best_score * HIGH_SEP_SCORE_RATIO:
                high_sep_candidates.append((sep, score, row))

        picked_high_sep: dict | None = None
        if high_sep_candidates:
            high_sep_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            picked_high_sep = high_sep_candidates[0][2]
        elif fallback_candidates:
            fallback_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            picked_high_sep = fallback_candidates[0][2]

        if picked_high_sep is not None:
            selected.append(("high_sep", picked_high_sep))

    return selected


def _build_diag_row(
    *,
    source_manifest: dict,
    source_summary: dict,
    tuned_params: dict[str, float],
    selection_kind: str,
    runtime_seed: int,
) -> dict:
    row = deepcopy(source_manifest)
    row["question"] = DIAG_QUESTION
    row["stage"] = DIAG_QUESTION
    row["run_axis"] = QUESTION_AXIS[DIAG_QUESTION]
    row["parent_job_id"] = str(source_summary.get("job_id", "") or "")
    row["setting_key"] = f"{selection_kind}_diag"
    row["setting_label"] = f"{selection_kind} diag"
    row["variant_label"] = selection_kind
    row["variant_group"] = "sep_main_diag"
    row["variant_order"] = 0 if selection_kind == "best" else 1
    row["runtime_seed"] = int(runtime_seed)
    row["seed_id"] = int(source_summary.get("seed_id", row.get("seed_id", 1)) or 1)
    fixed_context = deepcopy(row.get("fixed_context") or {})
    for key, value in tuned_params.items():
        fixed_context[key] = float(value)
    row["fixed_context"] = fixed_context
    row["search_space"] = {}
    row["search_space_types"] = {}
    row["max_evals"] = 1
    row["job_id"] = f"{source_summary.get('job_id', row.get('job_id', ''))}_{selection_kind.upper()}_DIAG"
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
        "++special_logging=true",
        "++log_unseen_target_metrics=true",
    ]


def _run_diag_job(row: dict, gpu_id: str) -> dict:
    log_path = log_path_for_row(DIAG_QUESTION, row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_route_command(row, gpu_id, search_algo="tpe")
    cmd.extend(_diag_extra_args())
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    print(f"[{DIAG_QUESTION}][gpu={gpu_id}] START {row['job_id']} dataset={row['dataset']}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# cmd={' '.join(cmd)}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT / "experiments"),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
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
    print(f"[{DIAG_QUESTION}][gpu={gpu_id}] END {row['job_id']} status={status}", flush=True)
    return summary


def _run_report() -> int:
    report_script = REPO_ROOT / "writing" / "260419_real_final_exp" / "sep_main_report.py"
    if not report_script.exists():
        print(f"[{QUESTION}] WARN missing report script: {report_script}", flush=True)
        return 0
    subprocess.run([python_bin(), str(report_script)], check=True, cwd=str(REPO_ROOT))
    return 0


def _run_phase2(
    *,
    datasets: list[str],
    gpus: list[str],
    skip_case_eval: bool,
    dry_run: bool,
    skip_report: bool,
) -> int:
    summary_rows = [
        row for row in read_summary_rows(QUESTION)
        if str(row.get("dataset", "")).strip() in set(datasets)
    ]
    selected_sources = _select_phase2_sources(summary_rows)
    if not selected_sources:
        print(f"[{DIAG_QUESTION}] No completed {QUESTION} rows found; skipping Phase-2.", flush=True)
        return 0

    print(f"[{DIAG_QUESTION}] Selected {len(selected_sources)} source runs for diag:", flush=True)
    for selection_kind, row in selected_sources:
        params = _load_best_params(row)
        print(
            f"  dataset={row['dataset']} kind={selection_kind} test={_safe_float(row.get('test_score')):.4f} sep={_safe_float(params.get('route_separation_lambda')):.4g}",
            flush=True,
        )

    if dry_run:
        print(f"[{DIAG_QUESTION}] dry-run: skipping diag reruns/case-eval/report.", flush=True)
        return 0

    manifest_rows = _load_manifest_rows(QUESTION)
    diag_jobs: list[tuple[dict, dict]] = []
    runtime_seed = 7_500_000
    for selection_kind, source_summary in selected_sources:
        source_job_id = str(source_summary.get("job_id", "") or "").strip()
        source_manifest = manifest_rows.get(source_job_id)
        if source_manifest is None:
            print(f"[{DIAG_QUESTION}] WARN missing manifest row for job_id={source_job_id}", flush=True)
            continue
        try:
            tuned_params = _load_best_params(source_summary)
        except Exception as exc:
            print(f"[{DIAG_QUESTION}] WARN cannot load tuned params for {source_job_id}: {exc}", flush=True)
            continue
        runtime_seed += 1
        diag_row = _build_diag_row(
            source_manifest=source_manifest,
            source_summary=source_summary,
            tuned_params=tuned_params,
            selection_kind=selection_kind,
            runtime_seed=runtime_seed,
        )
        diag_jobs.append((diag_row, source_summary))

    if not diag_jobs:
        print(f"[{DIAG_QUESTION}] No valid diag jobs built; skipping.", flush=True)
        return 0

    diag_summary_path = LOG_ROOT / DIAG_QUESTION / "summary.csv"
    diag_summary_path.parent.mkdir(parents=True, exist_ok=True)
    pending: Queue[tuple[dict, dict]] = Queue()
    for item in diag_jobs:
        pending.put(item)
    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    completed: list[tuple[dict, dict]] = []
    lock = threading.Lock()

    def _worker() -> None:
        while True:
            try:
                row, source_summary = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = _run_diag_job(row, gpu_id)
                with lock:
                    completed.append((summary, source_summary))
                    append_csv_row(diag_summary_path, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if not skip_case_eval:
        case_rows: list[dict] = []
        for summary, _source_summary in completed:
            if str(summary.get("status", "")).lower() != "ok":
                continue
            existing = find_completed_case_eval_row(DIAG_QUESTION, summary)
            if existing is not None:
                case_rows.append(existing)
                continue
            try:
                bundle = run_case_eval_pipeline(
                    question=DIAG_QUESTION,
                    source_summary_row=summary,
                    output_root=LOG_ROOT / DIAG_QUESTION / "case_eval" / str(summary.get("job_id", "")),
                    skip_by_group=False,
                )
                case_rows.append(bundle)
                print(f"[{DIAG_QUESTION}] case-eval OK: {summary.get('dataset')} {summary.get('setting_key')}", flush=True)
            except Exception as exc:
                print(f"[{DIAG_QUESTION}] WARN case-eval failed: {exc}", flush=True)
                case_rows.append(
                    {
                        "question": DIAG_QUESTION,
                        "dataset": summary.get("dataset", ""),
                        "setting_key": summary.get("setting_key", ""),
                        "base_rank": summary.get("base_rank", ""),
                        "seed_id": summary.get("seed_id", ""),
                        "result_path": summary.get("result_path", ""),
                        "checkpoint_file": summary.get("checkpoint_file", ""),
                        "status": "error",
                        "error": str(exc),
                    }
                )
        if case_rows:
            write_index_rows(index_path(DIAG_QUESTION, f"{DIAG_QUESTION}_case_eval_index.csv"), case_rows)

    ok_count = sum(1 for summary, _ in completed if str(summary.get("status", "")).lower() == "ok")
    print(f"[{DIAG_QUESTION}] Phase-2 done: {ok_count}/{len(diag_jobs)} succeeded.", flush=True)
    if not skip_case_eval and not skip_report:
        _run_report()
    return 0


def parse_args():
    parser = common_arg_parser("Separation-centric tuning for KuaiRec/Foursquare", question=QUESTION)
    parser.add_argument("--skip-phase2", action="store_true", help="Only run Phase-1 and skip diag/case-eval/report.")
    parser.add_argument("--phase2-only", action="store_true", help="Skip Phase-1 and run only diag/case-eval/report.")
    parser.add_argument("--skip-case-eval", action="store_true", help="Skip case-eval after diag reruns.")
    parser.add_argument("--skip-report", action="store_true", help="Skip post-run report/figure export.")
    parser.set_defaults(
        datasets=",".join(DEFAULT_DATASETS),
        models="featured_moe_n3",
        top_k_configs=4,
        max_evals=15,
        max_run_hours=3.0,
        tune_epochs=100,
        tune_patience=5,
        lr_mode="fixed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    phase2_kwargs = dict(
        datasets=datasets,
        gpus=gpus,
        skip_case_eval=bool(args.skip_case_eval),
        dry_run=bool(args.dry_run),
        skip_report=bool(args.skip_report),
    )

    if bool(args.phase2_only):
        print(f"[{QUESTION}] --phase2-only: skipping Phase-1.", flush=True)
        return _run_phase2(**phase2_kwargs)

    rows = build_rows(args)
    manifest = write_manifest(QUESTION, rows)
    print(f"[{QUESTION}] manifest -> {manifest} ({len(rows)} jobs)", flush=True)
    rc = run_jobs(
        rows,
        question=QUESTION,
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or bool(args.dry_run) or bool(args.skip_phase2):
        return rc
    print(f"\n[{QUESTION}] Phase-1 complete. Starting Phase-2...", flush=True)
    return _run_phase2(**phase2_kwargs)


if __name__ == "__main__":
    raise SystemExit(main())