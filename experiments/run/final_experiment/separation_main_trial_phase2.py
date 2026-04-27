#!/usr/bin/env python3
"""Trial-level phase-2 reruns for sep_main.

This script mines successful hyperopt trials from completed sep_main runs,
selects 8 candidates per dataset, retrains them with fixed params under diag
logging, and runs case-eval on the completed checkpoints.
"""

from __future__ import annotations

import argparse
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
    LOG_ROOT,
    QUESTION_AXIS,
    REPO_ROOT,
    SUMMARY_FIELDS,
    append_csv_row,
    build_route_command,
    build_summary_row,
    extract_error_tail,
    find_completed_case_eval_row,
    has_run_status_end_normal,
    index_path,
    load_result_payload,
    log_path_for_row,
    parse_csv_list,
    parse_result_path_from_log,
    read_json,
    read_summary_rows,
    result_has_successful_trials,
    run_case_eval_pipeline,
    write_index_rows,
)


SOURCE_QUESTION = "sep_main"
QUESTION = "sep_main_diag_trials"
QUESTION_AXIS[QUESTION] = "separation_main_diag_trials"

DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2", "foursquare"]
METRIC_KEYS = [
    "hit@5",
    "hit@10",
    "hit@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr@5",
    "mrr@10",
    "mrr@20",
]
HIGH_SEP_MIN = 1.0e-2
HIGH_SEP_RATIO_FLOOR = 0.95
HIGH_SEP_RATIO_START = 0.985
HIGH_SEP_RATIO_STEP = 0.01


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _metric_mean(metrics: dict | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    values = [_safe_float(metrics.get(key), 0.0) for key in METRIC_KEYS]
    return sum(values) / float(len(values))


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


def _trial_signature(candidate: dict) -> tuple:
    params = candidate["params"]
    return (
        candidate["dataset"],
        round(_safe_float(params.get("learning_rate")), 12),
        round(_safe_float(params.get("weight_decay")), 12),
        round(_safe_float(params.get("route_consistency_lambda")), 12),
        round(_safe_float(params.get("route_separation_lambda")), 12),
    )


def _load_trial_pool(datasets: list[str]) -> list[dict]:
    pool: list[dict] = []
    for row in read_summary_rows(SOURCE_QUESTION):
        if str(row.get("status", "")).lower() != "ok":
            continue
        dataset = str(row.get("dataset", "")).strip()
        if dataset not in datasets:
            continue
        result_path = str(row.get("result_path", "") or "").strip()
        if not result_path:
            continue
        payload = read_json(Path(result_path))
        trials = list(payload.get("trials") or [])
        for trial in trials:
            if str(trial.get("status", "ok")).lower() != "ok":
                continue
            params = deepcopy(trial.get("params") or {})
            test_result = deepcopy(trial.get("test_result") or {})
            valid_result = deepcopy(trial.get("valid_result") or {})
            candidate = {
                "dataset": dataset,
                "source_job_id": str(row.get("job_id", "") or ""),
                "source_base_rank": str(row.get("base_rank", "") or ""),
                "source_base_tag": str(row.get("base_tag", "") or ""),
                "source_result_path": result_path,
                "source_seed_id": str(row.get("seed_id", "") or "1"),
                "source_gpu_id": str(row.get("gpu_id", "") or ""),
                "trial": int(trial.get("trial", 0) or 0),
                "params": params,
                "test_result": test_result,
                "valid_result": valid_result,
                "test_mean": _metric_mean(test_result),
                "valid_mean": _metric_mean(valid_result),
                "test_mrr20": _safe_float(test_result.get("mrr@20")),
                "valid_mrr20": _safe_float(valid_result.get("mrr@20")),
                "test_hr10": _safe_float(test_result.get("hit@10")),
                "sep_lambda": _safe_float(params.get("route_separation_lambda")),
                "cons_lambda": _safe_float(params.get("route_consistency_lambda")),
                "learning_rate": _safe_float(params.get("learning_rate")),
                "weight_decay": _safe_float(params.get("weight_decay")),
            }
            pool.append(candidate)
    return pool


def _select_dataset_candidates(
    pool: list[dict],
    *,
    performance_slots: int,
    high_sep_slots: int,
) -> list[dict]:
    selected: list[dict] = []
    used: set[tuple] = set()

    perf_sorted = sorted(
        pool,
        key=lambda item: (item["test_mean"], item["test_mrr20"], item["valid_mean"], item["sep_lambda"]),
        reverse=True,
    )
    perf_rank = 0
    for candidate in perf_sorted:
        signature = _trial_signature(candidate)
        if signature in used:
            continue
        perf_rank += 1
        used.add(signature)
        chosen = dict(candidate)
        chosen["selection_bucket"] = "perf"
        chosen["selection_rank"] = perf_rank
        selected.append(chosen)
        if perf_rank >= performance_slots:
            break

    if not perf_sorted:
        return selected

    best_mean = perf_sorted[0]["test_mean"]
    ratio = HIGH_SEP_RATIO_START
    high_sep_pool: list[dict] = []
    while ratio >= HIGH_SEP_RATIO_FLOOR:
        high_sep_pool = [
            item for item in pool
            if item["sep_lambda"] >= HIGH_SEP_MIN and item["test_mean"] >= best_mean * ratio
        ]
        if len(high_sep_pool) >= high_sep_slots:
            break
        ratio -= HIGH_SEP_RATIO_STEP

    if len(high_sep_pool) < high_sep_slots:
        high_sep_pool = [item for item in pool if item["sep_lambda"] >= HIGH_SEP_MIN]
    if len(high_sep_pool) < high_sep_slots:
        high_sep_pool = list(pool)

    high_sep_sorted = sorted(
        high_sep_pool,
        key=lambda item: (item["sep_lambda"], item["test_mean"], item["test_mrr20"]),
        reverse=True,
    )
    high_sep_rank = 0
    for candidate in high_sep_sorted:
        signature = _trial_signature(candidate)
        if signature in used:
            continue
        high_sep_rank += 1
        used.add(signature)
        chosen = dict(candidate)
        chosen["selection_bucket"] = "highsep"
        chosen["selection_rank"] = high_sep_rank
        selected.append(chosen)
        if high_sep_rank >= high_sep_slots:
            break
    return selected


def select_candidates(
    datasets: list[str],
    *,
    per_dataset: int,
    performance_slots: int,
) -> list[dict]:
    pool = _load_trial_pool(datasets)
    by_dataset: dict[str, list[dict]] = {}
    for candidate in pool:
        by_dataset.setdefault(candidate["dataset"], []).append(candidate)

    selected: list[dict] = []
    high_sep_slots = max(0, per_dataset - performance_slots)
    for dataset in datasets:
        chosen = _select_dataset_candidates(
            by_dataset.get(dataset, []),
            performance_slots=performance_slots,
            high_sep_slots=high_sep_slots,
        )
        selected.extend(chosen)
    return selected


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


def _build_diag_row(source_manifest: dict, candidate: dict, runtime_seed: int) -> dict:
    selection_bucket = str(candidate["selection_bucket"])
    selection_rank = int(candidate["selection_rank"])
    trial_id = int(candidate["trial"])
    row = deepcopy(source_manifest)
    row["question"] = QUESTION
    row["stage"] = QUESTION
    row["run_axis"] = QUESTION_AXIS[QUESTION]
    row["parent_job_id"] = candidate["source_job_id"]
    row["setting_key"] = f"{selection_bucket}_{selection_rank:02d}_trial{trial_id:02d}_diag"
    row["setting_label"] = f"{selection_bucket} #{selection_rank:02d} trial {trial_id:02d} diag"
    row["variant_label"] = f"{selection_bucket}{selection_rank:02d}"
    row["variant_group"] = "sep_main_diag_trials"
    row["variant_order"] = selection_rank
    row["runtime_seed"] = runtime_seed
    row["seed_id"] = int(candidate["source_seed_id"] or 1)
    row["search_space"] = {}
    row["search_space_types"] = {}
    row["max_evals"] = 1
    fixed_context = deepcopy(row.get("fixed_context") or {})
    fixed_context["learning_rate"] = candidate["learning_rate"]
    fixed_context["weight_decay"] = candidate["weight_decay"]
    fixed_context["route_consistency_lambda"] = candidate["cons_lambda"]
    fixed_context["route_separation_lambda"] = candidate["sep_lambda"]
    row["fixed_context"] = fixed_context
    row["job_id"] = (
        f"{candidate['source_job_id']}_{selection_bucket.upper()}{selection_rank:02d}_"
        f"T{trial_id:02d}_DIAG"
    )
    row["run_phase"] = row["job_id"]
    return row


def _run_diag_job(row: dict, gpu_id: str) -> dict:
    log_path = log_path_for_row(QUESTION, row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_route_command(row, gpu_id, search_algo="tpe")
    cmd.extend(_diag_extra_args())
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    print(f"[{QUESTION}][gpu={gpu_id}] START {row['job_id']} dataset={row['dataset']}", flush=True)
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
    print(f"[{QUESTION}][gpu={gpu_id}] END {row['job_id']} status={status}", flush=True)
    return summary


def _selection_row(candidate: dict, diag_row: dict, summary: dict | None = None, case_eval: dict | None = None) -> dict:
    row = {
        "dataset": candidate["dataset"],
        "selection_bucket": candidate["selection_bucket"],
        "selection_rank": candidate["selection_rank"],
        "source_job_id": candidate["source_job_id"],
        "source_base_rank": candidate["source_base_rank"],
        "source_base_tag": candidate["source_base_tag"],
        "source_trial": candidate["trial"],
        "source_result_path": candidate["source_result_path"],
        "source_test_mean": candidate["test_mean"],
        "source_test_mrr20": candidate["test_mrr20"],
        "source_test_hr10": candidate["test_hr10"],
        "source_valid_mean": candidate["valid_mean"],
        "source_valid_mrr20": candidate["valid_mrr20"],
        "learning_rate": candidate["learning_rate"],
        "weight_decay": candidate["weight_decay"],
        "route_consistency_lambda": candidate["cons_lambda"],
        "route_separation_lambda": candidate["sep_lambda"],
        "diag_job_id": diag_row["job_id"],
        "diag_setting_key": diag_row["setting_key"],
    }
    if summary is not None:
        row.update(
            {
                "diag_status": summary.get("status", ""),
                "diag_result_path": summary.get("result_path", ""),
                "diag_test_score": summary.get("test_score", ""),
                "diag_test_mrr20": summary.get("test_mrr20", ""),
                "diag_checkpoint_file": summary.get("checkpoint_file", ""),
            }
        )
    if case_eval is not None:
        row.update(
            {
                "case_eval_status": case_eval.get("status", "ok"),
                "case_eval_manifest": case_eval.get("case_eval_manifest", ""),
                "case_eval_export_dir": case_eval.get("case_eval_export_dir", ""),
            }
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trial-level sep_main phase-2 reruns")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--per-dataset", type=int, default=8)
    parser.add_argument("--performance-slots", type=int, default=4)
    parser.add_argument("--skip-case-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse existing case-eval bundles when present.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    if args.performance_slots > args.per_dataset:
        raise RuntimeError("performance-slots cannot exceed per-dataset")

    selected = select_candidates(
        datasets,
        per_dataset=int(args.per_dataset),
        performance_slots=int(args.performance_slots),
    )
    if not selected:
        raise RuntimeError("No trial candidates found from sep_main results")

    manifest_rows = _load_manifest_rows(SOURCE_QUESTION)
    selection_preview = []
    diag_jobs: list[tuple[dict, dict]] = []
    runtime_seed = 8_500_000
    for candidate in selected:
        source_manifest = manifest_rows.get(candidate["source_job_id"])
        if source_manifest is None:
            print(f"[{QUESTION}] WARN missing manifest row for {candidate['source_job_id']}", flush=True)
            continue
        runtime_seed += 1
        diag_row = _build_diag_row(source_manifest, candidate, runtime_seed)
        diag_jobs.append((candidate, diag_row))
        selection_preview.append(_selection_row(candidate, diag_row))

    if not diag_jobs:
        raise RuntimeError("No diag jobs could be built")

    selection_path = index_path(QUESTION, f"{QUESTION}_selection_index.csv")
    write_index_rows(selection_path, selection_preview)
    print(f"[{QUESTION}] selected {len(diag_jobs)} diag jobs -> {selection_path}", flush=True)
    for item in selection_preview:
        print(
            f"  dataset={item['dataset']} bucket={item['selection_bucket']} rank={item['selection_rank']} "
            f"trial={item['source_trial']} test_mean={float(item['source_test_mean']):.6f} "
            f"test_mrr20={float(item['source_test_mrr20']):.4f} sep={float(item['route_separation_lambda']):.4g}",
            flush=True,
        )

    if bool(args.dry_run):
        return 0

    summary_path = LOG_ROOT / QUESTION / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    existing_summary_by_job: dict[str, dict] = {}
    if bool(args.resume):
        for row in read_summary_rows(QUESTION):
            job_id = str(row.get("job_id", "") or "").strip()
            if not job_id:
                continue
            if str(row.get("status", "") or "").lower() != "ok":
                continue
            existing_summary_by_job[job_id] = row

    pending: Queue[tuple[dict, dict]] = Queue()
    completed: list[tuple[dict, dict]] = []
    for item in diag_jobs:
        candidate, diag_row = item
        existing = existing_summary_by_job.get(str(diag_row.get("job_id", "") or ""))
        if existing is not None:
            completed.append((candidate, existing))
            continue
        pending.put(item)

    reused_count = len(completed)
    if reused_count:
        print(f"[{QUESTION}] resume reused {reused_count}/{len(diag_jobs)} completed diag jobs", flush=True)

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)
    lock = threading.Lock()

    def _worker() -> None:
        while True:
            try:
                candidate, diag_row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = _run_diag_job(diag_row, gpu_id)
                with lock:
                    completed.append((candidate, summary))
                    append_csv_row(summary_path, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=_worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    case_eval_rows: list[dict] = []
    final_rows: list[dict] = []
    summary_by_job = {summary.get("job_id", ""): summary for _candidate, summary in completed}
    diag_by_job = {diag_row["job_id"]: diag_row for _candidate, diag_row in diag_jobs}

    for candidate, diag_row in diag_jobs:
        summary = summary_by_job.get(diag_row["job_id"])
        case_eval = None
        if summary is not None and str(summary.get("status", "")).lower() == "ok" and not bool(args.skip_case_eval):
            existing = find_completed_case_eval_row(QUESTION, summary) if bool(args.resume) else None
            if existing is not None:
                case_eval = existing
            else:
                try:
                    case_eval = run_case_eval_pipeline(
                        question=QUESTION,
                        source_summary_row=summary,
                        output_root=LOG_ROOT / QUESTION / "case_eval" / str(summary.get("job_id", "")),
                        skip_by_group=False,
                    )
                    print(f"[{QUESTION}] case-eval OK: {summary.get('job_id')}", flush=True)
                except Exception as exc:
                    case_eval = {
                        "status": "error",
                        "error": str(exc),
                        "case_eval_manifest": "",
                        "case_eval_export_dir": "",
                    }
                    print(f"[{QUESTION}] WARN case-eval failed: {summary.get('job_id')} -> {exc}", flush=True)
            if case_eval is not None:
                case_eval_rows.append(case_eval)
        final_rows.append(_selection_row(candidate, diag_row, summary=summary, case_eval=case_eval))

    write_index_rows(selection_path, final_rows)
    if case_eval_rows:
        write_index_rows(index_path(QUESTION, f"{QUESTION}_case_eval_index.csv"), case_eval_rows)
    ok_count = sum(1 for _candidate, summary in completed if str(summary.get("status", "")).lower() == "ok")
    print(f"[{QUESTION}] completed {ok_count}/{len(diag_jobs)} diag jobs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())