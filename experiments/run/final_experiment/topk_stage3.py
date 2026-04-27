#!/usr/bin/env python3
"""Top-k Stage 3: pick top2 per method and run 3-seed confirmation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import (
    CODE_DIR,
    DEFAULT_DATASETS,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    ROUTE_MODEL,
    TRACK,
    load_manifest,
    load_stage_payloads,
    log_path_for_row,
    now_utc,
    parse_csv_list,
    run_jobs,
    sanitize_token,
    selection_rows_to_csv,
    stage_manifest_path,
    top_unique_trials,
    validate_session_fixed_files,
    write_json,
)


RUN_AXIS = "topk_stage3_seed_confirm"

if TRACK == "final_experiment":
    SELECTED_JSON = CODE_DIR / "selected_configs.json"
    SELECTED_CSV = CODE_DIR / "selected_configs.csv"
else:
    SELECTED_JSON = CODE_DIR / f"selected_configs_{TRACK}.json"
    SELECTED_CSV = CODE_DIR / f"selected_configs_{TRACK}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k stage3 seed confirmation")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=940000)
    parser.add_argument("--max-run-hours", type=float, default=min(DEFAULT_MAX_RUN_HOURS, 0.55))
    parser.add_argument("--oom-retry-limit", type=int, default=min(DEFAULT_OOM_RETRY_LIMIT, 4))
    parser.add_argument("--seeds", type=int, default=3)
    return parser.parse_args()


def load_stage2_manifest_rows() -> Dict[str, Dict[str, Any]]:
    path = stage_manifest_path("stage2")
    if not path.exists():
        return {}
    payload = load_manifest(path)
    return {str(row.get("job_id", "")): row for row in list(payload.get("rows") or [])}


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    stage2_payloads = load_stage_payloads("stage2")
    stage2_manifest = load_stage2_manifest_rows()
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for (dataset, model, job_id), item in stage2_payloads.items():
        if model != ROUTE_MODEL:
            continue
        manifest_row = dict(stage2_manifest.get(job_id, {}) or {})
        method_id = str(manifest_row.get("method_id", manifest_row.get("family_id", "")))
        if not method_id:
            continue
        grouped[(dataset, method_id)] = {
            "job_id": job_id,
            "payload": item.get("payload") or {},
            "manifest_row": manifest_row,
        }

    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        validate_session_fixed_files(dataset)
        for method_id in ("G3K2", "G4K2"):
            base = grouped.get((dataset, method_id))
            if not base:
                continue
            manifest_row = dict(base["manifest_row"])
            trials = top_unique_trials(base["payload"], top_k=2)
            for cfg_rank, trial in enumerate(trials, start=1):
                for seed_id in range(1, int(args.seeds) + 1):
                    cursor += 1
                    rows.append(
                        {
                            "stage": "stage3",
                            "run_axis": RUN_AXIS,
                            "dataset": dataset,
                            "model": ROUTE_MODEL,
                            "family": "route",
                            "family_id": method_id,
                            "parent_job_id": str(base["job_id"]),
                            "job_id": f"S3_{sanitize_token(dataset, upper=True)}_{method_id}_C{cfg_rank}_SEED{seed_id}",
                            "run_phase": f"S3_{sanitize_token(dataset, upper=True)}_TOPK_{method_id}_C{cfg_rank}_SEED{seed_id}",
                            "seed_id": seed_id,
                            "runtime_seed": int(args.seed_base) + cursor,
                            "config_rank": cfg_rank,
                            "method_id": method_id,
                            "selection_reason": "stage2_top2_per_method",
                            "search_space": dict(trial["config"]),
                            "fixed_context": dict(manifest_row.get("fixed_context") or {}),
                            "overrides": dict(manifest_row.get("overrides") or {}),
                            "max_evals": 1,
                            "max_run_hours": float(args.max_run_hours),
                            "oom_retry_limit": int(args.oom_retry_limit),
                            "config_signature": json.dumps(trial["config"], ensure_ascii=True, sort_keys=True),
                            "log_header_lines": list(manifest_row.get("log_header_lines") or []) + [
                                f"stage3_parent_job_id={base['job_id']}",
                                f"stage3_method_id={method_id}",
                                f"stage3_config_rank={cfg_rank}",
                                f"stage3_seed_id={seed_id}",
                            ],
                        }
                    )
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage3")
    payload = {
        "generated_at": now_utc(),
        "stage": "stage3",
        "run_axis": RUN_AXIS,
        "track": TRACK,
        "run_count": len(rows),
        "rows": [
            {
                "dataset": row.get("dataset", ""),
                "model": row.get("model", ""),
                "family": row.get("family", ""),
                "family_id": row.get("family_id", ""),
                "job_id": row.get("job_id", ""),
                "parent_job_id": row.get("parent_job_id", ""),
                "run_phase": row.get("run_phase", ""),
                "runtime_seed": row.get("runtime_seed", 0),
                "seed_id": row.get("seed_id", 0),
                "method_id": row.get("method_id", ""),
                "config_rank": row.get("config_rank", 0),
                "config_signature": row.get("config_signature", ""),
                "max_run_hours": row.get("max_run_hours", 0.0),
                "oom_retry_limit": row.get("oom_retry_limit", 0),
                "selection_reason": row.get("selection_reason", ""),
                "search_space": row.get("search_space", {}),
                "fixed_context": row.get("fixed_context", {}),
                "overrides": row.get("overrides", {}),
                "log_header_lines": row.get("log_header_lines", []),
                "log_path": str(log_path_for_row("stage3", row)),
            }
            for row in rows
        ],
    }
    write_json(path, payload)
    return path


def aggregate_selected_configs() -> List[Dict[str, Any]]:
    stage3_payloads = load_stage_payloads("stage3")
    manifest_rows = {}
    path = stage_manifest_path("stage3")
    if path.exists():
        payload = load_manifest(path)
        manifest_rows = {str(row.get("job_id", "")): row for row in list(payload.get("rows") or [])}

    grouped: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (dataset, model, job_id), item in stage3_payloads.items():
        mrow = dict(manifest_rows.get(job_id, {}) or {})
        signature = str(mrow.get("config_signature", ""))
        method_id = str(mrow.get("method_id", ""))
        grouped[(dataset, model, method_id, signature)].append(item)

    aggregated: List[Dict[str, Any]] = []
    for (dataset, model, method_id, signature), items in grouped.items():
        valid_scores = [float((entry.get("summary") or {}).get("valid_score", 0.0) or 0.0) for entry in items]
        test_scores = [float((entry.get("summary") or {}).get("test_score", 0.0) or 0.0) for entry in items]
        aggregated.append(
            {
                "dataset": dataset,
                "model": model,
                "method_id": method_id,
                "signature": signature,
                "seed_count": len(items),
                "mean_valid_score": sum(valid_scores) / max(len(valid_scores), 1),
                "mean_test_score": sum(test_scores) / max(len(test_scores), 1),
                "result_paths": [str(entry.get("result_path", "")) for entry in items],
            }
        )

    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        by_dataset[row["dataset"]].append(row)

    selected: List[Dict[str, Any]] = []
    for dataset, rows in by_dataset.items():
        rows.sort(key=lambda row: (float(row["mean_valid_score"]), float(row["mean_test_score"])), reverse=True)
        for rank, row in enumerate(rows[:2], start=1):
            selected.append(
                {
                    "dataset": dataset,
                    "model": row["model"],
                    "method_id": row["method_id"],
                    "config_rank": rank,
                    "seed_count": int(row["seed_count"]),
                    "mean_valid_score": float(row["mean_valid_score"]),
                    "mean_test_score": float(row["mean_test_score"]),
                    "config_json": row["signature"],
                    "result_paths_json": json.dumps(row["result_paths"], ensure_ascii=False),
                }
            )
    selected.sort(key=lambda row: (row["dataset"], row["config_rank"]))
    return selected


def write_selected_outputs(rows: List[Dict[str, Any]]) -> None:
    selection_rows_to_csv(rows, SELECTED_CSV)
    write_json(SELECTED_JSON, {"generated_at": now_utc(), "rows": rows})


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest_path = write_stage_manifest(rows)
    print(f"[topk-stage3] manifest -> {manifest_path}")
    print(f"[topk-stage3] run_count={len(rows)}")
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    rc = run_jobs(
        rows,
        stage="stage3",
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if not bool(args.dry_run):
        selected_rows = aggregate_selected_configs()
        write_selected_outputs(selected_rows)
        print(f"[topk-stage3] selected -> {SELECTED_JSON}")
        print(f"[topk-stage3] selected csv -> {SELECTED_CSV}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
