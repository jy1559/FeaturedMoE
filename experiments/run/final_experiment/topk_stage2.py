#!/usr/bin/env python3
"""Top-k Stage 2: short refinement from stage1 winners per method."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DEFAULT_DATASETS,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    ROUTE_MODEL,
    TRACK,
    load_manifest,
    load_stage_payloads,
    log_path_for_row,
    narrow_space_from_configs,
    now_utc,
    parse_csv_list,
    run_jobs,
    sanitize_token,
    stage_manifest_path,
    top_unique_trials,
    validate_session_fixed_files,
    write_json,
)


RUN_AXIS = "topk_stage2_short"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k stage2 short run")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=930000)
    parser.add_argument("--max-evals", type=int, default=3)
    parser.add_argument("--max-run-hours", type=float, default=min(DEFAULT_MAX_RUN_HOURS, 0.8))
    parser.add_argument("--oom-retry-limit", type=int, default=min(DEFAULT_OOM_RETRY_LIMIT, 4))
    return parser.parse_args()


def load_stage1_manifest_rows() -> Dict[str, Dict[str, Any]]:
    path = stage_manifest_path("stage1")
    if not path.exists():
        return {}
    payload = load_manifest(path)
    return {str(row.get("job_id", "")): row for row in list(payload.get("rows") or [])}


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    stage1_payloads = load_stage_payloads("stage1")
    stage1_manifest_rows = load_stage1_manifest_rows()
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)

    grouped: Dict[tuple[str, str], Dict[str, Any]] = {}
    for (dataset, model, job_id), item in stage1_payloads.items():
        if model != ROUTE_MODEL:
            continue
        manifest_row = dict(stage1_manifest_rows.get(job_id, {}) or {})
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
            top_trials = top_unique_trials(base["payload"], top_k=3)
            trial_configs = [entry["config"] for entry in top_trials] or [dict(manifest_row.get("fixed_context") or {})]
            search_space = narrow_space_from_configs(
                dict(manifest_row.get("search_space") or {}),
                trial_configs,
                lr_points=3,
                max_other=3,
            )
            cursor += 1
            rows.append(
                {
                    "stage": "stage2",
                    "run_axis": RUN_AXIS,
                    "dataset": dataset,
                    "model": ROUTE_MODEL,
                    "family": "route",
                    "family_id": method_id,
                    "parent_job_id": str(base["job_id"]),
                    "job_id": f"S2_{sanitize_token(dataset, upper=True)}_{method_id}",
                    "run_phase": f"S2_{sanitize_token(dataset, upper=True)}_TOPK_{method_id}",
                    "seed_id": 1,
                    "runtime_seed": int(args.seed_base) + cursor,
                    "search_space": search_space,
                    "fixed_context": dict(manifest_row.get("fixed_context") or {}),
                    "overrides": dict(manifest_row.get("overrides") or {}),
                    "max_evals": int(args.max_evals),
                    "max_run_hours": float(args.max_run_hours),
                    "oom_retry_limit": int(args.oom_retry_limit),
                    "method_id": method_id,
                    "selection_reason": "top3_trials_narrow",
                    "log_header_lines": list(manifest_row.get("log_header_lines") or []) + [
                        f"stage2_parent_job_id={base['job_id']}",
                        "stage2_narrow_mode=top3_trials",
                    ],
                }
            )
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage2")
    payload = {
        "generated_at": now_utc(),
        "stage": "stage2",
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
                "max_evals": row.get("max_evals", 0),
                "max_run_hours": row.get("max_run_hours", 0.0),
                "oom_retry_limit": row.get("oom_retry_limit", 0),
                "method_id": row.get("method_id", ""),
                "selection_reason": row.get("selection_reason", ""),
                "search_space": row.get("search_space", {}),
                "fixed_context": row.get("fixed_context", {}),
                "overrides": row.get("overrides", {}),
                "log_header_lines": row.get("log_header_lines", []),
                "log_path": str(log_path_for_row("stage2", row)),
            }
            for row in rows
        ],
    }
    write_json(path, payload)
    return path


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest_path = write_stage_manifest(rows)
    print(f"[topk-stage2] manifest -> {manifest_path}")
    print(f"[topk-stage2] run_count={len(rows)}")
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return run_jobs(
        rows,
        stage="stage2",
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
