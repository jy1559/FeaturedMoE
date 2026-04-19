#!/usr/bin/env python3
"""Stage 1 broad discrete TPE search for final_experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DEFAULT_DATASETS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    ROUTE_MODEL,
    STAGE1_AXIS,
    STAGE1_BASELINE_MAX_EVALS,
    STAGE1_ROUTE_MAX_EVALS,
    load_manifest,
    log_path_for_row,
    merge_manifest_rows,
    manifest_pair_index,
    manifest_route_index,
    now_utc,
    parse_csv_list,
    parse_csv_ints,
    run_jobs,
    sanitize_token,
    stage_manifest_path,
    validate_session_fixed_files,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 broad search for final_experiment")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default="")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--max-run-hours", type=float, default=DEFAULT_MAX_RUN_HOURS)
    parser.add_argument("--oom-retry-limit", type=int, default=DEFAULT_OOM_RETRY_LIMIT)
    parser.add_argument("--seed-base", type=int, default=610000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-jobs", type=int, default=4)
    return parser.parse_args()


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    pair_index = manifest_pair_index(manifest)
    route_index = manifest_route_index(manifest)

    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) if str(args.models).strip() else list(manifest.get("all_models") or [])
    models = [model for model in models if model]

    rows: List[Dict[str, Any]] = []
    cursor = 0

    for dataset in datasets:
        validate_session_fixed_files(dataset)
        for model in models:
            if model == ROUTE_MODEL:
                bank = route_index.get(dataset)
                if not bank:
                    continue
                for family in list(bank.get("families") or []):
                    cursor += 1
                    family_id = str(family.get("family_id", f"R{cursor:02d}"))
                    rows.append(
                        {
                            "stage": "stage1",
                            "run_axis": STAGE1_AXIS,
                            "dataset": dataset,
                            "model": ROUTE_MODEL,
                            "family": "route",
                            "family_id": family_id,
                            "job_id": f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(family_id, upper=True)}",
                            "run_phase": f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(ROUTE_MODEL, upper=True)}_{sanitize_token(family_id, upper=True)}",
                            "seed_id": 1,
                            "runtime_seed": int(args.seed_base) + cursor,
                            "capacity_anchor": family.get("capacity_anchor", ""),
                            "source_family_id": family.get("source_family_id", ""),
                            "family_role": family.get("family_role", ""),
                            "history_valid": family.get("history_valid", 0.0),
                            "history_test": family.get("history_test", 0.0),
                            "search_space": dict(family.get("search_space") or {}),
                            "fixed_context": dict(family.get("fixed_context") or {}),
                            "overrides": dict(bank.get("overrides") or {}),
                            "max_evals": int(bank.get("stage1_max_evals", STAGE1_ROUTE_MAX_EVALS[dataset])),
                            "max_run_hours": float(args.max_run_hours),
                            "oom_retry_limit": int(args.oom_retry_limit),
                        }
                    )
                continue

            spec = pair_index.get((dataset, model))
            if not spec:
                continue
            cursor += 1
            rows.append(
                {
                    "stage": "stage1",
                    "run_axis": STAGE1_AXIS,
                    "dataset": dataset,
                    "model": model,
                    "family": "baseline",
                    "job_id": f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}",
                    "run_phase": f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}",
                    "seed_id": 1,
                    "runtime_seed": int(args.seed_base) + cursor,
                    "search_space": dict(spec.get("search_space") or {}),
                    "fixed_context": dict(spec.get("fixed_context") or {}),
                    "max_evals": int(spec.get("stage1_max_evals", STAGE1_BASELINE_MAX_EVALS[dataset])),
                    "max_run_hours": float(args.max_run_hours),
                    "oom_retry_limit": int(args.oom_retry_limit),
                }
            )

    if bool(args.smoke_test):
        rows = list(rows[: max(1, int(args.smoke_max_jobs))])
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage1")
    existing_payload = load_manifest(path) if path.exists() else {}
    manifest_rows = [
        {
            "dataset": row.get("dataset", ""),
            "model": row.get("model", ""),
            "family": row.get("family", ""),
            "family_id": row.get("family_id", ""),
            "job_id": row.get("job_id", ""),
            "run_phase": row.get("run_phase", ""),
            "runtime_seed": row.get("runtime_seed", 0),
            "max_evals": row.get("max_evals", 0),
            "max_run_hours": row.get("max_run_hours", 0.0),
            "oom_retry_limit": row.get("oom_retry_limit", 0),
            "family_role": row.get("family_role", ""),
            "history_valid": row.get("history_valid", 0.0),
            "history_test": row.get("history_test", 0.0),
            "search_space": row.get("search_space", {}),
            "fixed_context": row.get("fixed_context", {}),
            "overrides": row.get("overrides", {}),
            "log_path": str(log_path_for_row("stage1", row)),
        }
        for row in rows
    ]
    payload = {
        "generated_at": now_utc(),
        "stage": "stage1",
        "run_axis": STAGE1_AXIS,
        "run_count": len(rows),
        "rows": merge_manifest_rows(existing_payload.get("rows") or [], manifest_rows),
    }
    payload["run_count"] = len(payload["rows"])
    write_json(path, payload)
    return path


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest_path = write_stage_manifest(rows)
    print(f"[stage1] manifest -> {manifest_path}")
    print(f"[stage1] run_count={len(rows)}")
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return run_jobs(
        rows,
        stage="stage1",
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
