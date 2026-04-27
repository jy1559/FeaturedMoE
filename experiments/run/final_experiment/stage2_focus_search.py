#!/usr/bin/env python3
"""Stage 2 focused discrete TPE search for final_experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DEFAULT_DATASETS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    model_oom_retry_limit,
    ROUTE_MODEL,
    STAGE2_AXIS,
    STAGE2_MAX_EVALS,
    load_manifest,
    load_stage_payloads,
    log_path_for_row,
    manifest_pair_index,
    manifest_route_index,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 focus search for final_experiment")
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
    parser.add_argument("--seed-base", type=int, default=710000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-jobs", type=int, default=4)
    return parser.parse_args()


def load_stage1_manifest_rows() -> Dict[str, Dict[str, Any]]:
    path = stage_manifest_path("stage1")
    if not path.exists():
        return {}
    payload = load_manifest(path)
    return {str(row.get("job_id", "")): row for row in list(payload.get("rows") or [])}


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    pair_index = manifest_pair_index(manifest)
    route_index = manifest_route_index(manifest)
    stage1_manifest_rows = load_stage1_manifest_rows()
    stage1_payloads = load_stage_payloads("stage1")

    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) if str(args.models).strip() else list(manifest.get("all_models") or [])

    rows: List[Dict[str, Any]] = []
    cursor = 0

    for dataset in datasets:
        validate_session_fixed_files(dataset)

        if ROUTE_MODEL in models:
            route_candidates: List[Dict[str, Any]] = []
            for (_ds, _model, job_id), item in stage1_payloads.items():
                if _ds != dataset or _model != ROUTE_MODEL:
                    continue
                route_candidates.append(
                    {
                        "job_id": job_id,
                        "payload": item["payload"],
                        "summary": item["summary"],
                    }
                )
            route_candidates.sort(
                key=lambda item: (
                    float(item["summary"].get("valid_score", 0.0) or 0.0),
                    float(item["summary"].get("test_score", 0.0) or 0.0),
                ),
                reverse=True,
            )
            for rank, item in enumerate(route_candidates[:4], start=1):
                base_row = dict(stage1_manifest_rows.get(item["job_id"], {}) or {})
                if not base_row:
                    continue
                top_trial = top_unique_trials(item["payload"], top_k=1)
                configs = [entry["config"] for entry in top_trial] or [dict(base_row.get("fixed_context") or {})]
                cursor += 1
                rows.append(
                    {
                        "stage": "stage2",
                        "run_axis": STAGE2_AXIS,
                        "dataset": dataset,
                        "model": ROUTE_MODEL,
                        "family": "route",
                        "family_id": base_row.get("family_id", f"R{rank:02d}"),
                        "parent_job_id": item["job_id"],
                        "job_id": f"S2_{sanitize_token(dataset, upper=True)}_{sanitize_token(str(base_row.get('family_id', rank)), upper=True)}",
                        "run_phase": f"S2_{sanitize_token(dataset, upper=True)}_{sanitize_token(ROUTE_MODEL, upper=True)}_{sanitize_token(str(base_row.get('family_id', rank)), upper=True)}",
                        "seed_id": rank,
                        "runtime_seed": int(args.seed_base) + cursor,
                        "capacity_anchor": base_row.get("capacity_anchor", ""),
                        "search_space": narrow_space_from_configs(dict(base_row.get("search_space") or {}), configs, lr_points=5, max_other=4),
                        "fixed_context": dict(base_row.get("fixed_context") or {}),
                        "overrides": dict(route_index.get(dataset, {}).get("overrides") or {}),
                        "max_evals": int(route_index.get(dataset, {}).get("stage2_max_evals", STAGE2_MAX_EVALS[dataset])),
                        "max_run_hours": float(args.max_run_hours),
                        "oom_retry_limit": int(args.oom_retry_limit),
                    }
                )

        for model in models:
            if model == ROUTE_MODEL:
                continue
            stage1_item = stage1_payloads.get((dataset, model, f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}"))
            if not stage1_item:
                continue
            manifest_row = dict(stage1_manifest_rows.get(f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}", {}) or {})
            if not manifest_row:
                continue
            top_trials = top_unique_trials(stage1_item["payload"], top_k=4)
            if not top_trials:
                continue
            configs = [entry["config"] for entry in top_trials]
            spec = pair_index.get((dataset, model), {})
            cursor += 1
            rows.append(
                {
                    "stage": "stage2",
                    "run_axis": STAGE2_AXIS,
                    "dataset": dataset,
                    "model": model,
                    "family": "baseline",
                    "parent_job_id": manifest_row.get("job_id", ""),
                    "job_id": f"S2_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}",
                    "run_phase": f"S2_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}",
                    "seed_id": 1,
                    "runtime_seed": int(args.seed_base) + cursor,
                    "search_space": narrow_space_from_configs(dict(manifest_row.get("search_space") or {}), configs, lr_points=5, max_other=4),
                    "fixed_context": dict(spec.get("fixed_context") or manifest_row.get("fixed_context") or {}),
                    "max_evals": int(spec.get("stage2_max_evals", STAGE2_MAX_EVALS[dataset])),
                    "max_run_hours": float(args.max_run_hours),
                    "oom_retry_limit": model_oom_retry_limit(model, int(args.oom_retry_limit)),
                }
            )

    if bool(args.smoke_test):
        rows = list(rows[: max(1, int(args.smoke_max_jobs))])
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage2")
    payload = {
        "generated_at": now_utc(),
        "stage": "stage2",
        "run_axis": STAGE2_AXIS,
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
                "search_space": row.get("search_space", {}),
                "fixed_context": row.get("fixed_context", {}),
                "overrides": row.get("overrides", {}),
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
    print(f"[stage2] manifest -> {manifest_path}")
    print(f"[stage2] run_count={len(rows)}")
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
