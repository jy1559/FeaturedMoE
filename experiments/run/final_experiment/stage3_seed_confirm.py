#!/usr/bin/env python3
"""Stage 3 seed confirmation and final selection for final_experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import (
    CODE_DIR,
    DEFAULT_DATASETS,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    ROUTE_MODEL,
    STAGE3_AXIS,
    STAGE3_FALLBACK_TOPK_BY_DATASET,
    STAGE3_MODEL_SHORTLIST,
    STAGE3_SEED_COUNTS,
    STAGE3_TOPK_BY_DATASET,
    load_manifest,
    load_stage_payloads,
    log_path_for_row,
    merge_manifest_rows,
    manifest_pair_index,
    manifest_route_index,
    now_utc,
    parse_csv_list,
    route_family_history_score,
    run_jobs,
    sanitize_token,
    select_route_stage3_configs,
    selection_rows_to_csv,
    stage_manifest_path,
    top_unique_trials,
    validate_session_fixed_files,
    write_json,
)


SELECTED_JSON = CODE_DIR / "selected_configs.json"
SELECTED_CSV = CODE_DIR / "selected_configs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 seed confirmation for final_experiment")
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
    parser.add_argument("--seed-base", type=int, default=810000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-jobs", type=int, default=6)
    return parser.parse_args()


def load_stage2_manifest_rows() -> Dict[str, Dict[str, Any]]:
    path = stage_manifest_path("stage2")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("job_id", "")): row for row in list(payload.get("rows") or [])}


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    manifest = load_manifest(args.manifest)
    pair_index = manifest_pair_index(manifest)
    route_index = manifest_route_index(manifest)
    stage1_payloads = load_stage_payloads("stage1")
    stage2_payloads = load_stage_payloads("stage2")
    stage2_manifest_rows = load_stage2_manifest_rows()

    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) if str(args.models).strip() else list(manifest.get("all_models") or [])

    rows: List[Dict[str, Any]] = []
    cursor = 0

    for dataset in datasets:
        validate_session_fixed_files(dataset)
        seed_count = int(STAGE3_SEED_COUNTS[dataset])
        allowed_models = set(STAGE3_MODEL_SHORTLIST.get(dataset, []))
        stage3_top_k = int(STAGE3_TOPK_BY_DATASET.get(dataset, 1))
        fallback_top_k = int(STAGE3_FALLBACK_TOPK_BY_DATASET.get(dataset, 1))

        if ROUTE_MODEL in models:
            route_configs: List[Dict[str, Any]] = []
            for (_ds, _model, job_id), item in stage2_payloads.items():
                if _ds != dataset or _model != ROUTE_MODEL:
                    continue
                top_trials = top_unique_trials(item["payload"], top_k=2)
                for trial in top_trials:
                    manifest_row = dict(stage2_manifest_rows.get(job_id, {}) or {})
                    route_configs.append(
                        {
                            "config": dict(trial["config"]),
                            "valid_score": float(trial["valid_score"]),
                            "test_score": float(trial["test_score"]),
                            "parent_job_id": job_id,
                            "selection_reason": str(manifest_row.get("selection_reason", "")),
                            "family_role": str(manifest_row.get("family_role", "")),
                            "history_valid": float(manifest_row.get("history_valid", 0.0) or 0.0),
                            "history_test": float(manifest_row.get("history_test", 0.0) or 0.0),
                            "config_signature": json.dumps(dict(trial["config"]), ensure_ascii=True, sort_keys=True),
                            "manifest_row": manifest_row,
                        }
                    )
            route_configs = select_route_stage3_configs(dataset, route_configs)
            for config_rank, item in enumerate(route_configs, start=1):
                manifest_row = dict(item["manifest_row"] or {})
                for seed_idx in range(1, seed_count + 1):
                    cursor += 1
                    rows.append(
                        {
                            "stage": "stage3",
                            "run_axis": STAGE3_AXIS,
                            "dataset": dataset,
                            "model": ROUTE_MODEL,
                            "family": "route",
                            "family_id": manifest_row.get("family_id", f"R{config_rank:02d}"),
                            "parent_job_id": item["parent_job_id"],
                            "job_id": f"S3_{sanitize_token(dataset, upper=True)}_{sanitize_token(str(config_rank), upper=True)}_SEED{seed_idx}",
                            "run_phase": f"S3_{sanitize_token(dataset, upper=True)}_{sanitize_token(ROUTE_MODEL, upper=True)}_C{config_rank}_SEED{seed_idx}",
                            "seed_id": seed_idx,
                            "runtime_seed": int(args.seed_base) + cursor,
                            "search_space": dict(item["config"]),
                            "fixed_context": dict(manifest_row.get("fixed_context") or {}),
                            "overrides": dict(route_index.get(dataset, {}).get("overrides") or {}),
                            "max_evals": 1,
                            "max_run_hours": float(args.max_run_hours),
                            "oom_retry_limit": int(args.oom_retry_limit),
                            "config_rank": config_rank,
                            "config_signature": str(item.get("config_signature", "")),
                            "selection_reason": str(item.get("selection_reason", "")),
                        }
                    )

        for model in models:
            if model == ROUTE_MODEL:
                continue
            if allowed_models and model not in allowed_models:
                continue
            job_id = f"S2_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}"
            stage2_item = stage2_payloads.get((dataset, model, job_id))
            top_trials: List[Dict[str, Any]] = []
            parent_job_id = job_id
            selection_reason = "stage2_top"
            if stage2_item:
                top_trials = top_unique_trials(stage2_item["payload"], top_k=stage3_top_k)
            if not top_trials:
                stage1_job_id = f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}"
                stage1_item = stage1_payloads.get((dataset, model, stage1_job_id))
                if not stage1_item:
                    continue
                top_trials = top_unique_trials(stage1_item["payload"], top_k=fallback_top_k)
                if not top_trials:
                    continue
                parent_job_id = stage1_job_id
                selection_reason = "stage1_fallback"
            manifest_row = dict(stage2_manifest_rows.get(job_id, {}) or {})
            spec = pair_index.get((dataset, model), {})
            for config_rank, trial in enumerate(top_trials, start=1):
                for seed_idx in range(1, seed_count + 1):
                    cursor += 1
                    rows.append(
                        {
                            "stage": "stage3",
                            "run_axis": STAGE3_AXIS,
                            "dataset": dataset,
                            "model": model,
                            "family": "baseline",
                            "parent_job_id": parent_job_id,
                            "job_id": f"S3_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}_C{config_rank}_SEED{seed_idx}",
                            "run_phase": f"S3_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}_C{config_rank}_SEED{seed_idx}",
                            "seed_id": seed_idx,
                            "runtime_seed": int(args.seed_base) + cursor,
                            "search_space": dict(trial["config"]),
                            "fixed_context": dict(spec.get("fixed_context") or manifest_row.get("fixed_context") or {}),
                            "max_evals": 1,
                            "max_run_hours": float(args.max_run_hours),
                            "oom_retry_limit": int(args.oom_retry_limit),
                            "config_rank": config_rank,
                            "config_signature": json.dumps(trial["config"], ensure_ascii=True, sort_keys=True),
                            "selection_reason": selection_reason,
                        }
                    )

    if bool(args.smoke_test):
        rows = list(rows[: max(1, int(args.smoke_max_jobs))])
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage3")
    existing_payload = load_manifest(path) if path.exists() else {}
    manifest_rows = [
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
            "max_run_hours": row.get("max_run_hours", 0.0),
            "oom_retry_limit": row.get("oom_retry_limit", 0),
            "config_rank": row.get("config_rank", 0),
            "config_signature": row.get("config_signature", ""),
            "selection_reason": row.get("selection_reason", ""),
            "search_space": row.get("search_space", {}),
            "fixed_context": row.get("fixed_context", {}),
            "overrides": row.get("overrides", {}),
            "log_path": str(log_path_for_row("stage3", row)),
        }
        for row in rows
    ]
    payload = {
        "generated_at": now_utc(),
        "stage": "stage3",
        "run_axis": STAGE3_AXIS,
        "run_count": len(rows),
        "rows": merge_manifest_rows(existing_payload.get("rows") or [], manifest_rows),
    }
    payload["run_count"] = len(payload["rows"])
    write_json(path, payload)
    return path


def aggregate_selected_configs() -> List[Dict[str, Any]]:
    stage3_payloads = load_stage_payloads("stage3")
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for (dataset, model, _job_id), item in stage3_payloads.items():
        summary = dict(item["summary"] or {})
        signature = ""
        try:
            manifest_payload = json.loads(stage_manifest_path("stage3").read_text(encoding="utf-8"))
            manifest_rows = {str(row.get("job_id", "")): row for row in list(manifest_payload.get("rows") or [])}
            signature = str(manifest_rows.get(_job_id, {}).get("config_signature", ""))
        except Exception:
            signature = ""
        grouped[(dataset, model, signature)].append(item)

    aggregated: List[Dict[str, Any]] = []
    for (dataset, model, signature), items in grouped.items():
        if not items:
            continue
        valid_scores = [float((item["summary"] or {}).get("valid_score", 0.0) or 0.0) for item in items]
        test_scores = [float((item["summary"] or {}).get("test_score", 0.0) or 0.0) for item in items]
        aggregated.append(
            {
                "dataset": dataset,
                "model": model,
                "signature": signature,
                "seed_count": len(items),
                "mean_valid_score": sum(valid_scores) / len(valid_scores),
                "mean_test_score": sum(test_scores) / len(test_scores),
                "result_paths": [str(item["result_path"]) for item in items],
            }
        )

    by_pair: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in aggregated:
        by_pair[(row["dataset"], row["model"])].append(row)

    selected: List[Dict[str, Any]] = []
    for (dataset, model), rows in by_pair.items():
        rows.sort(key=lambda row: (float(row["mean_valid_score"]), float(row["mean_test_score"])), reverse=True)
        best = rows[0]
        selected.append(
            {
                "dataset": dataset,
                "model": model,
                "config_rank": 1,
                "seed_count": int(best["seed_count"]),
                "mean_valid_score": float(best["mean_valid_score"]),
                "mean_test_score": float(best["mean_test_score"]),
                "config_json": best["signature"],
                "result_paths_json": json.dumps(best["result_paths"], ensure_ascii=False),
            }
        )
    selected.sort(key=lambda row: (row["dataset"], row["model"]))
    return selected


def write_selected_outputs(rows: List[Dict[str, Any]]) -> None:
    selection_rows_to_csv(rows, SELECTED_CSV)
    write_json(SELECTED_JSON, {"generated_at": now_utc(), "rows": rows})


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest_path = write_stage_manifest(rows)
    print(f"[stage3] manifest -> {manifest_path}")
    print(f"[stage3] run_count={len(rows)}")
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
        print(f"[stage3] selected -> {SELECTED_JSON}")
        print(f"[stage3] selected csv -> {SELECTED_CSV}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
