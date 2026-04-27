#!/usr/bin/env python3
"""Top-k Stage 1: fast broad search on two top-k strategies."""

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
    log_path_for_row,
    now_utc,
    parse_csv_list,
    run_jobs,
    sanitize_token,
    stage_manifest_path,
    validate_session_fixed_files,
    write_json,
)
from topk_pipeline_common import (
    TOPK_STRATEGIES,
    base_route_overrides,
    build_fixed_context,
    compact_json,
    dropout_points,
    load_selected_rows,
    lr_points,
    parent_payload_from_selected,
    reg_points,
    stage_router_primitives,
    wd_points,
)


RUN_AXIS = "topk_stage1_short"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Top-k stage1 short run")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=920000)
    parser.add_argument("--max-evals", type=int, default=2)
    parser.add_argument("--max-run-hours", type=float, default=min(DEFAULT_MAX_RUN_HOURS, 0.7))
    parser.add_argument("--oom-retry-limit", type=int, default=min(DEFAULT_OOM_RETRY_LIMIT, 4))
    parser.add_argument("--parent-track", default="final_experiment")
    return parser.parse_args()


def build_search_space(base_fixed: Dict[str, Any]) -> Dict[str, List[Any]]:
    base_lr = float(base_fixed.get("learning_rate", 5e-4) or 5e-4)
    base_wd = float(base_fixed.get("weight_decay", 1e-6) or 1e-6)
    base_hidden = float(base_fixed.get("hidden_dropout_prob", 0.15) or 0.15)
    base_attn = float(base_fixed.get("attn_dropout_prob", 0.08) or 0.08)
    return {
        "learning_rate": lr_points(base_lr, [0.8, 1.0, 1.2]),
        "weight_decay": wd_points(base_wd, [0.5, 1.0, 2.0]),
        "hidden_dropout_prob": dropout_points(base_hidden, 0.02),
        "fixed_hidden_dropout_prob": dropout_points(float(base_fixed.get("fixed_hidden_dropout_prob", base_hidden) or base_hidden), 0.02),
        "attn_dropout_prob": [base_attn, min(base_attn + 0.02, 0.16)],
        "route_consistency_lambda": reg_points(float(base_fixed.get("route_consistency_lambda", 0.0) or 0.0), 2.5e-4, [1.0, 2.0]),
        "z_loss_lambda": reg_points(float(base_fixed.get("z_loss_lambda", 0.0) or 0.0), 1e-4, [1.0, 2.0]),
    }


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_rows = load_selected_rows(args.parent_track)
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        validate_session_fixed_files(dataset)
        selected = selected_rows.get(dataset)
        if not selected:
            raise KeyError(f"dataset missing from selected configs: {dataset}")
        payload = parent_payload_from_selected(selected)
        base_fixed = dict(payload.get("fixed_search") or {})
        if not base_fixed:
            raise RuntimeError(f"parent payload missing fixed_search for dataset={dataset}")
        parent_path = str(payload.get("__result_path", ""))
        parent_phase = str(payload.get("run_phase", ""))
        search_space = build_search_space(base_fixed)
        fixed_context = build_fixed_context(base_fixed, search_space)
        common_headers = [
            f"parent_track={args.parent_track}",
            f"parent_result_path={parent_path}",
            f"parent_run_phase={parent_phase}",
            f"parent_fixed_context_source=fixed_search",
            f"parent_fixed_context={compact_json(base_fixed)}",
        ]
        for spec in TOPK_STRATEGIES:
            cursor += 1
            method_id = str(spec["method_id"])
            overrides = base_route_overrides()
            overrides["moe_top_k"] = int(spec["moe_top_k"])
            overrides["stage_router_primitives"] = stage_router_primitives(int(spec["group_top_k"]), int(spec["expert_top_k"]))
            rows.append(
                {
                    "stage": "stage1",
                    "run_axis": RUN_AXIS,
                    "dataset": dataset,
                    "model": ROUTE_MODEL,
                    "family": "route",
                    "family_id": method_id,
                    "job_id": f"S1_{sanitize_token(dataset, upper=True)}_{method_id}",
                    "run_phase": f"S1_{sanitize_token(dataset, upper=True)}_TOPK_{method_id}",
                    "seed_id": 1,
                    "runtime_seed": int(args.seed_base) + cursor,
                    "capacity_anchor": f"parent_{args.parent_track}",
                    "source_family_id": "selected_parent",
                    "family_role": "topk_method",
                    "selection_reason": str(spec["combo_desc"]),
                    "parent_result_path": parent_path,
                    "search_space": dict(search_space),
                    "fixed_context": dict(fixed_context),
                    "overrides": overrides,
                    "max_evals": int(args.max_evals),
                    "max_run_hours": float(args.max_run_hours),
                    "oom_retry_limit": int(args.oom_retry_limit),
                    "method_id": method_id,
                    "log_header_lines": common_headers
                    + [
                        f"method_id={method_id}",
                        f"method_desc={spec['combo_desc']}",
                        f"group_top_k={int(spec['group_top_k'])}",
                        f"expert_top_k={int(spec['expert_top_k'])}",
                        f"moe_top_k={int(spec['moe_top_k'])}",
                    ],
                }
            )
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("stage1")
    payload = {
        "generated_at": now_utc(),
        "stage": "stage1",
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
                "run_phase": row.get("run_phase", ""),
                "runtime_seed": row.get("runtime_seed", 0),
                "max_evals": row.get("max_evals", 0),
                "max_run_hours": row.get("max_run_hours", 0.0),
                "oom_retry_limit": row.get("oom_retry_limit", 0),
                "method_id": row.get("method_id", ""),
                "selection_reason": row.get("selection_reason", ""),
                "parent_result_path": row.get("parent_result_path", ""),
                "search_space": row.get("search_space", {}),
                "fixed_context": row.get("fixed_context", {}),
                "overrides": row.get("overrides", {}),
                "log_header_lines": row.get("log_header_lines", []),
                "log_path": str(log_path_for_row("stage1", row)),
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
    print(f"[topk-stage1] manifest -> {manifest_path}")
    print(f"[topk-stage1] run_count={len(rows)}")
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
