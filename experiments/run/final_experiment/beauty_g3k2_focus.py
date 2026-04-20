#!/usr/bin/env python3
"""Beauty-only focused follow-up sweep for G3K2 top-k routing."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    ARTIFACT_ROOT,
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
    base_route_overrides,
    compact_json,
    load_selected_rows,
    parent_payload_from_selected,
    stage_router_primitives,
    unique_sorted,
)


RUN_AXIS = "beauty_g3k2_focus"
STAGE_NAME = "beauty_g3k2_focus"
DATASET = "beauty"
METHOD_ID = "G3K2"
METHOD_DESC = "activate 3 groups + in-group top2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beauty-only G3K2 focused sweep")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=960000)
    parser.add_argument("--max-evals", type=int, default=4)
    parser.add_argument("--max-run-hours", type=float, default=min(DEFAULT_MAX_RUN_HOURS, 0.45))
    parser.add_argument("--oom-retry-limit", type=int, default=min(DEFAULT_OOM_RETRY_LIMIT, 4))
    parser.add_argument("--parent-track", default="final_experiment")
    parser.add_argument("--reference-track", default="final_topk")
    return parser.parse_args()


def uniform_stage_dropout(value: float) -> Dict[str, float]:
    level = float(value)
    return {"macro": level, "mid": level, "micro": level}


def find_latest_result(track: str, pattern: str) -> str:
    result_root = ARTIFACT_ROOT / "results" / str(track).strip()
    matches = sorted(result_root.glob(pattern))
    if not matches:
        return ""
    return str(matches[-1])


def build_fixed_context(base_fixed: Dict[str, Any], search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    fixed = dict(base_fixed)
    for key in search_space:
        fixed.pop(key, None)
    return fixed


def build_family_specs(base_fixed: Dict[str, Any]) -> List[Dict[str, Any]]:
    base_lr = float(base_fixed.get("learning_rate", 5.672e-4) or 5.672e-4)
    base_wd = float(base_fixed.get("weight_decay", 1e-6) or 1e-6)
    base_hidden = float(base_fixed.get("hidden_dropout_prob", 0.2) or 0.2)
    base_fixed_hidden = float(base_fixed.get("fixed_hidden_dropout_prob", base_hidden) or base_hidden)
    base_attn = float(base_fixed.get("attn_dropout_prob", 0.1) or 0.1)
    base_stage_feat = float(base_fixed.get("stage_feature_dropout_prob", 0.1) or 0.1)
    base_router_hidden = int(base_fixed.get("d_router_hidden", 32) or 32)
    base_family_drop = float((base_fixed.get("stage_family_dropout_prob") or {}).get("macro", 0.02) or 0.02)

    return [
        {
            "family_id": "B01_DENSE4_RECOVER",
            "desc": "restore beauty dense-parent capacity and tune lightly around it",
            "search_space": {
                "learning_rate": unique_sorted([round(base_lr * mul, 8) for mul in (0.8, 1.0, 1.2)]),
                "weight_decay": unique_sorted([max(base_wd * mul, 5e-7) for mul in (0.5, 1.0, 2.0)]),
                "hidden_dropout_prob": unique_sorted([0.18, base_hidden, 0.22]),
                "fixed_hidden_dropout_prob": unique_sorted([0.18, base_fixed_hidden, 0.22]),
                "attn_dropout_prob": unique_sorted([base_attn, 0.12]),
                "route_consistency_lambda": unique_sorted([1.25e-4, 2.5e-4, 5e-4]),
                "z_loss_lambda": unique_sorted([5e-5, 1e-4, 2e-4]),
                "expert_scale": [4],
                "d_router_hidden": unique_sorted([base_router_hidden, 48]),
                "stage_feature_dropout_prob": unique_sorted([max(base_stage_feat - 0.02, 0.05), base_stage_feat, 0.12]),
            },
        },
        {
            "family_id": "B02_TOPK_S2_REBOUND",
            "desc": "stay near the earlier beauty G3K2 stage2 pocket but reopen expert scale",
            "search_space": {
                "learning_rate": [0.00045376, 0.0005672, 0.00068064],
                "weight_decay": [5e-7, 1e-6, 2e-6],
                "hidden_dropout_prob": [0.18, 0.2, 0.22],
                "fixed_hidden_dropout_prob": [0.18, 0.2, 0.22],
                "attn_dropout_prob": [0.1, 0.12, 0.14],
                "route_consistency_lambda": [2.5e-4, 5e-4, 1e-3],
                "z_loss_lambda": [1e-4, 2e-4, 4e-4],
                "expert_scale": [3, 4],
                "d_router_hidden": [32, 48],
                "stage_feature_dropout_prob": [0.08, 0.1, 0.12],
            },
        },
        {
            "family_id": "B03_LOWLR_STABLE",
            "desc": "lower learning-rate, stronger dropout, stability-biased beauty sweep",
            "search_space": {
                "learning_rate": [0.00028, 0.00035, 0.00045],
                "weight_decay": [1e-6, 2e-6, 4e-6],
                "hidden_dropout_prob": [0.2, 0.22, 0.24],
                "fixed_hidden_dropout_prob": [0.2, 0.22, 0.24],
                "attn_dropout_prob": [0.12, 0.14],
                "route_consistency_lambda": [2.5e-4, 5e-4, 1e-3],
                "z_loss_lambda": [1e-4, 2e-4, 4e-4],
                "expert_scale": [4],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": [0.1, 0.15],
                "stage_family_dropout_prob": [uniform_stage_dropout(base_family_drop), uniform_stage_dropout(0.04)],
            },
        },
        {
            "family_id": "B04_CAPACITY_UP",
            "desc": "try larger capacity with milder regularization under the same G3K2 routing",
            "search_space": {
                "learning_rate": [0.00035, 0.00045, 0.0005672],
                "weight_decay": [5e-7, 1e-6, 2e-6],
                "hidden_dropout_prob": [0.16, 0.18, 0.2],
                "fixed_hidden_dropout_prob": [0.16, 0.18, 0.2],
                "attn_dropout_prob": [0.08, 0.1, 0.12],
                "route_consistency_lambda": [0.0, 1.25e-4, 2.5e-4],
                "z_loss_lambda": [0.0, 5e-5, 1e-4],
                "expert_scale": [4, 5],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": [0.05, 0.08, 0.1],
                "stage_family_dropout_prob": [uniform_stage_dropout(0.0), uniform_stage_dropout(base_family_drop)],
            },
        },
        {
            "family_id": "B05_HIGHREG_SAFE",
            "desc": "test whether beauty needs heavier regularization for seed stability",
            "search_space": {
                "learning_rate": [0.00025, 0.00035, 0.00045],
                "weight_decay": [2e-6, 4e-6, 8e-6],
                "hidden_dropout_prob": [0.22, 0.24, 0.26],
                "fixed_hidden_dropout_prob": [0.22, 0.24, 0.26],
                "attn_dropout_prob": [0.12, 0.14, 0.16],
                "route_consistency_lambda": [5e-4, 1e-3, 2e-3],
                "z_loss_lambda": [2e-4, 4e-4, 8e-4],
                "expert_scale": [3, 4],
                "d_router_hidden": [32, 48],
                "stage_feature_dropout_prob": [0.1, 0.15],
                "stage_family_dropout_prob": [uniform_stage_dropout(base_family_drop), uniform_stage_dropout(0.04)],
            },
        },
    ]


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    validate_session_fixed_files(DATASET)
    selected_rows = load_selected_rows(args.parent_track)
    selected = selected_rows.get(DATASET)
    if not selected:
        raise KeyError(f"dataset missing from selected configs: {DATASET}")
    payload = parent_payload_from_selected(selected)
    base_fixed = dict(payload.get("fixed_search") or {})
    if not base_fixed:
        raise RuntimeError(f"parent payload missing fixed_search for dataset={DATASET}")

    parent_path = str(payload.get("__result_path", ""))
    parent_phase = str(payload.get("run_phase", ""))
    reference_path = find_latest_result(args.reference_track, f"{DATASET}_FeaturedMoE_N3_s2_beauty_topk_g3k2_*.json")

    overrides = base_route_overrides()
    overrides["moe_top_k"] = 0
    overrides["stage_router_primitives"] = stage_router_primitives(3, 2)

    common_headers = [
        f"parent_track={args.parent_track}",
        f"parent_result_path={parent_path}",
        f"parent_run_phase={parent_phase}",
        "parent_fixed_context_source=fixed_search",
        f"parent_fixed_context={compact_json(base_fixed)}",
        f"reference_track={args.reference_track}",
        f"reference_stage2_result_path={reference_path}",
    ]

    rows: List[Dict[str, Any]] = []
    for offset, spec in enumerate(build_family_specs(base_fixed), start=1):
        search_space = dict(spec["search_space"])
        rows.append(
            {
                "stage": "stage1",
                "run_axis": RUN_AXIS,
                "dataset": DATASET,
                "model": ROUTE_MODEL,
                "family": "route",
                "family_id": str(spec["family_id"]),
                "job_id": f"BF_{sanitize_token(DATASET, upper=True)}_{METHOD_ID}_{sanitize_token(spec['family_id'], upper=True)}",
                "run_phase": f"BF_{sanitize_token(DATASET, upper=True)}_TOPK_{METHOD_ID}_{sanitize_token(spec['family_id'], upper=True)}",
                "seed_id": 1,
                "runtime_seed": int(args.seed_base) + offset,
                "capacity_anchor": f"beauty_focus_{spec['family_id'].lower()}",
                "source_family_id": "beauty_dense_parent",
                "family_role": "beauty_topk_focus",
                "selection_reason": str(spec["desc"]),
                "parent_result_path": parent_path,
                "search_space": search_space,
                "fixed_context": build_fixed_context(base_fixed, search_space),
                "overrides": dict(overrides),
                "max_evals": int(args.max_evals),
                "max_run_hours": float(args.max_run_hours),
                "oom_retry_limit": int(args.oom_retry_limit),
                "method_id": METHOD_ID,
                "log_header_lines": common_headers
                + [
                    f"method_id={METHOD_ID}",
                    f"method_desc={METHOD_DESC}",
                    "group_top_k=3",
                    "expert_top_k=2",
                    "moe_top_k=0",
                    f"focus_bank_id={spec['family_id']}",
                    f"focus_bank_desc={spec['desc']}",
                ],
            }
        )
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path(STAGE_NAME)
    payload = {
        "generated_at": now_utc(),
        "stage": STAGE_NAME,
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
                "log_path": str(log_path_for_row(STAGE_NAME, row)),
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
    print(f"[beauty-g3k2-focus] manifest -> {manifest_path}")
    print(f"[beauty-g3k2-focus] run_count={len(rows)}")
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return run_jobs(
        rows,
        stage=STAGE_NAME,
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
