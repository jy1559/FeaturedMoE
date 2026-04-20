#!/usr/bin/env python3
"""Beauty-only G3K2 focused sweep v2 – more trials, wider diversity."""

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


RUN_AXIS = "beauty_g3k2_focus2"
STAGE_NAME = "beauty_g3k2_focus2"
DATASET = "beauty"
METHOD_ID = "G3K2"
METHOD_DESC = "activate 3 groups + in-group top2"

# B01 best from v1: lr=0.00068064, wd=2e-6, hidden=0.2, fixed_hidden=0.2,
#   attn=0.1, rcl=0.00025, feat_drop=0.12, d_router=48, z_loss=0.0001, es=4
B01_BEST = {
    "learning_rate": 0.00068064,
    "weight_decay": 2e-6,
    "hidden_dropout_prob": 0.2,
    "fixed_hidden_dropout_prob": 0.2,
    "attn_dropout_prob": 0.1,
    "route_consistency_lambda": 0.00025,
    "stage_feature_dropout_prob": 0.12,
    "d_router_hidden": 48,
    "z_loss_lambda": 0.0001,
    "expert_scale": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Beauty-only G3K2 focused sweep v2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed-base", type=int, default=970000)
    parser.add_argument("--max-evals", type=int, default=8)
    parser.add_argument("--max-run-hours", type=float, default=min(DEFAULT_MAX_RUN_HOURS, 0.65))
    parser.add_argument("--oom-retry-limit", type=int, default=min(DEFAULT_OOM_RETRY_LIMIT, 4))
    parser.add_argument("--parent-track", default="final_experiment")
    return parser.parse_args()


def uniform_drop(v: float) -> Dict[str, float]:
    return {"macro": float(v), "mid": float(v), "micro": float(v)}


def build_fixed_context(base_fixed: Dict[str, Any], search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    fixed = dict(base_fixed)
    for key in search_space:
        fixed.pop(key, None)
    return fixed


def build_family_specs() -> List[Dict[str, Any]]:
    lr_c = B01_BEST["learning_rate"]       # 0.00068064
    wd_c = B01_BEST["weight_decay"]        # 2e-6
    hd_c = B01_BEST["hidden_dropout_prob"] # 0.2
    at_c = B01_BEST["attn_dropout_prob"]   # 0.1
    fd_c = B01_BEST["stage_feature_dropout_prob"]  # 0.12

    return [
        # ── B01: tight circle around v1-best, more LR grid ──────────────────
        {
            "family_id": "B01_NEAR_BEST",
            "desc": "tight around v1-best B01 center, 8 trials",
            "search_space": {
                "learning_rate": unique_sorted([lr_c * m for m in (0.75, 0.85, 1.0, 1.15, 1.3)]),
                "weight_decay": [1e-6, 2e-6, 4e-6],
                "hidden_dropout_prob": unique_sorted([hd_c - 0.02, hd_c, hd_c + 0.02]),
                "fixed_hidden_dropout_prob": unique_sorted([hd_c - 0.02, hd_c, hd_c + 0.02]),
                "attn_dropout_prob": unique_sorted([at_c, at_c + 0.02, at_c - 0.02]),
                "route_consistency_lambda": [1.25e-4, 2.5e-4, 5e-4],
                "z_loss_lambda": [5e-5, 1e-4, 2e-4],
                "expert_scale": [4],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": unique_sorted([fd_c - 0.02, fd_c, fd_c + 0.02]),
            },
        },
        # ── B02: push LR higher (beauty train set is small, may want faster lr) ──
        {
            "family_id": "B02_HIGH_LR",
            "desc": "higher LR region (0.0007-0.0014), expert_scale 4-5",
            "search_space": {
                "learning_rate": [0.00075, 0.0009, 0.00105, 0.0012, 0.0014],
                "weight_decay": [1e-6, 2e-6, 4e-6],
                "hidden_dropout_prob": [0.18, 0.2, 0.22],
                "fixed_hidden_dropout_prob": [0.18, 0.2, 0.22],
                "attn_dropout_prob": [0.08, 0.1, 0.12],
                "route_consistency_lambda": [0.0, 1.25e-4, 2.5e-4],
                "z_loss_lambda": [0.0, 5e-5, 1e-4],
                "expert_scale": [4, 5],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": [0.1, 0.12, 0.15],
            },
        },
        # ── B03: vary d_feat_emb – haven't tried non-default (16) yet ──────
        {
            "family_id": "B03_FEAT_DIM",
            "desc": "vary d_feat_emb (12/16/24/32) and d_router_hidden together",
            "search_space": {
                "learning_rate": [0.00056, 0.00068, 0.00085],
                "weight_decay": [1e-6, 2e-6],
                "hidden_dropout_prob": [0.18, 0.2],
                "fixed_hidden_dropout_prob": [0.18, 0.2],
                "attn_dropout_prob": [0.1, 0.12],
                "route_consistency_lambda": [1.25e-4, 2.5e-4, 5e-4],
                "z_loss_lambda": [5e-5, 1e-4],
                "expert_scale": [4],
                "d_router_hidden": [32, 48, 64],
                "d_feat_emb": [12, 16, 24, 32],
                "stage_feature_dropout_prob": [0.1, 0.12],
            },
        },
        # ── B04: expert_scale ladder 3-6, moderate reg ──────────────────────
        {
            "family_id": "B04_SCALE_LADDER",
            "desc": "sweep expert_scale 3/4/5/6 with moderate regularization",
            "search_space": {
                "learning_rate": [0.00050, 0.00068, 0.00085],
                "weight_decay": [1e-6, 2e-6, 4e-6],
                "hidden_dropout_prob": [0.18, 0.2, 0.22],
                "fixed_hidden_dropout_prob": [0.18, 0.2, 0.22],
                "attn_dropout_prob": [0.08, 0.1, 0.12],
                "route_consistency_lambda": [1.25e-4, 2.5e-4, 5e-4],
                "z_loss_lambda": [5e-5, 1e-4, 2e-4],
                "expert_scale": [3, 4, 5, 6],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": [0.08, 0.1, 0.12],
            },
        },
        # ── B05: soften routing constraints (very low lambda, let routing free) ──
        {
            "family_id": "B05_FREE_ROUTING",
            "desc": "no routing aux losses, let G3K2 route freely",
            "search_space": {
                "learning_rate": [0.00056, 0.00068, 0.00085, 0.001],
                "weight_decay": [5e-7, 1e-6, 2e-6],
                "hidden_dropout_prob": [0.16, 0.18, 0.2],
                "fixed_hidden_dropout_prob": [0.16, 0.18, 0.2],
                "attn_dropout_prob": [0.06, 0.08, 0.1],
                "route_consistency_lambda": [0.0],
                "z_loss_lambda": [0.0],
                "expert_scale": [4, 5],
                "d_router_hidden": [48, 64],
                "stage_feature_dropout_prob": [0.08, 0.1],
                "stage_family_dropout_prob": [uniform_drop(0.0), uniform_drop(0.01)],
            },
        },
        # ── B06: wide random – largest combo space, let TPE roam ────────────
        {
            "family_id": "B06_WIDE_RANDOM",
            "desc": "wide grid, no strong anchor, maximize coverage",
            "search_space": {
                "learning_rate": [0.0003, 0.00045, 0.00056, 0.00068, 0.0009, 0.0012],
                "weight_decay": [5e-7, 1e-6, 2e-6, 4e-6, 8e-6],
                "hidden_dropout_prob": [0.14, 0.16, 0.18, 0.2, 0.22, 0.24],
                "fixed_hidden_dropout_prob": [0.14, 0.16, 0.18, 0.2, 0.22, 0.24],
                "attn_dropout_prob": [0.06, 0.08, 0.1, 0.12, 0.14],
                "route_consistency_lambda": [0.0, 1.25e-4, 2.5e-4, 5e-4, 1e-3],
                "z_loss_lambda": [0.0, 5e-5, 1e-4, 2e-4, 4e-4],
                "expert_scale": [3, 4, 5],
                "d_router_hidden": [32, 48, 64],
                "d_feat_emb": [16, 24, 32],
                "stage_feature_dropout_prob": [0.05, 0.08, 0.1, 0.12, 0.15],
                "stage_family_dropout_prob": [uniform_drop(0.0), uniform_drop(0.02), uniform_drop(0.04)],
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

    overrides = base_route_overrides()
    overrides["moe_top_k"] = 0
    overrides["stage_router_primitives"] = stage_router_primitives(3, 2)

    common_headers = [
        f"parent_track={args.parent_track}",
        f"parent_result_path={parent_path}",
        f"parent_run_phase={parent_phase}",
        "parent_fixed_context_source=fixed_search",
        f"parent_fixed_context={compact_json(base_fixed)}",
        f"v1_best={compact_json(B01_BEST)}",
    ]

    rows: List[Dict[str, Any]] = []
    for offset, spec in enumerate(build_family_specs(), start=1):
        search_space = dict(spec["search_space"])
        rows.append(
            {
                "stage": "stage1",
                "run_axis": RUN_AXIS,
                "dataset": DATASET,
                "model": ROUTE_MODEL,
                "family": "route",
                "family_id": str(spec["family_id"]),
                "job_id": f"BF2_{sanitize_token(DATASET, upper=True)}_{METHOD_ID}_{sanitize_token(spec['family_id'], upper=True)}",
                "run_phase": f"BF2_{sanitize_token(DATASET, upper=True)}_TOPK_{METHOD_ID}_{sanitize_token(spec['family_id'], upper=True)}",
                "seed_id": 1,
                "runtime_seed": int(args.seed_base) + offset,
                "capacity_anchor": f"beauty_focus2_{spec['family_id'].lower()}",
                "source_family_id": "beauty_dense_parent",
                "family_role": "beauty_topk_focus2",
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
                    f"focus2_bank_id={spec['family_id']}",
                    f"focus2_bank_desc={spec['desc']}",
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
    print(f"[beauty-g3k2-focus2] manifest -> {manifest_path}")
    print(f"[beauty-g3k2-focus2] run_count={len(rows)} max_evals_per_bank={args.max_evals}")
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
