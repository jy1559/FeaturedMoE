#!/usr/bin/env python3
"""Compact top-k routing comparison runner for selected final_experiment parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from common import (
    CODE_DIR,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    TRACK,
    load_result_payload,
    log_path_for_row,
    now_utc,
    parse_csv_list,
    run_jobs,
    sanitize_token,
    stage_manifest_path,
    validate_session_fixed_files,
    write_json,
)


DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "beauty",
    "foursquare",
    "lastfm0.03",
    "movielens1m",
    "retail_rocket",
]
ALL_STAGES = ("macro", "mid", "micro")
RUN_AXIS = "topk_test"
ROUTE_MODEL = "featured_moe_n3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact top-k routing comparison for final_experiment")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-run-hours", type=float, default=DEFAULT_MAX_RUN_HOURS)
    parser.add_argument("--oom-retry-limit", type=int, default=DEFAULT_OOM_RETRY_LIMIT)
    parser.add_argument("--seed-base", type=int, default=910000)
    parser.add_argument("--max-evals", type=int, default=4)
    parser.add_argument("--parent-track", default="final_experiment")
    return parser.parse_args()


def selected_configs_path(track: str) -> Path:
    safe_track = str(track or "").strip()
    if not safe_track or safe_track == "final_experiment":
        return CODE_DIR / "selected_configs.json"
    return CODE_DIR / f"selected_configs_{safe_track}.json"


def load_selected_rows(track: str) -> Dict[str, Dict[str, Any]]:
    path = selected_configs_path(track)
    if not path.exists():
        raise FileNotFoundError(f"selected configs not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, dict):
        iterable = rows.values()
    elif isinstance(rows, list):
        iterable = rows
    else:
        raise RuntimeError(f"unexpected selected configs payload format: {path}")

    selected: Dict[str, Dict[str, Any]] = {}
    for row in iterable:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset", "")).strip()
        if dataset:
            selected[dataset] = dict(row)
    if not selected:
        raise RuntimeError(f"no selected config rows found: {path}")
    return selected


def parse_result_paths(row: Dict[str, Any]) -> List[Path]:
    raw = str(row.get("result_paths_json", "")).strip()
    if not raw:
        return []
    try:
        return [Path(item) for item in json.loads(raw)]
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"invalid result_paths_json for dataset={row.get('dataset', '')}: {exc}") from exc


def first_existing_path(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("no existing result path found")


def clip(value: float, low: float, high: float, digits: int = 8) -> float:
    return round(min(max(float(value), low), high), digits)


def unique_sorted(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        token = json.dumps(value, ensure_ascii=True, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        out.append(value)
    try:
        return sorted(out)
    except Exception:
        return out


def lr_points(base: float) -> List[float]:
    return unique_sorted(
        [
            clip(base * 0.75, 1e-5, 5e-3),
            clip(base, 1e-5, 5e-3),
            clip(base * 1.25, 1e-5, 5e-3),
        ]
    )


def wd_points(base: float) -> List[float]:
    anchor = max(float(base), 1e-7)
    return unique_sorted(
        [
            clip(anchor * 0.5, 1e-7, 5e-5, digits=10),
            clip(anchor, 1e-7, 5e-5, digits=10),
            clip(anchor * 2.0, 1e-7, 5e-5, digits=10),
        ]
    )


def dropout_points(base: float) -> List[float]:
    return unique_sorted(
        [
            clip(base - 0.02, 0.08, 0.28),
            clip(base, 0.08, 0.28),
            clip(base + 0.02, 0.08, 0.28),
        ]
    )


def attn_dropout_points(base: float) -> List[float]:
    return unique_sorted(
        [
            clip(base, 0.04, 0.16),
            clip(base + 0.02, 0.04, 0.16),
        ]
    )


def reg_points(base: float, floor: float, boost: float) -> List[float]:
    anchor = max(float(base), floor)
    return unique_sorted([clip(base, floor, 0.01, digits=10), clip(anchor, floor, 0.01, digits=10), clip(anchor * boost, floor, 0.01, digits=10)])


def base_route_overrides() -> Dict[str, Any]:
    return {
        "layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"],
        "stage_compute_mode": {"macro": "moe", "mid": "moe", "micro": "moe"},
        "stage_router_mode": {"macro": "learned", "mid": "learned", "micro": "learned"},
        "stage_router_source": {"macro": "both", "mid": "both", "micro": "both"},
        "stage_feature_injection": {"macro": "none", "mid": "none", "micro": "none"},
        "stage_router_wrapper": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
        "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "balance_loss_lambda": 0.0,
        "gate_entropy_lambda": 0.0,
        "route_smoothness_lambda": 0.0,
        "route_sharpness_lambda": 0.0,
        "route_monopoly_lambda": 0.0,
        "route_monopoly_tau": 0.25,
        "route_prior_lambda": 0.0,
        "group_prior_align_lambda": 0.0,
        "factored_group_balance_lambda": 0.0,
        "rule_agreement_lambda": 0.0,
        "group_coverage_lambda": 0.0,
        "feature_group_bias_lambda": 0.0,
        "rule_bias_scale": 0.0,
        "primitive_balance_lambda": 0.0,
        "wrapper_group_feature_align_lambda": 0.0,
        "macro_history_window": 5,
        "route_consistency_pairs": 1,
        "route_consistency_min_sim": 0.995,
        "bias_mode": "none",
    }


def default_stage_router_primitives() -> Dict[str, Dict[str, Any]]:
    base_stage = {
        "a_joint": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "b_group": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "c_shared": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "d_cond": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "e_scalar": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "wrapper": {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0},
    }
    return {stage: json.loads(json.dumps(base_stage)) for stage in ("macro", "mid", "micro")}


def combo_stage_router_primitives(*, expert_top_k: int, group_top_k: int) -> Dict[str, Dict[str, Any]]:
    primitives = default_stage_router_primitives()
    for stage in ALL_STAGES:
        primitives[stage]["d_cond"]["top_k"] = int(expert_top_k)
        primitives[stage]["b_group"]["top_k"] = int(group_top_k)
    return primitives


def combo_specs() -> List[Dict[str, Any]]:
    return [
        {
            "family_id": "DENSE",
            "combo_desc": "dense routing without top-k pruning",
            "expert_top_k": 0,
            "group_top_k": 0,
            "moe_top_k": 0,
        },
        {
            "family_id": "PG1",
            "combo_desc": "per-group top1 across all stages",
            "expert_top_k": 1,
            "group_top_k": 0,
            "moe_top_k": 0,
        },
        {
            "family_id": "PG2",
            "combo_desc": "per-group top2 across all stages",
            "expert_top_k": 2,
            "group_top_k": 0,
            "moe_top_k": 0,
        },
        {
            "family_id": "GLB4",
            "combo_desc": "global top4 across all experts",
            "expert_top_k": 0,
            "group_top_k": 0,
            "moe_top_k": 4,
        },
        {
            "family_id": "GLB8",
            "combo_desc": "global top8 across all experts",
            "expert_top_k": 0,
            "group_top_k": 0,
            "moe_top_k": 8,
        },
        {
            "family_id": "AG2FULL",
            "combo_desc": "activate 2 groups and use all experts inside them",
            "expert_top_k": 0,
            "group_top_k": 2,
            "moe_top_k": 0,
        },
        {
            "family_id": "AG3K2",
            "combo_desc": "activate 3 groups with per-group top2",
            "expert_top_k": 2,
            "group_top_k": 3,
            "moe_top_k": 0,
        },
        {
            "family_id": "AG2K1",
            "combo_desc": "activate 2 groups with per-group top1",
            "expert_top_k": 1,
            "group_top_k": 2,
            "moe_top_k": 0,
        },
    ]


def build_search_space(base_fixed: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, List[Any]]:
    base_lr = float(base_fixed.get("learning_rate", 5e-4) or 5e-4)
    base_wd = float(base_fixed.get("weight_decay", 1e-6) or 1e-6)
    base_hidden = float(base_fixed.get("hidden_dropout_prob", 0.15) or 0.15)
    base_attn = float(base_fixed.get("attn_dropout_prob", 0.08) or 0.08)
    search = {
        "learning_rate": lr_points(base_lr),
        "weight_decay": wd_points(base_wd),
        "hidden_dropout_prob": dropout_points(base_hidden),
        "fixed_hidden_dropout_prob": dropout_points(float(base_fixed.get("fixed_hidden_dropout_prob", base_hidden) or base_hidden)),
        "attn_dropout_prob": attn_dropout_points(base_attn),
        "route_consistency_lambda": reg_points(float(base_fixed.get("route_consistency_lambda", 0.0) or 0.0), 2.5e-4, 2.0),
        "z_loss_lambda": reg_points(float(base_fixed.get("z_loss_lambda", 0.0) or 0.0), 1e-4, 2.0),
    }
    return search


def build_fixed_context(base_fixed: Dict[str, Any], search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    fixed = dict(base_fixed)
    fixed["expert_scale"] = 3
    for key in search_space:
        fixed.pop(key, None)
    return fixed


def parent_payload_from_selected(row: Dict[str, Any]) -> Dict[str, Any]:
    result_path = first_existing_path(parse_result_paths(row))
    payload = load_result_payload(result_path)
    if not payload:
        raise RuntimeError(f"failed to load parent payload: {result_path}")
    payload["__result_path"] = str(result_path)
    return payload


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


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
        parent_result_path = str(payload.get("__result_path", ""))
        parent_run_phase = str(payload.get("run_phase", ""))
        parent_valid = float(payload.get("best_mrr@20", 0.0) or 0.0)
        parent_test = float(payload.get("test_mrr@20", 0.0) or 0.0)
        selected_rank = int(selected.get("config_rank", 0) or 0)
        selected_seed_count = int(selected.get("seed_count", 0) or 0)
        selected_config_json = str(selected.get("config_json", "")).strip()
        log_header_base = [
            f"parent_track={args.parent_track}",
            f"parent_result_path={parent_result_path}",
            f"parent_run_phase={parent_run_phase}",
            f"parent_selected_rank={selected_rank}",
            f"parent_selected_seed_count={selected_seed_count}",
            f"parent_fixed_context_source=fixed_search",
            f"parent_fixed_context={compact_json(base_fixed)}",
        ]
        if selected_config_json:
            log_header_base.append(f"parent_selected_config_json={selected_config_json}")

        for spec in combo_specs():
            cursor += 1
            search_space = build_search_space(base_fixed, spec)
            fixed_context = build_fixed_context(base_fixed, search_space)
            overrides = base_route_overrides()
            overrides["moe_top_k"] = int(spec["moe_top_k"])
            overrides["stage_router_primitives"] = combo_stage_router_primitives(
                expert_top_k=int(spec["expert_top_k"]),
                group_top_k=int(spec["group_top_k"]),
            )
            log_header_lines = list(log_header_base)
            log_header_lines.extend(
                [
                    f"combo_desc={spec['combo_desc']}",
                    f"combo_family_id={spec['family_id']}",
                    f"combo_expert_top_k={int(spec['expert_top_k'])}",
                    f"combo_group_top_k={int(spec['group_top_k'])}",
                    f"combo_moe_top_k={int(spec['moe_top_k'])}",
                ]
            )
            rows.append(
                {
                    "stage": "topk_test",
                    "run_axis": RUN_AXIS,
                    "dataset": dataset,
                    "model": ROUTE_MODEL,
                    "family": "route",
                    "family_id": str(spec["family_id"]),
                    "job_id": f"TOPK_{sanitize_token(dataset, upper=True)}_{sanitize_token(spec['family_id'], upper=True)}",
                    "run_phase": f"TOPK_{sanitize_token(dataset, upper=True)}_{sanitize_token(ROUTE_MODEL, upper=True)}_{sanitize_token(spec['family_id'], upper=True)}",
                    "seed_id": 1,
                    "runtime_seed": int(args.seed_base) + cursor,
                    "capacity_anchor": f"parent_{args.parent_track}",
                    "source_family_id": "selected_parent",
                    "family_role": "topk_probe",
                    "history_valid": parent_valid,
                    "history_test": parent_test,
                    "selection_reason": str(spec["combo_desc"]),
                    "parent_result_path": parent_result_path,
                    "parent_run_phase": parent_run_phase,
                    "parent_selected_rank": selected_rank,
                    "parent_selected_seed_count": selected_seed_count,
                    "parent_selected_config_json": selected_config_json,
                    "search_space": search_space,
                    "fixed_context": fixed_context,
                    "overrides": overrides,
                    "log_header_lines": log_header_lines,
                    "max_evals": int(args.max_evals),
                    "max_run_hours": float(args.max_run_hours),
                    "oom_retry_limit": int(args.oom_retry_limit),
                    "combo_desc": str(spec["combo_desc"]),
                    "expert_top_k": int(spec["expert_top_k"]),
                    "group_top_k": int(spec["group_top_k"]),
                    "moe_top_k": int(spec["moe_top_k"]),
                    "apply_stages": list(ALL_STAGES),
                }
            )
    return rows


def write_stage_manifest(rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path("topk_test")
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
            "selection_reason": row.get("selection_reason", ""),
            "parent_result_path": row.get("parent_result_path", ""),
            "parent_run_phase": row.get("parent_run_phase", ""),
            "parent_selected_rank": row.get("parent_selected_rank", 0),
            "parent_selected_seed_count": row.get("parent_selected_seed_count", 0),
            "parent_selected_config_json": row.get("parent_selected_config_json", ""),
            "combo_desc": row.get("combo_desc", ""),
            "expert_top_k": row.get("expert_top_k", 0),
            "group_top_k": row.get("group_top_k", 0),
            "moe_top_k": row.get("moe_top_k", 0),
            "apply_stages": row.get("apply_stages", []),
            "search_space": row.get("search_space", {}),
            "fixed_context": row.get("fixed_context", {}),
            "overrides": row.get("overrides", {}),
            "log_header_lines": row.get("log_header_lines", []),
            "log_path": str(log_path_for_row("topk_test", row)),
        }
        for row in rows
    ]
    payload = {
        "generated_at": now_utc(),
        "stage": "topk_test",
        "run_axis": RUN_AXIS,
        "track": TRACK,
        "run_count": len(manifest_rows),
        "rows": manifest_rows,
    }
    write_json(path, payload)
    return path


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest_path = write_stage_manifest(rows)
    print(f"[topk_test] manifest -> {manifest_path}")
    print(f"[topk_test] run_count={len(rows)}")
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return run_jobs(
        rows,
        stage="topk_test",
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())