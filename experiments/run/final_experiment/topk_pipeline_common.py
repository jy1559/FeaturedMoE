#!/usr/bin/env python3
"""Shared helpers for fast top-k 3-stage pipeline in final_experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from common import CODE_DIR, load_result_payload


ALL_STAGES = ("macro", "mid", "micro")


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
    return [Path(item) for item in json.loads(raw)]


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


def lr_points(base: float, multipliers: Sequence[float]) -> List[float]:
    return unique_sorted([clip(base * mul, 1e-5, 5e-3) for mul in multipliers])


def wd_points(base: float, multipliers: Sequence[float]) -> List[float]:
    anchor = max(float(base), 1e-7)
    return unique_sorted([clip(anchor * mul, 1e-7, 5e-5, digits=10) for mul in multipliers])


def dropout_points(base: float, delta: float) -> List[float]:
    return unique_sorted([
        clip(base - delta, 0.08, 0.28),
        clip(base, 0.08, 0.28),
        clip(base + delta, 0.08, 0.28),
    ])


def reg_points(base: float, floor: float, multipliers: Sequence[float]) -> List[float]:
    anchor = max(float(base), floor)
    return unique_sorted([clip(anchor * mul, floor, 0.01, digits=10) for mul in multipliers])


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
    return {stage: json.loads(json.dumps(base_stage)) for stage in ALL_STAGES}


def stage_router_primitives(group_top_k: int, expert_top_k: int) -> Dict[str, Dict[str, Any]]:
    out = default_stage_router_primitives()
    for stage in ALL_STAGES:
        out[stage]["b_group"]["top_k"] = int(group_top_k)
        out[stage]["d_cond"]["top_k"] = int(expert_top_k)
    return out


TOPK_STRATEGIES: List[Dict[str, Any]] = [
    {
        "method_id": "G3K2",
        "combo_desc": "activate 3 groups + in-group top2",
        "group_top_k": 3,
        "expert_top_k": 2,
        "moe_top_k": 0,
    },
    {
        "method_id": "G4K2",
        "combo_desc": "activate 4 groups + in-group top2",
        "group_top_k": 4,
        "expert_top_k": 2,
        "moe_top_k": 0,
    },
]


def parent_payload_from_selected(row: Dict[str, Any]) -> Dict[str, Any]:
    result_path = first_existing_path(parse_result_paths(row))
    payload = load_result_payload(result_path)
    if not payload:
        raise RuntimeError(f"failed to load parent payload: {result_path}")
    payload["__result_path"] = str(result_path)
    return payload


def build_fixed_context(base_fixed: Dict[str, Any], search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    fixed = dict(base_fixed)
    fixed["expert_scale"] = 3
    for key in search_space:
        fixed.pop(key, None)
    return fixed


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
