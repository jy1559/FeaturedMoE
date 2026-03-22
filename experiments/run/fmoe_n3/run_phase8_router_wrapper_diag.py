#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase8 router/wrapper + diagnostics experiments.

Phase8 pipeline (KuaiRec-first, resume-safe):
- Stage A (wrapper core): all_w1..all_w6 + mixed_1..mixed_3
- Stage B (bias augmentation): top-A wrappers x {off, feature, rule, both}
- Stage C (primitive source): top-B settings x source profiles
- Stage D (top-k refinement): top-C settings x primitive/final top-k profiles
- Confirm: top-D finalists x multi-seed rerun

Design goals:
- Keep wrapper/primitive config in new schema only:
  stage_router_wrapper + stage_router_primitives
- Robust to interruption:
  skip completed runs by log/result scan and continue queue
- Stage-wise selection from actual result JSON (best_mrr@20 primary)
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"
RESULT_ROOT = ARTIFACT_ROOT / "results" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase8_router_wrapper_diag_v1"
PHASE = "P8"
MODEL_TAG = "FMoEN3"

STAGES = ("macro", "mid", "micro")
PRIMITIVES = ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar")


def hydra_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Invalid float for hydra literal: {value}")
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        items = [f"{k}:{hydra_literal(v)}" for k, v in value.items()]
        return "{" + ",".join(items) + "}"
    raise TypeError(f"Unsupported hydra literal type: {type(value).__name__}")


def sanitize_slug(value: str) -> str:
    text = str(value or "").strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "run"


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _parse_csv_ints(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text or "").split(",") if tok.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [tok.strip() for tok in str(text or "").split(",") if tok.strip()]


def _primitive_default_sources() -> Dict[str, str]:
    return {
        "a_joint": "both",
        "b_group": "both",
        "c_shared": "both",
        "d_cond": "feature",
        "e_scalar": "feature",
    }


def _build_stage_router_primitives(
    *,
    sources: Optional[Dict[str, str]] = None,
    temperatures: Optional[Dict[str, float]] = None,
    top_ks: Optional[Dict[str, int]] = None,
    wrapper_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    base_sources = _primitive_default_sources()
    if isinstance(sources, dict):
        for key, val in sources.items():
            if key in PRIMITIVES and val is not None:
                base_sources[key] = str(val)

    base_temps = {key: 1.0 for key in PRIMITIVES}
    if isinstance(temperatures, dict):
        for key, val in temperatures.items():
            if key in PRIMITIVES and val is not None:
                base_temps[key] = float(val)

    base_topk = {key: 0 for key in PRIMITIVES}
    if isinstance(top_ks, dict):
        for key, val in top_ks.items():
            if key in PRIMITIVES and val is not None:
                base_topk[key] = int(val)

    wrapper_cfg = {
        "alpha_d": 1.0,
        "alpha_struct": 1.0,
        "alpha_a": 1.0,
    }
    if isinstance(wrapper_params, dict):
        for key, val in wrapper_params.items():
            wrapper_cfg[str(key)] = float(val)

    out: Dict[str, Dict[str, Any]] = {}
    for stage_name in STAGES:
        stage_cfg: Dict[str, Any] = {}
        for primitive in PRIMITIVES:
            stage_cfg[primitive] = {
                "source": str(base_sources[primitive]),
                "temperature": float(base_temps[primitive]),
                "top_k": int(base_topk[primitive]),
            }
        stage_cfg["wrapper"] = dict(wrapper_cfg)
        out[stage_name] = stage_cfg
    return out


def _copy_stage_router_primitives(primitives: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(primitives or {}))


def _set_all_stage_wrapper(overrides: Dict[str, Any], wrapper_name: str) -> None:
    overrides["stage_router_wrapper"] = _all_stage_map(str(wrapper_name))


def _set_stage_specific_wrapper(overrides: Dict[str, Any], wrapper_map: Dict[str, str]) -> None:
    stage_wrapper = {}
    for stage in STAGES:
        stage_wrapper[stage] = str(wrapper_map.get(stage, "w1_flat"))
    overrides["stage_router_wrapper"] = stage_wrapper


def _update_primitive_sources(primitives: Dict[str, Any], source_map: Dict[str, str]) -> Dict[str, Any]:
    out = _copy_stage_router_primitives(primitives)
    for stage in STAGES:
        stage_cfg = dict(out.get(stage, {}))
        for primitive, source in source_map.items():
            if primitive not in PRIMITIVES:
                continue
            raw = dict(stage_cfg.get(primitive, {}))
            raw["source"] = str(source)
            stage_cfg[primitive] = raw
        out[stage] = stage_cfg
    return out


def _update_primitive_top_k(primitives: Dict[str, Any], top_k_map: Dict[str, int]) -> Dict[str, Any]:
    out = _copy_stage_router_primitives(primitives)
    for stage in STAGES:
        stage_cfg = dict(out.get(stage, {}))
        for primitive in PRIMITIVES:
            raw = dict(stage_cfg.get(primitive, {}))
            raw["top_k"] = int(top_k_map.get(primitive, 0))
            stage_cfg[primitive] = raw
        out[stage] = stage_cfg
    return out


def _base_fixed_overrides() -> Dict[str, Any]:
    return {
        "layer_layout": ["macro", "mid", "micro"],
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": _all_stage_map("both"),
        "stage_feature_injection": _all_stage_map("none"),
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "balance_loss_lambda": 0.002,
        "z_loss_lambda": 1e-4,
        "route_smoothness_lambda": 0.01,
        "route_consistency_lambda": 0.0,
        "route_sharpness_lambda": 0.0,
        "route_monopoly_lambda": 0.0,
        "route_monopoly_tau": 0.25,
        "route_prior_lambda": 0.0,
        "group_prior_align_lambda": 0.0,
        "factored_group_balance_lambda": 0.0,
        "feature_group_bias_lambda": 0.0,
        "rule_bias_scale": 0.0,
        "stage_router_primitives": _build_stage_router_primitives(),
    }


def _wrapper_candidates_all_and_mixed() -> Dict[str, Dict[str, str]]:
    return {
        "all_w1": {"macro": "w1_flat", "mid": "w1_flat", "micro": "w1_flat"},
        "all_w2": {"macro": "w2_a_plus_d", "mid": "w2_a_plus_d", "micro": "w2_a_plus_d"},
        "all_w3": {"macro": "w3_bxc", "mid": "w3_bxc", "micro": "w3_bxc"},
        "all_w4": {"macro": "w4_bxd", "mid": "w4_bxd", "micro": "w4_bxd"},
        "all_w5": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
        "all_w6": {"macro": "w6_bxd_plus_a", "mid": "w6_bxd_plus_a", "micro": "w6_bxd_plus_a"},
        # mixed 3 (stage-specific wrapper families)
        "mixed_1": {"macro": "w4_bxd", "mid": "w4_bxd", "micro": "w1_flat"},
        "mixed_2": {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
        "mixed_3": {"macro": "w6_bxd_plus_a", "mid": "w1_flat", "micro": "w1_flat"},
    }


def _stage_a_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    allow = {tok.lower() for tok in _parse_csv_strings(args.only_a_candidates)}
    settings: list[Dict[str, Any]] = []
    for combo_name, wrapper_map in _wrapper_candidates_all_and_mixed().items():
        if allow and combo_name.lower() not in allow:
            continue
        overrides = _base_fixed_overrides()
        _set_stage_specific_wrapper(overrides, wrapper_map)
        overrides["feature_group_bias_lambda"] = 0.0
        overrides["rule_bias_scale"] = 0.0
        overrides["route_prior_lambda"] = 0.0
        settings.append(
            {
                "stage": "A",
                "setting_id": f"A_{combo_name.upper()}",
                "setting_key": combo_name,
                "setting_desc": f"wrapper_core::{combo_name}",
                "overrides": overrides,
                "meta": {
                    "wrapper_combo": combo_name,
                    "bias_mode": "off",
                    "source_profile": "default",
                    "topk_profile": "dense",
                    "parent_id": "",
                },
            }
        )
    return settings


def _stage_b_settings(args: argparse.Namespace, parents: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    feature_bias = float(args.feature_group_bias_lambda)
    rule_bias = float(args.rule_bias_scale)
    modes = {
        "bias_off": (0.0, 0.0),
        "bias_feat": (feature_bias, 0.0),
        # Explicit group-feature prior bias mode (kept for readability in logs/selections).
        "bias_group_feat": (feature_bias, 0.0),
        "bias_rule": (0.0, rule_bias),
        "bias_both": (feature_bias, rule_bias),
        "bias_group_feat_rule": (feature_bias, rule_bias),
    }

    settings: list[Dict[str, Any]] = []
    for parent in parents:
        parent_id = str(parent["setting_id"])
        parent_key = str(parent["setting_key"])
        for mode_name, (feat_lambda, rule_scale) in modes.items():
            overrides = copy.deepcopy(parent["overrides"])
            overrides["feature_group_bias_lambda"] = float(feat_lambda)
            overrides["rule_bias_scale"] = float(rule_scale)
            settings.append(
                {
                    "stage": "B",
                    "setting_id": f"B_{parent_key.upper()}_{mode_name.upper()}",
                    "setting_key": f"{parent_key}__{mode_name}",
                    "setting_desc": f"bias_aug::{parent_key}::{mode_name}",
                    "overrides": overrides,
                    "meta": {
                        "wrapper_combo": parent["meta"].get("wrapper_combo", parent_key),
                        "bias_mode": mode_name,
                        "source_profile": parent["meta"].get("source_profile", "default"),
                        "topk_profile": parent["meta"].get("topk_profile", "dense"),
                        "parent_id": parent_id,
                    },
                }
            )
    return settings


def _source_profiles() -> Dict[str, Dict[str, str]]:
    return {
        "src_base": {
            "a_joint": "both",
            "b_group": "both",
            "c_shared": "both",
            "d_cond": "feature",
            "e_scalar": "feature",
        },
        "src_all_both": {
            "a_joint": "both",
            "b_group": "both",
            "c_shared": "both",
            "d_cond": "both",
            "e_scalar": "both",
        },
        "src_a_hidden_b_d_feature": {
            "a_joint": "hidden",
            "b_group": "feature",
            "c_shared": "both",
            "d_cond": "feature",
            "e_scalar": "feature",
        },
        "src_abc_feature": {
            "a_joint": "feature",
            "b_group": "feature",
            "c_shared": "feature",
            "d_cond": "feature",
            "e_scalar": "feature",
        },
    }


def _stage_c_settings(args: argparse.Namespace, parents: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    del args
    settings: list[Dict[str, Any]] = []
    profiles = _source_profiles()
    for parent in parents:
        parent_key = str(parent["setting_key"])
        parent_id = str(parent["setting_id"])
        for profile_name, source_map in profiles.items():
            overrides = copy.deepcopy(parent["overrides"])
            primitives = _update_primitive_sources(overrides.get("stage_router_primitives", {}), source_map)
            overrides["stage_router_primitives"] = primitives
            settings.append(
                {
                    "stage": "C",
                    "setting_id": f"C_{sanitize_slug(parent_key).upper()}_{profile_name.upper()}",
                    "setting_key": f"{parent_key}__{profile_name}",
                    "setting_desc": f"source::{parent_key}::{profile_name}",
                    "overrides": overrides,
                    "meta": {
                        "wrapper_combo": parent["meta"].get("wrapper_combo", "unknown"),
                        "bias_mode": parent["meta"].get("bias_mode", "off"),
                        "source_profile": profile_name,
                        "topk_profile": parent["meta"].get("topk_profile", "dense"),
                        "parent_id": parent_id,
                    },
                }
            )
    return settings


def _topk_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "tk_dense": {
            "final_top_k": 0,
            "primitive_top_k": {
                "a_joint": 0,
                "b_group": 0,
                "c_shared": 0,
                "d_cond": 0,
                "e_scalar": 0,
            },
        },
        "tk_d1": {
            "final_top_k": 0,
            "primitive_top_k": {
                "a_joint": 0,
                "b_group": 0,
                "c_shared": 0,
                "d_cond": 1,
                "e_scalar": 0,
            },
        },
        "tk_d1_final4": {
            "final_top_k": 4,
            "primitive_top_k": {
                "a_joint": 0,
                "b_group": 0,
                "c_shared": 0,
                "d_cond": 1,
                "e_scalar": 0,
            },
        },
        "tk_a3_d1_final4": {
            "final_top_k": 4,
            "primitive_top_k": {
                "a_joint": 3,
                "b_group": 0,
                "c_shared": 0,
                "d_cond": 1,
                "e_scalar": 0,
            },
        },
    }


def _stage_d_settings(args: argparse.Namespace, parents: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    del args
    settings: list[Dict[str, Any]] = []
    for parent in parents:
        parent_key = str(parent["setting_key"])
        parent_id = str(parent["setting_id"])
        for profile_name, profile_cfg in _topk_profiles().items():
            overrides = copy.deepcopy(parent["overrides"])
            overrides["moe_top_k"] = int(profile_cfg["final_top_k"])
            overrides["topk_scope_mode"] = "global_flat"
            primitives = _update_primitive_top_k(
                overrides.get("stage_router_primitives", {}),
                dict(profile_cfg.get("primitive_top_k", {}) or {}),
            )
            overrides["stage_router_primitives"] = primitives
            settings.append(
                {
                    "stage": "D",
                    "setting_id": f"D_{sanitize_slug(parent_key).upper()}_{profile_name.upper()}",
                    "setting_key": f"{parent_key}__{profile_name}",
                    "setting_desc": f"topk::{parent_key}::{profile_name}",
                    "overrides": overrides,
                    "meta": {
                        "wrapper_combo": parent["meta"].get("wrapper_combo", "unknown"),
                        "bias_mode": parent["meta"].get("bias_mode", "off"),
                        "source_profile": parent["meta"].get("source_profile", "default"),
                        "topk_profile": profile_name,
                        "parent_id": parent_id,
                    },
                }
            )
    return settings


def _run_phase_name(stage: str, pass_tag: str, setting_key: str, seed_id: int) -> str:
    return f"P8_{pass_tag}_{stage}_{sanitize_slug(setting_key).upper()}_S{int(seed_id)}"


def _expand_rows(
    *,
    dataset: str,
    settings: list[Dict[str, Any]],
    seed_ids: list[int],
    pass_tag: str,
    seed_cursor: int,
) -> tuple[list[Dict[str, Any]], int]:
    rows: list[Dict[str, Any]] = []
    cursor = int(seed_cursor)
    for setting in settings:
        stage = str(setting.get("stage", "X")).upper()
        for seed_id in seed_ids:
            run_phase = _run_phase_name(stage=stage, pass_tag=pass_tag, setting_key=setting["setting_key"], seed_id=seed_id)
            rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "pass_tag": pass_tag,
                    "seed_id": int(seed_id),
                    "seed_offset": int(cursor),
                    "run_phase": run_phase,
                    "setting_id": setting["setting_id"],
                    "setting_key": setting["setting_key"],
                    "setting_desc": setting["setting_desc"],
                    "overrides": copy.deepcopy(setting["overrides"]),
                    "meta": copy.deepcopy(setting.get("meta", {})),
                }
            )
            cursor += 1
    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
    return rows, cursor


def _phase8_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset) / MODEL_TAG


def _phase8_axis_dataset_dir(dataset: str) -> Path:
    root = LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _phase8_summary_csv_path(dataset: str) -> Path:
    return _phase8_axis_dataset_dir(dataset) / "summary.csv"


def _summary_fieldnames() -> list[str]:
    return [
        # Left-most requested columns.
        "global_best_valid_mrr20",
        "run_best_valid_mrr20",
        # Quick human-readable identifiers.
        "run_phase",
        "exp_brief",
        "stage",
        "pass_tag",
        "setting_id",
        "setting_key",
        "setting_desc",
        "wrapper_combo",
        "bias_mode",
        "source_profile",
        "topk_profile",
        # Meta + metrics.
        "trigger",
        "dataset",
        "seed_id",
        "gpu_id",
        "test_mrr20",
        "best_valid_hr10",
        "test_hr10",
        "n_completed",
        "interrupted",
        "result_path",
        "timestamp_utc",
    ]


def _ensure_summary_csv(path: Path) -> None:
    expected = ",".join(_summary_fieldnames())
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            first = ""
        if first == expected:
            return
        # Back up legacy/incompatible schema, then recreate.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.legacy_{ts}{path.suffix}")
        try:
            path.rename(backup)
        except Exception:
            pass
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writeheader()


def _summary_format_metric(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{int(ndigits)}f}"


def _summary_exp_brief(row: Dict[str, Any]) -> str:
    meta = dict(row.get("meta", {}) or {})
    stage = str(row.get("stage", "") or "")
    key = str(row.get("setting_key", "") or "")
    wrapper = str(meta.get("wrapper_combo", "") or "")
    bias = str(meta.get("bias_mode", "") or "")
    source = str(meta.get("source_profile", "") or "")
    topk = str(meta.get("topk_profile", "") or "")
    return f"{stage}:{key} | w={wrapper}, b={bias}, s={source}, k={topk}"


def _load_summary_state(path: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "global_best_valid": None,
        "run_complete_written": set(),
    }
    if not path.exists():
        return state
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trigger = str(row.get("trigger", "")).strip().lower()
            run_phase = str(row.get("run_phase", "")).strip()
            if trigger == "run_complete" and run_phase:
                state["run_complete_written"].add(run_phase)
            global_col = _metric_to_float(row.get("global_best_valid_mrr20"))
            run_col = _metric_to_float(row.get("run_best_valid_mrr20"))
            for valid in (global_col, run_col):
                if valid is None:
                    continue
                if state["global_best_valid"] is None or valid > float(state["global_best_valid"]):
                    state["global_best_valid"] = float(valid)
    return state


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path)
    payload = {key: row.get(key, "") for key in _summary_fieldnames()}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writerow(payload)


def _summary_row_base(
    *,
    row: Dict[str, Any],
    trigger: str,
    global_best_valid: Optional[float],
    run_best_valid: Optional[float],
    test_mrr20: Optional[float],
    best_valid_hr10: Optional[float],
    test_hr10: Optional[float],
    n_completed: Optional[int],
    interrupted: Optional[bool],
    result_path: str,
) -> Dict[str, Any]:
    meta = dict(row.get("meta", {}) or {})
    return {
        "global_best_valid_mrr20": _summary_format_metric(global_best_valid),
        "run_best_valid_mrr20": _summary_format_metric(run_best_valid),
        "run_phase": row.get("run_phase", ""),
        "exp_brief": _summary_exp_brief(row),
        "stage": row.get("stage", ""),
        "pass_tag": row.get("pass_tag", ""),
        "setting_id": row.get("setting_id", ""),
        "setting_key": row.get("setting_key", ""),
        "setting_desc": row.get("setting_desc", ""),
        "wrapper_combo": meta.get("wrapper_combo", ""),
        "bias_mode": meta.get("bias_mode", ""),
        "source_profile": meta.get("source_profile", ""),
        "topk_profile": meta.get("topk_profile", ""),
        "trigger": str(trigger),
        "dataset": row.get("dataset", ""),
        "seed_id": row.get("seed_id", ""),
        "gpu_id": row.get("assigned_gpu", ""),
        "test_mrr20": _summary_format_metric(test_mrr20),
        "best_valid_hr10": _summary_format_metric(best_valid_hr10),
        "test_hr10": _summary_format_metric(test_hr10),
        "n_completed": "" if n_completed is None else int(n_completed),
        "interrupted": "" if interrupted is None else bool(interrupted),
        "result_path": result_path,
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _get_result_row_for_run_phase(dataset: str, run_phase: str, retries: int = 5, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        index = _load_result_index(dataset, AXIS)
        row = index.get(str(run_phase))
        if isinstance(row, dict) and _metric_to_float(row.get("best_mrr")) is not None:
            return row
        time.sleep(max(float(sleep_sec), 0.0))
    return None


_TRIAL_METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")


def _parse_trial_metrics_line(line: str) -> Optional[Dict[str, float]]:
    text = str(line or "")
    if "[TRIAL_METRICS]" not in text:
        return None
    metrics: Dict[str, float] = {}
    for key, raw_val in _TRIAL_METRIC_KV_PATTERN.findall(text):
        val = _metric_to_float(raw_val)
        if val is None:
            continue
        metrics[str(key)] = float(val)
    return metrics or None


def _scan_trial_metric_updates(log_path: Path, start_offset: int) -> tuple[int, list[Dict[str, float]]]:
    updates: list[Dict[str, float]] = []
    new_offset = int(start_offset)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(max(int(start_offset), 0))
            while True:
                line = fh.readline()
                if not line:
                    break
                parsed = _parse_trial_metrics_line(line)
                if parsed:
                    updates.append(parsed)
            new_offset = fh.tell()
    except Exception:
        return int(start_offset), []
    return new_offset, updates


def _record_trial_new_best_if_any(
    *,
    summary_path: Path,
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    trial_metrics: Dict[str, float],
) -> None:
    run_best = _metric_to_float(trial_metrics.get("run_best_mrr20"))
    if run_best is None:
        run_best = _metric_to_float(trial_metrics.get("cur_best_mrr20"))
    if run_best is None:
        return
    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    if prev_global is not None and run_best <= prev_global:
        return
    new_global = run_best if prev_global is None else max(prev_global, run_best)
    payload = _summary_row_base(
        row=row,
        trigger="trial_new_best",
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=_metric_to_float(trial_metrics.get("run_test_mrr20")),
        best_valid_hr10=_metric_to_float(trial_metrics.get("run_best_hr10")),
        test_hr10=_metric_to_float(trial_metrics.get("run_test_hr10")),
        n_completed=None,
        interrupted=None,
        result_path="",
    )
    _append_summary_row(summary_path, payload)
    summary_state["global_best_valid"] = float(new_global)


def _record_run_complete_summary(
    *,
    dataset: str,
    summary_path: Path,
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
) -> None:
    run_phase = str(row.get("run_phase", "") or "")
    if not run_phase:
        return
    run_complete_written = summary_state.get("run_complete_written")
    if isinstance(run_complete_written, set) and run_phase in run_complete_written:
        return
    result_row = _get_result_row_for_run_phase(dataset, run_phase, retries=8, sleep_sec=1.0)
    if not isinstance(result_row, dict):
        return
    run_best = _metric_to_float(result_row.get("best_mrr"))
    if run_best is None:
        return
    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    new_global = run_best if prev_global is None else max(prev_global, run_best)
    payload = _summary_row_base(
        row=row,
        trigger="run_complete",
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=_metric_to_float(result_row.get("test_mrr")),
        best_valid_hr10=_metric_to_float(result_row.get("best_hr10")),
        test_hr10=_metric_to_float(result_row.get("test_hr10")),
        n_completed=int(result_row.get("n_completed", 0) or 0),
        interrupted=bool(result_row.get("interrupted", False)),
        result_path=str(result_row.get("path", "") or ""),
    )
    _append_summary_row(summary_path, payload)
    summary_state["global_best_valid"] = float(new_global)
    if not isinstance(run_complete_written, set):
        summary_state["run_complete_written"] = set()
    summary_state["run_complete_written"].add(run_phase)


def _phase_log_prefix(stage: str) -> str:
    text = sanitize_slug(str(stage or "X")).upper()
    return text or "X"


def _parse_log_phase_and_index(log_path: Path) -> tuple[str, Optional[int]]:
    stem = log_path.stem
    parts = stem.split("_", 2)
    if len(parts) < 2:
        return "", None

    # New format: A_001_...
    first = sanitize_slug(parts[0]).upper()
    try:
        index = int(parts[1])
    except Exception:
        index = None
    if first and index is not None:
        return first, index

    # Legacy format: 001_A_...
    try:
        legacy_index = int(parts[0])
    except Exception:
        legacy_index = None
    legacy_phase = sanitize_slug(parts[1]).upper()
    if legacy_phase and legacy_index is not None:
        return legacy_phase, legacy_index
    return "", None


def _next_log_index_by_phase(dataset: str) -> Dict[str, int]:
    root = _phase8_log_dir(dataset)
    next_index: Dict[str, int] = {}
    if not root.exists():
        return next_index
    for log_path in sorted(root.glob("*.log")):
        phase_prefix, index = _parse_log_phase_and_index(log_path)
        if not phase_prefix or index is None:
            continue
        next_index[phase_prefix] = max(int(next_index.get(phase_prefix, 1)), int(index) + 1)
    return next_index


def _make_log_stem(row: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase_prefix = _phase_log_prefix(str(row.get("stage", "X")))
    return (
        f"{phase_prefix}_"
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row.get('pass_tag', 'UNK'))}_"
        f"{sanitize_slug(row.get('setting_key', 'run'))}_"
        f"s{int(row.get('seed_id', 0))}_"
        f"{ts}"
    )


def _log_path(row: Dict[str, Any], dataset: str) -> Path:
    root = _phase8_log_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    stem = _make_log_stem(row)
    out_path = root / f"{stem}.log"
    if not out_path.exists():
        out_path.touch(exist_ok=False)
        return out_path
    retry_idx = 2
    while True:
        candidate = root / f"{stem}_r{retry_idx:02d}.log"
        if not candidate.exists():
            candidate.touch(exist_ok=False)
            return candidate
        retry_idx += 1


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--seed",
        str(int(args.seed_base) + int(row["seed_offset"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        row["run_phase"],
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(args.embedding_size)}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(args.d_ff)}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(args.d_expert_hidden)}",
        f"d_router_hidden={int(args.d_router_hidden)}",
        f"expert_scale={int(args.expert_scale)}",
        "++layer_layout=[macro,mid,micro]",
        f"++search.learning_rate={hydra_literal([float(args.search_lr_min), float(args.search_lr_max)])}",
        f"++search.weight_decay={hydra_literal([float(args.fixed_weight_decay)])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(args.fixed_hidden_dropout_prob)])}",
        f"++search.lr_scheduler_type={hydra_literal(_parse_csv_strings(args.search_lr_scheduler))}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++p8_stage={hydra_literal(row['stage'])}",
        f"++p8_pass={hydra_literal(row['pass_tag'])}",
        f"++p8_setting_id={hydra_literal(row['setting_id'])}",
        f"++p8_setting_key={hydra_literal(row['setting_key'])}",
        f"++p8_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++p8_parent_id={hydra_literal(row.get('meta', {}).get('parent_id', ''))}",
        f"++p8_wrapper_combo={hydra_literal(row.get('meta', {}).get('wrapper_combo', ''))}",
        f"++p8_bias_mode={hydra_literal(row.get('meta', {}).get('bias_mode', ''))}",
        f"++p8_source_profile={hydra_literal(row.get('meta', {}).get('source_profile', ''))}",
        f"++p8_topk_profile={hydra_literal(row.get('meta', {}).get('topk_profile', ''))}",
    ]

    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _write_log_preamble(
    log_file: Path,
    row: Dict[str, Any],
    gpu_id: str,
    args: argparse.Namespace,
    cmd: list[str],
) -> None:
    lines = [
        "[PHASE8_SETTING_HEADER]",
        f"run_phase={row['run_phase']} setting={row['setting_id']} stage={row['stage']} pass={row['pass_tag']} seed={row['seed_id']}",
        f"key={row['setting_key']}",
        f"desc={row['setting_desc']}",
        f"wrapper_combo={row.get('meta', {}).get('wrapper_combo', '')} bias_mode={row.get('meta', {}).get('bias_mode', '')}",
        f"source_profile={row.get('meta', {}).get('source_profile', '')} topk_profile={row.get('meta', {}).get('topk_profile', '')}",
        f"dataset={row['dataset']} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _extract_run_phase_from_log(log_path: Path) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(14):
                line = fh.readline()
                if not line:
                    break
                if line.startswith("run_phase="):
                    token = line.split()[0]
                    return token.split("=", 1)[1].strip()
    except Exception:
        return ""
    return ""


def _is_completed_log(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("[RUN_STATUS] END status=normal" in text) or ("\n  DONE  |  FeaturedMoE_N3 x " in text)


def _scan_completed_run_phases(dataset: str) -> set[str]:
    done: set[str] = set()
    root = _phase8_log_dir(dataset)
    if not root.exists():
        return done
    for log_path in sorted(root.glob("*.log")):
        run_phase = _extract_run_phase_from_log(log_path)
        if not run_phase:
            continue
        if _is_completed_log(log_path):
            done.add(run_phase)
    return done


def _metric_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except Exception:
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def _extract_valid_mrr(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("best_mrr@20", "best_valid_mrr@20", "best_valid_score"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    bvr = payload.get("best_valid_result")
    if isinstance(bvr, dict):
        for key in ("mrr@20", "MRR@20"):
            val = _metric_to_float(bvr.get(key))
            if val is not None:
                return val
    return None


def _extract_test_mrr(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("test_mrr@20", "best_test_mrr@20", "test_score"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    test = payload.get("test_result")
    if isinstance(test, dict):
        for key in ("mrr@20", "MRR@20"):
            val = _metric_to_float(test.get(key))
            if val is not None:
                return val
    return None


def _extract_valid_hr10(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("best_hr@10", "best_valid_hr@10"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    bvr = payload.get("best_valid_result")
    if isinstance(bvr, dict):
        for key in ("hr@10", "HR@10"):
            val = _metric_to_float(bvr.get(key))
            if val is not None:
                return val
    return None


def _extract_test_hr10(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("test_hr@10", "best_test_hr@10"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    test = payload.get("test_result")
    if isinstance(test, dict):
        for key in ("hr@10", "HR@10"):
            val = _metric_to_float(test.get(key))
            if val is not None:
                return val
    return None


def _load_result_index(dataset: str, axis: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not RESULT_ROOT.exists():
        return out
    for path in RESULT_ROOT.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_axis", "")) != str(axis):
            continue
        if str(payload.get("dataset", "")) != str(dataset):
            continue
        run_phase = str(payload.get("run_phase", "")).strip()
        if not run_phase:
            continue

        best_mrr = _extract_valid_mrr(payload)
        test_mrr = _extract_test_mrr(payload)
        best_hr10 = _extract_valid_hr10(payload)
        test_hr10 = _extract_test_hr10(payload)
        n_completed = int(payload.get("n_completed", 0) or 0)
        interrupted = bool(payload.get("interrupted", False))
        mtime = float(path.stat().st_mtime)

        candidate = {
            "run_phase": run_phase,
            "best_mrr": best_mrr,
            "test_mrr": test_mrr,
            "best_hr10": best_hr10,
            "test_hr10": test_hr10,
            "n_completed": n_completed,
            "interrupted": interrupted,
            "path": str(path),
            "mtime": mtime,
        }

        prev = out.get(run_phase)
        if prev is None:
            out[run_phase] = candidate
            continue

        prev_score = _metric_to_float(prev.get("best_mrr"))
        cur_score = _metric_to_float(candidate.get("best_mrr"))
        if cur_score is not None and (prev_score is None or cur_score > prev_score):
            out[run_phase] = candidate
            continue
        if cur_score == prev_score and mtime >= float(prev.get("mtime", 0.0)):
            out[run_phase] = candidate
            continue
        if prev_score is None and cur_score is None and mtime >= float(prev.get("mtime", 0.0)):
            out[run_phase] = candidate
    return out


def _completed_by_result(result_index: Dict[str, Dict[str, Any]]) -> set[str]:
    done = set()
    for run_phase, rec in result_index.items():
        if _metric_to_float(rec.get("best_mrr")) is not None and int(rec.get("n_completed", 0)) > 0:
            done.add(run_phase)
    return done


def _summarize_stage(
    *,
    settings: list[Dict[str, Any]],
    rows: list[Dict[str, Any]],
    result_index: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    rows_by_setting: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_setting[str(row["setting_id"])].append(row)

    out: list[Dict[str, Any]] = []
    for setting in settings:
        setting_id = str(setting["setting_id"])
        row_list = rows_by_setting.get(setting_id, [])
        valid_scores: list[float] = []
        test_scores: list[float] = []
        completed = 0
        run_phases: list[str] = []
        for row in row_list:
            run_phase = str(row["run_phase"])
            run_phases.append(run_phase)
            rec = result_index.get(run_phase)
            if not isinstance(rec, dict):
                continue
            score = _metric_to_float(rec.get("best_mrr"))
            test = _metric_to_float(rec.get("test_mrr"))
            if score is not None:
                valid_scores.append(score)
                completed += 1
            if test is not None:
                test_scores.append(test)

        valid_mean = statistics.fmean(valid_scores) if valid_scores else None
        valid_best = max(valid_scores) if valid_scores else None
        test_mean = statistics.fmean(test_scores) if test_scores else None

        out.append(
            {
                "stage": setting.get("stage", ""),
                "setting_id": setting_id,
                "setting_key": setting.get("setting_key", ""),
                "setting_desc": setting.get("setting_desc", ""),
                "n_expected": len(row_list),
                "n_scored": completed,
                "valid_mean": valid_mean,
                "valid_best": valid_best,
                "test_mean": test_mean,
                "run_phases": run_phases,
                "meta": copy.deepcopy(setting.get("meta", {})),
            }
        )
    return out


def _select_top_setting_ids(
    *,
    settings: list[Dict[str, Any]],
    summaries: list[Dict[str, Any]],
    top_n: int,
    allow_fallback: bool,
) -> list[str]:
    top_n = max(int(top_n), 0)
    if top_n <= 0:
        return []

    scored = [s for s in summaries if int(s.get("n_scored", 0)) > 0 and _metric_to_float(s.get("valid_mean")) is not None]
    if scored:
        scored.sort(
            key=lambda x: (
                int(x.get("n_scored", 0)),
                float(x.get("valid_mean", -1e9) if x.get("valid_mean") is not None else -1e9),
                float(x.get("test_mean", -1e9) if x.get("test_mean") is not None else -1e9),
                str(x.get("setting_id", "")),
            ),
            reverse=True,
        )
        return [str(item["setting_id"]) for item in scored[:top_n]]

    if allow_fallback:
        return [str(s["setting_id"]) for s in settings[:top_n]]
    return []


def _pick_settings_by_ids(settings: list[Dict[str, Any]], picked_ids: Iterable[str]) -> list[Dict[str, Any]]:
    picked = {str(x) for x in picked_ids}
    return [s for s in settings if str(s.get("setting_id")) in picked]


def _print_stage_summary(stage_label: str, summaries: list[Dict[str, Any]], top_ids: list[str]) -> None:
    print(f"[phase8][{stage_label}] summary_rows={len(summaries)} selected={len(top_ids)}")
    preview = sorted(
        summaries,
        key=lambda x: (
            int(x.get("n_scored", 0)),
            float(x.get("valid_mean", -1e9) if x.get("valid_mean") is not None else -1e9),
            float(x.get("test_mean", -1e9) if x.get("test_mean") is not None else -1e9),
        ),
        reverse=True,
    )[: min(8, len(summaries))]
    for row in preview:
        vmean = row.get("valid_mean")
        tmean = row.get("test_mean")
        print(
            "  -"
            f" {row.get('setting_id')}"
            f" n={row.get('n_scored')}/{row.get('n_expected')}"
            f" valid_mean={vmean if vmean is not None else 'NA'}"
            f" test_mean={tmean if tmean is not None else 'NA'}"
        )
    if top_ids:
        print(f"[phase8][{stage_label}] picked: {', '.join(top_ids)}")


def _launch_rows(
    *,
    rows: list[Dict[str, Any]],
    gpus: list[str],
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    if not rows:
        return []

    summary_path = _phase8_summary_csv_path(args.dataset)
    _ensure_summary_csv(summary_path)
    summary_state = _load_summary_state(summary_path)

    if args.resume_from_logs:
        done_logs = _scan_completed_run_phases(args.dataset)
        if done_logs:
            before = len(rows)
            rows = [r for r in rows if str(r.get("run_phase")) not in done_logs]
            skipped = before - len(rows)
            if skipped > 0:
                print(
                    f"[phase8] resume_from_logs=on: skipped {skipped} completed runs "
                    f"(remaining {len(rows)}/{before})"
                )

    if not rows:
        return []

    for idx, row in enumerate(rows):
        row["assigned_order"] = idx + 1
        row["assigned_gpu"] = gpus[idx % len(gpus)]

    next_log_index = _next_log_index_by_phase(args.dataset)
    phase_counters: Dict[str, int] = {}
    for row in rows:
        phase_prefix = _phase_log_prefix(str(row.get("stage", "X")))
        current = int(phase_counters.get(phase_prefix, next_log_index.get(phase_prefix, 1)))
        row["run_index"] = current
        phase_counters[phase_prefix] = current + 1

    if args.dry_run:
        for row in rows:
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, row["assigned_gpu"], args)
            _write_log_preamble(lp, row, row["assigned_gpu"], args, cmd)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} {row['run_phase']} "
                f"(setting={row['setting_id']}, seed={row['seed_id']}) -> {lp}"
            )
        return rows

    gpu_bins: Dict[str, deque[Dict[str, Any]]] = {gpu: deque() for gpu in gpus}
    for row in rows:
        gpu_bins[str(row["assigned_gpu"])].append(row)

    active: Dict[str, Dict[str, Any]] = {}
    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not gpu_bins[gpu_id]:
                continue
            row = gpu_bins[gpu_id].popleft()
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, gpu_id, args)
            _write_log_preamble(lp, row, gpu_id, args, cmd)
            env = dict(os.environ)
            env["HYPEROPT_RESULTS_DIR"] = str(ARTIFACT_ROOT / "results")
            with lp.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            start_offset = 0
            try:
                start_offset = int(lp.stat().st_size)
            except Exception:
                start_offset = 0
            active[gpu_id] = {
                "proc": proc,
                "row": row,
                "log_path": lp,
                "offset": start_offset,
            }
            print(f"[launch] gpu={gpu_id} {row['run_phase']}")

        done_gpu = []
        for gpu_id, slot in active.items():
            proc = slot["proc"]
            row = slot["row"]
            lp = slot["log_path"]
            prev_offset = int(slot.get("offset", 0))
            new_offset, updates = _scan_trial_metric_updates(lp, prev_offset)
            slot["offset"] = int(new_offset)
            for trial_metrics in updates:
                _record_trial_new_best_if_any(
                    summary_path=summary_path,
                    summary_state=summary_state,
                    row=row,
                    trial_metrics=trial_metrics,
                )
            rc = proc.poll()
            if rc is None:
                continue
            done_gpu.append(gpu_id)
            print(f"[done] gpu={gpu_id} {row['run_phase']} rc={rc} log={lp}")
            if int(rc) == 0:
                _record_run_complete_summary(
                    dataset=args.dataset,
                    summary_path=summary_path,
                    summary_state=summary_state,
                    row=row,
                )

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    return rows


def _run_stage(
    *,
    stage_label: str,
    pass_tag: str,
    settings: list[Dict[str, Any]],
    seed_ids: list[int],
    args: argparse.Namespace,
    gpus: list[str],
    seed_cursor: int,
    execute: bool,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], int]:
    rows, next_cursor = _expand_rows(
        dataset=args.dataset,
        settings=settings,
        seed_ids=seed_ids,
        pass_tag=pass_tag,
        seed_cursor=seed_cursor,
    )

    if execute:
        _launch_rows(rows=rows, gpus=gpus, args=args)

    result_index = _load_result_index(args.dataset, AXIS)
    summaries = _summarize_stage(settings=settings, rows=rows, result_index=result_index)
    return rows, summaries, next_cursor


def _selection_out_dir(dataset: str) -> Path:
    root = LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_selection_manifest(payload: Dict[str, Any], out_path: Optional[str], dataset: str) -> None:
    target = Path(out_path) if out_path else (_selection_out_dir(dataset) / "selection_manifest_latest.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase8 router/wrapper + diag launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")

    parser.add_argument("--screening-seeds", default="1", help="Seed IDs for Stage A~D screening")
    parser.add_argument("--confirm-seeds", default="1,2,3", help="Seed IDs for confirm rerun")
    parser.add_argument("--seed-base", type=int, default=28000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--top-a", type=int, default=4, help="How many Stage-A winners advance to Stage-B")
    parser.add_argument("--top-b", type=int, default=4, help="How many Stage-B winners advance to Stage-C")
    parser.add_argument("--top-c", type=int, default=3, help="How many Stage-C winners advance to Stage-D")
    parser.add_argument("--top-d", type=int, default=3, help="How many Stage-D winners advance to confirm")

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)

    parser.add_argument("--only-a-candidates", default="", help="Comma-separated subset from all_w1..all_w6,mixed_1..mixed_3")

    parser.add_argument("--run-screening", dest="run_screening", action="store_true")
    parser.add_argument("--no-run-screening", dest="run_screening", action="store_false")
    parser.set_defaults(run_screening=True)

    parser.add_argument("--run-confirm", dest="run_confirm", action="store_true")
    parser.add_argument("--no-run-confirm", dest="run_confirm", action="store_false")
    parser.set_defaults(run_confirm=True)

    parser.add_argument(
        "--stop-after-stage",
        default="none",
        choices=["none", "A", "B", "C", "D", "confirm"],
        help="Stop pipeline after this stage",
    )

    parser.add_argument("--resume-from-logs", dest="resume_from_logs", action="store_true")
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.set_defaults(resume_from_logs=True)

    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")

    # fixed architecture defaults (same spirit as phase7)
    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--d-expert-hidden", type=int, default=128)
    parser.add_argument("--d-router-hidden", type=int, default=64)
    parser.add_argument("--expert-scale", type=int, default=3)
    parser.add_argument("--fixed-weight-decay", type=float, default=1e-6)
    parser.add_argument("--fixed-hidden-dropout-prob", type=float, default=0.15)

    # Search only LR. Keep this range fixed by default.
    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.screening_seeds = "1"
    args.confirm_seeds = ""
    args.top_a = 1
    args.top_b = 1
    args.top_c = 1
    args.top_d = 1
    if not args.only_a_candidates:
        args.only_a_candidates = "all_w1"
    args.run_confirm = False


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided")

    screening_seeds = _parse_csv_ints(args.screening_seeds)
    if not screening_seeds:
        raise SystemExit("No screening seeds provided")

    confirm_seeds = _parse_csv_ints(args.confirm_seeds)

    if not args.run_screening and not args.run_confirm:
        raise SystemExit("Nothing to run: both screening and confirm are disabled")

    manifest: Dict[str, Any] = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "args": {
            "gpus": gpus,
            "screening_seeds": screening_seeds,
            "confirm_seeds": confirm_seeds,
            "max_evals": args.max_evals,
            "tune_epochs": args.tune_epochs,
            "tune_patience": args.tune_patience,
            "top_a": args.top_a,
            "top_b": args.top_b,
            "top_c": args.top_c,
            "top_d": args.top_d,
            "fixed_weight_decay": args.fixed_weight_decay,
            "fixed_hidden_dropout_prob": args.fixed_hidden_dropout_prob,
            "search_lr_min": args.search_lr_min,
            "search_lr_max": args.search_lr_max,
            "run_screening": args.run_screening,
            "run_confirm": args.run_confirm,
            "resume_from_logs": args.resume_from_logs,
            "dry_run": args.dry_run,
            "smoke_test": args.smoke_test,
        },
        "stages": {},
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    allow_fallback = bool(args.dry_run)
    seed_cursor = 0

    stage_a = _stage_a_settings(args)
    if not stage_a:
        raise SystemExit("Stage-A setting list is empty. Check --only-a-candidates values.")

    # Stage A
    rows_a, summary_a, seed_cursor = _run_stage(
        stage_label="A",
        pass_tag="SCR",
        settings=stage_a,
        seed_ids=screening_seeds,
        args=args,
        gpus=gpus,
        seed_cursor=seed_cursor,
        execute=args.run_screening,
    )
    top_a_ids = _select_top_setting_ids(
        settings=stage_a,
        summaries=summary_a,
        top_n=args.top_a,
        allow_fallback=allow_fallback,
    )
    _print_stage_summary("A", summary_a, top_a_ids)
    manifest["stages"]["A"] = {
        "selected": top_a_ids,
        "summary": summary_a,
        "n_rows": len(rows_a),
    }
    selected_a = _pick_settings_by_ids(stage_a, top_a_ids)
    if not selected_a:
        raise SystemExit("Stage A produced no selectable winners.")

    if args.stop_after_stage == "A":
        _write_selection_manifest(manifest, args.manifest_out, args.dataset)
        return 0

    # Stage B
    stage_b = _stage_b_settings(args, selected_a)
    rows_b, summary_b, seed_cursor = _run_stage(
        stage_label="B",
        pass_tag="SCR",
        settings=stage_b,
        seed_ids=screening_seeds,
        args=args,
        gpus=gpus,
        seed_cursor=seed_cursor,
        execute=args.run_screening,
    )
    top_b_ids = _select_top_setting_ids(
        settings=stage_b,
        summaries=summary_b,
        top_n=args.top_b,
        allow_fallback=allow_fallback,
    )
    _print_stage_summary("B", summary_b, top_b_ids)
    manifest["stages"]["B"] = {
        "selected": top_b_ids,
        "summary": summary_b,
        "n_rows": len(rows_b),
    }
    selected_b = _pick_settings_by_ids(stage_b, top_b_ids)
    if not selected_b:
        raise SystemExit("Stage B produced no selectable winners.")

    if args.stop_after_stage == "B":
        _write_selection_manifest(manifest, args.manifest_out, args.dataset)
        return 0

    # Stage C
    stage_c = _stage_c_settings(args, selected_b)
    rows_c, summary_c, seed_cursor = _run_stage(
        stage_label="C",
        pass_tag="SCR",
        settings=stage_c,
        seed_ids=screening_seeds,
        args=args,
        gpus=gpus,
        seed_cursor=seed_cursor,
        execute=args.run_screening,
    )
    top_c_ids = _select_top_setting_ids(
        settings=stage_c,
        summaries=summary_c,
        top_n=args.top_c,
        allow_fallback=allow_fallback,
    )
    _print_stage_summary("C", summary_c, top_c_ids)
    manifest["stages"]["C"] = {
        "selected": top_c_ids,
        "summary": summary_c,
        "n_rows": len(rows_c),
    }
    selected_c = _pick_settings_by_ids(stage_c, top_c_ids)
    if not selected_c:
        raise SystemExit("Stage C produced no selectable winners.")

    if args.stop_after_stage == "C":
        _write_selection_manifest(manifest, args.manifest_out, args.dataset)
        return 0

    # Stage D
    stage_d = _stage_d_settings(args, selected_c)
    rows_d, summary_d, seed_cursor = _run_stage(
        stage_label="D",
        pass_tag="SCR",
        settings=stage_d,
        seed_ids=screening_seeds,
        args=args,
        gpus=gpus,
        seed_cursor=seed_cursor,
        execute=args.run_screening,
    )
    top_d_ids = _select_top_setting_ids(
        settings=stage_d,
        summaries=summary_d,
        top_n=args.top_d,
        allow_fallback=allow_fallback,
    )
    _print_stage_summary("D", summary_d, top_d_ids)
    manifest["stages"]["D"] = {
        "selected": top_d_ids,
        "summary": summary_d,
        "n_rows": len(rows_d),
    }
    selected_d = _pick_settings_by_ids(stage_d, top_d_ids)
    if not selected_d:
        raise SystemExit("Stage D produced no selectable winners.")

    if args.stop_after_stage == "D":
        _write_selection_manifest(manifest, args.manifest_out, args.dataset)
        return 0

    # Confirm finalists
    if args.run_confirm and confirm_seeds:
        rows_confirm, summary_confirm, seed_cursor = _run_stage(
            stage_label="CONFIRM",
            pass_tag="CFM",
            settings=selected_d,
            seed_ids=confirm_seeds,
            args=args,
            gpus=gpus,
            seed_cursor=seed_cursor,
            execute=True,
        )
        top_confirm_ids = _select_top_setting_ids(
            settings=selected_d,
            summaries=summary_confirm,
            top_n=max(1, min(len(selected_d), args.top_d)),
            allow_fallback=allow_fallback,
        )
        _print_stage_summary("CONFIRM", summary_confirm, top_confirm_ids)
        manifest["stages"]["confirm"] = {
            "selected": top_confirm_ids,
            "summary": summary_confirm,
            "n_rows": len(rows_confirm),
        }

    _write_selection_manifest(manifest, args.manifest_out, args.dataset)
    print("[done] phase8 router/wrapper+diag pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
