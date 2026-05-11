#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase9 aux-loss screening runs (4x4x4 x seed)."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

THIS_FILE = Path(__file__).resolve()
EXP_DIR = THIS_FILE.parents[2]
REPO_ROOT = EXP_DIR.parent
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
RESULT_ROOT = ARTIFACT_ROOT / "results" / "fmoe_n3"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase9_auxloss_v1"
PHASE = "P9"
MODEL_TAG = "FMoEN3"

STAGES = ("macro", "mid", "micro")
PRIMITIVES = ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar")


def hydra_literal(value: Any) -> str:
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


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _parse_csv_ints(text: str) -> list[int]:
    return [int(tok.strip()) for tok in str(text or "").split(",") if tok.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [tok.strip() for tok in str(text or "").split(",") if tok.strip()]


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _sanitize_token(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum():
            out.append(ch.upper())
        else:
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "X"


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

    wrapper_cfg = {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0}
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
    # All aux defaults off; profile layer enables main/support losses only.
    return {
        "layer_layout": ["macro", "mid", "micro"],
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": _all_stage_map("both"),
        "stage_feature_injection": _all_stage_map("none"),
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "balance_loss_lambda": 0.0,
        "z_loss_lambda": 0.0,
        "gate_entropy_lambda": 0.0,
        "route_smoothness_lambda": 0.0,
        "route_consistency_lambda": 0.0,
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
        # Planned-but-not-yet-consumed keys; kept for forward compatibility.
        "primitive_balance_lambda": 0.0,
        "wrapper_group_feature_align_lambda": 0.0,
        "stage_router_primitives": _build_stage_router_primitives(),
    }


def _source_profiles() -> Dict[str, Dict[str, str]]:
    return {
        "src_base": {
            "a_joint": "both",
            "b_group": "both",
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


def _base_definitions() -> Dict[str, Dict[str, Any]]:
    return {
        "B1": {
            "run_phase_ref": "P8_SCR_B_ALL_W2_BIAS_RULE_S1",
            "desc": "all_w2 + bias_rule + src_base + dense",
            "wrapper_map": {"macro": "w2_a_plus_d", "mid": "w2_a_plus_d", "micro": "w2_a_plus_d"},
            "bias_mode": "bias_rule",
            "source_profile": "src_base",
        },
        "B2": {
            "run_phase_ref": "P8_SCR_B_MIXED_2_BIAS_GROUP_FEAT_S1",
            "desc": "mixed_2 + bias_group_feat + src_base + dense",
            "wrapper_map": {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
            "bias_mode": "bias_group_feat",
            "source_profile": "src_base",
        },
        "B3": {
            "run_phase_ref": "P8_SCR_B_MIXED_2_BIAS_BOTH_S1",
            "desc": "mixed_2 + bias_both + src_base + dense",
            "wrapper_map": {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
            "bias_mode": "bias_both",
            "source_profile": "src_base",
        },
        "B4": {
            "run_phase_ref": "P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_S1",
            "desc": "mixed_2 + bias_both + src_abc_feature + dense",
            "wrapper_map": {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
            "bias_mode": "bias_both",
            "source_profile": "src_abc_feature",
        },
    }


def _aux_profiles() -> list[Dict[str, Any]]:
    return [
        {
            "concept_id": "C0",
            "concept_name": "Natural",
            "combo_id": "N1",
            "main_aux": "none",
            "support_aux": "none",
            "scenario": "완전 baseline (aux off).",
            "overrides": {},
        },
        {
            "concept_id": "C0",
            "concept_name": "Natural",
            "combo_id": "N2",
            "main_aux": "route_smoothness",
            "support_aux": "none",
            "scenario": "route jitter 완화 자연형.",
            "overrides": {"route_smoothness_lambda": 0.01},
        },
        {
            "concept_id": "C0",
            "concept_name": "Natural",
            "combo_id": "N3",
            "main_aux": "balance",
            "support_aux": "z",
            "scenario": "약한 canonical 안정화.",
            "overrides": {"balance_loss_lambda": 0.001, "z_loss_lambda": 5e-5},
        },
        {
            "concept_id": "C0",
            "concept_name": "Natural",
            "combo_id": "N4",
            "main_aux": "z",
            "support_aux": "none",
            "scenario": "logit 안정화 단독.",
            "overrides": {"z_loss_lambda": 1e-4},
        },
        {
            "concept_id": "C1",
            "concept_name": "CanonicalBalance",
            "combo_id": "B1",
            "main_aux": "balance",
            "support_aux": "z",
            "scenario": "표준 균형 유도.",
            "overrides": {"balance_loss_lambda": 0.002, "z_loss_lambda": 1e-4},
        },
        {
            "concept_id": "C1",
            "concept_name": "CanonicalBalance",
            "combo_id": "B2",
            "main_aux": "balance_strong",
            "support_aux": "z",
            "scenario": "강한 균형 유도.",
            "overrides": {"balance_loss_lambda": 0.006, "z_loss_lambda": 3e-4},
        },
        {
            "concept_id": "C1",
            "concept_name": "CanonicalBalance",
            "combo_id": "B3",
            "main_aux": "factored_group_balance",
            "support_aux": "z",
            "scenario": "group-level collapse 완화(조건부).",
            "overrides": {"factored_group_balance_lambda": 1e-3, "z_loss_lambda": 1e-4},
        },
        {
            "concept_id": "C1",
            "concept_name": "CanonicalBalance",
            "combo_id": "B4",
            "main_aux": "primitive_balance",
            "support_aux": "z",
            "scenario": "primitive 분포 균형(신규 key).",
            "overrides": {"primitive_balance_lambda": 8e-4, "z_loss_lambda": 1e-4},
        },
        {
            "concept_id": "C2",
            "concept_name": "Specialization",
            "combo_id": "S1",
            "main_aux": "route_sharpness",
            "support_aux": "none",
            "scenario": "약한 특화.",
            "overrides": {"route_sharpness_lambda": 0.004},
        },
        {
            "concept_id": "C2",
            "concept_name": "Specialization",
            "combo_id": "S2",
            "main_aux": "route_sharpness_strong",
            "support_aux": "none",
            "scenario": "강한 특화.",
            "overrides": {"route_sharpness_lambda": 0.008},
        },
        {
            "concept_id": "C2",
            "concept_name": "Specialization",
            "combo_id": "S3",
            "main_aux": "route_sharpness",
            "support_aux": "route_monopoly",
            "scenario": "집중+과점 제어 동시.",
            "overrides": {"route_sharpness_lambda": 0.008, "route_monopoly_lambda": 0.02, "route_monopoly_tau": 0.25},
        },
        {
            "concept_id": "C2",
            "concept_name": "Specialization",
            "combo_id": "S4",
            "main_aux": "route_smoothness",
            "support_aux": "route_sharpness",
            "scenario": "안정성 안에서 특화.",
            "overrides": {"route_smoothness_lambda": 0.03, "route_sharpness_lambda": 0.004},
        },
        {
            "concept_id": "C3",
            "concept_name": "FeatureAlignment",
            "combo_id": "F1",
            "main_aux": "route_prior",
            "support_aux": "none",
            "scenario": "session-level prior 정렬.",
            "overrides": {"route_prior_lambda": 5e-4},
        },
        {
            "concept_id": "C3",
            "concept_name": "FeatureAlignment",
            "combo_id": "F2",
            "main_aux": "route_prior_strong",
            "support_aux": "z",
            "scenario": "prior 정렬 강도 증가 + 안정화.",
            "overrides": {"route_prior_lambda": 1e-3, "z_loss_lambda": 1e-4},
        },
        {
            "concept_id": "C3",
            "concept_name": "FeatureAlignment",
            "combo_id": "F3",
            "main_aux": "group_prior_align",
            "support_aux": "none",
            "scenario": "group prior 정렬.",
            "overrides": {"group_prior_align_lambda": 5e-4},
        },
        {
            "concept_id": "C3",
            "concept_name": "FeatureAlignment",
            "combo_id": "F4",
            "main_aux": "wrapper_group_feature_align",
            "support_aux": "group_prior_align",
            "scenario": "wrapper-group 특화 정렬(신규 key).",
            "overrides": {"wrapper_group_feature_align_lambda": 1e-3, "group_prior_align_lambda": 2e-4},
        },
    ]


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
        n_completed = int(payload.get("n_completed", 0) or 0)
        interrupted = bool(payload.get("interrupted", False))
        mtime = float(path.stat().st_mtime)
        rec = {
            "run_phase": run_phase,
            "best_mrr": best_mrr,
            "test_mrr": test_mrr,
            "n_completed": n_completed,
            "interrupted": interrupted,
            "path": str(path),
            "mtime": mtime,
        }
        prev = out.get(run_phase)
        if prev is None:
            out[run_phase] = rec
            continue
        prev_best = _metric_to_float(prev.get("best_mrr"))
        cur_best = _metric_to_float(rec.get("best_mrr"))
        if cur_best is not None and (prev_best is None or cur_best > prev_best):
            out[run_phase] = rec
            continue
        if cur_best == prev_best and mtime >= float(prev.get("mtime", 0.0)):
            out[run_phase] = rec
            continue
        if prev_best is None and cur_best is None and mtime >= float(prev.get("mtime", 0.0)):
            out[run_phase] = rec
    return out


def _completed_by_result(result_index: Dict[str, Dict[str, Any]]) -> set[str]:
    done = set()
    for run_phase, rec in result_index.items():
        if _metric_to_float(rec.get("best_mrr")) is not None and int(rec.get("n_completed", 0)) > 0:
            done.add(run_phase)
    return done


def _phase9_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset) / MODEL_TAG


def _phase9_axis_dataset_dir(dataset: str) -> Path:
    root = LOG_ROOT / AXIS / PHASE / _dataset_tag(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _phase9_summary_csv_path(dataset: str) -> Path:
    return _phase9_axis_dataset_dir(dataset) / "summary.csv"


def _summary_fieldnames() -> list[str]:
    return [
        "run_phase",
        "run_id",
        "dataset",
        "base_id",
        "base_ref",
        "concept_id",
        "concept_name",
        "combo_id",
        "main_aux",
        "support_aux",
        "n_active_aux",
        "wrapper_combo",
        "bias_mode",
        "source_profile",
        "seed_id",
        "gpu_id",
        "status",
        "run_best_valid_mrr20",
        "test_mrr20",
        "n_completed",
        "interrupted",
        "result_path",
        "timestamp_utc",
    ]


def _write_summary_csv(path: Path, rows: list[Dict[str, Any]], result_index: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writeheader()
        for row in rows:
            rec = result_index.get(str(row["run_phase"]))
            status = "pending"
            best = ""
            test = ""
            n_completed = ""
            interrupted = ""
            result_path = ""
            if isinstance(rec, dict):
                best_v = _metric_to_float(rec.get("best_mrr"))
                test_v = _metric_to_float(rec.get("test_mrr"))
                best = "" if best_v is None else f"{best_v:.6f}"
                test = "" if test_v is None else f"{test_v:.6f}"
                n_completed = int(rec.get("n_completed", 0) or 0)
                interrupted = bool(rec.get("interrupted", False))
                result_path = str(rec.get("path", "") or "")
                if best_v is not None and n_completed > 0:
                    status = "completed"
                elif result_path:
                    status = "result_found"
            writer.writerow(
                {
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "base_id": row["base_id"],
                    "base_ref": row["base_ref"],
                    "concept_id": row["concept_id"],
                    "concept_name": row["concept_name"],
                    "combo_id": row["combo_id"],
                    "main_aux": row["main_aux"],
                    "support_aux": row["support_aux"],
                    "n_active_aux": row["n_active_aux"],
                    "wrapper_combo": row["wrapper_combo"],
                    "bias_mode": row["bias_mode"],
                    "source_profile": row["source_profile"],
                    "seed_id": row["seed_id"],
                    "gpu_id": row.get("assigned_gpu", ""),
                    "status": status,
                    "run_best_valid_mrr20": best,
                    "test_mrr20": test,
                    "n_completed": n_completed,
                    "interrupted": interrupted,
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
            )


def _run_phase_name(base_id: str, concept_id: str, combo_id: str, seed_id: int) -> str:
    return f"P9_{base_id}_{concept_id}_{combo_id}_S{int(seed_id)}"


def _run_id(base_id: str, concept_id: str, combo_id: str, seed_id: int) -> str:
    return f"{base_id}_{concept_id}_{combo_id}_S{int(seed_id)}"


def _extract_run_phase_from_log(log_path: Path) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    for line in text.splitlines():
        parts = line.strip().split()
        for part in parts:
            if part.startswith("run_phase="):
                return part.split("=", 1)[1].strip()
    return ""


def _is_completed_log(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return ("[RUN_STATUS] END status=normal" in text) or ("\n  DONE  |  FeaturedMoE_N3 x " in text)


def _scan_completed_run_phases(dataset: str) -> set[str]:
    done = set()
    root = _phase9_log_dir(dataset)
    if not root.exists():
        return done
    for log_path in sorted(root.glob("*.log")):
        run_phase = _extract_run_phase_from_log(log_path)
        if not run_phase:
            continue
        if _is_completed_log(log_path):
            done.add(run_phase)
    return done


def _apply_base_overrides(
    *,
    overrides: Dict[str, Any],
    base_cfg: Dict[str, Any],
    feature_group_bias_lambda: float,
    rule_bias_scale: float,
) -> None:
    _set_stage_specific_wrapper(overrides, dict(base_cfg["wrapper_map"]))
    bias_mode = str(base_cfg["bias_mode"])
    if bias_mode == "bias_rule":
        overrides["feature_group_bias_lambda"] = 0.0
        overrides["rule_bias_scale"] = float(rule_bias_scale)
    elif bias_mode == "bias_group_feat":
        overrides["feature_group_bias_lambda"] = float(feature_group_bias_lambda)
        overrides["rule_bias_scale"] = 0.0
    elif bias_mode == "bias_both":
        overrides["feature_group_bias_lambda"] = float(feature_group_bias_lambda)
        overrides["rule_bias_scale"] = float(rule_bias_scale)
    else:
        overrides["feature_group_bias_lambda"] = 0.0
        overrides["rule_bias_scale"] = 0.0

    source_map = _source_profiles()[str(base_cfg["source_profile"])]
    primitives = _update_primitive_sources(overrides.get("stage_router_primitives", {}), source_map)
    primitives = _update_primitive_top_k(primitives, {p: 0 for p in PRIMITIVES})
    overrides["stage_router_primitives"] = primitives
    overrides["moe_top_k"] = 0


def _build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    base_allow = {tok.upper() for tok in _parse_csv_strings(args.only_base)}
    concept_allow = {tok.upper() for tok in _parse_csv_strings(args.only_concept)}
    combo_allow = {tok.upper() for tok in _parse_csv_strings(args.only_combo)}
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("No seeds provided")

    profiles = _aux_profiles()
    bases = _base_definitions()
    rows: list[Dict[str, Any]] = []
    seed_cursor = 0
    for base_id in ("B1", "B2", "B3", "B4"):
        if base_allow and base_id not in base_allow:
            continue
        base_cfg = bases[base_id]
        for profile in profiles:
            concept_id = str(profile["concept_id"]).upper()
            combo_id = str(profile["combo_id"]).upper()
            if concept_allow and concept_id not in concept_allow:
                continue
            if combo_allow and combo_id not in combo_allow:
                continue
            for seed_id in seeds:
                run_phase = _run_phase_name(base_id=base_id, concept_id=concept_id, combo_id=combo_id, seed_id=seed_id)
                run_id = _run_id(base_id=base_id, concept_id=concept_id, combo_id=combo_id, seed_id=seed_id)
                overrides = _base_fixed_overrides()
                _apply_base_overrides(
                    overrides=overrides,
                    base_cfg=base_cfg,
                    feature_group_bias_lambda=float(args.feature_group_bias_lambda),
                    rule_bias_scale=float(args.rule_bias_scale),
                )
                for key, value in dict(profile.get("overrides", {}) or {}).items():
                    overrides[str(key)] = value
                n_active = 0
                if str(profile.get("main_aux", "none")).lower() != "none":
                    n_active += 1
                if str(profile.get("support_aux", "none")).lower() != "none":
                    n_active += 1
                rows.append(
                    {
                        "dataset": args.dataset,
                        "base_id": base_id,
                        "base_ref": base_cfg["run_phase_ref"],
                        "base_desc": base_cfg["desc"],
                        "wrapper_combo": "all_w2" if base_id == "B1" else "mixed_2",
                        "bias_mode": base_cfg["bias_mode"],
                        "source_profile": base_cfg["source_profile"],
                        "concept_id": concept_id,
                        "concept_name": profile["concept_name"],
                        "combo_id": combo_id,
                        "main_aux": profile["main_aux"],
                        "support_aux": profile["support_aux"],
                        "scenario": profile["scenario"],
                        "n_active_aux": int(n_active),
                        "seed_id": int(seed_id),
                        "seed_offset": int(seed_cursor),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "overrides": overrides,
                    }
                )
                seed_cursor += 1
    return rows


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
        f"++p9_base_id={hydra_literal(row['base_id'])}",
        f"++p9_concept_id={hydra_literal(row['concept_id'])}",
        f"++p9_combo_id={hydra_literal(row['combo_id'])}",
        f"++p9_main_aux={hydra_literal(row['main_aux'])}",
        f"++p9_support_aux={hydra_literal(row['support_aux'])}",
        f"++p9_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _log_path(row: Dict[str, Any], dataset: str) -> Path:
    root = _phase9_log_dir(dataset)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{row['run_id']}.log"


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE9_SETTING_HEADER]",
        (
            f"run_id={row['run_id']} run_phase={row['run_phase']} "
            f"base={row['base_id']} concept={row['concept_id']} combo={row['combo_id']} seed={row['seed_id']}"
        ),
        (
            f"main_aux={row['main_aux']} support_aux={row['support_aux']} "
            f"n_active_aux={row['n_active_aux']}"
        ),
        f"base_ref={row['base_ref']}",
        f"wrapper={row['wrapper_combo']} bias={row['bias_mode']} source={row['source_profile']}",
        f"scenario={row['scenario']}",
        f"dataset={row['dataset']} gpu={gpu_id}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    # Overwrite stale/incomplete log by design.
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _write_matrix_manifest(rows: list[Dict[str, Any]], args: argparse.Namespace) -> Path:
    if args.manifest_out:
        out_path = Path(args.manifest_out)
    else:
        out_path = _phase9_axis_dataset_dir(args.dataset) / "auxloss_matrix.json"
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_rows": len(rows),
        "rows": [
            {
                "run_id": r["run_id"],
                "run_phase": r["run_phase"],
                "base_id": r["base_id"],
                "concept_id": r["concept_id"],
                "combo_id": r["combo_id"],
                "main_aux": r["main_aux"],
                "support_aux": r["support_aux"],
                "n_active_aux": r["n_active_aux"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def _launch_rows(rows: list[Dict[str, Any]], gpus: list[str], args: argparse.Namespace) -> int:
    if not rows:
        print("[phase9] no rows to run.")
        return 0

    for idx, row in enumerate(rows):
        row["assigned_gpu"] = gpus[idx % len(gpus)]
        row["assigned_order"] = idx + 1

    result_index = _load_result_index(args.dataset, AXIS)
    done_results = _completed_by_result(result_index)
    done_logs: set[str] = set()
    if args.resume_from_logs:
        done_logs = _scan_completed_run_phases(args.dataset)

    runnable: list[Dict[str, Any]] = []
    for row in rows:
        run_phase = str(row["run_phase"])
        if run_phase in done_results:
            continue
        lp = _log_path(row, args.dataset)
        if args.resume_from_logs and lp.exists() and _is_completed_log(lp):
            continue
        if args.resume_from_logs and run_phase in done_logs:
            continue
        runnable.append(row)

    summary_path = _phase9_summary_csv_path(args.dataset)
    _write_summary_csv(summary_path, rows, result_index)

    if not runnable:
        print("[phase9] all runs are already completed by result/log markers.")
        return 0

    if args.dry_run:
        for row in runnable:
            lp = _log_path(row, args.dataset)
            cmd = _build_command(row, row["assigned_gpu"], args)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} run={row['run_id']} "
                f"base={row['base_id']} concept={row['concept_id']} combo={row['combo_id']} -> {lp}"
            )
            print("          " + " ".join(cmd))
        return 0

    gpu_bins: Dict[str, deque[Dict[str, Any]]] = {gpu: deque() for gpu in gpus}
    for row in runnable:
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
            active[gpu_id] = {"proc": proc, "row": row, "log_path": lp}
            print(
                f"[launch] gpu={gpu_id} run={row['run_id']} "
                f"({row['base_id']}/{row['concept_id']}/{row['combo_id']})"
            )

        done_gpu = []
        for gpu_id, slot in active.items():
            proc = slot["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            row = slot["row"]
            lp = slot["log_path"]
            done_gpu.append(gpu_id)
            print(f"[done] gpu={gpu_id} run={row['run_id']} rc={rc} log={lp}")

            result_index = _load_result_index(args.dataset, AXIS)
            _write_summary_csv(summary_path, rows, result_index)

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    result_index = _load_result_index(args.dataset, AXIS)
    _write_summary_csv(summary_path, rows, result_index)
    print(f"[phase9] summary updated: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase9 aux-loss screening launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=48000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--d-expert-hidden", type=int, default=128)
    parser.add_argument("--d-router-hidden", type=int, default=64)
    parser.add_argument("--fixed-weight-decay", type=float, default=1e-6)
    parser.add_argument("--fixed-hidden-dropout-prob", type=float, default=0.15)

    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--only-base", default="", help="Comma-separated subset of {B1,B2,B3,B4}")
    parser.add_argument("--only-concept", default="", help="Comma-separated subset of {C0,C1,C2,C3}")
    parser.add_argument("--only-combo", default="", help="Comma-separated subset of combo ids (e.g. N1,B2,S3,F4)")

    parser.add_argument("--manifest-out", default="", help="Optional matrix JSON output path")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=4)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"
    args.gpus = _parse_csv_strings(args.gpus)[0] if _parse_csv_strings(args.gpus) else "0"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs provided")

    rows = _build_rows(args)
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise SystemExit("No rows matched filters")

    manifest_path = _write_matrix_manifest(rows, args)
    print(f"[phase9] dataset={args.dataset} total_rows={len(rows)} manifest={manifest_path}")
    return _launch_rows(rows=rows, gpus=gpus, args=args)


if __name__ == "__main__":
    raise SystemExit(main())
