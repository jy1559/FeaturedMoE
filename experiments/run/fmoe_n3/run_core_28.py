#!/usr/bin/env python3
"""Launch the FMoE_N3 core_ablation_v2 28-combo track."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"
INVENTORY_ROOT = ARTIFACT_ROOT / "inventory" / "fmoe_n3"

if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from common.phase_summary_csv import build_fmoe_n3_axis_summary, build_fmoe_n3_summaries
from fmoe_n3.update_artifact_views import build_artifact_views

TRACK = "fmoe_n3"
AXIS = "core_ablation_v2"
PHASE = "CORE28"
SUMMARY_REFRESH_SEC = 300.0

WD_CHOICES = [0.0, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4]
DROPOUT_CHOICES = [0.10, 0.15, 0.20, 0.25]

SMOKE_TOTAL_MIN_HINT = {
    "P00": 0.6,
    "P01": 0.6,
    "D10": 0.5,
    "D11": 0.5,
    "D12": 0.5,
    "D13": 0.6,
    "D14": 0.5,
    "D15": 0.6,
    "M20": 0.9,
    "M21": 1.2,
    "M22": 1.6,
    "R30": 1.6,
    "R31": 1.6,
    "R32": 1.7,
    "R33": 1.6,
    "R34": 1.6,
    "E40": 1.7,
    "E41": 1.6,
    "E42": 1.6,
    "T50": 1.7,
    "T51": 1.7,
    "X60": 1.6,
    "X61": 1.6,
    "X62": 1.8,
    "X63": 1.6,
    "C70": 1.6,
    "C71": 0.6,
    "C72": 0.6,
}

TRIAL_MIN_HINT = {
    "P00": 0.15,
    "P01": 1.0 / 6.0,
    "D10": 1.0 / 6.0,
    "D11": 1.0 / 6.0,
    "D12": 1.0 / 6.0,
    "D13": 11.0 / 60.0,
    "D14": 1.0 / 6.0,
    "D15": 11.0 / 60.0,
    "M20": 0.50,
    "M21": 0.85,
    "M22": 1.20,
    "R30": 1.20,
    "R31": 1.20,
    "R32": 1.30,
    "R33": 1.20,
    "R34": 1.20,
    "E40": 1.30,
    "E41": 1.20,
    "E42": 1.20,
    "T50": 1.30,
    "T51": 1.30,
    "X60": 1.20,
    "X61": 1.20,
    "X62": 1.40,
    "X63": 1.20,
    "C70": 1.20,
    "C71": 0.20,
    "C72": 13.0 / 60.0,
}


def hydra_literal(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
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
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "run"


def runtime_bucket(combo_id: str) -> str:
    if combo_id in {"P00", "P01", "D10", "D11", "D12", "D13", "D14", "D15", "C71", "C72"}:
        return "fast"
    if combo_id in {"M20", "M21"}:
        return "medium"
    return "heavy"


def recommended_max_evals(combo_id: str) -> int:
    if combo_id in {"P00", "P01"}:
        return 15
    fast = {"D10", "D11", "D12", "D13", "D14", "D15", "C71", "C72"}
    medium = {"M20", "M21"}
    very_heavy = {"R32", "E40", "T50", "T51", "X62"}
    if combo_id in fast:
        return 10
    if combo_id in medium:
        return 4
    if combo_id in very_heavy:
        return 3
    return 4


def estimated_runtime_profile(combo_id: str) -> dict:
    smoke_total = float(SMOKE_TOTAL_MIN_HINT.get(combo_id, 1.0))
    trial_min = float(TRIAL_MIN_HINT.get(combo_id, max(smoke_total - 0.3, 0.1)))
    fixed_overhead = max(smoke_total - trial_min, 0.0)
    est_50ep_single_eval = fixed_overhead + (trial_min * 50.0)
    return {
        "runtime_bucket": runtime_bucket(combo_id),
        "smoke_total_min_hint": smoke_total,
        "trial_min_hint": trial_min,
        "fixed_overhead_min_hint": fixed_overhead,
        "estimated_50ep_single_eval_min": est_50ep_single_eval,
        "recommended_max_evals": recommended_max_evals(combo_id),
    }


def plan_gpu_bins(
    combos: list[dict],
    gpus: list[str],
    *,
    cost_key: str,
    group_size: int | None = None,
) -> dict[str, list[dict]]:
    """Plan combos with local balancing while preserving the global combo order.

    We intentionally avoid global cost-based resorting. Instead, combos are split
    into small windows (default: one window per number of GPUs), and only within
    each window do we rebalance assignments to avoid one GPU getting all heavy
    jobs. This keeps the launch order close to the authored combo order, which
    makes early SASRec/plain comparisons appear first.
    """

    bins = {gpu: [] for gpu in gpus}
    loads = {gpu: 0.0 for gpu in gpus}
    window = max(1, int(group_size or len(gpus)))
    ordered = list(combos)

    for group_idx, start in enumerate(range(0, len(ordered), window), start=1):
        chunk = ordered[start : start + window]
        available_gpus = list(gpus)
        combo_to_gpu: dict[str, str] = {}

        # Within each local chunk, place heavier jobs on currently lighter GPUs.
        ranked_chunk = sorted(
            chunk,
            key=lambda row: (-float(row.get(cost_key, 0.0) or 0.0), row["combo_id"]),
        )
        for combo in ranked_chunk:
            gpu_id = min(available_gpus, key=lambda gpu: (loads[gpu], gpu))
            combo_to_gpu[str(combo["combo_id"])] = gpu_id
            available_gpus.remove(gpu_id)

        # Append back in the original authored order so per-GPU queues still read
        # naturally as "early combos first, later combos later".
        for local_order, combo in enumerate(chunk, start=1):
            combo["launch_group"] = group_idx
            combo["launch_group_order"] = local_order
            gpu_id = combo_to_gpu[str(combo["combo_id"])]
            bins[gpu_id].append(combo)
            loads[gpu_id] += float(combo.get(cost_key, 0.0) or 0.0)

    return bins


def dataset_profile(dataset: str) -> dict:
    key = dataset.lower()
    if key == "lastfm0.03":
        return {
            "c2_train_bs": 2048,
            "c2_eval_bs": 4096,
            "c3_train_bs": 2048,
            "c3_eval_bs": 4096,
            "c4_train_bs": 1024,
            "c4_eval_bs": 2048,
            "heavy_train_bs": 1024,
            "heavy_eval_bs": 2048,
        }
    return {
        "c2_train_bs": 4096,
        "c2_eval_bs": 8192,
        "c3_train_bs": 3072,
        "c3_eval_bs": 6144,
        "c4_train_bs": 2048,
        "c4_eval_bs": 4096,
        "heavy_train_bs": 1536,
        "heavy_eval_bs": 3072,
    }


def _all_stage_map(value: str) -> dict:
    return {"macro": value, "mid": value, "micro": value}


def _default_granularity() -> dict:
    return {"macro": "session", "mid": "session", "micro": "token"}


def _default_encoder_mode() -> dict:
    return _all_stage_map("linear")


def _default_router_source() -> dict:
    return _all_stage_map("both")


def _default_feature_injection() -> dict:
    return _all_stage_map("none")


def _default_router_mode() -> dict:
    return _all_stage_map("learned")


def _default_compute_mode() -> dict:
    return _all_stage_map("moe")


def base_combo(
    combo_id: str,
    *,
    combo_family: str,
    combo_role: str,
    desc: str,
    layer_layout: list[str],
    baseline_recipe: str,
    delta_from_base: str,
    train_bs: int,
    eval_bs: int,
    lr_min: float,
    lr_max: float,
    max_len: int = 10,
    embedding_size: int = 128,
    num_heads: int = 4,
    attn_dropout_prob: float = 0.10,
    stage_feature_encoder_mode: dict | None = None,
    stage_compute_mode: dict | None = None,
    stage_router_mode: dict | None = None,
    stage_router_source: dict | None = None,
    stage_feature_injection: dict | None = None,
    stage_router_granularity: dict | None = None,
    stage_router_type: dict | None = None,
    macro_window: int = 5,
    family_mask: dict | None = None,
    moe_top_k: int = 0,
    expert_scale: int = 1,
    dense_hidden_scale: float = 1.0,
    balance_loss_lambda: float = 0.002,
    z_loss_lambda: float = 0.0,
    gate_entropy_lambda: float = 0.0,
    gate_entropy_until: float = 0.0,
    rule_agreement_lambda: float = 0.0,
    group_coverage_lambda: float = 0.0,
    group_prior_align_lambda: float = 0.0,
    feature_group_bias_lambda: float = 0.0,
    feature_group_prior_temperature: float = 1.0,
    search_hidden_dropout_prob: list[float] | None = None,
    search_weight_decay: list[float] | None = None,
):
    return {
        "combo_id": combo_id,
        "combo_family": combo_family,
        "combo_role": combo_role,
        "desc": desc,
        "layer_layout": list(layer_layout),
        "baseline_recipe": baseline_recipe,
        "delta_from_base": delta_from_base,
        "train_batch_size": train_bs,
        "eval_batch_size": eval_bs,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "MAX_ITEM_LIST_LENGTH": max_len,
        "embedding_size": embedding_size,
        "d_ff": int(2 * embedding_size),
        "num_heads": num_heads,
        "attn_dropout_prob": attn_dropout_prob,
        "stage_feature_encoder_mode": stage_feature_encoder_mode or _default_encoder_mode(),
        "stage_compute_mode": stage_compute_mode or _default_compute_mode(),
        "stage_router_mode": stage_router_mode or _default_router_mode(),
        "stage_router_source": stage_router_source or _default_router_source(),
        "stage_feature_injection": stage_feature_injection or _default_feature_injection(),
        "stage_router_granularity": stage_router_granularity or _default_granularity(),
        "stage_router_type": stage_router_type or _all_stage_map("standard"),
        "macro_history_window": macro_window,
        "stage_feature_family_mask": family_mask or {},
        "moe_top_k": moe_top_k,
        "expert_scale": expert_scale,
        "dense_hidden_scale": dense_hidden_scale,
        "balance_loss_lambda": balance_loss_lambda,
        "z_loss_lambda": z_loss_lambda,
        "gate_entropy_lambda": gate_entropy_lambda,
        "gate_entropy_until": gate_entropy_until,
        "rule_agreement_lambda": rule_agreement_lambda,
        "group_coverage_lambda": group_coverage_lambda,
        "group_prior_align_lambda": group_prior_align_lambda,
        "feature_group_bias_lambda": feature_group_bias_lambda,
        "feature_group_prior_temperature": feature_group_prior_temperature,
        "search_hidden_dropout_prob": list(search_hidden_dropout_prob or DROPOUT_CHOICES),
        "search_weight_decay": list(search_weight_decay or WD_CHOICES),
        "has_diag": False,
    }


def build_combos(dataset: str) -> list[dict]:
    prof = dataset_profile(dataset)
    # LFM: SASRec baseline best at lr≈3.86e-4; collapses observed at <1e-4.
    # Narrow the search window relative to KuaiRec to avoid wasted trials in
    # confirmed dead zones (< 8e-5) and diminishing-return high end (> 1.2e-3).
    if dataset.lower() == "lastfm0.03":
        c2_lr = (1.5e-4, 2.5e-3)
        c4_lr = (1e-4, 1.2e-3)
        conservative_c4_lr = (8e-5, 8e-4)
    else:
        c2_lr = (7e-5, 5e-3)
        c4_lr = (3e-5, 2e-3)
        conservative_c4_lr = (2e-5, 1.2e-3)
    all_linear = _default_encoder_mode()
    all_both = _default_router_source()
    all_none_inj = _default_feature_injection()
    all_learned = _default_router_mode()
    all_moe = _default_compute_mode()
    all_dense = _all_stage_map("dense_plain")
    all_none = _all_stage_map("none")
    tempo_memory = {
        "macro": ["Tempo", "Memory"],
        "mid": ["Tempo", "Memory"],
        "micro": ["Tempo", "Memory"],
    }

    combos = [
        base_combo(
            "P00",
            combo_family="P",
            combo_role="exact plain C2",
            desc="plain_c2_one_layer",
            layer_layout=["layer"],
            baseline_recipe="C2",
            delta_from_base="1-layer plain SASRec-style stack",
            train_bs=prof["c2_train_bs"],
            eval_bs=prof["c2_eval_bs"],
            lr_min=c2_lr[0],
            lr_max=c2_lr[1],
            attn_dropout_prob=0.10,
            search_hidden_dropout_prob=[0.05, 0.10, 0.15, 0.20],
            stage_compute_mode=all_none,
            stage_router_mode=all_none,
            stage_feature_injection=all_none_inj,
        ),
        base_combo(
            "P01",
            combo_family="P",
            combo_role="exact plain C4 wide",
            desc="plain_c4_wide_two_layer",
            layer_layout=["layer", "layer"],
            baseline_recipe="C4",
            delta_from_base="2-layer wide SASRec-style stack",
            train_bs=prof["c4_train_bs"],
            eval_bs=prof["c4_eval_bs"],
            lr_min=3e-5,
            lr_max=2e-3,
            max_len=20,
            embedding_size=160,
            attn_dropout_prob=0.10,
            search_hidden_dropout_prob=[0.10, 0.15, 0.20, 0.25],
            stage_compute_mode=all_none,
            stage_router_mode=all_none,
            stage_feature_injection=all_none_inj,
        ),
        base_combo(
            "D10",
            combo_family="D",
            combo_role="dense plain wrapper",
            desc="dense_plain_macro_mid",
            layer_layout=["macro", "mid"],
            baseline_recipe="C3",
            delta_from_base="stage wrapper only, dense plain, macro+mid",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "none"},
            stage_router_mode=all_none,
        ),
        base_combo(
            "D11",
            combo_family="D",
            combo_role="dense plain wrapper full",
            desc="dense_plain_macro_mid_micro",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="stage wrapper only, dense plain, full",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "dense_plain"},
            stage_router_mode=all_none,
        ),
        base_combo(
            "D12",
            combo_family="D",
            combo_role="dense FiLM",
            desc="dense_film_macro_mid",
            layer_layout=["macro", "mid"],
            baseline_recipe="C3",
            delta_from_base="dense plain + FiLM, macro+mid",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "none"},
            stage_router_mode=all_none,
            stage_feature_injection={"macro": "film", "mid": "film", "micro": "none"},
        ),
        base_combo(
            "D13",
            combo_family="D",
            combo_role="dense FiLM full",
            desc="dense_film_macro_mid_micro",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="dense plain + FiLM, full",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "dense_plain"},
            stage_router_mode=all_none,
            stage_feature_injection=_all_stage_map("film"),
        ),
        base_combo(
            "D14",
            combo_family="D",
            combo_role="dense gated bias",
            desc="dense_gated_macro_mid",
            layer_layout=["macro", "mid"],
            baseline_recipe="C3",
            delta_from_base="dense plain + gated bias, macro+mid",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "none"},
            stage_router_mode=all_none,
            stage_feature_injection={"macro": "gated_bias", "mid": "gated_bias", "micro": "none"},
        ),
        base_combo(
            "D15",
            combo_family="D",
            combo_role="dense gated bias full",
            desc="dense_gated_macro_mid_micro",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="dense plain + gated bias, full",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "dense_plain"},
            stage_router_mode=all_none,
            stage_feature_injection=_all_stage_map("gated_bias"),
        ),
        base_combo(
            "M20",
            combo_family="M",
            combo_role="macro-only learned MoE both",
            desc="macro_only_moe_both",
            layer_layout=["macro"],
            baseline_recipe="C3",
            delta_from_base="macro-only learned router, hidden+feature",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "moe", "mid": "none", "micro": "none"},
            stage_router_mode={"macro": "learned", "mid": "none", "micro": "none"},
            stage_router_source={"macro": "both", "mid": "both", "micro": "both"},
        ),
        base_combo(
            "M21",
            combo_family="M",
            combo_role="macro+mid learned MoE both",
            desc="macro_mid_moe_both",
            layer_layout=["macro", "mid"],
            baseline_recipe="C3",
            delta_from_base="macro+mid learned router, hidden+feature",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode={"macro": "moe", "mid": "moe", "micro": "none"},
            stage_router_mode={"macro": "learned", "mid": "learned", "micro": "none"},
            stage_router_source={"macro": "both", "mid": "both", "micro": "both"},
        ),
        base_combo(
            "M22",
            combo_family="M",
            combo_role="full learned MoE both anchor",
            desc="full_moe_both_anchor",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="full learned router, hidden+feature",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=c4_lr[0],
            lr_max=c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
        ),
        base_combo(
            "R30",
            combo_family="R",
            combo_role="full rule_soft",
            desc="full_rule_soft",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="rule_soft routing on all stages",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=_all_stage_map("rule_soft"),
            stage_router_source=all_both,
        ),
        base_combo(
            "R31",
            combo_family="R",
            combo_role="full learned MoE hidden-only",
            desc="full_moe_hidden_only",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="learned router hidden-only",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=_all_stage_map("hidden"),
        ),
        base_combo(
            "R32",
            combo_family="R",
            combo_role="full learned MoE feature-only",
            desc="full_moe_feature_only",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="learned router feature-only",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=_all_stage_map("feature"),
        ),
        base_combo(
            "R33",
            combo_family="R",
            combo_role="full learned hidden-only plus gated bias",
            desc="full_moe_hidden_gated_bias",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="learned router hidden-only + gated bias injection",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=_all_stage_map("hidden"),
            stage_feature_injection=_all_stage_map("gated_bias"),
        ),
        base_combo(
            "R34",
            combo_family="R",
            combo_role="hybrid macro learned both plus mid micro rule",
            desc="hybrid_macro_learn_mid_micro_rule",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="macro learned both, mid/micro rule_soft",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode={"macro": "learned", "mid": "rule_soft", "micro": "rule_soft"},
            stage_router_source={"macro": "both", "mid": "both", "micro": "both"},
        ),
        base_combo(
            "E40",
            combo_family="E",
            combo_role="full learned both plus complex encoder all",
            desc="full_moe_both_complex_all",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="complex feature encoder on all stages",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_feature_encoder_mode=_all_stage_map("complex"),
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
        ),
        base_combo(
            "E41",
            combo_family="E",
            combo_role="full learned both plus complex encoder macro",
            desc="full_moe_both_complex_macro",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="complex feature encoder on macro only",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"},
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
        ),
        base_combo(
            "E42",
            combo_family="E",
            combo_role="full learned both plus complex encoder mid",
            desc="full_moe_both_complex_mid",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="complex feature encoder on mid only",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_feature_encoder_mode={"macro": "linear", "mid": "complex", "micro": "linear"},
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
        ),
        base_combo(
            "T50",
            combo_family="T",
            combo_role="token routing mid and micro",
            desc="full_moe_both_mid_token",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="macro session, mid token, micro token",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            stage_router_granularity={"macro": "session", "mid": "token", "micro": "token"},
        ),
        base_combo(
            "T51",
            combo_family="T",
            combo_role="token routing macro mid micro",
            desc="full_moe_both_all_token",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="macro token, mid token, micro token",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            stage_router_granularity={"macro": "token", "mid": "token", "micro": "token"},
        ),
        base_combo(
            "X60",
            combo_family="X",
            combo_role="macro window ablation",
            desc="m22_macro_window_10",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="M22 + macro window 10",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            macro_window=10,
        ),
        base_combo(
            "X61",
            combo_family="X",
            combo_role="feature family ablation",
            desc="m22_tempo_memory_only",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="M22 + Tempo+Memory only",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            family_mask=tempo_memory,
        ),
        base_combo(
            "X62",
            combo_family="X",
            combo_role="sequence length ablation",
            desc="m22_len_30",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="M22 + max len 30",
            train_bs=prof["heavy_train_bs"],
            eval_bs=prof["heavy_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            max_len=30,
        ),
        base_combo(
            "X63",
            combo_family="X",
            combo_role="sparse expert ablation",
            desc="m22_topk_2",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="M22 + top_k=2",
            train_bs=prof["c3_train_bs"],
            eval_bs=prof["c3_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            moe_top_k=2,
        ),
        base_combo(
            "C70",
            combo_family="C",
            combo_role="capacity full learned both",
            desc="m22_expert_scale_3",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="M22 + expert scale 3",
            train_bs=prof["heavy_train_bs"],
            eval_bs=prof["heavy_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_moe,
            stage_router_mode=all_learned,
            stage_router_source=all_both,
            expert_scale=3,
        ),
        base_combo(
            "C71",
            combo_family="C",
            combo_role="capacity dense plain control",
            desc="dense_plain_param_match_c70",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="C70 param-matched dense plain",
            train_bs=prof["heavy_train_bs"],
            eval_bs=prof["heavy_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_dense,
            stage_router_mode=all_none,
            dense_hidden_scale=3.0,
        ),
        base_combo(
            "C72",
            combo_family="C",
            combo_role="capacity dense FiLM control",
            desc="dense_film_param_match_c70",
            layer_layout=["macro", "mid", "micro"],
            baseline_recipe="C3",
            delta_from_base="C70 param-matched dense FiLM",
            train_bs=prof["heavy_train_bs"],
            eval_bs=prof["heavy_eval_bs"],
            lr_min=conservative_c4_lr[0],
            lr_max=conservative_c4_lr[1],
            stage_compute_mode=all_dense,
            stage_router_mode=all_none,
            stage_feature_injection=_all_stage_map("film"),
            dense_hidden_scale=3.0,
        ),
    ]

    assert len(combos) == 28
    for idx, row in enumerate(combos):
        if row["combo_id"] not in {"P00", "P01"}:
            row["baseline_recipe"] = "C4"
            row["train_batch_size"] = prof["c4_train_bs"]
            row["eval_batch_size"] = prof["c4_eval_bs"]
            row["attn_dropout_prob"] = 0.10
            row["search_hidden_dropout_prob"] = list(DROPOUT_CHOICES)
            row["search_weight_decay"] = list(WD_CHOICES)
        if row["combo_id"] in {"X62", "C70", "C71", "C72"}:
            row["train_batch_size"] = prof["heavy_train_bs"]
            row["eval_batch_size"] = prof["heavy_eval_bs"]
        row["seed_offset"] = idx
        row["dataset"] = dataset
        row["feature_mode"] = "full_v3"
        row["d_feat_emb"] = 16
        row["d_expert_hidden"] = 128
        row["d_router_hidden"] = 64
        row["has_diag"] = any(
            row["stage_compute_mode"].get(stage) == "moe" and row["stage_router_mode"].get(stage) in {"learned", "rule_soft"}
            for stage in ("macro", "mid", "micro")
        )
    return combos


def build_command(combo: dict, gpu_id: str, args) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    combo_max_evals = int(combo.get("launch_max_evals", args.max_evals))
    eval_logging_timing = str(getattr(args, "eval_logging_timing", "final_only") or "final_only").strip().lower()
    if eval_logging_timing not in {"final_only", "per_eval"}:
        eval_logging_timing = "final_only"
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(combo_max_evals),
        "--tune-epochs",
        str(args.tune_epochs),
        "--tune-patience",
        str(args.tune_patience),
        "--seed",
        str(args.seed_base + int(combo["seed_offset"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        combo["combo_id"],
        "model=featured_moe_n3_tune",
        f"dataset={combo['dataset']}",
        "eval_mode=session",
        f"feature_mode={combo['feature_mode']}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        f"fmoe_diag_logging={hydra_literal(combo['has_diag'])}",
        f"fmoe_eval_logging_timing={eval_logging_timing}",
        f"fmoe_feature_ablation_logging={hydra_literal(bool(getattr(args, 'feature_ablation_logging', False)))}",
        "fmoe_special_logging=true",
        f"MAX_ITEM_LIST_LENGTH={combo['MAX_ITEM_LIST_LENGTH']}",
        f"train_batch_size={combo['train_batch_size']}",
        f"eval_batch_size={combo['eval_batch_size']}",
        f"embedding_size={combo['embedding_size']}",
        f"num_heads={combo['num_heads']}",
        f"attn_dropout_prob={combo['attn_dropout_prob']}",
        f"d_ff={combo['d_ff']}",
        f"d_feat_emb={combo['d_feat_emb']}",
        f"d_expert_hidden={combo['d_expert_hidden']}",
        f"d_router_hidden={combo['d_router_hidden']}",
        f"expert_scale={combo['expert_scale']}",
        f"++layer_layout={hydra_literal(combo['layer_layout'])}",
        f"++stage_feature_encoder_mode={hydra_literal(combo['stage_feature_encoder_mode'])}",
        f"++stage_compute_mode={hydra_literal(combo['stage_compute_mode'])}",
        f"++stage_router_mode={hydra_literal(combo['stage_router_mode'])}",
        f"++stage_router_source={hydra_literal(combo['stage_router_source'])}",
        f"++stage_feature_injection={hydra_literal(combo['stage_feature_injection'])}",
        f"++stage_router_type={hydra_literal(combo['stage_router_type'])}",
        f"++stage_router_granularity={hydra_literal(combo['stage_router_granularity'])}",
        f"++stage_feature_family_mask={hydra_literal(combo['stage_feature_family_mask'])}",
        f"macro_history_window={combo['macro_history_window']}",
        f"dense_hidden_scale={combo['dense_hidden_scale']}",
        f"moe_top_k={combo['moe_top_k']}",
        "moe_top_k_policy=auto",
        "moe_top_k_ratio=0.5",
        "macro_session_pooling=mean",
        "feature_encoder_mode=linear",
        "router_impl=learned",
        "use_valid_ratio_gating=false",
        f"balance_loss_lambda={combo['balance_loss_lambda']}",
        f"z_loss_lambda={combo['z_loss_lambda']}",
        f"gate_entropy_lambda={combo['gate_entropy_lambda']}",
        f"gate_entropy_until={combo['gate_entropy_until']}",
        f"rule_agreement_lambda={combo['rule_agreement_lambda']}",
        f"group_coverage_lambda={combo['group_coverage_lambda']}",
        f"group_prior_align_lambda={combo['group_prior_align_lambda']}",
        f"feature_group_bias_lambda={combo['feature_group_bias_lambda']}",
        f"feature_group_prior_temperature={combo['feature_group_prior_temperature']}",
        "lr_scheduler_type=none",
        "alpha_warmup_until=0",
        "temperature_warmup_until=0",
        "mid_router_temperature=1.2",
        "micro_router_temperature=1.2",
        "mid_router_temperature_start=1.2",
        "micro_router_temperature_start=1.2",
        "moe_top_k_start=0",
        "moe_top_k_warmup_until=0",
        f"++combo_family={hydra_literal(combo['combo_family'])}",
        f"++combo_role={hydra_literal(combo['combo_role'])}",
        f"++baseline_recipe={hydra_literal(combo['baseline_recipe'])}",
        f"++delta_from_base={hydra_literal(combo['delta_from_base'])}",
        f"++combo_desc={hydra_literal(combo['desc'])}",
        f"++search.MAX_ITEM_LIST_LENGTH={hydra_literal([combo['MAX_ITEM_LIST_LENGTH']])}",
        f"++search.train_batch_size={hydra_literal([combo['train_batch_size']])}",
        f"++search.eval_batch_size={hydra_literal([combo['eval_batch_size']])}",
        f"++search.embedding_size={hydra_literal([combo['embedding_size']])}",
        f"++search.num_heads={hydra_literal([combo['num_heads']])}",
        f"++search.d_ff={hydra_literal([combo['d_ff']])}",
        f"++search.attn_dropout_prob={hydra_literal([combo['attn_dropout_prob']])}",
        f"++search.d_feat_emb={hydra_literal([combo['d_feat_emb']])}",
        f"++search.d_expert_hidden={hydra_literal([combo['d_expert_hidden']])}",
        f"++search.d_router_hidden={hydra_literal([combo['d_router_hidden']])}",
        f"++search.expert_scale={hydra_literal([combo['expert_scale']])}",
        f"++search.layer_layout={hydra_literal([combo['layer_layout']])}",
        f"++search.stage_feature_encoder_mode={hydra_literal([combo['stage_feature_encoder_mode']])}",
        f"++search.stage_compute_mode={hydra_literal([combo['stage_compute_mode']])}",
        f"++search.stage_router_mode={hydra_literal([combo['stage_router_mode']])}",
        f"++search.stage_router_source={hydra_literal([combo['stage_router_source']])}",
        f"++search.stage_feature_injection={hydra_literal([combo['stage_feature_injection']])}",
        f"++search.stage_router_granularity={hydra_literal([combo['stage_router_granularity']])}",
        f"++search.stage_feature_family_mask={hydra_literal([combo['stage_feature_family_mask']])}",
        f"++search.macro_history_window={hydra_literal([combo['macro_history_window']])}",
        f"++search.moe_top_k={hydra_literal([combo['moe_top_k']])}",
        f"++search.dense_hidden_scale={hydra_literal([combo['dense_hidden_scale']])}",
        f"++search.learning_rate={hydra_literal([combo['lr_min'], combo['lr_max']])}",
        f"++search.weight_decay={hydra_literal(combo['search_weight_decay'])}",
        f"++search.hidden_dropout_prob={hydra_literal(combo['search_hidden_dropout_prob'])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search.lr_scheduler_type=[none]",
    ]
    return cmd


def log_path(combo: dict, dataset: str) -> Path:
    dataset_tag = dataset.replace("/", "_")
    model_tag = "FMoEN3"
    phase_bucket = PHASE
    root = LOG_ROOT / AXIS / phase_bucket / dataset_tag / model_tag
    root.mkdir(parents=True, exist_ok=True)
    authored_idx = int(combo.get("seed_offset", 0)) + 1
    stem = f"{authored_idx:03d}_{sanitize_slug(combo['combo_id'])}_{sanitize_slug(combo['desc'])}"
    out_path = root / f"{stem}.log"
    if not out_path.exists():
        out_path.touch(exist_ok=False)
        return out_path
    rerun_idx = 2
    while True:
        candidate = root / f"{stem}_r{rerun_idx:02d}.log"
        if not candidate.exists():
            candidate.touch(exist_ok=False)
            return candidate
        rerun_idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=8300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--only", default="", help="Comma-separated combo ids to launch.")
    parser.add_argument("--use-recommended-budget", action="store_true")
    parser.add_argument("--eval-logging-timing", default="final_only", choices=["final_only", "per_eval"])
    parser.add_argument("--feature-ablation-logging", action="store_true")
    args = parser.parse_args()

    gpus = [token.strip() for token in args.gpus.split(",") if token.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    combos = build_combos(args.dataset)
    if args.only.strip():
        wanted = {token.strip().upper() for token in args.only.split(",") if token.strip()}
        combos = [combo for combo in combos if combo["combo_id"].upper() in wanted]
        if not combos:
            raise SystemExit(f"No combos matched --only={args.only}")
    for combo in combos:
        combo.update(estimated_runtime_profile(combo["combo_id"]))
        combo["launch_max_evals"] = (
            int(combo["recommended_max_evals"]) if args.use_recommended_budget else int(args.max_evals)
        )
        combo["estimated_total_budget_min"] = float(combo["estimated_50ep_single_eval_min"]) * float(combo["launch_max_evals"])

    planned_bins = plan_gpu_bins(combos, gpus, cost_key="estimated_total_budget_min", group_size=len(gpus))
    planned_queue = []
    for gpu_id in gpus:
        bucket = planned_bins[gpu_id]
        for order_idx, combo in enumerate(bucket, start=1):
            combo["assigned_gpu"] = gpu_id
            combo["assigned_order"] = order_idx
    planned_queue = list(combos)

    manifest = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "gpus": gpus,
        "max_evals": args.max_evals,
        "tune_epochs": args.tune_epochs,
        "tune_patience": args.tune_patience,
        "use_recommended_budget": bool(args.use_recommended_budget),
        "combos": combos,
    }
    INVENTORY_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_out) if args.manifest_out else INVENTORY_ROOT / f"core28_{int(time.time())}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Manifest] {manifest_path}")

    if args.dry_run:
        for combo in planned_queue:
            cmd = build_command(combo, combo["assigned_gpu"], args)
            print(
                f"[DryRun] group={combo.get('launch_group', 0):02d} "
                f"{combo['combo_id']} gpu={combo['assigned_gpu']} :: {' '.join(cmd)}"
            )
        return

    per_gpu_queues = {gpu: deque(planned_bins[gpu]) for gpu in gpus}
    active: dict[str, dict] = {}
    last_summary_refresh = 0.0

    def refresh_phase_summaries(*, force: bool = False, reason: str = "") -> None:
        nonlocal last_summary_refresh
        now = time.time()
        if not force and (now - last_summary_refresh) < SUMMARY_REFRESH_SEC:
            return
        try:
            phase_paths = build_fmoe_n3_summaries(AXIS, PHASE)
            axis_path = build_fmoe_n3_axis_summary(AXIS)
            artifact_paths = build_artifact_views(AXIS)
            last_summary_refresh = now
            if reason:
                if phase_paths:
                    print(f"[Summary] refreshed ({reason}) -> {axis_path}")
                    print(
                        "[Artifacts] "
                        f"special={artifact_paths['special_summary']} "
                        f"feature_ablation={artifact_paths['feature_ablation_summary']} "
                        f"diag={artifact_paths['diag_summary']}"
                    )
                else:
                    print(f"[Summary] refreshed ({reason})")
        except Exception as exc:
            print(f"[Summary] refresh failed ({reason or 'unknown'}): {exc}")

    refresh_phase_summaries(force=True, reason="start")

    while any(per_gpu_queues[gpu] for gpu in gpus) or active:
        saw_log_growth = False
        saw_completion = False
        for gpu_id in list(gpus):
            proc_info = active.get(gpu_id)
            if proc_info is not None:
                combo = proc_info["combo"]
                proc = proc_info["proc"]
                if proc.poll() is None:
                    log_file = proc_info["log_file"]
                    try:
                        current_size = int(log_file.stat().st_size)
                    except Exception:
                        current_size = -1
                    if current_size != int(proc_info.get("last_seen_size", -1)):
                        proc_info["last_seen_size"] = current_size
                        saw_log_growth = True
                    continue
                print(f"[Done] {combo['combo_id']} gpu={gpu_id} rc={proc.returncode}")
                del active[gpu_id]
                saw_completion = True
            if not per_gpu_queues[gpu_id]:
                continue
            combo = per_gpu_queues[gpu_id].popleft()
            cmd = build_command(combo, gpu_id, args)
            env = os.environ.copy()
            env.setdefault("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
            log_file = log_path(combo, args.dataset)
            env["LOG_FILE"] = str(log_file)
            env["RUN_LOGS_DIR"] = str((ARTIFACT_ROOT / "logs").resolve())
            env.pop("TRACK_PHASE_SUMMARY", None)
            env.pop("FMOE_N3_PHASE_SUMMARY", None)
            env.pop("TRACK_SUMMARY_PHASE", None)
            env.pop("TRACK_SUMMARY_AXIS", None)
            env.pop("TRACK_SUMMARY_SCRIPT", None)
            print(
                f"[Launch] {combo['combo_id']} gpu={gpu_id} "
                f"budget_evals={combo['launch_max_evals']} "
                f"est50epx1={combo['estimated_50ep_single_eval_min']:.1f}m"
            )
            proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), env=env)
            active[gpu_id] = {
                "combo": combo,
                "proc": proc,
                "log_file": log_file,
                "last_seen_size": 0,
            }

        if saw_completion:
            refresh_phase_summaries(force=True, reason="combo_done")
        elif saw_log_growth:
            refresh_phase_summaries(force=False, reason="log_progress")
        time.sleep(3)

    refresh_phase_summaries(force=True, reason="finished")
    print("[Finished] all combos completed")


if __name__ == "__main__":
    main()
