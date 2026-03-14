#!/usr/bin/env python3
"""Launch FMoE_N3 phase2 40-combo track.

Block allocation (40 total):
  A (16): Best exploitation from Phase 1 + some large structural variants
  B  (4): lr/dropout fixed; wd sweep
  C  (8): aux/reg parameter sweep
  D (12): Router structure + group feature exploration
"""

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
AXIS = "phase2_router_v1"
PHASE = "P2"
SUMMARY_REFRESH_SEC = 300.0

WD_CHOICES = [0.0, 1e-7, 1e-6, 5e-5, 1e-4]
DROPOUT_CHOICES = [0.10, 0.15, 0.20, 0.25]
DEFAULT_LR_BASE = (1.5e-4, 1.0e-2)
DEFAULT_LR_AGGR = (2.0e-4, 1.0e-2)


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


def runtime_bucket(combo: dict) -> str:
    if int(combo.get("MAX_ITEM_LIST_LENGTH", 20)) >= 50:
        return "heavy"
    if int(combo.get("embedding_size", 128)) >= 256:
        return "heavy"
    if combo.get("stage_router_type", {}).get("macro") == "factored":
        return "medium-heavy"
    return "medium"


def recommended_max_evals(combo: dict) -> int:
    bucket = runtime_bucket(combo)
    if bucket == "heavy":
        return 10
    if combo.get("combo_family") == "D":
        return 15  # new structures need slightly more trials
    return 20


def estimated_runtime_profile(combo: dict) -> dict:
    bucket = runtime_bucket(combo)
    trial_min = 1.6 if bucket == "heavy" else 1.1 if bucket == "medium-heavy" else 1.0
    fixed_overhead = 0.4
    est_50 = fixed_overhead + trial_min * 50.0
    return {
        "runtime_bucket": bucket,
        "trial_min_hint": trial_min,
        "fixed_overhead_min_hint": fixed_overhead,
        "estimated_50ep_single_eval_min": est_50,
        "recommended_max_evals": recommended_max_evals(combo),
    }


def plan_gpu_bins(combos: list[dict], gpus: list[str], *, cost_key: str) -> dict[str, list[dict]]:
    bins = {gpu: [] for gpu in gpus}
    loads = {gpu: 0.0 for gpu in gpus}
    window = max(1, len(gpus))
    ordered = list(combos)
    for group_idx, start in enumerate(range(0, len(ordered), window), start=1):
        chunk = ordered[start: start + window]
        available = list(gpus)
        combo_to_gpu: dict[str, str] = {}
        ranked = sorted(chunk, key=lambda row: (-float(row.get(cost_key, 0.0) or 0.0), row["combo_id"]))
        for combo in ranked:
            gpu = min(available, key=lambda gid: (loads[gid], gid))
            combo_to_gpu[combo["combo_id"]] = gpu
            available.remove(gpu)
        for local_order, combo in enumerate(chunk, start=1):
            combo["launch_group"] = group_idx
            combo["launch_group_order"] = local_order
            gpu = combo_to_gpu[combo["combo_id"]]
            bins[gpu].append(combo)
            loads[gpu] += float(combo.get(cost_key, 0.0) or 0.0)
    return bins


def dataset_profile(dataset: str) -> dict:
    key = dataset.lower()
    if key == "lastfm0.03":
        return {
            "c4_train_bs": 2048,
            "c4_eval_bs": 4096,
            "heavy_train_bs": 1536,
            "heavy_eval_bs": 3072,
            "len50_train_bs": 1024,
            "len50_eval_bs": 2048,
        }
    return {
        "c4_train_bs": 4096,
        "c4_eval_bs": 4096,
        "heavy_train_bs": 2048,
        "heavy_eval_bs": 4096,
        "len50_train_bs": 1536,
        "len50_eval_bs": 3072,
    }


def _all_stage_map(value: str) -> dict:
    return {"macro": value, "mid": value, "micro": value}


def _default_granularity() -> dict:
    return {"macro": "session", "mid": "session", "micro": "token"}


def _format_setting(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _default_combo_reference(dataset: str) -> dict:
    prof = dataset_profile(dataset)
    return {
        "MAX_ITEM_LIST_LENGTH": 20,
        "train_batch_size": prof["c4_train_bs"],
        "eval_batch_size": prof["c4_eval_bs"],
        "embedding_size": 128,
        "d_ff": 256,
        "num_heads": 4,
        "attn_dropout_prob": 0.10,
        "d_feat_emb": 16,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "expert_scale": 3,
        "layer_layout": ["macro", "mid", "micro"],
        "stage_feature_encoder_mode": _all_stage_map("linear"),
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": _all_stage_map("both"),
        "stage_feature_injection": _all_stage_map("none"),
        "stage_router_type": _all_stage_map("standard"),
        "stage_router_granularity": _default_granularity(),
        "macro_history_window": 5,
        "moe_top_k": 0,
        "balance_loss_lambda": 0.004,
        "z_loss_lambda": 1e-4,
        "gate_entropy_lambda": 0.0,
        "gate_entropy_until": 0.0,
        "rule_agreement_lambda": 0.0,
        "group_coverage_lambda": 0.0,
        "group_prior_align_lambda": 0.0,
        "feature_group_bias_lambda": 0.0,
        "feature_group_prior_temperature": 1.0,
        "factored_group_balance_lambda": 0.0,
        "lr_scheduler_type": "warmup_cosine",
        "lr_scheduler_warmup_ratio": 0.1,
        "lr_scheduler_min_lr_ratio": 0.1,
        "lr_scheduler_plateau_factor": 0.5,
        "lr_scheduler_plateau_patience": 3,
        "feature_mode": "full_v3",
        "lr_min": DEFAULT_LR_BASE[0],
        "lr_max": DEFAULT_LR_BASE[1],
        "search_hidden_dropout_prob": list(DROPOUT_CHOICES),
        "search_weight_decay": list(WD_CHOICES),
    }


def summarize_combo_changes(combo: dict) -> list[str]:
    reference = _default_combo_reference(combo["dataset"])
    labels = {
        "MAX_ITEM_LIST_LENGTH": "max_seq_length",
        "train_batch_size": "train_batch_size",
        "embedding_size": "embedding_size",
        "d_feat_emb": "d_feat_emb",
        "d_expert_hidden": "d_expert_hidden",
        "d_router_hidden": "d_router_hidden",
        "expert_scale": "expert_scale",
        "layer_layout": "layer_layout",
        "stage_feature_encoder_mode": "stage_feature_encoder_mode",
        "stage_router_source": "stage_router_source",
        "stage_feature_injection": "stage_feature_injection",
        "stage_router_type": "stage_router_type",
        "stage_router_granularity": "stage_router_granularity",
        "stage_feature_family_mask": "stage_feature_family_mask",
        "macro_history_window": "macro_history_window",
        "balance_loss_lambda": "balance_loss_lambda",
        "z_loss_lambda": "z_loss_lambda",
        "gate_entropy_lambda": "gate_entropy_lambda",
        "group_prior_align_lambda": "group_prior_align_lambda",
        "feature_group_bias_lambda": "feature_group_bias_lambda",
        "factored_group_balance_lambda": "factored_group_balance_lambda",
        "lr_scheduler_type": "lr_scheduler_type",
        "lr_min": "learning_rate.min",
        "lr_max": "learning_rate.max",
    }
    changed = []
    for key in labels:
        if combo.get(key) != reference.get(key):
            changed.append(
                f"{labels[key]}={_format_setting(combo.get(key))} (base={_format_setting(reference.get(key))})"
            )
    return changed


def write_log_preamble(log_file: Path, combo: dict, gpu_id: str, args, cmd: list[str]) -> None:
    important_changes = summarize_combo_changes(combo)
    fixed_fields = [
        "feature_mode",
        "MAX_ITEM_LIST_LENGTH",
        "train_batch_size",
        "eval_batch_size",
        "embedding_size",
        "d_ff",
        "num_heads",
        "attn_dropout_prob",
        "d_feat_emb",
        "d_expert_hidden",
        "d_router_hidden",
        "expert_scale",
        "layer_layout",
        "stage_feature_encoder_mode",
        "stage_compute_mode",
        "stage_router_mode",
        "stage_router_source",
        "stage_feature_injection",
        "stage_router_type",
        "stage_router_granularity",
        "stage_feature_family_mask",
        "macro_history_window",
        "moe_top_k",
        "balance_loss_lambda",
        "z_loss_lambda",
        "gate_entropy_lambda",
        "rule_agreement_lambda",
        "group_coverage_lambda",
        "group_prior_align_lambda",
        "feature_group_bias_lambda",
        "feature_group_prior_temperature",
        "factored_group_balance_lambda",
        "gate_entropy_until",
        "lr_scheduler_type",
        "lr_scheduler_warmup_ratio",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_plateau_factor",
        "lr_scheduler_plateau_patience",
    ]
    lines = [
        "[PHASE2_COMBO_HEADER]",
        f"combo_id={combo['combo_id']} family={combo['combo_family']} role={combo['combo_role']} desc={combo['desc']}",
        f"dataset={combo['dataset']} gpu={gpu_id} launch_group={combo.get('launch_group', 0)} launch_order={combo.get('assigned_order', 0)}",
        f"launch_max_evals={combo['launch_max_evals']} cli_max_evals={args.max_evals}",
        f"tune_epochs={args.tune_epochs} tune_patience={args.tune_patience} seed={args.seed_base + int(combo['seed_offset'])}",
        f"runtime_bucket={combo.get('runtime_bucket', 'unknown')} est_50ep_single_eval_min={combo.get('estimated_50ep_single_eval_min', 0.0):.1f}",
        "",
        "[PURPOSE]",
        f"- {combo['desc']} | family={combo['combo_family']} | role={combo['combo_role']}",
        "",
        "[KEY_CHANGES_VS_DEFAULT]",
    ]
    if important_changes:
        lines.extend(f"- {item}" for item in important_changes)
    else:
        lines.append("- none")
    lines.extend(["", "[FIXED_SETTINGS]"])
    lines.extend(f"- {field}={_format_setting(combo[field])}" for field in fixed_fields if field in combo)
    lines.extend([
        "",
        "[SEARCH_SPACE]",
        f"- learning_rate=loguniform[{combo['lr_min']}, {combo['lr_max']}]",
        f"- hidden_dropout_prob=choice{_format_setting(combo['search_hidden_dropout_prob'])}",
        f"- weight_decay=choice{_format_setting(combo['search_weight_decay'])}",
        f"- lr_scheduler_type=choice[{combo['lr_scheduler_type']}]",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ])
    log_file.write_text("\n".join(lines), encoding="utf-8")


def make_combo(
    combo_id: str,
    *,
    combo_family: str,
    combo_role: str,
    desc: str,
    layer_layout: list[str],
    train_bs: int,
    eval_bs: int,
    lr_min: float,
    lr_max: float,
    max_len: int = 20,
    embedding_size: int = 128,
    num_heads: int = 4,
    stage_feature_encoder_mode: dict | None = None,
    stage_router_source: dict | None = None,
    stage_feature_injection: dict | None = None,
    stage_router_granularity: dict | None = None,
    stage_router_type: dict | None = None,
    stage_feature_family_mask: dict | None = None,
    expert_scale: int = 3,
    d_feat_emb: int = 16,
    d_expert_hidden: int = 128,
    d_router_hidden: int = 64,
    balance_loss_lambda: float = 0.004,
    z_loss_lambda: float = 1e-4,
    gate_entropy_lambda: float = 0.0,
    gate_entropy_until: float = 0.0,
    rule_agreement_lambda: float = 0.0,
    group_coverage_lambda: float = 0.0,
    group_prior_align_lambda: float = 0.0,
    feature_group_bias_lambda: float = 0.0,
    feature_group_prior_temperature: float = 1.0,
    factored_group_balance_lambda: float = 0.0,
    lr_scheduler_type: str = "warmup_cosine",
    lr_scheduler_warmup_ratio: float = 0.1,
    lr_scheduler_min_lr_ratio: float = 0.1,
    search_weight_decay: list | None = None,
    search_dropout: list | None = None,
) -> dict:
    return {
        "combo_id": combo_id,
        "combo_family": combo_family,
        "combo_role": combo_role,
        "desc": desc,
        "baseline_recipe": "P2",
        "delta_from_base": desc,
        "layer_layout": list(layer_layout),
        "train_batch_size": train_bs,
        "eval_batch_size": eval_bs,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "MAX_ITEM_LIST_LENGTH": max_len,
        "embedding_size": embedding_size,
        "d_ff": int(2 * embedding_size),
        "num_heads": num_heads,
        "attn_dropout_prob": 0.10,
        "stage_feature_encoder_mode": stage_feature_encoder_mode or _all_stage_map("linear"),
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": stage_router_source or _all_stage_map("both"),
        "stage_feature_injection": stage_feature_injection or _all_stage_map("none"),
        "stage_router_type": stage_router_type or _all_stage_map("standard"),
        "stage_router_granularity": stage_router_granularity or _default_granularity(),
        "stage_feature_family_mask": stage_feature_family_mask or {},
        "macro_history_window": 10 if max_len >= 30 else 5,
        "moe_top_k": 0,
        "expert_scale": expert_scale,
        "dense_hidden_scale": 1.0,
        "search_hidden_dropout_prob": list(search_dropout or DROPOUT_CHOICES),
        "search_weight_decay": list(search_weight_decay or WD_CHOICES),
        "d_feat_emb": d_feat_emb,
        "d_expert_hidden": d_expert_hidden,
        "d_router_hidden": d_router_hidden,
        "balance_loss_lambda": balance_loss_lambda,
        "z_loss_lambda": z_loss_lambda,
        "gate_entropy_lambda": gate_entropy_lambda,
        "gate_entropy_until": gate_entropy_until,
        "rule_agreement_lambda": rule_agreement_lambda,
        "group_coverage_lambda": group_coverage_lambda,
        "group_prior_align_lambda": group_prior_align_lambda,
        "feature_group_bias_lambda": feature_group_bias_lambda,
        "feature_group_prior_temperature": feature_group_prior_temperature,
        "factored_group_balance_lambda": factored_group_balance_lambda,
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_warmup_ratio": lr_scheduler_warmup_ratio,
        "lr_scheduler_min_lr_ratio": lr_scheduler_min_lr_ratio,
        "lr_scheduler_plateau_factor": 0.5,
        "lr_scheduler_plateau_patience": 3,
        "has_diag": True,
    }


# ---------------------------------------------------------------------------
# Stage-level family mask helpers
# ---------------------------------------------------------------------------

def _mask_all(groups: list[str]) -> dict:
    """Apply the same group mask to all stages."""
    return {"macro": groups, "mid": groups, "micro": groups}


def build_combos(dataset: str) -> list[dict]:
    prof = dataset_profile(dataset)
    lr_base = DEFAULT_LR_BASE
    lr_aggr = DEFAULT_LR_AGGR
    # wd-only search (Block B uses fixed lr; search only over wd)
    wd_only_search = [0.0, 1e-7, 1e-6, 1e-5, 5e-5]

    # ===================================================================
    # BLOCK A (16): Best exploitation from Phase 1 + large structural variants
    # Rule: router_type=standard, no new group features — pure P1 architecture
    # ===================================================================
    a_combos = [
        # --- A1: Core 3-stage anchors ---
        make_combo("PA01", combo_family="A", combo_role="anchor",
                   desc="anchor_baseline_3stage",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1]),

        make_combo("PA04", combo_family="A", combo_role="anchor",
                   desc="macro_complex_enc_anchor",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"}),

        make_combo("PA05", combo_family="A", combo_role="anchor",
                   desc="all_gated_bias",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_injection=_all_stage_map("gated_bias")),

        # --- A2: Large structural variants (크게 다른 것들) ---
        # Very different: pure FFN stage blocks, no attention routing
        make_combo("PA_L04", combo_family="A", combo_role="layout",
                   desc="ffn_only_between_stages",
                   layer_layout=["layer", "macro_ffn", "mid_ffn", "micro_ffn"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1]),

        # Very different: 2 full transformer layers before any stage
        make_combo("PA_L08", combo_family="A", combo_role="layout",
                   desc="deep_layer_prefix_2x",
                   layer_layout=["layer", "layer", "macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1]),

        # Very different: large model (embed256)
        make_combo("PA_X65", combo_family="A", combo_role="scale",
                   desc="embed256_large_model",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   embedding_size=256, d_feat_emb=24, d_router_hidden=96,
                   d_expert_hidden=192,
                   balance_loss_lambda=0.006, z_loss_lambda=3e-4),

        # Very different: tiny model (embed64)
        make_combo("PA_X64", combo_family="A", combo_role="scale",
                   desc="embed64_tiny_model",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   embedding_size=64, d_feat_emb=10, d_router_hidden=48,
                   d_expert_hidden=96,
                   balance_loss_lambda=0.002, z_loss_lambda=1e-4),

        # Very different: long sequence (len50)
        make_combo("PA_N50", combo_family="A", combo_role="length",
                   desc="len50_probe",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["len50_train_bs"], eval_bs=prof["len50_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   max_len=50, expert_scale=3),

        # --- A3: Feature routing & encoder variants ---
        # Pure feature-driven routing (very different from hidden-driven)
        make_combo("PA_FS", combo_family="A", combo_role="routing",
                   desc="feature_source_only",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_source=_all_stage_map("feature")),

        # All-complex encoder (big encoding upgrade)
        make_combo("PA_F03", combo_family="A", combo_role="feature",
                   desc="all_complex_encoder",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_encoder_mode=_all_stage_map("complex")),

        # Macro-only gated bias (partial injection)
        make_combo("PA_F04", combo_family="A", combo_role="feature",
                   desc="macro_gated_bias_only",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_injection={"macro": "gated_bias", "mid": "none", "micro": "none"}),

        # Token granularity for all stages
        make_combo("PA_T3", combo_family="A", combo_role="routing",
                   desc="all_token_granularity",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_granularity=_all_stage_map("token")),

        # --- A4: Scheduler/LR variants ---
        make_combo("PA_S02", combo_family="A", combo_role="sched",
                   desc="warmup_cosine_aggressive_lr",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_aggr[0], lr_max=lr_aggr[1],
                   lr_scheduler_type="warmup_cosine", lr_scheduler_warmup_ratio=0.15),

        make_combo("PA_S03", combo_family="A", combo_role="sched",
                   desc="plateau_scheduler",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   lr_scheduler_type="plateau", lr_scheduler_warmup_ratio=0.0),

        # --- A5: Multi-axis combos (앞선 상위권 조합 섞기) ---
        # macro_complex + gated_bias + aggressive LR  → layer-wide injection + rich encoding
        make_combo("PA_MX1", combo_family="A", combo_role="mix",
                   desc="macro_complex_plus_gated_bias_aggr",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_aggr[0], lr_max=lr_aggr[1],
                   stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"},
                   stage_feature_injection=_all_stage_map("gated_bias"),
                   lr_scheduler_type="warmup_cosine", lr_scheduler_warmup_ratio=0.15),

        # len50 + complex encoder + plateau (long-sequence specialization)
        make_combo("PA_MX2", combo_family="A", combo_role="mix",
                   desc="len50_complex_plateau",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["len50_train_bs"], eval_bs=prof["len50_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   max_len=50, expert_scale=3,
                   stage_feature_encoder_mode=_all_stage_map("complex"),
                   lr_scheduler_type="plateau", lr_scheduler_warmup_ratio=0.0),
    ]
    assert len(a_combos) == 16, f"Block A must have 16 combos, got {len(a_combos)}"

    # ===================================================================
    # BLOCK B (4): Fixed lr/dropout + wd sweep (isolate weight_decay effect)
    # Base: 3-stage, both, standard, macro_complex since it's likely best anchor
    # lr=3.0e-4 fixed (single point), dropout=0.20 fixed
    # ===================================================================
    b_wd_search = [0.0, 1e-6, 1e-5, 5e-5]
    b_combos = [
        make_combo("PB01", combo_family="B", combo_role="wd_sweep",
                   desc="wd0_fixed_lr",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=3.0e-4, lr_max=3.0e-4,
                   search_weight_decay=[0.0],
                   search_dropout=[0.20]),

        make_combo("PB02", combo_family="B", combo_role="wd_sweep",
                   desc="wd1e-6_fixed_lr",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=3.0e-4, lr_max=3.0e-4,
                   search_weight_decay=[1e-6],
                   search_dropout=[0.20]),

        make_combo("PB03", combo_family="B", combo_role="wd_sweep",
                   desc="wd1e-5_fixed_lr",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=3.0e-4, lr_max=3.0e-4,
                   search_weight_decay=[1e-5],
                   search_dropout=[0.20]),

        make_combo("PB04", combo_family="B", combo_role="wd_sweep",
                   desc="wd1e-6_with_group_prior_align",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=3.0e-4, lr_max=3.0e-4,
                   search_weight_decay=[1e-6],
                   search_dropout=[0.20],
                   group_prior_align_lambda=5e-4),
    ]
    assert len(b_combos) == 4

    # ===================================================================
    # BLOCK C (8): aux/reg parameter exploration
    # Base structure: 3-stage, both, standard, macro_complex (B0) fixed
    # Vary: balance/z/entropy/group_prior_align — single-axis + combos
    # ===================================================================
    c_combos = [
        # --- Single-axis ---
        make_combo("PC01", combo_family="C", combo_role="aux_single",
                   desc="strong_balance_0006",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   balance_loss_lambda=0.006, z_loss_lambda=1e-4),

        make_combo("PC02", combo_family="C", combo_role="aux_single",
                   desc="strong_z_loss_3e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   balance_loss_lambda=0.004, z_loss_lambda=3e-4),

        make_combo("PC03", combo_family="C", combo_role="aux_single",
                   desc="gate_entropy_reg_2e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   gate_entropy_lambda=2e-4),

        make_combo("PC04", combo_family="C", combo_role="aux_single",
                   desc="group_prior_align_1e-3",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   group_prior_align_lambda=1e-3),

        # --- Two-axis combos ---
        make_combo("PC05", combo_family="C", combo_role="aux_combo",
                   desc="balance_0006_z_3e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   balance_loss_lambda=0.006, z_loss_lambda=3e-4),

        make_combo("PC06", combo_family="C", combo_role="aux_combo",
                   desc="balance_0004_group_prior_5e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   balance_loss_lambda=0.004, group_prior_align_lambda=5e-4),

        make_combo("PC07", combo_family="C", combo_role="aux_combo",
                   desc="entropy_2e-4_group_prior_5e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   gate_entropy_lambda=2e-4, group_prior_align_lambda=5e-4),

        # --- Three-axis heavy ---
        make_combo("PC08", combo_family="C", combo_role="aux_heavy",
                   desc="balance_0006_z_1e-4_group_prior_1e-3",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   balance_loss_lambda=0.006, z_loss_lambda=1e-4,
                   group_prior_align_lambda=1e-3),
    ]
    assert len(c_combos) == 8

    # ===================================================================
    # BLOCK D (12): Router structure + group feature exploration
    # Sub-D1 (4): New routing structures (factored / group_gated_bias)
    # Sub-D2 (3): Feature group ablation (limit to sub-groups of 4 families)
    # Sub-D3 (3): Lambda exploration on new mechanisms
    # Sub-D4 (2): Heavy aux/reg for group routing
    # ===================================================================

    # --- Sub-D1 (4): New routing structure variants ---
    d_combos = [
        # Factored router: feature→group + hidden→intra; source=both
        make_combo("PD01", combo_family="D", combo_role="factored_router",
                   desc="factored_router_source_both",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_router_source=_all_stage_map("both")),

        # Factored router: pure feature-driven (source=feature)
        # Very different: feature completely controls both group AND intra routing
        make_combo("PD02", combo_family="D", combo_role="factored_router",
                   desc="factored_router_pure_feature",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_router_source=_all_stage_map("feature")),

        # Group-scoped gated bias: per-group feature conditions its own experts
        make_combo("PD03", combo_family="D", combo_role="group_injection",
                   desc="group_gated_bias_injection",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_injection=_all_stage_map("group_gated_bias"),
                   stage_router_source=_all_stage_map("both")),

        # Factored + group_gated_bias: both routing AND injection are group-hierarchical
        make_combo("PD04", combo_family="D", combo_role="factored_plus_group",
                   desc="factored_router_plus_group_gated_bias",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_feature_injection=_all_stage_map("group_gated_bias"),
                   stage_router_source=_all_stage_map("both")),

        # --- Sub-D2 (3): Feature group ablation ---
        # Only Tempo+Focus (drop Memory+Exposure): time/attention signals only
        make_combo("PD05", combo_family="D", combo_role="grp_ablation",
                   desc="grp_tempo_focus_only",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_family_mask=_mask_all(["Tempo", "Focus"])),

        # Only Memory+Exposure (drop Tempo+Focus): repetition/popularity signals only
        make_combo("PD06", combo_family="D", combo_role="grp_ablation",
                   desc="grp_memory_exposure_only",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_family_mask=_mask_all(["Memory", "Exposure"])),

        # Extreme ablation: single group Focus only (what happens when routing is 1-group?)
        make_combo("PD07", combo_family="D", combo_role="grp_ablation",
                   desc="grp_focus_only_extreme",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_feature_family_mask=_mask_all(["Focus"])),

        # --- Sub-D3 (3): Lambda exploration on new mechanisms ---
        # Factored router + group_prior alignment (does alignment help factored routing?)
        make_combo("PD08", combo_family="D", combo_role="lambda_sweep",
                   desc="factored_plus_group_prior_align_5e-4",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_router_source=_all_stage_map("both"),
                   group_prior_align_lambda=5e-4),

        # feature_group_bias_lambda: prior injection into standard learned router
        make_combo("PD09", combo_family="D", combo_role="lambda_sweep",
                   desc="feature_group_bias_0p3",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   feature_group_bias_lambda=0.3),

        # group_bias strong + alignment: force prior + KL alignment together
        make_combo("PD10", combo_family="D", combo_role="lambda_sweep",
                   desc="group_bias_0p5_align_1e-3",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   feature_group_bias_lambda=0.5,
                   group_prior_align_lambda=1e-3),

        # --- Sub-D4 (2): Heavy aux/reg for group routing stability ---
        # Factored router + heavy balance on group logits
        # (prevent factored router group collapse with strong factored_group_balance)
        make_combo("PD11", combo_family="D", combo_role="group_reg",
                   desc="factored_strong_group_balance",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_router_source=_all_stage_map("both"),
                   factored_group_balance_lambda=0.01,
                   balance_loss_lambda=0.006),

        # Factored router + group coverage reward (diversity across groups)
        make_combo("PD12", combo_family="D", combo_role="group_reg",
                   desc="factored_group_coverage_0p5",
                   layer_layout=["macro", "mid", "micro"],
                   train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"],
                   lr_min=lr_base[0], lr_max=lr_base[1],
                   stage_router_type=_all_stage_map("factored"),
                   stage_router_source=_all_stage_map("both"),
                   group_coverage_lambda=0.5,
                   factored_group_balance_lambda=0.005),
    ]
    assert len(d_combos) == 12

    combos = a_combos + b_combos + c_combos + d_combos
    assert len(combos) == 40, f"Expected 40 combos, got {len(combos)}"

    for idx, row in enumerate(combos):
        row["seed_offset"] = idx
        row["dataset"] = dataset
        row["feature_mode"] = "full_v3"
        row["has_diag"] = True
        rp = estimated_runtime_profile(row)
        row["runtime_bucket"] = rp["runtime_bucket"]
        row["estimated_50ep_single_eval_min"] = rp["estimated_50ep_single_eval_min"]
        row["launch_max_evals"] = rp["recommended_max_evals"]
    return combos


def build_command(combo: dict, gpu_id: str, args) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    combo_max_evals = int(combo.get("launch_max_evals", args.max_evals))
    if args.max_evals > 0:
        combo_max_evals = min(combo_max_evals, args.max_evals) if args.use_recommended_budget else args.max_evals
    eval_logging_timing = str(getattr(args, "eval_logging_timing", "final_only") or "final_only").strip().lower()
    if eval_logging_timing not in {"final_only", "per_eval"}:
        eval_logging_timing = "final_only"

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name", "config",
        "--max-evals", str(combo_max_evals),
        "--tune-epochs", str(args.tune_epochs),
        "--tune-patience", str(args.tune_patience),
        "--seed", str(args.seed_base + int(combo["seed_offset"])),
        "--run-group", TRACK,
        "--run-axis", AXIS,
        "--run-phase", combo["combo_id"],
        "model=featured_moe_n3_tune",
        f"dataset={combo['dataset']}",
        "eval_mode=session",
        f"feature_mode={combo['feature_mode']}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
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
        f"moe_top_k={combo['moe_top_k']}",
        "moe_top_k_policy=auto",
        "moe_top_k_ratio=0.5",
        "macro_session_pooling=mean",
        f"balance_loss_lambda={combo['balance_loss_lambda']}",
        f"z_loss_lambda={combo['z_loss_lambda']}",
        f"gate_entropy_lambda={combo['gate_entropy_lambda']}",
        f"gate_entropy_until={combo['gate_entropy_until']}",
        f"rule_agreement_lambda={combo['rule_agreement_lambda']}",
        f"group_coverage_lambda={combo['group_coverage_lambda']}",
        f"group_prior_align_lambda={combo['group_prior_align_lambda']}",
        f"feature_group_bias_lambda={combo['feature_group_bias_lambda']}",
        f"feature_group_prior_temperature={combo['feature_group_prior_temperature']}",
        f"factored_group_balance_lambda={combo['factored_group_balance_lambda']}",
        f"lr_scheduler_type={combo['lr_scheduler_type']}",
        f"lr_scheduler_warmup_ratio={combo['lr_scheduler_warmup_ratio']}",
        f"lr_scheduler_min_lr_ratio={combo['lr_scheduler_min_lr_ratio']}",
        f"lr_scheduler_plateau_factor={combo['lr_scheduler_plateau_factor']}",
        f"lr_scheduler_plateau_patience={combo['lr_scheduler_plateau_patience']}",
        f"++combo_family={hydra_literal(combo['combo_family'])}",
        f"++combo_role={hydra_literal(combo['combo_role'])}",
        f"++combo_desc={hydra_literal(combo['desc'])}",
        f"++search.learning_rate={hydra_literal([combo['lr_min'], combo['lr_max']])}",
        f"++search.weight_decay={hydra_literal(combo['search_weight_decay'])}",
        f"++search.hidden_dropout_prob={hydra_literal(combo['search_hidden_dropout_prob'])}",
        f"++search.lr_scheduler_type={hydra_literal([combo['lr_scheduler_type']])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
    ]
    return cmd


def log_path(combo: dict, dataset: str) -> Path:
    dataset_tag = dataset.replace("/", "_")
    model_tag = "FMoEN3"
    root = LOG_ROOT / AXIS / PHASE / dataset_tag / model_tag
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


def main() -> None:
    parser = argparse.ArgumentParser(description="FMoE_N3 phase2 40-combo launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=0,
                        help="0=use recommended per combo, >0=override all")
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=9900)
    parser.add_argument("--use-recommended-budget", action="store_true",
                        help="Cap max-evals to recommended_max_evals per combo")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--only", default="",
                        help="Comma-separated list of combo_ids to run (subset)")
    parser.add_argument("--family", default="",
                        help="Run only this family letter(s) e.g. 'A' or 'D'")
    parser.add_argument("--eval-logging-timing", default="final_only",
                        choices=["final_only", "per_eval"])
    parser.add_argument("--feature-ablation-logging", action="store_true")
    args = parser.parse_args()

    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    combos = build_combos(args.dataset)

    if args.only:
        allowed = {cid.strip() for cid in args.only.split(",") if cid.strip()}
        combos = [c for c in combos if c["combo_id"] in allowed]
    if args.family:
        allowed_fam = {f.strip().upper() for f in args.family.split(",") if f.strip()}
        combos = [c for c in combos if str(c["combo_family"]).upper() in allowed_fam]
    if not combos:
        raise SystemExit("No combos selected after filtering.")

    cost_key = "estimated_50ep_single_eval_min"
    bins = plan_gpu_bins(combos, gpus, cost_key=cost_key)

    manifest = []
    all_assigned = []
    for gpu_id, gpu_combos in bins.items():
        for order, combo in enumerate(gpu_combos, start=1):
            combo["assigned_order"] = order
            cmd = build_command(combo, gpu_id, args)
            log_p = log_path(combo, args.dataset)
            write_log_preamble(log_p, combo, gpu_id, args, cmd)
            manifest.append({
                "combo_id": combo["combo_id"],
                "gpu": gpu_id,
                "family": combo["combo_family"],
                "desc": combo["desc"],
                "max_evals": int(combo.get("launch_max_evals", 20)),
                "bucket": combo.get("runtime_bucket", "medium"),
                "log": str(log_p),
                "cmd": cmd,
            })
            all_assigned.append((gpu_id, combo, cmd, log_p))

    if args.manifest_out:
        Path(args.manifest_out).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[manifest] Written {len(manifest)} entries → {args.manifest_out}")

    if args.dry_run:
        print(f"\n[dry-run] {len(all_assigned)} combos across GPUs {gpus}")
        for gpu_id, combo, cmd, log_p in all_assigned:
            print(f"  {combo['combo_id']:10s} [{combo['combo_family']}] gpu={gpu_id}  {combo['desc']}")
        return

    # Launch with GPU round-robin + concurrency guard
    active: dict[str, subprocess.Popen] = {}
    queue = deque(all_assigned)
    summary_timer = time.time()

    while queue or active:
        for gpu_id in list(active):
            proc = active[gpu_id]
            if proc.poll() is not None:
                del active[gpu_id]
        for gpu_id in gpus:
            if gpu_id not in active and queue:
                gpu_id_match = None
                for i, (g, combo, cmd, log_p) in enumerate(queue):
                    if g == gpu_id:
                        gpu_id_match = i
                        break
                if gpu_id_match is not None:
                    _, combo, cmd, log_p = queue[gpu_id_match]
                    del queue[gpu_id_match]
                    print(f"[launch] {combo['combo_id']} on GPU {gpu_id}: {combo['desc']}")
                    with open(log_p, "a") as lf:
                        proc = subprocess.Popen(
                            cmd,
                            cwd=str(EXP_DIR),
                            stdout=lf,
                            stderr=lf,
                        )
                    active[gpu_id] = proc
        if time.time() - summary_timer > SUMMARY_REFRESH_SEC:
            try:
                build_fmoe_n3_summaries(ARTIFACT_ROOT)
            except Exception:
                pass
            summary_timer = time.time()
        if active or queue:
            time.sleep(5)

    try:
        build_fmoe_n3_summaries(ARTIFACT_ROOT)
        build_fmoe_n3_axis_summary(ARTIFACT_ROOT, axis=AXIS)
    except Exception:
        pass
    print(f"\n[done] Phase 2 complete — {len(all_assigned)} combos launched.")


if __name__ == "__main__":
    main()
