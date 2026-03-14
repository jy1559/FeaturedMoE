#!/usr/bin/env python3
"""Launch FMoE_N3 phase1 28-combo upgrade track."""

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
AXIS = "phase1_upgrade_v1"
PHASE = "P1"
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
    if combo.get("lr_scheduler_type") == "warmup_cosine":
        return "medium"
    return "medium"


def recommended_max_evals(combo: dict) -> int:
    if int(combo.get("MAX_ITEM_LIST_LENGTH", 20)) >= 50:
        return 10
    if int(combo.get("embedding_size", 128)) >= 256:
        return 10
    if combo.get("combo_family") in {"G3", "G4"}:
        return 20
    return 20


def estimated_runtime_profile(combo: dict) -> dict:
    bucket = runtime_bucket(combo)
    trial_min = 1.4 if bucket == "heavy" else 1.0
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
        chunk = ordered[start : start + window]
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
        "stage_router_type": _all_stage_map("standard"),
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
        "eval_batch_size": "eval_batch_size",
        "embedding_size": "embedding_size",
        "d_ff": "d_ff",
        "num_heads": "num_heads",
        "d_feat_emb": "d_feat_emb",
        "d_expert_hidden": "d_expert_hidden",
        "d_router_hidden": "d_router_hidden",
        "expert_scale": "expert_scale",
        "layer_layout": "layer_layout",
        "stage_feature_encoder_mode": "stage_feature_encoder_mode",
        "stage_router_source": "stage_router_source",
        "stage_feature_injection": "stage_feature_injection",
        "stage_router_granularity": "stage_router_granularity",
        "macro_history_window": "macro_history_window",
        "balance_loss_lambda": "balance_loss_lambda",
        "z_loss_lambda": "z_loss_lambda",
        "gate_entropy_lambda": "gate_entropy_lambda",
        "gate_entropy_until": "gate_entropy_until",
        "rule_agreement_lambda": "rule_agreement_lambda",
        "group_coverage_lambda": "group_coverage_lambda",
        "group_prior_align_lambda": "group_prior_align_lambda",
        "feature_group_bias_lambda": "feature_group_bias_lambda",
        "feature_group_prior_temperature": "feature_group_prior_temperature",
        "stage_router_type": "stage_router_type",
        "lr_scheduler_type": "lr_scheduler_type",
        "lr_scheduler_warmup_ratio": "lr_scheduler_warmup_ratio",
        "lr_scheduler_min_lr_ratio": "lr_scheduler_min_lr_ratio",
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
        "stage_router_granularity",
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
        "stage_router_type",
        "gate_entropy_until",
        "lr_scheduler_type",
        "lr_scheduler_warmup_ratio",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_plateau_factor",
        "lr_scheduler_plateau_patience",
    ]
    budget_mode = "recommended" if args.use_recommended_budget else "manual"
    lines = [
        "[PHASE1_COMBO_HEADER]",
        f"combo_id={combo['combo_id']} family={combo['combo_family']} role={combo['combo_role']} desc={combo['desc']}",
        f"dataset={combo['dataset']} gpu={gpu_id} launch_group={combo.get('launch_group', 0)} launch_order={combo.get('assigned_order', 0)}",
        f"budget_mode={budget_mode} launch_max_evals={combo['launch_max_evals']} cli_max_evals={args.max_evals}",
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
    lines.extend([
        "",
        "[FIXED_SETTINGS]",
    ])
    lines.extend(f"- {field}={_format_setting(combo[field])}" for field in fixed_fields)
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
    lr_scheduler_type: str = "warmup_cosine",
    lr_scheduler_warmup_ratio: float = 0.1,
    lr_scheduler_min_lr_ratio: float = 0.1,
):
    return {
        "combo_id": combo_id,
        "combo_family": combo_family,
        "combo_role": combo_role,
        "desc": desc,
        "baseline_recipe": "P1",
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
        "stage_router_granularity": stage_router_granularity or _default_granularity(),
        "stage_router_type": stage_router_type or _all_stage_map("standard"),
        "macro_history_window": 10 if max_len >= 30 else 5,
        "stage_feature_family_mask": {},
        "moe_top_k": 0,
        "expert_scale": expert_scale,
        "dense_hidden_scale": 1.0,
        "search_hidden_dropout_prob": list(DROPOUT_CHOICES),
        "search_weight_decay": list(WD_CHOICES),
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
        "lr_scheduler_type": lr_scheduler_type,
        "lr_scheduler_warmup_ratio": lr_scheduler_warmup_ratio,
        "lr_scheduler_min_lr_ratio": lr_scheduler_min_lr_ratio,
        "lr_scheduler_plateau_factor": 0.5,
        "lr_scheduler_plateau_patience": 3,
        "has_diag": True,
    }


def build_combos(dataset: str) -> list[dict]:
    prof = dataset_profile(dataset)
    lr_base = DEFAULT_LR_BASE
    lr_aggr = DEFAULT_LR_AGGR

    combos = [
        # G1 Core anchors
        make_combo("A01", combo_family="G1", combo_role="core", desc="c70_anchor", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], expert_scale=3),
        make_combo("A02", combo_family="G1", combo_role="core", desc="x62_len30_anchor", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=30, expert_scale=3),
        make_combo("A03", combo_family="G1", combo_role="core", desc="t50_token_anchor", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_router_granularity={"macro": "session", "mid": "token", "micro": "token"}),
        make_combo("A04", combo_family="G1", combo_role="core", desc="e41_macro_complex", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"}),
        make_combo("A05", combo_family="G1", combo_role="core", desc="both_plus_gated_bias", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_injection=_all_stage_map("gated_bias")),
        make_combo("A06", combo_family="G1", combo_role="core", desc="core_with_layer_prefix", layer_layout=["layer", "macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),

        # G2 Length expansion
        make_combo("N30A", combo_family="G2", combo_role="len30", desc="len30_c70", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=30, expert_scale=3),
        make_combo("N30B", combo_family="G2", combo_role="len30", desc="len30_e41", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=30, stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"}),
        make_combo("N30C", combo_family="G2", combo_role="len30", desc="len30_t50", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=30, stage_router_granularity={"macro": "session", "mid": "token", "micro": "token"}),
        make_combo("N30D", combo_family="G2", combo_role="len30", desc="len30_layout_repeat", layer_layout=["layer", "macro", "macro_ffn", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=30),
        make_combo("N50A", combo_family="G2", combo_role="len50", desc="len50_probe", layer_layout=["macro", "mid", "micro"], train_bs=prof["len50_train_bs"], eval_bs=prof["len50_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], max_len=50, expert_scale=3),

        # G3 Layout topology
        make_combo("L01", combo_family="G3", combo_role="layout", desc="layer_macro_mid_micro", layer_layout=["layer", "macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L02", combo_family="G3", combo_role="layout", desc="layer_macroffn_mid_micro", layer_layout=["layer", "macro_ffn", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L03", combo_family="G3", combo_role="layout", desc="layer_macro_midffn_micro", layer_layout=["layer", "macro", "mid_ffn", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L04", combo_family="G3", combo_role="layout", desc="ffn_only_between_stages", layer_layout=["layer", "macro_ffn", "mid_ffn", "micro_ffn"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L05", combo_family="G3", combo_role="layout", desc="macro_repeated", layer_layout=["layer", "macro", "macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L06", combo_family="G3", combo_role="layout", desc="macro_micro_swapped", layer_layout=["layer", "micro", "mid", "macro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L07", combo_family="G3", combo_role="layout", desc="macro_micro_mid", layer_layout=["layer", "macro", "micro", "mid"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),
        make_combo("L08", combo_family="G3", combo_role="layout", desc="deeper_layer_prefix", layer_layout=["layer", "layer", "macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1]),

        # G4 Feature encoding/injection
        make_combo("F01", combo_family="G4", combo_role="feature", desc="all_linear", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_encoder_mode=_all_stage_map("linear")),
        make_combo("F02", combo_family="G4", combo_role="feature", desc="macro_complex", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_encoder_mode={"macro": "complex", "mid": "linear", "micro": "linear"}),
        make_combo("F03", combo_family="G4", combo_role="feature", desc="all_complex", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_encoder_mode=_all_stage_map("complex")),
        make_combo("F04", combo_family="G4", combo_role="feature", desc="macro_gated_bias_only", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], stage_feature_injection={"macro": "gated_bias", "mid": "none", "micro": "none"}),

        # G5 Scheduler/LR envelope
        make_combo("S01", combo_family="G5", combo_role="sched", desc="warmup_cosine_base", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], lr_scheduler_type="warmup_cosine", lr_scheduler_warmup_ratio=0.1),
        make_combo("S02", combo_family="G5", combo_role="sched", desc="warmup_cosine_aggressive", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_aggr[0], lr_max=lr_aggr[1], lr_scheduler_type="warmup_cosine", lr_scheduler_warmup_ratio=0.15),
        make_combo("S03", combo_family="G5", combo_role="sched", desc="plateau_control", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], lr_scheduler_type="plateau", lr_scheduler_warmup_ratio=0.0),

        # G6 Aux + embedding spread
        make_combo("X64", combo_family="G6", combo_role="embed", desc="embed64_aux_light", layer_layout=["macro", "mid", "micro"], train_bs=prof["c4_train_bs"], eval_bs=prof["c4_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], embedding_size=64, d_feat_emb=12, d_router_hidden=48, d_expert_hidden=96, balance_loss_lambda=0.002, z_loss_lambda=1e-4),
        make_combo("X65", combo_family="G6", combo_role="embed", desc="embed256_aux_strong", layer_layout=["macro", "mid", "micro"], train_bs=prof["heavy_train_bs"], eval_bs=prof["heavy_eval_bs"], lr_min=lr_base[0], lr_max=lr_base[1], embedding_size=256, d_feat_emb=24, d_router_hidden=96, d_expert_hidden=192, balance_loss_lambda=0.006, z_loss_lambda=3e-4),
    ]

    assert len(combos) == 28
    for idx, row in enumerate(combos):
        row["seed_offset"] = idx
        row["dataset"] = dataset
        row["feature_mode"] = "full_v3"
        row["has_diag"] = True
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=9800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--only", default="")
    parser.add_argument("--use-recommended-budget", action="store_true")
    parser.add_argument("--eval-logging-timing", default="final_only", choices=["final_only", "per_eval"])
    parser.add_argument("--feature-ablation-logging", action="store_true")
    args = parser.parse_args()

    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided.")

    combos = build_combos(args.dataset)
    if args.only.strip():
        wanted = {tok.strip().upper() for tok in args.only.split(",") if tok.strip()}
        combos = [c for c in combos if c["combo_id"].upper() in wanted]
        if not combos:
            raise SystemExit(f"No combos matched --only={args.only}")

    for combo in combos:
        combo.update(estimated_runtime_profile(combo))
        combo["launch_max_evals"] = int(combo["recommended_max_evals"]) if args.use_recommended_budget else int(args.max_evals)
        combo["estimated_total_budget_min"] = float(combo["estimated_50ep_single_eval_min"]) * float(combo["launch_max_evals"])

    planned_bins = plan_gpu_bins(combos, gpus, cost_key="estimated_total_budget_min")
    for gpu_id in gpus:
        bucket = planned_bins[gpu_id]
        for order_idx, combo in enumerate(bucket, start=1):
            combo["assigned_gpu"] = gpu_id
            combo["assigned_order"] = order_idx

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
    manifest_path = Path(args.manifest_out) if args.manifest_out else INVENTORY_ROOT / f"phase1_28_{int(time.time())}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Manifest] {manifest_path}")

    if args.dry_run:
        for combo in combos:
            cmd = build_command(combo, combo["assigned_gpu"], args)
            print(f"[DryRun] group={combo.get('launch_group', 0):02d} {combo['combo_id']} gpu={combo['assigned_gpu']} :: {' '.join(cmd)}")
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
            if reason and phase_paths:
                print(f"[Summary] refreshed ({reason}) -> {axis_path}")
                print("[Artifacts] "
                      f"special={artifact_paths['special_summary']} "
                      f"feature_ablation={artifact_paths['feature_ablation_summary']} "
                      f"diag={artifact_paths['diag_summary']}")
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
                    try:
                        current_size = int(proc_info["log_file"].stat().st_size)
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
            write_log_preamble(log_file, combo, gpu_id, args, cmd)
            env["LOG_FILE"] = str(log_file)
            env["RUN_LOGS_DIR"] = str((ARTIFACT_ROOT / "logs").resolve())
            print(f"[Launch] {combo['combo_id']} gpu={gpu_id} budget_evals={combo['launch_max_evals']} est50epx1={combo['estimated_50ep_single_eval_min']:.1f}m")
            proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), env=env)
            active[gpu_id] = {"combo": combo, "proc": proc, "log_file": log_file, "last_seen_size": 0}

        if saw_completion:
            refresh_phase_summaries(force=True, reason="combo_done")
        elif saw_log_growth:
            refresh_phase_summaries(force=False, reason="log_progress")
        time.sleep(3)

    refresh_phase_summaries(force=True, reason="finished")
    print("[Finished] all combos completed")


if __name__ == "__main__":
    main()
