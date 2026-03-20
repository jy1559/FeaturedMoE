#!/usr/bin/env python3
"""LFM all-in-one launcher for FMoE_N3 (phase0~phase2).

Design goals:
- Anchor-based fixed settings (wd/dropout/attn_dropout fixed per anchor)
- LR-only hyperopt tuning (search.learning_rate only)
- Suite-based axis expansion with optional router x feature_embed coupling
- Phase7-style logging path and rich run metadata in logs + manifest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"
INVENTORY_ROOT = ARTIFACT_ROOT / "inventory" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase_lfm_allinone_v1"


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    embedding_size: int
    d_ff: int
    d_router_hidden: int
    d_expert_hidden: int
    expert_scale: int
    weight_decay: float
    hidden_dropout: float
    attn_dropout: float
    lr_min: float
    lr_max: float


def _all_stage_map(value: str) -> dict[str, str]:
    return {"macro": value, "mid": value, "micro": value}


def _all_stage_float(value: float) -> dict[str, float]:
    return {"macro": float(value), "mid": float(value), "micro": float(value)}


def hydra_literal(value: Any) -> str:
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
        return "{" + ",".join(f"{k}:{hydra_literal(v)}" for k, v in value.items()) + "}"
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


def dataset_profile(dataset: str, max_len: int) -> dict[str, int]:
    key = dataset.lower()
    if key == "lastfm0.03":
        if max_len >= 50:
            return {"train_bs": 1024, "eval_bs": 2048}
        if max_len >= 30:
            return {"train_bs": 1536, "eval_bs": 3072}
        return {"train_bs": 2048, "eval_bs": 4096}
    if max_len >= 50:
        return {"train_bs": 1536, "eval_bs": 3072}
    if max_len >= 30:
        return {"train_bs": 2048, "eval_bs": 4096}
    return {"train_bs": 4096, "eval_bs": 4096}


def anchors() -> dict[str, Anchor]:
    return {
        "AN_S": Anchor(
            anchor_id="AN_S",
            embedding_size=96,
            d_ff=192,
            d_router_hidden=64,
            d_expert_hidden=128,
            expert_scale=3,
            weight_decay=1e-6,
            hidden_dropout=0.10,
            attn_dropout=0.10,
            lr_min=1.0e-4,
            lr_max=4.0e-3,
        ),
        "AN_M": Anchor(
            anchor_id="AN_M",
            embedding_size=128,
            d_ff=256,
            d_router_hidden=96,
            d_expert_hidden=192,
            expert_scale=4,
            weight_decay=1e-6,
            hidden_dropout=0.15,
            attn_dropout=0.10,
            lr_min=8.0e-5,
            lr_max=6.0e-3,
        ),
        "AN_L": Anchor(
            anchor_id="AN_L",
            embedding_size=192,
            d_ff=384,
            d_router_hidden=128,
            d_expert_hidden=256,
            expert_scale=5,
            weight_decay=1e-5,
            hidden_dropout=0.20,
            attn_dropout=0.15,
            lr_min=5.0e-5,
            lr_max=4.0e-3,
        ),
    }


def suite_defs() -> dict[str, list[dict[str, Any]]]:
    return {
        "layout_suite": [
            {"variant_id": "LAY_BASE", "overrides": {"layer_layout": ["macro", "mid", "micro"]}},
            {"variant_id": "LAY_SAS", "overrides": {"layer_layout": ["layer", "layer", "layer"]}},
            {"variant_id": "LAY_PFX2", "overrides": {"layer_layout": ["layer", "layer", "macro", "mid", "micro"]}},
            {"variant_id": "LAY_FFN", "overrides": {"layer_layout": ["layer", "macro_ffn", "mid_ffn", "micro_ffn"]}},
        ],
        "router_suite": [
            {
                "variant_id": "ROUT_STD",
                "router_variant": "standard",
                "overrides": {
                    "stage_router_type": _all_stage_map("standard"),
                    "stage_router_source": _all_stage_map("both"),
                    "stage_feature_injection": _all_stage_map("gated_bias"),
                    "topk_scope_mode": "global_flat",
                    "moe_top_k": 0,
                },
            },
            {
                "variant_id": "ROUT_FAC",
                "router_variant": "factored",
                "overrides": {
                    "stage_router_type": _all_stage_map("factored"),
                    "stage_router_source": _all_stage_map("both"),
                    "stage_feature_injection": _all_stage_map("group_gated_bias"),
                    "stage_factored_group_router_source": _all_stage_map("feature"),
                    "stage_factored_group_logit_scale": _all_stage_float(1.0),
                    "stage_factored_intra_logit_scale": _all_stage_float(1.0),
                    "stage_factored_combine_mode": _all_stage_map("add"),
                    "topk_scope_mode": "group_dense",
                    "moe_top_k": 0,
                    "group_prior_align_lambda": 5e-4,
                    "factored_group_balance_lambda": 1e-3,
                },
            },
            {
                "variant_id": "ROUT_FACH",
                "router_variant": "factored_heavy",
                "overrides": {
                    "stage_router_type": _all_stage_map("factored"),
                    "stage_router_source": _all_stage_map("feature"),
                    "stage_feature_injection": _all_stage_map("group_gated_bias"),
                    "stage_factored_group_router_source": _all_stage_map("both"),
                    "stage_factored_group_logit_scale": _all_stage_float(1.6),
                    "stage_factored_intra_logit_scale": _all_stage_float(1.0),
                    "stage_factored_combine_mode": _all_stage_map("add"),
                    "topk_scope_mode": "group_dense",
                    "moe_top_k": 0,
                    "group_prior_align_lambda": 5e-4,
                    "factored_group_balance_lambda": 1e-3,
                },
            },
        ],
        "feature_embed_suite": [
            {"variant_id": "FEMB_12", "d_feat_emb": 12, "overrides": {"d_feat_emb": 12}},
            {"variant_id": "FEMB_16", "d_feat_emb": 16, "overrides": {"d_feat_emb": 16}},
            {"variant_id": "FEMB_24", "d_feat_emb": 24, "overrides": {"d_feat_emb": 24}},
            {"variant_id": "FEMB_32", "d_feat_emb": 32, "overrides": {"d_feat_emb": 32}},
        ],
        "feature_family_mask_suite": [
            {"variant_id": "FMASK_TF", "overrides": {"stage_feature_family_mask": _all_stage_map(["Tempo", "Focus"]) }},
            {"variant_id": "FMASK_ME", "overrides": {"stage_feature_family_mask": _all_stage_map(["Memory", "Exposure"]) }},
            {"variant_id": "FMASK_F", "overrides": {"stage_feature_family_mask": _all_stage_map(["Focus"]) }},
        ],
        "topk_suite": [
            {"variant_id": "TOPK_GLOB0", "overrides": {"topk_scope_mode": "global_flat", "moe_top_k": 0, "moe_top_k_ratio": 0.5}},
            {"variant_id": "TOPK_GLOB3", "overrides": {"topk_scope_mode": "global_flat", "moe_top_k": 3, "moe_top_k_ratio": 0.5}},
            {"variant_id": "TOPK_GRP1", "overrides": {"topk_scope_mode": "group_top1_pergroup", "moe_top_k": 1, "moe_top_k_ratio": 0.34}},
            {"variant_id": "TOPK_GRP2", "overrides": {"topk_scope_mode": "group_top2_pergroup", "moe_top_k": 2, "moe_top_k_ratio": 0.5}},
        ],
        "expert_scale_suite": [
            {"variant_id": "ES_3", "overrides": {"expert_scale": 3}},
            {"variant_id": "ES_4", "overrides": {"expert_scale": 4}},
            {"variant_id": "ES_5", "overrides": {"expert_scale": 5}},
            {"variant_id": "ES_6", "overrides": {"expert_scale": 6}},
        ],
        "seq_len_suite": [
            {"variant_id": "LEN20", "overrides": {"MAX_ITEM_LIST_LENGTH": 20}},
            {"variant_id": "LEN30", "overrides": {"MAX_ITEM_LIST_LENGTH": 30}},
            {"variant_id": "LEN50", "overrides": {"MAX_ITEM_LIST_LENGTH": 50}},
        ],
        "aux_balance_suite": [
            {"variant_id": "AUXB_1", "overrides": {"balance_loss_lambda": 0.002, "z_loss_lambda": 1e-4}},
            {"variant_id": "AUXB_2", "overrides": {"balance_loss_lambda": 0.004, "z_loss_lambda": 1e-4}},
            {"variant_id": "AUXB_3", "overrides": {"balance_loss_lambda": 0.006, "z_loss_lambda": 3e-4}},
        ],
        "aux_spec_suite": [
            {"variant_id": "AUXS_SM", "overrides": {"route_smoothness_lambda": 0.04}},
            {"variant_id": "AUXS_CS", "overrides": {"route_consistency_lambda": 0.03}},
            {"variant_id": "AUXS_SH", "overrides": {"route_sharpness_lambda": 0.01, "route_monopoly_lambda": 0.04, "route_monopoly_tau": 0.25}},
        ],
        "residual_suite": [
            {"variant_id": "RES_BASE", "overrides": {"stage_residual_mode": _all_stage_map("base")}},
            {"variant_id": "RES_WARM", "overrides": {"stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"), "fmoe_schedule_enable": True, "alpha_warmup_until": 0.15, "alpha_warmup_start": 0.0, "alpha_warmup_end": 1.0}},
        ],
    }


def base_template(dataset: str, anchor: Anchor) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "feature_mode": "full_v3",
        "MAX_ITEM_LIST_LENGTH": 20,
        "embedding_size": anchor.embedding_size,
        "d_ff": anchor.d_ff,
        "num_heads": 4,
        "d_feat_emb": 16,
        "d_router_hidden": anchor.d_router_hidden,
        "d_expert_hidden": anchor.d_expert_hidden,
        "expert_scale": anchor.expert_scale,
        "hidden_dropout_prob": anchor.hidden_dropout,
        "attn_dropout_prob": anchor.attn_dropout,
        "weight_decay": anchor.weight_decay,
        "lr_min": anchor.lr_min,
        "lr_max": anchor.lr_max,
        "layer_layout": ["macro", "mid", "micro"],
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_type": _all_stage_map("standard"),
        "stage_router_source": _all_stage_map("both"),
        "stage_feature_encoder_mode": _all_stage_map("linear"),
        "stage_feature_injection": _all_stage_map("gated_bias"),
        "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
        "stage_feature_family_mask": {},
        "stage_residual_mode": _all_stage_map("base"),
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "moe_top_k_ratio": 0.5,
        "moe_top_k_policy": "auto",
        "macro_history_window": 5,
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
        "lr_scheduler_type": "warmup_cosine",
        "lr_scheduler_warmup_ratio": 0.1,
        "lr_scheduler_min_lr_ratio": 0.1,
        "lr_scheduler_plateau_factor": 0.5,
        "lr_scheduler_plateau_patience": 3,
    }


def merge_overrides(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        out[k] = v
    return out


def _phase_name(idx: int) -> str:
    return f"P{idx}"


def _suites_for_phase(phase_idx: int) -> list[str]:
    if phase_idx == 0:
        return [
            "layout_suite",
            "router_suite",
            "feature_embed_suite",
            "topk_suite",
            "expert_scale_suite",
            "seq_len_suite",
            "aux_balance_suite",
            "aux_spec_suite",
            "residual_suite",
        ]
    if phase_idx == 1:
        return [
            "router_suite",
            "feature_embed_suite",
            "topk_suite",
            "expert_scale_suite",
            "aux_balance_suite",
            "aux_spec_suite",
        ]
    return [
        "router_suite",
        "feature_embed_suite",
        "topk_suite",
        "seq_len_suite",
        "aux_balance_suite",
    ]


def _max_evals_for_phase(args: argparse.Namespace, phase_idx: int) -> int:
    if phase_idx == 0:
        return int(args.p0_max_evals)
    if phase_idx == 1:
        return int(args.p1_max_evals)
    return int(args.p2_max_evals)


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    anchor_map = anchors()
    selected_anchors = [tok.strip() for tok in args.anchors.split(",") if tok.strip()]
    if not selected_anchors:
        selected_anchors = list(anchor_map.keys())
    for a in selected_anchors:
        if a not in anchor_map:
            raise SystemExit(f"Unknown anchor: {a}")

    defs = suite_defs()
    requested_suites = {tok.strip() for tok in args.suites.split(",") if tok.strip()}

    rows: list[dict[str, Any]] = []
    seed_cursor = 0

    for phase_idx in range(args.from_phase, args.to_phase + 1):
        phase_name = _phase_name(phase_idx)
        max_evals = _max_evals_for_phase(args, phase_idx)
        phase_suites = _suites_for_phase(phase_idx)
        if requested_suites:
            phase_suites = [s for s in phase_suites if s in requested_suites]

        for anchor_id in selected_anchors:
            anchor = anchor_map[anchor_id]
            base = base_template(args.dataset, anchor)
            base_prof = dataset_profile(args.dataset, int(base.get("MAX_ITEM_LIST_LENGTH", 20)))
            base["train_batch_size"] = int(base_prof["train_bs"])
            base["eval_batch_size"] = int(base_prof["eval_bs"])
            base["macro_history_window"] = 10 if int(base.get("MAX_ITEM_LIST_LENGTH", 20)) >= 30 else 5

            # Anchor baseline per phase
            seed_cursor += 1
            rows.append(
                {
                    "run_phase": f"{phase_name}_{anchor_id}_BASE",
                    "phase": phase_name,
                    "phase_idx": phase_idx,
                    "category": "anchor_base",
                    "suite_id": "anchor_base",
                    "variant_id": "base",
                    "anchor_id": anchor_id,
                    "router_variant": "standard",
                    "d_feat_emb": int(base["d_feat_emb"]),
                    "lr_range": [float(base["lr_min"]), float(base["lr_max"])],
                    "axis_tags": ["anchor_base"],
                    "max_evals": max_evals,
                    "seed_offset": seed_cursor,
                    "cfg": base,
                }
            )

            for suite_name in phase_suites:
                variants = defs[suite_name]
                for var in variants:
                    seed_cursor += 1
                    cfg = merge_overrides(base, var.get("overrides", {}))
                    max_len = int(cfg.get("MAX_ITEM_LIST_LENGTH", 20))
                    prof = dataset_profile(args.dataset, max_len)
                    cfg["train_batch_size"] = int(prof["train_bs"])
                    cfg["eval_batch_size"] = int(prof["eval_bs"])
                    cfg["macro_history_window"] = 10 if max_len >= 30 else 5

                    row = {
                        "run_phase": f"{phase_name}_{anchor_id}_{sanitize_slug(suite_name)}_{sanitize_slug(var['variant_id'])}",
                        "phase": phase_name,
                        "phase_idx": phase_idx,
                        "category": suite_name,
                        "suite_id": suite_name,
                        "variant_id": var["variant_id"],
                        "anchor_id": anchor_id,
                        "router_variant": var.get("router_variant", "inherit"),
                        "d_feat_emb": int(cfg.get("d_feat_emb", 16)),
                        "lr_range": [float(cfg["lr_min"]), float(cfg["lr_max"])],
                        "axis_tags": [suite_name, var["variant_id"], anchor_id],
                        "max_evals": max_evals,
                        "seed_offset": seed_cursor,
                        "cfg": cfg,
                    }
                    rows.append(row)

            # Router x feature-embed coupled matrix (standard/factored, 3~4 embeds)
            if (not requested_suites) or ({"router_suite", "feature_embed_suite"} <= requested_suites):
                router_vars = [v for v in defs["router_suite"] if v.get("router_variant") in {"standard", "factored"}]
                embed_vars = defs["feature_embed_suite"]
                for rv in router_vars:
                    for ev in embed_vars:
                        seed_cursor += 1
                        cfg = merge_overrides(base, rv.get("overrides", {}))
                        cfg = merge_overrides(cfg, ev.get("overrides", {}))
                        prof = dataset_profile(args.dataset, int(cfg.get("MAX_ITEM_LIST_LENGTH", 20)))
                        cfg["train_batch_size"] = int(prof["train_bs"])
                        cfg["eval_batch_size"] = int(prof["eval_bs"])
                        rows.append(
                            {
                                "run_phase": f"{phase_name}_{anchor_id}_RXF_{rv['variant_id']}_{ev['variant_id']}",
                                "phase": phase_name,
                                "phase_idx": phase_idx,
                                "category": "router_x_feature_embed",
                                "suite_id": "router_x_feature_embed",
                                "variant_id": f"{rv['variant_id']}__{ev['variant_id']}",
                                "anchor_id": anchor_id,
                                "router_variant": rv.get("router_variant", "unknown"),
                                "d_feat_emb": int(cfg.get("d_feat_emb", 16)),
                                "lr_range": [float(cfg["lr_min"]), float(cfg["lr_max"])],
                                "axis_tags": ["router_suite", rv["variant_id"], "feature_embed_suite", ev["variant_id"], anchor_id],
                                "max_evals": max_evals,
                                "seed_offset": seed_cursor,
                                "cfg": cfg,
                            }
                        )

    # Filters
    if args.category:
        allowed = {tok.strip().lower() for tok in args.category.split(",") if tok.strip()}
        rows = [r for r in rows if str(r.get("category", "")).lower() in allowed]

    if args.only:
        allowed = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        rows = [r for r in rows if r["run_phase"] in allowed]

    if not rows:
        raise SystemExit("No runs selected after filters")

    # Deterministic order and gpu assignment
    rows.sort(key=lambda r: (r["phase_idx"], r["anchor_id"], r["category"], r["variant_id"], r["run_phase"]))
    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['phase'])}_"
        f"{sanitize_slug(row['suite_id'])}_"
        f"{sanitize_slug(row['anchor_id'])}_"
        f"{sanitize_slug(row['variant_id'])}_"
        f"{ts}"
    )


def log_path(row: dict[str, Any], dataset: str) -> Path:
    root = LOG_ROOT / AXIS / row["phase"] / dataset / "FMoEN3"
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


def build_command(row: dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    cfg = row["cfg"]
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name", "config",
        "--max-evals", str(int(row["max_evals"])),
        "--tune-epochs", str(int(args.tune_epochs)),
        "--tune-patience", str(int(args.tune_patience)),
        "--seed", str(int(args.seed_base) + int(row["seed_offset"])),
        "--run-group", TRACK,
        "--run-axis", AXIS,
        "--run-phase", row["run_phase"],
        "model=featured_moe_n3_tune",
        f"dataset={cfg['dataset']}",
        "eval_mode=session",
        f"feature_mode={cfg['feature_mode']}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_artifact_logging_policy=final_only",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(row['phase'])}",
        f"++lfm_aio_anchor_id={hydra_literal(row['anchor_id'])}",
        f"++lfm_aio_suite_id={hydra_literal(row['suite_id'])}",
        f"++lfm_aio_variant_id={hydra_literal(row['variant_id'])}",
        f"++lfm_aio_router_variant={hydra_literal(row['router_variant'])}",
        f"++lfm_aio_axis_tags={hydra_literal(row['axis_tags'])}",
        f"MAX_ITEM_LIST_LENGTH={int(cfg['MAX_ITEM_LIST_LENGTH'])}",
        f"train_batch_size={int(cfg['train_batch_size'])}",
        f"eval_batch_size={int(cfg['eval_batch_size'])}",
        f"embedding_size={int(cfg['embedding_size'])}",
        f"num_heads={int(cfg['num_heads'])}",
        f"d_ff={int(cfg['d_ff'])}",
        f"d_feat_emb={int(cfg['d_feat_emb'])}",
        f"d_router_hidden={int(cfg['d_router_hidden'])}",
        f"d_expert_hidden={int(cfg['d_expert_hidden'])}",
        f"expert_scale={int(cfg['expert_scale'])}",
        f"hidden_dropout_prob={hydra_literal(cfg['hidden_dropout_prob'])}",
        f"attn_dropout_prob={hydra_literal(cfg['attn_dropout_prob'])}",
        f"++weight_decay={hydra_literal(cfg['weight_decay'])}",
        f"++layer_layout={hydra_literal(cfg['layer_layout'])}",
        f"++stage_compute_mode={hydra_literal(cfg['stage_compute_mode'])}",
        f"++stage_router_mode={hydra_literal(cfg['stage_router_mode'])}",
        f"++stage_router_type={hydra_literal(cfg['stage_router_type'])}",
        f"++stage_router_source={hydra_literal(cfg['stage_router_source'])}",
        f"++stage_feature_encoder_mode={hydra_literal(cfg['stage_feature_encoder_mode'])}",
        f"++stage_feature_injection={hydra_literal(cfg['stage_feature_injection'])}",
        f"++stage_router_granularity={hydra_literal(cfg['stage_router_granularity'])}",
        f"++stage_feature_family_mask={hydra_literal(cfg['stage_feature_family_mask'])}",
        f"++stage_residual_mode={hydra_literal(cfg['stage_residual_mode'])}",
        f"++macro_history_window={int(cfg['macro_history_window'])}",
        f"++topk_scope_mode={hydra_literal(cfg['topk_scope_mode'])}",
        f"++moe_top_k={int(cfg['moe_top_k'])}",
        f"++moe_top_k_ratio={hydra_literal(cfg['moe_top_k_ratio'])}",
        f"++moe_top_k_policy={hydra_literal(cfg['moe_top_k_policy'])}",
        f"++balance_loss_lambda={hydra_literal(cfg['balance_loss_lambda'])}",
        f"++z_loss_lambda={hydra_literal(cfg['z_loss_lambda'])}",
        f"++route_smoothness_lambda={hydra_literal(cfg['route_smoothness_lambda'])}",
        f"++route_consistency_lambda={hydra_literal(cfg['route_consistency_lambda'])}",
        f"++route_sharpness_lambda={hydra_literal(cfg['route_sharpness_lambda'])}",
        f"++route_monopoly_lambda={hydra_literal(cfg['route_monopoly_lambda'])}",
        f"++route_monopoly_tau={hydra_literal(cfg['route_monopoly_tau'])}",
        f"++route_prior_lambda={hydra_literal(cfg['route_prior_lambda'])}",
        f"++group_prior_align_lambda={hydra_literal(cfg['group_prior_align_lambda'])}",
        f"++factored_group_balance_lambda={hydra_literal(cfg['factored_group_balance_lambda'])}",
        f"++lr_scheduler_type={hydra_literal(cfg['lr_scheduler_type'])}",
        f"++lr_scheduler_warmup_ratio={hydra_literal(cfg['lr_scheduler_warmup_ratio'])}",
        f"++lr_scheduler_min_lr_ratio={hydra_literal(cfg['lr_scheduler_min_lr_ratio'])}",
        f"++lr_scheduler_plateau_factor={hydra_literal(cfg['lr_scheduler_plateau_factor'])}",
        f"++lr_scheduler_plateau_patience={hydra_literal(cfg['lr_scheduler_plateau_patience'])}",
        # LR-only search
        f"++search.learning_rate={hydra_literal([cfg['lr_min'], cfg['lr_max']])}",
        "++search_space_type_overrides.learning_rate=loguniform",
    ]

    # Optional keys present in factored suites.
    for key in [
        "stage_factored_group_router_source",
        "stage_factored_group_logit_scale",
        "stage_factored_intra_logit_scale",
        "stage_factored_combine_mode",
        "fmoe_schedule_enable",
        "alpha_warmup_until",
        "alpha_warmup_start",
        "alpha_warmup_end",
    ]:
        if key in cfg:
            cmd.append(f"++{key}={hydra_literal(cfg[key])}")

    return cmd


def write_log_preamble(log_file: Path, row: dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    cfg = row["cfg"]
    lines = [
        "[LFM_ALLINONE_HEADER]",
        f"phase={row['phase']} run_phase={row['run_phase']} category={row['category']} suite={row['suite_id']} variant={row['variant_id']}",
        f"anchor={row['anchor_id']} router_variant={row['router_variant']} d_feat_emb={row['d_feat_emb']} lr_range={row['lr_range']}",
        f"dataset={cfg['dataset']} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={row['max_evals']} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={int(args.seed_base) + int(row['seed_offset'])}",
        f"axis_tags={json.dumps(row['axis_tags'], ensure_ascii=True)}",
        "",
        "[ANCHOR_FIXED]",
        f"embedding_size={cfg['embedding_size']} d_ff={cfg['d_ff']} d_router_hidden={cfg['d_router_hidden']} d_expert_hidden={cfg['d_expert_hidden']} expert_scale={cfg['expert_scale']}",
        f"weight_decay={cfg['weight_decay']} hidden_dropout_prob={cfg['hidden_dropout_prob']} attn_dropout_prob={cfg['attn_dropout_prob']}",
        "",
        "[SEARCH_POLICY]",
        f"learning_rate=loguniform[{cfg['lr_min']}, {cfg['lr_max']}], others=fixed",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LFM all-in-one phase launcher (P0~P2)")
    p.add_argument("--dataset", default="lastfm0.03")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--from-phase", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--to-phase", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--p0-max-evals", type=int, default=10)
    p.add_argument("--p1-max-evals", type=int, default=10)
    p.add_argument("--p2-max-evals", type=int, default=10)
    p.add_argument("--tune-epochs", type=int, default=100)
    p.add_argument("--tune-patience", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=22000)
    p.add_argument("--anchors", default="AN_S,AN_M,AN_L")
    p.add_argument("--suites", default="")
    p.add_argument("--category", default="")
    p.add_argument("--only", default="")
    p.add_argument("--manifest-out", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_phase > args.to_phase:
        raise SystemExit("--from-phase must be <= --to-phase")

    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided")

    rows = build_rows(args)
    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
        row["assigned_order"] = idx
        row["assigned_gpu"] = gpus[(idx - 1) % len(gpus)]

    payload = {
        "track": TRACK,
        "axis": AXIS,
        "dataset": args.dataset,
        "from_phase": args.from_phase,
        "to_phase": args.to_phase,
        "anchors": args.anchors,
        "suites": args.suites,
        "rows": rows,
    }
    INVENTORY_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_out) if args.manifest_out else INVENTORY_ROOT / f"lfm_allinone_{int(time.time())}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[lfm_allinone] dataset={args.dataset} phases=P{args.from_phase}..P{args.to_phase} "
        f"runs={len(rows)} gpus={','.join(gpus)} manifest={manifest_path}"
    )

    if args.dry_run:
        for row in rows:
            lp = log_path(row, args.dataset)
            cmd = build_command(row, row["assigned_gpu"], args)
            write_log_preamble(lp, row, row["assigned_gpu"], args, cmd)
            print(f"[dry-run] gpu={row['assigned_gpu']} {row['run_phase']} -> {lp}")
        return 0

    gpu_bins: dict[str, deque[dict[str, Any]]] = {g: deque() for g in gpus}
    for row in rows:
        gpu_bins[row["assigned_gpu"]].append(row)

    active: dict[str, tuple[subprocess.Popen[Any], dict[str, Any], Path]] = {}
    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not gpu_bins[gpu_id]:
                continue
            row = gpu_bins[gpu_id].popleft()
            lp = log_path(row, args.dataset)
            cmd = build_command(row, gpu_id, args)
            write_log_preamble(lp, row, gpu_id, args, cmd)
            env = dict(os.environ)
            env["HYPEROPT_RESULTS_DIR"] = str(ARTIFACT_ROOT / "results")
            with lp.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu_id] = (proc, row, lp)
            print(f"[launch] gpu={gpu_id} {row['run_phase']}")

        done_gpu: list[str] = []
        for gpu_id, (proc, row, lp) in active.items():
            rc = proc.poll()
            if rc is None:
                continue
            done_gpu.append(gpu_id)
            print(f"[done] gpu={gpu_id} {row['run_phase']} rc={rc} log={lp}")

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = any(gpu_bins[g] for g in gpus)
        if not pending and not active:
            break
        time.sleep(3)

    print("[done] all queues completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
