#!/usr/bin/env python3
"""Launch FMoE_N3 phase6 candidate reinforcement experiments.

Phase6 scope:
1) Candidate confirmation: A/B/C x 3 seeds (candidate rows use 3x max-evals)
2) Baseline bridge expansion (8 runs: SASRec + hidden/feature/both x std/factored)
3) Router x Injection 2x2 under 2 contexts (8 runs)
4) Specialization ablation (5 methods x candidate A/B = 10 runs)
5) Feature ablation sweep on candidate B (macro window 5/10, family mask size 1 or 2)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase6_candidate_reinfor_v2"
PHASE = "P6"
FAMILIES = ["Tempo", "Memory", "Focus", "Exposure"]


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


def _all_stage_map(value: str) -> dict[str, str]:
    return {"macro": value, "mid": value, "micro": value}


def _all_stage_float(value: float) -> dict[str, float]:
    v = float(value)
    return {"macro": v, "mid": v, "micro": v}


def _mask_all(families: list[str]) -> dict[str, list[str]]:
    uniq = []
    for fam in families:
        f = str(fam).strip()
        if f and f not in uniq:
            uniq.append(f)
    return {"macro": list(uniq), "mid": list(uniq), "micro": list(uniq)}


def _candidate_profiles() -> dict[str, dict[str, Any]]:
    return {
        "A": {
            "router_type": "standard",
            "injection": "gated_bias",
            "router_source": "both",
            "residual_mode": "base",
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
            "route_smoothness_lambda": 0.01,
            "group_prior_align_lambda": 0.0,
            "factored_group_balance_lambda": 0.0,
            "route_consistency_lambda": 0.0,
            "route_prior_lambda": 0.0,
            "warmup": False,
        },
        "B": {
            "router_type": "factored",
            "injection": "group_gated_bias",
            "router_source": "both",
            "residual_mode": "base",
            "topk_scope_mode": "group_dense",
            "moe_top_k": 0,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
            "route_smoothness_lambda": 0.01,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "route_consistency_lambda": 0.0,
            "route_prior_lambda": 0.0,
            "warmup": False,
        },
        "C": {
            "router_type": "factored",
            "injection": "group_gated_bias",
            "router_source": "both",
            "residual_mode": "shared_moe_learned_warmup",
            "topk_scope_mode": "group_dense",
            "moe_top_k": 0,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
            "route_smoothness_lambda": 0.01,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "route_consistency_lambda": 0.01,
            "route_prior_lambda": 0.005,
            "warmup": True,
        },
    }


def _base_row(dataset: str, *, seed_offset: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "seed_offset": int(seed_offset),
        "feature_mode": "full_v3",
        "max_item_list_length": 20,
        "batch_size": 4096,
        "layer_layout": ["macro", "mid", "micro"],
        "search_learning_rate": [2.0e-4, 2.0e-3],
        "search_weight_decay": [1e-7, 1e-6, 1e-5, 5e-5],
        "search_hidden_dropout_prob": [0.1, 0.15, 0.2],
        "search_lr_scheduler_type": ["warmup_cosine"],
        "overrides": {},
    }


def _apply_candidate(row: dict[str, Any], cand: dict[str, Any]) -> None:
    row["overrides"].update(
        {
            "stage_router_type": _all_stage_map(cand["router_type"]),
            "stage_feature_injection": _all_stage_map(cand["injection"]),
            "stage_router_source": _all_stage_map(cand["router_source"]),
            "stage_residual_mode": _all_stage_map(cand["residual_mode"]),
            "topk_scope_mode": cand["topk_scope_mode"],
            "moe_top_k": int(cand["moe_top_k"]),
            "balance_loss_lambda": float(cand["balance_loss_lambda"]),
            "z_loss_lambda": float(cand["z_loss_lambda"]),
            "route_smoothness_lambda": float(cand["route_smoothness_lambda"]),
            "group_prior_align_lambda": float(cand["group_prior_align_lambda"]),
            "factored_group_balance_lambda": float(cand["factored_group_balance_lambda"]),
            "route_consistency_lambda": float(cand["route_consistency_lambda"]),
            "route_prior_lambda": float(cand["route_prior_lambda"]),
        }
    )
    if cand.get("warmup"):
        row["overrides"].update(
            {
                "fmoe_schedule_enable": True,
                "alpha_warmup_until": 0.15,
                "alpha_warmup_start": 0.0,
                "alpha_warmup_end": 1.0,
                "residual_alpha_init": _all_stage_float(-2.0),
            }
        )


def _build_candidate_rows(dataset: str, *, max_evals: int, seed_base_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = _candidate_profiles()
    wide_lr = [8.0e-5, 8.0e-3]
    run_idx = 0
    for cand_key in ("A", "B", "C"):
        for seed_id in range(1, 4):
            run_idx += 1
            row = _base_row(dataset, seed_offset=seed_base_offset + run_idx)
            row["category"] = "cand3x"
            row["combo_id"] = f"CAND_{cand_key}_S{seed_id}"
            row["combo_desc"] = f"candidate_{cand_key}_seed{seed_id}"
            row["run_phase"] = f"P6_CAND_{cand_key}_S{seed_id}"
            row["max_evals"] = int(max_evals) * 3
            row["search_learning_rate"] = list(wide_lr)
            _apply_candidate(row, profiles[cand_key])
            rows.append(row)
    return rows


def _build_router_x_injection_rows(dataset: str, *, max_evals: int, seed_base_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contexts = [
        {
            "ctx": "X1",
            "base": {
                "stage_residual_mode": _all_stage_map("base"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 1,
                "route_smoothness_lambda": 0.01,
                "group_prior_align_lambda": 0.0,
            },
        },
        {
            "ctx": "X2",
            "base": {
                "stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"),
                "topk_scope_mode": "group_top2_pergroup",
                "moe_top_k": 2,
                "fmoe_schedule_enable": True,
                "alpha_warmup_until": 0.15,
                "alpha_warmup_start": 0.0,
                "alpha_warmup_end": 1.0,
                "residual_alpha_init": _all_stage_float(-2.0),
                "route_smoothness_lambda": 0.01,
                "group_prior_align_lambda": 5e-4,
            },
        },
    ]
    run_idx = 0
    for ctx in contexts:
        for router in ("standard", "factored"):
            for inj in ("gated_bias", "group_gated_bias"):
                run_idx += 1
                row = _base_row(dataset, seed_offset=seed_base_offset + run_idx)
                row["category"] = "router2x2"
                row["combo_id"] = f"RXI_{ctx['ctx']}_{router}_{inj}"
                row["combo_desc"] = f"router_x_inj_{ctx['ctx']}"
                row["run_phase"] = f"P6_RXI_{ctx['ctx']}_{router[:3].upper()}_{inj[:3].upper()}"
                row["max_evals"] = int(max_evals)
                row["overrides"].update(
                    {
                        "stage_router_source": _all_stage_map("both"),
                        "stage_router_type": _all_stage_map(router),
                        "stage_feature_injection": _all_stage_map(inj),
                        "balance_loss_lambda": 0.002,
                        "z_loss_lambda": 1e-4,
                        "factored_group_balance_lambda": 1e-3 if router == "factored" else 0.0,
                        "route_consistency_lambda": 0.0,
                        "route_prior_lambda": 0.0,
                        **ctx["base"],
                    }
                )
                rows.append(row)
    return rows


def _build_spec_rows(dataset: str, *, max_evals: int, seed_base_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profiles = _candidate_profiles()
    methods = [
        # M0 is a hard off switch for specialization regularizers.
        (
            "M0",
            {
                "route_smoothness_lambda": 0.0,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.0,
                "route_monopoly_lambda": 0.0,
                "route_prior_lambda": 0.0,
                "group_prior_align_lambda": 0.0,
                "factored_group_balance_lambda": 0.0,
            },
        ),
        # M1/M2/M3 use stronger-than-phase5 single-axis pressure.
        ("M1", {"route_smoothness_lambda": 0.04}),
        ("M2", {"route_consistency_lambda": 0.04, "route_consistency_pairs": 8}),
        ("M3", {"route_sharpness_lambda": 0.01, "route_monopoly_lambda": 0.04, "route_monopoly_tau": 0.25}),
        # M4 is a stricter mixed prior+consistency setting than phase5-style blends.
        (
            "M4",
            {
                "route_smoothness_lambda": 0.02,
                "route_consistency_lambda": 0.03,
                "route_consistency_pairs": 8,
                "route_prior_lambda": 0.01,
                "group_prior_align_lambda": 1e-3,
                "factored_group_balance_lambda": 2e-3,
            },
        ),
    ]
    run_idx = 0
    for cand_key in ("A", "B"):
        for method_id, aux in methods:
            run_idx += 1
            row = _base_row(dataset, seed_offset=seed_base_offset + run_idx)
            row["category"] = "spec_ablation"
            row["combo_id"] = f"SPEC_{cand_key}_{method_id}"
            row["combo_desc"] = f"spec_{cand_key}_{method_id}"
            row["run_phase"] = f"P6_SPEC_{cand_key}_{method_id}"
            row["max_evals"] = int(max_evals)
            _apply_candidate(row, profiles[cand_key])
            row["overrides"].update(aux)
            rows.append(row)
    return rows


def _build_feature_rows(dataset: str, *, max_evals: int, seed_base_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_b = _candidate_profiles()["B"]
    masks = []
    for fam in FAMILIES:
        masks.append([fam])
    for fam1, fam2 in itertools.combinations(FAMILIES, 2):
        masks.append([fam1, fam2])

    run_idx = 0
    for window in (5, 10):
        for fams in masks:
            run_idx += 1
            row = _base_row(dataset, seed_offset=seed_base_offset + run_idx)
            fam_slug = "_".join(f.lower()[:3] for f in fams)
            row["category"] = "feature_ablation"
            row["combo_id"] = f"FEAT_W{window}_{fam_slug}"
            row["combo_desc"] = f"feature_mask_{window}_{','.join(fams)}"
            row["run_phase"] = f"P6_FEAT_W{window}_{sanitize_slug(fam_slug).upper()}"
            row["max_evals"] = int(max_evals)
            _apply_candidate(row, base_b)
            row["overrides"].update(
                {
                    "macro_history_window": int(window),
                    "stage_feature_family_mask": _mask_all(fams),
                }
            )
            rows.append(row)
    return rows


def _build_baseline_bridge_rows(dataset: str, *, max_evals: int, seed_base_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        (
            "B0",
            "sasrec_equiv_len20",
            {
                "layer_layout": ["layer", "layer", "layer"],
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("none"),
                "stage_router_type": _all_stage_map("standard"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 0,
                "macro_history_window": 5,
            },
        ),
        (
            "B1",
            "sasrec_equiv_w10",
            {
                "layer_layout": ["layer", "layer", "layer"],
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("none"),
                "stage_router_type": _all_stage_map("standard"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 0,
                "macro_history_window": 10,
            },
        ),
        (
            "B2",
            "moe_hidden_only_std",
            {
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("none"),
                "stage_router_type": _all_stage_map("standard"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 1,
                "macro_history_window": 5,
            },
        ),
        (
            "B3",
            "moe_hidden_only_factored",
            {
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("none"),
                "stage_router_type": _all_stage_map("factored"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 1,
                "macro_history_window": 5,
            },
        ),
        (
            "B4",
            "moe_feature_only_std",
            {
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("feature"),
                "stage_feature_injection": _all_stage_map("none"),
                "stage_router_type": _all_stage_map("standard"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 1,
                "macro_history_window": 5,
            },
        ),
        (
            "B5",
                "moe_feature_only_factored",
            {
                "stage_compute_mode": _all_stage_map("moe"),
                    "stage_router_source": _all_stage_map("feature"),
                    "stage_feature_injection": _all_stage_map("none"),
                    "stage_router_type": _all_stage_map("factored"),
                    "topk_scope_mode": "group_dense",
                    "moe_top_k": 1,
                    "macro_history_window": 5,
                },
            ),
            (
                "B6",
                "moe_both_std",
                {
                    "stage_compute_mode": _all_stage_map("moe"),
                    "stage_router_source": _all_stage_map("both"),
                    "stage_feature_injection": _all_stage_map("gated_bias"),
                    "stage_router_type": _all_stage_map("standard"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 1,
                "macro_history_window": 5,
            },
        ),
        (
                "B7",
            "moe_both_factored",
            {
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_source": _all_stage_map("both"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_router_type": _all_stage_map("factored"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 1,
                "macro_history_window": 5,
            },
        ),
    ]

    run_idx = 0
    for bid, desc, cfg in specs:
        run_idx += 1
        row = _base_row(dataset, seed_offset=seed_base_offset + run_idx)
        row["category"] = "baseline_bridge"
        row["combo_id"] = bid
        row["combo_desc"] = desc
        row["run_phase"] = f"P6_BASE_{bid}"
        row["max_evals"] = int(max_evals)
        row["search_learning_rate"] = [2.0e-4, 2.0e-3]
        row["overrides"].update(
            {
                "balance_loss_lambda": 0.0,
                "z_loss_lambda": 0.0,
                "route_smoothness_lambda": 0.0,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.0,
                "route_monopoly_lambda": 0.0,
                "route_prior_lambda": 0.0,
                **cfg,
            }
        )
        rows.append(row)
    return rows


def build_rows(dataset: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    suites = [s.strip().lower() for s in str(args.suites or "all").split(",") if s.strip()]
    run_all = (not suites) or ("all" in suites)

    def use(name: str) -> bool:
        return run_all or (name in suites)

    if use("candidate"):
        rows.extend(_build_candidate_rows(dataset, max_evals=args.max_evals, seed_base_offset=offset))
        offset += 100
    if use("base"):
        rows.extend(_build_baseline_bridge_rows(dataset, max_evals=args.max_evals, seed_base_offset=offset))
        offset += 100
    if use("router"):
        rows.extend(_build_router_x_injection_rows(dataset, max_evals=args.max_evals, seed_base_offset=offset))
        offset += 100
    if use("spec"):
        rows.extend(_build_spec_rows(dataset, max_evals=args.max_evals, seed_base_offset=offset))
        offset += 100
    if use("feature"):
        rows.extend(_build_feature_rows(dataset, max_evals=args.max_evals, seed_base_offset=offset))
        offset += 100

    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['category'])}_"
        f"{sanitize_slug(row['combo_id'])}_"
        f"{ts}"
    )


def log_path(row: dict[str, Any], dataset: str) -> Path:
    dataset_tag = dataset.replace("/", "_")
    model_tag = "FMoEN3"
    root = LOG_ROOT / AXIS / PHASE / dataset_tag / model_tag
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
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name", "config",
        "--max-evals", str(int(row["max_evals"])),
        "--tune-epochs", str(args.tune_epochs),
        "--tune-patience", str(args.tune_patience),
        "--seed", str(args.seed_base + int(row["seed_offset"])),
        "--run-group", TRACK,
        "--run-axis", AXIS,
        "--run-phase", row["run_phase"],
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session",
        f"feature_mode={row['feature_mode']}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        f"fmoe_feature_ablation_logging={hydra_literal(bool(args.feature_ablation_logging))}",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE)}",
        f"MAX_ITEM_LIST_LENGTH={int(row['max_item_list_length'])}",
        f"train_batch_size={int(row['batch_size'])}",
        f"eval_batch_size={int(row['batch_size'])}",
        "embedding_size=128",
        "num_heads=4",
        "attn_dropout_prob=0.1",
        "d_ff=256",
        "d_feat_emb=16",
        "d_expert_hidden=128",
        "d_router_hidden=64",
        "expert_scale=3",
        f"++layer_layout={hydra_literal(row['layer_layout'])}",
        f"++search.learning_rate={hydra_literal(row['search_learning_rate'])}",
        f"++search.weight_decay={hydra_literal(row['search_weight_decay'])}",
        f"++search.hidden_dropout_prob={hydra_literal(row['search_hidden_dropout_prob'])}",
        f"++search.lr_scheduler_type={hydra_literal(row['search_lr_scheduler_type'])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        f"++p6_category={hydra_literal(row['category'])}",
        f"++p6_combo_id={hydra_literal(row['combo_id'])}",
        f"++p6_combo_desc={hydra_literal(row['combo_desc'])}",
    ]

    for key, value in row.get("overrides", {}).items():
        if isinstance(value, dict):
            cmd.append(f"++{key}={hydra_literal(value)}")
        else:
            if key in {"topk_scope_mode", "stage_feature_family_mask", "stage_compute_mode", "stage_router_source", "stage_router_type", "stage_feature_injection", "stage_residual_mode", "residual_alpha_init"}:
                cmd.append(f"++{key}={hydra_literal(value)}")
            else:
                cmd.append(f"{key}={hydra_literal(value)}")
    return cmd


def write_log_preamble(log_file: Path, row: dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE6_COMBO_HEADER]",
        f"run_phase={row['run_phase']} category={row['category']} combo={row['combo_id']} desc={row['combo_desc']}",
        f"dataset={row['dataset']} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={row['max_evals']} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={args.seed_base + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 phase6 candidate reinforcement launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--suites", default="all", help="all|candidate,router,spec,feature,base")
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=16000)
    parser.add_argument("--feature-ablation-logging", dest="feature_ablation_logging", action="store_true")
    parser.add_argument("--no-feature-ablation-logging", dest="feature_ablation_logging", action="store_false")
    parser.set_defaults(feature_ablation_logging=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated run_phase values")
    parser.add_argument("--category", default="", help="Comma-separated categories to keep")
    parser.add_argument("--manifest-out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided")

    rows = build_rows(args.dataset, args)

    if args.category:
        allowed = {tok.strip().lower() for tok in args.category.split(",") if tok.strip()}
        rows = [r for r in rows if str(r.get("category", "")).lower() in allowed]

    if args.only:
        allowed = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        rows = [r for r in rows if r["run_phase"] in allowed]

    if not rows:
        raise SystemExit("No phase6 runs selected")

    for idx, row in enumerate(rows):
        row["assigned_order"] = idx + 1
        row["assigned_gpu"] = gpus[idx % len(gpus)]

    if args.manifest_out:
        mp = Path(args.manifest_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"phase": PHASE, "axis": AXIS, "rows": rows}, indent=2), encoding="utf-8")

    print(f"[phase6] dataset={args.dataset} suites={args.suites} runs={len(rows)} gpus={','.join(gpus)}")

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
                p = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu_id] = (p, row, lp)
            print(f"[launch] gpu={gpu_id} {row['run_phase']}")

        done_gpu = []
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

    print("[done] phase6 candidate reinforcement queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
