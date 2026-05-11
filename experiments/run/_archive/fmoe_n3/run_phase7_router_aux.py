#!/usr/bin/env python3
"""Launch FMoE_N3 Phase7 router-vs-aux focused fixed-setting experiments.

Phase7 design (KuaiRecLargeStrictPosV2_0.2 first):
- 16 fixed settings x 4 seeds = 64 runs
- Group A (router variants): 8 settings x 4 seeds = 32 runs
- Group B (aux/reg variants): 8 settings x 4 seeds = 32 runs
- Default queue target: 8 GPUs, round-robin assignment, per-GPU independent FIFO.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/workspace/jy1559/FMoE")
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"

TRACK = "fmoe_n3"
AXIS = "phase7_router_aux_v1"
PHASE = "P7"


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


def _router_templates() -> dict[str, dict[str, Any]]:
    """Router-input variants requested by phase7 design."""
    return {
        # Stage feature + hidden together, standard router.
        "R0_STD": {
            "router_label": "standard",
            "overrides": {
                "stage_router_type": _all_stage_map("standard"),
                "stage_router_source": _all_stage_map("both"),
                "stage_feature_injection": _all_stage_map("gated_bias"),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 0,
            },
        },
        # Stage feature + hidden together, factored router.
        "R1_FAC": {
            "router_label": "factored",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("both"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("feature"),
                "stage_factored_group_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_intra_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_combine_mode": _all_stage_map("add"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # Factored-heavy: emphasize group feature in router input.
        "R2_FAC_HEAVY": {
            "router_label": "factored_heavy",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("feature"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("both"),
                "stage_factored_group_logit_scale": {"macro": 1.6, "mid": 1.6, "micro": 1.6},
                "stage_factored_intra_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_combine_mode": _all_stage_map("add"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # Factored-only (no stage-whole feature routing path).
        "R3_FAC_ONLY": {
            "router_label": "factored_only_no_stage_feature_route",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("feature"),
                "stage_factored_group_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_intra_logit_scale": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
                "stage_factored_combine_mode": _all_stage_map("add"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # HIR: group weights (column) x intra-group weights (row-like per group) -> expert weights.
        "R4_HIR": {
            "router_label": "hir_multiplicative",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("both"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("both"),
                "stage_factored_group_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_intra_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_combine_mode": _all_stage_map("hir"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # FAC_GROUP: dense expert logits + group-importance bias from grouped features.
        "R5_FAC_GROUP": {
            "router_label": "fac_group_importance_bias",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("both"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("feature"),
                "stage_factored_group_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_intra_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_combine_mode": _all_stage_map("fac_group"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # FAC_ONLY_BOTH: intra route disabled, group router sees both hidden+feature.
        "R6_FAC_ONLY_BOTH": {
            "router_label": "factored_only_both",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("hidden"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("both"),
                "stage_factored_group_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_intra_logit_scale": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
                "stage_factored_combine_mode": _all_stage_map("add"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
        # FAC_HEAVY_FEAT: heavy group scale but group router source fixed to feature.
        "R7_FAC_HEAVY_FEAT": {
            "router_label": "factored_heavy_groupfeat",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_router_source": _all_stage_map("feature"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "stage_factored_group_router_source": _all_stage_map("feature"),
                "stage_factored_group_logit_scale": {"macro": 1.6, "mid": 1.6, "micro": 1.6},
                "stage_factored_intra_logit_scale": {"macro": 1.0, "mid": 1.0, "micro": 1.0},
                "stage_factored_combine_mode": _all_stage_map("add"),
                "topk_scope_mode": "group_dense",
                "moe_top_k": 0,
            },
        },
    }


def _base_aux(router_key: str) -> dict[str, Any]:
    # All router variants except R0_STD are factored-family settings.
    factored = str(router_key) != "R0_STD"
    return {
        "balance_loss_lambda": 0.002,
        "z_loss_lambda": 1e-4,
        "route_smoothness_lambda": 0.01,
        "route_consistency_lambda": 0.0,
        "route_sharpness_lambda": 0.0,
        "route_monopoly_lambda": 0.0,
        "route_monopoly_tau": 0.25,
        "route_prior_lambda": 0.0,
        "group_prior_align_lambda": 5e-4 if factored else 0.0,
        "factored_group_balance_lambda": 1e-3 if factored else 0.0,
    }


def _settings_router_core() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for router_key, router_cfg in _router_templates().items():
        rows.append(
            {
                "setting_id": router_key,
                "setting_group": "router_core",
                "router_key": router_key,
                "setting_desc": f"router_only::{router_cfg['router_label']}",
                "overrides": {
                    **router_cfg["overrides"],
                    **_base_aux(router_key),
                },
            }
        )
    return rows


def _aux_variants() -> list[dict[str, Any]]:
    return [
        {
            "aux_key": "BAL_A",
            "aux_desc": "balance_light(load_balance+z)",
            "aux_overrides": {
                "balance_loss_lambda": 0.002,
                "z_loss_lambda": 1e-4,
                "route_smoothness_lambda": 0.0,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.0,
                "route_monopoly_lambda": 0.0,
                "route_prior_lambda": 0.0,
            },
        },
        {
            "aux_key": "BAL_B",
            "aux_desc": "balance_strong(load_balance+z)",
            "aux_overrides": {
                "balance_loss_lambda": 0.006,
                "z_loss_lambda": 3e-4,
                "route_smoothness_lambda": 0.0,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.0,
                "route_monopoly_lambda": 0.0,
                "route_prior_lambda": 0.0,
            },
        },
        {
            "aux_key": "SPEC_A",
            "aux_desc": "specialization_smoothness",
            "aux_overrides": {
                "balance_loss_lambda": 0.002,
                "z_loss_lambda": 1e-4,
                "route_smoothness_lambda": 0.04,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.0,
                "route_monopoly_lambda": 0.0,
                "route_prior_lambda": 0.0,
            },
        },
        {
            "aux_key": "SPEC_B",
            "aux_desc": "specialization_sharp_monopoly",
            "aux_overrides": {
                "balance_loss_lambda": 0.002,
                "z_loss_lambda": 1e-4,
                "route_smoothness_lambda": 0.0,
                "route_consistency_lambda": 0.0,
                "route_sharpness_lambda": 0.01,
                "route_monopoly_lambda": 0.04,
                "route_monopoly_tau": 0.25,
                "route_prior_lambda": 0.0,
            },
        },
    ]


def _settings_aux_reg() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Requested anchors: standard + factored_heavy
    for router_key in ("R0_STD", "R2_FAC_HEAVY"):
        router_cfg = _router_templates()[router_key]
        base_aux = _base_aux(router_key)
        for aux in _aux_variants():
            setting_id = f"AUX_{router_key}_{aux['aux_key']}"
            rows.append(
                {
                    "setting_id": setting_id,
                    "setting_group": "aux_reg",
                    "router_key": router_key,
                    "aux_key": aux["aux_key"],
                    "setting_desc": f"aux::{router_cfg['router_label']}::{aux['aux_desc']}",
                    "overrides": {
                        **router_cfg["overrides"],
                        **base_aux,
                        **aux["aux_overrides"],
                    },
                }
            )
    return rows


def build_settings() -> list[dict[str, Any]]:
    settings = _settings_router_core() + _settings_aux_reg()
    for idx, setting in enumerate(settings, start=1):
        setting["setting_index"] = idx
    return settings


def expand_rows(dataset: str, settings: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for setting in settings:
        for seed in seeds:
            run_phase = f"P7_{setting['setting_id']}_S{seed}"
            rows.append(
                {
                    "dataset": dataset,
                    "seed_id": int(seed),
                    "seed_offset": int(seed_offset),
                    "run_phase": run_phase,
                    "setting_id": setting["setting_id"],
                    "setting_index": setting["setting_index"],
                    "setting_group": setting["setting_group"],
                    "setting_desc": setting["setting_desc"],
                    "router_key": setting["router_key"],
                    "aux_key": setting.get("aux_key", "BASE"),
                    "overrides": dict(setting["overrides"]),
                }
            )
            seed_offset += 1
    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['setting_group'])}_"
        f"{sanitize_slug(row['setting_id'])}_"
        f"s{int(row['seed_id'])}_"
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
        "--max-evals", str(int(args.max_evals)),
        "--tune-epochs", str(args.tune_epochs),
        "--tune-patience", str(args.tune_patience),
        "--seed", str(args.seed_base + int(row["seed_offset"])),
        "--run-group", TRACK,
        "--run-axis", AXIS,
        "--run-phase", row["run_phase"],
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
        f"++search.weight_decay={hydra_literal([float(args.weight_decay)])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(args.hidden_dropout_prob)])}",
        f"++search.lr_scheduler_type={hydra_literal([args.lr_scheduler_type])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        f"++p7_setting_id={hydra_literal(row['setting_id'])}",
        f"++p7_setting_group={hydra_literal(row['setting_group'])}",
        f"++p7_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++p7_router_key={hydra_literal(row['router_key'])}",
        f"++p7_aux_key={hydra_literal(row['aux_key'])}",
    ]

    for key, value in row.get("overrides", {}).items():
        if key in {
            "topk_scope_mode",
            "stage_compute_mode",
            "stage_router_source",
            "stage_router_type",
            "stage_factored_group_router_source",
            "stage_factored_group_logit_scale",
            "stage_factored_intra_logit_scale",
            "stage_factored_combine_mode",
            "stage_feature_injection",
            "stage_residual_mode",
            "residual_alpha_init",
        }:
            cmd.append(f"++{key}={hydra_literal(value)}")
        else:
            cmd.append(f"{key}={hydra_literal(value)}")
    return cmd


def write_log_preamble(log_file: Path, row: dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    lines = [
        "[PHASE7_SETTING_HEADER]",
        f"run_phase={row['run_phase']} setting={row['setting_id']} group={row['setting_group']} seed={row['seed_id']}",
        f"router_key={row['router_key']} aux_key={row['aux_key']}",
        f"desc={row['setting_desc']}",
        f"dataset={row['dataset']} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience}",
        f"seed={args.seed_base + int(row['seed_offset'])}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 phase7 router/aux launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3,4", help="Logical seed IDs in run-phase naming")
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=19000)

    # Fixed architecture/layout defaults for phase7.
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

    # Fixed optimization defaults.
    parser.add_argument("--search-lr-min", type=float, default=2.0e-4)
    parser.add_argument("--search-lr-max", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.15)
    parser.add_argument("--lr-scheduler-type", default="warmup_cosine")

    parser.add_argument("--group", default="all", help="all|router|aux")
    parser.add_argument("--only-setting", default="", help="Comma-separated setting_id values")
    parser.add_argument("--resume-from-logs", dest="resume_from_logs", action="store_true")
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.set_defaults(resume_from_logs=True)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _filter_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = rows
    group = str(args.group or "all").strip().lower()
    if group == "router":
        out = [r for r in out if r["setting_group"] == "router_core"]
    elif group == "aux":
        out = [r for r in out if r["setting_group"] == "aux_reg"]

    if args.only_setting:
        allow = {tok.strip() for tok in str(args.only_setting).split(",") if tok.strip()}
        out = [r for r in out if r["setting_id"] in allow]
    return out


def _phase7_log_dir(dataset: str) -> Path:
    dataset_tag = dataset.replace("/", "_")
    return LOG_ROOT / AXIS / PHASE / dataset_tag / "FMoEN3"


def _extract_run_phase_from_log(log_path: Path) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(8):
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
    root = _phase7_log_dir(dataset)
    if not root.exists():
        return done
    for log_path in sorted(root.glob("*.log")):
        run_phase = _extract_run_phase_from_log(log_path)
        if not run_phase:
            continue
        if _is_completed_log(log_path):
            done.add(run_phase)
    return done


def main() -> int:
    args = parse_args()
    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided")

    seed_ids = [int(tok.strip()) for tok in str(args.seeds).split(",") if tok.strip()]
    if not seed_ids:
        raise SystemExit("No seed IDs provided")

    settings = build_settings()
    rows = expand_rows(args.dataset, settings, seed_ids)
    rows = _filter_rows(rows, args)
    planned_rows = len(rows)
    if args.resume_from_logs:
        completed_run_phases = _scan_completed_run_phases(args.dataset)
        if completed_run_phases:
            rows = [r for r in rows if r["run_phase"] not in completed_run_phases]
            skipped = planned_rows - len(rows)
            if skipped > 0:
                print(
                    f"[phase7] resume_from_logs=on: skipped {skipped} completed runs "
                    f"(remaining {len(rows)}/{planned_rows})"
                )

    if not rows:
        print("[phase7] nothing to run (all selected runs already completed or filtered).")
        return 0

    for idx, row in enumerate(rows):
        row["assigned_order"] = idx + 1
        row["assigned_gpu"] = gpus[idx % len(gpus)]

    if args.manifest_out:
        mp = Path(args.manifest_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"phase": PHASE, "axis": AXIS, "rows": rows}, indent=2), encoding="utf-8")

    gpu_counter = Counter(r["assigned_gpu"] for r in rows)
    print(f"[phase7] dataset={args.dataset} rows={len(rows)} gpus={','.join(gpus)} seeds={seed_ids}")
    print(f"[phase7] group={args.group} settings={len(set(r['setting_id'] for r in rows))}")
    print("[phase7] gpu assignment:", ", ".join(f"gpu{g}:{gpu_counter.get(g, 0)}" for g in gpus))

    if args.dry_run:
        for row in rows:
            lp = log_path(row, args.dataset)
            cmd = build_command(row, row["assigned_gpu"], args)
            write_log_preamble(lp, row, row["assigned_gpu"], args, cmd)
            print(
                f"[dry-run] gpu={row['assigned_gpu']} {row['run_phase']} "
                f"(setting={row['setting_id']}, seed={row['seed_id']}) -> {lp}"
            )
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

    print("[done] phase7 router/aux queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
