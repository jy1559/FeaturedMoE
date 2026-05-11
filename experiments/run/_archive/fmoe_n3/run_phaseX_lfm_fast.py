#!/usr/bin/env python3
"""Launch FMoE_N3 LastFM fast transfer phase (12 fixed combos).

This phase is designed to transfer Kuai-validated axes to LastFM quickly:
- exactly 12 distinct combos
- deterministic 4-GPU round-robin assignment (3 runs per GPU)
- includes control baselines and MoE feature-off control
"""

from __future__ import annotations

import argparse
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
AXIS = "phaseX_lfm_fast_v1"
PHASE_DEFAULT = "LFMFAST12"


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
    uniq: list[str] = []
    for fam in families:
        f = str(fam).strip()
        if f and f not in uniq:
            uniq.append(f)
    return {"macro": list(uniq), "mid": list(uniq), "micro": list(uniq)}


def _profile_for_dataset(dataset: str) -> dict[str, Any]:
    key = dataset.lower()
    if key == "lastfm0.03":
        return {
            "batch_size": 2048,
            "max_item_list_length": 20,
            "search_learning_rate": [3e-4, 6e-3],
            "search_lr_scheduler_type": ["warmup_cosine"],
        }
    return {
        "batch_size": 4096,
        "max_item_list_length": 20,
        "search_learning_rate": [3e-4, 6e-3],
        "search_lr_scheduler_type": ["warmup_cosine"],
    }


def _base_row(
    dataset: str,
    combo_id: str,
    category: str,
    combo_desc: str,
    run_phase: str,
    seed_offset: int,
    fixed_dropout: float,
    fixed_weight_decay: float,
) -> dict[str, Any]:
    prof = _profile_for_dataset(dataset)
    return {
        "dataset": dataset,
        "seed_offset": int(seed_offset),
        "category": category,
        "combo_id": combo_id,
        "combo_desc": combo_desc,
        "run_phase": run_phase,
        "feature_mode": "full_v3",
        "max_item_list_length": int(prof["max_item_list_length"]),
        "batch_size": int(prof["batch_size"]),
        "fixed_hidden_dropout_prob": float(fixed_dropout),
        "fixed_weight_decay": float(fixed_weight_decay),
        "layer_layout": ["macro", "mid", "micro"],
        "search_learning_rate": list(prof["search_learning_rate"]),
        "search_lr_scheduler_type": list(prof["search_lr_scheduler_type"]),
        "overrides": {},
    }


def build_rows(dataset: str, fixed_dropout: float, fixed_weight_decay: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # C0: SASRec-equivalent internal control (plain path).
    r = _base_row(dataset, "C0", "control", "sasrec_equiv_plain", "LFMFAST_C0", 1, fixed_dropout, fixed_weight_decay)
    r["layer_layout"] = ["layer", "layer", "layer"]
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("none"),
            "stage_router_mode": _all_stage_map("none"),
            "stage_router_source": _all_stage_map("hidden"),
            "stage_feature_injection": _all_stage_map("none"),
            "stage_router_type": _all_stage_map("standard"),
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "expert_scale": 1,
        }
    )
    rows.append(r)

    # C1: MoE on but no feature usage control.
    r = _base_row(dataset, "C1", "control", "moe_hidden_only_no_feature", "LFMFAST_C1", 2, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("hidden"),
            "stage_feature_injection": _all_stage_map("none"),
            "stage_router_type": _all_stage_map("standard"),
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "expert_scale": 3,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    # Kuai-transfer top-k style variants.
    r = _base_row(dataset, "C2", "transfer_topk", "k_12e_top6_transfer", "LFMFAST_C2", 3, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("gated_bias"),
            "stage_router_type": _all_stage_map("standard"),
            "expert_scale": 3,
            "moe_top_k": 6,
            "topk_scope_mode": "global_flat",
            "balance_loss_lambda": 0.004,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C3", "transfer_topk", "k_12e_top3_transfer", "LFMFAST_C3", 4, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("gated_bias"),
            "stage_router_type": _all_stage_map("standard"),
            "expert_scale": 3,
            "moe_top_k": 3,
            "topk_scope_mode": "global_flat",
            "balance_loss_lambda": 0.004,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C4", "transfer_topk", "k_group_dense_transfer", "LFMFAST_C4", 5, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("group_gated_bias"),
            "stage_router_type": _all_stage_map("factored"),
            "expert_scale": 3,
            "moe_top_k": 0,
            "topk_scope_mode": "group_dense",
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "balance_loss_lambda": 0.004,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    # Phase6-style candidates.
    r = _base_row(dataset, "C5", "transfer_candidate", "cand_A_standard_gated", "LFMFAST_C5", 6, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("gated_bias"),
            "stage_router_type": _all_stage_map("standard"),
            "stage_residual_mode": _all_stage_map("base"),
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "expert_scale": 3,
            "route_smoothness_lambda": 0.01,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C6", "transfer_candidate", "cand_B_factored_group_gated", "LFMFAST_C6", 7, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("group_gated_bias"),
            "stage_router_type": _all_stage_map("factored"),
            "stage_residual_mode": _all_stage_map("base"),
            "topk_scope_mode": "group_dense",
            "moe_top_k": 0,
            "expert_scale": 3,
            "route_smoothness_lambda": 0.01,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C7", "transfer_candidate", "cand_C_warmup_residual", "LFMFAST_C7", 8, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("group_gated_bias"),
            "stage_router_type": _all_stage_map("factored"),
            "stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"),
            "topk_scope_mode": "group_dense",
            "moe_top_k": 0,
            "expert_scale": 3,
            "fmoe_schedule_enable": True,
            "alpha_warmup_until": 0.15,
            "alpha_warmup_start": 0.0,
            "alpha_warmup_end": 1.0,
            "residual_alpha_init": _all_stage_float(-2.0),
            "route_smoothness_lambda": 0.01,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "route_consistency_lambda": 0.01,
            "route_prior_lambda": 0.005,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    # Contrast variants around top-k/scope.
    r = _base_row(dataset, "C8", "contrast", "topk_scope_contrast_low", "LFMFAST_C8", 9, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("group_gated_bias"),
            "stage_router_type": _all_stage_map("factored"),
            "topk_scope_mode": "group_top1_pergroup",
            "moe_top_k": 1,
            "expert_scale": 3,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "balance_loss_lambda": 0.004,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C9", "contrast", "topk_scope_contrast_high", "LFMFAST_C9", 10, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("both"),
            "stage_feature_injection": _all_stage_map("group_gated_bias"),
            "stage_router_type": _all_stage_map("factored"),
            "topk_scope_mode": "group_top2_pergroup",
            "moe_top_k": 2,
            "expert_scale": 3,
            "group_prior_align_lambda": 5e-4,
            "factored_group_balance_lambda": 1e-3,
            "balance_loss_lambda": 0.004,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    # Feature-role controls.
    r = _base_row(dataset, "C10", "feature_role", "injection_only_control", "LFMFAST_C10", 11, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("hidden"),
            "stage_feature_injection": _all_stage_map("gated_bias"),
            "stage_router_type": _all_stage_map("standard"),
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "expert_scale": 3,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
        }
    )
    rows.append(r)

    r = _base_row(dataset, "C11", "feature_role", "feature_only_control", "LFMFAST_C11", 12, fixed_dropout, fixed_weight_decay)
    r["overrides"].update(
        {
            "stage_compute_mode": _all_stage_map("moe"),
            "stage_router_mode": _all_stage_map("learned"),
            "stage_router_source": _all_stage_map("feature"),
            "stage_feature_injection": _all_stage_map("none"),
            "stage_router_type": _all_stage_map("standard"),
            "topk_scope_mode": "global_flat",
            "moe_top_k": 0,
            "expert_scale": 3,
            "balance_loss_lambda": 0.002,
            "z_loss_lambda": 1e-4,
            "stage_feature_family_mask": _mask_all(["Tempo", "Memory", "Focus", "Exposure"]),
        }
    )
    rows.append(r)

    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['combo_id'])}_"
        f"{sanitize_slug(row['combo_desc'])}_"
        f"{ts}"
    )


def log_path(row: dict[str, Any], dataset: str, phase_name: str) -> Path:
    dataset_tag = dataset.replace("/", "_")
    model_tag = "FMoEN3"
    root = LOG_ROOT / AXIS / phase_name / dataset_tag / model_tag
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
        f"feature_mode={row['feature_mode']}",
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
        f"++fmoe_phase={hydra_literal(args.phase_name)}",
        f"MAX_ITEM_LIST_LENGTH={int(row['max_item_list_length'])}",
        f"train_batch_size={int(row['batch_size'])}",
        f"eval_batch_size={int(row['batch_size'])}",
        "embedding_size=128",
        "num_heads=4",
        f"hidden_dropout_prob={hydra_literal(row['fixed_hidden_dropout_prob'])}",
        "attn_dropout_prob=0.1",
        "d_ff=256",
        "d_feat_emb=16",
        "d_expert_hidden=128",
        "d_router_hidden=64",
        "++search_space_type_overrides.learning_rate=loguniform",
        f"++lfm_fast_combo_id={hydra_literal(row['combo_id'])}",
        f"++lfm_fast_combo_desc={hydra_literal(row['combo_desc'])}",
        f"++lfm_fast_category={hydra_literal(row['category'])}",
        f"++layer_layout={hydra_literal(row['layer_layout'])}",
        f"++search.learning_rate={hydra_literal(row['search_learning_rate'])}",
        f"++search.weight_decay={hydra_literal([row['fixed_weight_decay']])}",
        f"++search.hidden_dropout_prob={hydra_literal([row['fixed_hidden_dropout_prob']])}",
        f"++search.lr_scheduler_type={hydra_literal(row['search_lr_scheduler_type'])}",
    ]

    for key, value in row.get("overrides", {}).items():
        if isinstance(value, dict):
            cmd.append(f"++{key}={hydra_literal(value)}")
        else:
            if key in {
                "topk_scope_mode",
                "stage_feature_family_mask",
                "stage_compute_mode",
                "stage_router_mode",
                "stage_router_source",
                "stage_router_type",
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
        f"[{args.phase_name}_COMBO_HEADER]",
        f"run_phase={row['run_phase']} category={row['category']} combo={row['combo_id']} desc={row['combo_desc']}",
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
    parser = argparse.ArgumentParser(description="FMoE_N3 LastFM fast transfer 12-combo launcher")
    parser.add_argument("--dataset", default="lastfm0.03")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=19000)
    parser.add_argument("--phase-name", default=PHASE_DEFAULT)
    parser.add_argument("--fixed-dropout", type=float, default=0.15)
    parser.add_argument("--fixed-weight-decay", type=float, default=1e-6)
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

    rows = build_rows(args.dataset, args.fixed_dropout, args.fixed_weight_decay)

    if args.category:
        allowed = {tok.strip().lower() for tok in args.category.split(",") if tok.strip()}
        rows = [r for r in rows if str(r.get("category", "")).lower() in allowed]

    if args.only:
        allowed = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        rows = [r for r in rows if r["run_phase"] in allowed]

    if not rows:
        raise SystemExit("No runs selected")

    for idx, row in enumerate(rows):
        row["run_index"] = idx + 1
        row["assigned_order"] = idx + 1
        row["assigned_gpu"] = gpus[idx % len(gpus)]

    if args.manifest_out:
        mp = Path(args.manifest_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase": args.phase_name,
            "axis": AXIS,
            "dataset": args.dataset,
            "max_evals": int(args.max_evals),
            "tune_epochs": int(args.tune_epochs),
            "tune_patience": int(args.tune_patience),
            "fixed_dropout": float(args.fixed_dropout),
            "fixed_weight_decay": float(args.fixed_weight_decay),
            "rows": rows,
        }
        mp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"[lfm_fast] dataset={args.dataset} phase={args.phase_name} runs={len(rows)} gpus={','.join(gpus)} "
        f"fixed_dropout={args.fixed_dropout} fixed_weight_decay={args.fixed_weight_decay}"
    )

    if args.dry_run:
        for row in rows:
            lp = log_path(row, args.dataset, args.phase_name)
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
            lp = log_path(row, args.dataset, args.phase_name)
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

    print(f"[done] {args.phase_name} queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
