#!/usr/bin/env python3
"""Launch FMoE_N3 Phase5 specialization-focused aux/reg experiments.

Phase5 design:
- Methods (5): baseline, smoothness, consistency, sharp_mono, soft_prior
- Combos (8): C0..C7 (baseline-centered variants incl. one outlier)
- Total: 40 runs per dataset
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
AXIS = "phase5_specialization_v1"
PHASE = "P5"


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


def _method_rows() -> list[dict[str, Any]]:
    return [
        {
            "method_id": "M0",
            "method_slug": "baseline",
            "desc": "baseline",
            "aux": {},
        },
        {
            "method_id": "M1",
            "method_slug": "smoothness",
            "desc": "smoothness",
            "aux": {
                "route_smoothness_lambda": 0.02,
                "route_smoothness_stage_weight": {"macro": 1.0, "mid": 0.5, "micro": 0.1},
            },
        },
        {
            "method_id": "M2",
            "method_slug": "consistency",
            "desc": "consistency",
            "aux": {
                "route_consistency_lambda": 0.02,
                "route_consistency_pairs": 4,
            },
        },
        {
            "method_id": "M3",
            "method_slug": "sharp_mono",
            "desc": "sharp_but_not_monopoly",
            "aux": {
                "route_sharpness_lambda": 0.005,
                "route_monopoly_lambda": 0.02,
                "route_monopoly_tau": 0.22,
            },
        },
        {
            "method_id": "M4",
            "method_slug": "soft_prior",
            "desc": "soft_prior",
            "aux": {
                "route_prior_lambda": 0.01,
                "route_prior_bias_scale": 0.5,
            },
        },
    ]


def _combo_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "combo_id": "C0",
            "combo_desc": "group_dense_base",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "topk_scope_mode": "group_dense",
                "stage_residual_mode": _all_stage_map("base"),
            },
        },
        {
            "combo_id": "C1",
            "combo_desc": "group_dense_lowtemp",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "topk_scope_mode": "group_dense",
                "stage_residual_mode": _all_stage_map("base"),
                "mid_router_temperature": 1.0,
                "micro_router_temperature": 1.0,
            },
        },
        {
            "combo_id": "C2",
            "combo_desc": "group_dense_fastwarm",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "topk_scope_mode": "group_dense",
                "stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"),
                "fmoe_schedule_enable": True,
                "alpha_warmup_until": 0.15,
                "alpha_warmup_start": 0.0,
                "alpha_warmup_end": 1.0,
                "residual_alpha_init": _all_stage_float(-2.0),
            },
        },
        {
            "combo_id": "C3",
            "combo_desc": "group_dense_fastwarm_lowtemp",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "topk_scope_mode": "group_dense",
                "stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"),
                "fmoe_schedule_enable": True,
                "alpha_warmup_until": 0.15,
                "alpha_warmup_start": 0.0,
                "alpha_warmup_end": 1.0,
                "residual_alpha_init": _all_stage_float(-2.0),
                "mid_router_temperature": 1.0,
                "micro_router_temperature": 1.0,
            },
        },
        {
            "combo_id": "C4",
            "combo_desc": "dense12_base",
            "overrides": {
                "stage_router_type": _all_stage_map("standard"),
                "stage_feature_injection": _all_stage_map("gated_bias"),
                "topk_scope_mode": "global_flat",
                "stage_residual_mode": _all_stage_map("base"),
            },
        },
        {
            "combo_id": "C5",
            "combo_desc": "dense12_fastwarm",
            "overrides": {
                "stage_router_type": _all_stage_map("standard"),
                "stage_feature_injection": _all_stage_map("gated_bias"),
                "topk_scope_mode": "global_flat",
                "stage_residual_mode": _all_stage_map("shared_moe_learned_warmup"),
                "fmoe_schedule_enable": True,
                "alpha_warmup_until": 0.15,
                "alpha_warmup_start": 0.0,
                "alpha_warmup_end": 1.0,
                "residual_alpha_init": _all_stage_float(-2.0),
            },
        },
        {
            "combo_id": "C6",
            "combo_desc": "group_top2_base",
            "overrides": {
                "stage_router_type": _all_stage_map("factored"),
                "stage_feature_injection": _all_stage_map("group_gated_bias"),
                "topk_scope_mode": "group_top2_pergroup",
                "moe_top_k": 2,
                "stage_residual_mode": _all_stage_map("base"),
            },
        },
        {
            "combo_id": "C7",
            "combo_desc": "dense12_lowtemp_outlier",
            "overrides": {
                "stage_router_type": _all_stage_map("standard"),
                "stage_feature_injection": _all_stage_map("gated_bias"),
                "topk_scope_mode": "global_flat",
                "stage_residual_mode": _all_stage_map("base"),
                "mid_router_temperature": 0.9,
                "micro_router_temperature": 0.9,
            },
        },
    ]
    return rows


def build_rows(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = 0
    for method in _method_rows():
        for combo in _combo_rows():
            method_slug = method["method_slug"]
            combo_id = combo["combo_id"]
            run_phase = f"P5_{method['method_id']}_{combo_id}"
            row = {
                "dataset": dataset,
                "method_id": method["method_id"],
                "method_slug": method_slug,
                "method_desc": method["desc"],
                "combo_id": combo_id,
                "combo_desc": combo["combo_desc"],
                "aux": dict(method.get("aux", {})),
                "combo_overrides": dict(combo.get("overrides", {})),
                "run_phase": run_phase,
                "seed_offset": seed,
            }
            rows.append(row)
            seed += 1
    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['method_slug'])}_"
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


def build_command(row: dict[str, Any], gpu_id: str, args) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name", "config",
        "--max-evals", str(args.max_evals),
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
        "fmoe_best_only_logging=true",
        "fmoe_special_logging=true",
        f"fmoe_feature_ablation_logging={hydra_literal(bool(args.feature_ablation_logging))}",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE)}",
        "MAX_ITEM_LIST_LENGTH=20",
        "train_batch_size=4096",
        "eval_batch_size=4096",
        "embedding_size=128",
        "num_heads=4",
        "attn_dropout_prob=0.1",
        "d_ff=256",
        "d_feat_emb=16",
        "d_expert_hidden=128",
        "d_router_hidden=64",
        "expert_scale=3",
        "moe_top_k=0",
        "balance_loss_lambda=0.002",
        "z_loss_lambda=0.0",
        "gate_entropy_lambda=0.0",
        "group_prior_align_lambda=0.0005",
        "factored_group_balance_lambda=0.001",
        "feature_group_bias_lambda=0.001",
        "feature_group_prior_temperature=0.8",
        "route_smoothness_lambda=0.0",
        "route_consistency_lambda=0.0",
        "route_sharpness_lambda=0.0",
        "route_monopoly_lambda=0.0",
        "route_prior_lambda=0.0",
        "++layer_layout=[macro,mid,micro]",
        "++stage_router_type={macro:factored,mid:factored,micro:factored}",
        "++stage_feature_injection={macro:group_gated_bias,mid:group_gated_bias,micro:group_gated_bias}",
        "++topk_scope_mode=group_dense",
        "++stage_residual_mode={macro:base,mid:base,micro:base}",
        f"++method_slug={hydra_literal(row['method_slug'])}",
        f"++combo_id={hydra_literal(row['combo_id'])}",
        f"++combo_desc={hydra_literal(row['combo_desc'])}",
        f"++search.learning_rate={hydra_literal([2.0e-4, 2.0e-3])}",
        f"++search.weight_decay={hydra_literal([1e-7, 1e-6, 1e-5, 5e-5])}",
        f"++search.hidden_dropout_prob={hydra_literal([0.1, 0.15, 0.2])}",
        f"++search.lr_scheduler_type={hydra_literal(['warmup_cosine'])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
    ]

    for key, value in row.get("aux", {}).items():
        cmd.append(f"{key}={hydra_literal(value)}")
    for key, value in row.get("combo_overrides", {}).items():
        if isinstance(value, dict):
            cmd.append(f"++{key}={hydra_literal(value)}")
        else:
            if key in {"topk_scope_mode"}:
                cmd.append(f"++{key}={hydra_literal(value)}")
            else:
                cmd.append(f"{key}={hydra_literal(value)}")
    return cmd


def write_log_preamble(log_file: Path, row: dict[str, Any], gpu_id: str, args, cmd: list[str]) -> None:
    lines = [
        "[PHASE5_COMBO_HEADER]",
        (
            f"run_phase={row['run_phase']} method={row['method_slug']} "
            f"combo={row['combo_id']} desc={row['combo_desc']}"
        ),
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
    parser = argparse.ArgumentParser(description="FMoE_N3 phase5 specialization launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=13000)
    parser.add_argument("--feature-ablation-logging", dest="feature_ablation_logging", action="store_true")
    parser.add_argument("--no-feature-ablation-logging", dest="feature_ablation_logging", action="store_false")
    parser.set_defaults(feature_ablation_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated run_phase values")
    parser.add_argument("--method", default="", help="Comma-separated method slugs")
    parser.add_argument("--combo", default="", help="Comma-separated combo ids (C0..C7)")
    parser.add_argument("--manifest-out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided")

    rows = build_rows(args.dataset)
    if args.method:
        allowed = {tok.strip() for tok in args.method.split(",") if tok.strip()}
        rows = [r for r in rows if r["method_slug"] in allowed]
    if args.combo:
        allowed = {tok.strip().upper() for tok in args.combo.split(",") if tok.strip()}
        rows = [r for r in rows if r["combo_id"].upper() in allowed]
    if args.only:
        allowed = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        rows = [r for r in rows if r["run_phase"] in allowed]

    if not rows:
        raise SystemExit("No phase5 runs selected")

    for idx, row in enumerate(rows):
        row["assigned_order"] = idx + 1
        row["assigned_gpu"] = gpus[idx % len(gpus)]

    if args.manifest_out:
        mp = Path(args.manifest_out)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps({"phase": PHASE, "axis": AXIS, "rows": rows}, indent=2), encoding="utf-8")

    print(f"[phase5] dataset={args.dataset} runs={len(rows)} gpus={','.join(gpus)}")
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
            env["FMOE_UNIFIED_LOGGING_LAYOUT"] = "1"
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

    print("[done] phase5 specialization queue completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
