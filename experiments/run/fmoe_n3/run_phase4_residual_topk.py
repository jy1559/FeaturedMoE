#!/usr/bin/env python3
"""Launch FMoE_N3 phase4 residual-first + top-k experiments.

Key behavior:
- Residual-first execution per GPU lane.
- Top-k lanes are router-family aware (flat lanes vs factored lanes).
- Each GPU executes its own queue sequentially with no cross-lane barrier.
- Per-run sidecar outputs are generated under results sidecar directory:
  - *.result.json
  - *.summary.json
  - *.analysis_bundle.json
"""

from __future__ import annotations

import argparse
import csv
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
RESULT_ROOT = ARTIFACT_ROOT / "results" / "fmoe_n3"

if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from common.phase_summary_csv import build_fmoe_n3_axis_summary, build_fmoe_n3_summaries
from fmoe_n3.update_artifact_views import build_artifact_views

TRACK = "fmoe_n3"
AXIS = "phase4_residual_topk_v2"
PHASE = "P4"
SUMMARY_REFRESH_SEC = 180.0


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


def _all_stage_map(value: str) -> dict:
    return {"macro": value, "mid": value, "micro": value}


def _all_stage_float(value: float) -> dict:
    v = float(value)
    return {"macro": v, "mid": v, "micro": v}


def _default_granularity() -> dict:
    return {"macro": "session", "mid": "session", "micro": "token"}


def _lane_defaults(lane: str) -> dict[str, Any]:
    lane_key = str(lane).upper()
    if lane_key == "C1":
        return {
            "router_type": "standard",
            "injection": "gated_bias",
            "balance": 0.004,
            "z": 1e-4,
            "group_prior_align": 0.0,
            "group_balance": 0.0,
            "wd": [1e-7, 1e-6, 1e-5, 5e-5],
            "drop": [0.10, 0.15, 0.20],
        }
    if lane_key == "C2":
        return {
            "router_type": "standard",
            "injection": "gated_bias",
            "balance": 0.004,
            "z": 1e-3,
            "group_prior_align": 0.0,
            "group_balance": 0.0,
            "wd": [1e-7, 1e-6, 1e-5, 5e-5],
            "drop": [0.10, 0.15, 0.20],
        }
    if lane_key == "C3":
        return {
            "router_type": "factored",
            "injection": "group_gated_bias",
            "balance": 0.006,
            "z": 1e-3,
            "group_prior_align": 5e-4,
            "group_balance": 1e-3,
            "wd": [1e-7, 1e-6, 1e-5, 5e-5],
            "drop": [0.10, 0.15, 0.20],
        }
    return {
        "router_type": "factored",
        "injection": "group_gated_bias",
        "balance": 0.008,
        "z": 2e-3,
        "group_prior_align": 1e-3,
        "group_balance": 2e-3,
        "wd": [1e-7, 1e-6, 1e-5, 5e-5],
        "drop": [0.10, 0.15, 0.20],
    }


def _make_common(
    *,
    dataset: str,
    lane: str,
    axis_tag: str,
    variation_slug: str,
    description: str,
    seed_offset: int,
) -> dict[str, Any]:
    lane_cfg = _lane_defaults(lane)
    return {
        "dataset": dataset,
        "axis_tag": axis_tag,
        "variation_slug": variation_slug,
        "description": description,
        "combo_lane": lane,
        "seed_offset": int(seed_offset),
        "feature_mode": "full_v3",
        "layer_layout": ["macro", "mid", "micro"],
        "stage_compute_mode": _all_stage_map("moe"),
        "stage_router_mode": _all_stage_map("learned"),
        "stage_router_source": _all_stage_map("both"),
        "stage_router_type": _all_stage_map(lane_cfg["router_type"]),
        "stage_feature_injection": _all_stage_map(lane_cfg["injection"]),
        "stage_router_granularity": _default_granularity(),
        "stage_feature_encoder_mode": _all_stage_map("linear"),
        "stage_residual_mode": _all_stage_map("base"),
        "residual_alpha_fixed": _all_stage_float(0.5),
        "residual_alpha_init": _all_stage_float(0.0),
        "residual_shared_ffn_scale": 1.0,
        "macro_history_window": 5,
        "expert_scale": 3,
        "moe_top_k": 0,
        "balance_loss_lambda": lane_cfg["balance"],
        "z_loss_lambda": lane_cfg["z"],
        "group_prior_align_lambda": lane_cfg["group_prior_align"],
        "factored_group_balance_lambda": lane_cfg["group_balance"],
        "search_learning_rate": [2.0e-4, 2.0e-3],
        "search_weight_decay": list(lane_cfg["wd"]),
        "search_hidden_dropout_prob": list(lane_cfg["drop"]),
        "search_lr_scheduler_type": ["warmup_cosine"],
        "run_phase": f"{axis_tag}_{variation_slug}_{lane}",
    }


def _build_residual_variants(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("base", "base", 0.5, 0.0),
        ("shared_only", "shared_only", 0.5, 0.0),
        ("shared_moe_fixed03", "shared_moe_fixed", 0.3, 0.0),
        ("shared_moe_fixed05", "shared_moe_fixed", 0.5, 0.0),
        ("shared_moe_global", "shared_moe_global", 0.5, 0.0),
        ("shared_moe_stage", "shared_moe_learned", 0.5, 0.0),
        ("shared_moe_warmup", "shared_moe_learned_warmup", 0.5, 0.0),
    ]
    seed = 0
    for lane in ("C1", "C2", "C3", "C4"):
        for slug, mode, alpha_fixed, alpha_init in specs:
            row = _make_common(
                dataset=dataset,
                lane=lane,
                axis_tag="R",
                variation_slug=slug,
                description=slug,
                seed_offset=seed,
            )
            row["stage_residual_mode"] = _all_stage_map(mode)
            row["residual_alpha_fixed"] = _all_stage_float(alpha_fixed)
            row["residual_alpha_init"] = _all_stage_float(alpha_init)
            rows.append(row)
            seed += 1
    return rows


def _build_topk_variants(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed = 1000

    specs = [
        ("4e_dense", 1, 0, "standard", "gated_bias", "global_flat"),
        ("4e_top1", 1, 1, "standard", "gated_bias", "global_flat"),
        ("4e_top2", 1, 2, "standard", "gated_bias", "global_flat"),
        ("12e_dense", 3, 0, "standard", "gated_bias", "global_flat"),
        ("12e_top3", 3, 3, "standard", "gated_bias", "global_flat"),
        ("12e_top6", 3, 6, "standard", "gated_bias", "global_flat"),
        ("group_dense", 3, 0, "factored", "group_gated_bias", "group_dense"),
        ("group_top1", 3, 1, "factored", "group_gated_bias", "group_top1_pergroup"),
        ("group_top2", 3, 2, "factored", "group_gated_bias", "group_top2_pergroup"),
        ("groupignore_global6", 3, 6, "factored", "group_gated_bias", "global_ignore_group"),
    ]
    for lane in ("C1", "C2", "C3", "C4"):
        for slug, expert_scale, topk, router_type, injection, scope in specs:
            row = _make_common(
                dataset=dataset,
                lane=lane,
                axis_tag="K",
                variation_slug=slug,
                description=slug,
                seed_offset=seed,
            )
            row["expert_scale"] = int(expert_scale)
            row["moe_top_k"] = int(topk)
            row["stage_router_type"] = _all_stage_map(router_type)
            row["stage_feature_injection"] = _all_stage_map(injection)
            row["topk_scope_mode"] = str(scope)
            rows.append(row)
            seed += 1

    return rows


def _build_feature_probe_variants(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # One feature-impact probe per lane (4 settings total, each exactly once).
    probe_specs = [
        # lane, slug, router_source, injection
        ("C1", "feat_full", "both", "gated_bias"),
        ("C2", "feat_hidden_only", "hidden", "none"),
        ("C3", "feat_injection_only", "hidden", "gated_bias"),
        ("C4", "feat_feature_only", "feature", "none"),
    ]
    seed = 3000
    for lane, slug, router_source, injection in probe_specs:
        row = _make_common(
            dataset=dataset,
            lane=lane,
            axis_tag="F",
            variation_slug=slug,
            description=slug,
            seed_offset=seed,
        )
        row["stage_residual_mode"] = _all_stage_map("base")
        row["moe_top_k"] = 0
        row["expert_scale"] = 3
        row["stage_router_type"] = _all_stage_map("standard")
        row["stage_router_source"] = _all_stage_map(router_source)
        row["stage_feature_injection"] = _all_stage_map(injection)
        row["topk_scope_mode"] = "global_flat"
        rows.append(row)
        seed += 1
    return rows


def build_combos(dataset: str) -> list[dict[str, Any]]:
    residual_rows = _build_residual_variants(dataset)
    topk_rows = _build_topk_variants(dataset)
    feature_rows = _build_feature_probe_variants(dataset)
    rows = residual_rows + topk_rows + feature_rows

    axis_order = {"R": 0, "K": 1, "F": 2}

    rows = sorted(
        rows,
        key=lambda r: (
            r["combo_lane"],
            axis_order.get(str(r["axis_tag"]).upper(), 9),
            r["seed_offset"],
        ),
    )

    for idx, row in enumerate(rows, start=1):
        row["run_index"] = idx
    return rows


def _make_log_stem(row: dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{int(row['run_index']):03d}_"
        f"{sanitize_slug(row['axis_tag'])}_"
        f"{sanitize_slug(row['description'])}_"
        f"{sanitize_slug(row['combo_lane'])}_"
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
        f"feature_mode={row['feature_mode']}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        f"fmoe_eval_logging_timing={args.eval_logging_timing}",
        f"fmoe_feature_ablation_logging={hydra_literal(bool(args.feature_ablation_logging))}",
        "fmoe_special_logging=true",
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
        f"expert_scale={row['expert_scale']}",
        f"++layer_layout={hydra_literal(row['layer_layout'])}",
        f"++stage_feature_encoder_mode={hydra_literal(row['stage_feature_encoder_mode'])}",
        f"++stage_compute_mode={hydra_literal(row['stage_compute_mode'])}",
        f"++stage_router_mode={hydra_literal(row['stage_router_mode'])}",
        f"++stage_router_source={hydra_literal(row['stage_router_source'])}",
        f"++stage_feature_injection={hydra_literal(row['stage_feature_injection'])}",
        f"++stage_router_type={hydra_literal(row['stage_router_type'])}",
        f"++stage_router_granularity={hydra_literal(row['stage_router_granularity'])}",
        f"++stage_residual_mode={hydra_literal(row['stage_residual_mode'])}",
        f"++residual_alpha_fixed={hydra_literal(row['residual_alpha_fixed'])}",
        f"++residual_alpha_init={hydra_literal(row['residual_alpha_init'])}",
        f"++residual_shared_ffn_scale={row['residual_shared_ffn_scale']}",
        f"macro_history_window={row['macro_history_window']}",
        f"moe_top_k={row['moe_top_k']}",
        f"++topk_scope_mode={hydra_literal(row.get('topk_scope_mode', 'global'))}",
        "moe_top_k_policy=auto",
        "moe_top_k_ratio=0.5",
        "macro_session_pooling=mean",
        f"balance_loss_lambda={row['balance_loss_lambda']}",
        f"z_loss_lambda={row['z_loss_lambda']}",
        f"group_prior_align_lambda={row['group_prior_align_lambda']}",
        f"factored_group_balance_lambda={row['factored_group_balance_lambda']}",
        f"++combo_lane={hydra_literal(row['combo_lane'])}",
        f"++axis_tag={hydra_literal(row['axis_tag'])}",
        f"++variation_slug={hydra_literal(row['variation_slug'])}",
        f"++combo_desc={hydra_literal(row['description'])}",
        f"++search.learning_rate={hydra_literal(row['search_learning_rate'])}",
        f"++search.weight_decay={hydra_literal(row['search_weight_decay'])}",
        f"++search.hidden_dropout_prob={hydra_literal(row['search_hidden_dropout_prob'])}",
        f"++search.lr_scheduler_type={hydra_literal(row['search_lr_scheduler_type'])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
    ]
    return cmd


def write_log_preamble(log_file: Path, row: dict[str, Any], gpu_id: str, args, cmd: list[str]) -> None:
    lines = [
        "[PHASE4_COMBO_HEADER]",
        (
            f"run_phase={row['run_phase']} axis={row['axis_tag']} variation={row['variation_slug']} "
            f"combo={row['combo_lane']} desc={row['description']}"
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


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _find_latest_result_payload(*, run_phase: str, dataset: str) -> tuple[dict[str, Any] | None, Path | None]:
    best_payload = None
    best_path = None
    best_mtime = -1.0
    for path in sorted(RESULT_ROOT.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_group", "")) != TRACK:
            continue
        if str(payload.get("run_axis", "")) != AXIS:
            continue
        if str(payload.get("run_phase", "")) != str(run_phase):
            continue
        if str(payload.get("dataset", "")) != str(dataset):
            continue
        mtime = float(path.stat().st_mtime)
        if mtime > best_mtime:
            best_mtime = mtime
            best_payload = payload
            best_path = path
    return best_payload, best_path


def _select_row(rows: list[dict[str, str]], *, filters: dict[str, str]) -> dict[str, str] | None:
    for row in rows:
        ok = True
        for key, expected in filters.items():
            if str(row.get(key, "")) != str(expected):
                ok = False
                break
        if ok:
            return row
    return None


def write_sidecar_bundle(
    *,
    row: dict[str, Any],
    log_file: Path,
    phase_summary_path: Path,
    axis_summary_path: Path,
    artifact_paths: dict[str, Path],
) -> None:
    phase_root = LOG_ROOT / AXIS / PHASE
    axis_root = LOG_ROOT / AXIS
    try:
        phase_rel = str(log_file.relative_to(phase_root))
    except Exception:
        phase_rel = ""
    try:
        axis_rel = str(log_file.relative_to(axis_root))
    except Exception:
        axis_rel = ""

    phase_rows = _read_csv_rows(phase_summary_path)
    axis_rows = _read_csv_rows(axis_summary_path)
    special_rows = _read_csv_rows(Path(artifact_paths.get("special_summary", "")))
    diag_rows = _read_csv_rows(Path(artifact_paths.get("diag_summary", "")))
    fab_rows = _read_csv_rows(Path(artifact_paths.get("feature_ablation_summary", "")))

    phase_row = _select_row(
        phase_rows,
        filters={"run_phase": row["run_phase"], "log_rel_path": phase_rel},
    )
    axis_row = _select_row(
        axis_rows,
        filters={"run_phase": row["run_phase"], "log_rel_path": axis_rel},
    )
    special_row = _select_row(
        special_rows,
        filters={"combo_id": row["run_phase"], "dataset": row["dataset"], "model": "FMoEN3"},
    )
    if special_row is None:
        special_row = _select_row(
            special_rows,
            filters={"combo_id": row["run_phase"], "dataset": row["dataset"], "model": "FeaturedMoE_N3"},
        )
    diag_row = _select_row(
        diag_rows,
        filters={"combo_id": row["run_phase"], "dataset": row["dataset"]},
    )
    fab_row = _select_row(
        fab_rows,
        filters={"combo_id": row["run_phase"], "dataset": row["dataset"]},
    )

    result_payload, result_path = _find_latest_result_payload(run_phase=row["run_phase"], dataset=row["dataset"])

    missing: dict[str, str] = {}
    if result_payload is None:
        missing["result"] = "not_found"
    if phase_row is None:
        missing["phase_summary"] = "not_found"
    if axis_row is None:
        missing["axis_summary"] = "not_found"
    if special_row is None:
        missing["special"] = "not_found_or_not_enabled"
    if diag_row is None:
        missing["diag"] = "not_found_or_not_enabled"
    if fab_row is None:
        missing["feature_ablation"] = "not_found_or_not_enabled"

    summary_payload = {
        "meta": {
            "track": TRACK,
            "axis": AXIS,
            "phase": PHASE,
            "run_phase": row["run_phase"],
            "dataset": row["dataset"],
            "axis_tag": row["axis_tag"],
            "variation_slug": row["variation_slug"],
            "combo_lane": row["combo_lane"],
            "log_file": str(log_file),
            "result_file": str(result_path) if result_path else "",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "availability": {
            "result_available": result_payload is not None,
            "phase_summary_available": phase_row is not None,
            "axis_summary_available": axis_row is not None,
            "special_available": special_row is not None,
            "diag_available": diag_row is not None,
            "feature_ablation_available": fab_row is not None,
        },
        "phase_summary": phase_row or {},
        "axis_summary": axis_row or {},
        "missing_sections": missing,
    }

    bundle_payload = {
        "meta": summary_payload["meta"],
        "summary": {
            "phase": phase_row or {},
            "axis": axis_row or {},
        },
        "result": result_payload or {},
        "diag": diag_row or {},
        "special": special_row or {},
        "feature_ablation": fab_row or {},
        "missing_sections": missing,
    }

    dataset_tag = sanitize_slug(str(row.get("dataset", "unknown")))
    model_tag = "FMoEN3"
    sidecar_root = RESULT_ROOT / "sidecar" / AXIS / PHASE / dataset_tag / model_tag
    sidecar_root.mkdir(parents=True, exist_ok=True)
    base = sidecar_root / log_file.stem
    if result_payload is not None:
        (base.with_suffix(".result.json")).write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (base.with_suffix(".summary.json")).write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base.with_suffix(".analysis_bundle.json")).write_text(
        json.dumps(bundle_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _migrate_log_dir_sidecars() -> None:
    """logs 디렉터리에 잘못 놓인 sidecar bundle JSON을 results/sidecar 로 이동.

    이전 버전의 write_sidecar_bundle이 log_file.parent 에 직접 써서 발생한 파일들을
    올바른 위치(RESULT_ROOT/sidecar/AXIS/PHASE/...)로 옮긴다.
    대상 위치에 이미 파일이 있으면 logs 쪽 사본을 삭제한다.
    """
    phase_log_dir = LOG_ROOT / AXIS / PHASE
    phase_sidecar_dir = RESULT_ROOT / "sidecar" / AXIS / PHASE
    sidecar_exts = (".result.json", ".summary.json", ".analysis_bundle.json")
    moved = 0
    removed = 0
    for json_path in sorted(phase_log_dir.rglob("*.json")):
        if not any(json_path.name.endswith(sfx) for sfx in sidecar_exts):
            continue
        try:
            rel = json_path.relative_to(phase_log_dir)
        except ValueError:
            continue
        target = phase_sidecar_dir / rel
        if target.exists():
            json_path.unlink()
            removed += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            json_path.rename(target)
            moved += 1
    if moved or removed:
        print(f"[migrate] logs→sidecar: moved={moved} dedup_removed={removed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FMoE_N3 phase4 residual-topk launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=12000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--axis-filter", default="", help="R, K, or F")
    parser.add_argument("--only", default="", help="Comma-separated run_phase values")
    parser.add_argument("--combo", default="", help="Comma-separated combo lanes (C1,C2,C3,C4)")
    parser.add_argument("--eval-logging-timing", default="final_only", choices=["final_only", "per_eval"])
    parser.add_argument("--feature-ablation-logging", action="store_true")
    args = parser.parse_args()

    gpus = [tok.strip() for tok in args.gpus.split(",") if tok.strip()]
    if not gpus:
        raise SystemExit("No GPUs provided")

    combos = build_combos(args.dataset)

    if args.axis_filter:
        allowed_axis = {tok.strip().upper() for tok in args.axis_filter.split(",") if tok.strip()}
        combos = [row for row in combos if row["axis_tag"].upper() in allowed_axis]
    if args.combo:
        allowed_combo = {tok.strip().upper() for tok in args.combo.split(",") if tok.strip()}
        combos = [row for row in combos if row["combo_lane"].upper() in allowed_combo]
    if args.only:
        allowed_phase = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        combos = [row for row in combos if row["run_phase"] in allowed_phase]

    if not combos:
        raise SystemExit("No combos selected")

    lane_to_gpu: dict[str, str] = {}
    lane_order = ["C1", "C2", "C3", "C4"]
    for idx, lane in enumerate(lane_order):
        lane_to_gpu[lane] = gpus[idx % len(gpus)]

    bins: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    for row in combos:
        gpu_id = lane_to_gpu.get(row["combo_lane"], gpus[0])
        bins[gpu_id].append(row)

    for gpu_id in gpus:
        bins[gpu_id] = sorted(
            bins[gpu_id],
            key=lambda r: (
                r["combo_lane"],
                0 if r["axis_tag"] == "R" else 1,
                r["seed_offset"],
            ),
        )
        for order, row in enumerate(bins[gpu_id], start=1):
            row["assigned_gpu"] = gpu_id
            row["assigned_order"] = order

    manifest = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE,
        "dataset": args.dataset,
        "gpus": gpus,
        "max_evals": args.max_evals,
        "tune_epochs": args.tune_epochs,
        "tune_patience": args.tune_patience,
        "combos": combos,
    }
    if args.manifest_out:
        out = Path(args.manifest_out)
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[manifest] {out}")

    _migrate_log_dir_sidecars()

    assignments: list[tuple[str, dict[str, Any], list[str], Path]] = []
    for gpu_id in gpus:
        for row in bins[gpu_id]:
            cmd = build_command(row, gpu_id, args)
            lp = log_path(row, args.dataset)
            write_log_preamble(lp, row, gpu_id, args, cmd)
            assignments.append((gpu_id, row, cmd, lp))

    if args.dry_run:
        print(f"[dry-run] total={len(assignments)}")
        for gpu_id, row, _, lp in assignments:
            print(f"  gpu={gpu_id} order={row['assigned_order']:02d} {row['run_phase']} -> {lp.name}")
        return

    per_gpu_queue: dict[str, deque] = {gpu: deque() for gpu in gpus}
    for gpu_id, row, cmd, lp in assignments:
        per_gpu_queue[gpu_id].append((row, cmd, lp))

    active: dict[str, dict[str, Any]] = {}
    last_summary_refresh = 0.0
    phase_summary_path = LOG_ROOT / AXIS / PHASE / args.dataset / f"{PHASE}_summary.csv"
    axis_summary_path = LOG_ROOT / AXIS / f"{AXIS}_summary.csv"

    def refresh_views(*, reason: str, force: bool = False) -> tuple[Path, Path, dict[str, Path]]:
        nonlocal last_summary_refresh
        now = time.time()
        if (not force) and (now - last_summary_refresh < SUMMARY_REFRESH_SEC):
            return phase_summary_path, axis_summary_path, {
                "special_summary": LOG_ROOT / AXIS / f"{AXIS}_special_summary.csv",
                "feature_ablation_summary": LOG_ROOT / AXIS / f"{AXIS}_feature_ablation_summary.csv",
                "diag_summary": LOG_ROOT / AXIS / f"{AXIS}_diag_summary.csv",
            }

        phase_paths = build_fmoe_n3_summaries(AXIS, PHASE)
        _ = phase_paths
        axis_path = build_fmoe_n3_axis_summary(AXIS)
        artifact_paths = build_artifact_views(AXIS)
        last_summary_refresh = now
        print(f"[refresh] {reason}: axis={axis_path}")
        return phase_summary_path, axis_summary_path, artifact_paths

    refresh_views(reason="start", force=True)

    while any(per_gpu_queue[gpu] for gpu in gpus) or active:
        for gpu_id in gpus:
            if gpu_id in active:
                proc = active[gpu_id]["proc"]
                if proc.poll() is None:
                    continue

                row = active[gpu_id]["row"]
                log_file = active[gpu_id]["log"]
                rc = int(proc.returncode)
                print(f"[done] gpu={gpu_id} phase={row['run_phase']} rc={rc}")

                p_path, a_path, artifact_paths = refresh_views(reason=f"complete:{row['run_phase']}", force=True)
                try:
                    write_sidecar_bundle(
                        row=row,
                        log_file=log_file,
                        phase_summary_path=p_path,
                        axis_summary_path=a_path,
                        artifact_paths=artifact_paths,
                    )
                except Exception as exc:
                    print(f"[bundle] failed for {row['run_phase']}: {exc}")

                del active[gpu_id]

            if gpu_id not in active and per_gpu_queue[gpu_id]:
                row, cmd, log_file = per_gpu_queue[gpu_id].popleft()
                existing_payload, existing_path = _find_latest_result_payload(
                    run_phase=row["run_phase"], dataset=row["dataset"]
                )
                if existing_payload is not None:
                    print(
                        f"[skip] gpu={gpu_id} {row['run_phase']} "
                        f"already completed ({existing_path.name if existing_path else '?'})"
                    )
                    continue
                print(f"[launch] gpu={gpu_id} order={row['assigned_order']:02d} {row['run_phase']}")
                with open(log_file, "a", encoding="utf-8") as lf:
                    proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), stdout=lf, stderr=lf)
                active[gpu_id] = {"proc": proc, "row": row, "log": log_file}

        refresh_views(reason="periodic", force=False)
        if active or any(per_gpu_queue[g] for g in gpus):
            time.sleep(5)

    refresh_views(reason="final", force=True)
    print("[done] phase4 residual+topk queue completed")


if __name__ == "__main__":
    main()
