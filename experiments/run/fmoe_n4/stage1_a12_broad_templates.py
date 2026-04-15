#!/usr/bin/env python3
"""FMoE_N4: Stage1-only broad A12 tuning with 8/16 lightweight templates.

Default dataset is KuaiRec; templates prioritize fast lr-centric tuning while
keeping A12 structure fixed. The template bank is biased toward v3 A12 results:
most runs stay at len 20, keep lr in the 1e-4 to low-1e-3 band, and avoid the
len 40/50 or 2.7e-3 regimes that were repeatedly unstable or OOM-prone.
Only the core aux terms remain active:
- route_consistency_lambda
- z_loss_lambda
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

THIS_DIR = Path(__file__).resolve().parent
FMOE_N3_DIR = THIS_DIR.parent / "fmoe_n3"
if str(FMOE_N3_DIR) not in sys.path:
    sys.path.append(str(FMOE_N3_DIR))

import run_final_all_datasets as base  # noqa: E402
from run_phase9_auxloss import (  # noqa: E402
    _apply_base_overrides,
    _base_fixed_overrides,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)
from run_phase_wide_common import (  # noqa: E402
    build_summary_fieldnames,
    launch_wide_rows,
    sanitize_token,
)

TRACK = "fmoe_n4"
AXIS = "Stage1_A12_BroadTemplates"
AXIS_ID = "N4S1A12"
AXIS_DESC = "stage1_a12_broad_templates"
ARCH_ID = "A12"
ARCH_KEY = "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5"
ARCH_NAME = "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5"
PHASE_ID = "P4S1"
PHASE_NAME = "FMOE_N4_STAGE1_A12_BROAD"
DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2"]

REPO_ROOT_REAL = THIS_DIR.parents[2]
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _dedupe_keep_order(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        token = json.dumps(value, ensure_ascii=True, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        out.append(value)
    return out


def _validate_session_fixed_files(dataset: str) -> None:
    ds_dir = REPO_ROOT_REAL / "Datasets" / "processed" / "feature_added_v4" / dataset
    required = [
        ds_dir / f"{dataset}.train.inter",
        ds_dir / f"{dataset}.valid.inter",
        ds_dir / f"{dataset}.test.inter",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"session_fixed files missing for dataset={dataset}: {missing}")


def _anchor_cfg(hid: str) -> Dict[str, Any]:
    return dict(base.HPARAM_BANK[str(hid)])


def _hidden_choices(anchor: str, mode: str) -> list[float]:
    center = float(_anchor_cfg(anchor)["fixed_hidden_dropout_prob"])
    if mode == "low":
        values = [max(0.06, center - 0.02), center]
    elif mode == "high":
        values = [center, min(0.22, center + 0.03)]
    else:
        values = [max(0.07, center - 0.02), center]
    return _dedupe_keep_order(round(float(v), 4) for v in values)


def _weight_decay_choices(anchor: str, scales: list[float]) -> list[float]:
    base_wd = float(_anchor_cfg(anchor)["fixed_weight_decay"])
    return _dedupe_keep_order(round(base_wd * float(scale), 12) for scale in scales)


def _choice_spec(values: Iterable[Any]) -> Dict[str, Any]:
    return {"type": "choice", "values": list(values)}


def _loguniform_spec(low: float, high: float) -> Dict[str, Any]:
    return {"type": "loguniform", "values": [float(low), float(high)]}


def _template_batches(template: Dict[str, Any], fixed_values: Dict[str, Any], args: argparse.Namespace) -> tuple[int, int, int]:
    train_batch = int(args.batch_size)
    eval_batch = int(args.eval_batch_size)
    max_evals = int(args.max_evals)

    max_len = int(template.get("len", fixed_values.get("MAX_ITEM_LIST_LENGTH", 20)))
    d_feat = int(template.get("d_feat", fixed_values.get("d_feat_emb", 16)))
    expert_scale = int(template.get("expert", fixed_values.get("expert_scale", 3)))

    heavy_score = 0
    if max_len >= 30:
        heavy_score += 2
    elif max_len <= 15:
        heavy_score -= 1
    if d_feat >= 20:
        heavy_score += 1
    if expert_scale >= 4:
        heavy_score += 1

    if heavy_score >= 3:
        return 2560, 4096, min(max_evals, 8)
    if heavy_score >= 1:
        return 3072, 4096, min(max_evals, 9)
    if heavy_score <= -1:
        return 5120, 6144, max_evals
    return train_batch, eval_batch, max_evals


def _build_overrides(cons_lambda: float, z_lambda: float, family_drop: float, feature_drop: float) -> Dict[str, Any]:
    base_cfg = {
        "wrapper_map": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
        "source_profile": "src_abc_feature",
        "bias_mode": "bias_both",
    }
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=0.0,
        rule_bias_scale=0.0,
    )
    overrides["layer_layout"] = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
    overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
    overrides["macro_history_window"] = 5
    overrides["route_consistency_pairs"] = 1
    overrides["route_consistency_min_sim"] = 0.995
    overrides["route_consistency_lambda"] = float(cons_lambda)
    overrides["z_loss_lambda"] = float(z_lambda)

    # Keep only core aux terms active.
    overrides["balance_loss_lambda"] = 0.0
    overrides["gate_entropy_lambda"] = 0.0
    overrides["route_smoothness_lambda"] = 0.0
    overrides["route_sharpness_lambda"] = 0.0
    overrides["route_monopoly_lambda"] = 0.0
    overrides["route_prior_lambda"] = 0.0
    overrides["group_prior_align_lambda"] = 0.0
    overrides["factored_group_balance_lambda"] = 0.0
    overrides["rule_agreement_lambda"] = 0.0
    overrides["group_coverage_lambda"] = 0.0
    overrides["feature_group_bias_lambda"] = 0.0
    overrides["rule_bias_scale"] = 0.0
    overrides["bias_mode"] = "none"

    # Defaults (actual trial values come from search space).
    overrides["stage_family_dropout_prob"] = _all_stage_map(float(family_drop))
    overrides["stage_feature_dropout_prob"] = _all_stage_map(float(feature_drop))
    overrides["stage_feature_dropout_scope"] = _all_stage_map("token")
    return overrides


def _template_bank_16() -> list[Dict[str, Any]]:
    # First 8: conservative follow-ups around the best v3 A12 regions.
    # Last 8: exploratory variants that still stay under the obvious OOM zones.
    return [
        {"id": "T01_lrlo_h7", "anchor": "H7", "lambda": (5e-4, 1e-4), "lr_bounds": (1.0e-4, 4.0e-4), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T02_lrmid_h7", "anchor": "H7", "lambda": (5e-4, 1e-4), "lr_bounds": (2.2e-4, 8.5e-4), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T03_lrhi_h7", "anchor": "H7", "lambda": (5e-4, 1e-4), "lr_bounds": (4.5e-4, 1.6e-3), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T04_lrwide_h7", "anchor": "H7", "lambda": (5e-4, 1e-4), "lr_bounds": (1.5e-4, 1.35e-3), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T05_capacity_h14_lo", "anchor": "H14", "lambda": (5e-4, 1e-4), "lr_bounds": (1.5e-4, 7.5e-4), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "low", "attn": [0.06, 0.08]},
        {"id": "T06_capacity_h14_hi", "anchor": "H14", "lambda": (5e-4, 1e-4), "lr_bounds": (3.0e-4, 1.25e-3), "len": 20, "d_feat": 20, "expert": 4, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "low", "attn": [0.06, 0.08]},
        {"id": "T07_reg_h2_lo", "anchor": "H2", "lambda": (8e-4, 2e-4), "lr_bounds": (1.5e-4, 8.0e-4), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [1.0, 2.0], "hidden_mode": "high", "attn": [0.10, 0.12]},
        {"id": "T08_reg_h2_mid", "anchor": "H2", "lambda": (8e-4, 2e-4), "lr_bounds": (3.0e-4, 1.2e-3), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [1.0, 2.0], "hidden_mode": "high", "attn": [0.10, 0.12]},
        {"id": "T09_feat24_h7", "anchor": "H7", "lambda": (5e-4, 1e-4), "lr_bounds": (2.5e-4, 1.1e-3), "len": 20, "d_feat": 24, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T10_expert2_h6", "anchor": "H6", "lambda": (5e-4, 1e-4), "lr_bounds": (2.5e-4, 1.1e-3), "len": 20, "d_feat": 16, "expert": 2, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T11_expert4_h11", "anchor": "H11", "lambda": (8e-4, 2e-4), "lr_bounds": (3.5e-4, 1.35e-3), "len": 20, "d_feat": 20, "expert": 4, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
        {"id": "T12_len30_h10", "anchor": "H10", "lambda": (5e-4, 1e-4), "lr_bounds": (2.0e-4, 9.0e-4), "len": 30, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0], "hidden_mode": "low", "attn": [0.06, 0.08]},
        {"id": "T13_len25_h5", "anchor": "H5", "lambda": (5e-4, 1e-4), "lr_bounds": (1.8e-4, 7.0e-4), "len": 25, "d_feat": 16, "expert": 3, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "low", "attn": [0.06, 0.08]},
        {"id": "T14_compact_h3", "anchor": "H3", "lambda": (8e-4, 2e-4), "lr_bounds": (2.2e-4, 1.0e-3), "len": 20, "d_feat": 12, "expert": 3, "wd_scales": [1.0, 2.0, 4.0], "hidden_mode": "high", "attn": [0.10, 0.12]},
        {"id": "T15_small_h8", "anchor": "H8", "lambda": (8e-4, 2e-4), "lr_bounds": (1.8e-4, 8.0e-4), "len": 20, "d_feat": 16, "expert": 3, "wd_scales": [1.0, 2.0, 4.0], "hidden_mode": "high", "attn": [0.10, 0.12]},
        {"id": "T16_fastwide_h11", "anchor": "H11", "lambda": (8e-4, 2e-4), "lr_bounds": (5.0e-4, 1.8e-3), "len": 20, "d_feat": 16, "expert": 2, "wd_scales": [0.5, 1.0, 2.0], "hidden_mode": "balanced", "attn": [0.08, 0.10]},
    ]


def _select_templates(n_templates: int) -> list[Dict[str, Any]]:
    bank = _template_bank_16()
    if int(n_templates) == 16:
        return bank
    # 8-template subset: 4 conservative + 4 exploratory.
    keep = {"T01_lrlo_h7", "T03_lrhi_h7", "T05_capacity_h14_lo", "T07_reg_h2_lo", "T09_feat24_h7", "T11_expert4_h11", "T12_len30_h10", "T15_small_h8"}
    out = [t for t in bank if str(t["id"]) in keep]
    if len(out) != 8:
        raise RuntimeError("template subset construction failed")
    return out


def _row(dataset: str, template: Dict[str, Any], seed_id: int, runtime_seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    anchor = str(template["anchor"])
    cfg = _anchor_cfg(anchor)
    template_id = str(template["id"])
    cons_lambda, z_lambda = template["lambda"]

    family_drop_default = 0.02 if str(template.get("hidden_mode", "balanced")) != "high" else 0.04
    feature_drop_default = 0.0
    overrides = _build_overrides(cons_lambda, z_lambda, family_drop_default, feature_drop_default)

    hidden_mode = str(template.get("hidden_mode", "balanced"))

    run_id = f"S1_{sanitize_token(dataset, upper=True)}_{sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
    run_phase = f"{PHASE_ID}_{run_id}"

    max_item_list_length = int(template["len"]) if "len" in template else int(cfg.get("MAX_ITEM_LIST_LENGTH", 20))
    d_feat_emb = int(template["d_feat"]) if "d_feat" in template else int(cfg.get("d_feat_emb", 16))
    expert_scale = int(template["expert"]) if "expert" in template else int(cfg.get("expert_scale", 3))

    fixed_values: Dict[str, Any] = {
        "embedding_size": int(cfg["embedding_size"]),
        "d_ff": int(cfg["d_ff"]),
        "d_expert_hidden": int(cfg["d_expert_hidden"]),
        "d_router_hidden": int(cfg["d_router_hidden"]),
        "MAX_ITEM_LIST_LENGTH": max_item_list_length,
        "d_feat_emb": d_feat_emb,
        "expert_scale": expert_scale,
        "lr_scheduler_type": "warmup_cosine",
        "num_heads": 4,
    }

    train_batch_size, eval_batch_size, max_evals = _template_batches(template, fixed_values, args)

    lr_low, lr_high = template["lr_bounds"]
    search_space: Dict[str, Any] = {
        "learning_rate": _loguniform_spec(float(lr_low), float(lr_high)),
        "hidden_dropout_prob": _choice_spec(_hidden_choices(anchor, hidden_mode)),
        "attn_dropout_prob": _choice_spec(float(v) for v in template["attn"]),
        "weight_decay": _choice_spec(_weight_decay_choices(anchor, list(template.get("wd_scales", [0.5, 1.0, 2.0])))),
    }

    return {
        "dataset": dataset,
        "phase_id": PHASE_ID,
        "axis_id": AXIS_ID,
        "axis_desc": AXIS_DESC,
        "architecture_id": ARCH_ID,
        "architecture_key": ARCH_KEY,
        "architecture_name": ARCH_NAME,
        "exp_brief": ARCH_NAME,
        "run_phase": run_phase,
        "run_id": run_id,
        "setting_id": template_id,
        "setting_key": template_id,
        "setting_desc": template_id,
        "stage": "stage1",
        "tuning_stage": "stage1",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": "broad",
        "capacity_anchor": anchor,
        "selected_from_stage": "manual_template_bank",
        "selection_score": "",
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "stage1",
        "source_family_id": template_id,
        "template_count": int(args.template_count),
        "aux_route_consistency_lambda": float(cons_lambda),
        "aux_z_loss_lambda": float(z_lambda),
        "fixed_values": fixed_values,
        "search_space": search_space,
        "overrides": overrides,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "max_evals": int(max_evals),
        "tune_epochs": int(args.tune_epochs),
        "tune_patience": int(args.tune_patience),
    }


def build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    datasets = _parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets selected")
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")

    templates = _select_templates(int(args.template_count))

    rows: list[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        _validate_session_fixed_files(dataset)
        for template in templates:
            for seed_id in seeds:
                cursor += 1
                rows.append(
                    _row(
                        dataset=dataset,
                        template=template,
                        seed_id=int(seed_id),
                        runtime_seed=int(args.seed_base) + cursor - 1,
                        args=args,
                    )
                )
    return rows


def _serialize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "dataset",
        "phase_id",
        "axis_id",
        "axis_desc",
        "architecture_id",
        "architecture_key",
        "architecture_name",
        "exp_brief",
        "run_phase",
        "run_id",
        "setting_id",
        "setting_key",
        "setting_desc",
        "stage",
        "tuning_stage",
        "family_id",
        "family_group",
        "variant_id",
        "capacity_anchor",
        "selected_from_stage",
        "selection_score",
        "search_algo",
        "seed_id",
        "runtime_seed",
        "stage_group",
        "source_family_id",
        "template_count",
        "aux_route_consistency_lambda",
        "aux_z_loss_lambda",
        "fixed_values",
        "search_space",
        "overrides",
        "train_batch_size",
        "eval_batch_size",
        "max_evals",
        "tune_epochs",
        "tune_patience",
    }
    return {key: row.get(key) for key in keep if key in row}


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        raw = Path(str(args.manifest_out))
        if raw.suffix:
            return raw
        return raw / "stage1_manifest.json"
    return LOG_ROOT / "stage1_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "stage1",
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_count": len(rows),
        "datasets": sorted({str(row.get("dataset", "")) for row in rows}),
        "rows": [_serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summary_path(dataset: str) -> Path:
    path = LOG_ROOT / str(dataset) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_log_path(*, log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    dataset_dir = log_dir / str(row["dataset"])
    template_dir = dataset_dir / str(row["family_id"])
    filename = f"{phase_id}_{sanitize_token(str(row['family_id']), upper=True)}_S{int(row['seed_id'])}.log"
    return template_dir / filename


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = str(Path("/venv/FMoE/bin/python"))
    if not Path(python_bin).exists():
        python_bin = sys.executable

    fixed_values = dict(row.get("fixed_values", {}) or {})
    search_space = dict(row.get("search_space", {}) or {})

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(row["search_algo"]),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        str(int(row["tune_epochs"])),
        "--tune-patience",
        str(int(row["tune_patience"])),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        "feature_mode=full_v4",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(ARCH_ID)}",
        f"++fmoe_architecture_key={hydra_literal(ARCH_KEY)}",
        f"++fmoe_hparam_id={hydra_literal(row['capacity_anchor'])}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"train_batch_size={int(row['train_batch_size'])}",
        f"eval_batch_size={int(row['eval_batch_size'])}",
        f"++phase_run_type={hydra_literal(str(row.get('tuning_stage', row.get('stage', 'stage1'))))}",
        f"++phase_axis_id={hydra_literal(AXIS_ID)}",
        f"++phase_axis_desc={hydra_literal(AXIS_DESC)}",
        f"++phase_setting_id={hydra_literal(str(row['family_id']))}",
        f"++phase_setting_key={hydra_literal(str(row['family_id']))}",
        f"++phase_hparam_id={hydra_literal(str(row['capacity_anchor']))}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(str(row['run_id']))}",
    ]

    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, value in fixed_values.items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, spec in search_space.items():
        if isinstance(spec, dict):
            values = list(spec.get("values", []))
            search_type = str(spec.get("type", "choice"))
        else:
            values = list(spec)
            search_type = "choice"
        cmd.append(f"++search.{key}={hydra_literal(values)}")
        cmd.append(f"++search_space_type_overrides.{key}={search_type}")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N4 Stage1-only A12 broad template tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="CSV datasets (default: KuaiRec)")
    parser.add_argument("--template-count", type=int, choices=[8, 16], default=16)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=260000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--tune-patience", type=int, default=5)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    args = parser.parse_args()
    if int(args.max_evals) < 1:
        raise RuntimeError("--max-evals must be >= 1")
    return args


def maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 2) or 2))])


def main() -> int:
    args = parse_args()
    rows = maybe_limit_smoke(build_rows(args), args)
    manifest = write_manifest(args, rows)
    print(f"[stage1] manifest -> {manifest}")

    fieldnames = build_summary_fieldnames(
        [
            "architecture_id",
            "architecture_name",
            "tuning_stage",
            "family_id",
            "family_group",
            "variant_id",
            "capacity_anchor",
            "selected_from_stage",
            "selection_score",
            "search_algo",
            "source_family_id",
            "stage_group",
            "template_count",
            "aux_route_consistency_lambda",
            "aux_z_loss_lambda",
        ]
    )

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    return int(
        launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=LOG_ROOT,
            summary_path=LOG_ROOT / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[
                col
                for col in fieldnames
                if col
                not in {
                    "global_best_valid_mrr20",
                    "run_best_valid_mrr20",
                    "run_phase",
                    "exp_brief",
                    "stage",
                    "trigger",
                    "dataset",
                    "seed_id",
                    "gpu_id",
                    "status",
                    "test_mrr20",
                    "n_completed",
                    "interrupted",
                    "special_ok",
                    "diag_ok",
                    "result_path",
                    "timestamp_utc",
                }
            ],
            build_command=build_command,
            build_log_path=build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: summary_path(str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
