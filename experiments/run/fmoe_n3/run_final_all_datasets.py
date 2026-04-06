#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 final all-datasets runs (session_fixed)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from run_phase9_auxloss import (  # noqa: E402
    LOG_ROOT,
    REPO_ROOT,
    TRACK,
    _apply_base_overrides,
    _base_fixed_overrides,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows, sanitize_token  # noqa: E402

AXIS = "Final_all_datasets"
PHASE_ID = "P14"
PHASE_NAME = "FINAL_ALL_DATASETS"
RUN_STAGE = "final"
AXIS_DESC = "final_all_datasets"

ARCH_ORDER = ("A1", "A2", "A3", "A4", "A5", "A6")
ARCH_METADATA: Dict[str, Dict[str, str]] = {
    "A1": {
        "arch_key": "FINAL_MAIN",
        "arch_name": "A1_MAIN_ATTN_MICRO_BEFORE",
        "setting_group": "final_main",
        "setting_key": "FINAL_MAIN_ATTN_MICRO_BEFORE",
        "setting_desc_prefix": "FINAL_MAIN_ATTN_MICRO_BEFORE",
        "setting_detail": "B4 + C0-N4 + family_dropout + macro_mid_micro + session_fixed",
    },
    "A2": {
        "arch_key": "A2_INTRA_NN_ZLOSS",
        "arch_name": "A2_INTRA_FEATURE_NN_STRICT_ZLOSS",
        "setting_group": "final_variant",
        "setting_key": "A2_INTRA_NN_ZLOSS",
        "setting_desc_prefix": "A2_INTRA_NN_ZLOSS",
        "setting_detail": "A1 + strict intra-feature NN consistency(top1, sim>=threshold) + z-loss only",
    },
    "A3": {
        "arch_key": "A3_NO_CATEGORY",
        "arch_name": "A3_NO_CATEGORY",
        "setting_group": "final_variant",
        "setting_key": "A3_NO_CATEGORY",
        "setting_desc_prefix": "A3_NO_CATEGORY",
        "setting_detail": "A1 + structural drop of category/theme-derived features",
    },
    "A4": {
        "arch_key": "A4_INTRA_GROUP_BIAS_GLS",
        "arch_name": "A4_INTRA_GROUP_BIAS_GLS",
        "setting_group": "final_variant",
        "setting_key": "A4_INTRA_GROUP_BIAS_GLS",
        "setting_desc_prefix": "A4_INTRA_GROUP_BIAS_GLS",
        "setting_detail": "A1 with rule/group bias off + direct intra-group GLS expert bias (macro+mid+micro)",
    },
    "A5": {
        "arch_key": "A5_NO_CATEGORY_NO_TIMESTAMP",
        "arch_name": "A5_NO_CATEGORY_NO_TIMESTAMP",
        "setting_group": "final_variant",
        "setting_key": "A5_NO_CATEGORY_NO_TIMESTAMP",
        "setting_desc_prefix": "A5_NO_CATEGORY_NO_TIMESTAMP",
        "setting_detail": "A1 + structural drop of category/theme and timestamp/pace/interval-derived features",
    },
    "A6": {
        "arch_key": "A6_NO_BIAS",
        "arch_name": "A6_NO_BIAS",
        "setting_group": "final_variant",
        "setting_key": "A6_NO_BIAS",
        "setting_desc_prefix": "A6_NO_BIAS",
        "setting_detail": "A1 + no_bias (bias_mode=none, rule/group bias off)",
    },
}

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "amazon_beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

HPARAM_BANK: Dict[str, Dict[str, float]] = {
    "H1": {
        "embedding_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H2": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 5e-7,
        "fixed_hidden_dropout_prob": 0.12,
    },
    "H3": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 2e-6,
        "fixed_hidden_dropout_prob": 0.18,
    },
    "H4": {
        "embedding_size": 112,
        "d_ff": 224,
        "d_expert_hidden": 112,
        "d_router_hidden": 56,
        "fixed_weight_decay": 3e-6,
        "fixed_hidden_dropout_prob": 0.20,
    },
    "H5": {
        "embedding_size": 168,
        "d_ff": 336,
        "d_expert_hidden": 168,
        "d_router_hidden": 84,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H6": {
        "embedding_size": 144,
        "d_ff": 288,
        "d_expert_hidden": 144,
        "d_router_hidden": 72,
        "fixed_weight_decay": 1.5e-6,
        "fixed_hidden_dropout_prob": 0.17,
    },
    "H7": {
        "embedding_size": 160,
        "d_ff": 320,
        "d_expert_hidden": 160,
        "d_router_hidden": 80,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.15,
    },
    "H8": {
        "embedding_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "fixed_weight_decay": 2.5e-6,
        "fixed_hidden_dropout_prob": 0.19,
    },
    "H9": {
        "embedding_size": 96,
        "d_ff": 192,
        "d_expert_hidden": 96,
        "d_router_hidden": 48,
        "fixed_weight_decay": 4e-6,
        "fixed_hidden_dropout_prob": 0.22,
    },
    "H10": {
        "embedding_size": 192,
        "d_ff": 384,
        "d_expert_hidden": 192,
        "d_router_hidden": 96,
        "fixed_weight_decay": 8e-7,
        "fixed_hidden_dropout_prob": 0.14,
    },
    "H11": {
        "embedding_size": 224,
        "d_ff": 448,
        "d_expert_hidden": 224,
        "d_router_hidden": 112,
        "fixed_weight_decay": 6e-7,
        "fixed_hidden_dropout_prob": 0.12,
    },
    "H12": {
        "embedding_size": 80,
        "d_ff": 160,
        "d_expert_hidden": 80,
        "d_router_hidden": 40,
        "fixed_weight_decay": 5e-6,
        "fixed_hidden_dropout_prob": 0.24,
    },
    "H13": {
        "embedding_size": 176,
        "d_ff": 352,
        "d_expert_hidden": 176,
        "d_router_hidden": 88,
        "fixed_weight_decay": 1e-6,
        "fixed_hidden_dropout_prob": 0.16,
    },
    "H14": {
        "embedding_size": 256,
        "d_ff": 512,
        "d_expert_hidden": 256,
        "d_router_hidden": 128,
        "fixed_weight_decay": 5e-7,
        "fixed_hidden_dropout_prob": 0.10,
    },
    "H15": {
        "embedding_size": 128,
        "d_ff": 384,
        "d_expert_hidden": 96,
        "d_router_hidden": 64,
        "fixed_weight_decay": 1.2e-6,
        "fixed_hidden_dropout_prob": 0.17,
    },
    "H16": {
        "embedding_size": 144,
        "d_ff": 216,
        "d_expert_hidden": 216,
        "d_router_hidden": 72,
        "fixed_weight_decay": 1.8e-6,
        "fixed_hidden_dropout_prob": 0.20,
    },
}

DEFAULT_OUTLIER_HPARAM = {
    "KuaiRecLargeStrictPosV2_0.2": "H4",
    "lastfm0.03": "H5",
    "amazon_beauty": "H8",
    "foursquare": "H2",
    "movielens1m": "H5",
    "retail_rocket": "H6",
}

DATASET_HPARAM_PRESET_12: Dict[str, list[str]] = {
    "KuaiRecLargeStrictPosV2_0.2": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H12", "H14"],
    "lastfm0.03": ["H1", "H2", "H3", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H15"],
    "amazon_beauty": ["H1", "H2", "H3", "H4", "H6", "H8", "H9", "H10", "H12", "H13", "H14", "H16"],
    "foursquare": ["H1", "H2", "H3", "H4", "H5", "H7", "H8", "H9", "H10", "H11", "H12", "H15"],
    "movielens1m": ["H1", "H2", "H3", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H14", "H16"],
    "retail_rocket": ["H1", "H2", "H3", "H4", "H6", "H8", "H9", "H10", "H12", "H13", "H15", "H16"],
}


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _apply_a2_aux_profile(overrides: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply strict NN + z-loss aux profile used by A2 family."""
    overrides["route_consistency_pairs"] = 1
    overrides["route_consistency_lambda"] = float(args.a2_route_consistency_lambda)
    overrides["route_consistency_min_sim"] = float(args.a2_route_consistency_min_sim)
    overrides["z_loss_lambda"] = float(args.a2_z_loss_lambda)
    overrides["route_monopoly_lambda"] = 0.0
    overrides["balance_loss_lambda"] = 0.0


def _core_overrides(
    args: argparse.Namespace,
    *,
    wrapper_map: Dict[str, str],
    source_profile: str,
    bias_mode: str,
) -> Dict[str, Any]:
    base_cfg = {
        "wrapper_map": dict(wrapper_map),
        "source_profile": str(source_profile),
        "bias_mode": str(bias_mode),
    }
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
    )
    # C0-N4 core (z-only mild stabilization) kept as the baseline default.
    overrides["z_loss_lambda"] = float(args.z_loss_lambda)
    overrides["balance_loss_lambda"] = float(args.balance_loss_lambda)
    overrides["macro_history_window"] = int(args.macro_history_window)
    return overrides


def _arch_overrides(arch_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    arch = str(arch_id).upper().strip()

    if arch in {"A1", "A2", "A3", "A4", "A5", "A6"}:
        overrides = _core_overrides(
            args,
            wrapper_map={"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
            source_profile="src_abc_feature",
            bias_mode="bias_both",
        )
        # Keep run naming stable for resume/skip compatibility, but switch the
        # actual architecture to the plain macro->mid->micro layout.
        overrides["layer_layout"] = ["macro", "mid", "micro"]
        overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
        overrides["stage_family_dropout_prob"] = _all_stage_map(float(args.family_dropout_prob))
        overrides["stage_feature_dropout_prob"] = _all_stage_map(float(args.feature_dropout_prob))
        overrides["stage_feature_dropout_scope"] = _all_stage_map("token")
        if arch == "A2":
            _apply_a2_aux_profile(overrides, args)
        elif arch == "A3":
            _apply_a2_aux_profile(overrides, args)
            overrides["stage_feature_drop_keywords"] = ["cat", "theme"]
        elif arch == "A4":
            # Disable legacy rule/group bias paths and use direct intra-group bias.
            overrides["rule_bias_scale"] = 0.0
            overrides["feature_group_bias_lambda"] = 0.0
            overrides["intra_group_bias_mode"] = _all_stage_map("gls_stats12")
            overrides["intra_group_bias_scale"] = _all_stage_map(float(args.a4_intra_group_bias_scale))
        elif arch == "A5":
            _apply_a2_aux_profile(overrides, args)
            overrides["stage_feature_drop_keywords"] = [
                "cat",
                "theme",
                "timestamp",
                "gap",
                "pace",
                "int_",
                "_int",
                "sess_age",
                "ctx_valid_r",
                "valid_r",
                "delta_vs_mid",
            ]
        elif arch == "A6":
            _apply_a2_aux_profile(overrides, args)
            overrides["bias_mode"] = "none"
            overrides["rule_bias_scale"] = 0.0
            overrides["feature_group_bias_lambda"] = 0.0
        return overrides

    raise ValueError(f"Unsupported architecture: {arch_id}")


def _phase_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / str(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_log_dir(dataset) / "summary.csv"


def _manifest_path(dataset: str, args: argparse.Namespace) -> Path:
    if args.manifest_out:
        return Path(str(args.manifest_out)).with_name(f"{Path(str(args.manifest_out)).name}_{dataset}.json")
    return _phase_log_dir(dataset) / "final_matrix.json"


def _validate_session_fixed_files(dataset: str) -> None:
    ds_dir = REPO_ROOT / "Datasets" / "processed" / "feature_added_v3" / dataset
    required = [
        ds_dir / f"{dataset}.train.inter",
        ds_dir / f"{dataset}.valid.inter",
        ds_dir / f"{dataset}.test.inter",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"session_fixed files missing for dataset={dataset}: {missing}")


def _hparam_num(hid: str) -> int:
    token = str(hid).upper().strip()
    if token.startswith("H") and token[1:].isdigit():
        return int(token[1:])
    raise ValueError(f"Invalid hparam id: {hid}")


def _parse_outlier_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not str(text or "").strip():
        return out
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        ds, hid = chunk.split(":", 1)
        ds = ds.strip()
        hid = hid.strip().upper()
        if ds and hid:
            out[ds] = hid
    return out


def _selected_hparams_for_dataset(dataset: str, args: argparse.Namespace) -> list[str]:
    common = [h.upper() for h in _parse_csv_strings(args.common_hparams)]
    if common == ["AUTO12"]:
        preset = list(DATASET_HPARAM_PRESET_12.get(dataset, []))
        selected = [h for h in preset if h in HPARAM_BANK]
        if selected:
            return selected

    outlier_map = dict(DEFAULT_OUTLIER_HPARAM)
    outlier_map.update(_parse_outlier_map(args.dataset_outlier_hparams))
    outlier = str(outlier_map.get(dataset, args.default_outlier_hparam)).upper()
    selected: list[str] = []
    for hid in [*common, outlier]:
        if hid in HPARAM_BANK and hid not in selected:
            selected.append(hid)
    if not selected:
        raise RuntimeError(f"No valid hparams selected for dataset={dataset}")
    return selected


def _selected_architectures(args: argparse.Namespace) -> list[str]:
    raw = args.architecture or args.architectures
    selected: list[str] = []
    for token in _parse_csv_strings(raw):
        arch = token.upper()
        if arch not in ARCH_METADATA:
            raise RuntimeError(f"Unknown architecture={token}. Supported: {','.join(ARCH_ORDER)}")
        if arch not in selected:
            selected.append(arch)
    if not selected:
        raise RuntimeError("No architecture selected")
    return selected


def _build_rows(dataset: str, args: argparse.Namespace, arch_id: str) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")
    hparams = _selected_hparams_for_dataset(dataset, args)
    overrides = _arch_overrides(arch_id, args)
    arch_meta = dict(ARCH_METADATA[str(arch_id)])

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    dataset_tag = sanitize_token(dataset, upper=True)

    for hid in hparams:
        hnum = _hparam_num(hid)
        for seed_id in seeds:
            run_cursor += 1

            if arch_id == "A1":
                # Keep A1 naming backward-compatible so existing completed logs are skippable after migration.
                run_id = f"FAD_{dataset_tag}_H{hnum}_S{int(seed_id)}"
                run_phase = f"{PHASE_ID}_{run_id}"
                setting_id = f"FINAL_MAIN_H{hnum}_S{int(seed_id)}"
                setting_key = arch_meta["setting_key"]
                setting_desc = f"{arch_meta['setting_desc_prefix']}_H{hnum}_S{int(seed_id)}"
            else:
                run_id = f"FAD_{arch_id}_{dataset_tag}_H{hnum}_S{int(seed_id)}"
                run_phase = f"{PHASE_ID}_{run_id}"
                setting_id = f"{arch_meta['arch_key']}_H{hnum}_S{int(seed_id)}"
                setting_key = arch_meta["setting_key"]
                setting_desc = f"{arch_meta['setting_desc_prefix']}_H{hnum}_S{int(seed_id)}"

            rows.append(
                {
                    "dataset": dataset,
                    "phase_id": PHASE_ID,
                    "axis_id": "FAD",
                    "axis_desc": AXIS_DESC,
                    "architecture_id": arch_id,
                    "architecture_key": arch_meta["arch_key"],
                    "architecture_name": arch_meta["arch_name"],
                    "setting_id": setting_id,
                    "setting_key": setting_key,
                    "setting_desc": setting_desc,
                    "setting_group": arch_meta["setting_group"],
                    "setting_detail": arch_meta["setting_detail"],
                    "hparam_id": hid,
                    "hparam_num": hnum,
                    "seed_id": int(seed_id),
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "exp_brief": arch_meta["arch_name"],
                    "stage": RUN_STAGE,
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "overrides": copy.deepcopy(overrides),
                }
            )
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    h = dict(HPARAM_BANK[str(row["hparam_id"])])
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    sched = _parse_csv_strings(args.search_lr_scheduler)
    if not sched:
        sched = ["warmup_cosine"]
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
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
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(row['architecture_id'])}",
        f"++fmoe_architecture_key={hydra_literal(row['architecture_key'])}",
        f"++fmoe_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(h['embedding_size'])}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(h['d_ff'])}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(h['d_expert_hidden'])}",
        f"d_router_hidden={int(h['d_router_hidden'])}",
        f"expert_scale={int(args.expert_scale)}",
        f"++search.learning_rate={hydra_literal([float(args.search_lr_min), float(args.search_lr_max)])}",
        f"++search.weight_decay={hydra_literal([float(h['fixed_weight_decay'])])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(h['fixed_hidden_dropout_prob'])])}",
        f"++search.lr_scheduler_type={hydra_literal(sched)}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(row['axis_id'])}",
        f"++phase_axis_desc={hydra_literal(row['axis_desc'])}",
        f"++phase_setting_id={hydra_literal(row['setting_id'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++phase_seed_id={hydra_literal(row['seed_id'])}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
    ]
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _build_log_path(log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    dataset = str(row.get("dataset", "") or "")
    base_dir = _phase_log_dir(dataset) if dataset else log_dir
    phase = sanitize_token(phase_id, upper=True)
    axis_id = sanitize_token(str(row.get("axis_id", "FAD")), upper=True)
    axis_desc = sanitize_token(str(row.get("axis_desc", AXIS_DESC)), upper=False)
    setting_id = sanitize_token(str(row.get("setting_id", "FINAL_MAIN")), upper=True)
    setting_desc = sanitize_token(str(row.get("setting_desc", "FINAL_MAIN_ATTN_MICRO_BEFORE")), upper=True)
    hparam = sanitize_token(str(row.get("hparam_id", "H1")), upper=True)
    arch = sanitize_token(str(row.get("architecture_id", "A1")), upper=True)
    filename = f"{phase}_{axis_id}_{axis_desc}_{setting_id}_{setting_desc}.log"
    return base_dir / arch / hparam / filename


def _write_manifest(
    dataset: str,
    args: argparse.Namespace,
    rows_by_arch: Dict[str, list[Dict[str, Any]]],
    architectures: list[str],
) -> Path:
    path = _manifest_path(dataset, args)
    seeds = _parse_csv_ints(args.seeds)
    selected_hparams = _selected_hparams_for_dataset(dataset, args)
    all_rows = [r for arch in architectures for r in rows_by_arch.get(arch, [])]

    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "dataset": dataset,
        "execution_type": RUN_STAGE,
        "architecture_count": len(architectures),
        "architectures": architectures,
        "hparam_count": len(selected_hparams),
        "seed_count": len(seeds),
        "run_count": len(all_rows),
        "run_count_formula": f"{len(architectures)} x {len(selected_hparams)} x {len(seeds)}",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_hparams": selected_hparams,
        "rows": [
            {
                "architecture_id": r["architecture_id"],
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "hparam_id": r["hparam_id"],
                "seed_id": r["seed_id"],
            }
            for r in all_rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _safe_move(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
        return True
    except FileNotFoundError:
        # Source can disappear between listing and move in concurrent environments.
        return False


def _extract_hparam_from_name(text: str) -> str:
    m = re.search(r"_H(\d+)_S\d+", str(text), flags=re.IGNORECASE)
    if m:
        return f"H{int(m.group(1))}"
    return "HX"


def _extract_hparam_from_run_dir(run_dir: Path) -> str:
    token = _extract_hparam_from_name(run_dir.name)
    if token != "HX":
        return token
    run_meta = run_dir / "run_meta.json"
    if run_meta.exists():
        try:
            payload = json.loads(run_meta.read_text(encoding="utf-8"))
            return _extract_hparam_from_name(str(payload.get("run_phase", "")))
        except Exception:
            return "HX"
    return "HX"


def _migrate_existing_logs_layout(dataset: str) -> tuple[int, int]:
    ds_root = _phase_log_dir(dataset)
    if not ds_root.exists():
        return 0, 0

    moved = 0
    skipped = 0
    a1_root = ds_root / "A1"
    a1_root.mkdir(parents=True, exist_ok=True)

    for path in sorted(ds_root.iterdir()):
        if not path.is_dir() or path.name.upper() == "A1":
            continue
        if re.fullmatch(r"H\d+", path.name.upper()) is None:
            continue
        target_dir = a1_root / path.name.upper()
        target_dir.mkdir(parents=True, exist_ok=True)
        for log_file in sorted(path.glob("*.log")):
            if _safe_move(log_file, target_dir / log_file.name):
                moved += 1
            else:
                skipped += 1
        try:
            path.rmdir()
        except Exception:
            pass

    for log_file in sorted(ds_root.glob("*.log")):
        hparam = _extract_hparam_from_name(log_file.name)
        target = a1_root / hparam / log_file.name
        if _safe_move(log_file, target):
            moved += 1
        else:
            skipped += 1

    return moved, skipped


def _migrate_existing_logging_layout(dataset: str) -> tuple[int, int]:
    legacy_root = REPO_ROOT / "experiments" / "run" / "artifacts" / "logging" / "fmoe_n3" / dataset / PHASE_ID
    if not legacy_root.exists():
        return 0, 0

    moved = 0
    skipped = 0
    a1_root = REPO_ROOT / "experiments" / "run" / "artifacts" / "logging" / "fmoe_n3" / AXIS / dataset / "A1"

    for run_dir in sorted(legacy_root.iterdir()):
        if not run_dir.is_dir():
            continue
        hparam = _extract_hparam_from_run_dir(run_dir)
        target = a1_root / hparam / run_dir.name
        if _safe_move(run_dir, target):
            moved += 1
        else:
            skipped += 1

    return moved, skipped


def _migrate_existing_layout_for_dataset(dataset: str) -> None:
    log_moved, log_skipped = _migrate_existing_logs_layout(dataset)
    detail_moved, detail_skipped = _migrate_existing_logging_layout(dataset)
    if any((log_moved, log_skipped, detail_moved, detail_skipped)):
        print(
            f"[{PHASE_ID}] migrated dataset={dataset} "
            f"logs(moved={log_moved},skipped={log_skipped}) "
            f"logging(moved={detail_moved},skipped={detail_skipped})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Final all-datasets launcher")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--seed-base", type=int, default=91000)

    parser.add_argument("--architectures", default="A2,A3,A5,A6", help="CSV from {A1,A2,A3,A4,A5,A6}")
    parser.add_argument("--architecture", default="", help="Alias of --architectures")

    parser.add_argument("--common-hparams", default="H1,H3")
    parser.add_argument("--default-outlier-hparam", default="H4")
    parser.add_argument(
        "--dataset-outlier-hparams",
        default="",
        help="CSV map: dataset:Hid,dataset2:Hid ... (overrides defaults)",
    )

    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--z-loss-lambda", type=float, default=1e-4)
    parser.add_argument("--balance-loss-lambda", type=float, default=0.0)
    parser.add_argument("--macro-history-window", type=int, default=5)
    parser.add_argument("--family-dropout-prob", type=float, default=0.10)
    parser.add_argument("--feature-dropout-prob", type=float, default=0.0)

    parser.add_argument("--a2-route-consistency-lambda", type=float, default=8e-4)
    parser.add_argument("--a2-route-consistency-min-sim", type=float, default=0.995)
    parser.add_argument("--a2-z-loss-lambda", type=float, default=2e-4)
    parser.add_argument("--a4-intra-group-bias-scale", type=float, default=0.12)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)
    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")

    parser.add_argument("--migrate-existing-layout", dest="migrate_existing_layout", action="store_true")
    parser.add_argument("--no-migrate-existing-layout", dest="migrate_existing_layout", action="store_false")
    parser.set_defaults(migrate_existing_layout=True)

    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"
    gpus = _parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"
    ds = _parse_csv_strings(args.datasets)
    args.datasets = ds[0] if ds else DEFAULT_DATASETS[0]


def _build_execution_plan(datasets: list[str], architectures: list[str]) -> tuple[list[tuple[str, str]], str]:
    # Requested priority mode for final v4:
    # - Default set A1..A5:
    #   1) Run Kuai A1 first.
    #   2) Then per-dataset A2->A3->A4->A5.
    # - Any other architecture list: run architecture-major.
    if architectures == ["A1", "A2", "A3", "A4", "A5"]:
        plan: list[tuple[str, str]] = []
        if "KuaiRecLargeStrictPosV2_0.2" in datasets:
            plan.append(("KuaiRecLargeStrictPosV2_0.2", "A1"))
        for dataset in datasets:
            for arch in ("A2", "A3", "A4", "A5"):
                plan.append((dataset, arch))
        return plan, "Kuai_A1_then_datasetwise_A2_A3_A4_A5"

    if architectures == ["A1", "A2", "A3"]:
        plan: list[tuple[str, str]] = []
        for dataset in datasets:
            plan.append((dataset, "A1"))
        for dataset in datasets:
            plan.append((dataset, "A2"))
            plan.append((dataset, "A3"))
        return plan, "A1_all_datasets_then_datasetwise_A2_A3"

    if architectures == ["A2", "A3", "A5", "A6"]:
        plan: list[tuple[str, str]] = []
        for dataset in datasets:
            plan.append((dataset, "A2"))
        for dataset in datasets:
            for arch in ("A3", "A5", "A6"):
                plan.append((dataset, arch))
        return plan, "A2_all_datasets_then_datasetwise_A3_A5_A6"

    plan = []
    for arch_id in architectures:
        for dataset in datasets:
            plan.append((dataset, arch_id))
    return plan, "architecture_major"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs provided")
    datasets = _parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets provided")
    architectures = _selected_architectures(args)

    extra_cols = [
        "phase_id",
        "axis_id",
        "axis_desc",
        "architecture_id",
        "architecture_key",
        "architecture_name",
        "setting_id",
        "setting_key",
        "setting_desc",
        "setting_group",
        "setting_detail",
        "hparam_id",
        "seed_id",
        "run_id",
    ]
    fieldnames = build_summary_fieldnames(extra_cols)

    dataset_payloads: Dict[str, Dict[str, Any]] = {}
    for d_idx, dataset in enumerate(datasets):
        _validate_session_fixed_files(dataset)

        if bool(args.migrate_existing_layout):
            _migrate_existing_layout_for_dataset(dataset)

        rows_by_arch: Dict[str, list[Dict[str, Any]]] = {}
        for arch_id in architectures:
            rows = _build_rows(dataset, args, arch_id)
            if args.smoke_test:
                rows = rows[: max(int(args.smoke_max_runs), 1)]
            rows_by_arch[arch_id] = rows

        manifest_path = _write_manifest(dataset, args, rows_by_arch, architectures)
        log_dir = _phase_log_dir(dataset)
        summary_path = _summary_path(dataset)
        total_rows = sum(len(v) for v in rows_by_arch.values())

        print(
            f"[{PHASE_ID}] dataset={dataset} ({d_idx+1}/{len(datasets)}) total_rows={total_rows} "
            f"architectures={','.join(architectures)} axis={AXIS} manifest={manifest_path}"
        )

        dataset_payloads[dataset] = {
            "rows_by_arch": rows_by_arch,
            "log_dir": log_dir,
            "summary_path": summary_path,
        }

    execution_plan, plan_name = _build_execution_plan(datasets, architectures)
    print(f"[{PHASE_ID}] execution_plan={plan_name} steps={len(execution_plan)}")

    scheduled_rows: list[Dict[str, Any]] = []
    for step_idx, (dataset, arch_id) in enumerate(execution_plan, start=1):
        payload = dataset_payloads.get(dataset, {})
        rows_by_arch = dict(payload.get("rows_by_arch", {}) or {})
        rows = list(rows_by_arch.get(arch_id, []) or [])
        if not rows:
            continue
        print(
            f"[{PHASE_ID}] schedule_step={step_idx}/{len(execution_plan)} "
            f"dataset={dataset} architecture={arch_id} rows={len(rows)}"
        )

        # Preserve execution-plan ordering while allowing a single global GPU queue.
        for row in rows:
            row["scheduled_step"] = int(step_idx)
        scheduled_rows.extend(rows)

    if not scheduled_rows:
        print(f"[{PHASE_ID}] nothing to schedule after plan filtering.")
        return 0

    rc = launch_wide_rows(
        rows=scheduled_rows,
        gpus=gpus,
        args=args,
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        log_dir=LOG_ROOT / AXIS,
        summary_path=_summary_path(datasets[0]),
        fieldnames=fieldnames,
        extra_cols=extra_cols,
        build_command=_build_command,
        build_log_path=_build_log_path,
        verify_logging=bool(args.verify_logging),
        summary_path_for_row=lambda row: _summary_path(str(row.get("dataset", ""))),
    )
    if rc != 0:
        return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
