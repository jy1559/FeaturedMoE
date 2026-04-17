#!/usr/bin/env python3
"""Pairwise transfer pilot launcher for FMoE_N4 ablations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
RUN_DIR = THIS_DIR.parent
FMOE_N3_DIR = RUN_DIR.parent / "fmoe_n3"
for extra in (THIS_DIR, RUN_DIR, FMOE_N3_DIR):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import common  # noqa: E402
from run_phase9_auxloss import _parse_csv_ints, _parse_csv_strings, hydra_literal  # noqa: E402
from run_phase_wide_common import launch_wide_rows, sanitize_token  # noqa: E402

AXIS = "ablation_transfer_pairwise_v1"
AXIS_ID = "N4ABL5"
AXIS_DESC = "transfer_pairwise"
PHASE_ID = "P45"
PHASE_NAME = "N4_TRANSFER_PAIRWISE"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS

COMMON_HPARAM_PRESETS: dict[str, dict[str, Any]] = {
    "shared_a": {
        "embedding_size": 128,
        "d_ff": 256,
        "d_expert_hidden": 128,
        "d_router_hidden": 64,
        "MAX_ITEM_LIST_LENGTH": 20,
        "d_feat_emb": 12,
        "expert_scale": 3,
        "num_heads": 4,
        "hidden_dropout_prob": 0.18,
        "attn_dropout_prob": 0.1,
        "weight_decay": 5e-7,
        "lr_scheduler_type": "warmup_cosine",
    },
    "shared_b": {
        "embedding_size": 192,
        "d_ff": 384,
        "d_expert_hidden": 192,
        "d_router_hidden": 96,
        "MAX_ITEM_LIST_LENGTH": 25,
        "d_feat_emb": 16,
        "expert_scale": 4,
        "num_heads": 4,
        "hidden_dropout_prob": 0.12,
        "attn_dropout_prob": 0.1,
        "weight_decay": 5e-7,
        "lr_scheduler_type": "warmup_cosine",
    },
}

PAIR_SPECS: dict[str, dict[str, str]] = {
    "beauty_to_kuairec": {"source_dataset": "beauty", "target_dataset": "KuaiRecLargeStrictPosV2_0.2"},
    "kuairec_to_beauty": {"source_dataset": "KuaiRecLargeStrictPosV2_0.2", "target_dataset": "beauty"},
    "lastfm_to_kuairec": {"source_dataset": "lastfm0.03", "target_dataset": "KuaiRecLargeStrictPosV2_0.2"},
    "beauty_to_lastfm": {"source_dataset": "beauty", "target_dataset": "lastfm0.03"},
}

LR_CENTER_BY_DATASET = {
    "beauty": 0.001841911333,
    "KuaiRecLargeStrictPosV2_0.2": 0.00052,
    "lastfm0.03": 0.00055,
}

TRANSFER_SETTINGS = [
    {
        "scope": "core",
        "tier": "essential",
        "setting_id": "TR-01",
        "setting_key": "SCRATCH",
        "setting_desc": "scratch",
        "setting_group": "transfer",
        "setting_detail": "Native target training with the shared hyperparameter preset.",
        "transfer_mode": "none",
    },
    {
        "scope": "core",
        "tier": "essential",
        "setting_id": "TR-02",
        "setting_key": "FEATURE_ENCODER_INIT",
        "setting_desc": "feature_encoder_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize all stage feature encoders from the source dataset.",
        "transfer_mode": "all_stage_feature_encoder",
    },
    {
        "scope": "core",
        "tier": "essential",
        "setting_id": "TR-03",
        "setting_key": "GROUP_ROUTER_INIT",
        "setting_desc": "group_router_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize group routers only from the source checkpoint.",
        "transfer_mode": "all_stage_group_router",
    },
    {
        "scope": "core",
        "tier": "essential",
        "setting_id": "TR-04",
        "setting_key": "ALL_ROUTER_INIT",
        "setting_desc": "all_router_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize all router blocks from the source checkpoint.",
        "transfer_mode": "all_stage_full_router",
    },
    {
        "scope": "core",
        "tier": "essential",
        "setting_id": "TR-05",
        "setting_key": "FULL_MODEL_INIT",
        "setting_desc": "full_model_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize every compatible tensor from the source checkpoint.",
        "transfer_mode": "full_model",
    },
    {
        "scope": "appendix",
        "tier": "extended",
        "setting_id": "TR-06",
        "setting_key": "MACRO_GROUP_ROUTER_INIT",
        "setting_desc": "macro_group_router_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize only the macro-stage group router.",
        "transfer_mode": "macro_group_router",
    },
    {
        "scope": "appendix",
        "tier": "extended",
        "setting_id": "TR-07",
        "setting_key": "MACRO_FEATURE_ENCODER_INIT",
        "setting_desc": "macro_feature_encoder_init",
        "setting_group": "transfer",
        "setting_detail": "Initialize only the macro-stage feature encoder.",
        "transfer_mode": "macro_feature_encoder",
    },
]


def _all_stage(value: str) -> dict[str, str]:
    return {"macro": value, "mid": value, "micro": value}


def _base_overrides() -> dict[str, object]:
    return {
        "layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"],
        "stage_compute_mode": _all_stage("moe"),
        "stage_router_mode": _all_stage("learned"),
        "stage_router_source": _all_stage("both"),
        "stage_feature_injection": _all_stage("none"),
        "stage_router_wrapper": _all_stage("w5_exd"),
        "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
        "macro_history_window": 5,
        "route_consistency_pairs": 1,
        "route_consistency_min_sim": 0.995,
        "route_consistency_lambda": 0.0008,
        "z_loss_lambda": 0.0002,
        "balance_loss_lambda": 0.0,
        "bias_mode": "none",
        "topk_scope_mode": "global_flat",
        "moe_top_k": 0,
        "feature_perturb_mode": "none",
        "feature_perturb_apply": "none",
        "feature_perturb_family": [],
        "feature_perturb_keywords": [],
        "stage_family_dropout_prob": {"macro": 0.03, "mid": 0.03, "micro": 0.03},
        "stage_feature_dropout_prob": {"macro": 0.0, "mid": 0.0, "micro": 0.0},
        "stage_feature_dropout_scope": {"macro": "token", "mid": "token", "micro": "token"},
        "stage_router_primitives": {
            stage: {
                "a_joint": {"source": "feature", "temperature": 1.0, "top_k": 0},
                "b_group": {"source": "feature", "temperature": 1.0, "top_k": 0},
                "c_shared": {"source": "feature", "temperature": 1.0, "top_k": 0},
                "d_cond": {"source": "feature", "temperature": 1.0, "top_k": 0},
                "e_scalar": {"source": "feature", "temperature": 1.0, "top_k": 0},
                "wrapper": {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0},
            }
            for stage in ("macro", "mid", "micro")
        },
    }


def _selected_pairs(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for token in _parse_csv_strings(str(args.pairs)):
        if token not in PAIR_SPECS:
            raise RuntimeError(f"Unknown pair={token}. Supported: {','.join(sorted(PAIR_SPECS))}")
        if token not in selected:
            selected.append(token)
    if not selected:
        raise RuntimeError("No transfer pair selected")
    return selected


def _selected_presets(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for token in _parse_csv_strings(str(args.hparam_presets)):
        if token not in COMMON_HPARAM_PRESETS:
            raise RuntimeError(f"Unknown hparam preset={token}. Supported: {','.join(sorted(COMMON_HPARAM_PRESETS))}")
        if token not in selected:
            selected.append(token)
    if not selected:
        raise RuntimeError("No shared hparam preset selected")
    return selected


def _selected_transfer_settings(args: argparse.Namespace) -> list[dict[str, object]]:
    scope = str(args.setting_scope)
    tier = str(args.setting_tier)
    only = {str(value) for value in _parse_csv_strings(str(args.only_setting or ""))}
    selected: list[dict[str, object]] = []
    for setting in TRANSFER_SETTINGS:
        if scope != "all" and str(setting.get("scope", "core")) != scope:
            continue
        if tier != "all" and str(setting.get("tier", "essential")) != tier:
            continue
        if only:
            tokens = {str(setting.get("setting_id", "")), str(setting.get("setting_key", ""))}
            if tokens.isdisjoint(only):
                continue
        selected.append(dict(setting))
    return selected


def _dataset_batch_sizes(dataset: str) -> tuple[int, int]:
    if dataset == "KuaiRecLargeStrictPosV2_0.2":
        return 3584, 4608
    return 4096, 4096


def _source_checkpoint_path(log_root: Path, dataset: str, preset_id: str, seed_id: int) -> Path:
    return log_root / "exports" / sanitize_token(dataset, upper=False) / f"{preset_id}_s{int(seed_id)}_best.pth"


def _source_rows(args: argparse.Namespace, pair_ids: list[str], preset_ids: list[str]) -> list[dict[str, object]]:
    seeds = _parse_csv_ints(args.seeds)
    rows: list[dict[str, object]] = []
    dataset_order: list[str] = []
    for pair_id in pair_ids:
        spec = PAIR_SPECS[pair_id]
        for dataset in (spec["source_dataset"], spec["target_dataset"]):
            if dataset not in dataset_order:
                dataset_order.append(dataset)
    cursor = 0
    for dataset in dataset_order:
        common.validate_session_fixed_files(dataset, args.feature_dataset_dir)
        train_batch_size, eval_batch_size = _dataset_batch_sizes(dataset)
        for preset_id in preset_ids:
            fixed_values = dict(COMMON_HPARAM_PRESETS[preset_id])
            lr_choices = common.build_lr_choices(LR_CENTER_BY_DATASET.get(dataset, 0.001), str(args.lr_mode))
            for seed_id in seeds:
                cursor += 1
                run_id = f"SRC_{sanitize_token(dataset, upper=True)}_{sanitize_token(preset_id, upper=True)}_S{int(seed_id)}"
                rows.append(
                    {
                        "track": common.TRACK,
                        "axis_id": AXIS_ID,
                        "axis_desc": AXIS_DESC,
                        "phase_id": PHASE_ID,
                        "architecture_id": common.ARCH_ID,
                        "architecture_key": common.ARCH_KEY,
                        "architecture_name": common.ARCH_NAME,
                        "exp_brief": f"transfer source {preset_id}",
                        "dataset": dataset,
                        "run_phase": f"{PHASE_ID}_{run_id}",
                        "run_id": run_id,
                        "stage": "transfer_native_shared",
                        "tuning_stage": "transfer_native_shared",
                        "phase_kind": "source_pretrain",
                        "setting_tier": "essential",
                        "setting_id": "TS-00",
                        "setting_key": "SOURCE_NATIVE",
                        "setting_desc": "source_native",
                        "setting_group": "transfer_source",
                        "setting_detail": f"shared preset={preset_id}",
                        "family_id": f"source_{sanitize_token(dataset, upper=False)}_{preset_id}",
                        "family_group": "transfer_source",
                        "variant_id": preset_id,
                        "capacity_anchor": preset_id,
                        "search_algo": str(args.search_algo),
                        "seed_id": int(seed_id),
                        "runtime_seed": int(args.seed_base) + cursor - 1,
                        "fixed_values": fixed_values,
                        "search_space": {"learning_rate": {"type": "choice", "values": lr_choices}},
                        "overrides": _base_overrides(),
                        "train_batch_size": int(train_batch_size),
                        "eval_batch_size": int(eval_batch_size),
                        "max_evals": int(args.max_evals),
                        "tune_epochs": int(args.tune_epochs),
                        "tune_patience": int(args.tune_patience),
                        "feature_mode": str(args.feature_mode),
                        "eval_mode": str(args.eval_mode),
                        "diag_logging": True,
                        "special_logging": True,
                        "feature_ablation_logging": False,
                        "hparam_preset": preset_id,
                        "source_dataset": dataset,
                        "target_dataset": dataset,
                        "pair_id": "native_shared",
                        "transfer_mode": "none",
                        "source_checkpoint_export_path": str(_source_checkpoint_path(LOG_ROOT, dataset, preset_id, seed_id)),
                    }
                )
    return rows


def _target_rows(
    args: argparse.Namespace,
    pair_ids: list[str],
    preset_ids: list[str],
    settings: list[dict[str, object]],
) -> list[dict[str, object]]:
    seeds = _parse_csv_ints(args.seeds)
    rows: list[dict[str, object]] = []
    cursor = 100000
    for pair_id in pair_ids:
        pair_spec = PAIR_SPECS[pair_id]
        source_dataset = pair_spec["source_dataset"]
        target_dataset = pair_spec["target_dataset"]
        common.validate_session_fixed_files(source_dataset, args.feature_dataset_dir)
        common.validate_session_fixed_files(target_dataset, args.feature_dataset_dir)
        train_batch_size, eval_batch_size = _dataset_batch_sizes(target_dataset)
        for preset_id in preset_ids:
            fixed_values = dict(COMMON_HPARAM_PRESETS[preset_id])
            lr_choices = common.build_lr_choices(LR_CENTER_BY_DATASET.get(target_dataset, 0.001), str(args.lr_mode))
            for setting in settings:
                for seed_id in seeds:
                    cursor += 1
                    run_id = (
                        f"TGT_{sanitize_token(pair_id, upper=True)}_"
                        f"{sanitize_token(preset_id, upper=True)}_{sanitize_token(str(setting['setting_id']), upper=True)}_S{int(seed_id)}"
                    )
                    rows.append(
                        {
                            "track": common.TRACK,
                            "axis_id": AXIS_ID,
                            "axis_desc": AXIS_DESC,
                            "phase_id": PHASE_ID,
                            "architecture_id": common.ARCH_ID,
                            "architecture_key": common.ARCH_KEY,
                            "architecture_name": common.ARCH_NAME,
                            "exp_brief": f"transfer target {pair_id}",
                            "dataset": target_dataset,
                            "run_phase": f"{PHASE_ID}_{run_id}",
                            "run_id": run_id,
                            "stage": "transfer_pairwise",
                            "tuning_stage": "transfer_pairwise",
                            "phase_kind": "target_transfer",
                            "setting_tier": str(setting.get("tier", "essential")),
                            "setting_id": str(setting["setting_id"]),
                            "setting_key": str(setting["setting_key"]),
                            "setting_desc": str(setting["setting_desc"]),
                            "setting_group": str(setting["setting_group"]),
                            "setting_detail": str(setting["setting_detail"]),
                            "family_id": f"{pair_id}_{preset_id}_{setting['setting_id']}",
                            "family_group": "transfer_target",
                            "variant_id": str(setting["setting_id"]),
                            "capacity_anchor": preset_id,
                            "search_algo": str(args.search_algo),
                            "seed_id": int(seed_id),
                            "runtime_seed": int(args.seed_base) + cursor - 1,
                            "fixed_values": fixed_values,
                            "search_space": {"learning_rate": {"type": "choice", "values": lr_choices}},
                            "overrides": _base_overrides(),
                            "train_batch_size": int(train_batch_size),
                            "eval_batch_size": int(eval_batch_size),
                            "max_evals": int(args.max_evals),
                            "tune_epochs": int(args.tune_epochs),
                            "tune_patience": int(args.tune_patience),
                            "feature_mode": str(args.feature_mode),
                            "eval_mode": str(args.eval_mode),
                            "diag_logging": True,
                            "special_logging": True,
                            "feature_ablation_logging": False,
                            "hparam_preset": preset_id,
                            "source_dataset": source_dataset,
                            "target_dataset": target_dataset,
                            "pair_id": pair_id,
                            "transfer_mode": str(setting.get("transfer_mode", "none")),
                            "source_checkpoint": str(_source_checkpoint_path(LOG_ROOT, source_dataset, preset_id, seed_id)),
                        }
                    )
    return rows


def _build_log_path(log_dir: Path, row: dict[str, object], phase_id: str) -> Path:
    dataset = sanitize_token(str(row.get("dataset", "dataset")), upper=False)
    phase_kind = sanitize_token(str(row.get("phase_kind", "phase")), upper=False)
    family = sanitize_token(str(row.get("family_id", "family")), upper=False)
    filename = f"{phase_id}_{sanitize_token(str(row.get('setting_id', '00')), upper=True)}_S{int(row.get('seed_id', 1))}.log"
    return log_dir / dataset / phase_kind / family / filename


def _device_assignments(gpu_id: str) -> list[str]:
    token = str(gpu_id).strip().lower()
    if token in {"cpu", "-1", "none"}:
        return ["gpu_id=", "use_gpu=false", "enable_tf32=false"]
    return [f"gpu_id={gpu_id}", "use_gpu=true", "enable_tf32=true"]


def _build_command(row: dict[str, object], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = str(Path("/venv/FMoE/bin/python"))
    if not Path(python_bin).exists():
        python_bin = sys.executable
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
        common.TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        f"eval_mode={row['eval_mode']}",
        f"feature_mode={row['feature_mode']}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
        *_device_assignments(gpu_id),
        "log_wandb=false",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(common.ARCH_ID)}",
        f"++fmoe_architecture_key={hydra_literal(common.ARCH_KEY)}",
        f"++fmoe_hparam_id={hydra_literal(str(row['capacity_anchor']))}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"train_batch_size={int(row['train_batch_size'])}",
        f"eval_batch_size={int(row['eval_batch_size'])}",
        f"++phase_run_type={hydra_literal(str(row['stage']))}",
        f"++phase_axis_id={hydra_literal(AXIS_ID)}",
        f"++phase_axis_desc={hydra_literal(AXIS_DESC)}",
        f"++phase_setting_id={hydra_literal(str(row['setting_id']))}",
        f"++phase_setting_key={hydra_literal(str(row['setting_key']))}",
        f"++phase_hparam_id={hydra_literal(str(row['capacity_anchor']))}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(str(row['run_id']))}",
        f"++phase_transfer_pair_id={hydra_literal(str(row.get('pair_id', '')))}",
        f"++phase_transfer_mode={hydra_literal(str(row.get('transfer_mode', 'none')))}",
        f"++phase_transfer_source_dataset={hydra_literal(str(row.get('source_dataset', '')))}",
        f"++phase_transfer_target_dataset={hydra_literal(str(row.get('target_dataset', '')))}",
        f"++phase_transfer_hparam_preset={hydra_literal(str(row.get('hparam_preset', '')))}",
    ]
    for key, value in dict(row.get("overrides") or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, value in dict(row.get("fixed_values") or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, spec in dict(row.get("search_space") or {}).items():
        normalized = common._normalize_search_spec(spec)
        cmd.append(f"++search.{key}={hydra_literal(normalized['values'])}")
        cmd.append(f"++search_space_type_overrides.{key}={normalized['type']}")
    if str(row["phase_kind"]) == "source_pretrain":
        cmd.append(f"++__artifact_combo_best_export_path={hydra_literal(str(row['source_checkpoint_export_path']))}")
        cmd.append("++transfer.enable=false")
        cmd.append("++transfer.mode=none")
    else:
        mode = str(row.get("transfer_mode", "none"))
        if mode == "none":
            cmd.append("++transfer.enable=false")
            cmd.append("++transfer.mode=none")
        else:
            cmd.append("++transfer.enable=true")
            cmd.append(f"++transfer.mode={hydra_literal(mode)}")
            cmd.append(f"++transfer.source_checkpoint={hydra_literal(str(row['source_checkpoint']))}")
            cmd.append(f"++transfer.source_architecture={hydra_literal(common.ARCH_ID)}")
            cmd.append("++transfer.strict_shape=false")
    return cmd


def _launch_phase(rows: list[dict[str, object]], args: argparse.Namespace, phase_name: str) -> int:
    gpus = _parse_csv_strings(args.gpus)
    fieldnames = common.build_fieldnames([
        "phase_kind",
        "hparam_preset",
        "pair_id",
        "source_dataset",
        "target_dataset",
        "transfer_mode",
    ])
    return int(
        launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=phase_name,
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
            build_command=_build_command,
            build_log_path=_build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: common.summary_path(LOG_ROOT, str(row["dataset"])),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = common.common_arg_parser(
        "FMoE_N4 pairwise transfer pilot",
        default_datasets=common.DEFAULT_DATASETS,
        default_scope="core",
    )
    parser.add_argument("--pairs", default="beauty_to_kuairec")
    parser.add_argument("--hparam-presets", default="shared_a")
    args = parser.parse_args()
    args = common.finalize_common_args(args)
    args.axis = AXIS
    return args


def main() -> int:
    args = parse_args()
    pair_ids = _selected_pairs(args)
    preset_ids = _selected_presets(args)
    settings = _selected_transfer_settings(args)
    source_rows = common.maybe_limit_smoke(_source_rows(args, pair_ids, preset_ids), args)
    source_manifest = common.manifest_path(LOG_ROOT, args, "transfer_source_manifest.json")
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(json.dumps({"axis": AXIS, "phase": "source_pretrain", "rows": source_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[transfer] source manifest -> {source_manifest}")
    rc = _launch_phase(source_rows, args, f"{PHASE_NAME}_SOURCE")
    if rc != 0:
        return rc
    if not bool(args.dry_run):
        missing = [str(row["source_checkpoint_export_path"]) for row in source_rows if not Path(str(row["source_checkpoint_export_path"])).exists()]
        if missing:
            raise RuntimeError(f"Missing exported source checkpoints after source phase: {missing[:4]}")

    target_rows = common.maybe_limit_smoke(_target_rows(args, pair_ids, preset_ids, settings), args)
    target_manifest = common.manifest_path(LOG_ROOT, args, "transfer_target_manifest.json")
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(json.dumps({"axis": AXIS, "phase": "target_transfer", "rows": target_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[transfer] target manifest -> {target_manifest}")
    return _launch_phase(target_rows, args, f"{PHASE_NAME}_TARGET")


if __name__ == "__main__":
    raise SystemExit(main())