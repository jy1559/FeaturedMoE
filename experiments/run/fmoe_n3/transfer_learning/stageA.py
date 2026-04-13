#!/usr/bin/env python3
"""StageA transfer-learning launcher for FeaturedMoE_N3."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import run_final_all_datasets as final_base  # noqa: E402
import run_final_a8_a12_wrapper_sweep as wrapper_base  # noqa: E402
from run_phase9_auxloss import LOG_ROOT, TRACK, _load_result_index, _parse_csv_ints, _parse_csv_strings, hydra_literal  # noqa: E402
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows, sanitize_token  # noqa: E402

AXIS = "transfer_learning_stageA_v1"
PHASE_ID = "P15A"
PHASE_NAME = "TRANSFER_LEARNING_STAGEA"
RUN_STAGE = "transfer_stageA"
AXIS_ID = "TLA"
AXIS_DESC = "transfer_learning_stageA_v1"

ARCH_ORDER = ("A10", "A12")
ARCH_WRAPPER_ALIAS = {
    "A10": "w4_bxd",
    "A12": "w5_exd",
}

PAIR_SPECS: Dict[str, Dict[str, Any]] = {
    "kuairec_to_lastfm": {
        "source_dataset": "KuaiRecLargeStrictPosV2_0.2",
        "target_dataset": "lastfm0.03",
        "source_hparams": ["H3", "H7"],
        "target_map": {"H3": ["H3", "H7"], "H7": ["H7", "H3"]},
    },
    "lastfm_to_kuairec": {
        "source_dataset": "lastfm0.03",
        "target_dataset": "KuaiRecLargeStrictPosV2_0.2",
        "source_hparams": ["H3", "H7"],
        "target_map": {"H3": ["H3", "H7"], "H7": ["H7", "H3"]},
    },
    "amazon_to_retail": {
        "source_dataset": "amazon_beauty",
        "target_dataset": "retail_rocket",
        "source_hparams": ["H2", "H3"],
        "target_map": {"H2": ["H2", "H3"], "H3": ["H3", "H2"]},
    },
    "foursquare_to_movielens": {
        "source_dataset": "foursquare",
        "target_dataset": "movielens1m",
        "source_hparams": ["H3", "H5"],
        "target_map": {"H3": ["H3", "H5"], "H5": ["H5", "H3"]},
    },
}

SOURCE_LR_BANDS: Dict[str, list[float]] = {
    "KuaiRecLargeStrictPosV2_0.2": [2.5e-4, 5.5e-4],
    "lastfm0.03": [4.0e-4, 6.5e-4],
    "amazon_beauty": [6.0e-4, 1.5e-3],
    "foursquare": [1.3e-3, 2.2e-3],
}

TARGET_LR_CHOICES: Dict[str, list[float]] = {
    "KuaiRecLargeStrictPosV2_0.2": [2.5e-4, 3.6e-4, 5.2e-4, 8.2e-4, 5.0e-3],
    "lastfm0.03": [2.0e-4, 3.5e-4, 5.2e-4, 9.0e-4, 5.0e-3],
    "retail_rocket": [3.0e-4, 5.0e-4, 9.0e-4, 1.4e-3, 5.0e-3],
    "movielens1m": [1.2e-3, 2.2e-3, 3.2e-3, 4.3e-3, 5.0e-3],
}

TRAIN_BATCH_SIZE_BY_DATASET = {
    "retail_rocket": 2048,
}

EVAL_BATCH_SIZE_BY_DATASET = {
    "movielens1m": 6144,
}

DEFAULT_TRAIN_BATCH_SIZE = 4096
DEFAULT_EVAL_BATCH_SIZE = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FeaturedMoE_N3 StageA transfer-learning launcher")
    parser.add_argument("--architectures", default="A12", help="CSV from {A10,A12}")
    parser.add_argument("--architecture", default="", help="Alias of --architectures")
    parser.add_argument("--pairs", default=",".join(PAIR_SPECS.keys()))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=151000)

    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--target-only", action="store_true")

    parser.add_argument("--source-max-evals", type=int, default=3)
    parser.add_argument("--source-tune-epochs", type=int, default=80)
    parser.add_argument("--source-tune-patience", type=int, default=8)
    parser.add_argument("--target-max-evals", type=int, default=5)
    parser.add_argument("--target-tune-epochs", type=int, default=80)
    parser.add_argument("--target-tune-patience", type=int, default=8)

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
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)

    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", action="store_true", default=True)
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected_architectures(args: argparse.Namespace) -> list[str]:
    raw = args.architecture or args.architectures
    selected: list[str] = []
    for token in _parse_csv_strings(raw):
        arch = token.upper()
        if arch not in ARCH_ORDER:
            raise RuntimeError(f"Unknown architecture={token}. Supported: {','.join(ARCH_ORDER)}")
        if arch not in selected:
            selected.append(arch)
    if not selected:
        raise RuntimeError("No architecture selected")
    return selected


def _selected_pairs(args: argparse.Namespace) -> list[str]:
    selected: list[str] = []
    for token in _parse_csv_strings(args.pairs):
        pair_id = str(token).strip()
        if pair_id not in PAIR_SPECS:
            raise RuntimeError(f"Unknown pair={token}. Supported: {','.join(PAIR_SPECS.keys())}")
        if pair_id not in selected:
            selected.append(pair_id)
    if not selected:
        raise RuntimeError("No pair selected")
    return selected


def _pair_log_dir(pair_id: str) -> Path:
    return LOG_ROOT / AXIS / str(pair_id)


def _pair_summary_path(pair_id: str) -> Path:
    return _pair_log_dir(pair_id) / "summary.csv"


def _source_ckpt_dir(arch_id: str, pair_id: str) -> Path:
    return SCRIPT_DIR / "ckpt" / "stageA" / str(arch_id) / str(pair_id)


def _source_ckpt_path(*, arch_id: str, pair_id: str, dataset: str, hparam_id: str, seed_id: int) -> Path:
    dataset_token = sanitize_token(dataset, upper=False)
    return _source_ckpt_dir(arch_id, pair_id) / f"{dataset_token}_{hparam_id.upper()}_s{int(seed_id)}_best.pth"


def _source_meta_path(*, arch_id: str, pair_id: str, dataset: str, hparam_id: str, seed_id: int) -> Path:
    dataset_token = sanitize_token(dataset, upper=False)
    return _source_ckpt_dir(arch_id, pair_id) / f"{dataset_token}_{hparam_id.upper()}_s{int(seed_id)}_best.json"


def _dataset_train_batch_size(dataset: str) -> int:
    return int(TRAIN_BATCH_SIZE_BY_DATASET.get(dataset, DEFAULT_TRAIN_BATCH_SIZE))


def _dataset_eval_batch_size(dataset: str) -> int:
    return int(EVAL_BATCH_SIZE_BY_DATASET.get(dataset, DEFAULT_EVAL_BATCH_SIZE))


def _native_target_hparams(pair_spec: Dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for source_h in list(pair_spec.get("source_hparams", []) or []):
        for target_h in list((pair_spec.get("target_map") or {}).get(source_h, []) or []):
            if target_h not in ordered:
                ordered.append(target_h)
    return ordered


def _source_row_artifacts(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "checkpoint_path": str(
            _source_ckpt_path(
                arch_id=str(row["architecture_id"]),
                pair_id=str(row["pair_id"]),
                dataset=str(row["dataset"]),
                hparam_id=str(row["hparam_id"]),
                seed_id=int(row["seed_id"]),
            )
        ),
        "metadata_path": str(
            _source_meta_path(
                arch_id=str(row["architecture_id"]),
                pair_id=str(row["pair_id"]),
                dataset=str(row["dataset"]),
                hparam_id=str(row["hparam_id"]),
                seed_id=int(row["seed_id"]),
            )
        ),
    }


def _base_row(
    *,
    args: argparse.Namespace,
    pair_id: str,
    pair_spec: Dict[str, Any],
    arch_id: str,
    dataset: str,
    hparam_id: str,
    seed_id: int,
    row_index: int,
) -> Dict[str, Any]:
    arch_meta = dict(wrapper_base.ARCH_METADATA[str(arch_id)])
    return {
        "phase_id": PHASE_ID,
        "axis_id": AXIS_ID,
        "axis_desc": AXIS_DESC,
        "pair_id": pair_id,
        "dataset": dataset,
        "source_dataset": str(pair_spec["source_dataset"]),
        "target_dataset": str(pair_spec["target_dataset"]),
        "architecture_id": arch_id,
        "architecture_key": arch_meta["arch_key"],
        "architecture_name": arch_meta["arch_name"],
        "seed_id": int(seed_id),
        "runtime_seed": int(args.seed_base) + int(row_index),
        "hparam_id": str(hparam_id).upper(),
        "hparam_num": final_base._hparam_num(hparam_id),
        "stage": RUN_STAGE,
    }


def _build_source_rows(pair_id: str, pair_spec: Dict[str, Any], arch_id: str, args: argparse.Namespace) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    rows: list[Dict[str, Any]] = []
    row_index = 0
    for source_h in list(pair_spec["source_hparams"]):
        for seed_id in seeds:
            row_index += 1
            row = _base_row(
                args=args,
                pair_id=pair_id,
                pair_spec=pair_spec,
                arch_id=arch_id,
                dataset=str(pair_spec["source_dataset"]),
                hparam_id=source_h,
                seed_id=seed_id,
                row_index=row_index,
            )
            row["phase_kind"] = "source_pretrain"
            row["transfer_mode"] = "none"
            row["source_hparam_id"] = str(source_h)
            row["target_hparam_id"] = ""
            row["source_run_phase"] = ""
            row["source_checkpoint"] = ""
            row["setting_group"] = "source_pretrain"
            row["setting_key"] = f"SRC_{pair_id}_{arch_id}_{source_h}"
            row["setting_id"] = f"{PHASE_ID}_SRC_{pair_id}_{arch_id}_{source_h}_S{int(seed_id)}"
            row["setting_desc"] = f"{pair_id} source pretrain {arch_id} {source_h}"
            row["setting_detail"] = f"source_dataset={pair_spec['source_dataset']} target_dataset={pair_spec['target_dataset']}"
            row["run_id"] = f"{pair_id}_{arch_id}_SRC_{source_h}_S{int(seed_id)}"
            row["run_phase"] = f"{PHASE_ID}_SRC_{pair_id}_{arch_id}_{source_h}_S{int(seed_id)}"
            row["exp_brief"] = f"{arch_id} source {source_h}"
            row.update(_source_row_artifacts(row))
            row["source_checkpoint_export_path"] = str(row["checkpoint_path"])
            row["source_metadata_path"] = str(row["metadata_path"])
            rows.append(row)
    return rows


def _make_target_row(
    *,
    args: argparse.Namespace,
    pair_id: str,
    pair_spec: Dict[str, Any],
    arch_id: str,
    transfer_mode: str,
    source_h: str,
    target_h: str,
    seed_id: int,
    row_index: int,
    source_checkpoint: str,
    source_run_phase: str,
) -> Dict[str, Any]:
    row = _base_row(
        args=args,
        pair_id=pair_id,
        pair_spec=pair_spec,
        arch_id=arch_id,
        dataset=str(pair_spec["target_dataset"]),
        hparam_id=target_h,
        seed_id=seed_id,
        row_index=row_index,
    )
    mode_label = {
        "none": "native",
        "macro_feature_encoder": "feature_init",
        "macro_group_router": "group_init",
        "macro_encoder_all": "encoder_all_init",
        "full_model": "full_arch_init",
    }[transfer_mode]
    row["phase_kind"] = "target_sweep"
    row["transfer_mode"] = str(transfer_mode)
    row["source_hparam_id"] = str(source_h)
    row["target_hparam_id"] = str(target_h)
    row["source_checkpoint"] = str(source_checkpoint)
    row["source_run_phase"] = str(source_run_phase)
    row["setting_group"] = f"target_{mode_label}"
    row["setting_key"] = f"{mode_label}_{pair_id}_{arch_id}_{source_h or 'native'}_{target_h}"
    row["setting_id"] = f"{PHASE_ID}_TGT_{pair_id}_{arch_id}_{mode_label}_{source_h or 'native'}_{target_h}_S{int(seed_id)}"
    row["setting_desc"] = f"{pair_id} {arch_id} {mode_label} {source_h or 'native'}->{target_h}"
    row["setting_detail"] = f"source={pair_spec['source_dataset']} target={pair_spec['target_dataset']} mode={transfer_mode}"
    row["run_id"] = f"{pair_id}_{arch_id}_{mode_label}_{source_h or 'native'}_{target_h}_S{int(seed_id)}"
    row["run_phase"] = f"{PHASE_ID}_TGT_{pair_id}_{arch_id}_{mode_label}_{source_h or 'native'}_{target_h}_S{int(seed_id)}"
    row["exp_brief"] = f"{arch_id} {mode_label} {target_h}"
    return row


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_result_index_compatible(dataset: str) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    candidate_axes = [str(AXIS)]
    axis_lower = str(AXIS).lower()
    if axis_lower not in candidate_axes:
        candidate_axes.append(axis_lower)

    for axis_name in candidate_axes:
        index = _load_result_index(dataset, axis_name)
        for run_phase, rec in dict(index or {}).items():
            prev = merged.get(run_phase)
            if prev is None or float(rec.get("mtime", 0.0) or 0.0) >= float(prev.get("mtime", 0.0) or 0.0):
                merged[run_phase] = dict(rec)
    return merged


def _lookup_result_row(result_index: Dict[str, Dict[str, Any]], run_phase: str) -> Dict[str, Any]:
    direct = dict(result_index.get(run_phase, {}) or {})
    if direct:
        return direct
    phase_key = str(run_phase).casefold()
    for key, value in result_index.items():
        if str(key).casefold() == phase_key:
            return dict(value or {})
    return {}


def _build_source_record_from_result(row: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(row["dataset"])
    run_phase = str(row["run_phase"])
    result_index = _load_result_index_compatible(dataset)
    result_row = _lookup_result_row(result_index, run_phase)
    if not result_row:
        raise RuntimeError(
            f"Missing source result index row for run_phase={run_phase} dataset={dataset} "
            f"(checked axis={AXIS} and axis={AXIS.lower()})"
        )

    result_path_text = str(result_row.get("path", "") or "").strip()
    if not result_path_text:
        raise RuntimeError(f"Source result index row has empty path for run_phase={run_phase} dataset={dataset}")

    result_path = Path(result_path_text)
    if not result_path.exists():
        raise RuntimeError(f"Missing source result for run_phase={row['run_phase']}: {result_path}")
    if not result_path.is_file():
        raise RuntimeError(
            f"Invalid source result path for run_phase={run_phase}: expected file, got {result_path}"
        )

    payload = _load_json(result_path)
    checkpoint_path = str(payload.get("best_checkpoint_file", "") or row.get("checkpoint_path", "") or "").strip()
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise RuntimeError(
            f"Missing exported source checkpoint for run_phase={row['run_phase']} "
            f"(expected {checkpoint_file})"
        )
    best_params = dict(payload.get("best_params", {}) or {})
    metadata = {
        "architecture_id": str(row["architecture_id"]),
        "dataset": str(row["dataset"]),
        "hparam_id": str(row["hparam_id"]),
        "wrapper_alias": str(ARCH_WRAPPER_ALIAS.get(str(row["architecture_id"]), "")),
        "best_valid": float(payload.get("best_mrr@20", 0.0) or 0.0),
        "test_at_best": float(payload.get("test_mrr@20", 0.0) or 0.0),
        "best_learning_rate": float(best_params.get("learning_rate", 0.0) or 0.0),
        "checkpoint_path": str(checkpoint_file.resolve()),
        "result_path": str(result_path.resolve()),
        "run_phase": str(row["run_phase"]),
        "pair_id": str(row["pair_id"]),
        "seed_id": int(row["seed_id"]),
    }
    meta_path = Path(str(row["metadata_path"]))
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _resolve_source_record(row: Dict[str, Any], *, allow_missing: bool) -> Dict[str, Any]:
    meta_path = Path(str(row["metadata_path"]))
    if meta_path.exists():
        metadata = _load_json(meta_path)
        checkpoint_path = Path(str(metadata.get("checkpoint_path", "") or ""))
        if checkpoint_path.exists():
            return metadata
    if allow_missing:
        return {
            "checkpoint_path": str(row["checkpoint_path"]),
            "run_phase": str(row["run_phase"]),
        }
    return _build_source_record_from_result(row)


def _build_target_rows(
    pair_id: str,
    pair_spec: Dict[str, Any],
    arch_id: str,
    args: argparse.Namespace,
    source_rows: list[Dict[str, Any]],
    *,
    allow_missing_checkpoints: bool,
) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    source_lookup: Dict[tuple[str, int], Dict[str, Any]] = {}
    for row in source_rows:
        source_lookup[(str(row["hparam_id"]), int(row["seed_id"]))] = _resolve_source_record(
            row,
            allow_missing=allow_missing_checkpoints,
        )

    rows: list[Dict[str, Any]] = []
    row_index = 10_000
    for target_h in _native_target_hparams(pair_spec):
        for seed_id in seeds:
            row_index += 1
            rows.append(
                _make_target_row(
                    args=args,
                    pair_id=pair_id,
                    pair_spec=pair_spec,
                    arch_id=arch_id,
                    transfer_mode="none",
                    source_h="",
                    target_h=target_h,
                    seed_id=seed_id,
                    row_index=row_index,
                    source_checkpoint="",
                    source_run_phase="",
                )
            )

    for source_h in list(pair_spec["source_hparams"]):
        for target_h in list((pair_spec.get("target_map") or {}).get(source_h, []) or []):
            for seed_id in seeds:
                source_record = dict(source_lookup[(str(source_h), int(seed_id))])
                source_checkpoint = str(source_record.get("checkpoint_path", "") or "")
                source_run_phase = str(source_record.get("run_phase", "") or "")
                for transfer_mode in ("macro_feature_encoder", "macro_group_router", "macro_encoder_all"):
                    row_index += 1
                    rows.append(
                        _make_target_row(
                            args=args,
                            pair_id=pair_id,
                            pair_spec=pair_spec,
                            arch_id=arch_id,
                            transfer_mode=transfer_mode,
                            source_h=source_h,
                            target_h=target_h,
                            seed_id=seed_id,
                            row_index=row_index,
                            source_checkpoint=source_checkpoint,
                            source_run_phase=source_run_phase,
                        )
                    )
                if str(target_h) == str(source_h):
                    row_index += 1
                    rows.append(
                        _make_target_row(
                            args=args,
                            pair_id=pair_id,
                            pair_spec=pair_spec,
                            arch_id=arch_id,
                            transfer_mode="full_model",
                            source_h=source_h,
                            target_h=target_h,
                            seed_id=seed_id,
                            row_index=row_index,
                            source_checkpoint=source_checkpoint,
                            source_run_phase=source_run_phase,
                        )
                    )
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    h = dict(final_base.HPARAM_BANK[str(row["hparam_id"])])
    dataset = str(row["dataset"])
    phase_kind = str(row["phase_kind"])
    is_source = phase_kind == "source_pretrain"
    max_evals = int(args.source_max_evals if is_source else args.target_max_evals)
    tune_epochs = int(args.source_tune_epochs if is_source else args.target_tune_epochs)
    tune_patience = int(args.source_tune_patience if is_source else args.target_tune_patience)
    train_batch_size = _dataset_train_batch_size(dataset)
    eval_batch_size = _dataset_eval_batch_size(dataset)
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")

    if is_source:
        lr_values = list(SOURCE_LR_BANDS[dataset])
        lr_space_type = "loguniform"
    else:
        lr_values = list(TARGET_LR_CHOICES[dataset])
        lr_space_type = "choice"

    hidden_dropout = float(h["fixed_hidden_dropout_prob"])
    if str(row["hparam_id"]).upper() == "H3" and abs(hidden_dropout - 0.18) < 1e-12:
        hidden_dropout = 0.1

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--max-evals",
        str(max_evals),
        "--tune-epochs",
        str(tune_epochs),
        "--tune-patience",
        str(tune_patience),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={dataset}",
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
        "fmoe_artifact_logging_policy=combo_best",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(row['architecture_id'])}",
        f"++fmoe_architecture_key={hydra_literal(row['architecture_key'])}",
        f"++fmoe_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(train_batch_size)}",
        f"eval_batch_size={int(eval_batch_size)}",
        f"embedding_size={int(h['embedding_size'])}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(h['d_ff'])}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(h['d_expert_hidden'])}",
        f"d_router_hidden={int(h['d_router_hidden'])}",
        f"expert_scale={int(args.expert_scale)}",
        f"++search.learning_rate={hydra_literal(lr_values)}",
        f"++search.weight_decay={hydra_literal([float(h['fixed_weight_decay'])])}",
        f"++search.hidden_dropout_prob={hydra_literal([hidden_dropout])}",
        f"++search.lr_scheduler_type={hydra_literal(['warmup_cosine'])}",
        f"++search_space_type_overrides.learning_rate={hydra_literal(lr_space_type)}",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(AXIS_ID)}",
        f"++phase_axis_desc={hydra_literal(AXIS_DESC)}",
        f"++phase_setting_id={hydra_literal(row['setting_id'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++phase_seed_id={hydra_literal(row['seed_id'])}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
        f"++phase_transfer_pair_id={hydra_literal(row['pair_id'])}",
        f"++phase_transfer_mode={hydra_literal(row['transfer_mode'])}",
        f"++phase_transfer_source_dataset={hydra_literal(row['source_dataset'])}",
        f"++phase_transfer_target_dataset={hydra_literal(row['target_dataset'])}",
        f"++phase_transfer_source_hparam_id={hydra_literal(row['source_hparam_id'])}",
        f"++phase_transfer_target_hparam_id={hydra_literal(row['target_hparam_id'])}",
    ]

    if is_source:
        cmd.append(f"++__artifact_combo_best_export_path={hydra_literal(row['source_checkpoint_export_path'])}")
        cmd.append("++transfer.enable=false")
        cmd.append("++transfer.mode=none")
    else:
        if str(row["transfer_mode"]) == "none":
            cmd.append("++transfer.enable=false")
            cmd.append("++transfer.mode=none")
        else:
            strict_shape = str(row["transfer_mode"]) == "full_model"
            cmd.append("++transfer.enable=true")
            cmd.append(f"++transfer.mode={hydra_literal(row['transfer_mode'])}")
            cmd.append(f"++transfer.source_checkpoint={hydra_literal(row['source_checkpoint'])}")
            cmd.append(f"++transfer.source_architecture={hydra_literal(row['architecture_id'])}")
            cmd.append(f"++transfer.strict_shape={hydra_literal(strict_shape)}")

    for key, value in dict(wrapper_base._arch_overrides(str(row["architecture_id"]), args) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd


def _build_log_path(log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    pair_id = sanitize_token(str(row.get("pair_id", "pair")), upper=False)
    arch = sanitize_token(str(row.get("architecture_id", "A12")), upper=True)
    phase_kind = sanitize_token(str(row.get("phase_kind", "phase")), upper=False)
    hparam = sanitize_token(str(row.get("hparam_id", "H1")), upper=True)
    phase = sanitize_token(phase_id, upper=True)
    setting_id = sanitize_token(str(row.get("setting_id", "setting")), upper=True)
    setting_desc = sanitize_token(str(row.get("setting_desc", "stageA")), upper=True)
    filename = f"{phase}_{setting_id}_{setting_desc}.log"
    return log_dir / pair_id / arch / phase_kind / hparam / filename


def _validate_all_pair_datasets(pair_ids: list[str]) -> None:
    seen = set()
    for pair_id in pair_ids:
        pair = dict(PAIR_SPECS[pair_id])
        for dataset in (str(pair["source_dataset"]), str(pair["target_dataset"])):
            if dataset in seen:
                continue
            final_base._validate_session_fixed_files(dataset)
            seen.add(dataset)


def _collect_source_rows(pair_ids: list[str], architectures: list[str], args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for pair_id in pair_ids:
        pair_spec = dict(PAIR_SPECS[pair_id])
        for arch_id in architectures:
            rows.extend(_build_source_rows(pair_id, pair_spec, arch_id, args))
    return rows


def _collect_target_rows(
    pair_ids: list[str],
    architectures: list[str],
    args: argparse.Namespace,
    source_rows_by_key: Dict[tuple[str, str], list[Dict[str, Any]]],
    *,
    allow_missing_checkpoints: bool,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for pair_id in pair_ids:
        pair_spec = dict(PAIR_SPECS[pair_id])
        for arch_id in architectures:
            key = (pair_id, arch_id)
            rows.extend(
                _build_target_rows(
                    pair_id,
                    pair_spec,
                    arch_id,
                    args,
                    list(source_rows_by_key[key]),
                    allow_missing_checkpoints=allow_missing_checkpoints,
                )
            )
    return rows


def _source_rows_by_pair_arch(source_rows: list[Dict[str, Any]]) -> Dict[tuple[str, str], list[Dict[str, Any]]]:
    grouped: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for row in source_rows:
        key = (str(row["pair_id"]), str(row["architecture_id"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _print_phase_counts(label: str, rows: list[Dict[str, Any]]) -> None:
    print(f"[{PHASE_ID}] {label} rows={len(rows)}")
    by_pair: Dict[str, int] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        by_pair[pair_id] = by_pair.get(pair_id, 0) + 1
    for pair_id, count in sorted(by_pair.items()):
        print(f"[{PHASE_ID}]   pair={pair_id} rows={count}")


def main() -> int:
    args = parse_args()
    if args.source_only and args.target_only:
        raise RuntimeError("--source-only and --target-only cannot be used together")

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs provided")
    architectures = _selected_architectures(args)
    pair_ids = _selected_pairs(args)
    _validate_all_pair_datasets(pair_ids)

    extra_cols = [
        "phase_id",
        "axis_id",
        "axis_desc",
        "pair_id",
        "phase_kind",
        "architecture_id",
        "architecture_key",
        "architecture_name",
        "setting_id",
        "setting_key",
        "setting_desc",
        "setting_group",
        "setting_detail",
        "hparam_id",
        "source_hparam_id",
        "target_hparam_id",
        "transfer_mode",
        "source_dataset",
        "target_dataset",
        "source_checkpoint",
        "source_run_phase",
        "run_id",
    ]
    fieldnames = build_summary_fieldnames(extra_cols)

    source_rows = _collect_source_rows(pair_ids, architectures, args)
    source_rows_by_key = _source_rows_by_pair_arch(source_rows)

    if not args.target_only:
        _print_phase_counts("source_pretrain", source_rows)
        rc = launch_wide_rows(
            rows=source_rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=LOG_ROOT / AXIS,
            summary_path=_pair_summary_path(pair_ids[0]),
            fieldnames=fieldnames,
            extra_cols=extra_cols,
            build_command=_build_command,
            build_log_path=_build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: _pair_summary_path(str(row.get("pair_id", ""))),
        )
        if rc != 0:
            return int(rc)

        if not args.dry_run:
            for row in source_rows:
                _resolve_source_record(row, allow_missing=False)

    if args.source_only:
        return 0

    target_rows = _collect_target_rows(
        pair_ids,
        architectures,
        args,
        source_rows_by_key,
        allow_missing_checkpoints=bool(args.dry_run),
    )
    _print_phase_counts("target_sweep", target_rows)
    rc = launch_wide_rows(
        rows=target_rows,
        gpus=gpus,
        args=args,
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        log_dir=LOG_ROOT / AXIS,
        summary_path=_pair_summary_path(pair_ids[0]),
        fieldnames=fieldnames,
        extra_cols=extra_cols,
        build_command=_build_command,
        build_log_path=_build_log_path,
        verify_logging=bool(args.verify_logging),
        summary_path_for_row=lambda row: _pair_summary_path(str(row.get("pair_id", ""))),
    )
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
