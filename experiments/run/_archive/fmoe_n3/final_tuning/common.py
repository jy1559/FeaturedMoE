#!/usr/bin/env python3
"""Common helpers for A12-only final tuning runners."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

THIS_DIR = Path(__file__).resolve().parent
FMOE_RUN_DIR = THIS_DIR.parent
if str(FMOE_RUN_DIR) not in sys.path:
    sys.path.append(str(FMOE_RUN_DIR))

import run_final_all_datasets as base  # noqa: E402
from run_phase9_auxloss import (  # noqa: E402
    LOG_ROOT,
    TRACK,
    _apply_base_overrides,
    _base_fixed_overrides,
    _load_result_index,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows, sanitize_token  # noqa: E402


REPO_ROOT_REAL = THIS_DIR.parents[3]
AXIS = "Final_tuning_A12"
AXIS_ID = "FTA12"
AXIS_DESC = "final_tuning_a12"
ARCH_ID = "A12"
ARCH_KEY = "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5"
ARCH_NAME = "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5"
PHASE_BY_STAGE = {
    "stage1": "P16A",
    "stage2": "P16B",
    "stage3": "P16C",
}
PHASE_NAME_BY_STAGE = {
    "stage1": "A12_FINAL_TUNING_STAGE1",
    "stage2": "A12_FINAL_TUNING_STAGE2",
    "stage3": "A12_FINAL_TUNING_STAGE3",
}
WEAK_DATASETS = {"amazon_beauty", "lastfm0.03", "movielens1m", "retail_rocket"}
STRONG_DATASETS = {"KuaiRecLargeStrictPosV2_0.2", "foursquare"}
DATASET_ORDER = list(base.DEFAULT_DATASETS)
DEFAULT_BATCH_SIZE = 4096
DEFAULT_DATASET_BATCH_SIZES = {"movielens1m": 8192, "retail_rocket": 8192}
DEFAULT_DATASET_EVAL_BATCH_SIZES = {"movielens1m": 12288, "retail_rocket": 12288}

STAGE_BUDGETS = {
    "stage1": {
        "weak": {"epochs": 35, "patience": 4, "max_evals": 20, "search_algo": "random"},
        "strong": {"epochs": 30, "patience": 4, "max_evals": 20, "search_algo": "random"},
    },
    "stage2": {
        "weak": {"epochs": 60, "patience": 6, "max_evals": 14, "search_algo": "tpe"},
        "strong": {"epochs": 50, "patience": 6, "max_evals": 12, "search_algo": "tpe"},
    },
    "stage3": {
        "weak": {"epochs": 100, "patience": 10, "max_evals": 18, "search_algo": "tpe"},
        "strong": {"epochs": 100, "patience": 10, "max_evals": 16, "search_algo": "tpe"},
    },
}

DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "amazon_beauty": {
        "group": "weak",
        "dense_easy": False,
        "lr_ladder": [8e-5, 1.8e-4, 4e-4, 9e-4, 1.35e-3],
        "max_len": [20, 30, 40, 50],
        "d_feat_emb": [8, 12, 16],
        "expert_scale": [2, 3],
        "family_dropout": [0.02, 0.04, 0.06],
        "feature_dropout": [0.00, 0.02, 0.05],
        "default_feature_dropout": 0.0,
    },
    "lastfm0.03": {
        "group": "weak",
        "dense_easy": True,
        "lr_ladder": [1e-4, 2.2e-4, 5e-4, 7.5e-4, 1.1e-3],
        "max_len": [15, 20, 30, 40],
        "d_feat_emb": [8, 12],
        "expert_scale": [2, 3],
        "family_dropout": [0.03, 0.05, 0.07],
        "feature_dropout": [0.02, 0.05, 0.08],
        "default_feature_dropout": 0.02,
    },
    "movielens1m": {
        "group": "weak",
        "dense_easy": True,
        "lr_ladder": [1e-4, 2.2e-4, 5e-4, 7.5e-4, 1.1e-3],
        "max_len": [20, 30, 40],
        "d_feat_emb": [8, 12],
        "expert_scale": [2, 3],
        "family_dropout": [0.03, 0.05, 0.07],
        "feature_dropout": [0.02, 0.05, 0.08],
        "default_feature_dropout": 0.02,
    },
    "retail_rocket": {
        "group": "weak",
        "dense_easy": False,
        "lr_ladder": [1.5e-4, 3.5e-4, 8e-4, 1.8e-3, 2.7e-3],
        "max_len": [20, 30, 40],
        "d_feat_emb": [12, 16],
        "expert_scale": [3, 4],
        "family_dropout": [0.02, 0.04, 0.06],
        "feature_dropout": [0.00, 0.02, 0.05],
        "default_feature_dropout": 0.0,
    },
    "KuaiRecLargeStrictPosV2_0.2": {
        "group": "strong",
        "dense_easy": False,
        "lr_ladder": [1.5e-4, 3.5e-4, 8e-4, 1.8e-3, 2.7e-3],
        "max_len": [20, 30, 40, 50],
        "d_feat_emb": [12, 16, 24],
        "expert_scale": [3, 4],
        "family_dropout": [0.02, 0.04, 0.06],
        "feature_dropout": [0.00, 0.02, 0.05],
        "default_feature_dropout": 0.0,
    },
    "foursquare": {
        "group": "strong",
        "dense_easy": False,
        "lr_ladder": [1.5e-4, 3.5e-4, 8e-4, 1.2e-3, 1.8e-3],
        "max_len": [15, 20, 30, 40],
        "d_feat_emb": [12, 16],
        "expert_scale": [3, 4],
        "family_dropout": [0.02, 0.04, 0.06],
        "feature_dropout": [0.00, 0.02, 0.05],
        "default_feature_dropout": 0.0,
    },
}

FAMILY_BANK: Dict[str, list[Dict[str, Any]]] = {
    "amazon_beauty": [
        {"family_id": "AB_H10_feature_light", "family_group": "feature_light", "capacity_anchor": "H10", "d_feat_emb": [8, 12], "expert_scale": [2], "max_len": [20, 30, 40]},
        {"family_id": "AB_H13_balanced", "family_group": "balanced", "capacity_anchor": "H13"},
        {"family_id": "AB_H14_capacity_control", "family_group": "capacity_control", "capacity_anchor": "H14", "d_feat_emb": [12, 16], "expert_scale": [2, 3], "max_len": [20, 30]},
        {"family_id": "AB_H3_long_context", "family_group": "long_context", "capacity_anchor": "H3", "max_len": [30, 40, 50], "d_feat_emb": [8, 12]},
        {"family_id": "AB_H10_low_drop", "family_group": "low_drop", "capacity_anchor": "H10"},
        {"family_id": "AB_H13_low_feat_dropout", "family_group": "low_feat_dropout", "capacity_anchor": "H13", "feature_dropout": [0.00, 0.02]},
        {"family_id": "AB_H8_small_regularized", "family_group": "small_regularized", "capacity_anchor": "H8", "d_feat_emb": [8, 12], "expert_scale": [2]},
    ],
    "lastfm0.03": [
        {"family_id": "LFM_H3_feature_light", "family_group": "feature_light", "capacity_anchor": "H3", "d_feat_emb": [8], "expert_scale": [2]},
        {"family_id": "LFM_H7_feature_light", "family_group": "feature_light", "capacity_anchor": "H7", "d_feat_emb": [8, 12], "expert_scale": [2]},
        {"family_id": "LFM_H11_balanced", "family_group": "balanced", "capacity_anchor": "H11"},
        {"family_id": "LFM_H15_medium_context", "family_group": "medium_context", "capacity_anchor": "H15", "max_len": [20, 30, 40]},
        {"family_id": "LFM_H3_regularized", "family_group": "regularized", "capacity_anchor": "H3"},
        {"family_id": "LFM_H9_small_head_light", "family_group": "small_regularized", "capacity_anchor": "H9", "d_feat_emb": [8], "expert_scale": [2]},
    ],
    "movielens1m": [
        {"family_id": "ML_H8_feature_light", "family_group": "feature_light", "capacity_anchor": "H8", "d_feat_emb": [8], "expert_scale": [2]},
        {"family_id": "ML_H3_balanced", "family_group": "balanced", "capacity_anchor": "H3"},
        {"family_id": "ML_H9_compact_reg", "family_group": "small_regularized", "capacity_anchor": "H9", "d_feat_emb": [8], "expert_scale": [2]},
        {"family_id": "ML_H5_control", "family_group": "capacity_control", "capacity_anchor": "H5"},
        {"family_id": "ML_H8_long_context", "family_group": "long_context", "capacity_anchor": "H8", "max_len": [20, 30, 40]},
        {"family_id": "ML_H12_small_regularized", "family_group": "small_regularized", "capacity_anchor": "H12", "d_feat_emb": [8], "expert_scale": [2]},
    ],
    "retail_rocket": [
        {"family_id": "RR_H2_balanced", "family_group": "balanced", "capacity_anchor": "H2"},
        {"family_id": "RR_H1_feature_strong", "family_group": "feature_strong", "capacity_anchor": "H1", "d_feat_emb": [12, 16], "expert_scale": [4]},
        {"family_id": "RR_H3_feature_light", "family_group": "feature_light", "capacity_anchor": "H3", "d_feat_emb": [12], "expert_scale": [3]},
        {"family_id": "RR_H6_control", "family_group": "capacity_control", "capacity_anchor": "H6"},
        {"family_id": "RR_H2_context_probe", "family_group": "long_context", "capacity_anchor": "H2", "max_len": [20, 30, 40]},
        {"family_id": "RR_H5_wider_control", "family_group": "capacity_control", "capacity_anchor": "H5"},
    ],
    "KuaiRecLargeStrictPosV2_0.2": [
        {"family_id": "KU_H14_feature_strong", "family_group": "feature_strong", "capacity_anchor": "H14", "d_feat_emb": [16, 24], "expert_scale": [4]},
        {"family_id": "KU_H7_balanced", "family_group": "balanced", "capacity_anchor": "H7"},
        {"family_id": "KU_H2_regularized", "family_group": "regularized", "capacity_anchor": "H2"},
        {"family_id": "KU_H10_long_context", "family_group": "long_context", "capacity_anchor": "H10", "max_len": [30, 40, 50]},
    ],
    "foursquare": [
        {"family_id": "FSQ_H3_balanced", "family_group": "balanced", "capacity_anchor": "H3"},
        {"family_id": "FSQ_H15_feature_strong", "family_group": "feature_strong", "capacity_anchor": "H15", "expert_scale": [4]},
        {"family_id": "FSQ_H5_feature_light", "family_group": "feature_light", "capacity_anchor": "H5", "d_feat_emb": [12], "expert_scale": [3]},
        {"family_id": "FSQ_H1_regularized", "family_group": "regularized", "capacity_anchor": "H1"},
    ],
}

NEIGHBOR_ANCHOR = {
    "amazon_beauty": {"H10": "H13", "H13": "H14", "H14": "H13", "H3": "H10", "H8": "H10"},
    "lastfm0.03": {"H3": "H7", "H7": "H11", "H11": "H15", "H15": "H11", "H9": "H7"},
    "movielens1m": {"H8": "H3", "H3": "H9", "H9": "H5", "H5": "H3", "H12": "H9"},
    "retail_rocket": {"H2": "H1", "H1": "H3", "H3": "H6", "H6": "H3", "H5": "H2"},
    "KuaiRecLargeStrictPosV2_0.2": {"H14": "H7", "H7": "H2", "H2": "H7", "H10": "H14"},
    "foursquare": {"H3": "H15", "H15": "H5", "H5": "H3", "H1": "H3"},
}


def axis_root() -> Path:
    root = LOG_ROOT / AXIS
    root.mkdir(parents=True, exist_ok=True)
    return root


def stage_manifest_path(stage_name: str, manifest_out: str = "") -> Path:
    if manifest_out:
        raw = Path(str(manifest_out))
        if raw.suffix:
            return raw.with_name(f"{raw.stem}_{stage_name}{raw.suffix}")
        return raw / f"{stage_name}_manifest.json"
    return axis_root() / f"{stage_name}_manifest.json"


def phase_log_dir(dataset: str) -> Path:
    path = axis_root() / str(dataset)
    path.mkdir(parents=True, exist_ok=True)
    return path


def summary_path(dataset: str) -> Path:
    return phase_log_dir(dataset) / "summary.csv"


def _dataset_profile(dataset: str) -> Dict[str, Any]:
    if dataset not in DATASET_PROFILES:
        raise KeyError(f"Unknown dataset profile: {dataset}")
    return dict(DATASET_PROFILES[dataset])


def _group_for_dataset(dataset: str) -> str:
    return str(_dataset_profile(dataset)["group"])


def _parse_dataset_int_map(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        ds, raw_val = chunk.split(":", 1)
        ds = ds.strip()
        raw_val = raw_val.strip()
        if not ds or not raw_val:
            continue
        try:
            out[ds] = int(raw_val)
        except ValueError:
            continue
    return out


def _effective_dataset_int_map(
    dataset: str,
    *,
    default: int,
    override_text: str,
) -> int:
    value = int(_parse_dataset_int_map(override_text).get(dataset, default))
    return max(value, 1)


def _validate_session_fixed_files(dataset: str) -> None:
    ds_dir = REPO_ROOT_REAL / "Datasets" / "processed" / "feature_added_v3" / dataset
    required = [
        ds_dir / f"{dataset}.train.inter",
        ds_dir / f"{dataset}.valid.inter",
        ds_dir / f"{dataset}.test.inter",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"session_fixed files missing for dataset={dataset}: {missing}")


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


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _stage_map_choices(values: Iterable[float]) -> list[Dict[str, Any]]:
    return [_all_stage_map(float(v)) for v in values]


def _anchor_cfg(hid: str) -> Dict[str, Any]:
    return dict(base.HPARAM_BANK[str(hid)])


def _default_num_heads() -> int:
    return 4


def _phase_id(stage_name: str) -> str:
    return str(PHASE_BY_STAGE[stage_name])


def _phase_name(stage_name: str) -> str:
    return str(PHASE_NAME_BY_STAGE[stage_name])


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    requested = _parse_csv_strings(args.datasets)
    selected: list[str] = []
    for dataset in requested:
        if dataset not in DATASET_ORDER:
            raise RuntimeError(f"Unknown dataset={dataset}. Supported: {','.join(DATASET_ORDER)}")
        if dataset not in selected:
            selected.append(dataset)
    if not selected:
        raise RuntimeError("No datasets selected")
    return selected


def _build_a12_overrides(args: argparse.Namespace, *, family_dropout_map: Dict[str, Any], feature_dropout_map: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = {
        "wrapper_map": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
        "source_profile": "src_abc_feature",
        "bias_mode": "bias_both",
    }
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
    )
    overrides["z_loss_lambda"] = float(args.z_loss_lambda)
    overrides["balance_loss_lambda"] = float(args.balance_loss_lambda)
    overrides["macro_history_window"] = int(args.macro_history_window)
    overrides["route_consistency_pairs"] = 1
    overrides["route_consistency_lambda"] = float(args.a2_route_consistency_lambda)
    overrides["route_consistency_min_sim"] = float(args.a2_route_consistency_min_sim)
    overrides["z_loss_lambda"] = float(args.a2_z_loss_lambda)
    overrides["route_monopoly_lambda"] = 0.0
    overrides["balance_loss_lambda"] = 0.0
    overrides["layer_layout"] = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
    overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
    overrides["stage_family_dropout_prob"] = dict(family_dropout_map)
    overrides["stage_feature_dropout_prob"] = dict(feature_dropout_map)
    overrides["stage_feature_dropout_scope"] = _all_stage_map("token")
    overrides["bias_mode"] = "none"
    overrides["rule_bias_scale"] = 0.0
    overrides["feature_group_bias_lambda"] = 0.0
    return overrides


def _family_specs(dataset: str) -> list[Dict[str, Any]]:
    return [dict(spec) for spec in FAMILY_BANK[str(dataset)]]


def _anchor_hidden_choices(anchor: str, family_group: str) -> list[float]:
    base_dropout = float(_anchor_cfg(anchor)["fixed_hidden_dropout_prob"])
    if family_group in {"low_drop", "feature_strong", "capacity_control"}:
        values = [max(0.05, base_dropout - 0.06), max(0.06, base_dropout - 0.03), base_dropout]
    elif family_group in {"regularized", "small_regularized"}:
        values = [base_dropout, min(0.22, base_dropout + 0.03), min(0.26, base_dropout + 0.06)]
    else:
        values = [max(0.05, base_dropout - 0.03), base_dropout, min(0.24, base_dropout + 0.03)]
    return _dedupe_keep_order(round(float(v), 4) for v in values)


def _base_stage1_search(dataset: str, family: Dict[str, Any]) -> Dict[str, list[Any]]:
    profile = _dataset_profile(dataset)
    anchor = str(family["capacity_anchor"])
    search: Dict[str, list[Any]] = {
        "learning_rate": _dedupe_keep_order(profile["lr_ladder"]),
        "hidden_dropout_prob": _anchor_hidden_choices(anchor, str(family["family_group"])),
        "MAX_ITEM_LIST_LENGTH": _dedupe_keep_order(family.get("max_len", profile["max_len"])),
        "d_feat_emb": _dedupe_keep_order(family.get("d_feat_emb", profile["d_feat_emb"])),
        "expert_scale": _dedupe_keep_order(family.get("expert_scale", profile["expert_scale"])),
        "stage_family_dropout_prob": _stage_map_choices(family.get("family_dropout", profile["family_dropout"])),
    }
    if profile["group"] == "weak":
        search["stage_feature_dropout_prob"] = _stage_map_choices(family.get("feature_dropout", profile["feature_dropout"]))
    return search


def _round_float(value: float, ndigits: int = 10) -> float:
    return round(float(value), int(ndigits))


def _float_from_stage_map(value: Any, default: float) -> float:
    if isinstance(value, dict):
        for key in ("macro", "mid", "micro"):
            raw = value.get(key)
            if raw is not None:
                try:
                    return float(raw)
                except Exception:
                    continue
    try:
        return float(value)
    except Exception:
        return float(default)


def _local_window(values: list[Any], center: Any, width: int) -> list[Any]:
    if not values:
        return []
    if center is None:
        return values[: max(1, width)]
    if isinstance(center, dict):
        center_val = _float_from_stage_map(center, 0.0)
        pairs = []
        for idx, value in enumerate(values):
            pairs.append((abs(_float_from_stage_map(value, 0.0) - center_val), idx, value))
        chosen = [item[2] for item in sorted(pairs)[: max(1, width)]]
        return _dedupe_keep_order(v for v in values if any(json.dumps(v, sort_keys=True) == json.dumps(c, sort_keys=True) for c in chosen))
    try:
        center_val = float(center)
        numeric_pairs = []
        for idx, value in enumerate(values):
            try:
                numeric_pairs.append((abs(float(value) - center_val), idx, value))
            except Exception:
                numeric_pairs.append((1e9 + idx, idx, value))
        chosen = [item[2] for item in sorted(numeric_pairs)[: max(1, width)]]
        return _dedupe_keep_order(v for v in values if any(v == c for c in chosen))
    except Exception:
        pass
    return values[: max(1, width)]


def _local_lr_window(dataset: str, center: Any, width: int) -> list[float]:
    return [_round_float(v) for v in _local_window(list(_dataset_profile(dataset)["lr_ladder"]), center, width)]


def _local_max_len_window(dataset: str, center: Any, width: int) -> list[int]:
    return [int(v) for v in _local_window(list(_dataset_profile(dataset)["max_len"]), center, width)]


def _local_d_feat_window(dataset: str, center: Any, width: int) -> list[int]:
    return [int(v) for v in _local_window(list(_dataset_profile(dataset)["d_feat_emb"]), center, width)]


def _local_expert_window(dataset: str, center: Any, width: int) -> list[int]:
    return [int(v) for v in _local_window(list(_dataset_profile(dataset)["expert_scale"]), center, width)]


def _local_family_dropout_window(dataset: str, center: Any, width: int) -> list[Dict[str, Any]]:
    profile = _dataset_profile(dataset)
    return _stage_map_choices(_local_window(list(profile["family_dropout"]), _float_from_stage_map(center, profile["family_dropout"][0]), width))


def _local_feature_dropout_window(dataset: str, center: Any, width: int) -> list[Dict[str, Any]]:
    profile = _dataset_profile(dataset)
    return _stage_map_choices(_local_window(list(profile["feature_dropout"]), _float_from_stage_map(center, profile["default_feature_dropout"]), width))


def _weight_decay_choices(anchor: str, scales: Iterable[float]) -> list[float]:
    base_wd = float(_anchor_cfg(anchor)["fixed_weight_decay"])
    return _dedupe_keep_order(_round_float(base_wd * float(scale), 12) for scale in scales)


def _topk_mean(values: list[float], k: int = 3) -> float:
    valid = sorted([float(v) for v in values], reverse=True)
    if not valid:
        return 0.0
    topk = valid[: max(1, min(int(k), len(valid)))]
    return float(sum(topk) / len(topk))


def _result_payload_for_run(dataset: str, run_phase: str) -> Optional[Dict[str, Any]]:
    index = _load_result_index(dataset, AXIS)
    rec = index.get(str(run_phase))
    if not isinstance(rec, dict):
        return None
    path = Path(str(rec.get("path", "") or ""))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload["_result_path"] = str(path)
    return payload


def _record_from_manifest_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    payload = _result_payload_for_run(str(row["dataset"]), str(row["run_phase"]))
    if payload is None:
        return None
    trials = [t for t in list(payload.get("trials", [])) if str(t.get("status", "")) == "ok"]
    trial_valids: list[float] = []
    for trial in trials:
        valid = (trial.get("valid_result") or {})
        try:
            trial_valids.append(float(valid.get("mrr@20", trial.get("mrr@20", 0.0)) or 0.0))
        except Exception:
            continue
    best_valid = float(payload.get("best_mrr@20", 0.0) or 0.0)
    top3_mean = _topk_mean(trial_valids, k=3)
    return {
        "row": dict(row),
        "payload": payload,
        "best_valid": best_valid,
        "top3_mean": top3_mean,
        "best_params": dict(payload.get("best_params", {}) or {}),
        "test_mrr20": float(payload.get("test_mrr@20", 0.0) or 0.0),
    }


def _selection_key(record: Dict[str, Any]) -> tuple[float, float]:
    return (
        float(record.get("best_valid", 0.0) or 0.0),
        float(record.get("top3_mean", 0.0) or 0.0),
    )


def _selection_score_text(record: Dict[str, Any]) -> str:
    return f"{float(record.get('best_valid', 0.0)):.6f}|{float(record.get('top3_mean', 0.0)):.6f}"


def _load_manifest_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"manifest rows missing: {path}")
    return [dict(row) for row in rows]


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


def write_manifest(stage_name: str, args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = stage_manifest_path(stage_name, getattr(args, "manifest_out", ""))
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": stage_name,
        "phase_id": _phase_id(stage_name),
        "phase_name": _phase_name(stage_name),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_count": len(rows),
        "datasets": sorted({str(row.get("dataset", "")) for row in rows}),
        "rows": [_serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def common_summary_fieldnames() -> list[str]:
    extra_cols = [
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
    ]
    return build_summary_fieldnames(extra_cols)


def summary_path_for_row(row: Dict[str, Any]) -> Path:
    return summary_path(str(row["dataset"]))


def build_log_path(*, log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    dataset_dir = phase_log_dir(str(row["dataset"]))
    family_dir = dataset_dir / str(row["family_id"])
    filename = f"{phase_id}_{sanitize_token(str(row['family_id']), upper=True)}_S{int(row['seed_id'])}.log"
    return family_dir / filename


def _base_cli_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", default=",".join(DATASET_ORDER))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=160000)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--dataset-batch-sizes",
        default="movielens1m:8192,retail_rocket:8192",
        help="CSV map: dataset:int,dataset2:int overriding train_batch_size",
    )
    parser.add_argument(
        "--dataset-eval-batch-sizes",
        default="movielens1m:12288,retail_rocket:12288",
        help="CSV map: dataset:int,dataset2:int overriding eval_batch_size",
    )
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--z-loss-lambda", type=float, default=1e-4)
    parser.add_argument("--balance-loss-lambda", type=float, default=0.0)
    parser.add_argument("--macro-history-window", type=int, default=5)
    parser.add_argument("--a2-route-consistency-lambda", type=float, default=8e-4)
    parser.add_argument("--a2-route-consistency-min-sim", type=float, default=0.995)
    parser.add_argument("--a2-z-loss-lambda", type=float, default=2e-4)
    return parser


def parse_stage_args(stage_name: str, description: str) -> argparse.Namespace:
    parser = _base_cli_parser(description)
    args = parser.parse_args()
    if _parse_csv_strings(args.seeds) != ["1"]:
        print(f"[warn] {stage_name}: final tuning is designed for seed=1; proceeding with provided seeds={args.seeds}")
    return args


def _row_base(
    *,
    dataset: str,
    stage_name: str,
    family_id: str,
    family_group: str,
    variant_id: str,
    capacity_anchor: str,
    selected_from_stage: str,
    selection_score: str,
    seed_id: int,
    runtime_seed: int,
    stage_group: str,
    search_algo: str,
    source_family_id: str,
    fixed_values: Dict[str, Any],
    search_space: Dict[str, list[Any]],
    overrides: Dict[str, Any],
    train_batch_size: int,
    eval_batch_size: int,
    max_evals: int,
    tune_epochs: int,
    tune_patience: int,
) -> Dict[str, Any]:
    phase_id = _phase_id(stage_name)
    dataset_tag = sanitize_token(dataset, upper=True)
    family_tag = sanitize_token(family_id, upper=True)
    run_id = f"{stage_name.upper()}_{dataset_tag}_{family_tag}_S{int(seed_id)}"
    run_phase = f"{phase_id}_{run_id}"
    return {
        "dataset": dataset,
        "phase_id": phase_id,
        "axis_id": AXIS_ID,
        "axis_desc": AXIS_DESC,
        "architecture_id": ARCH_ID,
        "architecture_key": ARCH_KEY,
        "architecture_name": ARCH_NAME,
        "exp_brief": ARCH_NAME,
        "run_phase": run_phase,
        "run_id": run_id,
        "setting_id": family_id,
        "setting_key": family_id,
        "setting_desc": family_id,
        "stage": stage_name,
        "tuning_stage": stage_name,
        "family_id": family_id,
        "family_group": family_group,
        "variant_id": variant_id,
        "capacity_anchor": capacity_anchor,
        "selected_from_stage": selected_from_stage,
        "selection_score": selection_score,
        "search_algo": search_algo,
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": stage_group,
        "source_family_id": source_family_id,
        "fixed_values": dict(fixed_values),
        "search_space": {key: list(values) for key, values in search_space.items()},
        "overrides": dict(overrides),
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "max_evals": int(max_evals),
        "tune_epochs": int(tune_epochs),
        "tune_patience": int(tune_patience),
    }


def _row_budget(stage_name: str, dataset: str) -> Dict[str, Any]:
    group = _group_for_dataset(dataset)
    out = dict(STAGE_BUDGETS[stage_name][group])
    out["stage_group"] = group
    return out


def _row_batch_sizes(dataset: str, args: argparse.Namespace) -> tuple[int, int]:
    train_batch = _effective_dataset_int_map(
        dataset,
        default=int(args.batch_size),
        override_text=str(args.dataset_batch_sizes),
    )
    eval_batch = _effective_dataset_int_map(
        dataset,
        default=train_batch,
        override_text=str(args.dataset_eval_batch_sizes),
    )
    return train_batch, eval_batch


def build_stage1_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    cursor = 0
    seeds = _parse_csv_ints(args.seeds)
    datasets = _selected_datasets(args)
    for dataset in datasets:
        _validate_session_fixed_files(dataset)
        budget = _row_budget("stage1", dataset)
        profile = _dataset_profile(dataset)
        train_batch, eval_batch = _row_batch_sizes(dataset, args)
        for family in _family_specs(dataset):
            for seed_id in seeds:
                cursor += 1
                family_search = _base_stage1_search(dataset, family)
                default_family_drop = float(profile["family_dropout"][0])
                default_feature_drop = float(profile["default_feature_dropout"])
                overrides = _build_a12_overrides(
                    args,
                    family_dropout_map=_all_stage_map(default_family_drop),
                    feature_dropout_map=_all_stage_map(default_feature_drop),
                )
                anchor_cfg = _anchor_cfg(str(family["capacity_anchor"]))
                fixed_values = {
                    "embedding_size": int(anchor_cfg["embedding_size"]),
                    "d_ff": int(anchor_cfg["d_ff"]),
                    "d_expert_hidden": int(anchor_cfg["d_expert_hidden"]),
                    "d_router_hidden": int(anchor_cfg["d_router_hidden"]),
                    "weight_decay": float(anchor_cfg["fixed_weight_decay"]),
                    "num_heads": _default_num_heads(),
                    "attn_dropout_prob": 0.10,
                    "lr_scheduler_type": "warmup_cosine",
                    "stage_feature_dropout_prob": _all_stage_map(default_feature_drop),
                }
                if profile["group"] == "weak":
                    fixed_values.pop("stage_feature_dropout_prob", None)
                rows.append(
                    _row_base(
                        dataset=dataset,
                        stage_name="stage1",
                        family_id=str(family["family_id"]),
                        family_group=str(family["family_group"]),
                        variant_id="family_sweep",
                        capacity_anchor=str(family["capacity_anchor"]),
                        selected_from_stage="manual_family_bank",
                        selection_score="",
                        seed_id=int(seed_id),
                        runtime_seed=int(args.seed_base) + cursor - 1,
                        stage_group=str(budget["stage_group"]),
                        search_algo=str(budget["search_algo"]),
                        source_family_id=str(family["family_id"]),
                        fixed_values=fixed_values,
                        search_space=family_search,
                        overrides=overrides,
                        train_batch_size=train_batch,
                        eval_batch_size=eval_batch,
                        max_evals=int(budget["max_evals"]),
                        tune_epochs=int(budget["epochs"]),
                        tune_patience=int(budget["patience"]),
                    )
                )
    return rows


def _select_stage1_records(args: argparse.Namespace) -> Dict[str, list[Dict[str, Any]]]:
    manifest_rows = _load_manifest_rows(stage_manifest_path("stage1"))
    selected_datasets = set(_selected_datasets(args))
    by_dataset: Dict[str, list[Dict[str, Any]]] = {dataset: [] for dataset in selected_datasets}
    for row in manifest_rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in selected_datasets:
            continue
        record = _record_from_manifest_row(row)
        if record is None:
            continue
        by_dataset.setdefault(dataset, []).append(record)
    for dataset, records in by_dataset.items():
        records.sort(key=_selection_key, reverse=True)
        top_n = 2 if _group_for_dataset(dataset) == "weak" else 1
        by_dataset[dataset] = records[:top_n]
    return by_dataset


def _neighbor_anchor(dataset: str, anchor: str) -> str:
    return str(NEIGHBOR_ANCHOR.get(dataset, {}).get(anchor, anchor))


def build_stage2_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    cursor = 0
    seeds = _parse_csv_ints(args.seeds)
    selected = _select_stage1_records(args)
    for dataset in _selected_datasets(args):
        budget = _row_budget("stage2", dataset)
        profile = _dataset_profile(dataset)
        train_batch, eval_batch = _row_batch_sizes(dataset, args)
        for record in selected.get(dataset, []):
            source_row = dict(record["row"])
            best_params = dict(record.get("best_params", {}) or {})
            source_anchor = str(source_row["capacity_anchor"])
            local_lr = _local_lr_window(dataset, best_params.get("learning_rate"), 5)
            local_hidden = _local_window(_anchor_hidden_choices(source_anchor, str(source_row["family_group"])), best_params.get("hidden_dropout_prob"), 3)
            local_max_len = _local_max_len_window(dataset, best_params.get("MAX_ITEM_LIST_LENGTH"), 3)
            local_d_feat = _local_d_feat_window(dataset, best_params.get("d_feat_emb"), 3)
            local_expert = _local_expert_window(dataset, best_params.get("expert_scale"), 2)
            local_fdrop = _local_family_dropout_window(dataset, best_params.get("stage_family_dropout_prob"), 3)
            local_xdrop = _local_feature_dropout_window(dataset, best_params.get("stage_feature_dropout_prob"), 3)
            for variant_id, anchor in (
                ("same_anchor_local", source_anchor),
                ("neighbor_anchor_local", _neighbor_anchor(dataset, source_anchor)),
            ):
                for seed_id in seeds:
                    cursor += 1
                    anchor_cfg = _anchor_cfg(anchor)
                    fixed_values: Dict[str, Any] = {
                        "embedding_size": int(anchor_cfg["embedding_size"]),
                        "d_ff": int(anchor_cfg["d_ff"]),
                        "d_expert_hidden": int(anchor_cfg["d_expert_hidden"]),
                        "d_router_hidden": int(anchor_cfg["d_router_hidden"]),
                        "stage_feature_dropout_prob": _all_stage_map(float(profile["default_feature_dropout"])),
                    }
                    search_space: Dict[str, list[Any]] = {
                        "learning_rate": local_lr,
                        "hidden_dropout_prob": [float(v) for v in local_hidden],
                        "MAX_ITEM_LIST_LENGTH": local_max_len,
                        "d_feat_emb": local_d_feat,
                        "expert_scale": local_expert,
                        "stage_family_dropout_prob": local_fdrop,
                        "num_heads": [2, 4],
                        "attn_dropout_prob": [0.05, 0.10, 0.15],
                        "lr_scheduler_type": ["warmup_cosine", "cosine"],
                    }
                    if profile["group"] == "weak":
                        search_space["stage_feature_dropout_prob"] = local_xdrop
                        fixed_values.pop("stage_feature_dropout_prob", None)
                    if profile["dense_easy"]:
                        search_space["weight_decay"] = _weight_decay_choices(anchor, [0.5, 1.0, 2.0])
                    else:
                        fixed_values["weight_decay"] = float(anchor_cfg["fixed_weight_decay"])
                    default_family_drop = _float_from_stage_map(local_fdrop[0], profile["family_dropout"][0])
                    default_feature_drop = _float_from_stage_map(
                        (local_xdrop[0] if local_xdrop else _all_stage_map(profile["default_feature_dropout"])),
                        profile["default_feature_dropout"],
                    )
                    overrides = _build_a12_overrides(
                        args,
                        family_dropout_map=_all_stage_map(default_family_drop),
                        feature_dropout_map=_all_stage_map(default_feature_drop),
                    )
                    family_id = f"{source_row['family_id']}__{variant_id}"
                    rows.append(
                        _row_base(
                            dataset=dataset,
                            stage_name="stage2",
                            family_id=family_id,
                            family_group=str(source_row["family_group"]),
                            variant_id=variant_id,
                            capacity_anchor=anchor,
                            selected_from_stage="stage1",
                            selection_score=_selection_score_text(record),
                            seed_id=int(seed_id),
                            runtime_seed=int(args.seed_base) + cursor - 1,
                            stage_group=str(budget["stage_group"]),
                            search_algo=str(budget["search_algo"]),
                            source_family_id=str(source_row["family_id"]),
                            fixed_values=fixed_values,
                            search_space=search_space,
                            overrides=overrides,
                            train_batch_size=train_batch,
                            eval_batch_size=eval_batch,
                            max_evals=int(budget["max_evals"]),
                            tune_epochs=int(budget["epochs"]),
                            tune_patience=int(budget["patience"]),
                        )
                    )
    return rows


def _select_stage2_records(args: argparse.Namespace) -> Dict[str, Optional[Dict[str, Any]]]:
    manifest_rows = _load_manifest_rows(stage_manifest_path("stage2"))
    selected_datasets = set(_selected_datasets(args))
    best_by_dataset: Dict[str, Optional[Dict[str, Any]]] = {dataset: None for dataset in selected_datasets}
    for row in manifest_rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in selected_datasets:
            continue
        record = _record_from_manifest_row(row)
        if record is None:
            continue
        prev = best_by_dataset.get(dataset)
        if prev is None or _selection_key(record) > _selection_key(prev):
            best_by_dataset[dataset] = record
    return best_by_dataset


def build_stage3_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    cursor = 0
    seeds = _parse_csv_ints(args.seeds)
    selected = _select_stage2_records(args)
    for dataset in _selected_datasets(args):
        record = selected.get(dataset)
        if record is None:
            continue
        budget = _row_budget("stage3", dataset)
        profile = _dataset_profile(dataset)
        source_row = dict(record["row"])
        best_params = dict(record.get("best_params", {}) or {})
        anchor = str(source_row["capacity_anchor"])
        anchor_cfg = _anchor_cfg(anchor)
        train_batch, eval_batch = _row_batch_sizes(dataset, args)
        variants = ["narrow_opt", "narrow_context_feature"] if profile["group"] == "weak" else ["narrow_opt"]
        fixed_num_heads = int(best_params.get("num_heads", 4) or 4)
        fixed_attn_dropout = float(best_params.get("attn_dropout_prob", 0.10) or 0.10)
        fixed_scheduler = str(best_params.get("lr_scheduler_type", "warmup_cosine") or "warmup_cosine")
        fixed_expert_scale = int(best_params.get("expert_scale", profile["expert_scale"][0]) or profile["expert_scale"][0])
        fixed_feature_dropout = _all_stage_map(_float_from_stage_map(best_params.get("stage_feature_dropout_prob"), profile["default_feature_dropout"]))
        fixed_family_dropout = _all_stage_map(_float_from_stage_map(best_params.get("stage_family_dropout_prob"), profile["family_dropout"][0]))
        for variant_id in variants:
            for seed_id in seeds:
                cursor += 1
                fixed_values: Dict[str, Any] = {
                    "embedding_size": int(anchor_cfg["embedding_size"]),
                    "d_ff": int(anchor_cfg["d_ff"]),
                    "d_expert_hidden": int(anchor_cfg["d_expert_hidden"]),
                    "d_router_hidden": int(anchor_cfg["d_router_hidden"]),
                    "num_heads": fixed_num_heads,
                    "attn_dropout_prob": fixed_attn_dropout,
                    "lr_scheduler_type": fixed_scheduler,
                    "expert_scale": fixed_expert_scale,
                    "stage_feature_dropout_prob": fixed_feature_dropout,
                    "stage_family_dropout_prob": fixed_family_dropout,
                    "weight_decay": float(best_params.get("weight_decay", anchor_cfg["fixed_weight_decay"]) or anchor_cfg["fixed_weight_decay"]),
                }
                search_space: Dict[str, list[Any]] = {
                    "learning_rate": _local_lr_window(dataset, best_params.get("learning_rate"), 4),
                    "hidden_dropout_prob": [float(v) for v in _local_window(_anchor_hidden_choices(anchor, str(source_row["family_group"])), best_params.get("hidden_dropout_prob"), 3)],
                    "MAX_ITEM_LIST_LENGTH": _local_max_len_window(dataset, best_params.get("MAX_ITEM_LIST_LENGTH"), 3),
                    "stage_family_dropout_prob": _local_family_dropout_window(dataset, best_params.get("stage_family_dropout_prob"), 3),
                }
                if variant_id == "narrow_context_feature":
                    search_space["stage_feature_dropout_prob"] = _local_feature_dropout_window(dataset, best_params.get("stage_feature_dropout_prob"), 3)
                    search_space["d_feat_emb"] = _local_d_feat_window(dataset, best_params.get("d_feat_emb"), 3)
                    fixed_values.pop("stage_feature_dropout_prob", None)
                elif profile["group"] == "weak":
                    search_space["stage_feature_dropout_prob"] = _local_feature_dropout_window(dataset, best_params.get("stage_feature_dropout_prob"), 2)
                    fixed_values.pop("stage_feature_dropout_prob", None)
                if profile["dense_easy"]:
                    center_wd = float(best_params.get("weight_decay", anchor_cfg["fixed_weight_decay"]) or anchor_cfg["fixed_weight_decay"])
                    search_space["weight_decay"] = _dedupe_keep_order(
                        _round_float(center_wd * scale, 12) for scale in [0.5, 1.0, 2.0]
                    )
                    fixed_values.pop("weight_decay", None)
                default_family_drop = _float_from_stage_map(search_space["stage_family_dropout_prob"][0], profile["family_dropout"][0])
                default_feature_drop = _float_from_stage_map(
                    (search_space.get("stage_feature_dropout_prob") or [fixed_feature_dropout])[0],
                    profile["default_feature_dropout"],
                )
                overrides = _build_a12_overrides(
                    args,
                    family_dropout_map=_all_stage_map(default_family_drop),
                    feature_dropout_map=_all_stage_map(default_feature_drop),
                )
                family_id = f"{source_row['family_id']}__{variant_id}"
                rows.append(
                    _row_base(
                        dataset=dataset,
                        stage_name="stage3",
                        family_id=family_id,
                        family_group=str(source_row["family_group"]),
                        variant_id=variant_id,
                        capacity_anchor=anchor,
                        selected_from_stage="stage2",
                        selection_score=_selection_score_text(record),
                        seed_id=int(seed_id),
                        runtime_seed=int(args.seed_base) + cursor - 1,
                        stage_group=str(budget["stage_group"]),
                        search_algo=str(budget["search_algo"]),
                        source_family_id=str(source_row["family_id"]),
                        fixed_values=fixed_values,
                        search_space=search_space,
                        overrides=overrides,
                        train_batch_size=train_batch,
                        eval_batch_size=eval_batch,
                        max_evals=int(budget["max_evals"]),
                        tune_epochs=int(budget["epochs"]),
                        tune_patience=int(budget["patience"]),
                    )
                )
    return rows


def maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 2) or 2))])


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = str(Path(str(Path("/venv/FMoE/bin/python"))))
    if not Path(python_bin).exists():
        python_bin = str(Path(sys.executable))
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
        f"++fmoe_architecture_id={hydra_literal(ARCH_ID)}",
        f"++fmoe_architecture_key={hydra_literal(ARCH_KEY)}",
        f"++fmoe_hparam_id={hydra_literal(row['capacity_anchor'])}",
        f"++fmoe_phase={hydra_literal(_phase_id(str(row['tuning_stage'])))}",
        f"train_batch_size={int(row['train_batch_size'])}",
        f"eval_batch_size={int(row['eval_batch_size'])}",
        f"++phase_run_type={hydra_literal(str(row['tuning_stage']))}",
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
    for key, values in search_space.items():
        cmd.append(f"++search.{key}={hydra_literal(list(values))}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")
    return cmd


def launch_stage(stage_name: str, args: argparse.Namespace, rows: list[Dict[str, Any]]) -> int:
    rows = maybe_limit_smoke(rows, args)
    manifest = write_manifest(stage_name, args, rows)
    print(f"[{stage_name}] manifest -> {manifest}")
    fieldnames = common_summary_fieldnames()
    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return int(
        launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=_phase_id(stage_name),
            phase_name=_phase_name(stage_name),
            log_dir=axis_root(),
            summary_path=axis_root() / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[col for col in fieldnames if col not in {
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
            }],
            build_command=build_command,
            build_log_path=build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=summary_path_for_row,
        )
    )
