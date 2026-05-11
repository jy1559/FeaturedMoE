#!/usr/bin/env python3
"""Stage G runner (cross 9 models x 6 datasets) for final baseline paper-quality sweep.

Design goals:
- Reuse run_final_all_datasets execution skeleton (resume/verify/logging/shared queue)
- Candidate-count policy:
  * old combos (A~F deeply explored 5 models x 2 datasets): 1 candidate
  * new combos: 2 candidates
- Seed policy: 1,2,3
- Balanced runtime budget with slight extra room for GRU4Rec/FAME
- Force amazon_beauty GRU4Rec/FAME to recovery templates
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import run_final_all_datasets as base


# ---------------------------------------------------------------------------
# Stage identity / defaults
# ---------------------------------------------------------------------------
base.AXIS = "StageG_Cross6x9_anchor2_core5"
base.PHASE_ID = "P22"
base.PHASE_NAME = "STAGEG_CROSS6X9_ANCHOR2_CORE5"
base.AXIS_DESC = "stageg_cross6x9_anchor2_core5"

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "amazon_beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

DEFAULT_MODELS = [m["model_option"] for m in base.MODEL_SPECS]

OLD_DATASETS = {"lastfm0.03", "amazon_beauty"}
OLD_MODELS = {"sasrec", "gru4rec", "duorec", "difsr", "fame"}

STAGE_AXES = [
    "StageA_LR_anchor2_core5",
    "StageB_Structure_anchor2_core5",
    "StageC_Focus_anchor2_core5",
    "StageD_MicroWide_anchor2_core5",
    "StageE_ReLRSeed_anchor2_core5",
    "StageF_TailBoost_anchor2_core5",
]

FINAL_AXIS = "Final_all_datasets"

TRANSFER_RULES = {
    "movielens1m": ("lastfm0.03", "foursquare"),
    "retail_rocket": ("amazon_beauty", "foursquare"),
}

LR_MIN = 8e-5
LR_MAX = 1e-2

SOURCE_BONUS = {
    "manual": 1.00,
    "stage_summary": 0.85,
    "final_same_dataset": 0.70,
    "final_transfer": 0.55,
    "global_fallback": 0.40,
}

OPTION_TO_LABEL = {str(m["model_option"]).lower(): str(m["model_label"]) for m in base.MODEL_SPECS}
LABEL_TO_OPTION = {v: k for k, v in OPTION_TO_LABEL.items()}


# ---------------------------------------------------------------------------
# Recovery templates (from Stage F findings)
# ---------------------------------------------------------------------------
MANUAL_AB_GRU_RECOVER = {
    "id": "MANUAL_AB_GRU_RECOVER_G",
    "config": {
        "hidden_size": 264,
        "embedding_size": 264,
        "layers": 3,
        "num_layers": 3,
        "max_len": 10,
        "dropout": 0.06,
        "weight_decay": 4.84e-5,
    },
    "lr_lo": 1.2e-3,
    "lr_hi": 9.0e-3,
}

MANUAL_AB_FAME_RECOVER = {
    "id": "MANUAL_AB_FAME_RECOVER_G",
    "config": {
        "hidden_size": 88,
        "embedding_size": 88,
        "layers": 3,
        "num_layers": 3,
        "heads": 8,
        "inner_size": 176,
        "max_len": 10,
        "dropout": 0.16,
        "weight_decay": 1.57e-4,
        "num_experts": 4,
    },
    "lr_lo": 1.1e-3,
    "lr_hi": 1.0e-2,
}

AGGR_SPARSE_GRU = {
    "id": "AGGR_SPARSE_GRU_G2",
    "config": {
        "hidden_size": 264,
        "embedding_size": 264,
        "layers": 3,
        "num_layers": 3,
        "max_len": 10,
        "dropout": 0.05,
        "weight_decay": 4.84e-5,
    },
    "lr_lo": 1.4e-3,
    "lr_hi": 1.0e-2,
}

AGGR_SPARSE_FAME = {
    "id": "AGGR_SPARSE_FAME_G2",
    "config": {
        "hidden_size": 88,
        "embedding_size": 88,
        "layers": 3,
        "num_layers": 3,
        "heads": 8,
        "inner_size": 176,
        "max_len": 10,
        "dropout": 0.14,
        "weight_decay": 1.2e-4,
        "num_experts": 4,
    },
    "lr_lo": 1.3e-3,
    "lr_hi": 1.0e-2,
}


# ---------------------------------------------------------------------------
# Utility / normalization
# ---------------------------------------------------------------------------
def _metric(v: Any) -> Optional[float]:
    return base._metric_to_float(v)


def _clamp_lr(lo: float, hi: float) -> tuple[float, float]:
    lo2 = max(float(LR_MIN), min(float(LR_MAX), float(lo)))
    hi2 = max(float(LR_MIN), min(float(LR_MAX), float(hi)))
    if hi2 <= lo2:
        hi2 = min(float(LR_MAX), max(lo2 * 1.25, lo2 + 1e-6))
    return float(lo2), float(hi2)


def _scale_lr_band(lo: float, hi: float, band_mult: float) -> tuple[float, float]:
    lo, hi = _clamp_lr(lo, hi)
    center = math.sqrt(float(lo) * float(hi))
    span = max(1.02, math.sqrt(float(hi) / float(lo)))
    span2 = max(1.02, float(span) * float(band_mult))
    return _clamp_lr(center / span2, center * span2)


def _candidate_score(valid: Optional[float], test: Optional[float], completion: Optional[float], source: str) -> float:
    v = float(valid) if valid is not None else 0.0
    t = float(test) if test is not None else 0.0
    c = float(completion) if completion is not None else 0.0
    b = float(SOURCE_BONUS.get(str(source), 0.35))
    return float(v + 0.20 * t + 0.03 * b + 0.01 * c)


def _default_cfg_for_model(model_option: str) -> Dict[str, Any]:
    h = base._effective_model_hparams({"model_option": str(model_option), "hparam_id": "H2"})
    cfg = {
        "hidden_size": int(h.get("hidden_size", 128)),
        "embedding_size": int(h.get("embedding_size", h.get("hidden_size", 128))),
        "layers": int(h.get("layers", 2)),
        "num_layers": int(h.get("layers", 2)),
        "heads": int(h.get("heads", 4)),
        "inner_size": int(h.get("inner_size", 256)),
        "dropout": float(h.get("dropout", 0.12)),
        "weight_decay": float(h.get("weight_decay", 1e-4)),
        "max_len": int(h.get("max_len", 20)),
        "time_span": int(h.get("time_span", 384)),
        "num_experts": int(h.get("num_experts", 3)),
        "sigma_state": int(h.get("sigma_state", 16)),
        "sigma_kernel": int(h.get("sigma_kernel", 6)),
        "sigma_remaining_ratio": float(h.get("sigma_remaining_ratio", 0.6)),
    }

    if model_option == "duorec":
        cfg.update({"contrast": "un", "tau": 0.30, "lmd": 0.04, "lmd_sem": 0.0, "semantic_sample_max_tries": 2})
    if model_option == "fearec":
        cfg.update({"contrast": "un", "tau": 0.20, "lmd": 0.04, "lmd_sem": 0.0, "global_ratio": 0.85, "semantic_sample_max_tries": 2})
    if model_option == "difsr":
        cfg.update({"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.10})
    return cfg


def _normalize_cfg(model_option: str, cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = _default_cfg_for_model(model_option)
    cfg = dict(base_cfg)
    cfg.update(dict(cfg_in or {}))

    cfg["hidden_size"] = max(64, int(cfg.get("hidden_size", base_cfg["hidden_size"])))
    cfg["embedding_size"] = max(64, int(cfg.get("embedding_size", cfg["hidden_size"])))

    layers = int(cfg.get("num_layers", cfg.get("layers", base_cfg["layers"])))
    layers = max(1, min(4, layers))
    cfg["layers"] = layers
    cfg["num_layers"] = layers

    heads = int(cfg.get("heads", cfg.get("num_heads", base_cfg["heads"])))
    heads = max(1, min(8, heads))
    cfg["heads"] = heads

    cfg["inner_size"] = max(cfg["hidden_size"], int(cfg.get("inner_size", base_cfg["inner_size"])))
    cfg["dropout"] = min(0.45, max(0.03, float(cfg.get("dropout", base_cfg["dropout"]))))
    cfg["weight_decay"] = max(1e-8, float(cfg.get("weight_decay", base_cfg["weight_decay"])))
    cfg["max_len"] = max(5, min(60, int(cfg.get("max_len", base_cfg["max_len"]))))

    cfg["time_span"] = max(64, int(cfg.get("time_span", base_cfg["time_span"])))
    cfg["num_experts"] = max(2, min(8, int(cfg.get("num_experts", base_cfg["num_experts"]))))
    cfg["sigma_state"] = max(4, int(cfg.get("sigma_state", base_cfg["sigma_state"])))
    cfg["sigma_kernel"] = max(2, int(cfg.get("sigma_kernel", base_cfg["sigma_kernel"])))
    cfg["sigma_remaining_ratio"] = min(0.95, max(0.1, float(cfg.get("sigma_remaining_ratio", base_cfg["sigma_remaining_ratio"]))))

    if model_option == "duorec":
        cfg["contrast"] = str(cfg.get("contrast", "un"))
        cfg["tau"] = float(cfg.get("tau", 0.30))
        cfg["lmd"] = float(cfg.get("lmd", 0.04))
        cfg["lmd_sem"] = float(cfg.get("lmd_sem", 0.0))
        cfg["semantic_sample_max_tries"] = int(cfg.get("semantic_sample_max_tries", 2))

    if model_option == "fearec":
        cfg["contrast"] = str(cfg.get("contrast", "un"))
        cfg["tau"] = float(cfg.get("tau", 0.20))
        cfg["lmd"] = float(cfg.get("lmd", 0.04))
        cfg["lmd_sem"] = float(cfg.get("lmd_sem", 0.0))
        cfg["global_ratio"] = float(cfg.get("global_ratio", 0.85))
        cfg["semantic_sample_max_tries"] = int(cfg.get("semantic_sample_max_tries", 2))

    if model_option == "difsr":
        cfg["fusion_type"] = str(cfg.get("fusion_type", "gate"))
        cfg["use_attribute_predictor"] = bool(cfg.get("use_attribute_predictor", True))
        cfg["lambda_attr"] = float(cfg.get("lambda_attr", 0.10))

    return cfg


def _config_from_result_payload(model_option: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _default_cfg_for_model(model_option)
    fixed = payload.get("fixed_search") if isinstance(payload.get("fixed_search"), dict) else {}
    context_fixed = payload.get("context_fixed") if isinstance(payload.get("context_fixed"), dict) else {}
    best_params = payload.get("best_params") if isinstance(payload.get("best_params"), dict) else {}

    def _pick_num(*keys: str) -> Optional[float]:
        for k in keys:
            v = _metric(fixed.get(k))
            if v is not None:
                return v
            v = _metric(context_fixed.get(k))
            if v is not None:
                return v
            v = _metric(best_params.get(k))
            if v is not None:
                return v
        return None

    hs = _pick_num("hidden_size")
    if hs is not None:
        cfg["hidden_size"] = int(hs)
    es = _pick_num("embedding_size")
    if es is not None:
        cfg["embedding_size"] = int(es)
    nl = _pick_num("num_layers", "n_layers")
    if nl is not None:
        cfg["layers"] = int(nl)
        cfg["num_layers"] = int(nl)
    nh = _pick_num("num_heads", "n_heads")
    if nh is not None:
        cfg["heads"] = int(nh)
    inner = _pick_num("inner_size")
    if inner is not None:
        cfg["inner_size"] = int(inner)
    dp = _pick_num("dropout_ratio", "dropout_prob", "hidden_dropout_prob", "attn_dropout_prob")
    if dp is not None:
        cfg["dropout"] = float(dp)
    wd = _pick_num("weight_decay")
    if wd is not None:
        cfg["weight_decay"] = float(wd)
    ml = _pick_num("MAX_ITEM_LIST_LENGTH")
    if ml is not None:
        cfg["max_len"] = int(ml)
    ts = _pick_num("time_span")
    if ts is not None:
        cfg["time_span"] = int(ts)
    ne = _pick_num("num_experts")
    if ne is not None:
        cfg["num_experts"] = int(ne)
    ss = _pick_num("state_size", "sigma_state")
    if ss is not None:
        cfg["sigma_state"] = int(ss)
    sk = _pick_num("conv_kernel", "sigma_kernel")
    if sk is not None:
        cfg["sigma_kernel"] = int(sk)
    rr = _pick_num("remaining_ratio", "sigma_remaining_ratio")
    if rr is not None:
        cfg["sigma_remaining_ratio"] = float(rr)

    return _normalize_cfg(model_option, cfg)


# ---------------------------------------------------------------------------
# Candidate loading
# ---------------------------------------------------------------------------
def _final_summary_path(dataset: str) -> Path:
    return base.LOG_ROOT / FINAL_AXIS / base._dataset_tag(dataset) / "summary.csv"


def _stage_summary_path(axis: str, dataset: str) -> Path:
    return base.LOG_ROOT / axis / base._dataset_tag(dataset) / "summary.csv"


def _safe_json(path_text: str) -> Optional[Dict[str, Any]]:
    p = Path(str(path_text or "")).expanduser()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _candidate_from_row(
    *,
    model_option: str,
    row: Dict[str, Any],
    source: str,
    source_stage: str,
    transfer_from_dataset: str,
    candidate_id: str,
    lr_policy_mult: Optional[float] = None,
    force_lr_narrow_around_best: bool = False,
) -> Optional[Dict[str, Any]]:
    payload = _safe_json(str(row.get("result_path", "")))
    if not isinstance(payload, dict):
        return None

    cfg = _config_from_result_payload(model_option, payload)

    best_lr = _metric((payload.get("best_params") or {}).get("learning_rate"))
    lo = _metric(row.get("lr_lo"))
    hi = _metric(row.get("lr_hi"))

    if best_lr is None and lo is not None and hi is not None and hi > lo:
        best_lr = math.sqrt(float(lo) * float(hi))
    if best_lr is None:
        lo2, hi2 = base._compute_lr_space(str(row.get("dataset", "")), model_option, "H2")
        best_lr = math.sqrt(float(lo2) * float(hi2))

    if force_lr_narrow_around_best:
        lo, hi = _clamp_lr(float(best_lr) / 1.6, float(best_lr) * 1.6)
    else:
        if lo is None or hi is None or hi <= lo:
            lo, hi = _clamp_lr(float(best_lr) / 2.2, float(best_lr) * 2.2)
        else:
            lo, hi = _clamp_lr(float(lo), float(hi))
        if lr_policy_mult is not None:
            lo, hi = _scale_lr_band(float(lo), float(hi), float(lr_policy_mult))

    valid = _metric(row.get("run_best_valid_mrr20"))
    test = _metric(row.get("run_best_test_mrr20"))
    n_completed = _metric(row.get("n_completed"))
    completion = None if n_completed is None else max(0.0, min(1.0, float(n_completed) / 10.0))

    return {
        "candidate_id": str(candidate_id),
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": str(source),
        "source_stage": str(source_stage),
        "transfer_from_dataset": str(transfer_from_dataset),
        "valid": valid,
        "test": test,
        "completion": completion,
        "score": _candidate_score(valid, test, completion, source),
    }


def _read_summary_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def _top_unique_hparam_rows(rows: list[Dict[str, Any]], model_label: str, topn: int = 2) -> list[Dict[str, Any]]:
    filtered = [r for r in rows if str(r.get("model", "")).strip() == str(model_label)]
    filtered.sort(
        key=lambda r: (
            float(_metric(r.get("run_best_valid_mrr20")) or -1e9),
            float(_metric(r.get("run_best_test_mrr20")) or -1e9),
            str(r.get("timestamp_utc", "")),
        ),
        reverse=True,
    )
    out = []
    seen = set()
    for r in filtered:
        hid = str(r.get("hparam_id", "")).strip().upper()
        if not hid or hid in seen:
            continue
        seen.add(hid)
        out.append(r)
        if len(out) >= max(1, int(topn)):
            break
    return out


def _load_final_hparam_cache() -> tuple[Dict[str, Dict[str, list[Dict[str, Any]]]], Dict[str, list[Dict[str, Any]]]]:
    by_dataset: Dict[str, Dict[str, list[Dict[str, Any]]]] = defaultdict(dict)
    by_model_global: Dict[str, list[Dict[str, Any]]] = defaultdict(list)

    for dataset in DEFAULT_DATASETS:
        spath = _final_summary_path(dataset)
        rows = _read_summary_rows(spath)
        if not rows:
            continue
        for spec in base.MODEL_SPECS:
            model_option = str(spec["model_option"]).lower()
            model_label = str(spec["model_label"])
            top_rows = _top_unique_hparam_rows(rows, model_label, topn=2)
            cands = []
            for idx, r in enumerate(top_rows, start=1):
                c = _candidate_from_row(
                    model_option=model_option,
                    row=r,
                    source="final_same_dataset",
                    source_stage=FINAL_AXIS,
                    transfer_from_dataset="",
                    candidate_id=f"FAD_{str(r.get('hparam_id','H2')).upper()}_R{idx}",
                    lr_policy_mult=(0.72 if idx == 1 else 0.95),
                    force_lr_narrow_around_best=False,
                )
                if c:
                    cands.append(c)
                    by_model_global[model_option].append(c)
            if cands:
                by_dataset[dataset][model_option] = cands

    for model_option, cands in by_model_global.items():
        cands.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)
        uniq: list[Dict[str, Any]] = []
        seen = set()
        for c in cands:
            key = (int(c["config"].get("hidden_size", 0)), int(c["config"].get("layers", 0)), int(c["config"].get("max_len", 0)), round(float(c["lr_lo"]), 7), round(float(c["lr_hi"]), 7))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
            if len(uniq) >= 4:
                break
        by_model_global[model_option] = uniq

    return by_dataset, by_model_global


def _load_stage_best_cache() -> Dict[tuple[str, str], Dict[str, Any]]:
    out: Dict[tuple[str, str], Dict[str, Any]] = {}
    for axis in STAGE_AXES:
        for dataset in sorted(OLD_DATASETS):
            rows = _read_summary_rows(_stage_summary_path(axis, dataset))
            if not rows:
                continue
            for r in rows:
                model_label = str(r.get("model", "")).strip()
                model_option = LABEL_TO_OPTION.get(model_label)
                if model_option is None:
                    continue
                key = (dataset, model_option)
                valid = _metric(r.get("run_best_valid_mrr20"))
                if valid is None:
                    continue
                prev = out.get(key)
                prev_valid = _metric(prev.get("run_best_valid_mrr20")) if isinstance(prev, dict) else None
                if prev is None or (prev_valid is None or float(valid) > float(prev_valid)):
                    rr = dict(r)
                    rr["__axis"] = axis
                    out[key] = rr
    return out


def _manual_candidate(template: Dict[str, Any], model_option: str, *, source_stage: str, transfer_from_dataset: str = "") -> Dict[str, Any]:
    lo, hi = _clamp_lr(float(template["lr_lo"]), float(template["lr_hi"]))
    cfg = _normalize_cfg(model_option, dict(template.get("config", {})))
    return {
        "candidate_id": str(template["id"]),
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "manual",
        "source_stage": str(source_stage),
        "transfer_from_dataset": str(transfer_from_dataset),
        "valid": None,
        "test": None,
        "completion": 1.0,
        "score": _candidate_score(None, None, 1.0, "manual"),
    }


def _dedup_candidates(cands: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out = []
    seen = set()
    for c in cands:
        key = (
            int(c["config"].get("hidden_size", 0)),
            int(c["config"].get("num_layers", c["config"].get("layers", 0))),
            int(c["config"].get("max_len", 0)),
            round(float(c["lr_lo"]), 7),
            round(float(c["lr_hi"]), 7),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _fallback_candidate(dataset: str, model_option: str, idx: int) -> Dict[str, Any]:
    cfg = _default_cfg_for_model(model_option)
    lo, hi = base._compute_lr_space(dataset, model_option, "H2")
    lo, hi = _scale_lr_band(lo, hi, 0.72 if idx == 1 else 0.95)
    return {
        "candidate_id": f"FB_{idx}",
        "config": _normalize_cfg(model_option, cfg),
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "global_fallback",
        "source_stage": "fallback",
        "transfer_from_dataset": "",
        "valid": None,
        "test": None,
        "completion": 0.0,
        "score": _candidate_score(None, None, 0.0, "global_fallback"),
    }


def _select_candidates_for_combo(
    *,
    dataset: str,
    model_option: str,
    model_label: str,
    final_cache_by_dataset: Dict[str, Dict[str, list[Dict[str, Any]]]],
    global_cache_by_model: Dict[str, list[Dict[str, Any]]],
    stage_best_cache: Dict[tuple[str, str], Dict[str, Any]],
) -> list[Dict[str, Any]]:
    old_combo = (dataset in OLD_DATASETS and model_option in OLD_MODELS)

    # Forced recovery for known collapse points.
    if dataset == "amazon_beauty" and model_option == "gru4rec":
        return [_manual_candidate(MANUAL_AB_GRU_RECOVER, model_option, source_stage="manual_bank")]
    if dataset == "amazon_beauty" and model_option == "fame":
        return [_manual_candidate(MANUAL_AB_FAME_RECOVER, model_option, source_stage="manual_bank")]

    if old_combo:
        row = stage_best_cache.get((dataset, model_option))
        if isinstance(row, dict):
            c = _candidate_from_row(
                model_option=model_option,
                row=row,
                source="stage_summary",
                source_stage=str(row.get("__axis", "")),
                transfer_from_dataset="",
                candidate_id=f"OLD_{str(row.get('hparam_id','BEST'))}",
                force_lr_narrow_around_best=True,
            )
            if c:
                return [c]
        return [_fallback_candidate(dataset, model_option, 1)]

    # non-old: need 2 candidates
    cands: list[Dict[str, Any]] = []

    same_ds = list(final_cache_by_dataset.get(dataset, {}).get(model_option, []))
    if same_ds:
        for idx, c in enumerate(same_ds[:2], start=1):
            c2 = dict(c)
            c2["source"] = "final_same_dataset"
            c2["source_stage"] = FINAL_AXIS
            c2["transfer_from_dataset"] = ""
            c2["score"] = _candidate_score(c2.get("valid"), c2.get("test"), c2.get("completion"), "final_same_dataset")
            if idx == 1:
                c2["lr_lo"], c2["lr_hi"] = _scale_lr_band(float(c2["lr_lo"]), float(c2["lr_hi"]), 0.72)
            else:
                c2["lr_lo"], c2["lr_hi"] = _scale_lr_band(float(c2["lr_lo"]), float(c2["lr_hi"]), 0.95)
            cands.append(c2)
    else:
        primary, secondary = TRANSFER_RULES.get(dataset, ("", ""))

        if primary:
            p = list(final_cache_by_dataset.get(primary, {}).get(model_option, []))
            if p:
                c1 = dict(p[0])
                c1["source"] = "final_transfer"
                c1["source_stage"] = FINAL_AXIS
                c1["transfer_from_dataset"] = primary
                c1["candidate_id"] = f"XFER_{primary}_{c1['candidate_id']}_C1"
                c1["lr_lo"], c1["lr_hi"] = _scale_lr_band(float(c1["lr_lo"]), float(c1["lr_hi"]), 0.72)
                c1["score"] = _candidate_score(c1.get("valid"), c1.get("test"), c1.get("completion"), "final_transfer")
                cands.append(c1)

        if secondary:
            s = list(final_cache_by_dataset.get(secondary, {}).get(model_option, []))
            if s:
                c2 = dict(s[0])
                c2["source"] = "final_transfer"
                c2["source_stage"] = FINAL_AXIS
                c2["transfer_from_dataset"] = secondary
                c2["candidate_id"] = f"XFER_{secondary}_{c2['candidate_id']}_C2"
                c2["lr_lo"], c2["lr_hi"] = _scale_lr_band(float(c2["lr_lo"]), float(c2["lr_hi"]), 0.95)
                c2["score"] = _candidate_score(c2.get("valid"), c2.get("test"), c2.get("completion"), "final_transfer")
                cands.append(c2)

        if len(cands) < 2 and primary:
            p = list(final_cache_by_dataset.get(primary, {}).get(model_option, []))
            if len(p) >= 2:
                c2 = dict(p[1])
                c2["source"] = "final_transfer"
                c2["source_stage"] = FINAL_AXIS
                c2["transfer_from_dataset"] = primary
                c2["candidate_id"] = f"XFER_{primary}_{c2['candidate_id']}_ALT"
                c2["lr_lo"], c2["lr_hi"] = _scale_lr_band(float(c2["lr_lo"]), float(c2["lr_hi"]), 0.95)
                c2["score"] = _candidate_score(c2.get("valid"), c2.get("test"), c2.get("completion"), "final_transfer")
                cands.append(c2)

    # Sparse recovery replacement rule for candidate2 on GRU/FAME
    if dataset in {"amazon_beauty", "retail_rocket"} and model_option in {"gru4rec", "fame"}:
        if model_option == "gru4rec":
            aggr = _manual_candidate(AGGR_SPARSE_GRU, model_option, source_stage="manual_sparse", transfer_from_dataset=("amazon_beauty" if dataset == "retail_rocket" else ""))
        else:
            aggr = _manual_candidate(AGGR_SPARSE_FAME, model_option, source_stage="manual_sparse", transfer_from_dataset=("amazon_beauty" if dataset == "retail_rocket" else ""))

        if not cands:
            cands = [aggr]
        elif len(cands) == 1:
            cands.append(aggr)
        else:
            cands[1] = aggr

    cands = _dedup_candidates(cands)

    # global fallback to guarantee exactly 2
    global_pool = list(global_cache_by_model.get(model_option, []))
    for g_idx, gc in enumerate(global_pool, start=1):
        if len(cands) >= 2:
            break
        c = dict(gc)
        c["source"] = "global_fallback"
        c["source_stage"] = FINAL_AXIS
        c["transfer_from_dataset"] = "GLOBAL"
        c["candidate_id"] = f"GLOBAL_{g_idx}_{c['candidate_id']}"
        c["lr_lo"], c["lr_hi"] = _scale_lr_band(float(c["lr_lo"]), float(c["lr_hi"]), 0.95)
        c["score"] = _candidate_score(c.get("valid"), c.get("test"), c.get("completion"), "global_fallback")
        cands.append(c)

    cands = _dedup_candidates(cands)

    while len(cands) < 2:
        cands.append(_fallback_candidate(dataset, model_option, len(cands) + 1))

    cands = cands[:2]

    # deterministic order: better score first
    cands.sort(key=lambda x: float(x.get("score", -1e9)), reverse=True)

    return cands


# ---------------------------------------------------------------------------
# Budget / row / command
# ---------------------------------------------------------------------------
def _balanced_budget(model_option: str) -> tuple[int, int, int]:
    m = str(model_option).lower()
    if m in {"sasrec", "tisasrec", "bsarec", "difsr"}:
        return 6, 44, 6
    if m in {"duorec", "fearec", "sigma"}:
        return 6, 42, 6
    if m in {"gru4rec", "fame"}:
        return 7, 48, 7
    return 6, 44, 6


def _stageg_estimated_cost(row: Dict[str, Any]) -> float:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"]).lower()
    max_evals = max(1, int(row.get("max_evals", 6)))
    tune_epochs = max(1, int(row.get("tune_epochs", 44)))
    return (
        float(base.DATASET_COST_WEIGHT.get(dataset, 1.0))
        * float(base.MODEL_COST_WEIGHT.get(model_option, 1.0))
        * (float(max_evals) / 6.0)
        * (float(tune_epochs) / 44.0)
    )


def _build_rows(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int) -> list[Dict[str, Any]]:
    seeds = base._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    requested_models = [m.lower() for m in base._parse_csv_strings(args.models)]
    if not requested_models:
        raise RuntimeError("No models provided")

    final_cache_by_dataset, global_cache_by_model = _load_final_hparam_cache()
    stage_best_cache = _load_stage_best_cache()

    model_specs = [m for m in base.MODEL_SPECS if str(m["model_option"]).lower() in requested_models]
    if not model_specs:
        raise RuntimeError(f"No valid model specs selected: {args.models}")

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    ds_tag = base._sanitize_token(dataset, upper=True)

    for model_order, spec in enumerate(model_specs, start=1):
        model_option = str(spec["model_option"]).lower()
        model_label = str(spec["model_label"])

        candidates = _select_candidates_for_combo(
            dataset=dataset,
            model_option=model_option,
            model_label=model_label,
            final_cache_by_dataset=final_cache_by_dataset,
            global_cache_by_model=global_cache_by_model,
            stage_best_cache=stage_best_cache,
        )

        target_candidates = 1 if (dataset in OLD_DATASETS and model_option in OLD_MODELS) else 2
        candidates = candidates[:target_candidates]

        for c_order, cand in enumerate(candidates, start=1):
            max_evals, tune_epochs, tune_patience = _balanced_budget(model_option)
            if args.smoke_test:
                max_evals, tune_epochs, tune_patience = 1, 1, 1

            hparam_id = base._sanitize_token(str(cand["candidate_id"]), upper=True)[:40] or f"C{c_order}"

            for seed_id in seeds:
                run_cursor += 1
                run_id = (
                    f"SG_{ds_tag}_{base._sanitize_token(model_label, upper=True)}_"
                    f"{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{int(seed_id)}"
                )
                run_phase = (
                    f"{base.PHASE_ID}_SG_D{dataset_order_idx:02d}_M{model_order:02d}_C{c_order:02d}_"
                    f"{base._sanitize_token(model_label, upper=True)}_"
                    f"{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{int(seed_id)}"
                )

                row = {
                    "dataset": dataset,
                    "phase_id": base.PHASE_ID,
                    "axis_id": "SG",
                    "axis_desc": base.AXIS_DESC,
                    "setting_id": f"STAGEG_{base._sanitize_token(model_label, upper=True)}_{hparam_id}_S{int(seed_id)}",
                    "setting_key": "STAGEG_CROSS6X9_FINAL",
                    "setting_desc": f"STAGEG_CROSS6X9_FINAL_{base._sanitize_token(model_label, upper=True)}_{hparam_id}_S{int(seed_id)}",
                    "hparam_id": hparam_id,
                    "seed_id": int(seed_id),
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "stage": "stageg",
                    "model_option": model_option,
                    "model_label": model_label,
                    "candidate_id": str(cand["candidate_id"]),
                    "candidate_source": str(cand["source"]),
                    "source_stage": str(cand["source_stage"]),
                    "transfer_from_dataset": str(cand["transfer_from_dataset"]),
                    "candidate_config": _normalize_cfg(model_option, dict(cand["config"])),
                    "lr_lo": float(cand["lr_lo"]),
                    "lr_hi": float(cand["lr_hi"]),
                    "max_evals": int(max_evals),
                    "tune_epochs": int(tune_epochs),
                    "tune_patience": int(tune_patience),
                }
                row["estimated_cost"] = _stageg_estimated_cost(row)
                rows.append(row)

    return rows


# Patch effective-hparam path so downstream helpers can reuse base logic
_orig_effective_model_hparams = base._effective_model_hparams


def _effective_model_hparams_stageg(row: Dict[str, Any]) -> Dict[str, Any]:
    cfg = row.get("candidate_config")
    model_option = str(row.get("model_option", "")).lower()
    if model_option and isinstance(cfg, dict) and cfg:
        return _normalize_cfg(model_option, cfg)
    return _orig_effective_model_hparams(row)


base._effective_model_hparams = _effective_model_hparams_stageg

_orig_model_algorithm_overrides = base._model_algorithm_overrides


def _model_algorithm_overrides_stageg(row: Dict[str, Any]) -> list[str]:
    model_option = str(row.get("model_option", "")).lower()
    cfg = row.get("candidate_config")
    if not isinstance(cfg, dict) or not cfg:
        return _orig_model_algorithm_overrides(row)

    if model_option == "duorec":
        return [
            f"++contrast={str(cfg.get('contrast', 'un'))}",
            f"++tau={float(cfg.get('tau', 0.30))}",
            f"++lmd={float(cfg.get('lmd', 0.04))}",
            f"++lmd_sem={float(cfg.get('lmd_sem', 0.0))}",
            f"++semantic_sample_max_tries={int(cfg.get('semantic_sample_max_tries', 2))}",
        ]

    if model_option == "fearec":
        return [
            f"++contrast={str(cfg.get('contrast', 'un'))}",
            f"++tau={float(cfg.get('tau', 0.20))}",
            f"++lmd={float(cfg.get('lmd', 0.04))}",
            f"++lmd_sem={float(cfg.get('lmd_sem', 0.0))}",
            f"++global_ratio={float(cfg.get('global_ratio', 0.85))}",
            f"++semantic_sample_max_tries={int(cfg.get('semantic_sample_max_tries', 2))}",
        ]

    if model_option == "difsr":
        return [
            f"++fusion_type={str(cfg.get('fusion_type', 'gate'))}",
            f"++use_attribute_predictor={'true' if bool(cfg.get('use_attribute_predictor', True)) else 'false'}",
            f"++lambda_attr={float(cfg.get('lambda_attr', 0.10))}",
        ]

    if model_option == "fame":
        return [f"++num_experts={int(cfg.get('num_experts', 3))}"]

    return _orig_model_algorithm_overrides(row)


base._model_algorithm_overrides = _model_algorithm_overrides_stageg


def _build_command(row: Dict[str, Any], gpu_id: str, _args: argparse.Namespace) -> list[str]:
    model_option = str(row["model_option"]).lower()
    h = base._effective_model_hparams(row)
    lr_lo = float(row["lr_lo"])
    lr_hi = float(row["lr_hi"])
    tune_epochs = int(row["tune_epochs"])
    tune_patience = int(row["tune_patience"])
    max_evals = int(row["max_evals"])

    dropout = float(h["dropout"])
    weight_decay = float(h["weight_decay"])

    search_lr_only = {"learning_rate": [float(lr_lo), float(lr_hi)]}
    search_type_lr_only = {"learning_rate": "loguniform"}

    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        base._dataset_config_name(str(row["dataset"])),
        "--max-evals",
        str(int(max_evals)),
        "--tune-epochs",
        str(int(tune_epochs)),
        "--tune-patience",
        str(int(tune_patience)),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        base.TRACK,
        "--run-axis",
        base.AXIS,
        "--run-phase",
        str(row["run_phase"]),
        f"model={model_option}",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        "++special_logging=true",
        f"++seed={int(row['runtime_seed'])}",
        f"++search={base.hydra_literal(search_lr_only)}",
        f"++search_space_type_overrides={base.hydra_literal(search_type_lr_only)}",
        f"++weight_decay={weight_decay}",
    ]

    fixed_search: Dict[str, Any] = {
        "weight_decay": weight_decay,
        "n_layers": int(h["layers"]),
        "num_layers": int(h["layers"]),
        "n_heads": int(h["heads"]),
        "num_heads": int(h["heads"]),
        "dropout_ratio": dropout,
        "dropout_prob": dropout,
        "hidden_dropout_prob": dropout,
        "attn_dropout_prob": dropout,
        "hidden_size": int(h["hidden_size"]),
        "embedding_size": int(h["embedding_size"]),
        "inner_size": int(h["inner_size"]),
        "time_span": int(h.get("time_span", 384)),
        "num_experts": int(h.get("num_experts", 3)),
        "state_size": int(h.get("sigma_state", 16)),
        "conv_kernel": int(h.get("sigma_kernel", 6)),
        "remaining_ratio": float(h.get("sigma_remaining_ratio", 0.6)),
        "MAX_ITEM_LIST_LENGTH": int(h.get("max_len", 20)),
    }
    for key, value in fixed_search.items():
        cmd.append(f"++search.{key}={base.hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    if model_option == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    cmd.extend(base._model_algorithm_overrides(row))
    cmd.extend(base._model_runtime_resource_overrides(row))
    cmd.extend(base._base_hparam_overrides(row))
    return cmd


# ---------------------------------------------------------------------------
# Summary / manifest / run loop
# ---------------------------------------------------------------------------
def _summary_fieldnames() -> list[str]:
    return [
        "model",
        "global_best_valid_mrr20",
        "global_best_test_mrr20",
        "model_best_valid_mrr20",
        "model_best_test_mrr20",
        "run_best_valid_mrr20",
        "run_best_test_mrr20",
        "run_phase",
        "run_id",
        "dataset",
        "hparam_id",
        "seed_id",
        "gpu_id",
        "status",
        "n_completed",
        "interrupted",
        "special_ok",
        "candidate_id",
        "candidate_source",
        "source_stage",
        "transfer_from_dataset",
        "result_path",
        "timestamp_utc",
    ]


base._summary_fieldnames = _summary_fieldnames


def _write_manifest(dataset: str, args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = base._manifest_path(dataset, args)
    payload = {
        "track": base.TRACK,
        "axis": base.AXIS,
        "phase": base.PHASE_ID,
        "dataset": dataset,
        "execution_type": "stageg_cross6x9",
        "model_count": len({str(r.get("model_option", "")).lower() for r in rows}),
        "candidate_count": len({str(r.get("candidate_id", "")) for r in rows}),
        "seed_count": len(base._parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "run_count_formula": "old_combo:1x3 seeds, non_old:2x3 seeds",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "model": r["model_label"],
                "candidate_id": r.get("candidate_id", ""),
                "candidate_source": r.get("candidate_source", ""),
                "source_stage": r.get("source_stage", ""),
                "transfer_from_dataset": r.get("transfer_from_dataset", ""),
                "seed_id": r["seed_id"],
                "lr_lo": r.get("lr_lo"),
                "lr_hi": r.get("lr_hi"),
                "config": r.get("candidate_config", {}),
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, _args: argparse.Namespace, cmd: list[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{base.PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"model={row.get('model_label','')} candidate_id={row.get('candidate_id','')} seed={row.get('seed_id','')}"
        ),
        (
            f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)} "
            f"candidate_source={row.get('candidate_source','')} source_stage={row.get('source_stage','')} "
            f"transfer_from_dataset={row.get('transfer_from_dataset','')}"
        ),
        (
            f"max_evals={row.get('max_evals','')} tune_epochs={row.get('tune_epochs','')} "
            f"tune_patience={row.get('tune_patience','')}"
        ),
        f"seed={row.get('runtime_seed','')}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _smoke_trim_rows(rows: list[Dict[str, Any]], max_runs: int) -> list[Dict[str, Any]]:
    return rows[: max(1, int(max_runs))]


def _run_dataset(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int, gpus: list[str]) -> int:
    base._validate_session_fixed_files(dataset)
    rows = _build_rows(dataset, args, dataset_order_idx=dataset_order_idx)
    if args.smoke_test:
        rows = _smoke_trim_rows(rows, args.smoke_max_runs)

    manifest_path = _write_manifest(dataset, args, rows)
    summary_path = base._summary_path(dataset)
    base._ensure_summary_csv(summary_path)
    global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model = base._load_summary_bests(
        summary_path
    )

    print(
        f"[{base.PHASE_ID}] dataset={dataset} ({dataset_order_idx + 1}) rows={len(rows)} "
        f"axis={base.AXIS} manifest={manifest_path}"
    )

    for row in rows:
        row["log_path"] = str(base._build_log_path(row))

    runnable: list[Dict[str, Any]] = []
    skipped = 0
    result_index_for_resume = base._scan_result_index(dataset) if args.verify_logging else {}
    for row in rows:
        completed = base._is_completed_any_log(row, use_resume=bool(args.resume_from_logs))
        if not completed:
            runnable.append(row)
            continue
        if not args.verify_logging:
            skipped += 1
            continue
        rec = result_index_for_resume.get(str(row["run_phase"]))
        if not rec:
            runnable.append(row)
            continue
        ok, detail = base._verify_special_from_result(str(rec.get("path", "")))
        if ok:
            skipped += 1
            continue
        print(f"[resume-check] run={row['run_phase']} special_check_failed -> rerun ({detail})")
        runnable.append(row)

    if skipped > 0:
        print(f"[{base.PHASE_ID}] resume_from_logs=on: skipped {skipped} completed runs for dataset={dataset}.")

    if not runnable:
        print(f"[{base.PHASE_ID}] all runs already completed for dataset={dataset}.")
        base._update_baseline_phase_summary(dataset, base.PHASE_ID)
        return 0

    # Bottleneck-aware queue: sort expensive jobs first.
    runnable.sort(key=lambda r: float(r.get("estimated_cost", 1.0)), reverse=True)
    shared_queue: deque = deque(runnable)
    print(
        f"[queue] dataset={dataset} mode=shared_gpu_queue(cost_desc) "
        f"tasks_total={len(shared_queue)} gpus={','.join(gpus)}"
    )

    if args.dry_run:
        gpu_bins = base._plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = _build_command(row, gpu_id, args)
                print(
                    f"[dry-run] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} candidate={row.get('candidate_id','')} seed={row['seed_id']}"
                )
                print("          " + " ".join(cmd))
        return 0

    active: Dict[str, Dict[str, Any]] = {}
    try:
        while True:
            for gpu_id in gpus:
                if gpu_id in active:
                    continue
                if not shared_queue:
                    continue
                row = shared_queue.popleft()
                cmd = _build_command(row, gpu_id, args)
                log_path = Path(str(row["log_path"]))
                _write_log_preamble(log_path, row, gpu_id, args, cmd)
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("HYPEROPT_RESULTS_DIR", str(base.ARTIFACT_ROOT / "results"))
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                with log_path.open("a", encoding="utf-8") as fh:
                    proc = subprocess.Popen(cmd, cwd=base.EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
                active[gpu_id] = {"proc": proc, "row": row, "log_path": str(log_path)}
                print(
                    f"[launch] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} candidate={row.get('candidate_id','')}"
                )

            done_gpus: list[str] = []
            for gpu_id, slot in active.items():
                proc = slot["proc"]
                rc = proc.poll()
                if rc is None:
                    continue
                done_gpus.append(gpu_id)
                row = slot["row"]
                print(f"[done] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} rc={rc}")

                run_best = None
                test_mrr = None
                n_completed = None
                interrupted = None
                result_path = ""
                special_ok = False
                status = "run_complete" if int(rc) == 0 else "run_fail"

                rec = base._get_result_row_from_log_or_scan(
                    dataset=dataset,
                    run_phase=str(row["run_phase"]),
                    log_path=Path(str(slot.get("log_path", ""))),
                    retries=4,
                    sleep_sec=0.75,
                )
                if rec:
                    run_best = _metric(rec.get("best_mrr"))
                    test_mrr = _metric(rec.get("test_mrr"))
                    n_completed = int(rec.get("n_completed", 0) or 0)
                    interrupted = bool(rec.get("interrupted", False))
                    result_path = str(rec.get("path", "") or "")
                    if result_path:
                        special_ok, detail = base._verify_special_from_result(result_path)
                        print(f"[logging-check] run={row['run_phase']} {detail} result={result_path}")
                        base._mirror_logging_bundle(row, result_path)
                        if args.verify_logging and int(rc) == 0 and not special_ok:
                            raise RuntimeError(
                                f"special logging verification failed: run_phase={row['run_phase']} "
                                f"special_ok={special_ok} result={result_path}"
                            )
                if run_best is not None:
                    global_best_valid = (
                        run_best if global_best_valid is None else max(float(global_best_valid), float(run_best))
                    )
                    model_name = str(row["model_label"])
                    prev_model_best = model_best_valid_by_model.get(model_name)
                    model_best_valid_by_model[model_name] = (
                        run_best if prev_model_best is None else max(float(prev_model_best), float(run_best))
                    )
                if test_mrr is not None:
                    global_best_test = (
                        test_mrr if global_best_test is None else max(float(global_best_test), float(test_mrr))
                    )
                    model_name = str(row["model_label"])
                    prev_model_test_best = model_best_test_by_model.get(model_name)
                    model_best_test_by_model[model_name] = (
                        test_mrr if prev_model_test_best is None else max(float(prev_model_test_best), float(test_mrr))
                    )
                current_model_best_valid = model_best_valid_by_model.get(str(row["model_label"]))
                current_model_best_test = model_best_test_by_model.get(str(row["model_label"]))
                summary_row = {
                    "model": row["model_label"],
                    "global_best_valid_mrr20": "" if global_best_valid is None else f"{float(global_best_valid):.6f}",
                    "global_best_test_mrr20": "" if global_best_test is None else f"{float(global_best_test):.6f}",
                    "model_best_valid_mrr20": "" if current_model_best_valid is None else f"{float(current_model_best_valid):.6f}",
                    "model_best_test_mrr20": "" if current_model_best_test is None else f"{float(current_model_best_test):.6f}",
                    "run_best_valid_mrr20": "" if run_best is None else f"{float(run_best):.6f}",
                    "run_best_test_mrr20": "" if test_mrr is None else f"{float(test_mrr):.6f}",
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": dataset,
                    "hparam_id": row["hparam_id"],
                    "seed_id": row["seed_id"],
                    "gpu_id": gpu_id,
                    "status": status,
                    "n_completed": "" if n_completed is None else int(n_completed),
                    "interrupted": "" if interrupted is None else bool(interrupted),
                    "special_ok": bool(special_ok),
                    "candidate_id": row.get("candidate_id", ""),
                    "candidate_source": row.get("candidate_source", ""),
                    "source_stage": row.get("source_stage", ""),
                    "transfer_from_dataset": row.get("transfer_from_dataset", ""),
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                base._append_summary_row(summary_path, summary_row)
                base._update_baseline_phase_summary(dataset, base.PHASE_ID)

                if int(rc) != 0:
                    raise RuntimeError(f"run failed: dataset={dataset} run_phase={row['run_phase']} rc={rc}")

            for gpu_id in done_gpus:
                active.pop(gpu_id, None)

            pending = bool(shared_queue)
            if not pending and not active:
                break
            time.sleep(1)
    except Exception:
        base._terminate_active(active)
        raise

    print(f"[{base.PHASE_ID}] summary updated: {summary_path}")
    base._update_baseline_phase_summary(dataset, base.PHASE_ID)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P22 StageG cross 6x9 launcher")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--seed-base", type=int, default=220000)

    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=8)
    return parser.parse_args()


def _check_runtime_models(models: list[str]) -> None:
    py = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    labels = [OPTION_TO_LABEL[m] for m in models if m in OPTION_TO_LABEL]
    script = (
        "import recbole_patch  # noqa: F401\n"
        "from recbole.utils import utils as rbu\n"
        f"models={labels!r}\n"
        "for name in models:\n"
        "    _ = rbu.get_model(name)\n"
        "print('[ENV_CHECK] StageG model registration OK')\n"
    )
    proc = subprocess.run([py, "-c", script], cwd=base.EXP_DIR, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Model registration check failed:\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"
    gpus = base._parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = list(dict.fromkeys(base._parse_csv_strings(args.gpus)))
    if not gpus:
        raise RuntimeError("No GPUs provided")

    datasets = base._parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets provided")

    models = [m.lower() for m in base._parse_csv_strings(args.models)]
    if not models:
        raise RuntimeError("No models provided")
    unknown = [m for m in models if m not in OPTION_TO_LABEL]
    if unknown:
        raise RuntimeError(f"Unknown models: {unknown}")

    print(
        f"[config] datasets={','.join(datasets)} models={','.join(models)} gpus={','.join(gpus)} "
        f"seeds={args.seeds} seed_base={args.seed_base} smoke_test={args.smoke_test}"
    )

    _check_runtime_models(models)

    for d_idx, dataset in enumerate(datasets):
        rc = _run_dataset(dataset, args, dataset_order_idx=d_idx, gpus=gpus)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
