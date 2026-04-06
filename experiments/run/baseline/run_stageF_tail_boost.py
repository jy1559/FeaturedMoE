#!/usr/bin/env python3
"""Stage F runner (tail-boost) for GRU4Rec/FAME/DuoRec.

Built on top of run_stageC_focus engine, but with:
- parent search widened to StageE + StageD + StageC + StageB + manual recovery anchors
- LR windows centered around observed A~E winners per model/dataset
- MAX_ITEM_LIST_LENGTH explicitly fixed in search space (singleton) for consistency/speed
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stageC_focus as base  # noqa: E402

# Stage identity
base.AXIS = "StageF_TailBoost_anchor2_core5"
base.PHASE_ID = "P21"
base.PHASE_NAME = "STAGEF_TAILBOOST_ANCHOR2_CORE5"
base.AXIS_DESC = "stagef_tailboost_anchor2_core5"
base.STAGE_ID = "F"

# Parent candidates are selected primarily from Stage E, then widened.
base.STAGEB_AXIS = "StageE_ReLRSeed_anchor2_core5"

# Stage F profiles: compact but diverse local search around model-specific sweet spots.
base.C_PROFILES = {
    "F1": {
        "name": "stable_mid",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.00,
        "lr_span_mult": 0.68,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.30, "lmd": 0.03, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.12},
        "fame": {"num_experts": 3},
    },
    "F2": {
        "name": "high_lr_push",
        "max_len": 10,
        "dropout_delta": -0.02,
        "wd_mult": 0.85,
        "lr_mult": 1.18,
        "lr_span_mult": 0.72,
        "sasrec": {"inner_ratio": 3, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "su", "tau": 0.35, "lmd": 0.02, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 4},
    },
    "F3": {
        "name": "low_lr_safe",
        "max_len": 10,
        "dropout_delta": 0.03,
        "wd_mult": 1.25,
        "lr_mult": 0.82,
        "lr_span_mult": 0.70,
        "sasrec": {"inner_ratio": 2, "heads_mode": "half_if_small"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.55, "lmd": 0.03, "lmd_sem": 0.02},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": True, "lambda_attr": 0.08},
        "fame": {"num_experts": 2},
    },
    "F4": {
        "name": "regularized_outlier",
        "max_len": 15,
        "dropout_delta": 0.06,
        "wd_mult": 1.70,
        "lr_mult": 0.90,
        "lr_span_mult": 0.80,
        "sasrec": {"inner_ratio": 1, "heads_mode": "half"},
        "gru4rec": {"layer_delta": -1},
        "duorec": {"contrast": "un", "tau": 0.75, "lmd": 0.08, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "concat", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 2},
    },
    "F5": {
        "name": "fame_expert_boost",
        "max_len": 10,
        "dropout_delta": -0.01,
        "wd_mult": 0.92,
        "lr_mult": 1.10,
        "lr_span_mult": 0.66,
        "sasrec": {"inner_ratio": 3, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "su", "tau": 0.45, "lmd": 0.00, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.18},
        "fame": {"num_experts": 6},
    },
    "F6": {
        "name": "duorec_recover",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.02,
        "lr_span_mult": 0.64,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.25, "lmd": 0.02, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.10},
        "fame": {"num_experts": 3},
    },
    "F7": {
        "name": "capacity_probe",
        "max_len": 10,
        "dropout_delta": -0.04,
        "wd_mult": 0.78,
        "lr_mult": 1.28,
        "lr_span_mult": 0.62,
        "sasrec": {"inner_ratio": 4, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "su", "tau": 0.38, "lmd": 0.01, "lmd_sem": 0.08},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.20},
        "fame": {"num_experts": 6},
    },
    "F8": {
        "name": "long_context_probe",
        "max_len": 15,
        "dropout_delta": 0.02,
        "wd_mult": 1.10,
        "lr_mult": 0.98,
        "lr_span_mult": 0.76,
        "sasrec": {"inner_ratio": 2, "heads_mode": "half_if_small"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.45, "lmd": 0.04, "lmd_sem": 0.02},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": True, "lambda_attr": 0.10},
        "fame": {"num_experts": 4},
    },
}

base.PROFILE_ORDER = list(base.C_PROFILES.keys())
base.PROFILE_COST_WEIGHT.update(
    {
        "F1": 1.00,
        "F2": 1.08,
        "F3": 1.00,
        "F4": 1.05,
        "F5": 1.10,
        "F6": 1.03,
        "F7": 1.15,
        "F8": 1.08,
    }
)

# Model-specific shortlist for Stage F:
# - keep promising directions
# - include at least one outlier/probe profile per model
MODEL_PROFILE_WHITELIST = {
    "gru4rec": {"F1", "F2", "F3", "F7"},
    "fame": {"F1", "F2", "F5", "F7"},
    "duorec": {"F1", "F2", "F4", "F6"},
}

# Manual recovery anchors from observed best runs (especially amazon tails).
MANUAL_PARENT_BANK: dict[str, dict[str, list[dict[str, Any]]]] = {
    "lastfm0.03": {
        "gru4rec": [
            {
                "id": "MANUAL_LFM_GRU_STABLE",
                "config": {
                    "hidden_size": 112,
                    "embedding_size": 112,
                    "layers": 3,
                    "num_layers": 3,
                    "max_len": 10,
                    "dropout": 0.18,
                    "weight_decay": 1.2e-4,
                },
                "lr_lo": 1.7e-3,
                "lr_hi": 9.0e-3,
                "lr_band_id": "M_GRU_LFM_1",
                "must_keep": False,
                "priority": 0.88,
            }
        ],
        "fame": [
            {
                "id": "MANUAL_LFM_FAME_STRONG",
                "config": {
                    "hidden_size": 208,
                    "embedding_size": 208,
                    "layers": 1,
                    "num_layers": 1,
                    "heads": 2,
                    "inner_size": 416,
                    "max_len": 10,
                    "dropout": 0.23,
                    "weight_decay": 3.0e-4,
                    "num_experts": 3,
                },
                "lr_lo": 2.7e-4,
                "lr_hi": 3.9e-3,
                "lr_band_id": "M_FAME_LFM_1",
                "must_keep": False,
                "priority": 0.90,
            }
        ],
        "duorec": [
            {
                "id": "MANUAL_LFM_DUO_STRONG",
                "config": {
                    "hidden_size": 152,
                    "embedding_size": 152,
                    "layers": 1,
                    "num_layers": 1,
                    "heads": 4,
                    "inner_size": 304,
                    "max_len": 10,
                    "dropout": 0.10,
                    "weight_decay": 1.0e-4,
                    "contrast": "un",
                    "tau": 0.55,
                    "lmd": 0.03,
                    "lmd_sem": 0.02,
                    "semantic_sample_max_tries": 2,
                },
                "lr_lo": 2.2e-4,
                "lr_hi": 2.8e-3,
                "lr_band_id": "M_DUO_LFM_1",
                "must_keep": False,
                "priority": 0.90,
            }
        ],
    },
    "amazon_beauty": {
        "gru4rec": [
            {
                "id": "MANUAL_AB_GRU_RECOVER",
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
                "lr_band_id": "M_GRU_AB_1",
                "must_keep": True,
                "priority": 0.94,
            },
            {
                "id": "MANUAL_AB_GRU_ALT",
                "config": {
                    "hidden_size": 160,
                    "embedding_size": 160,
                    "layers": 2,
                    "num_layers": 2,
                    "max_len": 10,
                    "dropout": 0.18,
                    "weight_decay": 2.0e-4,
                },
                "lr_lo": 9.0e-4,
                "lr_hi": 5.2e-3,
                "lr_band_id": "M_GRU_AB_2",
                "must_keep": False,
                "priority": 0.86,
            },
        ],
        "fame": [
            {
                "id": "MANUAL_AB_FAME_RECOVER",
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
                "lr_band_id": "M_FAME_AB_1",
                "must_keep": True,
                "priority": 0.93,
            },
            {
                "id": "MANUAL_AB_FAME_ALT",
                "config": {
                    "hidden_size": 208,
                    "embedding_size": 208,
                    "layers": 1,
                    "num_layers": 1,
                    "heads": 4,
                    "inner_size": 416,
                    "max_len": 10,
                    "dropout": 0.14,
                    "weight_decay": 1.65e-4,
                    "num_experts": 2,
                },
                "lr_lo": 1.0e-3,
                "lr_hi": 1.0e-2,
                "lr_band_id": "M_FAME_AB_2",
                "must_keep": False,
                "priority": 0.88,
            },
        ],
        "duorec": [
            {
                "id": "MANUAL_AB_DUO_RECOVER",
                "config": {
                    "hidden_size": 120,
                    "embedding_size": 120,
                    "layers": 2,
                    "num_layers": 2,
                    "heads": 2,
                    "inner_size": 240,
                    "max_len": 10,
                    "dropout": 0.21,
                    "weight_decay": 3.4425e-4,
                    "contrast": "un",
                    "tau": 0.25,
                    "lmd": 0.02,
                    "lmd_sem": 0.00,
                    "semantic_sample_max_tries": 2,
                },
                "lr_lo": 4.46e-4,
                "lr_hi": 2.34e-3,
                "lr_band_id": "M_DUO_AB_1",
                "must_keep": True,
                "priority": 0.96,
            },
            {
                "id": "MANUAL_AB_DUO_ALT",
                "config": {
                    "hidden_size": 152,
                    "embedding_size": 152,
                    "layers": 1,
                    "num_layers": 1,
                    "heads": 2,
                    "inner_size": 304,
                    "max_len": 10,
                    "dropout": 0.14,
                    "weight_decay": 1.65e-4,
                    "contrast": "un",
                    "tau": 0.30,
                    "lmd": 0.04,
                    "lmd_sem": 0.00,
                    "semantic_sample_max_tries": 2,
                },
                "lr_lo": 3.4e-4,
                "lr_hi": 4.9e-3,
                "lr_band_id": "M_DUO_AB_2",
                "must_keep": False,
                "priority": 0.89,
            },
        ],
    },
}


SOURCE_PRIORITY = {
    "StageE_ReLRSeed_anchor2_core5": 1.00,
    "StageD_MicroWide_anchor2_core5": 0.96,
    "StageC_Focus_anchor2_core5": 0.93,
    "StageB_Structure_anchor2_core5": 0.90,
    "summary": 0.94,
    "manual": 0.98,
}


def _metric(x: Any) -> float | None:
    return base._metric_to_float(x)


def _clamp_lr(lo: float, hi: float) -> tuple[float, float]:
    lo2 = max(float(base.LR_CLAMP_MIN), min(float(base.LR_CLAMP_MAX), float(lo)))
    hi2 = max(float(base.LR_CLAMP_MIN), min(float(base.LR_CLAMP_MAX), float(hi)))
    if hi2 <= lo2:
        hi2 = min(float(base.LR_CLAMP_MAX), max(lo2 * 1.35, lo2 + 1e-6))
    return float(lo2), float(hi2)


def _candidate_file(axis: str, dataset: str) -> Path | None:
    root = base.LOG_ROOT / axis / base._dataset_tag(dataset)
    p1 = root / "stageC_candidates.json"
    if p1.exists():
        return p1
    p2 = root / "stageB_candidates.json"
    if p2.exists():
        return p2
    return None


def _build_candidate_record(
    *,
    model_option: str,
    parent_id: str,
    config: dict[str, Any],
    lr_lo: float,
    lr_hi: float,
    lr_band_id: str,
    parent_run_phase: str,
    source: str,
    valid: float | None,
    test: float | None,
    completion_ratio: float | None,
    must_keep: bool,
    extra_priority: float,
) -> dict[str, Any]:
    cfg = base._normalize_parent_config(model_option, dict(config or {}))
    lo, hi = _clamp_lr(float(lr_lo), float(lr_hi))
    v = 0.0 if valid is None else float(valid)
    t = 0.0 if test is None else float(test)
    c = 0.0 if completion_ratio is None else max(0.0, min(1.0, float(completion_ratio)))
    src = float(SOURCE_PRIORITY.get(source, 0.85))
    score = 1.00 * v + 0.25 * t + 0.015 * c + 0.03 * src + float(extra_priority)
    return {
        "parent_profile_id": str(parent_id),
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "lr_band_id": str(lr_band_id),
        "parent_run_phase": str(parent_run_phase),
        "source": str(source),
        "valid": None if valid is None else float(valid),
        "test": None if test is None else float(test),
        "score": float(score),
        "must_keep": bool(must_keep),
    }


def _collect_axis_candidates(axis: str, dataset: str, model_option: str, model_label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cpath = _candidate_file(axis, dataset)
    if cpath is None:
        return out
    try:
        payload = json.loads(cpath.read_text(encoding="utf-8"))
    except Exception:
        return out

    node = payload.get("models", {}).get(str(model_label), {})
    if not isinstance(node, dict):
        return out

    records: list[dict[str, Any]] = []
    selected = node.get("selected_profiles") or []
    profiles = node.get("profiles") or []
    if isinstance(selected, list):
        records.extend(selected)
    if isinstance(profiles, list):
        records.extend(profiles[:6])

    for it in records:
        if not isinstance(it, dict):
            continue
        cfg = dict(it.get("config", {}) or {})
        lo = _metric(it.get("lr_lo"))
        hi = _metric(it.get("lr_hi"))
        best_lr = _metric(it.get("best_lr"))
        if lo is None or hi is None or hi <= lo:
            if best_lr is not None:
                lo = max(float(base.LR_CLAMP_MIN), float(best_lr) / 1.9)
                hi = min(float(base.LR_CLAMP_MAX), float(best_lr) * 1.9)
            else:
                lo, hi, _ = base._load_stagea_lr_window(dataset, model_label)

        valid = _metric(it.get("valid_mrr20_best"))
        test = _metric(it.get("test_mrr20_mean"))
        completion_ratio = _metric(it.get("completion_ratio"))
        parent_id = str(it.get("combo_id") or it.get("profile_id") or f"{axis}_P")

        out.append(
            _build_candidate_record(
                model_option=model_option,
                parent_id=parent_id,
                config=cfg,
                lr_lo=float(lo),
                lr_hi=float(hi),
                lr_band_id=str(it.get("lr_band_id", "PSEL")),
                parent_run_phase=str(it.get("representative_run_phase", "")),
                source=axis,
                valid=valid,
                test=test,
                completion_ratio=completion_ratio,
                must_keep=False,
                extra_priority=0.0,
            )
        )

    return out


def _config_from_result_payload(model_option: str, payload: dict[str, Any]) -> dict[str, Any]:
    cfg = base._default_parent_config(model_option)
    fixed = payload.get("fixed_search")
    if not isinstance(fixed, dict):
        fixed = {}

    h = _metric(fixed.get("hidden_size"))
    if h is not None:
        cfg["hidden_size"] = int(h)
        cfg["embedding_size"] = int(h)

    emb = _metric(fixed.get("embedding_size"))
    if emb is not None:
        cfg["embedding_size"] = int(emb)

    nl = _metric(fixed.get("num_layers"))
    if nl is None:
        nl = _metric(fixed.get("n_layers"))
    if nl is not None:
        cfg["layers"] = int(nl)
        cfg["num_layers"] = int(nl)

    heads = _metric(fixed.get("num_heads"))
    if heads is None:
        heads = _metric(fixed.get("n_heads"))
    if heads is not None:
        cfg["heads"] = int(heads)

    inner = _metric(fixed.get("inner_size"))
    if inner is not None:
        cfg["inner_size"] = int(inner)

    dp = _metric(fixed.get("dropout_ratio"))
    if dp is None:
        dp = _metric(fixed.get("dropout_prob"))
    if dp is None:
        dp = _metric(fixed.get("hidden_dropout_prob"))
    if dp is not None:
        cfg["dropout"] = float(dp)

    wd = _metric(fixed.get("weight_decay"))
    if wd is not None:
        cfg["weight_decay"] = float(wd)

    bp = payload.get("best_params")
    if isinstance(bp, dict):
        ml = _metric(bp.get("MAX_ITEM_LIST_LENGTH"))
        if ml is not None:
            cfg["max_len"] = int(ml)

    if model_option == "fame":
        ne = _metric(fixed.get("num_experts"))
        if ne is not None:
            cfg["num_experts"] = int(ne)

    return base._normalize_parent_config(model_option, cfg)


def _collect_summary_candidates(axis: str, dataset: str, model_option: str, model_label: str, limit: int = 3) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    spath = base.LOG_ROOT / axis / base._dataset_tag(dataset) / "summary.csv"
    if not spath.exists():
        return out

    rows: list[dict[str, Any]] = []
    try:
        with spath.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if str(row.get("model", "")).strip() != str(model_label):
                    continue
                valid = _metric(row.get("run_best_valid_mrr20"))
                if valid is None:
                    continue
                rows.append(row)
    except Exception:
        return out

    rows.sort(
        key=lambda r: (
            float(_metric(r.get("run_best_valid_mrr20")) or -1e9),
            float(_metric(r.get("run_best_test_mrr20")) or -1e9),
            str(r.get("timestamp_utc", "")),
        ),
        reverse=True,
    )

    for row in rows[: max(1, int(limit))]:
        rp = Path(str(row.get("result_path", "") or "").strip())
        if not rp.exists():
            continue
        try:
            payload = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue

        cfg = _config_from_result_payload(model_option, payload)
        lo = _metric(row.get("lr_lo"))
        hi = _metric(row.get("lr_hi"))
        best_lr = _metric(payload.get("best_params", {}).get("learning_rate"))
        if lo is None or hi is None or hi <= lo:
            if best_lr is not None:
                lo = max(float(base.LR_CLAMP_MIN), float(best_lr) / 1.8)
                hi = min(float(base.LR_CLAMP_MAX), float(best_lr) * 1.8)
            else:
                lo, hi, _ = base._load_stagea_lr_window(dataset, model_label)

        n_completed = _metric(payload.get("n_completed"))
        max_evals = _metric(payload.get("max_evals"))
        completion = None
        if n_completed is not None and max_evals is not None and max_evals > 0:
            completion = float(n_completed) / float(max_evals)

        out.append(
            _build_candidate_record(
                model_option=model_option,
                parent_id=str(row.get("profile_id", "S")),
                config=cfg,
                lr_lo=float(lo),
                lr_hi=float(hi),
                lr_band_id=str(row.get("lr_band_id", "S")),
                parent_run_phase=str(row.get("run_phase", "")),
                source="summary",
                valid=_metric(row.get("run_best_valid_mrr20")),
                test=_metric(row.get("run_best_test_mrr20")),
                completion_ratio=completion,
                must_keep=False,
                extra_priority=0.01,
            )
        )

    return out


def _collect_manual_candidates(dataset: str, model_option: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in MANUAL_PARENT_BANK.get(str(dataset), {}).get(str(model_option), []):
        out.append(
            _build_candidate_record(
                model_option=model_option,
                parent_id=str(item.get("id", "MANUAL")),
                config=dict(item.get("config", {}) or {}),
                lr_lo=float(item.get("lr_lo", 5e-4)),
                lr_hi=float(item.get("lr_hi", 2e-3)),
                lr_band_id=str(item.get("lr_band_id", "MANUAL")),
                parent_run_phase="",
                source="manual",
                valid=None,
                test=None,
                completion_ratio=1.0,
                must_keep=bool(item.get("must_keep", False)),
                extra_priority=float(item.get("priority", 0.0)),
            )
        )
    return out


def _cfg_signature(cfg: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(cfg.get("hidden_size", 0)),
        int(cfg.get("embedding_size", 0)),
        int(cfg.get("layers", cfg.get("num_layers", 0))),
        int(cfg.get("heads", 0)),
        int(cfg.get("inner_size", 0)),
        int(cfg.get("max_len", 0)),
        round(float(cfg.get("dropout", 0.0)), 4),
        round(float(cfg.get("weight_decay", 0.0)), 7),
        int(cfg.get("num_experts", 0)),
        str(cfg.get("contrast", "")),
        round(float(cfg.get("tau", 0.0)), 3),
        round(float(cfg.get("lmd", 0.0)), 3),
        round(float(cfg.get("lmd_sem", 0.0)), 3),
        str(cfg.get("fusion_type", "")),
        int(bool(cfg.get("use_attribute_predictor", True))),
        round(float(cfg.get("lambda_attr", 0.0)), 3),
    )


def _merge_candidates(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for c in cands:
        key = _cfg_signature(dict(c.get("config", {})))
        cur = merged.get(key)
        if cur is None:
            c["_sources"] = [str(c.get("source", "?"))]
            merged[key] = c
            continue
        cur["lr_lo"] = float(min(float(cur.get("lr_lo", 1e9)), float(c.get("lr_lo", 1e9))))
        cur["lr_hi"] = float(max(float(cur.get("lr_hi", 0.0)), float(c.get("lr_hi", 0.0))))
        cur["score"] = float(max(float(cur.get("score", -1e9)), float(c.get("score", -1e9))))
        cur["must_keep"] = bool(cur.get("must_keep", False) or c.get("must_keep", False))
        srcs = list(cur.get("_sources", []))
        srcs.append(str(c.get("source", "?")))
        cur["_sources"] = list(dict.fromkeys(srcs))
        if _metric(c.get("valid")) is not None:
            pv = _metric(cur.get("valid"))
            cv = _metric(c.get("valid"))
            if pv is None or (cv is not None and float(cv) > float(pv)):
                cur["valid"] = cv
        if _metric(c.get("test")) is not None:
            pt = _metric(cur.get("test"))
            ct = _metric(c.get("test"))
            if pt is None or (ct is not None and float(ct) > float(pt)):
                cur["test"] = ct
    out = list(merged.values())
    for c in out:
        lo, hi = _clamp_lr(float(c.get("lr_lo", 5e-4)), float(c.get("lr_hi", 2e-3)))
        c["lr_lo"] = lo
        c["lr_hi"] = hi
    return out


def _select_candidates(cands: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    if not cands:
        return []
    cands2 = _merge_candidates(cands)
    cands2.sort(
        key=lambda x: (
            1 if bool(x.get("must_keep", False)) else 0,
            float(x.get("score", -1e9)),
            float(_metric(x.get("valid")) or -1e9),
            float(_metric(x.get("test")) or -1e9),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    must = [c for c in cands2 if bool(c.get("must_keep", False))]
    for c in must:
        selected.append(c)
    for c in cands2:
        if len(selected) >= max(1, int(topk)):
            break
        if c in selected:
            continue
        selected.append(c)
    return selected[: max(1, int(topk))]


def _load_parent_candidates(dataset: str, model_option: str, model_label: str, topk: int):
    # 1) Stage E selected/top profiles
    # 2) Stage D selected/top profiles
    # 3) Stage C/B selected/top profiles
    # 4) Top summary rows (captures runs not promoted into candidate JSON)
    # 5) Manual anchors for tail recovery
    all_cands: list[dict[str, Any]] = []

    for axis in (
        "StageE_ReLRSeed_anchor2_core5",
        "StageD_MicroWide_anchor2_core5",
        "StageC_Focus_anchor2_core5",
        "StageB_Structure_anchor2_core5",
    ):
        all_cands.extend(_collect_axis_candidates(axis, dataset, model_option, model_label))

    all_cands.extend(_collect_summary_candidates("StageE_ReLRSeed_anchor2_core5", dataset, model_option, model_label, limit=4))
    all_cands.extend(_collect_summary_candidates("StageD_MicroWide_anchor2_core5", dataset, model_option, model_label, limit=4))
    all_cands.extend(_collect_summary_candidates("StageC_Focus_anchor2_core5", dataset, model_option, model_label, limit=5))

    all_cands.extend(_collect_manual_candidates(dataset, model_option))

    picked = _select_candidates(all_cands, max(1, int(topk)))
    if picked:
        out = []
        for c in picked:
            out.append(
                {
                    "parent_profile_id": str(c.get("parent_profile_id", "F?")),
                    "config": base._normalize_parent_config(model_option, dict(c.get("config", {}))),
                    "lr_lo": float(c.get("lr_lo", 5e-4)),
                    "lr_hi": float(c.get("lr_hi", 2e-3)),
                    "lr_band_id": str(c.get("lr_band_id", "FSEL")),
                    "parent_run_phase": str(c.get("parent_run_phase", "")),
                }
            )
        return out

    # Final fallback: Stage A-derived LR + default parent config.
    lo, hi, band = base._load_stagea_lr_window(dataset, model_label)
    return [
        {
            "parent_profile_id": "F_FALLBACK",
            "config": base._default_parent_config(model_option),
            "lr_lo": float(lo),
            "lr_hi": float(hi),
            "lr_band_id": str(band),
            "parent_run_phase": "",
        }
    ]


base._load_stageb_parent_candidates = _load_parent_candidates


# Ensure max_len is truly fixed (singleton) in hyperopt search to reduce runtime/noise.
_orig_build_command = base._build_command


def _build_command_fixed_maxlen(row: dict[str, Any], gpu_id: str) -> list[str]:
    cmd = _orig_build_command(row, gpu_id)
    cfg = dict(row.get("config", {}) or {})
    max_len = int(cfg.get("max_len", base.MAX_LEN_DEFAULT))
    fixed = f"++search.MAX_ITEM_LIST_LENGTH={base.hydra_literal([max_len])}"
    ovrd = "++search_space_type_overrides.MAX_ITEM_LIST_LENGTH=choice"
    if fixed not in cmd:
        cmd.append(fixed)
    if ovrd not in cmd:
        cmd.append(ovrd)
    return cmd


base._build_command = _build_command_fixed_maxlen


_orig_build_rows = base._build_rows


def _build_rows_stagef(dataset: str, args, *, dataset_order_idx: int):
    rows = _orig_build_rows(dataset, args, dataset_order_idx=dataset_order_idx)
    filtered = []
    for r in rows:
        model_opt = str(r.get("model_option", "")).lower()
        pid = str(r.get("profile_id", ""))
        allow = MODEL_PROFILE_WHITELIST.get(model_opt)
        if allow is not None and pid not in allow:
            continue
        filtered.append(r)
    return filtered


base._build_rows = _build_rows_stagef


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
