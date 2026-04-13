#!/usr/bin/env python3
"""Stage I progressive baseline recovery and verification runner.

Design goals:
- Replace Stage H's mixed targeted-recovery modes with a cleaner 3-pass flow
- Pass1: rebuild the worst combos with broad, general hparam families
- Pass2: refine all still-underperforming combos around the latest best anchor
- Pass3: verify only the bottom-3 models per dataset with small local variants
- Reuse Stage G / final baseline runtime, logging, and summary infrastructure
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

import run_stageG_cross6x9 as sg
from slack_progress import SlackProgressNotifier

base = sg.base


base.AXIS = "StageI_ProgressiveRecovery_anchor2_core5"
base.PHASE_ID = "P24"
base.PHASE_NAME = "STAGEI_PROGRESSIVE_RECOVERY_ANCHOR2_CORE5"
base.AXIS_DESC = "stagei_progressiverecovery_anchor2_core5"


STEP_ORDER = ["step1", "step2", "step3"]
PASS_ORDER = ["pass1", "pass2", "pass3"]
PASS_TOKEN = {"pass1": "I1", "pass2": "I2", "pass3": "I3"}
PASS_LABEL = {"pass1": "StageI Pass1", "pass2": "StageI Pass2", "pass3": "StageI Pass3"}
PASS_SPECS: Dict[str, Dict[str, int | float]] = {
    "pass1": {"candidate_count": 24, "max_evals": 10, "tune_epochs": 100, "tune_patience": 5},
    "pass2": {"candidate_count": 12, "max_evals": 8, "tune_epochs": 50, "tune_patience": 5},
    "pass3": {"candidate_count": 6, "max_evals": 10, "tune_epochs": 100, "tune_patience": 10},
}
PASS1_RATIO_CUT = 0.35
PASS2_RATIO_CUT = 0.80
PASS3_BOTTOM_K = 5
PASS1_HPARAM_BANK = [f"H{i}" for i in range(1, 13)]
STEP2_PASS_SEQUENCE = ["pass1", "pass2"]
STEP3_PASS_SEQUENCE = ["pass1", "pass2"]

SPARSE_DATASETS = {"amazon_beauty", "foursquare", "retail_rocket"}
OOM_SENSITIVE_COMBOS = {
    ("KuaiRecLargeStrictPosV2_0.2", "sigma"),
    ("KuaiRecLargeStrictPosV2_0.2", "bsarec"),
    ("retail_rocket", "gru4rec"),
    ("retail_rocket", "tisasrec"),
}
HOST_MEMORY_SERIAL_COMBOS = {
    ("retail_rocket", "tisasrec"),
    ("retail_rocket", "gru4rec"),
}

DEFAULT_DATASETS = list(base.DEFAULT_DATASETS)
OPTION_TO_LABEL = {
    str(m["model_option"]).lower(): ("DIF-SR" if str(m["model_label"]) == "DIFSR" else str(m["model_label"]))
    for m in base.MODEL_SPECS
}
DEFAULT_MODELS = list(OPTION_TO_LABEL.keys())

_orig_model_runtime_resource_overrides = base._model_runtime_resource_overrides
_orig_model_algorithm_overrides_stagei = base._model_algorithm_overrides


def _normalize_model_name(text: str) -> str:
    return str(text or "").strip().lower().replace("-", "").replace("_", "")


LABEL_TO_OPTION = {_normalize_model_name(label): option for option, label in OPTION_TO_LABEL.items()}
LABEL_TO_OPTION["difsr"] = "difsr"


def _step_name(args: argparse.Namespace) -> str:
    step_name = str(getattr(args, "step_name", "step1") or "step1").strip().lower()
    return step_name if step_name in STEP_ORDER else "step1"


def _step_pass_token(step_name: str, pass_name: str) -> str:
    base_token = PASS_TOKEN[str(pass_name)]
    if str(step_name) == "step1":
        return base_token
    if str(step_name) == "step2":
        return f"{base_token}S2"
    return f"{base_token}S3"


def _row_step_name(row: Dict[str, Any]) -> str:
    return str(row.get("step_name", "step1") or "step1").strip().lower()


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
        "anchor_run_id",
        "step_name",
        "pass_name",
        "hypothesis_family",
        "root_cause_tag",
        "batch_profile",
        "result_path",
        "timestamp_utc",
    ]


base._summary_fieldnames = _summary_fieldnames


def _qdim(value: float, *, minimum: int = 64, multiple: int = 8) -> int:
    v = max(float(minimum), float(value))
    q = int(round(v / float(max(1, multiple)))) * int(max(1, multiple))
    return max(int(minimum), q)


def _align_hidden(hidden: int, heads: int) -> int:
    hidden = max(int(hidden), max(int(heads), 8))
    hidden = _qdim(hidden, minimum=max(int(heads), 64), multiple=max(int(heads), 8))
    if hidden % max(int(heads), 1) != 0:
        hidden = ((hidden // max(int(heads), 1)) + 1) * max(int(heads), 1)
    if hidden % 8 != 0:
        hidden = ((hidden // 8) + 1) * 8
    return max(hidden, max(int(heads), 64))


def _normalize_stagei_cfg(model_option: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg2 = sg._normalize_cfg(model_option, dict(cfg))
    heads = int(cfg2.get("heads", cfg2.get("num_heads", 4)))
    heads = max(1, min(8, heads))
    hidden = _align_hidden(int(cfg2.get("hidden_size", 128)), heads)
    cfg2["heads"] = heads
    cfg2["hidden_size"] = hidden
    cfg2["embedding_size"] = hidden

    inner = int(cfg2.get("inner_size", hidden * 2))
    inner = max(hidden, _qdim(inner, minimum=hidden, multiple=8))
    if inner < hidden * 2:
        inner = _qdim(hidden * 2, minimum=hidden, multiple=8)
    cfg2["inner_size"] = inner
    if model_option == "bsarec":
        cfg2["bsarec_alpha"] = min(0.95, max(0.05, float(cfg2.get("bsarec_alpha", 0.5))))
        cfg2["bsarec_c"] = max(1, min(31, int(cfg2.get("bsarec_c", 3))))
    if model_option == "fame":
        layers = int(cfg2.get("num_layers", cfg2.get("layers", 2)))
        cfg2["fame_moe_last_k_layers"] = max(
            1,
            min(layers, int(cfg2.get("fame_moe_last_k_layers", 1))),
        )
    return sg._normalize_cfg(model_option, cfg2)


def _canonical_search_dict(search: Dict[str, Any]) -> str:
    try:
        return json.dumps(search, sort_keys=True, ensure_ascii=True)
    except Exception:
        return str(search)


def _dedup_candidates_stagei(cands: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for c in cands:
        cfg = dict(c.get("config", {}) or {})
        key = (
            int(cfg.get("hidden_size", 0)),
            int(cfg.get("num_layers", cfg.get("layers", 0))),
            int(cfg.get("max_len", 0)),
            round(float(c.get("lr_lo", 0.0)), 7),
            round(float(c.get("lr_hi", 0.0)), 7),
            _canonical_search_dict(dict(c.get("search_overrides", {}) or {})),
            _canonical_search_dict(dict(c.get("search_space_type_overrides", {}) or {})),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _cap_max_len(
    dataset: str,
    model_option: str,
    pass_name: str,
    max_len: int,
    *,
    step_name: str = "step1",
    hypothesis_family: str = "",
) -> int:
    dataset_sparse = dataset in SPARSE_DATASETS
    if pass_name == "pass1":
        cap = 12 if dataset_sparse else 24
    elif pass_name == "pass2":
        cap = 14 if dataset_sparse else 28
    else:
        cap = 16 if dataset_sparse else 30

    if model_option in {"gru4rec", "fame"}:
        cap = min(cap, 10 if pass_name == "pass1" else 12 if pass_name == "pass2" else 14)
    if model_option == "sigma":
        cap = min(cap, 18 if pass_name == "pass1" else 20 if pass_name == "pass2" else 24)
    if model_option == "duorec":
        cap = min(cap, 16 if pass_name == "pass1" else 18 if pass_name == "pass2" else 20)

    if step_name == "step2":
        if model_option == "bsarec":
            cap = max(cap, 18 if dataset_sparse else 28)
        elif model_option == "fame":
            cap = max(cap, 16 if dataset_sparse else 24)
        elif model_option == "gru4rec":
            cap = max(cap, 14 if dataset_sparse else 22)
        elif model_option == "tisasrec":
            cap = max(cap, 20 if dataset_sparse else 30)
        if str(hypothesis_family) in {"long_sparse", "time_balanced", "longer_context"}:
            cap = max(cap, 20 if dataset_sparse else 30)
    elif step_name == "step3":
        if model_option == "bsarec":
            cap = max(cap, 40 if dataset_sparse else 50)
        elif model_option == "fame":
            cap = max(cap, 32 if dataset_sparse else 40)
        elif model_option == "gru4rec":
            cap = max(cap, 32 if dataset_sparse else 40)
        elif model_option == "tisasrec":
            cap = max(cap, 28 if dataset_sparse else 36)
        if str(hypothesis_family) in {
            "beauty_long_context",
            "long_context",
            "long_context_probe",
            "beauty_official_sparse",
            "time_balanced",
            "long_sparse",
            "longer_context",
        }:
            cap = max(cap, 50 if dataset_sparse else 60)
    return max(5, min(int(max_len), int(cap)))


def _prepare_cfg_for_pass(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    pass_name: str,
    *,
    step_name: str = "step1",
    hypothesis_family: str = "",
) -> Dict[str, Any]:
    cfg2 = _normalize_stagei_cfg(model_option, cfg)
    cfg2["max_len"] = _cap_max_len(
        dataset,
        model_option,
        pass_name,
        int(cfg2.get("max_len", 20)),
        step_name=step_name,
        hypothesis_family=hypothesis_family,
    )
    if model_option == "tisasrec":
        cfg2["time_span"] = max(64, int(min(int(cfg2.get("time_span", 256)), int(cfg2["max_len"]) * 32)))
    return _normalize_stagei_cfg(model_option, cfg2)


def _parse_batch_overrides(overrides: list[str]) -> tuple[int, int]:
    train_bs = 2048
    eval_bs = 4096
    for item in overrides:
        text = str(item)
        if text.startswith("++train_batch_size="):
            try:
                train_bs = int(float(text.split("=", 1)[1]))
            except Exception:
                pass
        elif text.startswith("++eval_batch_size="):
            try:
                eval_bs = int(float(text.split("=", 1)[1]))
            except Exception:
                pass
    return max(256, train_bs), max(512, eval_bs)


def _default_batch_numbers(model_option: str, cfg: Dict[str, Any]) -> tuple[int, int]:
    row = {
        "model_option": model_option,
        "candidate_config": cfg,
        "hparam_id": "H2",
    }
    return _parse_batch_overrides(_orig_model_runtime_resource_overrides(row))


def _batch_profile(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    pass_name: str,
    *,
    step_name: str = "step1",
    hypothesis_family: str = "",
) -> Dict[str, Any]:
    base_train, base_eval = _default_batch_numbers(model_option, cfg)
    train_scale = {"pass1": 1.18, "pass2": 1.06, "pass3": 1.00}[pass_name]
    eval_scale = {"pass1": 1.08, "pass2": 0.98, "pass3": 0.78}[pass_name]

    hidden = int(cfg.get("hidden_size", 128))
    layers = int(cfg.get("num_layers", cfg.get("layers", 2)))
    max_len = int(cfg.get("max_len", 20))

    if hidden >= 224:
        train_scale *= 0.80
        eval_scale *= 0.76
    elif hidden >= 192:
        train_scale *= 0.88
        eval_scale *= 0.84
    elif hidden >= 160:
        train_scale *= 0.94
        eval_scale *= 0.92

    if layers >= 4:
        train_scale *= 0.88
        eval_scale *= 0.90
    elif layers >= 3:
        train_scale *= 0.94
        eval_scale *= 0.95

    if max_len >= 30:
        train_scale *= 0.78
        eval_scale *= 0.72
    elif max_len >= 24:
        train_scale *= 0.88
        eval_scale *= 0.84
    elif max_len <= 10:
        train_scale *= 1.06
        eval_scale *= 1.02

    if model_option in {"duorec", "fearec"}:
        train_scale *= 0.82
        eval_scale *= 0.80
    elif model_option in {"fame", "tisasrec"}:
        train_scale *= 0.88
        eval_scale *= 0.86
    elif model_option == "sigma":
        train_scale *= 0.84
        eval_scale *= 0.80

    if (dataset, model_option) in OOM_SENSITIVE_COMBOS:
        train_scale *= 0.80
        eval_scale *= 0.65

    if model_option == "sigma" and hidden >= 192:
        train_scale *= 0.78
        eval_scale *= 0.68

    if dataset == "retail_rocket" and model_option == "gru4rec":
        eval_scale *= 0.58
    if dataset == "retail_rocket" and model_option == "tisasrec":
        train_scale *= 0.74
        eval_scale *= 0.68

    if step_name == "step2":
        if model_option == "gru4rec" and str(hypothesis_family) in {"stable_low_lr", "mid_lr_stable"}:
            train_scale *= 1.18
            eval_scale *= 1.06
        if model_option == "tisasrec" and str(hypothesis_family) in {"time_balanced", "time_depth", "struct_balanced"}:
            train_scale *= 0.74
            eval_scale *= 0.70
        if model_option == "fame" and str(hypothesis_family) in {"moe_deeper", "facet_wide"}:
            train_scale *= 0.78
            eval_scale *= 0.74
        if model_option == "duorec":
            train_scale *= 0.88
            eval_scale *= 0.86
        if model_option == "fearec":
            train_scale *= 0.90
            eval_scale *= 0.88

    train_bs = max(256, int(base_train * train_scale))
    eval_bs = max(512, int(base_eval * eval_scale))

    if model_option == "duorec":
        train_bs = min(train_bs, 3072)
        eval_bs = min(eval_bs, 6144)
    elif model_option == "fearec":
        train_bs = min(train_bs, 4608)
        eval_bs = min(eval_bs, 8192 if pass_name != "pass3" else 6144)
    elif model_option == "sigma":
        train_bs = min(train_bs, 2560)
        eval_bs = min(eval_bs, 3072)

    def _round_down(v: int, quantum: int) -> int:
        return max(quantum, (int(v) // quantum) * quantum)

    train_bs = _round_down(train_bs, 128)
    eval_bs = _round_down(eval_bs, 256)
    label = f"{step_name}:{pass_name}:bs{train_bs}/{eval_bs}"
    return {
        "label": label,
        "train_bs": train_bs,
        "eval_bs": eval_bs,
        "base_train_bs": base_train,
        "base_eval_bs": base_eval,
    }


def _model_runtime_resource_overrides_stagei(row: Dict[str, Any]) -> List[str]:
    if str(row.get("stage", "")).lower() != "stagei":
        return _orig_model_runtime_resource_overrides(row)
    model_option = str(row.get("model_option", "")).lower()
    cfg = base._effective_model_hparams(row)
    profile = _batch_profile(
        str(row.get("dataset", "")),
        model_option,
        cfg,
        str(row.get("pass_name", "pass1")),
        step_name=_row_step_name(row),
        hypothesis_family=str(row.get("hypothesis_family", "")),
    )
    return [f"++train_batch_size={int(profile['train_bs'])}", f"++eval_batch_size={int(profile['eval_bs'])}"]


base._model_runtime_resource_overrides = _model_runtime_resource_overrides_stagei


def _model_algorithm_overrides_stagei(row: Dict[str, Any]) -> list[str]:
    model_option = str(row.get("model_option", "")).lower()
    cfg = row.get("candidate_config")
    if not isinstance(cfg, dict) or not cfg:
        return _orig_model_algorithm_overrides_stagei(row)

    overrides = list(_orig_model_algorithm_overrides_stagei(row))
    if model_option == "bsarec":
        overrides.append(f"++bsarec_alpha={float(cfg.get('bsarec_alpha', 0.5))}")
        overrides.append(f"++bsarec_c={int(cfg.get('bsarec_c', 3))}")
    if model_option == "fame":
        overrides.append(f"++fame_moe_last_k_layers={int(cfg.get('fame_moe_last_k_layers', 1))}")
    return overrides


base._model_algorithm_overrides = _model_algorithm_overrides_stagei


def _scale_center(lo: float, hi: float, *, center_mult: float = 1.0, span_mult: float = 1.0) -> tuple[float, float]:
    lo2, hi2 = sg._clamp_lr(float(lo), float(hi))
    center = math.sqrt(float(lo2) * float(hi2)) * float(center_mult)
    span = math.sqrt(float(hi2) / float(lo2)) * float(max(span_mult, 1e-3))
    return sg._clamp_lr(center / span, center * span)


def _lr_band_with_batch_scaling(
    model_option: str,
    profile: Dict[str, Any],
    pass_name: str,
    lr_lo: float,
    lr_hi: float,
) -> tuple[float, float]:
    lo, hi = sg._clamp_lr(float(lr_lo), float(lr_hi))
    base_train = max(int(profile.get("base_train_bs", 1)), 1)
    train_bs = max(int(profile.get("train_bs", 1)), 1)
    batch_center_mult = max(0.78, min(1.35, (float(train_bs) / float(base_train)) ** 0.35))
    span_mult = {"pass1": 1.10, "pass2": 0.92, "pass3": 0.72}[pass_name]
    return _scale_center(lo, hi, center_mult=batch_center_mult, span_mult=span_mult)


def _lower_lr_floor(lo: float, hi: float, *, factor: float = 0.40) -> tuple[float, float]:
    lo2, hi2 = sg._clamp_lr(float(lo), float(hi))
    return sg._clamp_lr(float(lo2) * float(factor), float(hi2))


def _style_lr_band(
    model_option: str,
    profile: Dict[str, Any],
    pass_name: str,
    style: str,
    lr_lo: float,
    lr_hi: float,
) -> tuple[float, float]:
    lo, hi = _lr_band_with_batch_scaling(model_option, profile, pass_name, lr_lo, lr_hi)
    if pass_name == "pass2":
        mapping = {
            "ANCHOR": (1.00, 0.80),
            "WIDTH_UP": (1.08, 0.88),
            "WIDTH_DN": (0.92, 0.88),
            "DEPTH_UP": (0.92, 0.84),
            "SHORT_HI": (1.18, 0.92),
            "REG_RELIEF": (1.05, 0.88),
        }
    else:
        mapping = {
            "VERIFY_BASE": (1.00, 0.72),
            "VERIFY_WIDTH": (1.03, 0.72),
            "VERIFY_REG": (0.98, 0.70),
        }
    center_mult, span_mult = mapping.get(style, (1.0, 1.0))
    return _scale_center(lo, hi, center_mult=center_mult, span_mult=span_mult)


def _load_overall_best_table(
    *,
    datasets: set[str],
    model_options: set[str],
) -> list[Dict[str, Any]]:
    best_by_combo: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(base.LOG_ROOT.glob("*/**/summary.csv")):
        if "old_" in str(path):
            continue
        axis = path.parts[-3]
        dataset = path.parts[-2]
        if dataset not in datasets:
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if "model" not in row:
                        continue
                    if str(row.get("status", "")).strip().lower() != "run_complete":
                        continue
                    raw_model = str(row.get("model", "")).strip()
                    model_option = LABEL_TO_OPTION.get(_normalize_model_name(raw_model))
                    if model_option is None or model_option not in OPTION_TO_LABEL:
                        continue
                    valid = base._metric_to_float(row.get("run_best_valid_mrr20"))
                    if valid is None:
                        continue
                    test = base._metric_to_float(row.get("run_best_test_mrr20"))
                    if test is None:
                        test = base._metric_to_float(row.get("test_mrr20"))
                    rec = {
                        "dataset": dataset,
                        "model_option": model_option,
                        "model_label": OPTION_TO_LABEL[model_option],
                        "best_valid": float(valid),
                        "best_test": None if test is None else float(test),
                        "axis": axis,
                        "run_phase": str(row.get("run_phase", "")).strip(),
                        "run_id": str(row.get("run_id", "")).strip(),
                        "hparam_id": str(row.get("hparam_id", "")).strip(),
                        "seed_id": str(row.get("seed_id", "")).strip(),
                        "timestamp_utc": str(row.get("timestamp_utc", "")).strip(),
                        "result_path": str(row.get("result_path", "")).strip(),
                        "candidate_id": str(row.get("candidate_id", "")).strip(),
                        "candidate_source": str(row.get("candidate_source", "")).strip(),
                        "source_stage": str(row.get("source_stage", "")).strip(),
                        "transfer_from_dataset": str(row.get("transfer_from_dataset", "")).strip(),
                        "step_name": str(row.get("step_name", "step1") or "step1").strip(),
                        "pass_name": str(row.get("pass_name", "")).strip(),
                        "hypothesis_family": str(row.get("hypothesis_family", "")).strip(),
                        "root_cause_tag": str(row.get("root_cause_tag", "")).strip(),
                    }
                    key = (dataset, model_option)
                    prev = best_by_combo.get(key)
                    if prev is None:
                        best_by_combo[key] = rec
                        continue
                    prev_test = -1e9 if prev.get("best_test") is None else float(prev["best_test"])
                    cur_test = -1e9 if rec.get("best_test") is None else float(rec["best_test"])
                    prev_key = (float(prev["best_valid"]), prev_test, str(prev.get("timestamp_utc", "")))
                    cur_key = (float(rec["best_valid"]), cur_test, str(rec.get("timestamp_utc", "")))
                    if cur_key > prev_key:
                        best_by_combo[key] = rec
        except Exception:
            continue

    dataset_best: Dict[str, float] = {}
    for rec in best_by_combo.values():
        dataset = str(rec["dataset"])
        dataset_best[dataset] = max(float(rec["best_valid"]), dataset_best.get(dataset, float("-inf")))

    out: list[Dict[str, Any]] = []
    for rec in best_by_combo.values():
        if str(rec["model_option"]) not in model_options:
            continue
        ds_best = dataset_best.get(str(rec["dataset"]), float(rec["best_valid"]))
        rec2 = dict(rec)
        rec2["dataset_best"] = float(ds_best)
        rec2["ratio"] = 0.0 if float(ds_best) <= 0 else float(rec["best_valid"]) / float(ds_best)
        out.append(rec2)
    return out


def _select_pass1_targets(best_rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out = [dict(r) for r in best_rows if float(r["ratio"]) <= float(PASS1_RATIO_CUT)]
    out.sort(key=lambda r: (float(r["ratio"]), str(r["dataset"]), str(r["model_option"])))
    return out


def _select_pass2_targets(best_rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out = [dict(r) for r in best_rows if float(r["ratio"]) <= float(PASS2_RATIO_CUT)]
    out.sort(key=lambda r: (float(r["ratio"]), str(r["dataset"]), str(r["model_option"])))
    return out


def _select_pass3_targets(best_rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for rec in best_rows:
        grouped.setdefault(str(rec["dataset"]), []).append(dict(rec))
    out: list[Dict[str, Any]] = []
    for dataset in sorted(grouped.keys()):
        items = sorted(grouped[dataset], key=lambda r: (float(r["best_valid"]), float(r["ratio"]), str(r["model_option"])))
        out.extend(items[: min(PASS3_BOTTOM_K, len(items))])
    return out


def _payload_pick_value(payload: Dict[str, Any], *keys: str) -> Any:
    fixed = payload.get("fixed_search") if isinstance(payload.get("fixed_search"), dict) else {}
    context_fixed = payload.get("context_fixed") if isinstance(payload.get("context_fixed"), dict) else {}
    best_params = payload.get("best_params") if isinstance(payload.get("best_params"), dict) else {}
    for key in keys:
        if key in fixed:
            return fixed.get(key)
        if key in context_fixed:
            return context_fixed.get(key)
        if key in best_params:
            return best_params.get(key)
    return None


def _payload_pick_numeric(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _payload_pick_value(payload, key)
        metric = base._metric_to_float(value)
        if metric is not None:
            return float(metric)
    return None


def _stagei_config_from_result_payload(model_option: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _normalize_stagei_cfg(model_option, sg._config_from_result_payload(model_option, payload))
    if model_option == "bsarec":
        alpha = _payload_pick_numeric(payload, "bsarec_alpha")
        if alpha is not None:
            cfg["bsarec_alpha"] = float(alpha)
        cutoff = _payload_pick_numeric(payload, "bsarec_c", "c")
        if cutoff is not None:
            cfg["bsarec_c"] = int(cutoff)
    if model_option == "fame":
        last_k = _payload_pick_numeric(payload, "fame_moe_last_k_layers")
        if last_k is not None:
            cfg["fame_moe_last_k_layers"] = int(last_k)
    if model_option == "duorec":
        contrast = _payload_pick_value(payload, "contrast")
        if contrast is not None:
            cfg["contrast"] = str(contrast)
        for key in ("tau", "lmd", "lmd_sem", "semantic_sample_max_tries"):
            value = _payload_pick_numeric(payload, key)
            if value is not None:
                cfg[key] = int(value) if key == "semantic_sample_max_tries" else float(value)
    if model_option == "fearec":
        contrast = _payload_pick_value(payload, "contrast")
        if contrast is not None:
            cfg["contrast"] = str(contrast)
        for key in ("tau", "lmd", "lmd_sem", "global_ratio", "semantic_sample_max_tries"):
            value = _payload_pick_numeric(payload, key)
            if value is not None:
                cfg[key] = int(value) if key == "semantic_sample_max_tries" else float(value)
    if model_option == "difsr":
        fusion_type = _payload_pick_value(payload, "fusion_type")
        if fusion_type is not None:
            cfg["fusion_type"] = str(fusion_type)
        predictor = _payload_pick_value(payload, "use_attribute_predictor")
        if predictor is not None:
            if isinstance(predictor, bool):
                cfg["use_attribute_predictor"] = predictor
            else:
                cfg["use_attribute_predictor"] = str(predictor).strip().lower() in {"1", "true", "yes", "on"}
        lambda_attr = _payload_pick_numeric(payload, "lambda_attr")
        if lambda_attr is not None:
            cfg["lambda_attr"] = float(lambda_attr)
    return _normalize_stagei_cfg(model_option, cfg)


def _candidate_from_row_stagei(
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
    payload = sg._safe_json(str(row.get("result_path", "")))
    if not isinstance(payload, dict):
        return None

    cfg = _stagei_config_from_result_payload(model_option, payload)
    best_lr = _payload_pick_numeric(payload, "learning_rate")
    lo = base._metric_to_float(row.get("lr_lo"))
    hi = base._metric_to_float(row.get("lr_hi"))

    if best_lr is None and lo is not None and hi is not None and hi > lo:
        best_lr = math.sqrt(float(lo) * float(hi))
    if best_lr is None:
        lo2, hi2 = base._compute_lr_space(str(row.get("dataset", "")), model_option, "H2")
        best_lr = math.sqrt(float(lo2) * float(hi2))

    if force_lr_narrow_around_best:
        lo, hi = sg._clamp_lr(float(best_lr) / 1.6, float(best_lr) * 1.6)
    else:
        if lo is None or hi is None or hi <= lo:
            lo, hi = sg._clamp_lr(float(best_lr) / 2.2, float(best_lr) * 2.2)
        else:
            lo, hi = sg._clamp_lr(float(lo), float(hi))
        if lr_policy_mult is not None:
            lo, hi = sg._scale_lr_band(float(lo), float(hi), float(lr_policy_mult))

    valid = base._metric_to_float(row.get("run_best_valid_mrr20"))
    test = base._metric_to_float(row.get("run_best_test_mrr20"))
    n_completed = base._metric_to_float(row.get("n_completed"))
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
        "score": sg._candidate_score(valid, test, completion, source),
    }


def _anchor_candidate(
    dataset: str,
    model_option: str,
    best_rec: Optional[Dict[str, Any]],
    *,
    step_name: str = "step1",
) -> tuple[Dict[str, Any], str, str]:
    if isinstance(best_rec, dict):
        cand = _candidate_from_row_stagei(
            model_option=model_option,
            row=best_rec,
            source="stage_summary",
            source_stage=str(best_rec.get("axis", "")),
            transfer_from_dataset="",
            candidate_id="ANCHOR",
            force_lr_narrow_around_best=True,
        )
        if cand:
            return cand, str(best_rec.get("run_id", "")).strip(), str(best_rec.get("axis", "")).strip()

    cfg = _prepare_cfg_for_pass(
        dataset,
        model_option,
        sg._default_cfg_for_model(model_option),
        "pass2",
        step_name=step_name,
    )
    lo, hi = base._compute_lr_space(dataset, model_option, "H2")
    return {
        "candidate_id": "ANCHOR",
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "global_fallback",
        "source_stage": "fallback",
        "transfer_from_dataset": "",
    }, "", "fallback"


def _pass1_hparam_candidate(dataset: str, model_option: str, hid: str) -> Dict[str, Any]:
    cfg = _prepare_cfg_for_pass(
        dataset,
        model_option,
        dict(base._effective_model_hparams({"model_option": model_option, "hparam_id": hid})),
        "pass1",
    )
    lo, hi = base._compute_lr_space(dataset, model_option, hid)
    profile = _batch_profile(dataset, model_option, cfg, "pass1")
    lo, hi = _lr_band_with_batch_scaling(model_option, profile, "pass1", lo, hi)
    lo, hi = _lower_lr_floor(lo, hi, factor=0.40)
    model_tag = base._sanitize_token(OPTION_TO_LABEL[model_option], upper=True)
    return {
        "candidate_id": f"I1_{model_tag}_{hid}",
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "manual_rebuild",
        "source_stage": "general_hparam_bank",
        "transfer_from_dataset": "",
        "anchor_run_id": "",
        "batch_profile": str(profile["label"]),
    }


def _apply_style(dataset: str, model_option: str, anchor_cfg: Dict[str, Any], style: str, pass_name: str) -> Dict[str, Any]:
    cfg = dict(anchor_cfg)
    if style in {"ANCHOR", "VERIFY_BASE"}:
        pass
    elif style == "WIDTH_UP":
        cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * 1.20)
        cfg["embedding_size"] = int(cfg["hidden_size"])
        cfg["inner_size"] = int(cfg.get("inner_size", 256) * 1.28)
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.80
        if model_option == "fame":
            cfg["num_experts"] = int(cfg.get("num_experts", 3)) + 1
        if model_option == "sigma":
            cfg["sigma_state"] = int(cfg.get("sigma_state", 16) * 1.25)
    elif style == "WIDTH_DN":
        cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * 0.85)
        cfg["embedding_size"] = int(cfg["hidden_size"])
        cfg["inner_size"] = int(cfg.get("inner_size", 256) * 0.85)
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) + 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 1.20
        if model_option == "sigma":
            cfg["sigma_state"] = max(8, int(cfg.get("sigma_state", 16) * 0.85))
    elif style == "DEPTH_UP":
        depth = int(cfg.get("num_layers", cfg.get("layers", 2))) + 1
        cfg["num_layers"] = depth
        cfg["layers"] = depth
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) + 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 1.10
        if model_option == "sigma":
            cfg["sigma_kernel"] = min(12, int(cfg.get("sigma_kernel", 6)) + 2)
    elif style == "SHORT_HI":
        cfg["max_len"] = max(5, int(round(int(cfg.get("max_len", 20)) * 0.65)))
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.85
        if model_option == "tisasrec":
            cfg["time_span"] = max(64, int(round(int(cfg.get("time_span", 256)) * 0.70)))
    elif style == "REG_RELIEF":
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.03
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.65
        if model_option in {"duorec", "fearec"}:
            cfg["lmd"] = float(cfg.get("lmd", 0.04)) * 0.75
            cfg["tau"] = float(cfg.get("tau", 0.2)) * 0.90
        if model_option == "fearec":
            cfg["global_ratio"] = max(0.35, float(cfg.get("global_ratio", 0.85)) - 0.10)
        if model_option == "sigma":
            cfg["sigma_remaining_ratio"] = max(0.25, float(cfg.get("sigma_remaining_ratio", 0.6)) - 0.10)
    elif style == "VERIFY_WIDTH":
        cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * 1.08)
        cfg["embedding_size"] = int(cfg["hidden_size"])
        cfg["inner_size"] = int(cfg.get("inner_size", 256) * 1.12)
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.01
    elif style == "VERIFY_REG":
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.80
        if model_option in {"duorec", "fearec"}:
            cfg["lmd"] = float(cfg.get("lmd", 0.04)) * 0.85
        if model_option == "sigma":
            cfg["sigma_remaining_ratio"] = max(0.25, float(cfg.get("sigma_remaining_ratio", 0.6)) - 0.05)
    return _prepare_cfg_for_pass(dataset, model_option, cfg, pass_name)


def _make_history_candidate(
    dataset: str,
    model_option: str,
    style: str,
    *,
    pass_name: str,
    anchor_cfg: Dict[str, Any],
    anchor_lo: float,
    anchor_hi: float,
    anchor_run_id: str,
    source_stage: str,
) -> Dict[str, Any]:
    cfg = _apply_style(dataset, model_option, anchor_cfg, style, pass_name)
    profile = _batch_profile(dataset, model_option, cfg, pass_name)
    lo, hi = _style_lr_band(model_option, profile, pass_name, style, anchor_lo, anchor_hi)
    token = PASS_TOKEN[pass_name]
    model_tag = base._sanitize_token(OPTION_TO_LABEL[model_option], upper=True)
    return {
        "candidate_id": f"{token}_{model_tag}_{style}",
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "history_perturb" if pass_name == "pass2" else "verify_local",
        "source_stage": source_stage,
        "transfer_from_dataset": "",
        "anchor_run_id": anchor_run_id,
        "batch_profile": str(profile["label"]),
    }


def _expand_candidate_bank(
    dataset: str,
    model_option: str,
    pass_name: str,
    candidates: list[Dict[str, Any]],
    *,
    target_count: int,
    anchor_cfg: Optional[Dict[str, Any]] = None,
    anchor_lo: Optional[float] = None,
    anchor_hi: Optional[float] = None,
    anchor_run_id: str = "",
    source_stage: str = "fallback",
    source: str = "history_perturb",
) -> list[Dict[str, Any]]:
    out = _dedup_candidates_stagei(list(candidates))
    if len(out) >= int(target_count):
        return out[: int(target_count)]

    base_cfg = dict(anchor_cfg or (out[0]["config"] if out else sg._default_cfg_for_model(model_option)))
    lo = float(anchor_lo) if anchor_lo is not None else float(base._compute_lr_space(dataset, model_option, "H2")[0])
    hi = float(anchor_hi) if anchor_hi is not None else float(base._compute_lr_space(dataset, model_option, "H2")[1])
    model_tag = base._sanitize_token(OPTION_TO_LABEL[model_option], upper=True)
    token = PASS_TOKEN[pass_name]

    attempts = 0
    max_attempts = max(int(target_count) * 10, 24)
    while len(out) < int(target_count) and attempts < max_attempts:
        attempts += 1
        variant = (attempts - 1) % 6
        cycle = (attempts - 1) // 6
        cfg = dict(base_cfg)

        if variant == 0:
            cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * (1.10 + 0.07 * cycle))
            cfg["embedding_size"] = int(cfg["hidden_size"])
            cfg["inner_size"] = int(cfg.get("inner_size", 256) * (1.12 + 0.08 * cycle))
            center_mult, span_mult = 1.06 + 0.03 * cycle, 0.94 - 0.03 * min(cycle, 3)
        elif variant == 1:
            cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * max(0.66, 0.90 - 0.06 * cycle))
            cfg["embedding_size"] = int(cfg["hidden_size"])
            cfg["inner_size"] = int(cfg.get("inner_size", 256) * max(0.68, 0.92 - 0.06 * cycle))
            center_mult, span_mult = 0.94 - 0.03 * min(cycle, 4), 0.94 - 0.02 * min(cycle, 3)
        elif variant == 2:
            depth = int(cfg.get("num_layers", cfg.get("layers", 2))) + 1 + cycle
            cfg["num_layers"] = depth
            cfg["layers"] = depth
            center_mult, span_mult = max(0.78, 0.96 - 0.04 * cycle), max(0.68, 0.88 - 0.04 * cycle)
        elif variant == 3:
            cfg["max_len"] = max(5, int(round(int(cfg.get("max_len", 20)) * max(0.42, 0.80 - 0.08 * cycle))))
            center_mult, span_mult = 1.10 + 0.04 * cycle, max(0.72, 0.92 - 0.04 * cycle)
        elif variant == 4:
            cfg["dropout"] = float(cfg.get("dropout", 0.1)) - (0.02 + 0.01 * cycle)
            cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * max(0.40, 0.80 - 0.10 * cycle)
            center_mult, span_mult = 1.02 + 0.03 * cycle, max(0.64, 0.86 - 0.05 * cycle)
        else:
            cfg["dropout"] = float(cfg.get("dropout", 0.1)) + (0.02 + 0.01 * cycle)
            cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * (1.20 + 0.16 * cycle)
            center_mult, span_mult = max(0.62, 0.90 - 0.05 * cycle), max(0.60, 0.86 - 0.05 * cycle)

        cfg = _prepare_cfg_for_pass(dataset, model_option, cfg, pass_name)
        profile = _batch_profile(dataset, model_option, cfg, pass_name)
        lr_lo, lr_hi = _lr_band_with_batch_scaling(model_option, profile, pass_name, lo, hi)
        lr_lo, lr_hi = _scale_center(lr_lo, lr_hi, center_mult=center_mult, span_mult=span_mult)
        if pass_name == "pass1":
            lr_lo, lr_hi = _lower_lr_floor(lr_lo, lr_hi, factor=0.40)
        cand = {
            "candidate_id": f"{token}_{model_tag}_FB{attempts:02d}",
            "config": cfg,
            "lr_lo": float(lr_lo),
            "lr_hi": float(lr_hi),
            "source": source,
            "source_stage": source_stage,
            "transfer_from_dataset": "",
            "anchor_run_id": anchor_run_id,
            "batch_profile": str(profile["label"]),
        }
        out = _dedup_candidates_stagei(out + [cand])

    if len(out) < int(target_count):
        raise RuntimeError(
            f"failed to build enough StageI candidates: pass={pass_name} dataset={dataset} "
            f"model={model_option} have={len(out)} need={int(target_count)}"
        )
    return out[: int(target_count)]


def _build_pass1_candidates(dataset: str, model_option: str) -> list[Dict[str, Any]]:
    candidates = [_pass1_hparam_candidate(dataset, model_option, hid) for hid in PASS1_HPARAM_BANK]
    return _expand_candidate_bank(
        dataset,
        model_option,
        "pass1",
        candidates,
        target_count=int(PASS_SPECS["pass1"]["candidate_count"]),
        source_stage="general_hparam_bank",
        source="manual_rebuild",
    )


def _build_pass2_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    anchor, anchor_run_id, source_stage = _anchor_candidate(dataset, model_option, best_rec)
    styles = ["ANCHOR", "WIDTH_UP", "WIDTH_DN", "DEPTH_UP", "SHORT_HI", "REG_RELIEF"]
    candidates = [
        _make_history_candidate(
            dataset,
            model_option,
            style,
            pass_name="pass2",
            anchor_cfg=dict(anchor["config"]),
            anchor_lo=float(anchor["lr_lo"]),
            anchor_hi=float(anchor["lr_hi"]),
            anchor_run_id=anchor_run_id,
            source_stage=source_stage,
        )
        for style in styles
    ]
    return _expand_candidate_bank(
        dataset,
        model_option,
        "pass2",
        candidates,
        target_count=int(PASS_SPECS["pass2"]["candidate_count"]),
        anchor_cfg=dict(anchor["config"]),
        anchor_lo=float(anchor["lr_lo"]),
        anchor_hi=float(anchor["lr_hi"]),
        anchor_run_id=anchor_run_id,
        source_stage=source_stage,
        source="history_perturb",
    )


def _build_pass3_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    anchor, anchor_run_id, source_stage = _anchor_candidate(dataset, model_option, best_rec)
    styles = ["VERIFY_BASE", "VERIFY_WIDTH", "VERIFY_REG"]
    candidates = [
        _make_history_candidate(
            dataset,
            model_option,
            style,
            pass_name="pass3",
            anchor_cfg=dict(anchor["config"]),
            anchor_lo=float(anchor["lr_lo"]),
            anchor_hi=float(anchor["lr_hi"]),
            anchor_run_id=anchor_run_id,
            source_stage=source_stage,
        )
        for style in styles
    ]
    return _expand_candidate_bank(
        dataset,
        model_option,
        "pass3",
        candidates,
        target_count=int(PASS_SPECS["pass3"]["candidate_count"]),
        anchor_cfg=dict(anchor["config"]),
        anchor_lo=float(anchor["lr_lo"]),
        anchor_hi=float(anchor["lr_hi"]),
        anchor_run_id=anchor_run_id,
        source_stage=source_stage,
        source="verify_local",
    )


def _step2_family_spec(
    name: str,
    cfg: Dict[str, Any],
    *,
    lr_policy: str,
    root_cause_tag: str,
    is_control: bool = False,
    search_overrides: Optional[Dict[str, Any]] = None,
    search_space_type_overrides: Optional[Dict[str, Any]] = None,
    lr_lo_override: Optional[float] = None,
    lr_hi_override: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "name": str(name),
        "config": dict(cfg),
        "lr_policy": str(lr_policy),
        "root_cause_tag": str(root_cause_tag),
        "is_control": bool(is_control),
        "search_overrides": dict(search_overrides or {}),
        "search_space_type_overrides": dict(search_space_type_overrides or {}),
        "lr_lo_override": None if lr_lo_override is None else float(lr_lo_override),
        "lr_hi_override": None if lr_hi_override is None else float(lr_hi_override),
    }


def _family_batch_policy(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    pass_name: str,
    *,
    step_name: str = "step2",
    hypothesis_family: str,
) -> Dict[str, Any]:
    return _batch_profile(
        dataset,
        model_option,
        cfg,
        pass_name,
        step_name=step_name,
        hypothesis_family=hypothesis_family,
    )


def _family_lr_band(
    dataset: str,
    model_option: str,
    pass_name: str,
    profile: Dict[str, Any],
    *,
    lr_lo: float,
    lr_hi: float,
    lr_policy: str,
    variant: str,
) -> tuple[float, float]:
    raw_lo, raw_hi = sg._clamp_lr(float(lr_lo), float(lr_hi))
    scaled_lo, scaled_hi = _lr_band_with_batch_scaling(model_option, profile, pass_name, raw_lo, raw_hi)
    if raw_lo > 0 and raw_hi > raw_lo:
        # Preserve family-specific absolute priors even when batch scaling nudges the
        # center. This keeps low-stable step3 families genuinely low-LR.
        base_lo = min(float(raw_lo), float(scaled_lo))
        base_hi = min(float(raw_hi), float(scaled_hi))
        if base_hi <= base_lo:
            base_lo, base_hi = raw_lo, raw_hi
    else:
        base_lo, base_hi = scaled_lo, scaled_hi
    policy_map = {
        "low_stable": (0.62, 0.92),
        "mid_balanced": (0.88, 0.86),
        "high_control": (1.12, 0.88),
    }
    center_mult, span_mult = policy_map.get(str(lr_policy), (0.88, 0.86))
    variant_map = {
        "BASE": (1.00, 1.00),
        "WIDTH": (1.02, 0.95),
        "DEPTH": (0.92, 0.92),
        "LEN": (0.90, 0.95),
        "REG": (0.82, 0.92),
        "CTRL": (1.18, 0.92),
        "LR_DN": (0.80, 0.82),
        "LR_UP": (1.08, 0.82),
        "WIDTH_TUNE": (1.01, 0.84),
        "DEPTH_TUNE": (0.95, 0.84),
        "LEN_TUNE": (0.92, 0.86),
        "REG_TUNE": (0.84, 0.84),
        "ANCHOR": (1.00, 0.80),
    }
    vm_center, vm_span = variant_map.get(str(variant), (1.0, 1.0))
    lo2, hi2 = _scale_center(
        base_lo,
        base_hi,
        center_mult=center_mult * vm_center,
        span_mult=span_mult * vm_span,
    )
    # Keep low-stable families from being pushed upward by batch scaling.
    if str(lr_policy) == "low_stable":
        lo2 = min(float(lo2), float(raw_lo))
        hi2 = min(float(hi2), float(raw_hi))
    lo2 = max(float(raw_lo), float(lo2))
    hi2 = min(float(raw_hi), float(hi2))
    return sg._clamp_lr(lo2, hi2)


def _step2_prepare_family_cfg(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    *,
    family_name: str,
    pass_name: str = "pass1",
) -> Dict[str, Any]:
    return _prepare_cfg_for_pass(
        dataset,
        model_option,
        cfg,
        pass_name,
        step_name="step2",
        hypothesis_family=family_name,
    )


def _select_step2_hypothesis_bank(
    dataset: str,
    model_option: str,
    best_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    anchor, anchor_run_id, source_stage = _anchor_candidate(dataset, model_option, best_rec, step_name="step2")
    anchor_cfg = _normalize_stagei_cfg(model_option, dict(anchor["config"]))
    anchor_lo = float(anchor["lr_lo"])
    anchor_hi = float(anchor["lr_hi"])
    families: list[Dict[str, Any]] = []

    def fam(name: str, *, lr_policy: str, root: str, is_control: bool = False, **cfg_updates: Any) -> None:
        cfg = dict(anchor_cfg)
        cfg.update(cfg_updates)
        families.append(
            _step2_family_spec(
                name,
                _step2_prepare_family_cfg(dataset, model_option, cfg, family_name=name, pass_name="pass1"),
                lr_policy=lr_policy,
                root_cause_tag=root,
                is_control=is_control,
            )
        )

    combo = (dataset, model_option)
    if combo == ("amazon_beauty", "bsarec"):
        fam("freq_lowmix", lr_policy="low_stable", root="bsarec_missing_alpha", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=16, bsarec_alpha=0.20, dropout=0.12, weight_decay=1.6e-4)
        fam("freq_midmix", lr_policy="mid_balanced", root="bsarec_missing_alpha", hidden_size=192, embedding_size=192, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=384, max_len=14, bsarec_alpha=0.50, dropout=0.10, weight_decay=1.2e-4)
        fam("freq_highmix", lr_policy="high_control", root="bsarec_missing_alpha", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=10, bsarec_alpha=0.75, dropout=0.11, weight_decay=1.4e-4, is_control=True)
        fam("long_sparse", lr_policy="low_stable", root="bsarec_short_len_bias", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=20, bsarec_alpha=0.35, dropout=0.12, weight_decay=1.6e-4)
    elif combo == ("amazon_beauty", "fame"):
        fam("moe_shallow_stable", lr_policy="low_stable", root="fame_degenerate_short_high_lr", hidden_size=128, embedding_size=128, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=256, num_experts=2, fame_moe_last_k_layers=1, max_len=14, dropout=0.12, weight_decay=1.2e-4)
        fam("moe_deeper", lr_policy="low_stable", root="fame_degenerate_short_high_lr", hidden_size=144, embedding_size=144, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=288, num_experts=4, fame_moe_last_k_layers=2, max_len=14, dropout=0.10, weight_decay=1.1e-4)
        fam("expert_sparse", lr_policy="mid_balanced", root="fame_expert_head_mismatch", hidden_size=128, embedding_size=128, num_layers=2, layers=2, num_heads=2, heads=2, inner_size=256, num_experts=2, fame_moe_last_k_layers=1, max_len=12, dropout=0.12, weight_decay=1.3e-4)
        fam("facet_wide", lr_policy="mid_balanced", root="fame_expert_head_mismatch", hidden_size=192, embedding_size=192, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=384, num_experts=4, fame_moe_last_k_layers=2, max_len=16, dropout=0.10, weight_decay=1.0e-4)
    elif combo == ("amazon_beauty", "gru4rec"):
        fam("stable_low_lr", lr_policy="low_stable", root="gru_short_high_lr_bias", hidden_size=128, embedding_size=128, num_layers=2, layers=2, inner_size=256, max_len=14, dropout=0.15, weight_decay=8e-5)
        fam("mid_len_mid_lr", lr_policy="mid_balanced", root="gru_short_high_lr_bias", hidden_size=160, embedding_size=160, num_layers=2, layers=2, inner_size=320, max_len=16, dropout=0.12, weight_decay=7e-5)
        fam("deep_small", lr_policy="mid_balanced", root="gru_short_high_lr_bias", hidden_size=112, embedding_size=112, num_layers=3, layers=3, inner_size=224, max_len=12, dropout=0.16, weight_decay=1e-4)
        fam("high_lr_control", lr_policy="high_control", root="gru_short_high_lr_bias", hidden_size=192, embedding_size=192, num_layers=2, layers=2, inner_size=384, max_len=10, dropout=0.08, weight_decay=5e-5, is_control=True)
    elif combo == ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"):
        fam("high_lr_control", lr_policy="high_control", root="gru_single_family_lockin", hidden_size=320, embedding_size=320, num_layers=3, layers=3, inner_size=640, max_len=10, dropout=0.06, weight_decay=2.5e-5, is_control=True)
        fam("mid_lr_stable", lr_policy="mid_balanced", root="gru_single_family_lockin", hidden_size=224, embedding_size=224, num_layers=2, layers=2, inner_size=448, max_len=12, dropout=0.10, weight_decay=4.0e-5)
        fam("longer_context", lr_policy="low_stable", root="gru_single_family_lockin", hidden_size=192, embedding_size=192, num_layers=2, layers=2, inner_size=384, max_len=18, dropout=0.10, weight_decay=4.0e-5)
        fam("small_width_fast", lr_policy="mid_balanced", root="gru_single_family_lockin", hidden_size=128, embedding_size=128, num_layers=2, layers=2, inner_size=256, max_len=10, dropout=0.12, weight_decay=7.0e-5)
    elif combo == ("retail_rocket", "tisasrec"):
        fam("time_short_dense", lr_policy="mid_balanced", root="tisas_struct_time_coupling", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=12, time_span=192, dropout=0.10, weight_decay=1.2e-4)
        fam("time_balanced", lr_policy="low_stable", root="tisas_struct_time_coupling", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=18, time_span=432, dropout=0.10, weight_decay=1.1e-4)
        fam("time_depth", lr_policy="mid_balanced", root="tisas_struct_time_coupling", hidden_size=128, embedding_size=128, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=256, max_len=16, time_span=384, dropout=0.12, weight_decay=1.3e-4)
    elif combo == ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"):
        fam("struct_small", lr_policy="low_stable", root="tisas_structure_lockin", hidden_size=128, embedding_size=128, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=256, max_len=16, time_span=256, dropout=0.10, weight_decay=1.3e-4)
        fam("struct_balanced", lr_policy="mid_balanced", root="tisas_structure_lockin", hidden_size=160, embedding_size=160, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=320, max_len=20, time_span=384, dropout=0.10, weight_decay=1.2e-4)
        fam("time_compressed", lr_policy="high_control", root="tisas_structure_lockin", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=12, time_span=192, dropout=0.10, weight_decay=1.1e-4, is_control=True)
    elif combo == ("retail_rocket", "gru4rec"):
        fam("mid_lr_stable", lr_policy="mid_balanced", root="gru_memory_and_instability", hidden_size=192, embedding_size=192, num_layers=2, layers=2, inner_size=384, max_len=12, dropout=0.10, weight_decay=5.0e-5)
        fam("small_depth", lr_policy="low_stable", root="gru_memory_and_instability", hidden_size=128, embedding_size=128, num_layers=1, layers=1, inner_size=256, max_len=14, dropout=0.14, weight_decay=9.0e-5)
        fam("short_control", lr_policy="high_control", root="gru_memory_and_instability", hidden_size=224, embedding_size=224, num_layers=2, layers=2, inner_size=448, max_len=10, dropout=0.08, weight_decay=4.0e-5, is_control=True)
    elif combo == ("foursquare", "gru4rec"):
        fam("stable_low_lr", lr_policy="low_stable", root="gru_bimodal_instability", hidden_size=128, embedding_size=128, num_layers=2, layers=2, inner_size=256, max_len=12, dropout=0.14, weight_decay=8.0e-5)
        fam("short_sparse", lr_policy="high_control", root="gru_bimodal_instability", hidden_size=160, embedding_size=160, num_layers=2, layers=2, inner_size=320, max_len=8, dropout=0.08, weight_decay=4.0e-5, is_control=True)
        fam("mid_len_balanced", lr_policy="mid_balanced", root="gru_bimodal_instability", hidden_size=144, embedding_size=144, num_layers=2, layers=2, inner_size=288, max_len=14, dropout=0.12, weight_decay=6.0e-5)
    elif combo == ("retail_rocket", "sasrec"):
        fam("reg_relief_low_lr", lr_policy="low_stable", root="sasrec_reg_lr_mismatch", hidden_size=160, embedding_size=160, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=320, max_len=14, dropout=0.08, weight_decay=7.5e-5)
        fam("depth_balanced", lr_policy="mid_balanced", root="sasrec_reg_lr_mismatch", hidden_size=160, embedding_size=160, num_layers=4, layers=4, num_heads=4, heads=4, inner_size=320, max_len=12, dropout=0.10, weight_decay=1.0e-4)
        fam("mid_len_lowdrop", lr_policy="mid_balanced", root="sasrec_reg_lr_mismatch", hidden_size=144, embedding_size=144, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=288, max_len=18, dropout=0.08, weight_decay=8.0e-5)
    elif combo == ("retail_rocket", "duorec"):
        fam("ce_dominant", lr_policy="low_stable", root="duorec_overstrong_ssl", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=12, contrast="un", tau=0.25, lmd=0.01, lmd_sem=0.0, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
        fam("weak_ssl", lr_policy="mid_balanced", root="duorec_overstrong_ssl", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=12, contrast="un", tau=0.35, lmd=0.02, lmd_sem=0.01, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
        fam("semantic_light", lr_policy="mid_balanced", root="duorec_overstrong_ssl", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=12, contrast="su", tau=0.40, lmd=0.0, lmd_sem=0.02, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
    elif combo == ("retail_rocket", "fearec"):
        fam("freq_balanced", lr_policy="low_stable", root="fearec_freq_contrast_mismatch", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=12, contrast="un", tau=0.18, lmd=0.02, lmd_sem=0.0, global_ratio=0.70, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
        fam("wide_freq", lr_policy="mid_balanced", root="fearec_freq_contrast_mismatch", hidden_size=176, embedding_size=176, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=352, max_len=12, contrast="un", tau=0.20, lmd=0.03, lmd_sem=0.0, global_ratio=0.85, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
        fam("contrast_light", lr_policy="mid_balanced", root="fearec_freq_contrast_mismatch", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=14, contrast="un", tau=0.25, lmd=0.01, lmd_sem=0.0, global_ratio=0.80, semantic_sample_max_tries=2, dropout=0.10, weight_decay=1.0e-4)
    elif combo == ("foursquare", "sigma"):
        fam("sigma_state_mid", lr_policy="mid_balanced", root="sigma_state_kernel_ratio_mismatch", hidden_size=128, embedding_size=128, num_layers=2, layers=2, inner_size=256, sigma_state=16, sigma_kernel=6, sigma_remaining_ratio=0.45, max_len=12, dropout=0.10, weight_decay=1.1e-4)
        fam("sigma_wide_kernel", lr_policy="low_stable", root="sigma_state_kernel_ratio_mismatch", hidden_size=144, embedding_size=144, num_layers=2, layers=2, inner_size=288, sigma_state=24, sigma_kernel=8, sigma_remaining_ratio=0.55, max_len=14, dropout=0.10, weight_decay=1.0e-4)
        fam("sigma_compact", lr_policy="high_control", root="sigma_state_kernel_ratio_mismatch", hidden_size=128, embedding_size=128, num_layers=1, layers=1, inner_size=256, sigma_state=12, sigma_kernel=4, sigma_remaining_ratio=0.35, max_len=10, dropout=0.12, weight_decay=1.2e-4, is_control=True)
    else:
        if model_option == "gru4rec":
            fam("stable_low_lr", lr_policy="low_stable", root="gru_generic_root", hidden_size=128, embedding_size=128, num_layers=2, layers=2, inner_size=256, max_len=12, dropout=0.14, weight_decay=8e-5)
            fam("mid_balanced", lr_policy="mid_balanced", root="gru_generic_root", hidden_size=160, embedding_size=160, num_layers=2, layers=2, inner_size=320, max_len=14, dropout=0.10, weight_decay=6e-5)
            fam("high_lr_control", lr_policy="high_control", root="gru_generic_root", hidden_size=192, embedding_size=192, num_layers=2, layers=2, inner_size=384, max_len=10, dropout=0.08, weight_decay=5e-5, is_control=True)
        elif model_option == "tisasrec":
            fam("struct_balanced", lr_policy="mid_balanced", root="tisas_generic_root", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=320, max_len=16, time_span=320, dropout=0.10, weight_decay=1.2e-4)
            fam("time_short_dense", lr_policy="mid_balanced", root="tisas_generic_root", hidden_size=144, embedding_size=144, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=288, max_len=12, time_span=192, dropout=0.10, weight_decay=1.3e-4)
            fam("time_depth", lr_policy="low_stable", root="tisas_generic_root", hidden_size=128, embedding_size=128, num_layers=3, layers=3, num_heads=4, heads=4, inner_size=256, max_len=14, time_span=256, dropout=0.12, weight_decay=1.4e-4)
        else:
            fam("anchor_balanced", lr_policy="mid_balanced", root=f"{model_option}_generic_root")
            fam("regularized", lr_policy="low_stable", root=f"{model_option}_generic_root", dropout=float(anchor_cfg.get("dropout", 0.1)) - 0.02, weight_decay=float(anchor_cfg.get("weight_decay", 1e-4)) * 0.75)
            fam("control", lr_policy="high_control", root=f"{model_option}_generic_root", is_control=True, max_len=max(5, int(anchor_cfg.get("max_len", 12)) - 2))

    return {
        "anchor": anchor,
        "anchor_run_id": anchor_run_id,
        "source_stage": source_stage,
        "families": families,
        "anchor_cfg": anchor_cfg,
        "anchor_lo": anchor_lo,
        "anchor_hi": anchor_hi,
    }


def _apply_step2_variant(
    dataset: str,
    model_option: str,
    base_cfg: Dict[str, Any],
    *,
    family_name: str,
    variant: str,
    pass_name: str,
) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    if variant in {"WIDTH", "WIDTH_TUNE"}:
        cfg["hidden_size"] = int(cfg.get("hidden_size", 128) * 1.10)
        cfg["embedding_size"] = int(cfg["hidden_size"])
        cfg["inner_size"] = int(cfg.get("inner_size", 256) * 1.15)
    elif variant == "DEPTH":
        depth = int(cfg.get("num_layers", cfg.get("layers", 2))) + 1
        cfg["num_layers"] = depth
        cfg["layers"] = depth
    elif variant == "DEPTH_TUNE":
        depth = int(cfg.get("num_layers", cfg.get("layers", 2)))
        cfg["num_layers"] = max(1, depth + 1)
        cfg["layers"] = int(cfg["num_layers"])
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) + 0.01
    elif variant in {"LEN", "LEN_TUNE"}:
        cfg["max_len"] = int(round(int(cfg.get("max_len", 12)) * (1.20 if variant == "LEN" else 1.10)))
        if model_option == "tisasrec":
            cfg["time_span"] = int(round(int(cfg.get("time_span", 256)) * 1.20))
    elif variant == "REG":
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.02
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.70
        if model_option in {"duorec", "fearec"}:
            cfg["lmd"] = float(cfg.get("lmd", 0.04)) * 0.80
            cfg["lmd_sem"] = float(cfg.get("lmd_sem", 0.0)) * 0.80
    elif variant == "REG_TUNE":
        cfg["dropout"] = float(cfg.get("dropout", 0.1)) - 0.015
        cfg["weight_decay"] = float(cfg.get("weight_decay", 1e-4)) * 0.75
    return _step2_prepare_family_cfg(dataset, model_option, cfg, family_name=family_name, pass_name=pass_name)


def _make_step_candidate(
    step_name: str,
    dataset: str,
    model_option: str,
    pass_name: str,
    family: Dict[str, Any],
    *,
    variant: str,
    anchor_lo: float,
    anchor_hi: float,
    anchor_run_id: str,
    source_stage: str,
) -> Dict[str, Any]:
    if str(step_name) == "step2":
        cfg = _apply_step2_variant(
            dataset,
            model_option,
            dict(family["config"]),
            family_name=str(family["name"]),
            variant=variant,
            pass_name=pass_name,
        )
    else:
        cfg = _prepare_cfg_for_pass(
            dataset,
            model_option,
            dict(family["config"]),
            pass_name,
            step_name=str(step_name),
            hypothesis_family=str(family["name"]),
        )
    profile = _family_batch_policy(
        dataset,
        model_option,
        cfg,
        pass_name,
        step_name=str(step_name),
        hypothesis_family=str(family["name"]),
    )
    fam_lo = float(family.get("lr_lo_override")) if family.get("lr_lo_override") is not None else float(anchor_lo)
    fam_hi = float(family.get("lr_hi_override")) if family.get("lr_hi_override") is not None else float(anchor_hi)
    lo, hi = _family_lr_band(
        dataset,
        model_option,
        pass_name,
        profile,
        lr_lo=fam_lo,
        lr_hi=fam_hi,
        lr_policy=str(family["lr_policy"]),
        variant=variant,
    )
    model_tag = base._sanitize_token(OPTION_TO_LABEL[model_option], upper=True)
    token = _step_pass_token(str(step_name), pass_name)
    fam_tag = base._sanitize_token(str(family["name"]), upper=True)
    suffix = base._sanitize_token(str(variant), upper=True)
    return {
        "candidate_id": f"{token}_{model_tag}_{fam_tag}_{suffix}",
        "config": cfg,
        "lr_lo": float(lo),
        "lr_hi": float(hi),
        "source": "root_cause_hypothesis" if pass_name == "pass1" else "family_refine",
        "source_stage": source_stage,
        "transfer_from_dataset": "",
        "anchor_run_id": anchor_run_id,
        "batch_profile": str(profile["label"]),
        "hypothesis_family": str(family["name"]),
        "root_cause_tag": str(family["root_cause_tag"]),
        "search_overrides": dict(family.get("search_overrides", {}) or {}),
        "search_space_type_overrides": dict(family.get("search_space_type_overrides", {}) or {}),
    }


def _make_step2_candidate(
    dataset: str,
    model_option: str,
    pass_name: str,
    family: Dict[str, Any],
    *,
    variant: str,
    anchor_lo: float,
    anchor_hi: float,
    anchor_run_id: str,
    source_stage: str,
) -> Dict[str, Any]:
    return _make_step_candidate(
        "step2",
        dataset,
        model_option,
        pass_name,
        family,
        variant=variant,
        anchor_lo=anchor_lo,
        anchor_hi=anchor_hi,
        anchor_run_id=anchor_run_id,
        source_stage=source_stage,
    )


def _allocate_even_counts(total: int, families: list[Dict[str, Any]]) -> list[int]:
    if not families:
        return []
    base_count = int(total) // len(families)
    remainder = int(total) % len(families)
    counts = [base_count] * len(families)
    for i in range(remainder):
        counts[i] += 1
    return counts


def _load_step_family_stats(step_name: str, dataset: str, model_label: str) -> Dict[str, Dict[str, Any]]:
    path = base._summary_path(dataset)
    stats: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return stats
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if str(row.get("model", "")).strip() != str(model_label):
                    continue
                if str(row.get("step_name", "step1") or "step1").strip().lower() != str(step_name):
                    continue
                if str(row.get("pass_name", "")).strip() != "pass1":
                    continue
                family = str(row.get("hypothesis_family", "")).strip()
                if not family:
                    continue
                bucket = stats.setdefault(
                    family,
                    {"vals": [], "fails": 0, "runs": 0, "root_cause_tag": str(row.get("root_cause_tag", "")).strip()},
                )
                bucket["runs"] += 1
                if str(row.get("status", "")).strip().lower() != "run_complete":
                    bucket["fails"] += 1
                    continue
                metric = base._metric_to_float(row.get("run_best_valid_mrr20"))
                if metric is not None:
                    bucket["vals"].append(float(metric))
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for family, bucket in stats.items():
        vals = sorted(bucket["vals"], reverse=True)
        best = vals[0] if vals else 0.0
        topk = vals[: min(3, len(vals))]
        top_mean = sum(topk) / float(len(topk)) if topk else 0.0
        variance = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        fail_ratio = float(bucket["fails"]) / float(max(1, bucket["runs"]))
        score = best * 0.60 + top_mean * 0.35 - variance * 0.20 - fail_ratio * max(best, top_mean, 0.01) * 0.20
        out[family] = {
            "best_valid": best,
            "top_mean": top_mean,
            "variance": variance,
            "fail_ratio": fail_ratio,
            "score": score,
            "all_failed": not bool(vals),
            "root_cause_tag": bucket["root_cause_tag"],
        }
    return out


def _load_step2_family_stats(dataset: str, model_label: str) -> Dict[str, Dict[str, Any]]:
    return _load_step_family_stats("step2", dataset, model_label)


def _rank_step2_families(families: list[Dict[str, Any]], family_stats: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    def _key(family: Dict[str, Any]) -> tuple[float, float, str]:
        stat = family_stats.get(str(family["name"]), {})
        return (
            float(stat.get("score", -1.0)),
            float(stat.get("best_valid", -1.0)),
            str(family["name"]),
        )

    return sorted(families, key=_key, reverse=True)


def _build_step2_pass1_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    bank = _select_step2_hypothesis_bank(dataset, model_option, best_rec)
    families = list(bank["families"])
    counts = _allocate_even_counts(int(PASS_SPECS["pass1"]["candidate_count"]), families)
    variant_order = [
        "BASE",
        "WIDTH",
        "DEPTH",
        "LEN",
        "REG",
        "CTRL",
        "WIDTH_TUNE",
        "LEN_TUNE",
        "LR_DN",
        "LR_UP",
        "DEPTH_TUNE",
        "ANCHOR",
    ]
    candidates: list[Dict[str, Any]] = []
    for family, count in zip(families, counts):
        for idx in range(int(count)):
            variant = variant_order[idx % len(variant_order)]
            candidates.append(
                _make_step2_candidate(
                    dataset,
                    model_option,
                    "pass1",
                    family,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step2_hypothesis_bank",
                )
            )
    deduped = _dedup_candidates_stagei(candidates)
    return deduped[: int(PASS_SPECS["pass1"]["candidate_count"])]


def _build_step2_pass2_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    bank = _select_step2_hypothesis_bank(dataset, model_option, best_rec)
    families = list(bank["families"])
    family_stats = _load_step2_family_stats(dataset, OPTION_TO_LABEL[model_option])
    ranked = _rank_step2_families(families, family_stats)
    winning = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    control = next((f for f in families if bool(f.get("is_control")) and f is not winning and f is not second), None)
    if control is None:
        control = next((f for f in ranked if f is not winning and f is not second), second or winning)

    candidates: list[Dict[str, Any]] = []
    if winning is not None:
        for variant in ["BASE", "LR_DN", "LR_UP", "WIDTH_TUNE", "DEPTH_TUNE", "LEN_TUNE"]:
            candidates.append(
                _make_step2_candidate(
                    dataset,
                    model_option,
                    "pass2",
                    winning,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step2_family_refine",
                )
            )
    if second is not None:
        for variant in ["BASE", "LR_DN", "REG_TUNE"]:
            candidates.append(
                _make_step2_candidate(
                    dataset,
                    model_option,
                    "pass2",
                    second,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step2_family_refine",
                )
            )
    if control is not None:
        for variant in ["BASE", "CTRL"]:
            candidates.append(
                _make_step2_candidate(
                    dataset,
                    model_option,
                    "pass2",
                    control,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step2_family_refine",
                )
            )
    anchor_family = winning or control or families[0]
    candidates.append(
        _make_step2_candidate(
            dataset,
            model_option,
            "pass2",
            anchor_family,
            variant="ANCHOR",
            anchor_lo=float(bank["anchor_lo"]),
            anchor_hi=float(bank["anchor_hi"]),
            anchor_run_id=str(bank["anchor_run_id"]),
            source_stage="step2_family_refine",
        )
    )
    deduped = _dedup_candidates_stagei(candidates)
    if len(deduped) < int(PASS_SPECS["pass2"]["candidate_count"]) and winning is not None:
        for variant in ["REG", "CTRL", "WIDTH", "LEN"]:
            deduped = _dedup_candidates_stagei(
                deduped
                + [
                    _make_step2_candidate(
                        dataset,
                        model_option,
                        "pass2",
                        winning,
                        variant=variant,
                        anchor_lo=float(bank["anchor_lo"]),
                        anchor_hi=float(bank["anchor_hi"]),
                        anchor_run_id=str(bank["anchor_run_id"]),
                        source_stage="step2_family_refine",
                    )
                ]
            )
            if len(deduped) >= int(PASS_SPECS["pass2"]["candidate_count"]):
                break
    return deduped[: int(PASS_SPECS["pass2"]["candidate_count"])]


def _choice_overrides(**kwargs: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    search = {str(k): list(v) for k, v in kwargs.items()}
    types = {str(k): "choice" for k in search.keys()}
    return search, types


def _select_step3_hypothesis_bank(
    dataset: str,
    model_option: str,
    best_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    anchor, anchor_run_id, source_stage = _anchor_candidate(dataset, model_option, best_rec, step_name="step3")
    anchor_cfg = _normalize_stagei_cfg(model_option, dict(anchor["config"]))
    anchor_lo = float(anchor["lr_lo"])
    anchor_hi = float(anchor["lr_hi"])
    families: list[Dict[str, Any]] = []

    def fam(
        name: str,
        *,
        lr_policy: str,
        root: str,
        is_control: bool = False,
        search: Optional[Dict[str, Any]] = None,
        search_types: Optional[Dict[str, Any]] = None,
        lr_lo_override: Optional[float] = None,
        lr_hi_override: Optional[float] = None,
        **cfg_updates: Any,
    ) -> None:
        cfg = dict(anchor_cfg)
        cfg.update(cfg_updates)
        families.append(
            _step2_family_spec(
                name,
                _prepare_cfg_for_pass(
                    dataset,
                    model_option,
                    cfg,
                    "pass1",
                    step_name="step3",
                    hypothesis_family=name,
                ),
                lr_policy=lr_policy,
                root_cause_tag=root,
                is_control=is_control,
                search_overrides=search,
                search_space_type_overrides=search_types,
                lr_lo_override=lr_lo_override,
                lr_hi_override=lr_hi_override,
            )
        )

    combo = (dataset, model_option)
    if combo == ("amazon_beauty", "bsarec"):
        s1, t1 = _choice_overrides(
            bsarec_alpha=[0.55, 0.70, 0.85],
            bsarec_c=[3, 5, 7, 9],
            num_heads=[1, 2],
            n_heads=[1, 2],
            hidden_size=[64, 96, 128],
            embedding_size=[64, 96, 128],
            inner_size=[256, 384, 512],
            MAX_ITEM_LIST_LENGTH=[20, 32, 50],
            hidden_dropout_prob=[0.30, 0.50],
            dropout_prob=[0.30, 0.50],
            dropout_ratio=[0.30, 0.50],
            attn_dropout_prob=[0.30, 0.50],
            weight_decay=[5e-5, 1e-4, 2e-4],
        )
        fam("beauty_official_sparse", lr_policy="low_stable", root="bsarec_official_recipe_gap", hidden_size=64, embedding_size=64, num_layers=2, layers=2, num_heads=1, heads=1, inner_size=256, max_len=32, bsarec_alpha=0.7, bsarec_c=5, dropout=0.5, weight_decay=1e-4, search=s1, search_types=t1, lr_lo_override=1.2e-4, lr_hi_override=8.0e-4)
        s2, t2 = _choice_overrides(
            bsarec_alpha=[0.40, 0.55, 0.70],
            bsarec_c=[5, 7, 9, 11],
            num_heads=[1, 2],
            n_heads=[1, 2],
            hidden_size=[64, 96],
            embedding_size=[64, 96],
            inner_size=[256, 384],
            MAX_ITEM_LIST_LENGTH=[32, 40, 50],
            hidden_dropout_prob=[0.20, 0.30, 0.50],
            dropout_prob=[0.20, 0.30, 0.50],
            dropout_ratio=[0.20, 0.30, 0.50],
            attn_dropout_prob=[0.20, 0.30, 0.50],
        )
        fam("beauty_long_context", lr_policy="low_stable", root="bsarec_context_underreach", hidden_size=96, embedding_size=96, num_layers=2, layers=2, num_heads=1, heads=1, inner_size=384, max_len=40, bsarec_alpha=0.55, bsarec_c=7, dropout=0.3, weight_decay=1e-4, search=s2, search_types=t2, lr_lo_override=8.0e-5, lr_hi_override=6.0e-4)
        s3, t3 = _choice_overrides(
            bsarec_alpha=[0.50, 0.70],
            bsarec_c=[3, 5, 7],
            num_heads=[1, 2, 4],
            n_heads=[1, 2, 4],
            num_layers=[2, 3],
            n_layers=[2, 3],
            hidden_size=[96, 128, 160],
            embedding_size=[96, 128, 160],
            inner_size=[256, 384, 640],
            MAX_ITEM_LIST_LENGTH=[16, 24, 32],
            hidden_dropout_prob=[0.15, 0.30],
            dropout_prob=[0.15, 0.30],
            dropout_ratio=[0.15, 0.30],
            attn_dropout_prob=[0.15, 0.30],
        )
        fam("beauty_mixed_capacity", lr_policy="mid_balanced", root="bsarec_depth_head_mismatch", hidden_size=128, embedding_size=128, num_layers=3, layers=3, num_heads=2, heads=2, inner_size=384, max_len=24, bsarec_alpha=0.6, bsarec_c=5, dropout=0.2, weight_decay=1.2e-4, search=s3, search_types=t3, lr_lo_override=1.5e-4, lr_hi_override=1.2e-3)
        s4, t4 = _choice_overrides(
            bsarec_alpha=[0.35, 0.50, 0.65],
            bsarec_c=[3, 5, 7],
            num_heads=[2, 4],
            n_heads=[2, 4],
            hidden_size=[128, 192],
            embedding_size=[128, 192],
            inner_size=[384, 768],
            MAX_ITEM_LIST_LENGTH=[12, 16, 24],
        )
        fam("beauty_attention_control", lr_policy="high_control", root="bsarec_control_family", hidden_size=160, embedding_size=160, num_layers=2, layers=2, num_heads=4, heads=4, inner_size=640, max_len=16, bsarec_alpha=0.5, bsarec_c=3, dropout=0.15, weight_decay=8e-5, is_control=True, search=s4, search_types=t4)
    elif combo == ("amazon_beauty", "fame"):
        s1, t1 = _choice_overrides(
            num_heads=[1, 2],
            n_heads=[1, 2],
            num_experts=[1, 2, 4],
            fame_moe_last_k_layers=[1],
            hidden_size=[64, 96, 128],
            embedding_size=[64, 96, 128],
            num_layers=[2, 3],
            n_layers=[2, 3],
            MAX_ITEM_LIST_LENGTH=[16, 24, 32],
            hidden_dropout_prob=[0.10, 0.20, 0.30],
            dropout_prob=[0.10, 0.20, 0.30],
            dropout_ratio=[0.10, 0.20, 0.30],
            attn_dropout_prob=[0.10, 0.20, 0.30],
            weight_decay=[5e-5, 1e-4, 2e-4],
        )
        fam("beauty_few_heads", lr_policy="low_stable", root="fame_beauty_head_expert_mismatch", hidden_size=96, embedding_size=96, num_layers=2, layers=2, num_heads=1, heads=1, num_experts=2, fame_moe_last_k_layers=1, max_len=24, dropout=0.2, weight_decay=1e-4, search=s1, search_types=t1, lr_lo_override=1.5e-4, lr_hi_override=1.2e-3)
        s2, t2 = _choice_overrides(
            num_heads=[1, 2],
            n_heads=[1, 2],
            num_experts=[2, 4],
            fame_moe_last_k_layers=[1, 2],
            hidden_size=[96, 128],
            embedding_size=[96, 128],
            num_layers=[2, 3],
            n_layers=[2, 3],
            MAX_ITEM_LIST_LENGTH=[20, 32, 50],
            hidden_dropout_prob=[0.10, 0.20],
            dropout_prob=[0.10, 0.20],
            dropout_ratio=[0.10, 0.20],
            attn_dropout_prob=[0.10, 0.20],
        )
        fam("beauty_long_context", lr_policy="low_stable", root="fame_context_underreach", hidden_size=128, embedding_size=128, num_layers=2, layers=2, num_heads=2, heads=2, num_experts=2, fame_moe_last_k_layers=1, max_len=32, dropout=0.15, weight_decay=1e-4, search=s2, search_types=t2, lr_lo_override=1.0e-4, lr_hi_override=9.0e-4)
        s3, t3 = _choice_overrides(
            num_heads=[2, 4],
            n_heads=[2, 4],
            num_experts=[2, 4],
            fame_moe_last_k_layers=[1, 2],
            hidden_size=[128, 160, 192],
            embedding_size=[128, 160, 192],
            num_layers=[2, 3],
            n_layers=[2, 3],
            MAX_ITEM_LIST_LENGTH=[16, 24, 32],
        )
        fam("beauty_capacity_probe", lr_policy="mid_balanced", root="fame_capacity_shape_mismatch", hidden_size=160, embedding_size=160, num_layers=3, layers=3, num_heads=2, heads=2, num_experts=4, fame_moe_last_k_layers=2, max_len=24, dropout=0.1, weight_decay=1e-4, search=s3, search_types=t3, lr_lo_override=2.0e-4, lr_hi_override=1.5e-3)
        s4, t4 = _choice_overrides(
            num_heads=[2, 4],
            n_heads=[2, 4],
            num_experts=[2, 4],
            fame_moe_last_k_layers=[1, 2],
            hidden_size=[128, 192],
            embedding_size=[128, 192],
            MAX_ITEM_LIST_LENGTH=[12, 16, 24],
        )
        fam("beauty_midlr_control", lr_policy="high_control", root="fame_control_family", hidden_size=192, embedding_size=192, num_layers=2, layers=2, num_heads=4, heads=4, num_experts=4, fame_moe_last_k_layers=2, max_len=16, dropout=0.1, weight_decay=8e-5, is_control=True, search=s4, search_types=t4)
    elif combo == ("amazon_beauty", "gru4rec"):
        s1, t1 = _choice_overrides(
            hidden_size=[64, 128, 192],
            embedding_size=[64, 128, 192],
            num_layers=[1],
            n_layers=[1],
            MAX_ITEM_LIST_LENGTH=[16, 24, 32, 50],
            dropout_prob=[0.20, 0.30, 0.40],
            weight_decay=[0.0, 1e-5, 1e-4],
        )
        fam("classic_single_layer", lr_policy="low_stable", root="gru_classic_recipe_missing", hidden_size=128, embedding_size=128, num_layers=1, layers=1, max_len=24, dropout=0.3, weight_decay=1e-5, search=s1, search_types=t1, lr_lo_override=1.5e-4, lr_hi_override=2.0e-3)
        s2, t2 = _choice_overrides(
            hidden_size=[64, 128],
            embedding_size=[64, 128],
            num_layers=[2, 3],
            n_layers=[2, 3],
            MAX_ITEM_LIST_LENGTH=[16, 24, 32],
            dropout_prob=[0.20, 0.30, 0.40],
            weight_decay=[1e-5, 1e-4, 3e-4],
        )
        fam("deeper_dropout", lr_policy="mid_balanced", root="gru_depth_regularization_gap", hidden_size=128, embedding_size=128, num_layers=2, layers=2, max_len=24, dropout=0.3, weight_decay=1e-4, search=s2, search_types=t2, lr_lo_override=2.0e-4, lr_hi_override=2.5e-3)
        s3, t3 = _choice_overrides(
            hidden_size=[96, 160, 224],
            embedding_size=[96, 160, 224],
            num_layers=[1, 2],
            n_layers=[1, 2],
            MAX_ITEM_LIST_LENGTH=[24, 32, 50],
            dropout_prob=[0.10, 0.20, 0.30],
            weight_decay=[0.0, 1e-5, 1e-4],
        )
        fam("long_context", lr_policy="low_stable", root="gru_context_underreach", hidden_size=160, embedding_size=160, num_layers=1, layers=1, max_len=32, dropout=0.2, weight_decay=1e-5, search=s3, search_types=t3, lr_lo_override=1.5e-4, lr_hi_override=1.6e-3)
        s4, t4 = _choice_overrides(
            hidden_size=[160, 224, 320],
            embedding_size=[160, 224, 320],
            num_layers=[1, 2],
            n_layers=[1, 2],
            MAX_ITEM_LIST_LENGTH=[10, 16, 24],
            dropout_prob=[0.05, 0.10, 0.20],
            weight_decay=[0.0, 1e-5],
        )
        fam("high_capacity_control", lr_policy="high_control", root="gru_control_family", hidden_size=224, embedding_size=224, num_layers=2, layers=2, max_len=16, dropout=0.1, weight_decay=1e-5, is_control=True, search=s4, search_types=t4)
    elif combo == ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"):
        s1, t1 = _choice_overrides(
            hidden_size=[128, 224, 320],
            embedding_size=[128, 224, 320],
            num_layers=[1, 2],
            n_layers=[1, 2],
            MAX_ITEM_LIST_LENGTH=[12, 20, 32, 50],
            dropout_prob=[0.10, 0.20, 0.30],
            weight_decay=[0.0, 1e-5, 5e-5],
        )
        fam("stable_midlr_classic", lr_policy="mid_balanced", root="gru_joint_capacity_search_needed", hidden_size=224, embedding_size=224, num_layers=1, layers=1, max_len=20, dropout=0.2, weight_decay=1e-5, search=s1, search_types=t1)
        s2, t2 = _choice_overrides(
            hidden_size=[128, 192, 256],
            embedding_size=[128, 192, 256],
            num_layers=[1, 2],
            n_layers=[1, 2],
            MAX_ITEM_LIST_LENGTH=[20, 32, 50],
            dropout_prob=[0.10, 0.20, 0.30],
            weight_decay=[0.0, 1e-5, 5e-5],
        )
        fam("long_context_probe", lr_policy="low_stable", root="gru_context_underreach", hidden_size=192, embedding_size=192, num_layers=1, layers=1, max_len=32, dropout=0.2, weight_decay=1e-5, search=s2, search_types=t2)
        s3, t3 = _choice_overrides(
            hidden_size=[64, 128, 192],
            embedding_size=[64, 128, 192],
            num_layers=[1, 2, 3],
            n_layers=[1, 2, 3],
            MAX_ITEM_LIST_LENGTH=[12, 20, 32],
            dropout_prob=[0.20, 0.30, 0.40],
            weight_decay=[1e-5, 1e-4],
        )
        fam("compact_dropout", lr_policy="mid_balanced", root="gru_overcapacity_bias", hidden_size=128, embedding_size=128, num_layers=1, layers=1, max_len=20, dropout=0.3, weight_decay=1e-4, search=s3, search_types=t3)
        s4, t4 = _choice_overrides(
            hidden_size=[224, 320, 384],
            embedding_size=[224, 320, 384],
            num_layers=[2, 3],
            n_layers=[2, 3],
            MAX_ITEM_LIST_LENGTH=[10, 16, 24],
            dropout_prob=[0.05, 0.10, 0.20],
            weight_decay=[0.0, 1e-5],
        )
        fam("high_capacity_control", lr_policy="high_control", root="gru_control_family", hidden_size=320, embedding_size=320, num_layers=2, layers=2, max_len=16, dropout=0.1, weight_decay=1e-5, is_control=True, search=s4, search_types=t4)
    elif model_option == "gru4rec":
        s, t = _choice_overrides(
            hidden_size=[64, 128, 192],
            embedding_size=[64, 128, 192],
            num_layers=[1, 2],
            n_layers=[1, 2],
            MAX_ITEM_LIST_LENGTH=[12, 20, 32],
            dropout_prob=[0.10, 0.20, 0.30],
            weight_decay=[0.0, 1e-5, 1e-4],
        )
        fam("generic_gru_joint", lr_policy="mid_balanced", root="gru_joint_search_generic", hidden_size=128, embedding_size=128, num_layers=1, layers=1, max_len=20, dropout=0.2, weight_decay=1e-5, search=s, search_types=t)
    else:
        families = list(_select_step2_hypothesis_bank(dataset, model_option, best_rec)["families"])

    return {
        "anchor": anchor,
        "anchor_run_id": anchor_run_id,
        "source_stage": source_stage,
        "families": families,
        "anchor_cfg": anchor_cfg,
        "anchor_lo": anchor_lo,
        "anchor_hi": anchor_hi,
    }


def _build_step3_pass1_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    bank = _select_step3_hypothesis_bank(dataset, model_option, best_rec)
    families = list(bank["families"])
    counts = _allocate_even_counts(int(PASS_SPECS["pass1"]["candidate_count"]), families)
    variant_order = ["BASE", "WIDTH", "DEPTH", "LEN", "REG", "CTRL", "WIDTH_TUNE", "LEN_TUNE", "LR_DN", "LR_UP", "DEPTH_TUNE", "ANCHOR"]
    candidates: list[Dict[str, Any]] = []
    for family, count in zip(families, counts):
        for idx in range(int(count)):
            variant = variant_order[idx % len(variant_order)]
            candidates.append(
                _make_step_candidate(
                    "step3",
                    dataset,
                    model_option,
                    "pass1",
                    family,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step3_hypothesis_bank",
                )
            )
    deduped = _dedup_candidates_stagei(candidates)
    return deduped[: int(PASS_SPECS["pass1"]["candidate_count"])]


def _build_step3_pass2_candidates(dataset: str, model_option: str, best_rec: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
    bank = _select_step3_hypothesis_bank(dataset, model_option, best_rec)
    families = list(bank["families"])
    family_stats = _load_step_family_stats("step3", dataset, OPTION_TO_LABEL[model_option])
    ranked = _rank_step2_families(families, family_stats)
    winning = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None
    control = next((f for f in families if bool(f.get("is_control")) and f is not winning and f is not second), None)
    if control is None:
        control = next((f for f in ranked if f is not winning and f is not second), second or winning)

    candidates: list[Dict[str, Any]] = []
    if winning is not None:
        for variant in ["BASE", "LR_DN", "LR_UP", "WIDTH_TUNE", "DEPTH_TUNE", "LEN_TUNE"]:
            candidates.append(
                _make_step_candidate(
                    "step3",
                    dataset,
                    model_option,
                    "pass2",
                    winning,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step3_family_refine",
                )
            )
    if second is not None:
        for variant in ["BASE", "LR_DN", "REG_TUNE"]:
            candidates.append(
                _make_step_candidate(
                    "step3",
                    dataset,
                    model_option,
                    "pass2",
                    second,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step3_family_refine",
                )
            )
    if control is not None:
        for variant in ["BASE", "CTRL"]:
            candidates.append(
                _make_step_candidate(
                    "step3",
                    dataset,
                    model_option,
                    "pass2",
                    control,
                    variant=variant,
                    anchor_lo=float(bank["anchor_lo"]),
                    anchor_hi=float(bank["anchor_hi"]),
                    anchor_run_id=str(bank["anchor_run_id"]),
                    source_stage="step3_family_refine",
                )
            )
    anchor_family = winning or control or families[0]
    candidates.append(
        _make_step_candidate(
            "step3",
            dataset,
            model_option,
            "pass2",
            anchor_family,
            variant="ANCHOR",
            anchor_lo=float(bank["anchor_lo"]),
            anchor_hi=float(bank["anchor_hi"]),
            anchor_run_id=str(bank["anchor_run_id"]),
            source_stage="step3_family_refine",
        )
    )
    deduped = _dedup_candidates_stagei(candidates)
    if len(deduped) < int(PASS_SPECS["pass2"]["candidate_count"]) and winning is not None:
        for variant in ["REG", "CTRL", "WIDTH", "LEN"]:
            deduped = _dedup_candidates_stagei(
                deduped
                + [
                    _make_step_candidate(
                        "step3",
                        dataset,
                        model_option,
                        "pass2",
                        winning,
                        variant=variant,
                        anchor_lo=float(bank["anchor_lo"]),
                        anchor_hi=float(bank["anchor_hi"]),
                        anchor_run_id=str(bank["anchor_run_id"]),
                        source_stage="step3_family_refine",
                    )
                ]
            )
            if len(deduped) >= int(PASS_SPECS["pass2"]["candidate_count"]):
                break
    return deduped[: int(PASS_SPECS["pass2"]["candidate_count"])]


def _estimated_cost(row: Dict[str, Any]) -> float:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"]).lower()
    cfg = dict(row.get("candidate_config", {}) or {})
    max_evals = max(1, int(row.get("max_evals", 5)))
    tune_epochs = max(1, int(row.get("tune_epochs", 30)))
    length_factor = max(0.7, min(1.5, float(int(cfg.get("max_len", 20))) / 20.0))
    width_factor = max(0.8, min(1.4, float(int(cfg.get("hidden_size", 144))) / 144.0))
    return (
        float(base.DATASET_COST_WEIGHT.get(dataset, 1.0))
        * float(base.MODEL_COST_WEIGHT.get(model_option, 1.0))
        * (float(max_evals) / 5.0)
        * (float(tune_epochs) / 30.0)
        * length_factor
        * width_factor
    )


def _selected_datasets(args: argparse.Namespace) -> list[str]:
    datasets = list(dict.fromkeys(base._parse_csv_strings(args.datasets)))
    return datasets or list(DEFAULT_DATASETS)


def _selected_model_options(args: argparse.Namespace) -> list[str]:
    raw = [m.lower() for m in base._parse_csv_strings(args.models)]
    selected = [m for m in raw if m in OPTION_TO_LABEL]
    return selected or list(DEFAULT_MODELS)


def _build_rows_for_pass(
    pass_name: str,
    args: argparse.Namespace,
    *,
    best_rows: Optional[list[Dict[str, Any]]] = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    step_name = _step_name(args)
    datasets = _selected_datasets(args)
    model_options = _selected_model_options(args)
    rows_snapshot = best_rows
    if rows_snapshot is None:
        rows_snapshot = _load_overall_best_table(datasets=set(datasets), model_options=set(model_options))
    best_by_combo = {(str(r["dataset"]), str(r["model_option"])): r for r in rows_snapshot}

    if pass_name == "pass1":
        targets = _select_pass1_targets(rows_snapshot)
    elif pass_name == "pass2":
        targets = _select_pass2_targets(rows_snapshot)
    else:
        targets = _select_pass3_targets(rows_snapshot)

    rows: list[Dict[str, Any]] = []
    seeds = base._parse_csv_ints(args.seeds)
    run_cursor = 0
    pass_spec = PASS_SPECS[pass_name]
    for combo_idx, target in enumerate(targets, start=1):
        dataset = str(target["dataset"])
        model_option = str(target["model_option"])
        model_label = OPTION_TO_LABEL[model_option]
        best_rec = best_by_combo.get((dataset, model_option))
        if step_name == "step2" and pass_name == "pass1":
            candidates = _build_step2_pass1_candidates(dataset, model_option, best_rec)
        elif step_name == "step2" and pass_name == "pass2":
            candidates = _build_step2_pass2_candidates(dataset, model_option, best_rec)
        elif step_name == "step3" and pass_name == "pass1":
            candidates = _build_step3_pass1_candidates(dataset, model_option, best_rec)
        elif step_name == "step3" and pass_name == "pass2":
            candidates = _build_step3_pass2_candidates(dataset, model_option, best_rec)
        elif pass_name == "pass1":
            candidates = _build_pass1_candidates(dataset, model_option)
        elif pass_name == "pass2":
            candidates = _build_pass2_candidates(dataset, model_option, best_rec)
        else:
            candidates = _build_pass3_candidates(dataset, model_option, best_rec)

        for cand_idx, cand in enumerate(candidates, start=1):
            hypothesis_family = str(cand.get("hypothesis_family", ""))
            cfg = _prepare_cfg_for_pass(
                dataset,
                model_option,
                dict(cand["config"]),
                pass_name,
                step_name=step_name,
                hypothesis_family=hypothesis_family,
            )
            if str(cand.get("batch_profile", "")).strip():
                batch_profile_label = str(cand["batch_profile"])
            else:
                batch_profile_label = str(
                    _batch_profile(
                        dataset,
                        model_option,
                        cfg,
                        pass_name,
                        step_name=step_name,
                        hypothesis_family=hypothesis_family,
                    )["label"]
                )

            for seed_id in seeds:
                run_cursor += 1
                token = _step_pass_token(step_name, pass_name)
                model_tag = base._sanitize_token(model_label, upper=True)
                cand_tag = base._sanitize_token(str(cand["candidate_id"]), upper=True)[:48]
                if step_name == "step1":
                    phase_prefix = f"{base.PHASE_ID}_{token}"
                elif step_name == "step2":
                    phase_prefix = f"{base.PHASE_ID}_S2_{token}"
                else:
                    phase_prefix = f"{base.PHASE_ID}_S3_{token}"
                row = {
                    "dataset": dataset,
                    "phase_id": base.PHASE_ID,
                    "axis_id": "SI",
                    "axis_desc": base.AXIS_DESC,
                    "setting_id": f"STAGEI_{token}_{model_tag}_{cand_tag}_S{int(seed_id)}",
                    "setting_key": f"STAGEI_{token}",
                    "setting_desc": f"STAGEI_{token}_{model_tag}_{cand_tag}_S{int(seed_id)}",
                    "hparam_id": cand_tag[:40],
                    "seed_id": int(seed_id),
                    "run_phase": (
                        f"{phase_prefix}_D{combo_idx:02d}_M{base._sanitize_token(model_tag, upper=True)}_"
                        f"C{cand_idx:02d}_{cand_tag}_S{int(seed_id)}"
                    ),
                    "run_id": f"{token}_{base._sanitize_token(dataset, upper=True)}_{model_tag}_{cand_tag}_S{int(seed_id)}",
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "stage": "stagei",
                    "step_name": step_name,
                    "pass_name": pass_name,
                    "model_option": model_option,
                    "model_label": model_label,
                    "candidate_id": str(cand["candidate_id"]),
                    "candidate_source": str(cand["source"]),
                    "source_stage": str(cand["source_stage"]),
                    "transfer_from_dataset": str(cand.get("transfer_from_dataset", "")),
                    "anchor_run_id": str(cand.get("anchor_run_id", "")),
                    "hypothesis_family": hypothesis_family,
                    "root_cause_tag": str(cand.get("root_cause_tag", "")),
                    "batch_profile": batch_profile_label,
                    "candidate_config": cfg,
                    "lr_lo": float(cand["lr_lo"]),
                    "lr_hi": float(cand["lr_hi"]),
                    "search_overrides": dict(cand.get("search_overrides", {}) or {}),
                    "search_space_type_overrides": dict(cand.get("search_space_type_overrides", {}) or {}),
                    "max_evals": int(pass_spec["max_evals"]),
                    "tune_epochs": int(pass_spec["tune_epochs"]),
                    "tune_patience": int(pass_spec["tune_patience"]),
                    "target_ratio": float(target["ratio"]),
                }
                row["estimated_cost"] = _estimated_cost(row)
                rows.append(row)

    if args.smoke_test:
        rows = rows[: max(1, int(args.smoke_max_runs))]
    return targets, rows


def _write_manifest(pass_name: str, args: argparse.Namespace, rows: list[Dict[str, Any]], targets: list[Dict[str, Any]]) -> Path:
    step_name = _step_name(args)
    axis_dir = base.LOG_ROOT / base.AXIS
    axis_dir.mkdir(parents=True, exist_ok=True)
    if str(args.manifest_out or "").strip():
        out = Path(str(args.manifest_out)).expanduser()
        path = out.with_name(f"{out.stem}_{step_name}_{pass_name}.json")
    else:
        path = axis_dir / f"{step_name}_{pass_name}_manifest.json"
    payload = {
        "track": base.TRACK,
        "axis": base.AXIS,
        "phase": base.PHASE_ID,
        "step_name": step_name,
        "pass_name": pass_name,
        "pass_spec": PASS_SPECS[pass_name],
        "target_count": len(targets),
        "run_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "targets": [
            {
                "dataset": t["dataset"],
                "model_option": t["model_option"],
                "model_label": t["model_label"],
                "best_valid": t["best_valid"],
                "ratio": t["ratio"],
                "axis": t["axis"],
                "run_id": t["run_id"],
                "step_name": t.get("step_name", "step1"),
            }
            for t in targets
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, cmd: list[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{base.PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"step_name={row.get('step_name','')} pass_name={row.get('pass_name','')} model={row.get('model_label','')} "
            f"candidate_id={row.get('candidate_id','')} seed={row.get('seed_id','')}"
        ),
        (
            f"dataset={row.get('dataset','')} gpu={gpu_id} "
            f"candidate_source={row.get('candidate_source','')} source_stage={row.get('source_stage','')} "
            f"anchor_run_id={row.get('anchor_run_id','')} hypothesis_family={row.get('hypothesis_family','')} "
            f"root_cause_tag={row.get('root_cause_tag','')} batch_profile={row.get('batch_profile','')}"
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


def _result_trial_health(result_json_path: str) -> Dict[str, Any]:
    path = Path(str(result_json_path or "")).expanduser()
    if not path.exists():
        return {
            "parse_ok": False,
            "ok_trials": 0,
            "fail_trials": 0,
            "completed_trials": 0,
            "all_failed": False,
            "top_error": "",
            "detail": f"result_missing:{path}",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "parse_ok": False,
            "ok_trials": 0,
            "fail_trials": 0,
            "completed_trials": 0,
            "all_failed": False,
            "top_error": "",
            "detail": f"result_parse_error:{exc}",
        }

    ok_trials = 0
    fail_trials = 0
    error_counter: Counter[str] = Counter()
    for trial in list(payload.get("trials", []) or []):
        status = str(trial.get("status", "")).strip().lower()
        if status == "ok":
            ok_trials += 1
        elif status == "fail":
            fail_trials += 1
            err = str(trial.get("error", "")).strip() or "unknown_error"
            error_counter[err] += 1
    completed_trials = ok_trials + fail_trials
    top_error = error_counter.most_common(1)[0][0] if error_counter else ""
    return {
        "parse_ok": True,
        "ok_trials": int(ok_trials),
        "fail_trials": int(fail_trials),
        "completed_trials": int(completed_trials),
        "all_failed": bool(completed_trials > 0 and ok_trials == 0),
        "top_error": top_error,
        "detail": (
            f"ok_trials={ok_trials} fail_trials={fail_trials} completed_trials={completed_trials} "
            f"top_error={top_error}"
        ),
    }


def _host_memory_group(row: Dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).strip()
    model_option = str(row.get("model_option", "")).strip().lower()
    if (dataset, model_option) in HOST_MEMORY_SERIAL_COMBOS:
        return f"serial::{dataset}::{model_option}"
    return ""


def _can_launch_with_host_memory_limit(row: Dict[str, Any], active: Dict[str, Dict[str, Any]]) -> bool:
    group = _host_memory_group(row)
    if not group:
        return True
    active_same = 0
    for slot in active.values():
        if _host_memory_group(slot.get("row", {})) == group:
            active_same += 1
    return active_same < 1


def _pop_next_launchable_row(shared_queue: deque, active: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not shared_queue:
        return None
    qlen = len(shared_queue)
    for _ in range(qlen):
        row = shared_queue.popleft()
        if _can_launch_with_host_memory_limit(row, active):
            return row
        shared_queue.append(row)
    return None


def _build_command(row: Dict[str, Any], gpu_id: str) -> list[str]:
    model_option = str(row["model_option"]).lower()
    h = base._effective_model_hparams(row)
    lr_lo = float(row["lr_lo"])
    lr_hi = float(row["lr_hi"])
    tune_epochs = int(row["tune_epochs"])
    tune_patience = int(row["tune_patience"])
    max_evals = int(row["max_evals"])

    dropout = float(h["dropout"])
    weight_decay = float(h["weight_decay"])
    search: Dict[str, Any] = {"learning_rate": [float(lr_lo), float(lr_hi)]}
    search_types: Dict[str, str] = {"learning_rate": "loguniform"}

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
        "time_span": int(h["time_span"]),
        "num_experts": int(h["num_experts"]),
        "bsarec_c": int(dict(row.get("candidate_config", {}) or {}).get("bsarec_c", 3)),
        "fame_moe_last_k_layers": int(dict(row.get("candidate_config", {}) or {}).get("fame_moe_last_k_layers", 1)),
        "state_size": int(h["sigma_state"]),
        "conv_kernel": int(h["sigma_kernel"]),
        "remaining_ratio": float(h["sigma_remaining_ratio"]),
        "MAX_ITEM_LIST_LENGTH": int(h["max_len"]),
    }

    extra_search = dict(row.get("search_overrides", {}) or {})
    extra_search_types = dict(row.get("search_space_type_overrides", {}) or {})
    searched_keys = {str(k) for k in extra_search.keys()}
    for key, values in extra_search.items():
        fixed_search.pop(str(key), None)
        if isinstance(values, (list, tuple)):
            search[str(key)] = list(values)
        else:
            search[str(key)] = [values]
        search_types[str(key)] = str(extra_search_types.get(str(key), "choice"))

    for key, value in fixed_search.items():
        search[key] = [value]
        search_types[key] = "choice"

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
        f"++search={base.hydra_literal(search)}",
        f"++search_space_type_overrides={base.hydra_literal(search_types)}",
        f"++weight_decay={weight_decay}",
    ]

    if model_option == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    def _filter_cli_overrides(overrides: list[str]) -> list[str]:
        if not searched_keys:
            return list(overrides)
        out: list[str] = []
        for item in overrides:
            token = str(item)
            if not token.startswith("++"):
                out.append(token)
                continue
            key = token[2:].split("=", 1)[0].strip()
            if key in searched_keys:
                continue
            out.append(token)
        return out

    cmd.extend(_filter_cli_overrides(base._model_algorithm_overrides(row)))
    cmd.extend(base._model_runtime_resource_overrides(row))
    cmd.extend(_filter_cli_overrides(base._base_hparam_overrides(row)))
    return cmd


def _print_pass_plan(pass_name: str, targets: list[Dict[str, Any]], rows: list[Dict[str, Any]]) -> None:
    step_name = _row_step_name(rows[0]) if rows else "step1"
    spec = PASS_SPECS[pass_name]
    combos = len(targets)
    candidates = len({(str(r["dataset"]), str(r["model_option"]), str(r["candidate_id"])) for r in rows})
    print(
        f"[{_step_pass_token(step_name, pass_name)}] combos={combos} candidates={candidates} runs={len(rows)} "
        f"epochs={spec['tune_epochs']} patience={spec['tune_patience']} max_evals={spec['max_evals']}"
    )
    for target in targets:
        print(
            f"[plan] {pass_name} dataset={target['dataset']} model={target['model_label']} "
            f"ratio={float(target['ratio']):.4f} best_valid={float(target['best_valid']):.4f} axis={target['axis']}"
        )


def _run_rows(
    pass_name: str,
    args: argparse.Namespace,
    rows: list[Dict[str, Any]],
    *,
    global_total_runs: Optional[int] = None,
    global_done_base: int = 0,
) -> int:
    step_name = _step_name(args)
    if not rows:
        print(f"[{_step_pass_token(step_name, pass_name)}] no rows to run.")
        return 0

    gpus = list(dict.fromkeys(base._parse_csv_strings(args.gpus)))
    if not gpus:
        raise RuntimeError("No GPUs provided")

    datasets = sorted({str(r["dataset"]) for r in rows})
    for dataset in datasets:
        base._validate_session_fixed_files(dataset)

    summary_paths = {dataset: base._summary_path(dataset) for dataset in datasets}
    for path in summary_paths.values():
        base._ensure_summary_csv(path)
    summary_state = {dataset: list(base._load_summary_bests(path)) for dataset, path in summary_paths.items()}

    completed_keys_by_dataset: Dict[str, set[Tuple[str, str, str]]] = {}
    for dataset, path in summary_paths.items():
        done_keys: set[Tuple[str, str, str]] = set()
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                for rec in csv.DictReader(fh):
                    if str(rec.get("status", "")).strip().lower() != "run_complete":
                        continue
                    model = str(rec.get("model", "")).strip()
                    candidate_id = str(rec.get("candidate_id", "")).strip()
                    seed_id = str(rec.get("seed_id", "")).strip()
                    rec_step_name = str(rec.get("step_name", "step1") or "step1").strip().lower()
                    if model and candidate_id and seed_id:
                        done_keys.add((rec_step_name, model, candidate_id, seed_id))
        except Exception:
            pass
        completed_keys_by_dataset[dataset] = done_keys

    runnable: list[Dict[str, Any]] = []
    skipped_rows: list[Dict[str, Any]] = []
    result_indexes = {dataset: (base._scan_result_index(dataset) if args.verify_logging else {}) for dataset in datasets}

    for row in rows:
        row["log_path"] = str(base._build_log_path(row))
        dataset = str(row["dataset"])
        resume_key = (
            step_name,
            str(row.get("model_label", "")).strip(),
            str(row.get("candidate_id", "")).strip(),
            str(row.get("seed_id", "")).strip(),
        )
        if resume_key in completed_keys_by_dataset.get(dataset, set()):
            skipped_rows.append(row)
            continue
        completed = base._is_completed_any_log(row, use_resume=bool(args.resume_from_logs))
        if not completed:
            runnable.append(row)
            continue
        if not args.verify_logging:
            skipped_rows.append(row)
            continue
        rec = result_indexes.get(dataset, {}).get(str(row["run_phase"]))
        if not rec:
            runnable.append(row)
            continue
        result_path = str(rec.get("path", "") or "")
        health = _result_trial_health(result_path)
        if int(health.get("ok_trials", 0)) <= 0:
            print(f"[resume-check] run={row['run_phase']} no_ok_trials -> rerun ({health['detail']})")
            runnable.append(row)
            continue
        ok, detail = base._verify_special_from_result(result_path)
        if ok:
            skipped_rows.append(row)
            continue
        print(f"[resume-check] run={row['run_phase']} special_check_failed -> rerun ({detail})")
        runnable.append(row)

    notifier = None
    prev_total_env = os.environ.get("SLACK_NOTIFY_TOTAL_RUNS")
    prev_done_env = os.environ.get("SLACK_NOTIFY_GLOBAL_DONE_BASE")
    if not args.dry_run:
        if global_total_runs is not None:
            os.environ["SLACK_NOTIFY_TOTAL_RUNS"] = str(int(global_total_runs))
        os.environ["SLACK_NOTIFY_GLOBAL_DONE_BASE"] = str(max(int(global_done_base), 0))
        notifier = SlackProgressNotifier(phase_label=f"baseline StageI {step_name} {pass_name}", rows=rows)
        notifier.notify_plan(precompleted_rows=skipped_rows)

    try:
        if not runnable:
            print(f"[{_step_pass_token(step_name, pass_name)}] all runs already complete.")
            for dataset in datasets:
                base._update_baseline_phase_summary(dataset, base.PHASE_ID)
            return 0

        runnable.sort(key=lambda r: float(r.get("estimated_cost", 1.0)), reverse=True)
        print(
            f"[queue] {pass_name} planned={len(rows)} runnable={len(runnable)} "
            f"skipped={len(skipped_rows)} gpus={','.join(gpus)}"
        )

        if args.dry_run:
            gpu_bins = base._plan_gpu_bins(runnable, gpus)
            for gpu_id in gpus:
                for row in gpu_bins[gpu_id]:
                    cmd = _build_command(row, gpu_id)
                    print(
                        f"[dry-run] pass={pass_name} dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} "
                        f"model={row['model_label']} candidate={row['candidate_id']} seed={row['seed_id']}"
                    )
                    print("          " + " ".join(cmd))
            return 0

        active: Dict[str, Dict[str, Any]] = {}
        shared_queue: deque = deque(runnable)
        try:
            while True:
                for gpu_id in gpus:
                    if gpu_id in active or not shared_queue:
                        continue
                    row = _pop_next_launchable_row(shared_queue, active)
                    if row is None:
                        continue
                    cmd = _build_command(row, gpu_id)
                    log_path = Path(str(row["log_path"]))
                    _write_log_preamble(log_path, row, gpu_id, cmd)
                    env = dict(os.environ)
                    env["PYTHONUNBUFFERED"] = "1"
                    env.setdefault("HYPEROPT_RESULTS_DIR", str(base.ARTIFACT_ROOT / "results"))
                    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                    with log_path.open("a", encoding="utf-8") as fh:
                        proc = subprocess.Popen(cmd, cwd=base.EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
                    active[gpu_id] = {"proc": proc, "row": row, "log_path": str(log_path)}
                    print(
                        f"[launch] pass={pass_name} dataset={row['dataset']} gpu={gpu_id} "
                        f"run_phase={row['run_phase']} model={row['model_label']} candidate={row['candidate_id']}"
                    )

                done_gpus: list[str] = []
                for gpu_id, slot in active.items():
                    rc = slot["proc"].poll()
                    if rc is None:
                        continue
                    done_gpus.append(gpu_id)
                    row = slot["row"]
                    dataset = str(row["dataset"])
                    print(f"[done] pass={pass_name} dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} rc={rc}")

                    rec = base._get_result_row_from_log_or_scan(
                        dataset=dataset,
                        run_phase=str(row["run_phase"]),
                        log_path=Path(str(slot["log_path"])),
                        retries=4,
                        sleep_sec=0.75,
                    )
                    run_best = None
                    test_mrr = None
                    n_completed = None
                    interrupted = None
                    result_path = ""
                    special_ok = False
                    ok_trials = 0
                    trial_health_detail = ""
                    if rec:
                        run_best = base._metric_to_float(rec.get("best_mrr"))
                        test_mrr = base._metric_to_float(rec.get("test_mrr"))
                        n_completed = int(rec.get("n_completed", 0) or 0)
                        interrupted = bool(rec.get("interrupted", False))
                        result_path = str(rec.get("path", "") or "")
                        if result_path:
                            trial_health = _result_trial_health(result_path)
                            ok_trials = int(trial_health.get("ok_trials", 0))
                            trial_health_detail = str(trial_health.get("detail", ""))
                            base._mirror_logging_bundle(row, result_path)
                            if ok_trials > 0:
                                special_ok, detail = base._verify_special_from_result(result_path)
                                print(f"[logging-check] run={row['run_phase']} {detail} result={result_path}")
                                if args.verify_logging and int(rc) == 0 and not special_ok:
                                    raise RuntimeError(
                                        f"special logging verification failed: run_phase={row['run_phase']} "
                                        f"special_ok={special_ok} result={result_path}"
                                    )
                            else:
                                print(
                                    f"[result-check] run={row['run_phase']} no_ok_trials "
                                    f"({trial_health_detail}) result={result_path}"
                                )

                    global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model = summary_state[dataset]
                    if run_best is not None:
                        global_best_valid = run_best if global_best_valid is None else max(float(global_best_valid), float(run_best))
                        prev_model_best = model_best_valid_by_model.get(str(row["model_label"]))
                        model_best_valid_by_model[str(row["model_label"])] = (
                            run_best if prev_model_best is None else max(float(prev_model_best), float(run_best))
                        )
                    if test_mrr is not None:
                        global_best_test = test_mrr if global_best_test is None else max(float(global_best_test), float(test_mrr))
                        prev_model_test = model_best_test_by_model.get(str(row["model_label"]))
                        model_best_test_by_model[str(row["model_label"])] = (
                            test_mrr if prev_model_test is None else max(float(prev_model_test), float(test_mrr))
                        )

                    summary_state[dataset] = [global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model]
                    status = "run_complete" if int(rc) == 0 and int(ok_trials) > 0 else "run_fail"
                    summary_row = {
                        "model": row["model_label"],
                        "global_best_valid_mrr20": "" if global_best_valid is None else f"{float(global_best_valid):.6f}",
                        "global_best_test_mrr20": "" if global_best_test is None else f"{float(global_best_test):.6f}",
                        "model_best_valid_mrr20": (
                            "" if model_best_valid_by_model.get(str(row["model_label"])) is None else f"{float(model_best_valid_by_model[str(row['model_label'])]):.6f}"
                        ),
                        "model_best_test_mrr20": (
                            "" if model_best_test_by_model.get(str(row["model_label"])) is None else f"{float(model_best_test_by_model[str(row['model_label'])]):.6f}"
                        ),
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
                        "candidate_id": row["candidate_id"],
                        "candidate_source": row["candidate_source"],
                        "source_stage": row["source_stage"],
                        "transfer_from_dataset": row["transfer_from_dataset"],
                        "anchor_run_id": row["anchor_run_id"],
                        "step_name": row["step_name"],
                        "pass_name": row["pass_name"],
                        "hypothesis_family": row.get("hypothesis_family", ""),
                        "root_cause_tag": row.get("root_cause_tag", ""),
                        "batch_profile": row["batch_profile"],
                        "result_path": result_path,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                    base._append_summary_row(summary_paths[dataset], summary_row)
                    base._update_baseline_phase_summary(dataset, base.PHASE_ID)
                    if notifier is not None:
                        notifier.mark_complete(row)

                    if int(rc) != 0:
                        raise RuntimeError(f"run failed: pass={pass_name} dataset={dataset} run_phase={row['run_phase']} rc={rc}")
                    if int(ok_trials) <= 0:
                        print(
                            f"[warn] pass={pass_name} dataset={dataset} run_phase={row['run_phase']} "
                            f"finished without successful trials; recorded as run_fail ({trial_health_detail})"
                        )

                for gpu_id in done_gpus:
                    active.pop(gpu_id, None)
                if not shared_queue and not active:
                    break
                time.sleep(1)
        except Exception:
            base._terminate_active(active)
            raise
        return 0
    finally:
        if prev_total_env is None:
            os.environ.pop("SLACK_NOTIFY_TOTAL_RUNS", None)
        else:
            os.environ["SLACK_NOTIFY_TOTAL_RUNS"] = prev_total_env
        if prev_done_env is None:
            os.environ.pop("SLACK_NOTIFY_GLOBAL_DONE_BASE", None)
        else:
            os.environ["SLACK_NOTIFY_GLOBAL_DONE_BASE"] = prev_done_env


def _pass_sequence(args: argparse.Namespace) -> list[str]:
    step_name = _step_name(args)
    requested = str(args.stagei_pass)
    if step_name == "step2":
        if requested == "pass3":
            raise RuntimeError("step2 does not support pass3; use --pass pass1|pass2|all")
        if requested == "all":
            return list(STEP2_PASS_SEQUENCE)
        return [requested]
    if step_name == "step3":
        if requested == "pass3":
            raise RuntimeError("step3 does not support pass3; use --pass pass1|pass2|all")
        if requested == "all":
            return list(STEP3_PASS_SEQUENCE)
        return [requested]
    if requested == "all":
        return list(PASS_ORDER)
    return [requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P24 StageI progressive recovery launcher")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=240000)
    parser.add_argument("--step", dest="step_name", choices=STEP_ORDER, default="step1")
    parser.add_argument("--stagei-pass", "--pass", dest="stagei_pass", choices=["all", "pass1", "pass2", "pass3"], default="all")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=6)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"
    gpus = base._parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    selected_models = _selected_model_options(args)
    sg._check_runtime_models(selected_models)

    selected_datasets = _selected_datasets(args)
    step_name = _step_name(args)
    pass_names = _pass_sequence(args)

    if step_name == "step1":
        pass_plans: list[tuple[str, list[Dict[str, Any]], list[Dict[str, Any]]]] = []
        best_rows_snapshot = _load_overall_best_table(datasets=set(selected_datasets), model_options=set(selected_models))
        for pass_name in pass_names:
            targets, rows = _build_rows_for_pass(pass_name, args, best_rows=best_rows_snapshot)
            pass_plans.append((pass_name, targets, rows))
        total_runs = sum(len(rows) for _, _, rows in pass_plans)
        completed_base = 0
        for pass_name, targets, rows in pass_plans:
            _print_pass_plan(pass_name, targets, rows)
            manifest_path = _write_manifest(pass_name, args, rows, targets)
            print(f"[manifest] step={step_name} pass={pass_name} path={manifest_path}")
            _run_rows(
                pass_name,
                args,
                rows,
                global_total_runs=total_runs if total_runs > 0 else None,
                global_done_base=completed_base,
            )
            completed_base += len(rows)
        print(
            f"[All Done] baseline StageI progressive recovery completed: "
            f"step={step_name} passes={','.join(pass_names)} total_runs={total_runs}"
        )
        return 0

    completed_base = 0
    total_runs = 0
    for pass_name in pass_names:
        current_best_rows = _load_overall_best_table(datasets=set(selected_datasets), model_options=set(selected_models))
        targets, rows = _build_rows_for_pass(pass_name, args, best_rows=current_best_rows)
        total_runs += len(rows)
        _print_pass_plan(pass_name, targets, rows)
        manifest_path = _write_manifest(pass_name, args, rows, targets)
        print(f"[manifest] step={step_name} pass={pass_name} path={manifest_path}")
        _run_rows(pass_name, args, rows)
        completed_base += len(rows)

    print(
        f"[All Done] baseline StageI progressive recovery completed: "
        f"step={step_name} passes={','.join(pass_names)} total_runs={completed_base}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
