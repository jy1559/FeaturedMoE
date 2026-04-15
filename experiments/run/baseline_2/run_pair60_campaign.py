#!/usr/bin/env python3
"""Run 60-pair x 3-combo baseline_2 campaign on feature_added_v3 datasets."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

TRACK = "baseline_2"
DEFAULT_AXIS = "PAIR60_V3_LR10"

LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK
RESULT_ROOT = ARTIFACT_ROOT / "results" / TRACK

BASELINE_P14_LOG_ROOT = ARTIFACT_ROOT / "logs" / "baseline" / "Final_all_datasets"
FMOE_P14_LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3" / "Final_all_datasets"
BASELINE2_STAGEA_SUMMARY = ARTIFACT_ROOT / "logs" / TRACK / "ABCD_v2_lean" / "stages" / "stageA" / "summary.csv"

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

MODEL_SPECS: List[Dict[str, str]] = [
    {"model_option": "sasrec", "model_label": "SASRec", "history_label": "SASRec", "history_model": "sasrec", "family": "baseline"},
    {"model_option": "gru4rec", "model_label": "GRU4Rec", "history_label": "GRU4Rec", "history_model": "gru4rec", "family": "baseline"},
    {"model_option": "tisasrec", "model_label": "TiSASRec", "history_label": "TiSASRec", "history_model": "tisasrec", "family": "baseline"},
    {"model_option": "duorec", "model_label": "DuoRec", "history_label": "DuoRec", "history_model": "duorec", "family": "baseline"},
    {"model_option": "bsarec", "model_label": "BSARec", "history_label": "BSARec", "history_model": "bsarec", "family": "baseline"},
    {"model_option": "fearec", "model_label": "FEARec", "history_label": "FEARec", "history_model": "fearec", "family": "baseline"},
    {"model_option": "difsr", "model_label": "DIFSR", "history_label": "DIFSR", "history_model": "difsr", "family": "baseline"},
    {"model_option": "fame", "model_label": "FAME", "history_label": "FAME", "history_model": "fame", "family": "baseline"},
    {"model_option": "fdsa", "model_label": "FDSA", "history_label": "DIFSR", "history_model": "difsr", "family": "baseline"},
    {"model_option": "featured_moe_n3", "model_label": "FeaturedMoE_N3", "history_label": "FeaturedMoE_N3", "history_model": "featured_moe_n3", "family": "fmoe"},
]
MODEL_SPEC_BY_OPTION = {spec["model_option"]: dict(spec) for spec in MODEL_SPECS}
DEFAULT_MODELS = [spec["model_option"] for spec in MODEL_SPECS]

DATASET_CONFIG_MAP = {
    "kuaireclargestrictposv2_0.2": "tune_kuai_strict_small",
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
    "beauty": "tune_ab",
    "foursquare": "tune_fs",
    "movielens1m": "tune_ml",
    "retail_rocket": "tune_rr",
}

HISTORY_FALLBACK_DATASET = {
    "beauty": "amazon_beauty",
}

DATASET_LR_BASE: Dict[str, Tuple[float, float]] = {
    "KuaiRecLargeStrictPosV2_0.2": (2e-4, 8e-3),
    "lastfm0.03": (5e-5, 2e-3),
    "beauty": (5e-5, 6e-3),
    "foursquare": (5e-5, 7e-3),
    "movielens1m": (5e-5, 7e-3),
    "retail_rocket": (5e-5, 7e-3),
}

MODEL_LR_MULT: Dict[str, float] = {
    "sasrec": 1.00,
    "gru4rec": 1.45,
    "tisasrec": 0.78,
    "duorec": 0.55,
    "bsarec": 0.95,
    "fearec": 0.60,
    "difsr": 0.74,
    "fdsa": 0.72,
    "fame": 0.62,
    "featured_moe_n3": 0.88,
}

MODEL_LR_BAND_RATIO: Dict[str, float] = {
    "sasrec": 6.0,
    "gru4rec": 6.0,
    "tisasrec": 4.2,
    "duorec": 4.0,
    "bsarec": 6.0,
    "fearec": 4.0,
    "difsr": 6.0,
    "fdsa": 5.0,
    "fame": 4.0,
    "featured_moe_n3": 4.0,
}

TRANSFORMER_MODELS = {
    "sasrec",
    "tisasrec",
    "duorec",
    "bsarec",
    "fearec",
    "difsr",
    "fdsa",
    "fame",
}

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "model_label",
    "pair_id",
    "combo_id",
    "combo_source",
    "hparam_id",
    "architecture_id",
    "run_phase",
    "run_id",
    "runtime_seed",
    "gpu_id",
    "status",
    "best_valid_mrr20",
    "test_mrr20",
    "valid_unseen_mrr20",
    "valid_unseen_hit20",
    "test_unseen_mrr20",
    "test_unseen_hit20",
    "valid_main_seen_count",
    "valid_main_unseen_count",
    "test_main_seen_count",
    "test_main_unseen_count",
    "lr_lo",
    "lr_hi",
    "result_path",
    "log_path",
    "elapsed_sec",
    "error",
    "timestamp_utc",
    "params_json",
]

MATRIX_FIELDS = [
    "dataset",
    "model_option",
    "model_label",
    "pair_id",
    "combo_id",
    "combo_rank",
    "combo_source",
    "hparam_id",
    "architecture_id",
    "lr_lo",
    "lr_hi",
    "params_json",
]

RUN_STATUS_END_NORMAL_RE = re.compile(r"\[RUN_STATUS\]\s*END\s+status=normal\b", re.IGNORECASE)

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()


@dataclass
class ComboCandidate:
    params: Dict[str, Any]
    score: float
    source: str
    hparam_id: str = ""
    architecture_id: str = ""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_token(text: str, *, upper: bool = False) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    if not token:
        token = "NA"
    return token.upper() if upper else token.lower()


def parse_csv_list(text: str) -> List[str]:
    out: List[str] = []
    for item in str(text or "").split(","):
        t = item.strip()
        if t:
            out.append(t)
    return out


def parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for token in parse_csv_list(text):
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Invalid float for hydra literal: {value}")
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG_MAP:
        raise KeyError(f"Unknown dataset config mapping: {dataset}")
    return DATASET_CONFIG_MAP[key]


def history_dataset_order(dataset: str) -> List[str]:
    ds = str(dataset)
    out = [ds]
    fallback = HISTORY_FALLBACK_DATASET.get(ds)
    if fallback and fallback not in out:
        out.append(fallback)
    return out


def load_module(module_path: Path, module_name: str, prepend_path: Path | None = None) -> Any:
    remove_path = False
    if prepend_path is not None:
        prepend = str(prepend_path)
        if prepend not in sys.path:
            sys.path.insert(0, prepend)
            remove_path = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if remove_path and prepend_path is not None:
            prepend = str(prepend_path)
            if prepend in sys.path:
                sys.path.remove(prepend)


def load_external_modules() -> Tuple[Any, Any, Any]:
    baseline_mod = load_module(
        EXP_DIR / "run" / "baseline" / "run_final_all_datasets.py",
        "baseline_final_all_pair60",
    )
    fmoe_mod = load_module(
        EXP_DIR / "run" / "fmoe_n3" / "run_final_all_datasets.py",
        "fmoe_final_all_pair60",
        prepend_path=EXP_DIR / "run" / "fmoe_n3",
    )
    slack_mod = load_module(
        EXP_DIR / "run" / "common" / "slack_progress.py",
        "slack_progress_pair60",
    )
    return baseline_mod, fmoe_mod, slack_mod


def parse_override_scalar(raw: str) -> Any:
    token = str(raw).strip()
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        if token.startswith("0") and len(token) > 1 and token[1].isdigit():
            raise ValueError
        return int(token)
    except Exception:
        pass
    try:
        return float(token)
    except Exception:
        return token


def parse_plus_overrides(tokens: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for token in tokens:
        t = str(token).strip()
        if not t.startswith("++") or "=" not in t:
            continue
        key, value = t[2:].split("=", 1)
        key = str(key).strip()
        if not key:
            continue
        out[key] = parse_override_scalar(value)
    return out


def params_signature(params: Dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def stage_a_candidate_pool(stage_a_rows: List[Dict[str, str]], dataset: str, model_option: str) -> List[ComboCandidate]:
    ds = str(dataset).strip().lower()
    model = str(model_option).strip().lower()
    grouped: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
    for row in stage_a_rows:
        if str(row.get("dataset", "")).strip().lower() != ds:
            continue
        if str(row.get("model", "")).strip().lower() != model:
            continue
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        params_text = str(row.get("params_json", "")).strip()
        if not params_text:
            continue
        try:
            params = json.loads(params_text)
        except Exception:
            continue
        params.pop("learning_rate", None)
        params.pop("lr_group", None)
        parent_id = str(row.get("parent_candidate_id", "")).strip() or str(row.get("candidate_id", "")).strip()
        score = safe_float(row.get("best_valid_mrr20", 0.0), 0.0)
        prev = grouped.get(parent_id)
        if prev is None or score > prev[0]:
            grouped[parent_id] = (score, params, parent_id)

    out: List[ComboCandidate] = []
    for parent_id, (score, params, cand_id) in grouped.items():
        out.append(
            ComboCandidate(
                params=dict(params),
                score=float(score),
                source=f"baseline2_stageA:{dataset}:{model_option}:{parent_id}",
                hparam_id=str(cand_id),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def _effective_history_model(model_option: str) -> str:
    if str(model_option).lower() == "fdsa":
        return "difsr"
    return str(model_option).lower()


def baseline_hparam_to_params(model_option: str, hparam_id: str, baseline_mod: Any) -> Dict[str, Any]:
    history_model = _effective_history_model(model_option)
    row = {"model_option": history_model, "hparam_id": str(hparam_id).upper()}
    h = dict(baseline_mod._effective_model_hparams(row))

    hidden_size = int(h.get("hidden_size", h.get("embedding_size", 128)))
    embedding_size = int(h.get("embedding_size", hidden_size))
    params: Dict[str, Any] = {
        "max_len": int(h.get("max_len", 20)),
        "hidden_size": hidden_size,
        "embedding_size": embedding_size,
        "num_layers": int(h.get("layers", 2)),
        "num_heads": int(h.get("heads", 2)),
        "inner_size": int(h.get("inner_size", hidden_size * 2)),
        "dropout": float(h.get("dropout", 0.15)),
        "weight_decay": float(h.get("weight_decay", 1e-4)),
    }

    if history_model == "gru4rec":
        params.pop("num_heads", None)
        params.pop("inner_size", None)
        params["num_layers"] = int(h.get("layers", 1))
    if history_model == "tisasrec":
        params["time_span"] = int(h.get("time_span", 256))
    if history_model in {"fame"}:
        params["num_experts"] = int(h.get("num_experts", 3))
    if history_model in {"difsr"}:
        params["attribute_hidden_size"] = hidden_size

    algo_overrides = parse_plus_overrides(baseline_mod._model_algorithm_overrides(row))
    params.update(algo_overrides)

    if str(model_option).lower() == "fdsa":
        params["selected_features"] = ["category"]
        params["pooling_mode"] = "mean"
        params["attribute_hidden_size"] = hidden_size
        params.setdefault("fusion_type", "sum")
        params["use_attribute_predictor"] = True
        params["lambda_attr"] = float(params.get("lambda_attr", 0.1))

    return params


def baseline_history_pool(dataset: str, model_spec: Dict[str, str], baseline_mod: Any) -> List[ComboCandidate]:
    history_label = str(model_spec["history_label"])
    model_option = str(model_spec["model_option"])

    best_by_hparam: Dict[str, Dict[str, Any]] = {}
    for ds in history_dataset_order(dataset):
        summary_path = BASELINE_P14_LOG_ROOT / ds / "summary.csv"
        rows = read_csv(summary_path)
        if not rows:
            continue
        for row in rows:
            if str(row.get("model", "")).strip() != history_label:
                continue
            hparam_id = str(row.get("hparam_id", "")).strip().upper()
            if hparam_id not in baseline_mod.HPARAM_BANK:
                continue
            status = str(row.get("status", "")).strip().lower()
            if "run_complete" not in status and status != "ok":
                continue
            score = safe_float(row.get("run_best_valid_mrr20", row.get("model_best_valid_mrr20", 0.0)), 0.0)
            prev = best_by_hparam.get(hparam_id)
            if prev is None or score > prev["score"]:
                best_by_hparam[hparam_id] = {"score": float(score), "dataset": ds}

    out: List[ComboCandidate] = []
    for hparam_id, meta in best_by_hparam.items():
        params = baseline_hparam_to_params(model_option, hparam_id, baseline_mod)
        out.append(
            ComboCandidate(
                params=params,
                score=float(meta["score"]),
                source=f"baseline_p14:{meta['dataset']}:{history_label}:{hparam_id}",
                hparam_id=hparam_id,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def fmoe_hparam_to_params(dataset: str, hparam_id: str, fmoe_mod: Any, architecture_id: str) -> Dict[str, Any]:
    h = dict(fmoe_mod.HPARAM_BANK[str(hparam_id).upper()])
    max_len_by_dataset = {
        "KuaiRecLargeStrictPosV2_0.2": 20,
        "lastfm0.03": 30,
        "beauty": 20,
        "foursquare": 20,
        "movielens1m": 25,
        "retail_rocket": 20,
    }
    emb = int(h.get("embedding_size", 128))
    params: Dict[str, Any] = {
        "max_len": int(max_len_by_dataset.get(dataset, 20)),
        "embedding_size": emb,
        "hidden_size": emb,
        "num_heads": 4 if emb % 4 == 0 else 2,
        "d_ff": int(h.get("d_ff", emb * 2)),
        "d_expert_hidden": int(h.get("d_expert_hidden", emb)),
        "d_router_hidden": int(h.get("d_router_hidden", max(emb // 2, 32))),
        "d_feat_emb": 12 if emb <= 192 else 16,
        "expert_scale": 2 if emb <= 160 else 3,
        "dropout": float(h.get("fixed_hidden_dropout_prob", 0.15)),
        "weight_decay": float(h.get("fixed_weight_decay", 1e-6)),
        "fmoe_v2_layout_id": 0,
        "fmoe_stage_execution_mode": "serial",
        "router_impl": "learned",
        "router_use_hidden": True,
        "router_use_feature": True,
        "fmoe_architecture_id": str(architecture_id).upper(),
    }
    return params


def fmoe_history_pool(dataset: str, fmoe_mod: Any) -> List[ComboCandidate]:
    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ds in history_dataset_order(dataset):
        summary_path = FMOE_P14_LOG_ROOT / ds / "summary.csv"
        rows = read_csv(summary_path)
        if not rows:
            continue
        for row in rows:
            hparam_id = str(row.get("hparam_id", "")).strip().upper()
            if hparam_id not in fmoe_mod.HPARAM_BANK:
                continue
            status = str(row.get("status", "")).strip().lower()
            trigger = str(row.get("trigger", "")).strip().lower()
            if "run_complete" not in status and "run_complete" not in trigger:
                continue
            arch_id = str(row.get("architecture_id", "A1")).strip().upper() or "A1"
            score = safe_float(row.get("run_best_valid_mrr20", row.get("global_best_valid_mrr20", 0.0)), 0.0)
            key = (arch_id, hparam_id)
            prev = best_by_key.get(key)
            if prev is None or score > prev["score"]:
                best_by_key[key] = {"score": float(score), "dataset": ds}

    out: List[ComboCandidate] = []
    for (arch_id, hparam_id), meta in best_by_key.items():
        params = fmoe_hparam_to_params(dataset, hparam_id, fmoe_mod, arch_id)
        out.append(
            ComboCandidate(
                params=params,
                score=float(meta["score"]),
                source=f"fmoe_p14:{meta['dataset']}:{arch_id}:{hparam_id}",
                hparam_id=hparam_id,
                architecture_id=arch_id,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def default_baseline_candidates(dataset: str, model_option: str, baseline_mod: Any) -> List[ComboCandidate]:
    history_model = _effective_history_model(model_option)
    priority = list(baseline_mod.MODEL_HPARAM_PRIORITY.get(history_model, []))
    if not priority:
        priority = sorted(baseline_mod.HPARAM_BANK.keys())
    out: List[ComboCandidate] = []
    for rank, hparam_id in enumerate(priority[:4], start=1):
        if hparam_id not in baseline_mod.HPARAM_BANK:
            continue
        params = baseline_hparam_to_params(model_option, hparam_id, baseline_mod)
        out.append(
            ComboCandidate(
                params=params,
                score=float(-rank),
                source=f"fallback_baseline:{dataset}:{model_option}:{hparam_id}",
                hparam_id=hparam_id,
            )
        )
    return out


def default_fmoe_candidates(dataset: str, fmoe_mod: Any) -> List[ComboCandidate]:
    preset_key = dataset if dataset in fmoe_mod.DATASET_HPARAM_PRESET_12 else HISTORY_FALLBACK_DATASET.get(dataset, dataset)
    preset = list(fmoe_mod.DATASET_HPARAM_PRESET_12.get(preset_key, []))
    if not preset:
        preset = sorted(fmoe_mod.HPARAM_BANK.keys())
    out: List[ComboCandidate] = []
    for rank, hparam_id in enumerate(preset[:4], start=1):
        if hparam_id not in fmoe_mod.HPARAM_BANK:
            continue
        arch_id = "A2" if rank % 2 == 0 else "A1"
        params = fmoe_hparam_to_params(dataset, hparam_id, fmoe_mod, arch_id)
        out.append(
            ComboCandidate(
                params=params,
                score=float(-rank),
                source=f"fallback_fmoe:{dataset}:{arch_id}:{hparam_id}",
                hparam_id=hparam_id,
                architecture_id=arch_id,
            )
        )
    return out


def select_top_unique(pool: List[ComboCandidate], n: int) -> List[ComboCandidate]:
    seen = set()
    out: List[ComboCandidate] = []
    for cand in sorted(pool, key=lambda c: (c.score, c.source), reverse=True):
        sig = params_signature(cand.params)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def build_exploration_params(dataset: str, model_option: str, top1: Dict[str, Any], top2: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(top1)
    model = str(model_option).lower()
    ds = str(dataset)

    if model in TRANSFORMER_MODELS:
        max_len = safe_int(p.get("max_len", 20), 20)
        if ds in {"lastfm0.03", "movielens1m"}:
            max_len += 10
        elif ds == "KuaiRecLargeStrictPosV2_0.2":
            max_len += 5
        else:
            max_len -= 5
        p["max_len"] = clamp_int(max_len, 20, 100)

        dropout = safe_float(p.get("dropout", 0.15), 0.15)
        if ds in {"beauty", "foursquare", "retail_rocket"}:
            dropout += 0.03
        else:
            dropout -= 0.02
        p["dropout"] = round(clamp_float(dropout, 0.08, 0.35), 4)

        if "num_layers" in p:
            layers = safe_int(p.get("num_layers", 2), 2)
            if ds in {"lastfm0.03", "movielens1m", "KuaiRecLargeStrictPosV2_0.2"}:
                layers += 1
            else:
                layers = max(1, layers - 1)
            p["num_layers"] = clamp_int(layers, 1, 4)

        if model == "tisasrec":
            span = safe_int(p.get("time_span", 256), 256)
            p["time_span"] = clamp_int(int(span * 1.5) if ds in {"lastfm0.03", "movielens1m"} else int(span * 0.8), 128, 1024)

    if model == "gru4rec":
        p["num_layers"] = 2 if safe_int(p.get("num_layers", 1), 1) == 1 else 1
        dropout = safe_float(p.get("dropout", 0.2), 0.2)
        p["dropout"] = round(clamp_float(dropout + 0.03, 0.1, 0.4), 4)
        max_len = safe_int(p.get("max_len", 20), 20)
        p["max_len"] = clamp_int(max_len + (10 if ds == "lastfm0.03" else -5), 20, 80)

    if model in {"duorec", "fearec"}:
        cycle = ["un", "su", "us_x"]
        cur = str(p.get("contrast", "un")).lower()
        if cur not in cycle:
            cur = "un"
        p["contrast"] = cycle[(cycle.index(cur) + 1) % len(cycle)]
        p["tau"] = round(clamp_float(safe_float(p.get("tau", 0.2), 0.2) * 1.2, 0.08, 1.0), 4)
        if p["contrast"] == "su":
            p["lmd"] = 0.0
            p["lmd_sem"] = round(clamp_float(safe_float(p.get("lmd_sem", 0.06), 0.06) + 0.02, 0.0, 0.2), 4)
        elif p["contrast"] == "un":
            p["lmd"] = round(clamp_float(safe_float(p.get("lmd", 0.04), 0.04) + 0.02, 0.0, 0.2), 4)
            p["lmd_sem"] = 0.0

    if model in {"difsr", "fdsa"}:
        cycle = ["sum", "gate", "concat"]
        cur = str(p.get("fusion_type", "sum")).lower()
        if cur not in cycle:
            cur = "sum"
        p["fusion_type"] = cycle[(cycle.index(cur) + 1) % len(cycle)]
        p["use_attribute_predictor"] = True
        p["lambda_attr"] = round(clamp_float(safe_float(p.get("lambda_attr", 0.1), 0.1) + 0.03, 0.02, 0.3), 4)
        if model == "fdsa":
            p["selected_features"] = ["category"]
            p["pooling_mode"] = "mean"

    if model == "fame":
        experts = safe_int(p.get("num_experts", 3), 3)
        p["num_experts"] = clamp_int(experts + 1, 2, 6)

    if model == "featured_moe_n3":
        max_len = safe_int(p.get("max_len", 20), 20)
        p["max_len"] = clamp_int(max_len + (10 if ds == "lastfm0.03" else 5), 20, 60)
        scale = safe_int(p.get("expert_scale", 2), 2)
        p["expert_scale"] = clamp_int(scale + 1, 2, 4)
        d_feat = safe_int(p.get("d_feat_emb", 12), 12)
        p["d_feat_emb"] = 16 if d_feat <= 12 else 12
        d_router = safe_int(p.get("d_router_hidden", 64), 64)
        p["d_router_hidden"] = clamp_int(int(d_router * 1.25), 40, 160)

    if params_signature(p) in {params_signature(top1), params_signature(top2)}:
        p["weight_decay"] = round(clamp_float(safe_float(p.get("weight_decay", 1e-4), 1e-4) * 1.5, 1e-7, 5e-3), 8)

    return p


def compute_lr_bounds(dataset: str, model_option: str, combo_rank: int, params: Dict[str, Any]) -> Tuple[float, float]:
    base_lo, base_hi = DATASET_LR_BASE[str(dataset)]
    center = math.sqrt(base_lo * base_hi)
    center *= float(MODEL_LR_MULT.get(str(model_option), 1.0))
    center *= {1: 1.0, 2: 0.88, 3: 1.12}.get(int(combo_rank), 1.0)

    max_len = safe_int(params.get("max_len", 20), 20)
    if max_len >= 50:
        center *= 0.9

    dropout = safe_float(params.get("dropout", 0.15), 0.15)
    if dropout >= 0.2:
        center *= 0.9

    ratio = float(MODEL_LR_BAND_RATIO.get(str(model_option), 6.0))
    lo = max(2e-5, center / math.sqrt(ratio))
    hi = min(8e-3, center * math.sqrt(ratio))
    if hi <= lo:
        hi = min(8e-3, lo * 2.5)
    return float(lo), float(hi)


def build_pair_combos(
    dataset: str,
    model_spec: Dict[str, str],
    stage_a_rows: List[Dict[str, str]],
    baseline_mod: Any,
    fmoe_mod: Any,
) -> List[Dict[str, Any]]:
    model_option = str(model_spec["model_option"])
    family = str(model_spec["family"])

    pool: List[ComboCandidate] = []
    if family == "fmoe":
        pool.extend(fmoe_history_pool(dataset, fmoe_mod))
        pool.extend(default_fmoe_candidates(dataset, fmoe_mod))
    else:
        pool.extend(stage_a_candidate_pool(stage_a_rows, dataset, model_option))
        pool.extend(baseline_history_pool(dataset, model_spec, baseline_mod))
        pool.extend(default_baseline_candidates(dataset, model_option, baseline_mod))

    top2 = select_top_unique(pool, 2)
    if len(top2) < 2:
        # Ensure two non-identical base combos are always available.
        defaults = default_fmoe_candidates(dataset, fmoe_mod) if family == "fmoe" else default_baseline_candidates(dataset, model_option, baseline_mod)
        for cand in defaults:
            sig = params_signature(cand.params)
            if all(params_signature(x.params) != sig for x in top2):
                top2.append(cand)
            if len(top2) >= 2:
                break

    while len(top2) < 2:
        top2.append(top2[0])

    explore_params = build_exploration_params(dataset, model_option, top2[0].params, top2[1].params)

    combos = [
        {
            "combo_id": "C1",
            "combo_rank": 1,
            "combo_source": top2[0].source,
            "hparam_id": top2[0].hparam_id,
            "architecture_id": top2[0].architecture_id,
            "params": dict(top2[0].params),
        },
        {
            "combo_id": "C2",
            "combo_rank": 2,
            "combo_source": top2[1].source,
            "hparam_id": top2[1].hparam_id,
            "architecture_id": top2[1].architecture_id,
            "params": dict(top2[1].params),
        },
        {
            "combo_id": "C3",
            "combo_rank": 3,
            "combo_source": f"explore_from:{top2[0].source}",
            "hparam_id": top2[0].hparam_id,
            "architecture_id": top2[0].architecture_id,
            "params": dict(explore_params),
        },
    ]

    seen = set()
    for combo in combos:
        sig = params_signature(combo["params"])
        if sig in seen:
            combo["params"]["weight_decay"] = round(
                clamp_float(safe_float(combo["params"].get("weight_decay", 1e-4), 1e-4) * 1.3, 1e-7, 5e-3),
                8,
            )
            sig = params_signature(combo["params"])
        seen.add(sig)

    return combos


def build_matrix_rows(
    datasets: List[str],
    model_options: List[str],
    stage_a_rows: List[Dict[str, str]],
    baseline_mod: Any,
    fmoe_mod: Any,
) -> List[Dict[str, Any]]:
    matrix_rows: List[Dict[str, Any]] = []
    pair_cursor = 0
    for dataset in datasets:
        for model_option in model_options:
            pair_cursor += 1
            model_spec = MODEL_SPEC_BY_OPTION[model_option]
            pair_id = f"P{pair_cursor:03d}"
            combos = build_pair_combos(dataset, model_spec, stage_a_rows, baseline_mod, fmoe_mod)
            for combo in combos:
                lo, hi = compute_lr_bounds(dataset, model_option, int(combo["combo_rank"]), combo["params"])
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "model_option": model_option,
                        "model_label": model_spec["model_label"],
                        "pair_id": pair_id,
                        "combo_id": combo["combo_id"],
                        "combo_rank": int(combo["combo_rank"]),
                        "combo_source": combo["combo_source"],
                        "hparam_id": combo["hparam_id"],
                        "architecture_id": combo["architecture_id"],
                        "lr_lo": float(lo),
                        "lr_hi": float(hi),
                        "params_json": json.dumps(combo["params"], ensure_ascii=False, sort_keys=True),
                    }
                )
    return matrix_rows


def build_run_rows(matrix_rows: List[Dict[str, Any]], axis: str, seeds: List[int], runtime_seed_base: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for item in matrix_rows:
        for seed in seeds:
            cursor += 1
            runtime_seed = int(runtime_seed_base) + cursor - 1
            ds_tag = sanitize_token(item["dataset"], upper=True)
            model_tag = sanitize_token(item["model_option"], upper=True)
            pair_tag = sanitize_token(item["pair_id"], upper=True)
            combo_tag = sanitize_token(item["combo_id"], upper=True)
            run_phase = f"{axis}_D{ds_tag}_M{model_tag}_{pair_tag}_{combo_tag}_S{int(seed)}"
            run_id = f"R_{ds_tag}_{model_tag}_{pair_tag}_{combo_tag}_S{int(seed)}"
            rows.append(
                {
                    **item,
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "seed_id": int(seed),
                    "runtime_seed": int(runtime_seed),
                }
            )
    return rows


def resolve_log_path(axis_root: Path, row: Dict[str, Any]) -> Path:
    ds = sanitize_token(row["dataset"], upper=False)
    model = sanitize_token(row["model_option"], upper=False)
    combo = sanitize_token(row["combo_id"], upper=True)
    return axis_root / ds / model / f"combo_{combo}" / "logs" / f"{row['run_phase']}.log"


def metrics_from_special(special_payload: Dict[str, Any]) -> Dict[str, float]:
    out = {
        "unseen_mrr20": 0.0,
        "unseen_hit20": 0.0,
        "unseen_count": 0.0,
        "seen_count": 0.0,
    }
    if not isinstance(special_payload, dict):
        return out
    overall = special_payload.get("overall", {}) or {}
    slices = special_payload.get("slices", {}) or {}
    pop = slices.get("target_popularity_abs", {}) or {}
    cold = pop.get("cold_0", {}) or {}
    total = float(overall.get("count", 0.0) or 0.0)
    cold_count = float(cold.get("count", 0.0) or 0.0)
    out["unseen_mrr20"] = float(cold.get("mrr@20", 0.0) or 0.0)
    out["unseen_hit20"] = float(cold.get("hit@20", 0.0) or 0.0)
    out["unseen_count"] = cold_count
    out["seen_count"] = max(0.0, total - cold_count)
    return out


def parse_result_metrics(result_path: Path) -> Dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    best_valid_result = payload.get("best_valid_result", {}) or {}
    test_result = payload.get("test_result", {}) or {}
    best_valid_special = payload.get("best_valid_special_metrics", {}) or {}
    test_special = payload.get("test_special_metrics", {}) or {}

    valid_filter = payload.get("best_valid_main_eval_filter", {}) or {}
    test_filter = payload.get("test_main_eval_filter", {}) or {}
    valid_cold = payload.get("best_valid_cold_target_metrics", {}) or {}
    test_cold = payload.get("test_cold_target_metrics", {}) or {}

    if not valid_cold:
        v = metrics_from_special(best_valid_special)
        valid_cold = {"mrr@20": v["unseen_mrr20"], "hit@20": v["unseen_hit20"], "count": v["unseen_count"]}
    if not test_cold:
        t = metrics_from_special(test_special)
        test_cold = {"mrr@20": t["unseen_mrr20"], "hit@20": t["unseen_hit20"], "count": t["unseen_count"]}

    if not valid_filter:
        v = metrics_from_special(best_valid_special)
        valid_filter = {
            "seen_targets": int(v["seen_count"]),
            "unseen_targets": int(v["unseen_count"]),
        }
    if not test_filter:
        t = metrics_from_special(test_special)
        test_filter = {
            "seen_targets": int(t["seen_count"]),
            "unseen_targets": int(t["unseen_count"]),
        }

    return {
        "best_valid_mrr20": float(best_valid_result.get("mrr@20", payload.get("best_mrr@20", 0.0)) or 0.0),
        "test_mrr20": float(test_result.get("mrr@20", payload.get("test_mrr@20", 0.0)) or 0.0),
        "valid_unseen_mrr20": float(valid_cold.get("mrr@20", 0.0) or 0.0),
        "valid_unseen_hit20": float(valid_cold.get("hit@20", 0.0) or 0.0),
        "test_unseen_mrr20": float(test_cold.get("mrr@20", 0.0) or 0.0),
        "test_unseen_hit20": float(test_cold.get("hit@20", 0.0) or 0.0),
        "valid_main_seen_count": int(valid_filter.get("seen_targets", 0) or 0),
        "valid_main_unseen_count": int(valid_filter.get("unseen_targets", 0) or 0),
        "test_main_seen_count": int(test_filter.get("seen_targets", 0) or 0),
        "test_main_unseen_count": int(test_filter.get("unseen_targets", 0) or 0),
    }


def parse_result_path_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    pat = re.compile(r"Results\s*->\s*(.+)$")
    for line in reversed(lines):
        m = pat.search(line.strip())
        if m:
            return Path(m.group(1).strip()).expanduser()
    return None


def has_run_status_end_normal(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        token = str(line).strip()
        if not token:
            continue
        return bool(RUN_STATUS_END_NORMAL_RE.search(token))
    return False


def _active_proc_add(proc: subprocess.Popen[Any]) -> None:
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.add(proc)


def _active_proc_remove(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def _terminate_process(proc: subprocess.Popen[Any], sig_num: int) -> None:
    try:
        if proc.poll() is not None:
            return
        if int(sig_num) == int(signal.SIGKILL):
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        return


def terminate_active_children(*, grace_sec: float = 0.4) -> None:
    with ACTIVE_PROCESS_LOCK:
        procs = list(ACTIVE_PROCESSES)
    if not procs:
        return

    for proc in procs:
        _terminate_process(proc, signal.SIGTERM)
    if grace_sec > 0:
        time.sleep(float(grace_sec))
    for proc in procs:
        _terminate_process(proc, signal.SIGKILL)


def install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        first = not STOP_EVENT.is_set()
        STOP_EVENT.set()
        if first:
            sig_name = str(signum)
            try:
                sig_name = signal.Signals(signum).name
            except Exception:
                pass
            print(f"[launch] interrupt signal={sig_name} -> terminating active process groups")
            terminate_active_children(grace_sec=0.2)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def extract_error_tail(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for line in reversed(lines[-80:]):
        text = line.strip()
        if text:
            return text[:400]
    return ""


def _append_override(cmd: List[str], key: str, value: Any) -> None:
    cmd.append(f"++{key}={hydra_literal(value)}")


def base_runtime_overrides(model_option: str, params: Dict[str, Any]) -> List[str]:
    model = str(model_option).lower()
    out: List[str] = [
        f"++MAX_ITEM_LIST_LENGTH={safe_int(params.get('max_len', 20), 20)}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
    ]

    if model in TRANSFORMER_MODELS:
        hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
        embed = safe_int(params.get("embedding_size", hidden), hidden)
        layers = safe_int(params.get("num_layers", 2), 2)
        heads = safe_int(params.get("num_heads", 2), 2)
        inner = safe_int(params.get("inner_size", hidden * 2), hidden * 2)
        dropout = safe_float(params.get("dropout", 0.15), 0.15)

        out.extend(
            [
                f"++hidden_size={hidden}",
                f"++embedding_size={embed}",
                f"++n_layers={layers}",
                f"++num_layers={layers}",
                f"++n_heads={heads}",
                f"++num_heads={heads}",
                f"++inner_size={inner}",
                f"++dropout_ratio={dropout}",
                f"++hidden_dropout_prob={dropout}",
                f"++attn_dropout_prob={dropout}",
                f"++weight_decay={safe_float(params.get('weight_decay', 1e-4), 1e-4)}",
            ]
        )

        if model == "tisasrec":
            out.append(f"++time_span={safe_int(params.get('time_span', 256), 256)}")

        if model in {"duorec", "fearec"}:
            if "contrast" in params:
                _append_override(out, "contrast", params["contrast"])
            if "tau" in params:
                _append_override(out, "tau", safe_float(params.get("tau", 0.2), 0.2))
            if "lmd" in params:
                _append_override(out, "lmd", safe_float(params.get("lmd", 0.04), 0.04))
            if "lmd_sem" in params:
                _append_override(out, "lmd_sem", safe_float(params.get("lmd_sem", 0.0), 0.0))
            if "semantic_sample_max_tries" in params:
                _append_override(out, "semantic_sample_max_tries", safe_int(params.get("semantic_sample_max_tries", 2), 2))
            if "global_ratio" in params:
                _append_override(out, "global_ratio", safe_float(params.get("global_ratio", 0.9), 0.9))

        if model in {"difsr", "fdsa"}:
            _append_override(out, "attribute_hidden_size", safe_int(params.get("attribute_hidden_size", hidden), hidden))
            if "fusion_type" in params:
                _append_override(out, "fusion_type", params["fusion_type"])
            if "use_attribute_predictor" in params:
                _append_override(out, "use_attribute_predictor", bool(params["use_attribute_predictor"]))
            if "lambda_attr" in params:
                _append_override(out, "lambda_attr", safe_float(params.get("lambda_attr", 0.1), 0.1))

        if model == "fdsa":
            _append_override(out, "selected_features", params.get("selected_features", ["category"]))
            _append_override(out, "pooling_mode", params.get("pooling_mode", "mean"))

        if model == "fame":
            _append_override(out, "num_experts", safe_int(params.get("num_experts", 3), 3))

    elif model == "gru4rec":
        hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
        layers = safe_int(params.get("num_layers", 1), 1)
        dropout = safe_float(params.get("dropout", 0.2), 0.2)
        out.extend(
            [
                f"++hidden_size={hidden}",
                f"++embedding_size={hidden}",
                f"++num_layers={layers}",
                f"++dropout_prob={dropout}",
                f"++hidden_dropout_prob={dropout}",
                f"++weight_decay={safe_float(params.get('weight_decay', 1e-4), 1e-4)}",
            ]
        )

    elif model == "featured_moe_n3":
        emb = safe_int(params.get("embedding_size", 128), 128)
        heads = safe_int(params.get("num_heads", 4), 4)
        out.extend(
            [
                f"++embedding_size={emb}",
                f"++hidden_size={emb}",
                f"++num_heads={heads}",
                f"++d_ff={safe_int(params.get('d_ff', emb * 2), emb * 2)}",
                f"++d_expert_hidden={safe_int(params.get('d_expert_hidden', emb), emb)}",
                f"++d_router_hidden={safe_int(params.get('d_router_hidden', max(emb // 2, 32)), max(emb // 2, 32))}",
                f"++d_feat_emb={safe_int(params.get('d_feat_emb', 12), 12)}",
                f"++expert_scale={safe_int(params.get('expert_scale', 2), 2)}",
                f"++fixed_hidden_dropout_prob={safe_float(params.get('dropout', 0.15), 0.15)}",
                f"++fixed_weight_decay={safe_float(params.get('weight_decay', 1e-6), 1e-6)}",
                f"++fmoe_v2_layout_id={safe_int(params.get('fmoe_v2_layout_id', 0), 0)}",
                f"++fmoe_stage_execution_mode={hydra_literal(params.get('fmoe_stage_execution_mode', 'serial'))}",
                f"++router_impl={hydra_literal(params.get('router_impl', 'learned'))}",
                f"++router_use_hidden={hydra_literal(bool(params.get('router_use_hidden', True)))}",
                f"++router_use_feature={hydra_literal(bool(params.get('router_use_feature', True)))}",
                "++fmoe_special_logging=true",
                "++fmoe_diag_logging=true",
            ]
        )
        if "fmoe_architecture_id" in params:
            _append_override(out, "fmoe_architecture_id", str(params["fmoe_architecture_id"]))

    return out


def fixed_search_entries(model_option: str, params: Dict[str, Any]) -> Dict[str, Any]:
    model = str(model_option).lower()
    fixed: Dict[str, Any] = {
        "MAX_ITEM_LIST_LENGTH": safe_int(params.get("max_len", 20), 20),
        "weight_decay": safe_float(params.get("weight_decay", 1e-4), 1e-4),
    }

    if model in TRANSFORMER_MODELS:
        hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
        embed = safe_int(params.get("embedding_size", hidden), hidden)
        layers = safe_int(params.get("num_layers", 2), 2)
        heads = safe_int(params.get("num_heads", 2), 2)
        inner = safe_int(params.get("inner_size", hidden * 2), hidden * 2)
        dropout = safe_float(params.get("dropout", 0.15), 0.15)

        fixed.update(
            {
                "hidden_size": hidden,
                "embedding_size": embed,
                "n_layers": layers,
                "num_layers": layers,
                "n_heads": heads,
                "num_heads": heads,
                "inner_size": inner,
                "dropout_ratio": dropout,
                "hidden_dropout_prob": dropout,
                "attn_dropout_prob": dropout,
            }
        )

        if model == "tisasrec":
            fixed["time_span"] = safe_int(params.get("time_span", 256), 256)

        if model in {"duorec", "fearec"}:
            if "contrast" in params:
                fixed["contrast"] = params["contrast"]
            if "tau" in params:
                fixed["tau"] = safe_float(params.get("tau", 0.2), 0.2)
            if "lmd" in params:
                fixed["lmd"] = safe_float(params.get("lmd", 0.04), 0.04)
            if "lmd_sem" in params:
                fixed["lmd_sem"] = safe_float(params.get("lmd_sem", 0.0), 0.0)
            if "global_ratio" in params:
                fixed["global_ratio"] = safe_float(params.get("global_ratio", 1.0), 1.0)
            if "semantic_sample_max_tries" in params:
                fixed["semantic_sample_max_tries"] = safe_int(params.get("semantic_sample_max_tries", 2), 2)

        if model in {"difsr", "fdsa"}:
            fixed["attribute_hidden_size"] = safe_int(params.get("attribute_hidden_size", hidden), hidden)
            if "fusion_type" in params:
                fixed["fusion_type"] = params["fusion_type"]
            if "use_attribute_predictor" in params:
                fixed["use_attribute_predictor"] = bool(params["use_attribute_predictor"])
            if "lambda_attr" in params:
                fixed["lambda_attr"] = safe_float(params.get("lambda_attr", 0.1), 0.1)

        if model == "fdsa":
            fixed["selected_features"] = params.get("selected_features", ["category"])
            fixed["pooling_mode"] = params.get("pooling_mode", "mean")

        if model == "fame":
            fixed["num_experts"] = safe_int(params.get("num_experts", 3), 3)

    elif model == "gru4rec":
        hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
        layers = safe_int(params.get("num_layers", 1), 1)
        dropout = safe_float(params.get("dropout", 0.2), 0.2)
        fixed.update(
            {
                "hidden_size": hidden,
                "embedding_size": hidden,
                "num_layers": layers,
                "dropout_prob": dropout,
                "hidden_dropout_prob": dropout,
            }
        )

    elif model == "featured_moe_n3":
        emb = safe_int(params.get("embedding_size", 128), 128)
        fixed.update(
            {
                "embedding_size": emb,
                "hidden_size": emb,
                "num_heads": safe_int(params.get("num_heads", 4), 4),
                "d_ff": safe_int(params.get("d_ff", emb * 2), emb * 2),
                "d_expert_hidden": safe_int(params.get("d_expert_hidden", emb), emb),
                "d_router_hidden": safe_int(params.get("d_router_hidden", max(emb // 2, 32)), max(emb // 2, 32)),
                "d_feat_emb": safe_int(params.get("d_feat_emb", 12), 12),
                "expert_scale": safe_int(params.get("expert_scale", 2), 2),
                "fixed_hidden_dropout_prob": safe_float(params.get("dropout", 0.15), 0.15),
                "fixed_weight_decay": safe_float(params.get("weight_decay", 1e-6), 1e-6),
                "fmoe_v2_layout_id": safe_int(params.get("fmoe_v2_layout_id", 0), 0),
            }
        )
        if "fmoe_architecture_id" in params:
            fixed["fmoe_architecture_id"] = str(params["fmoe_architecture_id"])

    return fixed


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> List[str]:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"])
    params = json.loads(str(row["params_json"]))

    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        dataset_config_name(dataset),
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--search-algo",
        str(args.search_algo),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        str(args.axis),
        "--run-phase",
        str(row["run_phase"]),
        f"model={model_option}",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        f"++seed={int(row['runtime_seed'])}",
    ]

    cmd.extend(base_runtime_overrides(model_option, params))

    lr_range = [float(row["lr_lo"]), float(row["lr_hi"])]
    cmd.append(f"++search={hydra_literal({'learning_rate': lr_range})}")
    cmd.append(f"++search_space_type_overrides={hydra_literal({'learning_rate': 'loguniform'})}")

    fixed = fixed_search_entries(model_option, params)
    for key, value in fixed.items():
        cmd.append(f"++search.{key}={hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    return cmd


def build_summary_row(
    *,
    row: Dict[str, Any],
    gpu_id: str,
    status: str,
    metrics: Dict[str, Any],
    result_path: str,
    log_path: Path,
    elapsed_sec: float,
    error: str,
) -> Dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "model": row["model_option"],
        "model_label": row["model_label"],
        "pair_id": row["pair_id"],
        "combo_id": row["combo_id"],
        "combo_source": row["combo_source"],
        "hparam_id": row.get("hparam_id", ""),
        "architecture_id": row.get("architecture_id", ""),
        "run_phase": row["run_phase"],
        "run_id": row["run_id"],
        "runtime_seed": int(row["runtime_seed"]),
        "gpu_id": str(gpu_id),
        "status": status,
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "valid_unseen_mrr20": metrics.get("valid_unseen_mrr20", 0.0),
        "valid_unseen_hit20": metrics.get("valid_unseen_hit20", 0.0),
        "test_unseen_mrr20": metrics.get("test_unseen_mrr20", 0.0),
        "test_unseen_hit20": metrics.get("test_unseen_hit20", 0.0),
        "valid_main_seen_count": metrics.get("valid_main_seen_count", 0),
        "valid_main_unseen_count": metrics.get("valid_main_unseen_count", 0),
        "test_main_seen_count": metrics.get("test_main_seen_count", 0),
        "test_main_unseen_count": metrics.get("test_main_unseen_count", 0),
        "lr_lo": float(row["lr_lo"]),
        "lr_hi": float(row["lr_hi"]),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "elapsed_sec": float(elapsed_sec),
        "error": str(error or ""),
        "timestamp_utc": now_utc(),
        "params_json": row["params_json"],
    }


def run_one(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, axis_root: Path) -> Dict[str, Any]:
    log_path = resolve_log_path(axis_root, row)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(row, gpu_id, args)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    start = time.time()
    rc = 1
    proc: subprocess.Popen[Any] | None = None
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# pair={row['pair_id']} dataset={row['dataset']} model={row['model_option']} "
            f"combo={row['combo_id']} seed={row['seed_id']}\n"
        )
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXP_DIR),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _active_proc_add(proc)

        try:
            while True:
                if STOP_EVENT.is_set() and proc.poll() is None:
                    _terminate_process(proc, signal.SIGTERM)
                    time.sleep(0.3)
                    if proc.poll() is None:
                        _terminate_process(proc, signal.SIGKILL)
                polled = proc.poll()
                if polled is not None:
                    rc = int(polled)
                    break
                time.sleep(0.2)
        finally:
            _active_proc_remove(proc)
    elapsed = time.time() - start

    result_path_obj = parse_result_path_from_log(log_path)
    metrics = parse_result_metrics(result_path_obj) if result_path_obj is not None else {}
    normal_end = has_run_status_end_normal(log_path)

    status = "ok" if (rc == 0 and (normal_end or result_path_obj is not None)) else "fail"
    if status == "ok":
        error = ""
    else:
        interrupt_prefix = "interrupted " if STOP_EVENT.is_set() else ""
        error = f"{interrupt_prefix}rc={rc} tail={extract_error_tail(log_path)}"

    return build_summary_row(
        row=row,
        gpu_id=gpu_id,
        status=status,
        metrics=metrics,
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=elapsed,
        error=error,
    )


def build_resumed_row(row: Dict[str, Any], axis_root: Path) -> Dict[str, Any] | None:
    log_path = resolve_log_path(axis_root, row)
    if not has_run_status_end_normal(log_path):
        return None
    result_path_obj = parse_result_path_from_log(log_path)
    metrics = parse_result_metrics(result_path_obj) if result_path_obj is not None else {}
    return build_summary_row(
        row=row,
        gpu_id="resume",
        status="ok",
        metrics=metrics,
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=0.0,
        error="",
    )


def read_existing_summary(summary_csv: Path) -> Dict[str, Dict[str, str]]:
    existing: Dict[str, Dict[str, str]] = {}
    for row in read_csv(summary_csv):
        run_phase = str(row.get("run_phase", "")).strip()
        if run_phase:
            existing[run_phase] = dict(row)
    return existing


def rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        safe_float(row.get("best_valid_mrr20", 0.0), 0.0),
        safe_float(row.get("test_mrr20", 0.0), 0.0),
        safe_float(row.get("test_unseen_mrr20", 0.0), 0.0),
    )


def build_leaderboard(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        grouped.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, model), items in grouped.items():
        best = sorted(items, key=rank_key, reverse=True)[0]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "model_label": best.get("model_label", ""),
                "pair_id": best.get("pair_id", ""),
                "combo_id": best.get("combo_id", ""),
                "best_valid_mrr20": best.get("best_valid_mrr20", 0.0),
                "test_mrr20": best.get("test_mrr20", 0.0),
                "test_unseen_mrr20": best.get("test_unseen_mrr20", 0.0),
                "test_main_seen_count": best.get("test_main_seen_count", 0),
                "test_main_unseen_count": best.get("test_main_unseen_count", 0),
                "result_path": best.get("result_path", ""),
            }
        )
    out.sort(key=lambda r: (str(r["dataset"]), str(r["model"])))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", type=str, default=DEFAULT_AXIS)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--seeds", type=str, default="1")
    parser.add_argument("--runtime-seed-base", type=int, default=1)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--search-algo", type=str, default="tpe")

    parser.add_argument("--resume-from-logs", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--slack-progress-step", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    STOP_EVENT.clear()
    install_signal_handlers()

    datasets = parse_csv_list(args.datasets)
    if not datasets:
        datasets = list(DEFAULT_DATASETS)

    model_options = [m.lower() for m in parse_csv_list(args.models)]
    if not model_options:
        model_options = list(DEFAULT_MODELS)
    for model in model_options:
        if model not in MODEL_SPEC_BY_OPTION:
            raise RuntimeError(f"Unsupported model option: {model}")

    gpus = parse_csv_list(args.gpus)
    if not gpus:
        gpus = ["0"]

    seeds = parse_csv_ints(args.seeds)
    if not seeds:
        seeds = [1]

    baseline_mod, fmoe_mod, slack_mod = load_external_modules()
    SlackProgressNotifier = getattr(slack_mod, "SlackProgressNotifier")

    stage_a_rows = read_csv(BASELINE2_STAGEA_SUMMARY)
    matrix_rows = build_matrix_rows(datasets, model_options, stage_a_rows, baseline_mod, fmoe_mod)

    axis_root = LOG_ROOT / str(args.axis)
    axis_root.mkdir(parents=True, exist_ok=True)
    matrix_csv = axis_root / "combo_matrix.csv"
    write_csv(matrix_csv, matrix_rows, MATRIX_FIELDS)

    run_rows = build_run_rows(matrix_rows, str(args.axis), seeds, int(args.runtime_seed_base))
    if int(args.max_runs) > 0:
        run_rows = run_rows[: int(args.max_runs)]

    summary_csv = axis_root / "summary.csv"
    existing_by_run_phase = read_existing_summary(summary_csv) if args.resume_from_logs else {}

    all_rows: List[Dict[str, Any]] = []
    precompleted_rows: List[Dict[str, Any]] = []
    pending_jobs: List[Dict[str, Any]] = []

    for row in run_rows:
        run_phase = str(row["run_phase"])
        if args.resume_from_logs:
            prev = existing_by_run_phase.get(run_phase)
            if prev is not None and str(prev.get("status", "")) == "ok":
                all_rows.append(dict(prev))
                precompleted_rows.append(dict(prev))
                continue
            resumed = build_resumed_row(row, axis_root)
            if resumed is not None:
                all_rows.append(resumed)
                precompleted_rows.append(resumed)
                continue
        pending_jobs.append(dict(row))

    notifier = SlackProgressNotifier(
        phase_label=f"{args.axis}",
        rows=run_rows,
        progress_step=int(args.slack_progress_step),
    )
    notifier.notify_plan(precompleted_rows=precompleted_rows)

    print(f"[launch] pairs={len(datasets) * len(model_options)} combos={len(matrix_rows)} runs={len(run_rows)}")
    print(f"[launch] precompleted={len(precompleted_rows)} pending={len(pending_jobs)}")
    print(f"[launch] matrix={matrix_csv}")

    if args.dry_run:
        leaderboard = build_leaderboard(all_rows)
        write_csv(summary_csv, all_rows, SUMMARY_FIELDS)
        write_csv(
            axis_root / "leaderboard.csv",
            leaderboard,
            [
                "dataset",
                "model",
                "model_label",
                "pair_id",
                "combo_id",
                "best_valid_mrr20",
                "test_mrr20",
                "test_unseen_mrr20",
                "test_main_seen_count",
                "test_main_unseen_count",
                "result_path",
            ],
        )
        manifest = {
            "track": TRACK,
            "axis": args.axis,
            "datasets": datasets,
            "models": model_options,
            "pairs": len(datasets) * len(model_options),
            "combos": len(matrix_rows),
            "seeds": seeds,
            "runs": len(run_rows),
            "precompleted": len(precompleted_rows),
            "pending": len(pending_jobs),
            "dry_run": True,
            "timestamp_utc": now_utc(),
            "matrix_csv": str(matrix_csv),
            "summary_csv": str(summary_csv),
        }
        (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[launch] dry_run_completed=true")
        return

    if pending_jobs:
        job_queue: Queue = Queue()
        for row in pending_jobs:
            job_queue.put(row)

        lock = threading.Lock()

        def worker(gpu_id: str) -> None:
            while True:
                if STOP_EVENT.is_set():
                    return
                try:
                    job = job_queue.get_nowait()
                except Empty:
                    return
                if STOP_EVENT.is_set():
                    return
                print(
                    "[launch] "
                    f"dataset={job['dataset']} model={job['model_option']} combo={job['combo_id']} "
                    f"seed={job['seed_id']} gpu={gpu_id} status=start"
                )
                try:
                    result_row = run_one(job, gpu_id, args, axis_root)
                except Exception as exc:
                    result_row = build_summary_row(
                        row=job,
                        gpu_id=gpu_id,
                        status="fail",
                        metrics={},
                        result_path="",
                        log_path=resolve_log_path(axis_root, job),
                        elapsed_sec=0.0,
                        error=f"worker_exception={exc}",
                    )
                with lock:
                    all_rows.append(result_row)
                    notifier.mark_complete(result_row)
                print(
                    "[launch] "
                    f"dataset={job['dataset']} model={job['model_option']} combo={job['combo_id']} "
                    f"seed={job['seed_id']} gpu={gpu_id} status={result_row['status']}"
                )
                job_queue.task_done()

        threads: List[threading.Thread] = []
        for gpu_id in gpus:
            t = threading.Thread(target=worker, args=(str(gpu_id),), daemon=False)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    all_rows.sort(
        key=lambda r: (
            str(r.get("dataset", "")),
            str(r.get("model", "")),
            str(r.get("pair_id", "")),
            str(r.get("combo_id", "")),
            safe_int(r.get("runtime_seed", 0), 0),
        )
    )

    leaderboard = build_leaderboard(all_rows)
    write_csv(summary_csv, all_rows, SUMMARY_FIELDS)
    write_csv(
        axis_root / "leaderboard.csv",
        leaderboard,
        [
            "dataset",
            "model",
            "model_label",
            "pair_id",
            "combo_id",
            "best_valid_mrr20",
            "test_mrr20",
            "test_unseen_mrr20",
            "test_main_seen_count",
            "test_main_unseen_count",
            "result_path",
        ],
    )

    ok_rows = [r for r in all_rows if str(r.get("status", "")) == "ok"]
    manifest = {
        "track": TRACK,
        "axis": args.axis,
        "datasets": datasets,
        "models": model_options,
        "pairs": len(datasets) * len(model_options),
        "combos": len(matrix_rows),
        "seeds": seeds,
        "runs": len(run_rows),
        "ok": len(ok_rows),
        "fail": len(all_rows) - len(ok_rows),
        "precompleted": len(precompleted_rows),
        "pending_executed": len(pending_jobs),
        "interrupted": bool(STOP_EVENT.is_set()),
        "timestamp_utc": now_utc(),
        "matrix_csv": str(matrix_csv),
        "summary_csv": str(summary_csv),
        "leaderboard_csv": str(axis_root / "leaderboard.csv"),
    }
    (axis_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        f"[launch] ok={manifest['ok']} fail={manifest['fail']} "
        f"interrupted={str(manifest['interrupted']).lower()} summary={summary_csv}"
    )


if __name__ == "__main__":
    main()
