#!/usr/bin/env python3
"""Shared helpers for the final_experiment unified tuning pipeline."""

from __future__ import annotations

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

CODE_DIR = Path(__file__).resolve().parent
TRACK = "final_experiment"
RESULT_ROOT = ARTIFACT_ROOT / "results" / TRACK
LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK

DEFAULT_MANIFEST_PATH = CODE_DIR / "space_manifest.json"
DEFAULT_TUNING_SPACE_PATH = CODE_DIR / "tuning_space.csv"
DEFAULT_SERVER_SPLIT_PATH = CODE_DIR / "server_split.json"

STAGE1_AXIS = "stage1_broad_search"
STAGE2_AXIS = "stage2_focus_search"
STAGE3_AXIS = "stage3_seed_confirm"

DEFAULT_DATASETS = [
    "beauty",
    "foursquare",
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "movielens1m",
    "retail_rocket",
]

BASELINE_MODELS = [
    "sasrec",
    "gru4rec",
    "tisasrec",
    "duorec",
    "bsarec",
    "fearec",
    "difsr",
    "fame",
    "fdsa",
]
ROUTE_MODEL = "featured_moe_n3"
ALL_MODELS = [*BASELINE_MODELS, ROUTE_MODEL]

MODEL_LABELS = {
    "sasrec": "SASRec",
    "gru4rec": "GRU4Rec",
    "tisasrec": "TiSASRec",
    "duorec": "DuoRec",
    "bsarec": "BSARec",
    "fearec": "FEARec",
    "difsr": "DIFSR",
    "fame": "FAME",
    "fdsa": "FDSA",
    "featured_moe_n3": "FeaturedMoE_N3",
}

SERVER_MODEL_SPLITS = {
    "server_a": {
        "baseline_models": ["tisasrec", "bsarec", "sasrec", "difsr"],
        "route_model": ROUTE_MODEL,
    },
    "server_b": {
        "baseline_models": ["fearec", "duorec", "fame", "fdsa", "gru4rec"],
        "route_model": "",
    },
}

DATASET_CONFIG_MAP = {
    "beauty": "tune_ab",
    "amazon_beauty": "tune_ab",
    "foursquare": "tune_fs",
    "kuaireclargestrictposv2_0.2": "tune_kuai_strict_small",
    "lastfm0.03": "tune_lfm_small",
    "movielens1m": "tune_ml",
    "retail_rocket": "tune_rr",
}

MODEL_ALIASES = {
    "sasrec": "sasrec",
    "gru4rec": "gru4rec",
    "tisasrec": "tisasrec",
    "duorec": "duorec",
    "bsarec": "bsarec",
    "fearec": "fearec",
    "difsr": "difsr",
    "fame": "fame",
    "fdsa": "fdsa",
    "featured_moe_n3": "featured_moe_n3",
    "featuredmoe_n3": "featured_moe_n3",
    "featured_moe_n3_tune": "featured_moe_n3",
    "featuredmoe_n3_tune": "featured_moe_n3",
    "fmoen3": "featured_moe_n3",
    "fmoe_n3": "featured_moe_n3",
}

TRANSFORMER_STYLE_MODELS = {"sasrec", "tisasrec", "duorec", "bsarec", "fearec", "difsr", "fame", "fdsa"}

BASELINE_MODEL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "sasrec": {
        "layer_key": "num_layers",
        "head_key": "num_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": [],
    },
    "gru4rec": {
        "layer_key": "num_layers",
        "head_key": "",
        "dropout_keys": ["dropout_prob"],
        "fixed_aliases": {"hidden_dropout_prob": "dropout_prob", "embedding_size": "hidden_size"},
        "extra_keys": [],
    },
    "tisasrec": {
        "layer_key": "n_layers",
        "head_key": "n_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["time_span"],
    },
    "duorec": {
        "layer_key": "n_layers",
        "head_key": "n_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["tau", "lmd", "lmd_sem"],
    },
    "bsarec": {
        "layer_key": "num_layers",
        "head_key": "num_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["bsarec_alpha", "bsarec_c"],
    },
    "fearec": {
        "layer_key": "n_layers",
        "head_key": "n_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["tau", "lmd", "lmd_sem"],
    },
    "difsr": {
        "layer_key": "num_layers",
        "head_key": "num_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["attribute_hidden_size", "fusion_type", "lambda_attr"],
    },
    "fame": {
        "layer_key": "num_layers",
        "head_key": "num_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["num_experts"],
    },
    "fdsa": {
        "layer_key": "n_layers",
        "head_key": "n_heads",
        "dropout_keys": ["hidden_dropout_prob", "attn_dropout_prob"],
        "fixed_aliases": {"dropout_ratio": "hidden_dropout_prob"},
        "extra_keys": ["attribute_hidden_size", "lambda_attr"],
    },
}

BASELINE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "sasrec": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "num_layers": 2, "num_heads": 4, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.15, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "gru4rec": {"hidden_size": 128, "embedding_size": 128, "num_layers": 1, "dropout_prob": 0.2, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "tisasrec": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "n_layers": 2, "n_heads": 4, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.15, "time_span": 256, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "duorec": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "n_layers": 2, "n_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.0, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "bsarec": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "num_layers": 2, "num_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "bsarec_alpha": 0.55, "bsarec_c": 3, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "fearec": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "n_layers": 2, "n_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.08, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "difsr": {"hidden_size": 160, "embedding_size": 160, "inner_size": 320, "num_layers": 2, "num_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "attribute_hidden_size": 160, "fusion_type": "gate", "lambda_attr": 0.1, "use_attribute_predictor": True, "selected_features": ["category"], "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "fame": {"hidden_size": 128, "embedding_size": 128, "inner_size": 256, "num_layers": 2, "num_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "num_experts": 4, "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
    "fdsa": {"hidden_size": 160, "embedding_size": 160, "inner_size": 320, "n_layers": 2, "n_heads": 4, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12, "attribute_hidden_size": 160, "lambda_attr": 0.12, "selected_features": ["category"], "pooling_mode": "mean", "weight_decay": 1e-4, "MAX_ITEM_LIST_LENGTH": 20},
}

NUMERIC_FALLBACKS: Dict[str, List[float]] = {
    "hidden_size": [64, 96, 128, 160, 192, 224, 256],
    "embedding_size": [64, 96, 128, 160, 192, 224, 256],
    "inner_size": [128, 192, 256, 320, 384, 448, 512],
    "num_layers": [1, 2, 3, 4],
    "n_layers": [1, 2, 3, 4],
    "num_heads": [1, 2, 4, 8],
    "n_heads": [1, 2, 4, 8],
    "hidden_dropout_prob": [0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3],
    "attn_dropout_prob": [0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
    "dropout_prob": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    "dropout_ratio": [0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3],
    "weight_decay": [0.0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4],
    "time_span": [128, 256, 384, 512, 1024],
    "tau": [0.12, 0.16, 0.18, 0.2, 0.22, 0.25],
    "lmd": [0.02, 0.03, 0.04, 0.05, 0.06],
    "lmd_sem": [0.0, 0.05, 0.08, 0.1, 0.12],
    "bsarec_alpha": [0.35, 0.5, 0.55, 0.7, 0.78, 0.9],
    "bsarec_c": [2, 3, 5, 7],
    "attribute_hidden_size": [64, 96, 128, 160, 192, 224, 256],
    "lambda_attr": [0.0, 0.08, 0.1, 0.12, 0.14, 0.16],
    "num_experts": [2, 3, 4, 5, 6],
}

CHOICE_FALLBACKS: Dict[str, List[Any]] = {
    "fusion_type": ["gate", "sum", "concat"],
}

STAGE1_BASELINE_MAX_EVALS = {
    "beauty": 48,
    "foursquare": 36,
    "KuaiRecLargeStrictPosV2_0.2": 36,
    "retail_rocket": 24,
    "movielens1m": 24,
    "lastfm0.03": 24,
}
STAGE1_ROUTE_BANK_COUNTS = {
    "beauty": 24,
    "KuaiRecLargeStrictPosV2_0.2": 24,
    "foursquare": 16,
    "retail_rocket": 12,
    "movielens1m": 12,
    "lastfm0.03": 10,
}
STAGE1_ROUTE_MAX_EVALS = {
    "beauty": 20,
    "KuaiRecLargeStrictPosV2_0.2": 20,
    "foursquare": 18,
    "retail_rocket": 12,
    "movielens1m": 12,
    "lastfm0.03": 12,
}
STAGE2_MAX_EVALS = {
    "beauty": 16,
    "foursquare": 14,
    "KuaiRecLargeStrictPosV2_0.2": 14,
    "retail_rocket": 10,
    "movielens1m": 10,
    "lastfm0.03": 10,
}
STAGE3_SEED_COUNTS = {
    "beauty": 3,
    "foursquare": 3,
    "KuaiRecLargeStrictPosV2_0.2": 3,
    "retail_rocket": 2,
    "movielens1m": 2,
    "lastfm0.03": 2,
}

STAGE_TUNE_EPOCHS = {
    "stage1": 35,
    "stage2": 60,
    "stage3": 100,
}
STAGE_TUNE_PATIENCE = {
    "stage1": 4,
    "stage2": 6,
    "stage3": 10,
}
DEFAULT_MAX_RUN_HOURS = 2.0
DEFAULT_OOM_RETRY_LIMIT = 2
MODEL_OOM_RETRY_MIN = {
    "tisasrec": 5,
}
TISASREC_SAFE_BATCH_SIZES: Dict[str, Tuple[int, int]] = {
    "beauty": (2048, 4096),
    "foursquare": (1024, 2048),
    "KuaiRecLargeStrictPosV2_0.2": (1536, 3072),
    "lastfm0.03": (1536, 3072),
    "movielens1m": (2048, 4096),
    "retail_rocket": (1024, 2048),
}

TRUSTED_BASELINE_AXES = {
    "abcd_v2_lean",
    "pair60_v4",
    "pair60_addtuning",
    "pair60_addtuning2",
    "pair60_addtuning3",
    "pair60_addtuning3_2",
    "pair60_addtuning4",
    "pair60_v4_revised",
    "pair60_v4_revised_long12h",
}
TRUSTED_ROUTE_AXES = {
    "stage1_a12_broadtemplates",
    "stage2_a12_mixedtemplates",
    "stage3_a12_seenfocus",
    "crossdataset_a12_portfolio",
}

FMOE_ROUTE_SEARCH_DEFAULTS = {
    "d_router_hidden": [32, 64, 96, 128],
    "stage_feature_dropout_prob": [0.0, 0.03, 0.05, 0.1],
    "attn_dropout_prob": [0.05, 0.08, 0.1, 0.12],
    "route_consistency_lambda": [2.5e-4, 5e-4, 8e-4, 1.2e-3],
    "z_loss_lambda": [5e-5, 1e-4, 2e-4],
}
FMOE_EXPERT_SCALE_DEFAULT = [2, 3, 4]
FMOE_EXPERT_SCALE_KUAI = [2, 3, 4, 5]
FMOE_D_FEAT_FALLBACK = [8, 12, 16, 24]
FMOE_WEIGHT_DECAY_FALLBACK = [5e-7, 1e-6, 2e-6, 5e-6]
FMOE_MAX_LEN_DEFAULT = [10, 20, 30]
FMOE_MAX_LEN_LONG = [10, 20, 30, 50]

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

NINE_METRIC_KEYS = [
    "hit@5",
    "hit@10",
    "hit@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "mrr@5",
    "mrr@10",
    "mrr@20",
]

SUMMARY_FIELDS = [
    "stage",
    "dataset",
    "model",
    "family",
    "job_id",
    "parent_job_id",
    "run_phase",
    "runtime_seed",
    "gpu_id",
    "status",
    "valid_score",
    "test_score",
    "best_valid_mrr20",
    "test_mrr20",
    "result_path",
    "log_path",
    "elapsed_sec",
    "error",
    "timestamp_utc",
]

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_token(text: str, *, upper: bool = False) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_")
    if not token:
        token = "NA"
    return token.upper() if upper else token.lower()


def parse_csv_list(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for token in parse_csv_list(text):
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


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


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dedupe_keep_order(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for value in values:
        token = json.dumps(value, ensure_ascii=True, sort_keys=True)
        if token in seen:
            continue
        seen.add(token)
        out.append(value)
    return out


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"invalid hydra float: {value}")
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{hydra_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"unsupported hydra literal type: {type(value).__name__}")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG_MAP:
        raise KeyError(f"unknown dataset config mapping: {dataset}")
    return DATASET_CONFIG_MAP[key]


def normalize_dataset(dataset: str) -> str:
    raw = str(dataset or "").strip()
    if raw == "amazon_beauty":
        return "beauty"
    return raw


def normalize_model(model: str) -> str:
    key = sanitize_token(model, upper=False)
    key = key.replace("__", "_")
    return MODEL_ALIASES.get(key, key)


def metric_mean(metrics: Dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    values: List[float] = []
    for key in NINE_METRIC_KEYS:
        if key in metrics:
            values.append(safe_float(metrics.get(key), 0.0))
    if values:
        return float(sum(values) / len(values))
    if "mrr@20" in metrics:
        return safe_float(metrics.get("mrr@20"), 0.0)
    return 0.0


def special_seen_mean(special_payload: Dict[str, Any] | None) -> float:
    if not isinstance(special_payload, dict):
        return 0.0
    for key in ("overall_seen_target", "overall"):
        mean_val = metric_mean(special_payload.get(key) or {})
        if mean_val > 0.0:
            return mean_val
    return 0.0


def result_valid_mean(payload: Dict[str, Any]) -> float:
    return max(
        special_seen_mean(payload.get("best_valid_special_metrics") or {}),
        metric_mean(payload.get("best_valid_result") or {}),
        safe_float(payload.get("best_mrr@20"), 0.0),
    )


def result_test_mean(payload: Dict[str, Any]) -> float:
    return max(
        special_seen_mean(payload.get("test_special_metrics") or {}),
        metric_mean(payload.get("test_result") or {}),
        safe_float(payload.get("test_mrr@20"), 0.0),
    )


def trial_valid_mean(trial: Dict[str, Any]) -> float:
    return max(metric_mean(trial.get("valid_result") or {}), safe_float(trial.get("mrr@20"), 0.0))


def trial_test_mean(trial: Dict[str, Any]) -> float:
    return max(metric_mean(trial.get("test_result") or {}), safe_float(trial.get("test_mrr@20"), 0.0))


def config_signature(config: Dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def merge_result_config(payload: Dict[str, Any], sampled_params: Dict[str, Any] | None) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    fixed = dict(payload.get("fixed_search") or {})
    cfg.update(fixed)
    if sampled_params:
        cfg.update(dict(sampled_params))
    return cfg


def top_unique_trials(payload: Dict[str, Any], *, top_k: int) -> List[Dict[str, Any]]:
    ranked = []
    for idx, trial in enumerate(list(payload.get("trials") or []), start=1):
        if str((trial or {}).get("status", "")).strip().lower() != "ok":
            continue
        config = merge_result_config(payload, dict((trial or {}).get("params") or {}))
        ranked.append(
            {
                "trial_rank": idx,
                "valid_score": trial_valid_mean(trial or {}),
                "test_score": trial_test_mean(trial or {}),
                "config": config,
                "signature": config_signature(config),
            }
        )
    ranked.sort(key=lambda row: (float(row["valid_score"]), float(row["test_score"])), reverse=True)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        if row["signature"] in seen:
            continue
        seen.add(row["signature"])
        out.append(row)
        if len(out) >= int(top_k):
            break
    return out


def total_elapsed_from_payload(payload: Dict[str, Any]) -> float:
    total = 0.0
    for trial in list(payload.get("trials") or []):
        total += safe_float((trial or {}).get("elapsed"), 0.0)
    return float(total)


def _load_module(module_path: Path, module_name: str, prepend_path: Path | None = None) -> Any:
    remove_path = False
    if prepend_path is not None:
        prepend = str(prepend_path)
        if prepend not in sys.path:
            sys.path.insert(0, prepend)
            remove_path = True
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load module spec: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if remove_path and prepend_path is not None:
            prepend = str(prepend_path)
            if prepend in sys.path:
                sys.path.remove(prepend)


def load_baseline_bank_module() -> Any:
    return _load_module(
        EXP_DIR / "run" / "baseline" / "run_final_all_datasets.py",
        "final_experiment_baseline_bank",
    )


def load_fmoe_bank_module() -> Any:
    base_dir = EXP_DIR / "run" / "fmoe_n3"
    return _load_module(
        base_dir / "run_final_all_datasets.py",
        "final_experiment_fmoe_bank",
        prepend_path=base_dir,
    )


def load_fmoe_aux_module() -> Any:
    base_dir = EXP_DIR / "run" / "fmoe_n3"
    return _load_module(
        base_dir / "run_phase9_auxloss.py",
        "final_experiment_fmoe_aux",
        prepend_path=base_dir,
    )


def validate_session_fixed_files(dataset: str) -> None:
    ds_dir = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4" / dataset
    required = [
        ds_dir / f"{dataset}.train.inter",
        ds_dir / f"{dataset}.valid.inter",
        ds_dir / f"{dataset}.test.inter",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"feature_added_v4/session_fixed files missing for dataset={dataset}: {missing}")


def historical_result_paths(track: str) -> Iterator[Path]:
    root = ARTIFACT_ROOT / "results" / track
    if not root.exists():
        return iter(())
    return root.rglob("*.json")


def iter_historical_results() -> Iterator[Dict[str, Any]]:
    seen: set[str] = set()
    for track, allowed_axes in (("baseline_2", TRUSTED_BASELINE_AXES), ("fmoe_n4", TRUSTED_ROUTE_AXES)):
        root = ARTIFACT_ROOT / "results" / track
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = read_json(path)
            except Exception:
                continue
            run_phase = str(payload.get("run_phase", "")).strip()
            run_axis = sanitize_token(payload.get("run_axis", ""), upper=False)
            if not run_phase or run_phase in seen:
                continue
            if run_axis not in allowed_axes:
                continue
            seen.add(run_phase)
            model = normalize_model(payload.get("model", ""))
            dataset = normalize_dataset(payload.get("dataset", ""))
            if dataset not in DEFAULT_DATASETS or model not in ALL_MODELS:
                continue
            yield {
                "track": track,
                "path": path,
                "payload": payload,
                "dataset": dataset,
                "model": model,
                "run_axis": run_axis,
                "valid_score": result_valid_mean(payload),
                "test_score": result_test_mean(payload),
                "elapsed_sec": total_elapsed_from_payload(payload),
            }


def gather_history_by_pair() -> Tuple[Dict[Tuple[str, str], List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    pair_records: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    route_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in iter_historical_results():
        dataset = str(record["dataset"])
        model = str(record["model"])
        pair_records[(dataset, model)].append(record)
        if model == ROUTE_MODEL:
            route_records[dataset].append(record)
    for rows in pair_records.values():
        rows.sort(key=lambda row: (float(row["valid_score"]), float(row["test_score"])), reverse=True)
    for rows in route_records.values():
        rows.sort(key=lambda row: (float(row["valid_score"]), float(row["test_score"])), reverse=True)
    return pair_records, route_records


def _top_records(records: List[Dict[str, Any]], *, top_frac: float = 0.2, min_count: int = 8) -> List[Dict[str, Any]]:
    if not records:
        return []
    n = max(int(math.ceil(len(records) * float(top_frac))), int(min_count))
    return list(records[: min(len(records), n)])


def best_history_config(records: List[Dict[str, Any]], *, default: Dict[str, Any]) -> Dict[str, Any]:
    for record in records:
        payload = dict(record.get("payload") or {})
        cfg = merge_result_config(payload, dict(payload.get("best_params") or {}))
        if cfg:
            merged = dict(default)
            merged.update(cfg)
            return merged
    return dict(default)


def support_values(
    records: List[Dict[str, Any]],
    key: str,
    *,
    default: Any = None,
    fallback: Sequence[Any] | None = None,
    limit: int = 5,
) -> List[Any]:
    counter: Counter[str] = Counter()
    token_to_value: Dict[str, Any] = {}
    top_records = _top_records(records)
    for record in top_records:
        payload = dict(record.get("payload") or {})
        best_cfg = merge_result_config(payload, dict(payload.get("best_params") or {}))
        if key not in best_cfg:
            continue
        value = best_cfg[key]
        token = json.dumps(value, ensure_ascii=True, sort_keys=True)
        counter[token] += 1
        token_to_value[token] = value

    ordered: List[Any] = []
    for token, _count in counter.most_common():
        ordered.append(token_to_value[token])
    if default is not None:
        ordered.append(default)
    if fallback:
        ordered.extend(list(fallback))
    ordered = dedupe_keep_order(ordered)
    if not ordered:
        return []

    all_numeric = all(isinstance(value, (int, float)) for value in ordered)
    if all_numeric:
        ordered = sorted(ordered, key=lambda value: float(value))
        if len(ordered) > int(limit):
            center = float(default) if isinstance(default, (int, float)) else float(ordered[len(ordered) // 2])
            ordered = sorted(ordered, key=lambda value: (abs(float(value) - center), float(value)))[: int(limit)]
            ordered = sorted(ordered, key=lambda value: float(value))
    else:
        ordered = ordered[: int(limit)]
    return list(ordered)


def build_lr_grid(dataset: str, model: str, records: List[Dict[str, Any]], *, points: int = 7) -> List[float]:
    base_lo, base_hi = DATASET_LR_BASE[dataset]
    mult = MODEL_LR_MULT.get(model, 1.0)
    safe_lo = max(1e-6, base_lo * mult)
    safe_hi = max(safe_lo * 1.5, base_hi * mult)

    observed = []
    top = []
    for record in records:
        payload = dict(record.get("payload") or {})
        cfg = merge_result_config(payload, dict(payload.get("best_params") or {}))
        lr = cfg.get("learning_rate")
        if lr is None:
            continue
        lr_val = safe_float(lr, 0.0)
        if lr_val <= 0.0:
            continue
        observed.append(lr_val)
    for record in _top_records(records, min_count=5):
        payload = dict(record.get("payload") or {})
        cfg = merge_result_config(payload, dict(payload.get("best_params") or {}))
        lr = cfg.get("learning_rate")
        if lr is None:
            continue
        lr_val = safe_float(lr, 0.0)
        if lr_val > 0.0:
            top.append(lr_val)

    if observed:
        safe_lo = max(safe_lo, min(observed) / 1.8)
        safe_hi = min(safe_hi, max(observed) * 1.8)
        if safe_hi <= safe_lo:
            safe_hi = safe_lo * 2.0

    if top:
        center = math.exp(sum(math.log(v) for v in top) / len(top))
    elif observed:
        center = math.exp(sum(math.log(v) for v in observed) / len(observed))
    else:
        center = math.sqrt(safe_lo * safe_hi)

    ratio = MODEL_LR_BAND_RATIO.get(model, 4.0)
    lo = max(safe_lo, center / math.sqrt(ratio))
    hi = min(safe_hi, center * math.sqrt(ratio))
    if hi <= lo:
        lo = safe_lo
        hi = max(safe_hi, lo * 2.0)
    if points <= 1:
        return [round(float(center), 8)]
    step = (math.log(hi) - math.log(lo)) / float(points - 1)
    values = [round(math.exp(math.log(lo) + step * idx), 8) for idx in range(points)]
    return dedupe_keep_order(values)


def runtime_median_seconds(records: List[Dict[str, Any]]) -> float:
    elapsed = sorted(float(row.get("elapsed_sec", 0.0) or 0.0) for row in records if float(row.get("elapsed_sec", 0.0) or 0.0) > 0.0)
    if not elapsed:
        return 0.0
    mid = len(elapsed) // 2
    if len(elapsed) % 2 == 1:
        return float(elapsed[mid])
    return float((elapsed[mid - 1] + elapsed[mid]) / 2.0)


def allow_len50(records: List[Dict[str, Any]], dataset: str, model: str) -> bool:
    if dataset in {"movielens1m", "retail_rocket", "lastfm0.03"}:
        return True
    for record in _top_records(records, min_count=5):
        payload = dict(record.get("payload") or {})
        cfg = merge_result_config(payload, dict(payload.get("best_params") or {}))
        seq_len = safe_int(cfg.get("MAX_ITEM_LIST_LENGTH", cfg.get("max_len", 0)), 0)
        if seq_len >= 40:
            return True
    return model in {"bsarec", "gru4rec"}


def tisasrec_safe_batch_sizes(dataset: str) -> Tuple[int, int]:
    return TISASREC_SAFE_BATCH_SIZES.get(str(dataset), (1024, 2048))


def build_baseline_pair_spec(dataset: str, model: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    default_cfg = best_history_config(records, default=BASELINE_DEFAULTS[model])
    schema = BASELINE_MODEL_SCHEMA[model]
    search_space: Dict[str, List[Any]] = {}

    search_space["learning_rate"] = build_lr_grid(dataset, model, records, points=5)
    search_space["weight_decay"] = support_values(records, "weight_decay", default=default_cfg.get("weight_decay"), fallback=NUMERIC_FALLBACKS["weight_decay"], limit=4)

    max_len_values = list(FMOE_MAX_LEN_LONG if allow_len50(records, dataset, model) else FMOE_MAX_LEN_DEFAULT)
    search_space["MAX_ITEM_LIST_LENGTH"] = max_len_values

    hidden_values = support_values(records, "hidden_size", default=default_cfg.get("hidden_size"), fallback=NUMERIC_FALLBACKS["hidden_size"], limit=4)
    emb_values = support_values(records, "embedding_size", default=default_cfg.get("embedding_size"), fallback=NUMERIC_FALLBACKS["embedding_size"], limit=4)
    if hidden_values:
        search_space["hidden_size"] = hidden_values
    if emb_values and model not in {"difsr", "tisasrec"}:
        search_space["embedding_size"] = emb_values

    layer_key = str(schema.get("layer_key", "")).strip()
    if layer_key:
        search_space[layer_key] = support_values(records, layer_key, default=default_cfg.get(layer_key), fallback=NUMERIC_FALLBACKS[layer_key], limit=3)

    head_key = str(schema.get("head_key", "")).strip()
    if head_key:
        search_space[head_key] = support_values(records, head_key, default=default_cfg.get(head_key), fallback=NUMERIC_FALLBACKS[head_key], limit=3)

    for dropout_key in list(schema.get("dropout_keys") or []):
        search_space[dropout_key] = support_values(records, dropout_key, default=default_cfg.get(dropout_key), fallback=NUMERIC_FALLBACKS.get(dropout_key, []), limit=4)

    fixed_context: Dict[str, Any] = {
        "inner_size": safe_int(default_cfg.get("inner_size", max(256, safe_int(default_cfg.get("hidden_size", 128), 128) * 2)), 256),
    }
    for alias_key, source_key in dict(schema.get("fixed_aliases") or {}).items():
        if source_key in search_space and alias_key not in search_space:
            fixed_context[alias_key] = default_cfg.get(source_key)

    extra_keys = list(schema.get("extra_keys") or [])[:3]
    for key in extra_keys:
        fallback = NUMERIC_FALLBACKS.get(key, CHOICE_FALLBACKS.get(key, []))
        values = support_values(records, key, default=default_cfg.get(key), fallback=fallback, limit=4)
        if values:
            search_space[key] = values

    if model in {"difsr", "tisasrec"}:
        if hidden_values:
            fixed_context["embedding_size"] = int(hidden_values[0])
        else:
            fixed_context["embedding_size"] = safe_int(default_cfg.get("hidden_size", default_cfg.get("embedding_size", 160)), 160)
    if model == "difsr":
        fixed_context["use_attribute_predictor"] = bool(default_cfg.get("use_attribute_predictor", True))
        fixed_context["selected_features"] = default_cfg.get("selected_features", ["category"])
    if model == "tisasrec":
        train_batch_size, eval_batch_size = tisasrec_safe_batch_sizes(dataset)
        fixed_context["train_batch_size"] = train_batch_size
        fixed_context["eval_batch_size"] = eval_batch_size
    if model == "fdsa":
        fixed_context["selected_features"] = default_cfg.get("selected_features", ["category"])
        fixed_context["pooling_mode"] = default_cfg.get("pooling_mode", "mean")

    runtime_median = runtime_median_seconds(records)
    best_score = float(records[0]["valid_score"]) if records else 0.0
    best_test = float(records[0]["test_score"]) if records else 0.0

    return {
        "dataset": dataset,
        "model": model,
        "family": "baseline",
        "model_label": MODEL_LABELS[model],
        "history_count": len(records),
        "history_best_valid": best_score,
        "history_best_test": best_test,
        "runtime_median_sec": runtime_median,
        "stage1_max_evals": STAGE1_BASELINE_MAX_EVALS[dataset],
        "stage2_max_evals": STAGE2_MAX_EVALS[dataset],
        "stage3_seed_count": STAGE3_SEED_COUNTS[dataset],
        "search_space": search_space,
        "fixed_context": fixed_context,
        "default_config": default_cfg,
    }


def _route_family_signature(row: Dict[str, Any]) -> str:
    keep = {}
    fixed_values = dict(row.get("fixed_values") or {})
    for key in ("embedding_size", "d_ff", "d_expert_hidden", "d_feat_emb", "hidden_dropout_prob", "fixed_hidden_dropout_prob", "stage_family_dropout_prob"):
        if key in fixed_values:
            keep[key] = fixed_values[key]
    keep["family_id"] = row.get("family_id", "")
    keep["capacity_anchor"] = row.get("capacity_anchor", "")
    return config_signature(keep)


def route_base_overrides() -> Dict[str, Any]:
    aux_mod = load_fmoe_aux_module()
    overrides = dict(aux_mod._base_fixed_overrides())
    aux_mod._apply_base_overrides(
        overrides=overrides,
        base_cfg={
            "wrapper_map": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
            "source_profile": "src_abc_feature",
            "bias_mode": "bias_both",
        },
        feature_group_bias_lambda=0.0,
        rule_bias_scale=0.0,
    )
    overrides["layer_layout"] = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
    overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
    overrides["macro_history_window"] = 5
    overrides["route_consistency_pairs"] = 1
    overrides["route_consistency_min_sim"] = 0.995
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
    return overrides


def load_route_manifest_candidates() -> Dict[str, List[Dict[str, Any]]]:
    fmoe_manifest_paths = [
        ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage1_A12_BroadTemplates" / "stage1_manifest.json",
        ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage2_A12_MixedTemplates" / "stage2_manifest.json",
        ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage3_A12_SeenFocus" / "stage3_manifest.json",
        ARTIFACT_ROOT / "logs" / "fmoe_n4" / "CrossDataset_A12_Portfolio" / "cross_dataset_manifest.json",
        ARTIFACT_ROOT / "logs" / "fmoe_n4" / "CrossDataset_A12_Portfolio" / "cross_dataset_fs_rr_ml_followup_manifest.json",
    ]
    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for summary_path in (ARTIFACT_ROOT / "logs" / "fmoe_n4").rglob("summary.csv"):
        for row in read_csv_rows(summary_path):
            run_phase = str(row.get("run_phase", "")).strip()
            dataset = normalize_dataset(row.get("dataset", ""))
            if run_phase and dataset:
                summary_index[(dataset, run_phase)] = row

    for manifest_path in fmoe_manifest_paths:
        if not manifest_path.exists():
            continue
        try:
            payload = read_json(manifest_path)
        except Exception:
            continue
        for row in list(payload.get("rows") or []):
            dataset = normalize_dataset(row.get("dataset", ""))
            if dataset not in DEFAULT_DATASETS:
                continue
            run_phase = str(row.get("run_phase", "")).strip()
            summary_row = summary_index.get((dataset, run_phase), {})
            result_path = Path(str(summary_row.get("result_path", "")).strip()) if str(summary_row.get("result_path", "")).strip() else None
            result_payload: Dict[str, Any] = {}
            if result_path is not None and result_path.exists():
                try:
                    result_payload = read_json(result_path)
                except Exception:
                    result_payload = {}
            valid_score = result_valid_mean(result_payload) if result_payload else safe_float(summary_row.get("run_best_valid_mrr20"), 0.0)
            test_score = result_test_mean(result_payload) if result_payload else safe_float(summary_row.get("test_mrr20"), 0.0)
            candidate = dict(row)
            candidate["result_path"] = str(result_path) if result_path else ""
            candidate["valid_score"] = valid_score
            candidate["test_score"] = test_score
            candidate["signature"] = _route_family_signature(candidate)
            candidates[dataset].append(candidate)

    for dataset, rows in candidates.items():
        unique_rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        rows.sort(key=lambda row: (float(row.get("valid_score", 0.0)), float(row.get("test_score", 0.0))), reverse=True)
        for row in rows:
            signature = str(row.get("signature", ""))
            if not signature or signature in seen:
                continue
            seen.add(signature)
            unique_rows.append(row)
        candidates[dataset] = unique_rows
    return candidates


def build_route_bank(dataset: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fmoe_bank = load_fmoe_bank_module()
    manifest_candidates = load_route_manifest_candidates().get(dataset, [])

    families: List[Dict[str, Any]] = []
    seen: set[str] = set()
    bank_count = STAGE1_ROUTE_BANK_COUNTS[dataset]
    best_valid = float(records[0]["valid_score"]) if records else 0.0
    best_test = float(records[0]["test_score"]) if records else 0.0
    runtime_median = runtime_median_seconds(records)

    for idx, row in enumerate(manifest_candidates, start=1):
        fixed_values = dict(row.get("fixed_values") or {})
        search_space_hist = dict(row.get("search_space") or {})
        d_feat_default = fixed_values.get("d_feat_emb")
        if d_feat_default is None:
            d_feat_candidates = list(search_space_hist.get("d_feat_emb") or [])
            d_feat_default = d_feat_candidates[0] if d_feat_candidates else 16
        hidden_default = fixed_values.get("fixed_hidden_dropout_prob", fixed_values.get("hidden_dropout_prob", 0.15))
        family_dropout_default = fixed_values.get("stage_family_dropout_prob", {"macro": 0.02, "mid": 0.02, "micro": 0.02})
        if isinstance(family_dropout_default, (int, float)):
            family_dropout_default = {"macro": float(family_dropout_default), "mid": float(family_dropout_default), "micro": float(family_dropout_default)}
        signature = str(row.get("signature", ""))
        if signature in seen:
            continue
        seen.add(signature)
        family_search = {
            "learning_rate": build_lr_grid(dataset, ROUTE_MODEL, records, points=7),
            "weight_decay": support_values(records, "weight_decay", default=fixed_values.get("fixed_weight_decay", 1e-6), fallback=FMOE_WEIGHT_DECAY_FALLBACK, limit=4),
            "MAX_ITEM_LIST_LENGTH": list(FMOE_MAX_LEN_LONG if allow_len50(records, dataset, ROUTE_MODEL) else FMOE_MAX_LEN_DEFAULT),
            "expert_scale": list(FMOE_EXPERT_SCALE_KUAI if dataset == "KuaiRecLargeStrictPosV2_0.2" else FMOE_EXPERT_SCALE_DEFAULT),
            "d_router_hidden": list(FMOE_ROUTE_SEARCH_DEFAULTS["d_router_hidden"]),
            "stage_feature_dropout_prob": list(FMOE_ROUTE_SEARCH_DEFAULTS["stage_feature_dropout_prob"]),
            "attn_dropout_prob": list(FMOE_ROUTE_SEARCH_DEFAULTS["attn_dropout_prob"]),
            "route_consistency_lambda": list(FMOE_ROUTE_SEARCH_DEFAULTS["route_consistency_lambda"]),
            "z_loss_lambda": list(FMOE_ROUTE_SEARCH_DEFAULTS["z_loss_lambda"]),
        }
        fixed_context = {
            "embedding_size": safe_int(fixed_values.get("embedding_size", 128), 128),
            "hidden_size": safe_int(fixed_values.get("embedding_size", 128), 128),
            "d_ff": safe_int(fixed_values.get("d_ff", 256), 256),
            "d_expert_hidden": safe_int(fixed_values.get("d_expert_hidden", 128), 128),
            "d_feat_emb": safe_int(d_feat_default, 16),
            "hidden_dropout_prob": safe_float(hidden_default, 0.15),
            "fixed_hidden_dropout_prob": safe_float(hidden_default, 0.15),
            "stage_family_dropout_prob": family_dropout_default,
            "router_feature_proj_dim": 0,
            "router_use_hidden": True,
            "router_use_feature": True,
            "router_impl": "learned",
            "lr_scheduler_type": "warmup_cosine",
        }
        families.append(
            {
                "dataset": dataset,
                "model": ROUTE_MODEL,
                "family": "route",
                "family_id": f"R{idx:02d}",
                "source_family_id": row.get("family_id", row.get("setting_id", "")),
                "capacity_anchor": row.get("capacity_anchor", ""),
                "history_valid": safe_float(row.get("valid_score"), 0.0),
                "history_test": safe_float(row.get("test_score"), 0.0),
                "search_space": family_search,
                "fixed_context": fixed_context,
            }
        )
        if len(families) >= bank_count:
            break

    if len(families) < bank_count:
        preset = list(fmoe_bank.DATASET_HPARAM_PRESET_12.get(dataset, []))
        if not preset and dataset == "beauty":
            preset = list(fmoe_bank.DATASET_HPARAM_PRESET_12.get("amazon_beauty", []))
        all_hids = sorted(str(key) for key in dict(fmoe_bank.HPARAM_BANK or {}).keys())
        if not preset:
            preset = list(all_hids)
        else:
            preset = list(dict.fromkeys([*preset, *all_hids]))
        for hid in preset:
            if len(families) >= bank_count:
                break
            cfg = dict(fmoe_bank.HPARAM_BANK.get(str(hid), {}) or {})
            signature = config_signature({"hid": hid})
            if signature in seen:
                continue
            seen.add(signature)
            fixed_context = {
                "embedding_size": safe_int(cfg.get("embedding_size", 128), 128),
                "hidden_size": safe_int(cfg.get("embedding_size", 128), 128),
                "d_ff": safe_int(cfg.get("d_ff", 256), 256),
                "d_expert_hidden": safe_int(cfg.get("d_expert_hidden", 128), 128),
                "d_feat_emb": 16,
                "hidden_dropout_prob": safe_float(cfg.get("fixed_hidden_dropout_prob", 0.15), 0.15),
                "fixed_hidden_dropout_prob": safe_float(cfg.get("fixed_hidden_dropout_prob", 0.15), 0.15),
                "stage_family_dropout_prob": {"macro": 0.02, "mid": 0.02, "micro": 0.02},
                "router_feature_proj_dim": 0,
                "router_use_hidden": True,
                "router_use_feature": True,
                "router_impl": "learned",
                "lr_scheduler_type": "warmup_cosine",
            }
            families.append(
                {
                    "dataset": dataset,
                    "model": ROUTE_MODEL,
                    "family": "route",
                    "family_id": f"R{len(families) + 1:02d}",
                    "source_family_id": hid,
                    "capacity_anchor": hid,
                    "history_valid": 0.0,
                    "history_test": 0.0,
                    "search_space": {
                        "learning_rate": build_lr_grid(dataset, ROUTE_MODEL, records, points=7),
                        "weight_decay": list(FMOE_WEIGHT_DECAY_FALLBACK),
                        "MAX_ITEM_LIST_LENGTH": list(FMOE_MAX_LEN_LONG if allow_len50(records, dataset, ROUTE_MODEL) else FMOE_MAX_LEN_DEFAULT),
                        "expert_scale": list(FMOE_EXPERT_SCALE_KUAI if dataset == "KuaiRecLargeStrictPosV2_0.2" else FMOE_EXPERT_SCALE_DEFAULT),
                        "d_router_hidden": list(FMOE_ROUTE_SEARCH_DEFAULTS["d_router_hidden"]),
                        "stage_feature_dropout_prob": list(FMOE_ROUTE_SEARCH_DEFAULTS["stage_feature_dropout_prob"]),
                        "attn_dropout_prob": list(FMOE_ROUTE_SEARCH_DEFAULTS["attn_dropout_prob"]),
                        "route_consistency_lambda": list(FMOE_ROUTE_SEARCH_DEFAULTS["route_consistency_lambda"]),
                        "z_loss_lambda": list(FMOE_ROUTE_SEARCH_DEFAULTS["z_loss_lambda"]),
                    },
                    "fixed_context": fixed_context,
                }
            )

    if families and len(families) < bank_count:
        d_feat_cycle = [8, 12, 16, 24]
        hidden_cycle = [0.12, 0.15, 0.18, 0.2]
        family_drop_cycle = [0.0, 0.02, 0.04]
        seed_families = list(families)
        synthetic_idx = 0
        while len(families) < bank_count:
            base = dict(seed_families[synthetic_idx % len(seed_families)])
            fixed_context = dict(base.get("fixed_context") or {})
            fixed_context["d_feat_emb"] = d_feat_cycle[synthetic_idx % len(d_feat_cycle)]
            fixed_context["hidden_dropout_prob"] = hidden_cycle[synthetic_idx % len(hidden_cycle)]
            fixed_context["fixed_hidden_dropout_prob"] = hidden_cycle[synthetic_idx % len(hidden_cycle)]
            family_drop = family_drop_cycle[synthetic_idx % len(family_drop_cycle)]
            fixed_context["stage_family_dropout_prob"] = {"macro": family_drop, "mid": family_drop, "micro": family_drop}
            families.append(
                {
                    "dataset": dataset,
                    "model": ROUTE_MODEL,
                    "family": "route",
                    "family_id": f"R{len(families) + 1:02d}",
                    "source_family_id": f"{base.get('source_family_id', 'synthetic')}_X{synthetic_idx + 1:02d}",
                    "capacity_anchor": base.get("capacity_anchor", ""),
                    "history_valid": float(base.get("history_valid", 0.0) or 0.0),
                    "history_test": float(base.get("history_test", 0.0) or 0.0),
                    "search_space": dict(base.get("search_space") or {}),
                    "fixed_context": fixed_context,
                }
            )
            synthetic_idx += 1

    return {
        "dataset": dataset,
        "model": ROUTE_MODEL,
        "family": "route",
        "model_label": MODEL_LABELS[ROUTE_MODEL],
        "history_count": len(records),
        "history_best_valid": best_valid,
        "history_best_test": best_test,
        "runtime_median_sec": runtime_median,
        "stage1_bank_count": bank_count,
        "stage1_max_evals": STAGE1_ROUTE_MAX_EVALS[dataset],
        "stage2_max_evals": STAGE2_MAX_EVALS[dataset],
        "stage3_seed_count": STAGE3_SEED_COUNTS[dataset],
        "families": families[:bank_count],
        "overrides": route_base_overrides(),
    }


def build_space_manifest() -> Dict[str, Any]:
    pair_records, route_records = gather_history_by_pair()
    pair_specs: List[Dict[str, Any]] = []
    route_banks: List[Dict[str, Any]] = []

    for dataset in DEFAULT_DATASETS:
        for model in BASELINE_MODELS:
            pair_specs.append(build_baseline_pair_spec(dataset, model, pair_records.get((dataset, model), [])))
        route_bank = build_route_bank(dataset, route_records.get(dataset, []))
        route_banks.append(route_bank)
        pair_specs.append(
            {
                "dataset": dataset,
                "model": ROUTE_MODEL,
                "family": "route_summary",
                "model_label": MODEL_LABELS[ROUTE_MODEL],
                "history_count": route_bank["history_count"],
                "history_best_valid": route_bank["history_best_valid"],
                "history_best_test": route_bank["history_best_test"],
                "runtime_median_sec": route_bank["runtime_median_sec"],
                "stage1_bank_count": route_bank["stage1_bank_count"],
                "stage1_max_evals": route_bank["stage1_max_evals"],
                "stage2_max_evals": route_bank["stage2_max_evals"],
                "stage3_seed_count": route_bank["stage3_seed_count"],
                "search_bank_ref": dataset,
            }
        )

    server_runtime = {}
    for server_name, split in SERVER_MODEL_SPLITS.items():
        total = 0.0
        for spec in pair_specs:
            if spec["model"] in split["baseline_models"]:
                total += float(spec.get("runtime_median_sec", 0.0) or 0.0)
        if split.get("route_model"):
            total += sum(float(bank.get("runtime_median_sec", 0.0) or 0.0) for bank in route_banks)
        server_runtime[server_name] = round(total, 3)

    manifest = {
        "generated_at": now_utc(),
        "track": TRACK,
        "datasets": list(DEFAULT_DATASETS),
        "baseline_models": list(BASELINE_MODELS),
        "route_model": ROUTE_MODEL,
        "all_models": list(ALL_MODELS),
        "stage_tune_epochs": dict(STAGE_TUNE_EPOCHS),
        "stage_tune_patience": dict(STAGE_TUNE_PATIENCE),
        "pair_count": len(pair_specs),
        "baseline_pair_count": len([spec for spec in pair_specs if spec["model"] in BASELINE_MODELS]),
        "route_bank_count": len(route_banks),
        "pair_specs": pair_specs,
        "route_banks": route_banks,
        "server_splits": {
            "server_a": dict(SERVER_MODEL_SPLITS["server_a"], predicted_runtime_sec=server_runtime["server_a"]),
            "server_b": dict(SERVER_MODEL_SPLITS["server_b"], predicted_runtime_sec=server_runtime["server_b"]),
        },
    }
    return manifest


def tuning_space_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in list(manifest.get("pair_specs") or []):
        model = str(spec.get("model", ""))
        if model == ROUTE_MODEL:
            rows.append(
                {
                    "dataset": spec.get("dataset", ""),
                    "model": model,
                    "stage": "stage1",
                    "search_family": "route_bank",
                    "param": "bank_count",
                    "values_json": json.dumps([spec.get("stage1_bank_count", 0)], ensure_ascii=False),
                }
            )
            continue
        for key, values in dict(spec.get("search_space") or {}).items():
            rows.append(
                {
                    "dataset": spec.get("dataset", ""),
                    "model": model,
                    "stage": "stage1",
                    "search_family": "baseline",
                    "param": key,
                    "values_json": json.dumps(values, ensure_ascii=False),
                }
            )
    for bank in list(manifest.get("route_banks") or []):
        dataset = str(bank.get("dataset", ""))
        for family in list(bank.get("families") or []):
            for key, values in dict(family.get("search_space") or {}).items():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": ROUTE_MODEL,
                        "stage": "stage1",
                        "search_family": str(family.get("family_id", "")),
                        "param": key,
                        "values_json": json.dumps(values, ensure_ascii=False),
                    }
                )
    return rows


def write_space_outputs(manifest: Dict[str, Any], manifest_path: Path, tuning_space_path: Path, server_split_path: Path) -> None:
    write_json(manifest_path, manifest)
    write_csv_rows(
        tuning_space_path,
        tuning_space_rows(manifest),
        ["dataset", "model", "stage", "search_family", "param", "values_json"],
    )
    write_json(server_split_path, dict(manifest.get("server_splits") or {}))


def load_manifest(path: Path | str | None = None) -> Dict[str, Any]:
    target = Path(path) if path else DEFAULT_MANIFEST_PATH
    return read_json(target)


def manifest_pair_index(manifest: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for spec in list(manifest.get("pair_specs") or []):
        out[(str(spec.get("dataset", "")), str(spec.get("model", "")))] = spec
    return out


def manifest_route_index(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(bank.get("dataset", "")): bank for bank in list(manifest.get("route_banks") or [])}


def log_path_for_row(stage: str, row: Dict[str, Any]) -> Path:
    dataset = sanitize_token(row.get("dataset", ""), upper=False)
    model = sanitize_token(row.get("model", ""), upper=False)
    family = sanitize_token(row.get("family", ""), upper=False)
    job_id = sanitize_token(row.get("job_id", ""), upper=True)
    return LOG_ROOT / stage / dataset / model / family / f"{job_id}.log"


def stage_summary_path(stage: str) -> Path:
    return LOG_ROOT / stage / "summary.csv"


def stage_manifest_path(stage: str) -> Path:
    return LOG_ROOT / stage / "manifest.json"


def build_common_search_entries(search_space: Dict[str, Sequence[Any]], fixed_context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    search: Dict[str, Any] = {}
    types: Dict[str, str] = {}
    for key, values in dict(search_space or {}).items():
        if isinstance(values, (list, tuple, set)):
            ordered = dedupe_keep_order(list(values))
        else:
            ordered = [values]
        if not ordered:
            continue
        search[key] = ordered
        types[key] = "choice"
    for key, value in dict(fixed_context or {}).items():
        if key in search:
            continue
        search[key] = [value]
        types[key] = "choice"
    return search, types


def python_bin() -> str:
    candidate = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    if Path(candidate).exists():
        return candidate
    return sys.executable


def stage_tune_epochs(stage: str) -> int:
    return int(STAGE_TUNE_EPOCHS.get(str(stage), STAGE_TUNE_EPOCHS["stage3"]))


def stage_tune_patience(stage: str) -> int:
    return int(STAGE_TUNE_PATIENCE.get(str(stage), STAGE_TUNE_PATIENCE["stage3"]))


def model_oom_retry_limit(model: str, requested: int) -> int:
    return max(int(requested), int(MODEL_OOM_RETRY_MIN.get(str(model), DEFAULT_OOM_RETRY_LIMIT if requested <= 0 else requested)))


def build_baseline_command(row: Dict[str, Any], gpu_id: str, search_algo: str) -> List[str]:
    search, types = build_common_search_entries(row.get("search_space") or {}, row.get("fixed_context") or {})
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    oom_retry_limit = int(row.get("oom_retry_limit", 0) or 0)
    tune_epochs = stage_tune_epochs(str(row.get("stage", "")))
    tune_patience = stage_tune_patience(str(row.get("stage", "")))
    cmd = [
        python_bin(),
        "hyperopt_tune.py",
        "--config-name",
        dataset_config_name(str(row["dataset"])),
        "--search-algo",
        str(search_algo),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        str(tune_epochs),
        "--tune-patience",
        str(tune_patience),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        str(row["run_axis"]),
        "--run-phase",
        str(row["run_phase"]),
        f"model={row['model']}",
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
        "show_progress=false",
        f"++seed={int(row['runtime_seed'])}",
    ]
    if max_run_hours > 0.0:
        cmd.extend(["--max-run-hours", f"{max_run_hours:.6g}"])
    if oom_retry_limit > 0:
        cmd.extend(["--oom-retry-limit", str(oom_retry_limit)])
    for key, values in search.items():
        cmd.append(f"++search.{key}={hydra_literal(list(values))}")
        cmd.append(f"++search_space_type_overrides.{key}={types[key]}")
    return cmd


def build_route_command(row: Dict[str, Any], gpu_id: str, search_algo: str) -> List[str]:
    search, types = build_common_search_entries(row.get("search_space") or {}, row.get("fixed_context") or {})
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    oom_retry_limit = int(row.get("oom_retry_limit", 0) or 0)
    tune_epochs = stage_tune_epochs(str(row.get("stage", "")))
    tune_patience = stage_tune_patience(str(row.get("stage", "")))
    cmd = [
        python_bin(),
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(search_algo),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        str(tune_epochs),
        "--tune-patience",
        str(tune_patience),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        str(row["run_axis"]),
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
        f"++fmoe_architecture_id={hydra_literal('A12')}",
        f"++fmoe_architecture_key={hydra_literal('A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5')}",
        f"++fmoe_phase={hydra_literal(str(row['stage']).upper())}",
        f"++phase_run_type={hydra_literal(str(row['stage']))}",
        f"++phase_axis_id={hydra_literal(str(row['run_axis']))}",
        f"++phase_setting_id={hydra_literal(str(row.get('family_id', '')))}",
        f"++phase_hparam_id={hydra_literal(str(row.get('capacity_anchor', '')))}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(str(row['job_id']))}",
    ]
    if max_run_hours > 0.0:
        cmd.extend(["--max-run-hours", f"{max_run_hours:.6g}"])
    if oom_retry_limit > 0:
        cmd.extend(["--oom-retry-limit", str(oom_retry_limit)])
    overrides = dict(row.get("overrides") or {})
    for key, value in overrides.items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, values in search.items():
        cmd.append(f"++search.{key}={hydra_literal(list(values))}")
        cmd.append(f"++search_space_type_overrides.{key}={types[key]}")
    return cmd


RUN_STATUS_END_NORMAL_RE = re.compile(r"\[RUN_STATUS\]\s*END\s+status=normal\b", re.IGNORECASE)


def parse_result_path_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    pat = re.compile(r"Results\s*->\s*(.+)$")
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        match = pat.search(line.strip())
        if match:
            return Path(match.group(1).strip()).expanduser()
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
            terminate_active_children(grace_sec=0.2)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def parse_result_summary(result_path: Path) -> Dict[str, Any]:
    payload = load_result_payload(result_path)
    if not payload:
        return {}
    return result_summary_from_payload(payload)


def load_result_payload(result_path: Path) -> Dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        return read_json(result_path)
    except Exception:
        return {}


def result_summary_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not payload:
        return {}
    return {
        "valid_score": result_valid_mean(payload),
        "test_score": result_test_mean(payload),
        "best_valid_mrr20": safe_float((payload.get("best_valid_result") or {}).get("mrr@20", payload.get("best_mrr@20")), 0.0),
        "test_mrr20": safe_float((payload.get("test_result") or {}).get("mrr@20", payload.get("test_mrr@20")), 0.0),
    }


def result_has_successful_trials(payload: Dict[str, Any]) -> bool:
    if not payload:
        return False
    trials = list(payload.get("trials") or [])
    for trial in trials:
        if str((trial or {}).get("status", "")).strip().lower() == "ok":
            return True
    best_valid_result = payload.get("best_valid_result") or {}
    return bool(best_valid_result)


def build_summary_row(
    row: Dict[str, Any],
    *,
    gpu_id: str,
    status: str,
    result_path: str,
    log_path: Path,
    elapsed_sec: float,
    error: str,
) -> Dict[str, Any]:
    metrics = parse_result_summary(Path(result_path)) if result_path else {}
    return {
        "stage": row.get("stage", ""),
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "family": row.get("family", ""),
        "job_id": row.get("job_id", ""),
        "parent_job_id": row.get("parent_job_id", ""),
        "run_phase": row.get("run_phase", ""),
        "runtime_seed": int(row.get("runtime_seed", 0) or 0),
        "gpu_id": str(gpu_id),
        "status": status,
        "valid_score": metrics.get("valid_score", 0.0),
        "test_score": metrics.get("test_score", 0.0),
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "result_path": str(result_path or ""),
        "log_path": str(log_path),
        "elapsed_sec": float(elapsed_sec),
        "error": str(error or ""),
        "timestamp_utc": now_utc(),
    }


def describe_job(row: Dict[str, Any]) -> str:
    base = (
        f"stage={row.get('stage', '')} "
        f"dataset={row.get('dataset', '')} "
        f"model={row.get('model', '')} "
        f"job={row.get('job_id', '')} "
        f"max_evals={row.get('max_evals', '')}"
    )
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    if max_run_hours > 0.0:
        base += f" max_run_hours={max_run_hours:.3f}"
    oom_retry_limit = int(row.get("oom_retry_limit", 0) or 0)
    if oom_retry_limit > 0:
        base += f" oom_retries={oom_retry_limit}"
    if str(row.get("family", "")) == "route":
        base += (
            f" family={row.get('family_id', '')} "
            f"anchor={row.get('capacity_anchor', '')}"
        )
    return base


def run_one_job(row: Dict[str, Any], gpu_id: str, *, search_algo: str) -> Dict[str, Any]:
    log_path = log_path_for_row(str(row["stage"]), row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if str(row.get("family", "")).startswith("route"):
        cmd = build_route_command(row, gpu_id, search_algo)
    else:
        cmd = build_baseline_command(row, gpu_id, search_algo)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    start = time.time()
    rc = 1
    proc: subprocess.Popen[Any] | None = None
    print(
        f"[launch][gpu={gpu_id}] START {describe_job(row)} log={log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# job_id={row['job_id']} dataset={row['dataset']} model={row['model']} stage={row['stage']}\n")
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
    normal_end = has_run_status_end_normal(log_path)
    result_payload = load_result_payload(result_path_obj) if result_path_obj is not None else {}
    has_success = result_has_successful_trials(result_payload)
    status = "ok" if (rc == 0 and has_success and (normal_end or result_path_obj is not None)) else "fail"
    if status == "ok":
        error = ""
    elif rc == 0 and normal_end and not has_success:
        error = "completed_without_successful_trial"
    else:
        error = f"rc={rc} tail={extract_error_tail(log_path)}"
    summary = build_summary_row(
        row,
        gpu_id=gpu_id,
        status=status,
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=elapsed,
        error=error,
    )
    print(
        f"[launch][gpu={gpu_id}] END {describe_job(row)} "
        f"status={status} valid={summary.get('valid_score', 0.0):.6f} "
        f"test={summary.get('test_score', 0.0):.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def resumed_summary_row(row: Dict[str, Any]) -> Dict[str, Any] | None:
    log_path = log_path_for_row(str(row["stage"]), row)
    if not has_run_status_end_normal(log_path):
        return None
    result_path_obj = parse_result_path_from_log(log_path)
    if result_path_obj is None:
        return None
    result_payload = load_result_payload(result_path_obj)
    if not result_has_successful_trials(result_payload):
        return None
    return build_summary_row(
        row,
        gpu_id="resume",
        status="ok",
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=0.0,
        error="",
    )


def run_jobs(
    rows: List[Dict[str, Any]],
    *,
    stage: str,
    gpus: List[str],
    search_algo: str,
    resume_from_logs: bool,
    dry_run: bool,
) -> int:
    install_signal_handlers()
    summary_path = stage_summary_path(stage)
    if summary_path.exists():
        summary_path.unlink()

    if dry_run:
        for row in rows:
            append_csv_row(
                summary_path,
                SUMMARY_FIELDS,
                {
                    "stage": stage,
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "family": row.get("family", ""),
                    "job_id": row.get("job_id", ""),
                    "parent_job_id": row.get("parent_job_id", ""),
                    "run_phase": row.get("run_phase", ""),
                    "runtime_seed": int(row.get("runtime_seed", 0) or 0),
                    "gpu_id": "dry-run",
                    "status": "planned",
                    "valid_score": "",
                    "test_score": "",
                    "best_valid_mrr20": "",
                    "test_mrr20": "",
                    "result_path": "",
                    "log_path": str(log_path_for_row(stage, row)),
                    "elapsed_sec": 0.0,
                    "error": "",
                    "timestamp_utc": now_utc(),
                },
            )
        return 0

    pending: Queue[Dict[str, Any]] = Queue()
    for row in rows:
        if resume_from_logs:
            resumed = resumed_summary_row(row)
            if resumed is not None:
                print(
                    f"[resume] SKIP completed {describe_job(row)} "
                    f"valid={float(resumed.get('valid_score', 0.0) or 0.0):.6f} "
                    f"test={float(resumed.get('test_score', 0.0) or 0.0):.6f}",
                    flush=True,
                )
                append_csv_row(summary_path, SUMMARY_FIELDS, resumed)
                continue
        pending.put(row)

    if pending.empty():
        return 0

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))

    def worker() -> None:
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo=search_algo)
                append_csv_row(summary_path, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 0 if not STOP_EVENT.is_set() else 1


def load_stage_payloads(stage: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in read_csv_rows(stage_summary_path(stage)):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        result_path = Path(str(row.get("result_path", "")).strip())
        if not result_path.exists():
            continue
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        key = (normalize_dataset(row.get("dataset", "")), normalize_model(row.get("model", "")), str(row.get("job_id", "")))
        out[key] = {"summary": row, "payload": payload, "result_path": result_path}
    return out


def pick_top_configs_from_stage_payload(payload: Dict[str, Any], *, top_k: int) -> List[Dict[str, Any]]:
    return top_unique_trials(payload, top_k=top_k)


def narrow_space_from_configs(
    original_space: Dict[str, Sequence[Any]],
    configs: List[Dict[str, Any]],
    *,
    lr_points: int = 5,
    max_other: int = 4,
) -> Dict[str, List[Any]]:
    narrowed: Dict[str, List[Any]] = {}
    for key, values in dict(original_space or {}).items():
        ordered = list(values)
        if not ordered:
            continue
        if key == "learning_rate":
            selected = sorted({safe_float(cfg.get(key), 0.0) for cfg in configs if safe_float(cfg.get(key), 0.0) > 0.0})
            if not selected:
                narrowed[key] = list(ordered[:lr_points])
                continue
            center = math.exp(sum(math.log(v) for v in selected) / len(selected))
            nearest = sorted(ordered, key=lambda value: (abs(math.log(max(safe_float(value, 1e-8), 1e-8)) - math.log(center)), safe_float(value, 0.0)))
            picked = dedupe_keep_order(list(selected))
            target_size = max(int(lr_points), len(picked))
            for value in nearest:
                if value in picked:
                    continue
                picked.append(value)
                if len(picked) >= target_size:
                    break
            narrowed[key] = sorted(dedupe_keep_order(picked), key=lambda value: safe_float(value, 0.0))
            continue
        selected_values = [cfg.get(key) for cfg in configs if key in cfg]
        selected_tokens = {json.dumps(value, ensure_ascii=True, sort_keys=True) for value in selected_values}
        picked = dedupe_keep_order(selected_values)
        for value in ordered:
            token = json.dumps(value, ensure_ascii=True, sort_keys=True)
            if token in selected_tokens and token not in {json.dumps(v, ensure_ascii=True, sort_keys=True) for v in picked}:
                picked.append(value)
        if not picked:
            picked = list(ordered[: max_other])
        if len(picked) < min(len(ordered), max_other):
            seen_tokens = {json.dumps(v, ensure_ascii=True, sort_keys=True) for v in picked}
            for value in ordered:
                token = json.dumps(value, ensure_ascii=True, sort_keys=True)
                if token in seen_tokens:
                    continue
                picked.append(value)
                seen_tokens.add(token)
                if len(picked) >= min(len(ordered), max_other):
                    break
        narrowed[key] = dedupe_keep_order(picked)
    return narrowed


def selection_rows_to_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "dataset",
        "model",
        "config_rank",
        "seed_count",
        "mean_valid_score",
        "mean_test_score",
        "config_json",
        "result_paths_json",
    ]
    write_csv_rows(path, rows, fieldnames)
