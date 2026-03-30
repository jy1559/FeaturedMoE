#!/usr/bin/env python3
"""Launch baseline Final_all_datasets runs with wide hparam banks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
RESULT_ROOT = ARTIFACT_ROOT / "results" / "baseline"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "baseline"

TRACK = "baseline"
AXIS = "Final_all_datasets"
PHASE_ID = "P14"
PHASE_NAME = "FINAL_ALL_DATASETS"
AXIS_DESC = "final_all_datasets"


DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "amazon_beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

MODEL_SPECS = [
    {"model_option": "sasrec", "model_label": "SASRec"},
    {"model_option": "gru4rec", "model_label": "GRU4Rec"},
    {"model_option": "tisasrec", "model_label": "TiSASRec"},
    {"model_option": "duorec", "model_label": "DuoRec"},
    {"model_option": "sigma", "model_label": "SIGMA"},
    {"model_option": "bsarec", "model_label": "BSARec"},
    {"model_option": "fearec", "model_label": "FEARec"},
    {"model_option": "difsr", "model_label": "DIFSR"},
    {"model_option": "fame", "model_label": "FAME"},
]

HPARAM_BANK: Dict[str, Dict[str, Any]] = {
    # H1-H4: 기존 bank
    "H1": {
        "hidden_size": 128,
        "embedding_size": 128,
        "layers": 2,
        "heads": 4,
        "inner_size": 256,
        "dropout": 0.10,
        "weight_decay": 1e-4,
        "max_len": 20,
        "time_span": 256,
        "num_experts": 4,
        "sigma_state": 16,
        "sigma_kernel": 4,
        "sigma_remaining_ratio": 0.50,
    },
    "H2": {
        "hidden_size": 144,
        "embedding_size": 144,
        "layers": 2,
        "heads": 4,
        "inner_size": 288,
        "dropout": 0.12,
        "weight_decay": 1.5e-4,
        "max_len": 25,
        "time_span": 384,
        "num_experts": 6,
        "sigma_state": 24,
        "sigma_kernel": 6,
        "sigma_remaining_ratio": 0.60,
    },
    "H3": {
        "hidden_size": 160,
        "embedding_size": 160,
        "layers": 3,
        "heads": 4,
        "inner_size": 320,
        "dropout": 0.15,
        "weight_decay": 2e-4,
        "max_len": 30,
        "time_span": 512,
        "num_experts": 8,
        "sigma_state": 32,
        "sigma_kernel": 8,
        "sigma_remaining_ratio": 0.70,
    },
    "H4": {
        "hidden_size": 112,
        "embedding_size": 112,
        "layers": 1,
        "heads": 4,
        "inner_size": 224,
        "dropout": 0.25,
        "weight_decay": 5e-4,
        "max_len": 50,
        "time_span": 1024,
        "num_experts": 2,
        "sigma_state": 8,
        "sigma_kernel": 8,
        "sigma_remaining_ratio": 0.90,
    },
    # H5-H12: 확장 bank (모델별로 큰 dim/다른 정규화/같은 구조 다른 LR 중심)
    "H5": {
        "hidden_size": 192,
        "embedding_size": 192,
        "layers": 3,
        "heads": 4,
        "inner_size": 384,
        "dropout": 0.10,
        "weight_decay": 1e-4,
        "max_len": 24,
        "time_span": 384,
        "num_experts": 6,
        "sigma_state": 24,
        "sigma_kernel": 6,
        "sigma_remaining_ratio": 0.60,
    },
    "H6": {
        "hidden_size": 224,
        "embedding_size": 224,
        "layers": 3,
        "heads": 4,
        "inner_size": 448,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "max_len": 28,
        "time_span": 512,
        "num_experts": 8,
        "sigma_state": 32,
        "sigma_kernel": 8,
        "sigma_remaining_ratio": 0.65,
    },
    "H7": {
        "hidden_size": 96,
        "embedding_size": 96,
        "layers": 1,
        "heads": 2,
        "inner_size": 192,
        "dropout": 0.30,
        "weight_decay": 8e-4,
        "max_len": 50,
        "time_span": 256,
        "num_experts": 2,
        "sigma_state": 8,
        "sigma_kernel": 6,
        "sigma_remaining_ratio": 0.40,
    },
    "H8": {
        "hidden_size": 160,
        "embedding_size": 160,
        "layers": 2,
        "heads": 4,
        "inner_size": 320,
        "dropout": 0.20,
        "weight_decay": 3e-4,
        "max_len": 40,
        "time_span": 640,
        "num_experts": 4,
        "sigma_state": 16,
        "sigma_kernel": 10,
        "sigma_remaining_ratio": 0.80,
    },
    "H9": {
        "hidden_size": 144,
        "embedding_size": 144,
        "layers": 2,
        "heads": 4,
        "inner_size": 288,
        "dropout": 0.10,
        "weight_decay": 2e-4,
        "max_len": 25,
        "time_span": 384,
        "num_experts": 4,
        "sigma_state": 16,
        "sigma_kernel": 6,
        "sigma_remaining_ratio": 0.55,
    },
    "H10": {
        "hidden_size": 144,
        "embedding_size": 144,
        "layers": 2,
        "heads": 4,
        "inner_size": 288,
        "dropout": 0.15,
        "weight_decay": 1e-4,
        "max_len": 25,
        "time_span": 384,
        "num_experts": 6,
        "sigma_state": 24,
        "sigma_kernel": 6,
        "sigma_remaining_ratio": 0.60,
    },
    "H11": {
        "hidden_size": 160,
        "embedding_size": 160,
        "layers": 3,
        "heads": 4,
        "inner_size": 320,
        "dropout": 0.18,
        "weight_decay": 4e-4,
        "max_len": 30,
        "time_span": 512,
        "num_experts": 6,
        "sigma_state": 24,
        "sigma_kernel": 8,
        "sigma_remaining_ratio": 0.70,
    },
    "H12": {
        "hidden_size": 192,
        "embedding_size": 192,
        "layers": 2,
        "heads": 4,
        "inner_size": 384,
        "dropout": 0.12,
        "weight_decay": 1e-4,
        "max_len": 30,
        "time_span": 512,
        "num_experts": 8,
        "sigma_state": 32,
        "sigma_kernel": 8,
        "sigma_remaining_ratio": 0.75,
    },
}

# Model-specific hparam overlays to improve fairness/speed while keeping
# per-run LR-only search. Each H* remains "preset choice", not in-run search.
MODEL_HPARAM_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "sasrec": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "dropout": 0.10},
        "H2": {"layers": 2, "heads": 4, "max_len": 25, "dropout": 0.12},
        "H3": {"layers": 3, "heads": 4, "max_len": 30, "dropout": 0.15},
        "H4": {"layers": 1, "heads": 4, "max_len": 50, "dropout": 0.25, "hidden_size": 112, "embedding_size": 112},
    },
    "gru4rec": {
        "H1": {"layers": 1, "dropout": 0.10, "max_len": 20, "hidden_size": 128, "embedding_size": 128},
        "H2": {"layers": 2, "dropout": 0.15, "max_len": 25, "hidden_size": 144, "embedding_size": 144},
        "H3": {"layers": 2, "dropout": 0.20, "max_len": 30, "hidden_size": 160, "embedding_size": 160},
        "H4": {"layers": 1, "dropout": 0.30, "max_len": 50, "hidden_size": 112, "embedding_size": 112},
    },
    "tisasrec": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "time_span": 256, "dropout": 0.10},
        "H2": {"layers": 2, "heads": 4, "max_len": 24, "time_span": 320, "dropout": 0.12},
        "H3": {"layers": 2, "heads": 4, "max_len": 28, "time_span": 384, "dropout": 0.15},
        "H4": {"layers": 1, "heads": 2, "max_len": 24, "time_span": 256, "dropout": 0.22, "hidden_size": 96, "embedding_size": 96, "inner_size": 192},
    },
    "duorec": {
        "H1": {"layers": 1, "heads": 2, "max_len": 20, "dropout": 0.10, "hidden_size": 112, "embedding_size": 112, "inner_size": 224},
        "H2": {"layers": 2, "heads": 2, "max_len": 22, "dropout": 0.12, "hidden_size": 128, "embedding_size": 128, "inner_size": 256},
        "H3": {"layers": 2, "heads": 4, "max_len": 26, "dropout": 0.15, "hidden_size": 144, "embedding_size": 144, "inner_size": 288},
        "H4": {"layers": 2, "heads": 4, "max_len": 30, "dropout": 0.20, "hidden_size": 160, "embedding_size": 160, "inner_size": 320},
    },
    "sigma": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "dropout": 0.10, "sigma_state": 12, "sigma_kernel": 4, "sigma_remaining_ratio": 0.50},
        "H2": {"layers": 2, "heads": 4, "max_len": 24, "dropout": 0.12, "sigma_state": 16, "sigma_kernel": 6, "sigma_remaining_ratio": 0.60},
        "H3": {"layers": 3, "heads": 4, "max_len": 28, "dropout": 0.15, "sigma_state": 20, "sigma_kernel": 8, "sigma_remaining_ratio": 0.70},
        "H4": {"layers": 1, "heads": 4, "max_len": 36, "dropout": 0.20, "sigma_state": 8, "sigma_kernel": 10, "sigma_remaining_ratio": 0.35},
    },
    "bsarec": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "dropout": 0.10},
        "H2": {"layers": 2, "heads": 4, "max_len": 25, "dropout": 0.12},
        "H3": {"layers": 3, "heads": 4, "max_len": 30, "dropout": 0.15},
        "H4": {"layers": 1, "heads": 4, "max_len": 50, "dropout": 0.25},
    },
    "fearec": {
        "H1": {"layers": 2, "heads": 2, "max_len": 20, "dropout": 0.10, "hidden_size": 112, "embedding_size": 112, "inner_size": 224},
        "H2": {"layers": 2, "heads": 2, "max_len": 22, "dropout": 0.12, "hidden_size": 128, "embedding_size": 128, "inner_size": 256},
        "H3": {"layers": 2, "heads": 4, "max_len": 26, "dropout": 0.15, "hidden_size": 144, "embedding_size": 144, "inner_size": 288},
        "H4": {"layers": 3, "heads": 4, "max_len": 30, "dropout": 0.18, "hidden_size": 160, "embedding_size": 160, "inner_size": 320},
    },
    "difsr": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "dropout": 0.10},
        "H2": {"layers": 2, "heads": 4, "max_len": 25, "dropout": 0.12},
        "H3": {"layers": 3, "heads": 4, "max_len": 30, "dropout": 0.15},
        "H4": {"layers": 1, "heads": 4, "max_len": 50, "dropout": 0.25},
    },
    "fame": {
        "H1": {"layers": 2, "heads": 4, "max_len": 20, "dropout": 0.10, "num_experts": 2},
        "H2": {"layers": 2, "heads": 4, "max_len": 24, "dropout": 0.12, "num_experts": 3},
        "H3": {"layers": 2, "heads": 4, "max_len": 28, "dropout": 0.15, "num_experts": 4},
        "H4": {"layers": 3, "heads": 4, "max_len": 30, "dropout": 0.18, "num_experts": 6},
    },
}

DATASET_LR_BASE: Dict[str, tuple[float, float]] = {
    # Wide but sane defaults per dataset scale/sparsity.
    "KuaiRecLargeStrictPosV2_0.2": (2e-4, 8e-3),
    "lastfm0.03": (5e-5, 2e-3),
    "amazon_beauty": (5e-5, 7e-3),
    "foursquare": (5e-5, 7e-3),
    "movielens1m": (5e-5, 7e-3),
    "retail_rocket": (5e-5, 7e-3),
}

MODEL_LR_MULT: Dict[str, float] = {
    "sasrec": 1.00,
    "gru4rec": 1.45,
    "tisasrec": 0.78,
    "duorec": 0.52,
    "sigma": 0.44,
    "bsarec": 0.95,
    "fearec": 0.56,
    "difsr": 0.74,
    "fame": 0.62,
}

HPARAM_LR_MULT: Dict[str, float] = {
    "H1": 1.00,
    "H2": 0.92,
    "H3": 0.82,
    "H4": 0.70,
    "H5": 0.95,
    "H6": 0.80,
    "H7": 0.65,
    "H8": 0.75,
    "H9": 0.55,
    "H10": 1.30,
    "H11": 0.70,
    "H12": 1.15,
}

MODEL_LR_BAND_RATIO: Dict[str, float] = {
    "sasrec": 6.0,
    "gru4rec": 6.0,
    "tisasrec": 4.0,
    "duorec": 4.0,
    "sigma": 4.0,
    "bsarec": 6.0,
    "fearec": 4.0,
    "difsr": 6.0,
    "fame": 4.0,
}

MAX_AUTO_HPARAM_CAP = 12

# 모델별 관측 결과(현재 summary/log 기준)를 반영한 AUTO 우선순위.
# 앞쪽일수록 먼저 시도되며, --max-hparams-per-model 값으로 잘린다.
MODEL_HPARAM_PRIORITY: Dict[str, list[str]] = {
    "sasrec": ["H2", "H3", "H5", "H6", "H1", "H8", "H9", "H10", "H11", "H12", "H4", "H7"],
    "gru4rec": ["H3", "H2", "H1", "H5", "H9", "H10", "H8", "H11", "H12", "H4", "H6", "H7"],
    "tisasrec": ["H3", "H2", "H1", "H5", "H9", "H10", "H8", "H11", "H12", "H4", "H6", "H7"],
    "duorec": ["H1", "H2", "H3", "H5", "H9", "H10", "H8", "H11", "H12", "H4", "H6", "H7"],
    "sigma": ["H4", "H2", "H1", "H3", "H8", "H9", "H10", "H5", "H11", "H12", "H6", "H7"],
    "bsarec": ["H4", "H3", "H2", "H1", "H5", "H6", "H8", "H9", "H10", "H11", "H12", "H7"],
    "fearec": ["H1", "H2", "H3", "H5", "H9", "H10", "H8", "H11", "H12", "H4", "H6", "H7"],
    "difsr": ["H3", "H4", "H2", "H1", "H5", "H6", "H8", "H9", "H10", "H11", "H12", "H7"],
    "fame": ["H1", "H3", "H2", "H4", "H5", "H6", "H8", "H9", "H10", "H11", "H12", "H7"],
}

DATASET_COST_WEIGHT: Dict[str, float] = {
    "KuaiRecLargeStrictPosV2_0.2": 1.00,
    "lastfm0.03": 1.80,
    "amazon_beauty": 0.60,
    "foursquare": 0.80,
    "movielens1m": 1.20,
    "retail_rocket": 1.00,
}

MODEL_COST_WEIGHT: Dict[str, float] = {
    "sasrec": 1.00,
    "gru4rec": 0.80,
    "tisasrec": 1.35,
    "duorec": 1.55,
    "sigma": 1.45,
    "bsarec": 1.10,
    "fearec": 1.75,
    "difsr": 1.30,
    "fame": 1.70,
}

HPARAM_COST_WEIGHT: Dict[str, float] = {
    "H1": 1.00,
    "H2": 1.10,
    "H3": 1.20,
    "H4": 0.90,
    "H5": 1.30,
    "H6": 1.45,
    "H7": 0.85,
    "H8": 1.15,
    "H9": 1.00,
    "H10": 1.00,
    "H11": 1.20,
    "H12": 1.25,
}


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


def _parse_csv_strings(text: str) -> list[str]:
    return [tok.strip() for tok in str(text or "").split(",") if tok.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for tok in _parse_csv_strings(text):
        out.append(int(tok))
    return out


def _sanitize_token(text: str, *, upper: bool = True) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum():
            out.append(ch.upper() if upper else ch.lower())
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "X"


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _model_dir_name(model_label: str) -> str:
    raw = str(model_label or "").strip()
    if not raw:
        return "unknown_model"
    return raw.replace("/", "_")


def _dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    mapping = {
        "kuaireclargestrictposv2_0.2": "tune_kuai_strict_small",
        "lastfm0.03": "tune_lfm_small",
        "amazon_beauty": "tune_ab",
        "foursquare": "tune_fs",
        "movielens1m": "tune_ml",
        "retail_rocket": "tune_rr",
    }
    if key not in mapping:
        raise RuntimeError(f"Unsupported dataset for final baseline runner: {dataset}")
    return mapping[key]


def _phase_dataset_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / _dataset_tag(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_dataset_dir(dataset) / "summary.csv"


def _manifest_path(dataset: str, args: argparse.Namespace) -> Path:
    if str(args.manifest_out or "").strip():
        p = Path(str(args.manifest_out))
        return p.with_name(f"{p.name}_{dataset}.json")
    return _phase_dataset_dir(dataset) / "final_matrix.json"


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


def _is_completed_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        text = str(line).strip()
        if not text:
            continue
        if text == "[RUN_STATUS] END status=normal":
            return True
        return text.startswith("[RUN_STATUS] END status=normal ")
    return False


def _metric_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, str):
        try:
            v = float(value.strip())
        except Exception:
            return None
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def _extract_valid_mrr(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("best_mrr@20", "best_valid_mrr@20", "best_valid_score"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    bvr = payload.get("best_valid_result")
    if isinstance(bvr, dict):
        for key in ("mrr@20", "MRR@20"):
            val = _metric_to_float(bvr.get(key))
            if val is not None:
                return val
    return None


def _extract_test_mrr(payload: Dict[str, Any]) -> Optional[float]:
    for key in ("test_mrr@20", "best_test_mrr@20", "test_score"):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    tr = payload.get("test_result")
    if isinstance(tr, dict):
        for key in ("mrr@20", "MRR@20"):
            val = _metric_to_float(tr.get(key))
            if val is not None:
                return val
    return None


def _scan_result_index(dataset: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not RESULT_ROOT.exists():
        return out
    axis_lc = str(AXIS).strip().lower()
    for path in RESULT_ROOT.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload_axis = str(payload.get("run_axis", "")).strip().lower()
        if payload_axis != axis_lc:
            continue
        if str(payload.get("dataset", "")) != str(dataset):
            continue
        run_phase = str(payload.get("run_phase", "")).strip()
        if not run_phase:
            continue
        rec = {
            "run_phase": run_phase,
            "best_mrr": _extract_valid_mrr(payload),
            "test_mrr": _extract_test_mrr(payload),
            "n_completed": int(payload.get("n_completed", 0) or 0),
            "interrupted": bool(payload.get("interrupted", False)),
            "path": str(path.resolve()),
            "mtime": float(path.stat().st_mtime),
        }
        prev = out.get(run_phase)
        if prev is None or float(rec["mtime"]) >= float(prev.get("mtime", 0.0)):
            out[run_phase] = rec
    return out


def _get_result_row(dataset: str, run_phase: str, retries: int = 10, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        rec = _scan_result_index(dataset).get(str(run_phase))
        if isinstance(rec, dict):
            return rec
        time.sleep(max(float(sleep_sec), 0.0))
    return None


def _extract_result_json_path_from_log(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for line in reversed(lines):
        text = str(line).strip()
        if "Results ->" in text:
            return text.split("Results ->", 1)[1].strip()
        if "Normal mirror ->" in text:
            return text.split("Normal mirror ->", 1)[1].strip()
    return ""


def _result_row_from_json(path_text: str, run_phase: str) -> Optional[Dict[str, Any]]:
    p = Path(str(path_text or "")).expanduser()
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload_run_phase = str(payload.get("run_phase", "")).strip()
    if payload_run_phase and payload_run_phase != str(run_phase):
        return None
    return {
        "run_phase": str(run_phase),
        "best_mrr": _extract_valid_mrr(payload),
        "test_mrr": _extract_test_mrr(payload),
        "n_completed": int(payload.get("n_completed", 0) or 0),
        "interrupted": bool(payload.get("interrupted", False)),
        "path": str(p.resolve()),
        "mtime": float(p.stat().st_mtime),
    }


def _get_result_row_from_log_or_scan(
    *,
    dataset: str,
    run_phase: str,
    log_path: Path,
    retries: int = 4,
    sleep_sec: float = 0.75,
) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        result_path = _extract_result_json_path_from_log(log_path)
        if result_path:
            rec = _result_row_from_json(result_path, run_phase)
            if rec:
                return rec
        time.sleep(max(float(sleep_sec), 0.0))
    return _get_result_row(dataset=dataset, run_phase=run_phase, retries=2, sleep_sec=0.5)


def _verify_special_from_result(result_json_path: str) -> tuple[bool, str]:
    path = Path(str(result_json_path or "")).expanduser()
    if not path.exists():
        return False, f"result_missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"result_parse_error:{exc}"

    special_result_file = str(payload.get("special_result_file", "") or "").strip()
    special_log_file = str(payload.get("special_log_file", "") or "").strip()
    special_result_ok = bool(special_result_file) and Path(special_result_file).exists()
    special_log_ok = bool(special_log_file) and Path(special_log_file).exists()
    ok = bool(special_result_ok and special_log_ok)
    detail = (
        f"special_result_ok={special_result_ok} special_log_ok={special_log_ok} "
        f"special_result_file={special_result_file} special_log_file={special_log_file}"
    )
    return ok, detail


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
        "result_path",
        "timestamp_utc",
    ]


def _ensure_summary_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_fieldnames()
    expected = set(fieldnames)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                existing_fields = [f for f in (reader.fieldnames or []) if f]
                existing_rows = list(reader)
        except Exception:
            existing_fields = []
            existing_rows = []
        if set(existing_fields) == expected:
            return
        backup = path.with_name(f"{path.stem}.legacy_{int(time.time())}{path.suffix}")
        try:
            path.rename(backup)
        except Exception:
            pass
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                payload = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(payload)
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


def _load_summary_bests(path: Path) -> tuple[Optional[float], Optional[float], Dict[str, float], Dict[str, float]]:
    if not path.exists():
        return None, None, {}, {}
    global_best_valid = None
    global_best_test = None
    model_best_valid: Dict[str, float] = {}
    model_best_test: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                model = str(row.get("model", "") or "").strip()
                run_valid = _metric_to_float(row.get("run_best_valid_mrr20"))
                run_test = _metric_to_float(row.get("run_best_test_mrr20"))
                if run_test is None:
                    # Backward compatibility with legacy summary format.
                    run_test = _metric_to_float(row.get("test_mrr20"))
                model_valid = _metric_to_float(row.get("model_best_valid_mrr20"))
                model_test = _metric_to_float(row.get("model_best_test_mrr20"))
                global_valid = _metric_to_float(row.get("global_best_valid_mrr20"))
                global_test = _metric_to_float(row.get("global_best_test_mrr20"))

                if run_valid is not None:
                    global_best_valid = (
                        run_valid if global_best_valid is None else max(float(global_best_valid), float(run_valid))
                    )
                    if model:
                        prev = model_best_valid.get(model)
                        model_best_valid[model] = run_valid if prev is None else max(float(prev), float(run_valid))
                if run_test is not None:
                    global_best_test = (
                        run_test if global_best_test is None else max(float(global_best_test), float(run_test))
                    )
                    if model:
                        prev = model_best_test.get(model)
                        model_best_test[model] = run_test if prev is None else max(float(prev), float(run_test))
                if model and model_valid is not None:
                    prev = model_best_valid.get(model)
                    model_best_valid[model] = model_valid if prev is None else max(float(prev), float(model_valid))
                if model and model_test is not None:
                    prev = model_best_test.get(model)
                    model_best_test[model] = model_test if prev is None else max(float(prev), float(model_test))
                if global_valid is not None:
                    global_best_valid = (
                        global_valid
                        if global_best_valid is None
                        else max(float(global_best_valid), float(global_valid))
                    )
                if global_test is not None:
                    global_best_test = (
                        global_test
                        if global_best_test is None
                        else max(float(global_best_test), float(global_test))
                    )
    except Exception:
        return None, None, {}, {}
    return global_best_valid, global_best_test, model_best_valid, model_best_test


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path)
    payload = {k: row.get(k, "") for k in _summary_fieldnames()}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writerow(payload)


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, args: argparse.Namespace, cmd: list[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"model={row.get('model_label','')} hparam_id={row.get('hparam_id','')} seed={row.get('seed_id','')}"
        ),
        f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={getattr(args, 'max_evals', '')} tune_epochs={getattr(args, 'tune_epochs', '')} tune_patience={getattr(args, 'tune_patience', '')}",
        f"seed={row.get('runtime_seed','')}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _build_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    hparam_id = _sanitize_token(str(row["hparam_id"]), upper=True)
    model_dir = _model_dir_name(str(row["model_label"]))
    model_token = _sanitize_token(str(row["model_label"]), upper=True)
    seed_id = int(row["seed_id"])
    filename = f"{PHASE_ID}_FAD_{AXIS_DESC}_{model_token}_{hparam_id}_S{seed_id}.log"
    return ds_dir / model_dir / hparam_id / filename


def _legacy_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    hparam_id = _sanitize_token(str(row["hparam_id"]), upper=True)
    model_token = _sanitize_token(str(row["model_label"]), upper=True)
    seed_id = int(row["seed_id"])
    filename = f"{PHASE_ID}_FAD_{AXIS_DESC}_{model_token}_{hparam_id}_S{seed_id}.log"
    return ds_dir / hparam_id / filename


def _is_completed_any_log(row: Dict[str, Any], *, use_resume: bool) -> bool:
    if not use_resume:
        return False
    log_paths = [Path(str(row.get("log_path", ""))), _legacy_log_path(row)]
    for p in log_paths:
        if _is_completed_log(p):
            return True
    return False


def _logging_hparam_dir(row: Dict[str, Any]) -> Path:
    dataset = _dataset_tag(str(row.get("dataset", "")))
    model_dir = _model_dir_name(str(row.get("model_label", "")))
    hparam_id = _sanitize_token(str(row.get("hparam_id", "")), upper=True)
    return ARTIFACT_ROOT / "logging" / "baseline" / AXIS / dataset / model_dir / hparam_id


def _mirror_logging_bundle(row: Dict[str, Any], result_json_path: str) -> None:
    path = Path(str(result_json_path or "")).expanduser()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return

    bundle_dir = Path(str(payload.get("logging_bundle_dir", "") or "").strip()).expanduser()
    if not bundle_dir.exists():
        return

    dst_root = _logging_hparam_dir(row)
    dst_root.mkdir(parents=True, exist_ok=True)
    link_name = dst_root / bundle_dir.name
    if link_name.exists():
        return
    try:
        link_name.symlink_to(bundle_dir, target_is_directory=True)
    except Exception:
        # Fallback pointer file when symlink is unavailable.
        pointer = dst_root / f"{bundle_dir.name}.path.txt"
        pointer.write_text(str(bundle_dir.resolve()) + "\n", encoding="utf-8")


def _compute_lr_space(dataset: str, model_option: str, hparam_id: str) -> tuple[float, float]:
    base_lo, base_hi = DATASET_LR_BASE[str(dataset)]
    model_mult = float(MODEL_LR_MULT.get(str(model_option), 1.0))
    h_mult = float(HPARAM_LR_MULT.get(str(hparam_id).upper(), 1.0))
    # Keep LR band intentionally narrow for max_evals=10 regime.
    center = math.sqrt(float(base_lo) * float(base_hi)) * model_mult * h_mult
    band_ratio = float(MODEL_LR_BAND_RATIO.get(str(model_option), 6.0))
    lo = max(2e-5, center / math.sqrt(band_ratio))
    hi = min(6e-3, center * math.sqrt(band_ratio))
    if hi <= lo:
        hi = min(6e-3, lo * 2.5)
    return lo, hi


def _effective_model_hparams(row: Dict[str, Any]) -> Dict[str, Any]:
    """Apply model-specific overlays and safety caps on top of shared H* bank."""
    model_option = str(row["model_option"]).lower()
    hparam_id = str(row["hparam_id"]).upper()
    h = dict(HPARAM_BANK[hparam_id])
    model_overlay = MODEL_HPARAM_OVERRIDES.get(model_option, {}).get(hparam_id, {})
    h.update(model_overlay)

    if model_option == "fearec":
        # RecBole FEARec internally assumes >1 layer in HybridAttention slide-step path.
        h["layers"] = max(2, int(h.get("layers", 2)))
        h["max_len"] = min(int(h.get("max_len", 30)), 30)
        h["hidden_size"] = min(int(h.get("hidden_size", 128)), 160)
        h["embedding_size"] = min(int(h.get("embedding_size", h["hidden_size"])), 160)
        h["heads"] = 2 if int(h.get("hidden_size", 128)) <= 128 else 4
        h["inner_size"] = max(int(h.get("hidden_size", 128)) * 2, int(h.get("inner_size", 256)))

    if model_option == "tisasrec":
        # TiSASRec has O(B*L^2*H) time-relation tensors; cap long outlier length.
        h["max_len"] = min(int(h.get("max_len", 30)), 28)
        h["time_span"] = min(int(h.get("time_span", 512)), 384)
        h["layers"] = min(int(h.get("layers", 2)), 2)
        h["hidden_size"] = min(int(h.get("hidden_size", 128)), 160)
        h["embedding_size"] = min(int(h.get("embedding_size", h["hidden_size"])), 160)

    if model_option == "duorec":
        h["layers"] = min(int(h.get("layers", 2)), 2)
        h["max_len"] = min(int(h.get("max_len", 30)), 30)
        h["hidden_size"] = min(int(h.get("hidden_size", 128)), 160)
        h["embedding_size"] = min(int(h.get("embedding_size", h["hidden_size"])), 160)
        h["heads"] = 2 if int(h.get("hidden_size", 128)) <= 128 else 4
        h["inner_size"] = max(int(h.get("hidden_size", 128)) * 2, int(h.get("inner_size", 256)))

    if model_option == "sigma":
        h["max_len"] = min(int(h.get("max_len", 30)), 32)
        h["layers"] = min(int(h.get("layers", 2)), 3)

    if model_option == "fame":
        h["num_experts"] = min(max(int(h.get("num_experts", 4)), 2), 6)

    return h


def _model_algorithm_overrides(row: Dict[str, Any]) -> list[str]:
    model_option = str(row["model_option"]).lower()
    hparam_id = str(row["hparam_id"]).upper()

    if model_option == "duorec":
        if hparam_id in {"H1", "H2", "H5", "H7", "H9"}:
            return ["++contrast=un", "++tau=0.2", "++lmd=0.04", "++lmd_sem=0.0", "++semantic_sample_max_tries=2"]
        if hparam_id in {"H3", "H6", "H8", "H11"}:
            return ["++contrast=su", "++tau=0.45", "++lmd=0.0", "++lmd_sem=0.06", "++semantic_sample_max_tries=2"]
        if hparam_id in {"H10"}:
            return ["++contrast=un", "++tau=0.3", "++lmd=0.06", "++lmd_sem=0.0", "++semantic_sample_max_tries=2"]
        return ["++contrast=us_x", "++tau=0.8", "++lmd=0.1", "++lmd_sem=0.08", "++semantic_sample_max_tries=3"]

    if model_option == "fearec":
        if hparam_id in {"H1", "H2", "H5", "H7", "H9"}:
            return ["++contrast=un", "++tau=0.2", "++lmd=0.04", "++lmd_sem=0.0", "++global_ratio=0.85", "++semantic_sample_max_tries=2"]
        if hparam_id in {"H3", "H6", "H8", "H11"}:
            return ["++contrast=su", "++tau=0.45", "++lmd=0.0", "++lmd_sem=0.06", "++global_ratio=1.0", "++semantic_sample_max_tries=2"]
        if hparam_id in {"H10"}:
            return ["++contrast=un", "++tau=0.3", "++lmd=0.06", "++lmd_sem=0.0", "++global_ratio=0.9", "++semantic_sample_max_tries=2"]
        return ["++contrast=us_x", "++tau=0.8", "++lmd=0.1", "++lmd_sem=0.08", "++global_ratio=1.0", "++semantic_sample_max_tries=3"]

    if model_option == "difsr":
        if hparam_id in {"H1", "H2", "H5", "H9"}:
            return ["++fusion_type=sum", "++use_attribute_predictor=true", "++lambda_attr=0.10"]
        if hparam_id in {"H3", "H6", "H8", "H11"}:
            return ["++fusion_type=gate", "++use_attribute_predictor=true", "++lambda_attr=0.15"]
        return ["++fusion_type=concat", "++use_attribute_predictor=false", "++lambda_attr=0.0"]

    if model_option == "fame":
        if hparam_id in {"H1", "H2", "H5", "H9"}:
            return ["++num_experts=2"]
        if hparam_id in {"H3", "H6", "H8", "H11"}:
            return ["++num_experts=3"]
        if hparam_id in {"H10", "H12"}:
            return ["++num_experts=4"]
        return ["++num_experts=5"]

    return []


def _model_runtime_resource_overrides(row: Dict[str, Any]) -> list[str]:
    model_option = str(row["model_option"]).lower()
    hparam_id = str(row["hparam_id"]).upper()
    h = _effective_model_hparams(row)
    hidden_size = int(h.get("hidden_size", 128))

    def _scale_by_width(bs: int) -> int:
        factor = 1.0
        if hidden_size >= 224:
            factor = 0.55
        elif hidden_size >= 192:
            factor = 0.70
        elif hidden_size >= 160:
            factor = 0.85
        return max(256, int(bs * factor))

    if model_option == "sasrec":
        train_bs = {"H1": 4096, "H2": 3584, "H3": 3072, "H4": 3072}.get(hparam_id, 3072)
        eval_bs = {"H1": 6144, "H2": 5120, "H3": 4096, "H4": 4096}.get(hparam_id, 4096)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "gru4rec":
        train_bs = {"H1": 8192, "H2": 7168, "H3": 6144, "H4": 6144}.get(hparam_id, 6144)
        eval_bs = {"H1": 12288, "H2": 10240, "H3": 8192, "H4": 8192}.get(hparam_id, 8192)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "tisasrec":
        train_bs = {"H1": 1280, "H2": 1024, "H3": 768, "H4": 512}.get(hparam_id, 768)
        eval_bs = {"H1": 1792, "H2": 1536, "H3": 1024, "H4": 768}.get(hparam_id, 1024)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "fearec":
        train_bs = {"H1": 896, "H2": 768, "H3": 640, "H4": 512}.get(hparam_id, 640)
        eval_bs = {"H1": 1280, "H2": 1024, "H3": 896, "H4": 768}.get(hparam_id, 896)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "duorec":
        # DuoRec contrastive InfoNCE builds O(B^2) similarity matrices; very large
        # batches can be slower overall than moderate batches.
        train_bs = {"H1": 896, "H2": 768, "H3": 640, "H4": 640}.get(hparam_id, 640)
        eval_bs = {"H1": 1280, "H2": 1024, "H3": 896, "H4": 896}.get(hparam_id, 896)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "sigma":
        train_bs = {"H1": 1536, "H2": 1280, "H3": 1024, "H4": 896}.get(hparam_id, 1024)
        eval_bs = {"H1": 2048, "H2": 1792, "H3": 1536, "H4": 1280}.get(hparam_id, 1536)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "bsarec":
        train_bs = {"H1": 3072, "H2": 2560, "H3": 2048, "H4": 2048}.get(hparam_id, 2048)
        eval_bs = {"H1": 4096, "H2": 3584, "H3": 3072, "H4": 3072}.get(hparam_id, 3072)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "difsr":
        train_bs = {"H1": 3072, "H2": 2560, "H3": 2048, "H4": 2048}.get(hparam_id, 2048)
        eval_bs = {"H1": 4096, "H2": 3584, "H3": 3072, "H4": 3072}.get(hparam_id, 3072)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    if model_option == "fame":
        train_bs = {"H1": 1024, "H2": 896, "H3": 768, "H4": 640}.get(hparam_id, 768)
        eval_bs = {"H1": 1536, "H2": 1280, "H3": 1024, "H4": 896}.get(hparam_id, 1024)
        train_bs = _scale_by_width(int(train_bs))
        eval_bs = _scale_by_width(int(eval_bs))
        return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]

    return []


def _base_hparam_overrides(row: Dict[str, Any]) -> list[str]:
    h = _effective_model_hparams(row)
    model_option = str(row["model_option"]).lower()
    o: list[str] = []
    hidden_size = int(h["hidden_size"])
    embedding_size = int(h["embedding_size"])
    layers = int(h["layers"])
    heads = int(h["heads"])
    inner_size = int(h["inner_size"])
    max_len = int(h["max_len"])

    o.extend(
        [
            f"++hidden_size={hidden_size}",
            f"++embedding_size={embedding_size}",
            f"++MAX_ITEM_LIST_LENGTH={max_len}",
        ]
    )

    if model_option in {"sasrec", "tisasrec", "duorec", "fearec"}:
        o.extend(
            [
                f"++n_layers={layers}",
                f"++num_layers={layers}",
                f"++n_heads={heads}",
                f"++num_heads={heads}",
                f"++inner_size={inner_size}",
            ]
        )
        if model_option == "tisasrec":
            o.append(f"++time_span={int(h['time_span'])}")
    elif model_option in {"bsarec", "fame", "difsr"}:
        o.extend(
            [
                f"++num_layers={layers}",
                f"++n_layers={layers}",
                f"++num_heads={heads}",
                f"++n_heads={heads}",
                f"++inner_size={inner_size}",
            ]
        )
        if model_option == "fame":
            o.append(f"++num_experts={int(h['num_experts'])}")
        if model_option == "difsr":
            o.append(f"++attribute_hidden_size={hidden_size}")
    elif model_option == "gru4rec":
        o.extend(
            [
                f"++num_layers={layers}",
            ]
        )
    elif model_option == "sigma":
        o.extend(
            [
                f"++num_layers={layers}",
                f"++inner_size={inner_size}",
                f"++state_size={int(h['sigma_state'])}",
                f"++conv_kernel={int(h['sigma_kernel'])}",
                f"++remaining_ratio={float(h['sigma_remaining_ratio'])}",
            ]
        )
    return o


def _model_tune_budget(row: Dict[str, Any], args: argparse.Namespace) -> tuple[int, int]:
    model_option = str(row["model_option"]).lower()
    hparam_id = str(row["hparam_id"]).upper()
    epochs = int(args.tune_epochs)
    patience = int(args.tune_patience)

    if model_option in {"duorec", "fearec"}:
        if hparam_id in {"H1", "H2"}:
            epochs = min(epochs, 40)
            patience = min(patience, 6)
        else:
            epochs = min(epochs, 48)
            patience = min(patience, 7)
    elif model_option == "tisasrec":
        epochs = min(epochs, 50)
        patience = min(patience, 7)
    elif model_option in {"fame", "sigma"}:
        epochs = min(epochs, 52)
        patience = min(patience, 7)

    epochs = max(1, int(epochs))
    patience = max(1, min(int(patience), int(epochs)))
    return epochs, patience


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    model_option = str(row["model_option"]).lower()
    h = _effective_model_hparams(row)
    lr_lo, lr_hi = _compute_lr_space(str(row["dataset"]), model_option, str(row["hparam_id"]))
    layers = int(h["layers"])
    heads = int(h["heads"])
    dropout = float(h["dropout"])
    weight_decay = float(h["weight_decay"])
    tune_epochs, tune_patience = _model_tune_budget(row, args)
    search_lr_only = {"learning_rate": [float(lr_lo), float(lr_hi)]}
    search_type_lr_only = {"learning_rate": "loguniform"}

    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        _dataset_config_name(str(row["dataset"])),
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(tune_epochs)),
        "--tune-patience",
        str(int(tune_patience)),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
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
        f"++search={hydra_literal(search_lr_only)}",
        f"++search_space_type_overrides={hydra_literal(search_type_lr_only)}",
        f"++weight_decay={weight_decay}",
    ]

    # Force all known non-LR knobs to singleton search values so Hydra search merge
    # cannot reintroduce model-specific tuning keys from dataset defaults.
    fixed_search: Dict[str, Any] = {
        "weight_decay": weight_decay,
        "n_layers": layers,
        "num_layers": layers,
        "n_heads": heads,
        "num_heads": heads,
        "dropout_ratio": dropout,
        "dropout_prob": dropout,
        "hidden_dropout_prob": dropout,
        "attn_dropout_prob": dropout,
        "hidden_size": int(h["hidden_size"]),
        "embedding_size": int(h["embedding_size"]),
        "inner_size": int(h["inner_size"]),
        "time_span": int(h["time_span"]),
        "num_experts": int(h["num_experts"]),
        "state_size": int(h["sigma_state"]),
        "conv_kernel": int(h["sigma_kernel"]),
        "remaining_ratio": float(h["sigma_remaining_ratio"]),
    }
    for key, value in fixed_search.items():
        cmd.append(f"++search.{key}={hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    if model_option == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    cmd.extend(_model_algorithm_overrides(row))
    cmd.extend(_model_runtime_resource_overrides(row))
    cmd.extend(_base_hparam_overrides(row))
    return cmd


def _dataset_estimated_cost(dataset: str, model_option: str, hparam_id: str) -> float:
    return (
        float(DATASET_COST_WEIGHT.get(str(dataset), 1.0))
        * float(MODEL_COST_WEIGHT.get(str(model_option), 1.0))
        * float(HPARAM_COST_WEIGHT.get(str(hparam_id).upper(), 1.0))
    )


def _history_ranked_hparams(dataset: str, model_label: str, ordered_hparams: list[str]) -> list[str]:
    """Re-rank candidate hparams using existing per-dataset summary (best-first)."""
    summary = _summary_path(dataset)
    if not summary.exists():
        return ordered_hparams

    best_by_h: Dict[str, float] = {}
    try:
        with summary.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if str(row.get("model", "")).strip() != str(model_label):
                    continue
                hid = str(row.get("hparam_id", "")).strip().upper()
                if hid not in ordered_hparams:
                    continue
                score = _metric_to_float(row.get("run_best_valid_mrr20"))
                if score is None:
                    continue
                prev = best_by_h.get(hid)
                best_by_h[hid] = float(score) if prev is None else max(float(prev), float(score))
    except Exception:
        return ordered_hparams

    if not best_by_h:
        return ordered_hparams

    pos = {hid: idx for idx, hid in enumerate(ordered_hparams)}
    ranked = sorted(
        ordered_hparams,
        key=lambda hid: (
            0 if hid in best_by_h else 1,
            -float(best_by_h.get(hid, -1.0)),
            pos[hid],
        ),
    )
    return ranked


def _build_rows(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    requested_hparams = [hid.upper() for hid in _parse_csv_strings(args.hparams)]
    if not requested_hparams:
        raise RuntimeError(f"No valid hparams selected: {args.hparams}")
    max_hparams_per_model = max(1, min(int(getattr(args, "max_hparams_per_model", MAX_AUTO_HPARAM_CAP)), MAX_AUTO_HPARAM_CAP))

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    ds_tag = _sanitize_token(dataset, upper=True)
    for model_order, spec in enumerate(MODEL_SPECS, start=1):
        model_label = str(spec["model_label"])
        model_option = str(spec["model_option"])
        model_hparam_order = list(MODEL_HPARAM_PRIORITY.get(model_option, [])) or list(HPARAM_BANK.keys())
        model_hparam_order = [hid for hid in model_hparam_order if hid in HPARAM_BANK]
        model_hparam_order = _history_ranked_hparams(dataset, model_label, model_hparam_order)
        model_auto_limit = max_hparams_per_model
        if any(hid == "AUTO" for hid in requested_hparams):
            selected_hparams = model_hparam_order[:model_auto_limit]
        else:
            selected_hparams = [hid for hid in model_hparam_order if hid in requested_hparams]
            if not selected_hparams:
                selected_hparams = [hid for hid in requested_hparams if hid in HPARAM_BANK][:max_hparams_per_model]
            else:
                selected_hparams = selected_hparams[:max_hparams_per_model]
        if not selected_hparams:
            continue

        for h_idx, hid in enumerate(selected_hparams, start=1):
            for seed_id in seeds:
                run_cursor += 1
                run_id = f"FAD_{ds_tag}_{_sanitize_token(model_label, upper=True)}_{hid}_S{int(seed_id)}"
                run_phase = (
                    f"{PHASE_ID}_FAD_D{dataset_order_idx:02d}_M{model_order:02d}_"
                    f"{_sanitize_token(model_label, upper=True)}_{hid}_S{int(seed_id)}"
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "phase_id": PHASE_ID,
                        "axis_id": "FAD",
                        "axis_desc": AXIS_DESC,
                        "setting_id": f"FINAL_BASELINE_{_sanitize_token(model_label, upper=True)}_{hid}_S{int(seed_id)}",
                        "setting_key": "FINAL_BASELINE_WIDE_HPARAM",
                        "setting_desc": f"FINAL_BASELINE_WIDE_HPARAM_{_sanitize_token(model_label, upper=True)}_{hid}_S{int(seed_id)}",
                        "hparam_id": hid,
                        "seed_id": int(seed_id),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                        "stage": "final",
                        "model_option": model_option,
                        "model_label": model_label,
                        "estimated_cost": _dataset_estimated_cost(dataset, model_option, hid),
                    }
                )
    return rows


def _plan_gpu_bins(rows: list[Dict[str, Any]], gpus: list[str]) -> Dict[str, deque]:
    bins = {gpu: deque() for gpu in gpus}
    loads = {gpu: 0.0 for gpu in gpus}
    for idx, row in enumerate(rows, start=1):
        gpu = min(gpus, key=lambda gid: (loads[gid], gid))
        row["assigned_gpu"] = gpu
        row["assigned_order"] = idx
        bins[gpu].append(row)
        loads[gpu] += float(row.get("estimated_cost", 1.0) or 0.0)
    return bins


def _write_manifest(dataset: str, args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(dataset, args)
    requested_hparams = [hid.upper() for hid in _parse_csv_strings(args.hparams)]
    if any(hid == "AUTO" for hid in requested_hparams):
        hparams = sorted({str(r.get("hparam_id", "")).upper() for r in rows if str(r.get("hparam_id", "")).upper() in HPARAM_BANK})
        run_formula = f"{len(rows)} planned rows (model-adaptive AUTO presets)"
    else:
        hparams = [hid for hid in requested_hparams if hid in HPARAM_BANK]
        run_formula = f"{len(MODEL_SPECS)} x {len(hparams)} x {len(_parse_csv_ints(args.seeds))}"
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "dataset": dataset,
        "execution_type": "final",
        "model_count": len(MODEL_SPECS),
        "hparam_count": len(hparams),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "run_count_formula": run_formula,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "selected_hparams": hparams,
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "model": r["model_label"],
                "hparam_id": r["hparam_id"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _smoke_trim_rows(rows: list[Dict[str, Any]], max_runs: int) -> list[Dict[str, Any]]:
    limit = max(int(max_runs), 1)
    return rows[:limit]


def _terminate_active(active: Dict[str, Dict[str, Any]]) -> None:
    for slot in active.values():
        proc = slot.get("proc")
        if proc is None:
            continue
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


def _update_baseline_phase_summary(dataset: str, phase: str) -> None:
    script = RUN_DIR / "baseline" / "update_phase_summary.py"
    py = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    try:
        subprocess.run(
            [py, str(script), "--dataset", str(dataset), "--phase", str(phase)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _check_runtime_models() -> None:
    py = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    script = r"""
import recbole_patch  # noqa: F401
from recbole.utils import utils as rbu
models = ["SASRec","GRU4Rec","TiSASRec","DuoRec","SIGMA","BSARec","FEARec","DIFSR","FAME"]
for name in models:
    _ = rbu.get_model(name)
print("[ENV_CHECK] baseline final runner model registration OK")
"""
    proc = subprocess.run([py, "-c", script], cwd=EXP_DIR, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Model registration check failed:\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P14 Final all-datasets launcher (wide hparam)")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=120000)
    parser.add_argument("--hparams", default="AUTO")
    parser.add_argument("--max-hparams-per-model", type=int, default=12)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=60)
    parser.add_argument("--tune-patience", type=int, default=8)

    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=4)
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


def _run_dataset(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int, gpus: list[str]) -> int:
    _validate_session_fixed_files(dataset)
    rows = _build_rows(dataset, args, dataset_order_idx=dataset_order_idx)
    if args.smoke_test:
        rows = _smoke_trim_rows(rows, args.smoke_max_runs)

    manifest_path = _write_manifest(dataset, args, rows)
    summary_path = _summary_path(dataset)
    _ensure_summary_csv(summary_path)
    global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model = _load_summary_bests(
        summary_path
    )

    print(
        f"[{PHASE_ID}] dataset={dataset} ({dataset_order_idx+1}) rows={len(rows)} "
        f"axis={AXIS} manifest={manifest_path}"
    )

    for row in rows:
        row["log_path"] = str(_build_log_path(row))

    runnable: list[Dict[str, Any]] = []
    skipped = 0
    result_index_for_resume = _scan_result_index(dataset) if args.verify_logging else {}
    for row in rows:
        lp = Path(str(row["log_path"]))
        completed = _is_completed_any_log(row, use_resume=bool(args.resume_from_logs))
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
        ok, detail = _verify_special_from_result(str(rec.get("path", "")))
        if ok:
            skipped += 1
            continue
        print(f"[resume-check] run={row['run_phase']} special_check_failed -> rerun ({detail})")
        runnable.append(row)

    if skipped > 0:
        print(f"[{PHASE_ID}] resume_from_logs=on: skipped {skipped} completed runs for dataset={dataset}.")
    if len(runnable) < len(gpus):
        print(
            f"[queue] dataset={dataset} runnable={len(runnable)} < gpus={len(gpus)} "
            f"(resume/filters may have reduced immediate parallelism)"
        )

    if not runnable:
        print(f"[{PHASE_ID}] all runs already completed for dataset={dataset}.")
        _update_baseline_phase_summary(dataset, PHASE_ID)
        return 0

    # Runtime uses a shared queue so free GPUs can immediately steal the next job.
    # This avoids long-tail idle time when model cost estimates are inaccurate.
    shared_queue: deque = deque(runnable)
    print(
        f"[queue] dataset={dataset} mode=shared_gpu_queue "
        f"tasks_total={len(shared_queue)} gpus={','.join(gpus)}"
    )
    if len(gpus) > 1:
        preview = ", ".join(str(r["run_phase"]) for r in list(shared_queue)[: min(6, len(shared_queue))])
        print(f"[queue] preview(first)={preview}")

    if args.dry_run:
        # For dry-run, keep deterministic projection of assignments for readability.
        gpu_bins = _plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = _build_command(row, gpu_id, args)
                print(
                    f"[dry-run] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} hparam={row['hparam_id']} seed={row['seed_id']}"
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
                env.setdefault("HYPEROPT_RESULTS_DIR", str(ARTIFACT_ROOT / "results"))
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                with log_path.open("a", encoding="utf-8") as fh:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=EXP_DIR,
                        env=env,
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                    )
                active[gpu_id] = {"proc": proc, "row": row, "log_path": str(log_path)}
                print(
                    f"[launch] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} hparam={row['hparam_id']}"
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

                rec = _get_result_row_from_log_or_scan(
                    dataset=dataset,
                    run_phase=str(row["run_phase"]),
                    log_path=Path(str(slot.get("log_path", ""))),
                    retries=4,
                    sleep_sec=0.75,
                )
                if rec:
                    run_best = _metric_to_float(rec.get("best_mrr"))
                    test_mrr = _metric_to_float(rec.get("test_mrr"))
                    n_completed = int(rec.get("n_completed", 0) or 0)
                    interrupted = bool(rec.get("interrupted", False))
                    result_path = str(rec.get("path", "") or "")
                    if result_path:
                        special_ok, detail = _verify_special_from_result(result_path)
                        print(f"[logging-check] run={row['run_phase']} {detail} result={result_path}")
                        _mirror_logging_bundle(row, result_path)
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
                    "global_best_valid_mrr20": (
                        "" if global_best_valid is None else f"{float(global_best_valid):.6f}"
                    ),
                    "global_best_test_mrr20": "" if global_best_test is None else f"{float(global_best_test):.6f}",
                    "model_best_valid_mrr20": (
                        "" if current_model_best_valid is None else f"{float(current_model_best_valid):.6f}"
                    ),
                    "model_best_test_mrr20": (
                        "" if current_model_best_test is None else f"{float(current_model_best_test):.6f}"
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
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                _append_summary_row(summary_path, summary_row)
                _update_baseline_phase_summary(dataset, PHASE_ID)

                if int(rc) != 0:
                    raise RuntimeError(f"run failed: dataset={dataset} run_phase={row['run_phase']} rc={rc}")

            for gpu_id in done_gpus:
                active.pop(gpu_id, None)

            pending = bool(shared_queue)
            if not pending and not active:
                break
            time.sleep(1)
    except Exception:
        _terminate_active(active)
        raise

    print(f"[{PHASE_ID}] summary updated: {summary_path}")
    _update_baseline_phase_summary(dataset, PHASE_ID)
    return 0


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = list(dict.fromkeys(_parse_csv_strings(args.gpus)))
    if not gpus:
        raise RuntimeError("No GPUs provided")

    datasets = _parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets provided")

    print(
        f"[config] datasets={','.join(datasets)} gpus={','.join(gpus)} "
        f"seeds={args.seeds} hparams={args.hparams} max_hparams_per_model={args.max_hparams_per_model} "
        f"max_evals={args.max_evals} tune_epochs={args.tune_epochs} tune_patience={args.tune_patience} "
        f"smoke_test={args.smoke_test}"
    )

    _check_runtime_models()

    for d_idx, dataset in enumerate(datasets):
        rc = _run_dataset(dataset, args, dataset_order_idx=d_idx, gpus=gpus)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
