#!/usr/bin/env python3
"""Launch baseline wide-search runs for anchor datasets and core models."""

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
AXIS = "WideSearch_anchor2_core5"
PHASE_ID = "P15"
PHASE_NAME = "WIDESEARCH_ANCHOR2_CORE5"
AXIS_DESC = "widesearch_anchor2_core5"

DEFAULT_DATASETS = [
    "lastfm0.03",
    "amazon_beauty",
]

MODEL_SPECS = [
    {"model_option": "sasrec", "model_label": "SASRec"},
    {"model_option": "gru4rec", "model_label": "GRU4Rec"},
    {"model_option": "duorec", "model_label": "DuoRec"},
    {"model_option": "difsr", "model_label": "DIFSR"},
    {"model_option": "fame", "model_label": "FAME"},
]
MODEL_LABEL_BY_OPTION = {str(m["model_option"]): str(m["model_label"]) for m in MODEL_SPECS}

ALL_PROFILE_IDS = [f"C{c}D{d}" for c in range(1, 5) for d in range(1, 5)]

DEFAULT_LR_SPACE_SPECS = [
    {"id": "LR1_2e4_6e4", "band_id": "LR1", "lr_lo": 2.0e-4, "lr_hi": 6.0e-4},
    {"id": "LR2_6e4_2e3", "band_id": "LR2", "lr_lo": 6.0e-4, "lr_hi": 2.0e-3},
    {"id": "LR3_2e3_6e3", "band_id": "LR3", "lr_lo": 2.0e-3, "lr_hi": 6.0e-3},
    {"id": "LR4_3e3_1e2", "band_id": "LR4", "lr_lo": 3.0e-3, "lr_hi": 1.0e-2},
]

MODEL_HPARAM_FAST4: Dict[str, list[str]] = {
    "sasrec": ["C1D3", "C3D3", "C2D2", "C2D3"],
    "gru4rec": ["C3D3", "C2D3", "C2D2", "C4D3"],
    "duorec": ["C1D4", "C2D1", "C1D3", "C2D3"],
    "difsr": ["C1D3", "C2D3", "C2D2", "C3D3"],
    "fame": ["C1D4", "C2D3", "C2D2", "C3D3"],
}

CONCEPT_CFG: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "Compact-Short",
        "hidden_mult": 0.85,
        "emb_mult": 0.95,
        "layer_delta": -1,
        "layer_min": 1,
        "layer_cap": 3,
        "max_len_bias": -6,
        "dropout_delta": 0.02,
        "wd_mult": 1.4,
        "inner_ratio_delta": 0,
        "lr_mult": 1.0,
    },
    "C2": {
        "name": "Balanced-Short",
        "hidden_mult": 1.0,
        "emb_mult": 1.0,
        "layer_delta": 0,
        "layer_min": 1,
        "layer_cap": 3,
        "max_len_bias": 0,
        "dropout_delta": 0.0,
        "wd_mult": 1.0,
        "inner_ratio_delta": 0,
        "lr_mult": 1.0,
    },
    "C3": {
        "name": "Capacity-Short",
        "hidden_mult": 1.20,
        "emb_mult": 1.05,
        "layer_delta": 1,
        "layer_min": 1,
        "layer_cap": 3,
        "max_len_bias": -4,
        "dropout_delta": -0.02,
        "wd_mult": 0.85,
        "inner_ratio_delta": 1,
        "lr_mult": 1.0,
    },
    "C4": {
        "name": "Offbeat-Mixed",
        "hidden_mult": 1.05,
        "emb_mult": 0.90,
        "hidden_min": 96,
        "layer_delta": 0,
        "layer_min": 1,
        "layer_cap": 3,
        "max_len_bias": -8,
        "dropout_delta": 0.04,
        "wd_mult": 1.8,
        "inner_ratio_delta": 0,
        "lr_mult": 1.0,
    },
}

DETAIL_CFG: Dict[str, Dict[str, Any]] = {
    "D1": {
        "name": "TinyCtx-Stable",
        "lr_lo": 1.0e-4,
        "lr_hi": 3.0e-4,
        "width_mult": 0.90,
        "emb_mult": 0.90,
        "max_len": 10,
        "dropout_delta": 0.03,
        "wd_mult": 1.4,
        "inner_ratio": 2,
        "layer_delta": -1,
        "attr_mult": 1.0,
    },
    "D2": {
        "name": "ShortStd",
        "lr_lo": 3.0e-4,
        "lr_hi": 9.0e-4,
        "width_mult": 1.00,
        "emb_mult": 1.00,
        "max_len": 20,
        "dropout_delta": 0.0,
        "wd_mult": 1.0,
        "inner_ratio": 2,
        "layer_delta": 0,
        "attr_mult": 1.0,
    },
    "D3": {
        "name": "ShortWide",
        "lr_lo": 9.0e-4,
        "lr_hi": 2.4e-3,
        "width_mult": 1.20,
        "emb_mult": 1.10,
        "max_len": 16,
        "dropout_delta": -0.01,
        "wd_mult": 0.8,
        "inner_ratio": 3,
        "layer_delta": 1,
        "attr_mult": 1.2,
    },
    "D4": {
        "name": "UnusualMix",
        "lr_lo": 2.4e-3,
        "lr_hi": 6.0e-3,
        "width_mult": 1.10,
        "emb_mult": 0.80,
        "max_len": 12,
        "dropout_delta": 0.05,
        "wd_mult": 2.0,
        "inner_ratio": 4,
        "layer_delta": 0,
        "attr_mult": 0.8,
    },
}

MODEL_DETAIL_LR_BANDS: Dict[str, Dict[str, tuple[float, float]]] = {
    "sasrec": {
        "D1": (2.0e-4, 6.0e-4),
        "D2": (6.0e-4, 2.0e-3),
        "D3": (2.0e-3, 6.0e-3),
        "D4": (3.0e-3, 1.0e-2),
    },
    "gru4rec": {
        "D1": (2.5e-4, 8.0e-4),
        "D2": (8.0e-4, 2.5e-3),
        "D3": (2.5e-3, 7.0e-3),
        "D4": (4.0e-3, 1.0e-2),
    },
    "duorec": {
        "D1": (2.0e-4, 5.0e-4),
        "D2": (5.0e-4, 1.5e-3),
        "D3": (1.2e-3, 4.0e-3),
        "D4": (3.0e-3, 1.0e-2),
    },
    "difsr": {
        "D1": (2.0e-4, 6.0e-4),
        "D2": (6.0e-4, 2.0e-3),
        "D3": (2.0e-3, 6.0e-3),
        "D4": (3.0e-3, 1.0e-2),
    },
    "fame": {
        "D1": (2.0e-4, 5.0e-4),
        "D2": (5.0e-4, 1.5e-3),
        "D3": (1.2e-3, 4.0e-3),
        "D4": (3.0e-3, 1.0e-2),
    },
}

DATASET_LR_MULT = {
    "amazon_beauty": 1.0,
    "lastfm0.03": 1.0,
}

BASE_MODEL_CFG: Dict[str, Dict[str, Any]] = {
    "sasrec": {
        "hidden_size": 128,
        "layers": 2,
        "heads": 4,
        "inner_ratio_base": 2.0,
        "max_len": 30,
        "dropout": 0.15,
        "weight_decay": 2e-4,
    },
    "gru4rec": {
        "hidden_size": 160,
        "layers": 2,
        "max_len": 30,
        "dropout": 0.20,
        "weight_decay": 2e-4,
    },
    "duorec": {
        "hidden_size": 128,
        "layers": 2,
        "heads": 2,
        "inner_ratio_base": 2.0,
        "max_len": 24,
        "dropout": 0.12,
        "weight_decay": 1.5e-4,
    },
    "difsr": {
        "hidden_size": 160,
        "layers": 2,
        "heads": 4,
        "inner_ratio_base": 2.0,
        "max_len": 30,
        "dropout": 0.15,
        "weight_decay": 2e-4,
    },
    "fame": {
        "hidden_size": 144,
        "layers": 2,
        "heads": 4,
        "inner_ratio_base": 2.0,
        "max_len": 30,
        "dropout": 0.15,
        "weight_decay": 2e-4,
    },
}

DUOREC_DETAIL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "D1": {"contrast": "un", "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.0, "semantic_sample_max_tries": 2},
    "D2": {"contrast": "su", "tau": 0.45, "lmd": 0.0, "lmd_sem": 0.06, "semantic_sample_max_tries": 2},
    "D3": {"contrast": "us_x", "tau": 0.8, "lmd": 0.1, "lmd_sem": 0.08, "semantic_sample_max_tries": 3},
    "D4": {"contrast": "un", "tau": 0.3, "lmd": 0.06, "lmd_sem": 0.0, "semantic_sample_max_tries": 2},
}

DIFSR_DETAIL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "D1": {"fusion_type": "sum", "use_attribute_predictor": True, "lambda_attr": 0.05},
    "D2": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.10},
    "D3": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
    "D4": {"fusion_type": "concat", "use_attribute_predictor": False, "lambda_attr": 0.0},
}

FAME_DETAIL_NUM_EXPERTS = {
    "D1": 2,
    "D2": 3,
    "D3": 4,
    "D4": 6,
}

DATASET_CONFIG_MAP = {
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
}

DATASET_COST_WEIGHT = {
    "lastfm0.03": 1.8,
    "amazon_beauty": 0.6,
}

MODEL_COST_WEIGHT = {
    "sasrec": 1.0,
    "gru4rec": 0.8,
    "duorec": 1.55,
    "difsr": 1.3,
    "fame": 1.7,
}

CONCEPT_COST_WEIGHT = {
    "C1": 0.9,
    "C2": 1.0,
    "C3": 1.35,
    "C4": 1.2,
}

DETAIL_COST_WEIGHT = {
    "D1": 0.95,
    "D2": 1.0,
    "D3": 1.1,
    "D4": 1.0,
}

MODEL_EPOCH_CAP = {
    "sasrec": 42,
    "gru4rec": 34,
    "duorec": 46,
    "difsr": 44,
    "fame": 46,
}

CONCEPT_EPOCH_ADJ = {
    "C1": -6,
    "C2": 0,
    "C3": 4,
    "C4": -2,
}

DETAIL_EPOCH_ADJ = {
    "D1": -6,
    "D2": 0,
    "D3": 4,
    "D4": -4,
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


def _resolve_lr_space_specs(text: str) -> list[Dict[str, Any]]:
    raw = _parse_csv_strings(text)
    if not raw or (len(raw) == 1 and raw[0].strip().upper() == "AUTO4"):
        return [dict(x) for x in DEFAULT_LR_SPACE_SPECS]

    out: list[Dict[str, Any]] = []
    for idx, token in enumerate(raw, start=1):
        t = str(token).strip()
        if ":" not in t:
            raise RuntimeError(
                "Invalid --lr-spaces token. Expected 'lo:hi' format, e.g. 2e-4:6e-4"
            )
        lo_text, hi_text = t.split(":", 1)
        lo = float(lo_text.strip())
        hi = float(hi_text.strip())
        if not (lo > 0.0 and hi > lo):
            raise RuntimeError(f"Invalid lr space bounds: {token}")
        out.append(
            {
                "id": f"LR{idx}_{_sanitize_token(lo_text, upper=False)}_{_sanitize_token(hi_text, upper=False)}",
                "band_id": f"LR{idx}",
                "lr_lo": float(lo),
                "lr_hi": float(hi),
            }
        )
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


def _profile_dir_name(profile_id: str) -> str:
    pid = str(profile_id or "").strip()
    if pid.upper().startswith("LR"):
        out = []
        for ch in pid:
            out.append(ch if ch.isalnum() else "_")
        token = "".join(out)
        while "__" in token:
            token = token.replace("__", "_")
        return token.strip("_") or "LR"
    return _sanitize_token(pid, upper=True)


def _phase_dataset_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / _dataset_tag(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_dataset_dir(dataset) / "summary.csv"


def _manifest_path(dataset: str, args: argparse.Namespace) -> Path:
    if str(args.manifest_out or "").strip():
        p = Path(str(args.manifest_out))
        return p.with_name(f"{p.name}_{dataset}.json")
    return _phase_dataset_dir(dataset) / "final_matrix.json"


def _dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip()
    if key not in DATASET_CONFIG_MAP:
        raise RuntimeError(f"Unsupported dataset for widesearch runner: {dataset}")
    return DATASET_CONFIG_MAP[key]


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
        "profile_id",
        "concept_id",
        "detail_id",
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
                        global_valid if global_best_valid is None else max(float(global_best_valid), float(global_valid))
                    )
                if global_test is not None:
                    global_best_test = (
                        global_test if global_best_test is None else max(float(global_best_test), float(global_test))
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
    tune_epochs = int(row.get("effective_tune_epochs", getattr(args, "tune_epochs", 0)) or 0)
    tune_patience = int(row.get("effective_tune_patience", getattr(args, "tune_patience", 0)) or 0)
    lines = [
        f"[{PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"model={row.get('model_label','')} profile_id={row.get('profile_id','')} seed={row.get('seed_id','')}"
        ),
        f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={getattr(args, 'max_evals', '')} tune_epochs={tune_epochs} tune_patience={tune_patience}",
        f"seed={row.get('runtime_seed','')}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _build_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    profile_id = _profile_dir_name(str(row["profile_id"]))
    model_dir = _model_dir_name(str(row["model_label"]))
    model_token = _sanitize_token(str(row["model_label"]), upper=True)
    seed_id = int(row["seed_id"])
    filename = f"{PHASE_ID}_WS_{AXIS_DESC}_{model_token}_{profile_id}_S{seed_id}.log"
    return ds_dir / model_dir / profile_id / filename


def _legacy_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    profile_id = _profile_dir_name(str(row["profile_id"]))
    model_token = _sanitize_token(str(row["model_label"]), upper=True)
    seed_id = int(row["seed_id"])
    filename = f"{PHASE_ID}_WS_{AXIS_DESC}_{model_token}_{profile_id}_S{seed_id}.log"
    return ds_dir / profile_id / filename


def _is_completed_any_log(row: Dict[str, Any], *, use_resume: bool) -> bool:
    if not use_resume:
        return False
    log_paths = [Path(str(row.get("log_path", ""))), _legacy_log_path(row)]
    for p in log_paths:
        if _is_completed_log(p):
            return True
    return False


def _logging_profile_dir(row: Dict[str, Any]) -> Path:
    dataset = _dataset_tag(str(row.get("dataset", "")))
    model_dir = _model_dir_name(str(row.get("model_label", "")))
    profile_id = _profile_dir_name(str(row.get("profile_id", "")))
    return ARTIFACT_ROOT / "logging" / "baseline" / AXIS / dataset / model_dir / profile_id


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

    dst_root = _logging_profile_dir(row)
    dst_root.mkdir(parents=True, exist_ok=True)
    link_name = dst_root / bundle_dir.name
    if link_name.exists():
        return
    try:
        link_name.symlink_to(bundle_dir, target_is_directory=True)
    except Exception:
        pointer = dst_root / f"{bundle_dir.name}.path.txt"
        pointer.write_text(str(bundle_dir.resolve()) + "\n", encoding="utf-8")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _parse_profile_id(profile_id: str) -> tuple[str, str]:
    pid = str(profile_id or "").strip().upper()
    if len(pid) != 4 or pid[0] != "C" or pid[2] != "D":
        raise RuntimeError(f"Invalid profile_id: {profile_id}")
    concept = pid[:2]
    detail = pid[2:]
    if concept not in CONCEPT_CFG or detail not in DETAIL_CFG:
        raise RuntimeError(f"Unknown profile_id: {profile_id}")
    return concept, detail


def _align_hidden_to_heads(hidden: int, heads: int) -> int:
    if heads <= 0:
        return max(1, int(hidden))
    h = max(int(hidden), int(heads))
    aligned = (h // heads) * heads
    if aligned < heads:
        aligned = heads
    return aligned


def _quantize_dim(value: int, *, multiple: int = 8, minimum: int = 32) -> int:
    if multiple <= 1:
        return max(int(minimum), int(value))
    v = max(int(minimum), int(value))
    return (v // multiple) * multiple


def _effective_profile(row: Dict[str, Any]) -> Dict[str, Any]:
    model_option = str(row["model_option"]).lower()
    dataset = str(row["dataset"])
    search_mode = str(row.get("search_mode", "hparam")).strip().lower()
    profile_id = str(row["profile_id"]).upper()

    if search_mode == "lr":
        base_profile_id = str(row.get("base_profile_id", "C2D2")).upper()
        concept_id, detail_id = _parse_profile_id(base_profile_id)
    else:
        concept_id, detail_id = _parse_profile_id(profile_id)

    concept = CONCEPT_CFG[concept_id]
    detail = DETAIL_CFG[detail_id]
    base = dict(BASE_MODEL_CFG[model_option])

    hidden_mult = float(concept["hidden_mult"]) * float(detail.get("width_mult", 1.0))
    hidden = int(round(float(base["hidden_size"]) * hidden_mult))
    if "hidden_min" in concept:
        hidden = max(hidden, int(concept["hidden_min"]))

    layers = int(base["layers"]) + int(concept["layer_delta"]) + int(detail.get("layer_delta", 0))
    layers = max(int(concept["layer_min"]), min(int(concept["layer_cap"]), layers))

    max_len = int(detail["max_len"]) + int(concept.get("max_len_bias", 0))
    max_len = max(8, min(30, max_len))
    if concept_id == "C4" and detail_id in {"D2", "D3"}:
        max_len = min(28, max_len + 4)
    if concept_id == "C4" and detail_id == "D4":
        max_len = 14

    dropout = float(base["dropout"]) + float(concept["dropout_delta"]) + float(detail["dropout_delta"])
    dropout = _clamp(dropout, 0.05, 0.45)

    weight_decay = float(base["weight_decay"]) * float(concept["wd_mult"]) * float(detail["wd_mult"])

    if search_mode == "lr":
        lr_pair = (
            float(row.get("lr_lo_override", 2.0e-4)),
            float(row.get("lr_hi_override", 1.0e-2)),
        )
    else:
        lr_cfg = MODEL_DETAIL_LR_BANDS.get(model_option, {})
        lr_pair = lr_cfg.get(detail_id)
        if lr_pair is None:
            lr_pair = (float(detail["lr_lo"]), float(detail["lr_hi"]))
    lr_lo_raw, lr_hi_raw = float(lr_pair[0]), float(lr_pair[1])
    lr_mult = float(DATASET_LR_MULT.get(dataset, 1.0))
    lr_mult *= float(concept.get("lr_mult", 1.0))
    lr_lo = _clamp(lr_lo_raw * lr_mult, 2e-4, 1e-2)
    lr_hi = _clamp(lr_hi_raw * lr_mult, 2e-4, 1e-2)
    if lr_hi <= lr_lo:
        lr_hi = min(1e-2, max(lr_lo * 1.5, lr_lo + 1e-6))

    inner_ratio = int(detail["inner_ratio"]) + int(concept.get("inner_ratio_delta", 0))
    inner_ratio = max(2, min(5, inner_ratio))
    if concept_id == "C4" and detail_id == "D4":
        inner_ratio = 5

    # GRU4Rec-specific layer adjustments by detail.
    if model_option == "gru4rec":
        if detail_id == "D1":
            layers = max(1, layers - 1)
        elif detail_id == "D3":
            layers = min(3, layers + 1)
        elif detail_id == "D4":
            layers = 1

    heads = int(base.get("heads", 1))
    if model_option == "sasrec":
        heads = 4
        if detail_id == "D4" and hidden < 128:
            heads = 2
    elif model_option in {"difsr", "fame"}:
        if detail_id in {"D1", "D4"} and hidden < 144:
            heads = 2
        else:
            heads = int(base.get("heads", 4))

    if model_option in {"sasrec", "duorec", "difsr", "fame"}:
        hidden = _align_hidden_to_heads(hidden, heads)
    hidden = _quantize_dim(hidden, multiple=8, minimum=64)

    emb_mult = float(concept.get("emb_mult", 1.0)) * float(detail.get("emb_mult", 1.0))
    embedding_size = _quantize_dim(int(round(float(base["hidden_size"]) * emb_mult)), multiple=8, minimum=64)
    if model_option in {"sasrec", "duorec"}:
        embedding_size = int(hidden)
    if model_option in {"difsr", "fame"} and detail_id == "D4":
        embedding_size = _quantize_dim(int(round(hidden * 0.75)), multiple=8, minimum=64)

    inner_size = int(hidden) * int(inner_ratio)
    if detail_id == "D4":
        inner_size = _quantize_dim(inner_size, multiple=16, minimum=128)

    attr_hidden_size = hidden
    if model_option == "difsr":
        attr_mult = float(detail.get("attr_mult", 1.0))
        attr_hidden_size = _quantize_dim(int(round(hidden * attr_mult)), multiple=max(2, heads), minimum=max(heads, 32))
        if attr_hidden_size % heads != 0:
            attr_hidden_size = (attr_hidden_size // heads) * heads
        attr_hidden_size = max(heads, attr_hidden_size)

    out: Dict[str, Any] = {
        "profile_id": profile_id,
        "concept_id": concept_id,
        "detail_id": detail_id,
        "hidden_size": int(hidden),
        "embedding_size": int(embedding_size),
        "layers": int(layers),
        "heads": int(heads),
        "inner_size": int(inner_size),
        "max_len": int(max_len),
        "dropout": float(dropout),
        "weight_decay": float(weight_decay),
        "lr_lo": float(lr_lo),
        "lr_hi": float(lr_hi),
        "attr_hidden_size": int(attr_hidden_size),
    }

    if model_option == "duorec":
        out.update(DUOREC_DETAIL_OVERRIDES[detail_id])

    if model_option == "difsr":
        out.update(DIFSR_DETAIL_OVERRIDES[detail_id])

    if model_option == "fame":
        out["num_experts"] = int(FAME_DETAIL_NUM_EXPERTS[detail_id])

    return out


def _model_runtime_resource_overrides(row: Dict[str, Any], prof: Dict[str, Any]) -> list[str]:
    model_option = str(row["model_option"]).lower()
    hidden = int(prof["hidden_size"])
    max_len = int(prof["max_len"])
    detail_id = str(prof["detail_id"])

    def _scale(base_bs: int) -> int:
        factor = 1.0
        if max_len <= 10:
            factor *= 1.35
        elif max_len <= 14:
            factor *= 1.20
        elif max_len <= 20:
            factor *= 1.08
        if hidden >= 192:
            factor *= 0.70
        elif hidden >= 160:
            factor *= 0.85
        elif hidden <= 112:
            factor *= 1.10
        if detail_id == "D3":
            factor *= 0.92
        elif detail_id == "D4":
            factor *= 0.88
        return max(256, int(base_bs * factor))

    if model_option == "sasrec":
        return [f"++train_batch_size={_scale(4096)}", f"++eval_batch_size={_scale(6144)}"]
    if model_option == "gru4rec":
        return [f"++train_batch_size={_scale(8192)}", f"++eval_batch_size={_scale(10240)}"]
    if model_option == "duorec":
        return [f"++train_batch_size={_scale(1280)}", f"++eval_batch_size={_scale(1920)}"]
    if model_option == "difsr":
        return [f"++train_batch_size={_scale(3072)}", f"++eval_batch_size={_scale(4096)}"]
    if model_option == "fame":
        return [f"++train_batch_size={_scale(1536)}", f"++eval_batch_size={_scale(2304)}"]
    return []


def _base_hparam_overrides(row: Dict[str, Any], prof: Dict[str, Any]) -> list[str]:
    model_option = str(row["model_option"]).lower()
    hidden_size = int(prof["hidden_size"])
    embedding_size = int(prof["embedding_size"])
    layers = int(prof["layers"])
    heads = int(prof.get("heads", 1))
    inner_size = int(prof["inner_size"])
    max_len = int(prof["max_len"])
    out = [
        f"++hidden_size={hidden_size}",
        f"++embedding_size={embedding_size}",
        f"++MAX_ITEM_LIST_LENGTH={max_len}",
    ]

    if model_option in {"sasrec", "duorec"}:
        out.extend(
            [
                f"++n_layers={layers}",
                f"++num_layers={layers}",
                f"++n_heads={heads}",
                f"++num_heads={heads}",
                f"++inner_size={inner_size}",
            ]
        )
    elif model_option in {"difsr", "fame"}:
        out.extend(
            [
                f"++num_layers={layers}",
                f"++n_layers={layers}",
                f"++num_heads={heads}",
                f"++n_heads={heads}",
                f"++inner_size={inner_size}",
            ]
        )
        if model_option == "difsr":
            out.append(f"++attribute_hidden_size={int(prof['attr_hidden_size'])}")
        if model_option == "fame":
            out.append(f"++num_experts={int(prof['num_experts'])}")
    elif model_option == "gru4rec":
        out.append(f"++num_layers={layers}")

    return out


def _model_algorithm_overrides(row: Dict[str, Any], prof: Dict[str, Any]) -> list[str]:
    model_option = str(row["model_option"]).lower()
    if model_option == "duorec":
        return [
            f"++contrast={prof['contrast']}",
            f"++tau={float(prof['tau'])}",
            f"++lmd={float(prof['lmd'])}",
            f"++lmd_sem={float(prof['lmd_sem'])}",
            f"++semantic_sample_max_tries={int(prof['semantic_sample_max_tries'])}",
        ]
    if model_option == "difsr":
        return [
            f"++fusion_type={prof['fusion_type']}",
            f"++use_attribute_predictor={'true' if bool(prof['use_attribute_predictor']) else 'false'}",
            f"++lambda_attr={float(prof['lambda_attr'])}",
        ]
    if model_option == "fame":
        return [f"++num_experts={int(prof['num_experts'])}"]
    return []


def _effective_tune_budget(row: Dict[str, Any], args: argparse.Namespace, prof: Dict[str, Any]) -> tuple[int, int]:
    max_epochs = max(1, int(args.tune_epochs))
    max_patience = max(1, int(args.tune_patience))
    model_option = str(row["model_option"]).lower()
    dataset = str(row["dataset"])
    concept_id = str(prof["concept_id"])
    detail_id = str(prof["detail_id"])

    base_cap = int(MODEL_EPOCH_CAP.get(model_option, max_epochs))
    epochs = base_cap
    epochs += int(CONCEPT_EPOCH_ADJ.get(concept_id, 0))
    epochs += int(DETAIL_EPOCH_ADJ.get(detail_id, 0))
    if dataset == "lastfm0.03":
        epochs += 6
    if int(prof.get("max_len", 20)) <= 12:
        epochs -= 4
    if int(prof.get("layers", 2)) >= 3:
        epochs += 4
    search_mode = str(row.get("search_mode", "hparam")).strip().lower()
    if search_mode == "lr":
        epochs = int(round(epochs * 0.7))
    if search_mode == "hparam" and model_option in {"duorec", "difsr", "fame"}:
        epochs = int(round(epochs * 0.88))

    epochs = max(16, min(max_epochs, epochs))
    patience_target = max(3, min(12, int(round(epochs * 0.2))))
    patience = max(3, min(max_patience, patience_target))
    return int(epochs), int(patience)


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    model_option = str(row["model_option"]).lower()
    prof = _effective_profile(row)
    tune_epochs, tune_patience = _effective_tune_budget(row, args, prof)
    row["effective_tune_epochs"] = int(tune_epochs)
    row["effective_tune_patience"] = int(tune_patience)

    lr_lo = float(prof["lr_lo"])
    lr_hi = float(prof["lr_hi"])
    layers = int(prof["layers"])
    heads = int(prof.get("heads", 1))
    dropout = float(prof["dropout"])
    weight_decay = float(prof["weight_decay"])

    search_lr_only = {"learning_rate": [lr_lo, lr_hi]}
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

    fixed_search: Dict[str, Any] = {
        "weight_decay": weight_decay,
        "hidden_size": int(prof["hidden_size"]),
        "embedding_size": int(prof["embedding_size"]),
        "inner_size": int(prof["inner_size"]),
        "n_layers": layers,
        "num_layers": layers,
        "n_heads": heads,
        "num_heads": heads,
        "dropout_ratio": dropout,
        "dropout_prob": dropout,
        "hidden_dropout_prob": dropout,
        "attn_dropout_prob": dropout,
        "num_experts": int(prof.get("num_experts", 4)),
    }
    for key, value in fixed_search.items():
        cmd.append(f"++search.{key}={hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    if model_option == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    cmd.extend(_model_algorithm_overrides(row, prof))
    cmd.extend(_model_runtime_resource_overrides(row, prof))
    cmd.extend(_base_hparam_overrides(row, prof))
    return cmd


def _dataset_estimated_cost(dataset: str, model_option: str, profile_id: str) -> float:
    try:
        concept_id, detail_id = _parse_profile_id(profile_id)
    except RuntimeError:
        concept_id, detail_id = "C2", "D2"
    profile = _effective_profile(
        {
            "dataset": str(dataset),
            "model_option": str(model_option).lower(),
            "profile_id": str(profile_id).upper(),
            "search_mode": "lr" if str(profile_id).upper().startswith("LR") else "hparam",
            "base_profile_id": "C2D2",
        }
    )
    base = BASE_MODEL_CFG[str(model_option).lower()]
    hidden_factor = float(profile["hidden_size"]) / float(base["hidden_size"])
    depth_factor = float(profile["layers"]) / float(base["layers"])
    seq_factor = max(0.75, float(profile["max_len"]) / 20.0)
    complexity = hidden_factor * depth_factor * seq_factor
    return (
        float(DATASET_COST_WEIGHT.get(str(dataset), 1.0))
        * float(MODEL_COST_WEIGHT.get(str(model_option), 1.0))
        * float(CONCEPT_COST_WEIGHT.get(concept_id, 1.0))
        * float(DETAIL_COST_WEIGHT.get(detail_id, 1.0))
        * float(complexity)
    )


def _build_rows(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    requested_models = [m.lower() for m in _parse_csv_strings(args.models)]
    if not requested_models:
        raise RuntimeError("No models provided")
    for m in requested_models:
        if m not in MODEL_LABEL_BY_OPTION:
            raise RuntimeError(f"Unknown model option: {m}")

    search_mode = str(args.search_mode).strip().lower()
    if search_mode not in {"hparam", "lr"}:
        raise RuntimeError(f"Unsupported --search-mode: {args.search_mode}")

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    ds_tag = _sanitize_token(dataset, upper=True)
    model_specs = [m for m in MODEL_SPECS if str(m["model_option"]).lower() in requested_models]

    selected_profiles: list[str] = []
    selected_profiles_explicit: list[str] = []
    use_auto16 = False
    use_auto4 = False
    lr_specs: list[Dict[str, Any]] = []
    if search_mode == "hparam":
        requested_profiles_raw = [p.upper() for p in _parse_csv_strings(args.profiles)]
        if not requested_profiles_raw:
            raise RuntimeError("No profiles provided")
        use_auto16 = any(p == "AUTO16" for p in requested_profiles_raw)
        use_auto4 = any(p == "AUTO4" for p in requested_profiles_raw)
        if use_auto16 and use_auto4:
            raise RuntimeError("Choose only one of AUTO16 or AUTO4 for --profiles.")
        if use_auto16:
            selected_profiles = list(ALL_PROFILE_IDS)
        elif use_auto4:
            selected_profiles = []
        else:
            seen = set()
            for pid in requested_profiles_raw:
                if pid not in ALL_PROFILE_IDS:
                    raise RuntimeError(f"Unknown profile id: {pid}")
                if pid in seen:
                    continue
                selected_profiles_explicit.append(pid)
                seen.add(pid)
            selected_profiles = list(selected_profiles_explicit)
    else:
        lr_specs = _resolve_lr_space_specs(args.lr_spaces)
        selected_profiles = [str(x["id"]) for x in lr_specs]
        if not selected_profiles:
            raise RuntimeError("No lr spaces resolved for --search-mode lr")
    lr_spec_by_id = {str(x["id"]): x for x in lr_specs}

    for model_order, spec in enumerate(model_specs, start=1):
        model_label = str(spec["model_label"])
        model_option = str(spec["model_option"])
        selected_profiles_for_model = selected_profiles
        if search_mode == "hparam" and use_auto4:
            selected_profiles_for_model = list(MODEL_HPARAM_FAST4.get(model_option, ["C2D2", "C1D3", "C2D3", "C3D3"]))
        if search_mode == "hparam" and selected_profiles_explicit:
            selected_profiles_for_model = list(selected_profiles_explicit)
        for profile_id in selected_profiles_for_model:
            base_profile_id = "C2D2"
            lr_lo_override = None
            lr_hi_override = None
            if search_mode == "hparam":
                concept_id, detail_id = _parse_profile_id(profile_id)
            else:
                lr_spec = lr_spec_by_id[str(profile_id)]
                concept_id = "LR"
                detail_id = str(lr_spec["band_id"])
                lr_lo_override = float(lr_spec["lr_lo"])
                lr_hi_override = float(lr_spec["lr_hi"])
            for seed_id in seeds:
                run_cursor += 1
                run_id = f"WS_{ds_tag}_{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}"
                run_phase = (
                    f"{PHASE_ID}_WS_D{dataset_order_idx:02d}_M{model_order:02d}_"
                    f"{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}"
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "phase_id": PHASE_ID,
                        "axis_id": "WS",
                        "axis_desc": AXIS_DESC,
                        "setting_id": f"WIDESEARCH_{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}",
                        "setting_key": "BASELINE_WIDESEARCH_ANCHOR_CORE",
                        "setting_desc": (
                            f"BASELINE_WIDESEARCH_ANCHOR_CORE_{_sanitize_token(model_label, upper=True)}"
                            f"_{profile_id}_S{int(seed_id)}"
                        ),
                        "profile_id": profile_id,
                        "concept_id": concept_id,
                        "detail_id": detail_id,
                        "hparam_id": profile_id,
                        "seed_id": int(seed_id),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                        "stage": "final",
                        "search_mode": search_mode,
                        "base_profile_id": base_profile_id,
                        "lr_lo_override": lr_lo_override,
                        "lr_hi_override": lr_hi_override,
                        "model_option": model_option,
                        "model_label": model_label,
                        "estimated_cost": _dataset_estimated_cost(dataset, model_option, profile_id),
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
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "dataset": dataset,
        "search_mode": str(args.search_mode),
        "profiles_arg": str(args.profiles),
        "lr_spaces_arg": str(args.lr_spaces),
        "execution_type": "final",
        "model_count": len({str(r.get("model_option", "")).lower() for r in rows}),
        "profile_count": len({str(r.get("profile_id", "")).upper() for r in rows}),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "run_count_formula": (
            f"{len({str(r.get('model_option','')).lower() for r in rows})} x "
            f"{len({str(r.get('profile_id','')).upper() for r in rows})} x {len(_parse_csv_ints(args.seeds))}"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "model": r["model_label"],
                "profile_id": r["profile_id"],
                "concept_id": r["concept_id"],
                "detail_id": r["detail_id"],
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


def _check_runtime_models(models: list[str]) -> None:
    py = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    model_names = [MODEL_LABEL_BY_OPTION[m] for m in models]
    script = (
        "import recbole_patch  # noqa: F401\n"
        "from recbole.utils import utils as rbu\n"
        f"models={model_names!r}\n"
        "for name in models:\n"
        "    _ = rbu.get_model(name)\n"
        "print('[ENV_CHECK] baseline widesearch model registration OK')\n"
    )
    proc = subprocess.run([py, "-c", script], cwd=EXP_DIR, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Model registration check failed:\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P15 WideSearch launcher (anchor-2, core-5)")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default="sasrec,gru4rec,duorec,difsr,fame")
    parser.add_argument("--search-mode", choices=["hparam", "lr"], default="hparam")
    parser.add_argument("--profiles", default="AUTO4")
    parser.add_argument("--lr-spaces", default="AUTO4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=150000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=12)

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

    shared_queue: deque = deque(runnable)
    print(
        f"[queue] dataset={dataset} mode=shared_gpu_queue "
        f"tasks_total={len(shared_queue)} gpus={','.join(gpus)}"
    )
    if len(gpus) > 1:
        preview = ", ".join(str(r["run_phase"]) for r in list(shared_queue)[: min(6, len(shared_queue))])
        print(f"[queue] preview(first)={preview}")

    if args.dry_run:
        gpu_bins = _plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = _build_command(row, gpu_id, args)
                print(
                    f"[dry-run] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} profile={row['profile_id']} seed={row['seed_id']}"
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
                    f"model={row['model_label']} profile={row['profile_id']}"
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
                    "global_best_valid_mrr20": "" if global_best_valid is None else f"{float(global_best_valid):.6f}",
                    "global_best_test_mrr20": "" if global_best_test is None else f"{float(global_best_test):.6f}",
                    "model_best_valid_mrr20": ""
                    if current_model_best_valid is None
                    else f"{float(current_model_best_valid):.6f}",
                    "model_best_test_mrr20": ""
                    if current_model_best_test is None
                    else f"{float(current_model_best_test):.6f}",
                    "run_best_valid_mrr20": "" if run_best is None else f"{float(run_best):.6f}",
                    "run_best_test_mrr20": "" if test_mrr is None else f"{float(test_mrr):.6f}",
                    "profile_id": row["profile_id"],
                    "concept_id": row["concept_id"],
                    "detail_id": row["detail_id"],
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": dataset,
                    "hparam_id": row["profile_id"],
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

    models = [m.lower() for m in _parse_csv_strings(args.models)]
    if not models:
        raise RuntimeError("No models provided")

    search_mode = str(args.search_mode).strip().lower()
    if search_mode == "hparam":
        profiles = [p.upper() for p in _parse_csv_strings(args.profiles)]
        if not profiles:
            raise RuntimeError("No profiles provided")
    elif search_mode == "lr":
        _ = _resolve_lr_space_specs(args.lr_spaces)
    else:
        raise RuntimeError(f"Unsupported --search-mode: {args.search_mode}")

    print(
        f"[config] datasets={','.join(datasets)} gpus={','.join(gpus)} models={','.join(models)} "
        f"search_mode={args.search_mode} profiles={args.profiles} lr_spaces={args.lr_spaces} "
        f"seeds={args.seeds} max_evals={args.max_evals} "
        f"tune_epochs={args.tune_epochs} tune_patience={args.tune_patience} smoke_test={args.smoke_test}"
    )

    _check_runtime_models(models)

    for d_idx, dataset in enumerate(datasets):
        rc = _run_dataset(dataset, args, dataset_order_idx=d_idx, gpus=gpus)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
