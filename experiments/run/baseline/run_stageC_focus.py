#!/usr/bin/env python3
"""Launch baseline Stage C (focus knobs) runs using Stage B selected configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict, deque
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
AXIS = "StageC_Focus_anchor2_core5"
PHASE_ID = "P18"
PHASE_NAME = "STAGEC_FOCUS_ANCHOR2_CORE5"
AXIS_DESC = "stagec_focus_anchor2_core5"
STAGE_ID = "C"

DEFAULT_DATASETS = ["lastfm0.03", "amazon_beauty"]
MODEL_SPECS = [
    {"model_option": "sasrec", "model_label": "SASRec"},
    {"model_option": "gru4rec", "model_label": "GRU4Rec"},
    {"model_option": "duorec", "model_label": "DuoRec"},
    {"model_option": "difsr", "model_label": "DIFSR"},
    {"model_option": "fame", "model_label": "FAME"},
]
MODEL_LABEL_BY_OPTION = {str(m["model_option"]): str(m["model_label"]) for m in MODEL_SPECS}

DATASET_CONFIG_MAP = {
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
}
DATASET_SHORT_TAG = {
    "lastfm0.03": "LFM",
    "amazon_beauty": "AB",
}

LR_CLAMP_MIN = 8.0e-5
LR_CLAMP_MAX = 1.0e-2
MAX_LEN_DEFAULT = 10

STAGEA_AXIS = "StageA_LR_anchor2_core5"
STAGEB_AXIS = "StageB_Structure_anchor2_core5"

BASE_MODEL_CFG: Dict[str, Dict[str, Any]] = {
    "sasrec": {
        "hidden_size": 128,
        "embedding_size": 128,
        "layers": 2,
        "heads": 4,
        "inner_ratio": 2,
        "dropout": 0.20,
        "weight_decay": 3e-4,
    },
    "gru4rec": {
        "hidden_size": 160,
        "embedding_size": 160,
        "layers": 1,
        "dropout": 0.20,
        "weight_decay": 2e-4,
    },
    "duorec": {
        "hidden_size": 96,
        "embedding_size": 96,
        "layers": 1,
        "heads": 2,
        "inner_ratio": 2,
        "dropout": 0.20,
        "weight_decay": 3e-4,
        "contrast": "un",
        "tau": 0.30,
        "lmd": 0.04,
        "lmd_sem": 0.0,
        "semantic_sample_max_tries": 2,
    },
    "difsr": {
        "hidden_size": 128,
        "embedding_size": 128,
        "layers": 1,
        "heads": 4,
        "inner_ratio": 2,
        "dropout": 0.20,
        "weight_decay": 2e-4,
        "attribute_hidden_size": 128,
        "fusion_type": "gate",
        "use_attribute_predictor": True,
        "lambda_attr": 0.10,
    },
    "fame": {
        "hidden_size": 128,
        "embedding_size": 128,
        "layers": 1,
        "heads": 4,
        "inner_ratio": 2,
        "dropout": 0.20,
        "weight_decay": 3e-4,
        "num_experts": 2,
    },
}

# 6 focus profiles: keep structure from Stage B, then tune max_len + model key knobs.
C_PROFILES: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "short_reg",
        "max_len": 10,
        "dropout_delta": 0.03,
        "wd_mult": 1.35,
        "lr_mult": 0.92,
        "lr_span_mult": 0.85,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": -1},
        "duorec": {"contrast": "un", "tau": 0.25, "lmd": 0.02, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": True, "lambda_attr": 0.05},
        "fame": {"num_experts": 2},
    },
    "C2": {
        "name": "short_balanced",
        "max_len": 10,
        "dropout_delta": 0.00,
        "wd_mult": 1.00,
        "lr_mult": 1.00,
        "lr_span_mult": 0.90,
        "sasrec": {"inner_ratio": 2, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "su", "tau": 0.45, "lmd": 0.00, "lmd_sem": 0.06},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.10},
        "fame": {"num_experts": 3},
    },
    "C3": {
        "name": "mid_aggressive",
        "max_len": 15,
        "dropout_delta": -0.02,
        "wd_mult": 0.85,
        "lr_mult": 1.08,
        "lr_span_mult": 0.95,
        "sasrec": {"inner_ratio": 3, "heads_mode": "base"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "us_x", "tau": 0.80, "lmd": 0.10, "lmd_sem": 0.08},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.15},
        "fame": {"num_experts": 4},
    },
    "C4": {
        "name": "long_reg",
        "max_len": 20,
        "dropout_delta": 0.05,
        "wd_mult": 1.70,
        "lr_mult": 0.88,
        "lr_span_mult": 0.90,
        "sasrec": {"inner_ratio": 2, "heads_mode": "half_if_small"},
        "gru4rec": {"layer_delta": 0},
        "duorec": {"contrast": "un", "tau": 0.30, "lmd": 0.06, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "concat", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 6},
    },
    "C5": {
        "name": "mid_sparse_outlier",
        "max_len": 15,
        "dropout_delta": 0.08,
        "wd_mult": 2.20,
        "lr_mult": 0.80,
        "lr_span_mult": 1.00,
        "sasrec": {"inner_ratio": 1, "heads_mode": "half"},
        "gru4rec": {"layer_delta": -1},
        "duorec": {"contrast": "un", "tau": 1.00, "lmd": 0.12, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 2},
    },
    "C6": {
        "name": "long_dense_outlier",
        "max_len": 20,
        "dropout_delta": -0.03,
        "wd_mult": 0.70,
        "lr_mult": 1.12,
        "lr_span_mult": 1.05,
        "sasrec": {"inner_ratio": 4, "heads_mode": "double_if_fit"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "su", "tau": 0.55, "lmd": 0.00, "lmd_sem": 0.12},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.20},
        "fame": {"num_experts": 6},
    },
    "C7": {
        "name": "expert_boost_outlier",
        "max_len": 20,
        "dropout_delta": -0.06,
        "wd_mult": 0.55,
        "lr_mult": 1.22,
        "lr_span_mult": 1.30,
        "sasrec": {"inner_ratio": 4, "heads_mode": "double_if_fit"},
        "gru4rec": {"layer_delta": 1},
        "duorec": {"contrast": "us_x", "tau": 0.18, "lmd": 0.08, "lmd_sem": 0.15},
        "difsr": {"fusion_type": "gate", "use_attribute_predictor": True, "lambda_attr": 0.25},
        "fame": {"num_experts": 5},
    },
    "C8": {
        "name": "hard_regularized_sparse_outlier",
        "max_len": 15,
        "dropout_delta": 0.11,
        "wd_mult": 2.5,
        "lr_mult": 0.72,
        "lr_span_mult": 1.20,
        "sasrec": {"inner_ratio": 1, "heads_mode": "half"},
        "gru4rec": {"layer_delta": -1},
        "duorec": {"contrast": "un", "tau": 1.20, "lmd": 0.14, "lmd_sem": 0.00},
        "difsr": {"fusion_type": "sum", "use_attribute_predictor": False, "lambda_attr": 0.00},
        "fame": {"num_experts": 2},
    },
}

PROFILE_ORDER = list(C_PROFILES.keys())

MODEL_COST_WEIGHT = {
    "sasrec": 1.0,
    "gru4rec": 0.8,
    "duorec": 1.5,
    "difsr": 1.2,
    "fame": 1.3,
}
DATASET_COST_WEIGHT = {
    "lastfm0.03": 1.8,
    "amazon_beauty": 0.6,
}
PROFILE_COST_WEIGHT = {
    "C1": 0.95,
    "C2": 1.0,
    "C3": 1.12,
    "C4": 1.15,
    "C5": 1.05,
    "C6": 1.18,
    "C7": 1.28,
    "C8": 1.15,
}

DEFAULT_STAGEA_LR_FALLBACK = {
    "sasrec": (7.2e-4, 2.2e-3),
    "gru4rec": (2.4e-3, 6.8e-3),
    "duorec": (5.5e-4, 1.8e-3),
    "difsr": (1.1e-3, 3.7e-3),
    "fame": (9.0e-4, 2.8e-3),
}

TRIAL_MRR_RE = re.compile(r"^\[TRIAL_METRICS\].*cur_best_mrr20=([0-9eE+\-.]+)")


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


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _dataset_short_tag(dataset: str) -> str:
    key = str(dataset or "").strip()
    if key in DATASET_SHORT_TAG:
        return str(DATASET_SHORT_TAG[key])
    return _sanitize_token(key, upper=True)[:10]


def _model_dir_name(model_label: str) -> str:
    raw = str(model_label or "").strip()
    if not raw:
        return "unknown_model"
    return raw.replace("/", "_")


def _phase_dataset_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / _dataset_tag(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_dataset_dir(dataset) / "summary.csv"


def _manifest_path(dataset: str, args: argparse.Namespace) -> Path:
    if str(args.manifest_out or "").strip():
        p = Path(str(args.manifest_out))
        return p.with_name(f"{p.name}_{dataset}.json")
    return _phase_dataset_dir(dataset) / "final_matrix.json"


def _candidates_path(dataset: str) -> Path:
    return _phase_dataset_dir(dataset) / "stageC_candidates.json"


def _dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip()
    if key not in DATASET_CONFIG_MAP:
        raise RuntimeError(f"Unsupported dataset for Stage C runner: {dataset}")
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


def _extract_best_lr(payload: Dict[str, Any]) -> Optional[float]:
    bp = payload.get("best_params")
    if isinstance(bp, dict):
        lr = _metric_to_float(bp.get("learning_rate"))
        if lr is not None:
            return lr
    return _metric_to_float(payload.get("best_learning_rate"))


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
        "best_lr": _extract_best_lr(payload),
        "n_completed": int(payload.get("n_completed", 0) or 0),
        "interrupted": bool(payload.get("interrupted", False)),
        "path": str(p.resolve()),
    }


def _get_result_row_from_log(log_path: Path, run_phase: str, retries: int = 5, sleep_sec: float = 0.7) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        path = _extract_result_json_path_from_log(log_path)
        if path:
            rec = _result_row_from_json(path, run_phase)
            if rec:
                return rec
        time.sleep(max(float(sleep_sec), 0.0))
    return None


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
        "global_best_valid_mrr20",
        "global_best_test_mrr20",
        "model",
        "model_best_valid_mrr20",
        "model_best_test_mrr20",
        "run_best_valid_mrr20",
        "run_best_test_mrr20",
        "profile_id",
        "parent_profile_id",
        "concept_id",
        "detail_id",
        "parent_run_phase",
        "lr_band_id",
        "lr_lo",
        "lr_hi",
        "lr_center",
        "stage_id",
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
    fields = _summary_fieldnames()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                existing_fields = [f for f in (reader.fieldnames or []) if f]
                existing_rows = list(reader)
        except Exception:
            existing_fields = []
            existing_rows = []
        if existing_fields == fields:
            return
        backup = path.with_name(f"{path.stem}.legacy_{int(time.time())}{path.suffix}")
        try:
            path.rename(backup)
        except Exception:
            pass
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()


def _load_summary_bests(path: Path) -> tuple[Optional[float], Optional[float], Dict[str, float], Dict[str, float]]:
    if not path.exists():
        return None, None, {}, {}
    g_valid = None
    g_test = None
    m_valid: Dict[str, float] = {}
    m_test: Dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                model = str(row.get("model", "") or "").strip()
                run_valid = _metric_to_float(row.get("run_best_valid_mrr20"))
                run_test = _metric_to_float(row.get("run_best_test_mrr20"))
                if run_valid is not None:
                    g_valid = run_valid if g_valid is None else max(float(g_valid), float(run_valid))
                    if model:
                        prev = m_valid.get(model)
                        m_valid[model] = run_valid if prev is None else max(float(prev), float(run_valid))
                if run_test is not None:
                    g_test = run_test if g_test is None else max(float(g_test), float(run_test))
                    if model:
                        prev = m_test.get(model)
                        m_test[model] = run_test if prev is None else max(float(prev), float(run_test))
    except Exception:
        return None, None, {}, {}
    return g_valid, g_test, m_valid, m_test


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writerow({k: row.get(k, "") for k in _summary_fieldnames()})


def _build_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    model_dir = _model_dir_name(str(row["model_label"]))
    parent = _sanitize_token(str(row.get("parent_profile_id", "BP?")), upper=False)
    profile = _sanitize_token(str(row["profile_id"]), upper=False)
    model = str(row["model_label"])
    seed_id = int(row["seed_id"])
    seed_suffix = "" if seed_id == 1 else f"_S{seed_id}"
    fname = f"C_{profile}_{parent}_{_dataset_short_tag(str(row['dataset']))}_{model}{seed_suffix}.log"
    return ds_dir / model_dir / fname


def _legacy_log_paths(row: Dict[str, Any]) -> list[Path]:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    model_dir = _model_dir_name(str(row["model_label"]))
    profile = _sanitize_token(str(row["profile_id"]), upper=False)
    model = str(row["model_label"])
    seed_id = int(row["seed_id"])
    seed_suffix = "" if seed_id == 1 else f"_S{seed_id}"
    old = f"{PHASE_ID}_SC_{AXIS_DESC}_{model}_{profile}{seed_suffix}.log"
    return [ds_dir / model_dir / profile / old]


def _is_completed_any_log(row: Dict[str, Any], *, use_resume: bool) -> bool:
    if not use_resume:
        return False
    log_paths = [Path(str(row.get("log_path", ""))), *_legacy_log_paths(row)]
    for p in log_paths:
        if _is_completed_log(p):
            return True
    return False


def _logging_profile_dir(row: Dict[str, Any]) -> Path:
    dataset = _dataset_tag(str(row.get("dataset", "")))
    model_dir = _model_dir_name(str(row.get("model_label", "")))
    profile = _sanitize_token(str(row.get("profile_id", "")), upper=False)
    return ARTIFACT_ROOT / "logging" / "baseline" / AXIS / dataset / model_dir / profile


def _mirror_logging_bundle(row: Dict[str, Any], result_json_path: str) -> None:
    path = Path(str(result_json_path or "")).expanduser()
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    bundle = Path(str(payload.get("logging_bundle_dir", "") or "").strip()).expanduser()
    if not bundle.exists():
        return
    dst_root = _logging_profile_dir(row)
    dst_root.mkdir(parents=True, exist_ok=True)
    link_name = dst_root / bundle.name
    if link_name.exists():
        return
    try:
        link_name.symlink_to(bundle, target_is_directory=True)
    except Exception:
        pointer = dst_root / f"{bundle.name}.path.txt"
        pointer.write_text(str(bundle.resolve()) + "\n", encoding="utf-8")


def _align_hidden_to_heads(hidden: int, heads: int) -> int:
    if heads <= 0:
        return max(1, int(hidden))
    h = max(int(hidden), int(heads))
    aligned = (h // heads) * heads
    if aligned < heads:
        aligned = heads
    return aligned


def _quantize_dim(value: int, *, multiple: int = 8, minimum: int = 32) -> int:
    v = max(int(minimum), int(value))
    if multiple <= 1:
        return v
    return (v // multiple) * multiple


def _load_stagea_lr_window(dataset: str, model_label: str) -> tuple[float, float, str]:
    # Prefer Stage-A summary/result artifacts with early-stop aware scoring.
    summary = LOG_ROOT / STAGEA_AXIS / _dataset_tag(dataset) / "summary.csv"
    scored: list[Dict[str, Any]] = []
    if summary.exists():
        try:
            with summary.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if str(row.get("model", "")).strip() != str(model_label):
                        continue
                    valid = _metric_to_float(row.get("run_best_valid_mrr20"))
                    lo = _metric_to_float(row.get("lr_lo"))
                    hi = _metric_to_float(row.get("lr_hi"))
                    if valid is None or lo is None or hi is None or hi <= lo:
                        continue
                    result_path = str(row.get("result_path", "") or "").strip()
                    completion_ratio = 1.0
                    early_stop_ratio = 0.0
                    epoch_usage_ratio = 1.0
                    if result_path:
                        p = Path(result_path)
                        if p.exists():
                            try:
                                payload = json.loads(p.read_text(encoding="utf-8"))
                                trials = payload.get("trials") or []
                                if isinstance(trials, list) and trials:
                                    early_cnt = 0
                                    epoch_frac_sum = 0.0
                                    epoch_frac_n = 0
                                    tune_epochs = max(1, int(payload.get("tune_epochs", 1) or 1))
                                    for t in trials:
                                        if not isinstance(t, dict):
                                            continue
                                        if bool(t.get("early_stopped", False)):
                                            early_cnt += 1
                                        er = _metric_to_float(t.get("epochs_run"))
                                        if er is not None:
                                            epoch_frac_sum += max(0.0, min(1.0, float(er) / float(tune_epochs)))
                                            epoch_frac_n += 1
                                    early_stop_ratio = float(early_cnt) / float(max(1, len(trials)))
                                    if epoch_frac_n > 0:
                                        epoch_usage_ratio = float(epoch_frac_sum) / float(epoch_frac_n)
                                max_evals = _metric_to_float(payload.get("max_evals"))
                                nc = _metric_to_float(payload.get("n_completed"))
                                if max_evals is not None and max_evals > 0 and nc is not None:
                                    completion_ratio = max(0.0, min(1.0, float(nc) / float(max_evals)))
                            except Exception:
                                pass

                    score = float(valid) - 0.004 * float(early_stop_ratio) + 0.002 * float(epoch_usage_ratio) - 0.010 * (
                        1.0 - float(completion_ratio)
                    )
                    scored.append(
                        {
                            "lr_lo": float(lo),
                            "lr_hi": float(hi),
                            "lr_band_id": str(row.get("lr_band_id", "LR?")),
                            "valid": float(valid),
                            "score": float(score),
                            "early_stop_ratio": float(early_stop_ratio),
                            "epoch_usage_ratio": float(epoch_usage_ratio),
                            "completion_ratio": float(completion_ratio),
                        }
                    )
        except Exception:
            scored = []

    if scored:
        scored.sort(
            key=lambda x: (
                float(x["score"]),
                float(x["valid"]),
                -float(x["early_stop_ratio"]),
                float(x["epoch_usage_ratio"]),
            ),
            reverse=True,
        )
        top = scored[:2]
        lo = min(float(x["lr_lo"]) for x in top)
        hi = max(float(x["lr_hi"]) for x in top)
        bid = ",".join(str(x["lr_band_id"]) for x in top)
        if lo > 0 and hi > lo:
            return float(lo), float(hi), bid

    path = LOG_ROOT / STAGEA_AXIS / _dataset_tag(dataset) / "stageA_candidates.json"
    model = str(model_label)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mnode = payload.get("models", {}).get(model)
            if isinstance(mnode, dict):
                bands = mnode.get("bands") or []
                selected = mnode.get("selected_bands") or []
                if isinstance(selected, list) and selected:
                    lo = min(float(x.get("lr_lo", 0.0) or 0.0) for x in selected)
                    hi = max(float(x.get("lr_hi", 0.0) or 0.0) for x in selected)
                    bid = ",".join(str(x.get("lr_band_id", "LR?")) for x in selected[:2])
                    if lo > 0 and hi > lo:
                        return float(lo), float(hi), bid
                if isinstance(bands, list) and bands:
                    best = bands[0]
                    lo = float(best.get("lr_lo", 0.0) or 0.0)
                    hi = float(best.get("lr_hi", 0.0) or 0.0)
                    bid = str(best.get("lr_band_id", "LR?"))
                    if lo > 0 and hi > lo:
                        return lo, hi, bid
        except Exception:
            pass
    opt = next((k for k, v in MODEL_LABEL_BY_OPTION.items() if str(v) == model_label), "sasrec")
    lo, hi = DEFAULT_STAGEA_LR_FALLBACK.get(opt, (5e-4, 2e-3))
    return float(lo), float(hi), "A_FALLBACK"


def _default_parent_config(model_option: str) -> Dict[str, Any]:
    base = dict(BASE_MODEL_CFG[model_option])
    cfg: Dict[str, Any] = {
        "hidden_size": int(base["hidden_size"]),
        "embedding_size": int(base["embedding_size"]),
        "layers": int(base.get("layers", 1)),
        "num_layers": int(base.get("layers", 1)),
        "max_len": MAX_LEN_DEFAULT,
        "dropout": float(base["dropout"]),
        "weight_decay": float(base["weight_decay"]),
    }
    if model_option in {"sasrec", "duorec", "difsr", "fame"}:
        heads = int(base.get("heads", 1))
        hidden = _align_hidden_to_heads(int(cfg["hidden_size"]), heads)
        hidden = _quantize_dim(hidden, multiple=8, minimum=max(64, heads))
        cfg["hidden_size"] = int(hidden)
        cfg["embedding_size"] = int(hidden)
        cfg["heads"] = int(heads)
        ratio = int(base.get("inner_ratio", 2))
        cfg["inner_size"] = int(hidden * ratio)
    if model_option == "duorec":
        cfg.update(
            {
                "contrast": str(base.get("contrast", "un")),
                "tau": float(base.get("tau", 0.30)),
                "lmd": float(base.get("lmd", 0.04)),
                "lmd_sem": float(base.get("lmd_sem", 0.0)),
                "semantic_sample_max_tries": int(base.get("semantic_sample_max_tries", 2)),
            }
        )
    if model_option == "difsr":
        cfg.update(
            {
                "attribute_hidden_size": int(cfg["hidden_size"]),
                "fusion_type": str(base.get("fusion_type", "gate")),
                "use_attribute_predictor": bool(base.get("use_attribute_predictor", True)),
                "lambda_attr": float(base.get("lambda_attr", 0.10)),
            }
        )
    if model_option == "fame":
        cfg["num_experts"] = int(base.get("num_experts", 2))
    return cfg


def _normalize_parent_config(model_option: str, cfg_in: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_parent_config(model_option)
    cfg = dict(base)
    cfg.update(dict(cfg_in or {}))

    hidden = _quantize_dim(int(cfg.get("hidden_size", base["hidden_size"])), multiple=8, minimum=64)
    emb = _quantize_dim(int(cfg.get("embedding_size", hidden)), multiple=8, minimum=64)
    layers = int(cfg.get("layers", cfg.get("num_layers", base.get("layers", 1))))
    layers = max(1, min(3, layers))
    dropout = _clamp(float(cfg.get("dropout", base["dropout"])), 0.05, 0.45)
    wd = max(1e-8, float(cfg.get("weight_decay", base["weight_decay"])))
    max_len = int(cfg.get("max_len", MAX_LEN_DEFAULT))
    max_len = max(10, min(20, max_len))

    out: Dict[str, Any] = {
        "hidden_size": int(hidden),
        "embedding_size": int(emb),
        "layers": int(layers),
        "num_layers": int(layers),
        "max_len": int(max_len),
        "dropout": float(dropout),
        "weight_decay": float(wd),
    }

    if model_option in {"sasrec", "duorec", "difsr", "fame"}:
        heads = int(cfg.get("heads", cfg.get("n_heads", cfg.get("num_heads", base.get("heads", 1)))))
        heads = max(1, min(8, heads))
        hidden2 = _align_hidden_to_heads(int(hidden), heads)
        hidden2 = _quantize_dim(hidden2, multiple=8, minimum=max(64, heads))
        out["hidden_size"] = int(hidden2)
        out["embedding_size"] = int(hidden2)
        out["heads"] = int(heads)
        inner = int(cfg.get("inner_size", max(int(hidden2) * 2, 64)))
        out["inner_size"] = _quantize_dim(inner, multiple=8, minimum=int(hidden2))

    if model_option == "duorec":
        out.update(
            {
                "contrast": str(cfg.get("contrast", base.get("contrast", "un"))),
                "tau": float(cfg.get("tau", base.get("tau", 0.30))),
                "lmd": float(cfg.get("lmd", base.get("lmd", 0.04))),
                "lmd_sem": float(cfg.get("lmd_sem", base.get("lmd_sem", 0.0))),
                "semantic_sample_max_tries": int(cfg.get("semantic_sample_max_tries", base.get("semantic_sample_max_tries", 2))),
            }
        )
    elif model_option == "difsr":
        out.update(
            {
                "attribute_hidden_size": int(out["hidden_size"]),
                "fusion_type": str(cfg.get("fusion_type", base.get("fusion_type", "gate"))),
                "use_attribute_predictor": bool(cfg.get("use_attribute_predictor", base.get("use_attribute_predictor", True))),
                "lambda_attr": float(cfg.get("lambda_attr", base.get("lambda_attr", 0.10))),
            }
        )
    elif model_option == "fame":
        out["num_experts"] = int(cfg.get("num_experts", base.get("num_experts", 2)))
        out["num_experts"] = max(2, min(6, int(out["num_experts"])))

    return out


def _load_stageb_parent_candidates(dataset: str, model_option: str, model_label: str, topk: int) -> list[Dict[str, Any]]:
    path = LOG_ROOT / STAGEB_AXIS / _dataset_tag(dataset) / "stageB_candidates.json"
    selected: list[Dict[str, Any]] = []
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            node = payload.get("models", {}).get(str(model_label), {})
            if isinstance(node, dict):
                picked = node.get("selected_profiles") or []
                if not picked:
                    picked = (node.get("profiles") or [])[: max(1, int(topk))]
                for it in picked:
                    cfg_raw = dict(it.get("config", {}) or {})
                    cfg = _normalize_parent_config(model_option, cfg_raw)
                    lo = _metric_to_float(it.get("lr_lo"))
                    hi = _metric_to_float(it.get("lr_hi"))
                    best_lr = _metric_to_float(it.get("best_lr"))
                    if lo is None or hi is None or hi <= lo:
                        if best_lr is not None:
                            lo = max(LR_CLAMP_MIN, best_lr / 1.8)
                            hi = min(LR_CLAMP_MAX, best_lr * 1.8)
                        else:
                            lo, hi, _ = _load_stagea_lr_window(dataset, model_label)
                    selected.append(
                        {
                            "parent_profile_id": str(it.get("profile_id", "B?")),
                            "config": cfg,
                            "lr_lo": float(lo),
                            "lr_hi": float(hi),
                            "lr_band_id": str(it.get("lr_band_id", "BSEL")),
                            "parent_run_phase": str(it.get("representative_run_phase", "")),
                        }
                    )
        except Exception:
            selected = []

    if selected:
        return selected[: max(1, int(topk))]

    lo, hi, band = _load_stagea_lr_window(dataset, model_label)
    return [
        {
            "parent_profile_id": "B_FALLBACK",
            "config": _default_parent_config(model_option),
            "lr_lo": float(lo),
            "lr_hi": float(hi),
            "lr_band_id": str(band),
            "parent_run_phase": "",
        }
    ]


def _effective_budget(model_option: str, args: argparse.Namespace) -> tuple[int, int, int]:
    model = str(model_option).lower()
    if model == "duorec":
        return int(args.max_evals_duorec), int(args.tune_epochs_duorec), int(args.tune_patience_duorec)
    if model == "fame":
        return int(args.max_evals_fame), int(args.tune_epochs_fame), int(args.tune_patience_fame)
    return int(args.max_evals_default), int(args.tune_epochs_default), int(args.tune_patience_default)


def _build_profile_config(model_option: str, parent_cfg: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    base = _normalize_parent_config(model_option, parent_cfg)
    prof = dict(C_PROFILES[profile_id])

    hidden = int(base["hidden_size"])
    layers = int(base.get("layers", base.get("num_layers", 1)))
    dropout = _clamp(float(base["dropout"]) + float(prof["dropout_delta"]), 0.05, 0.45)
    wd = max(1e-8, float(base["weight_decay"]) * float(prof["wd_mult"]))
    max_len = int(prof["max_len"])
    max_len = max(10, min(20, max_len))

    cfg: Dict[str, Any] = {
        "profile_id": str(profile_id),
        "profile_name": str(prof["name"]),
        "hidden_size": int(hidden),
        "embedding_size": int(hidden),
        "layers": int(layers),
        "num_layers": int(layers),
        "max_len": int(max_len),
        "dropout": float(dropout),
        "weight_decay": float(wd),
        "lr_mult": float(prof["lr_mult"]),
        "lr_span_mult": float(prof["lr_span_mult"]),
    }

    if model_option in {"sasrec", "duorec", "difsr", "fame"}:
        base_heads = int(base.get("heads", 4))
        heads_mode = str(prof.get("sasrec", {}).get("heads_mode", "base"))
        heads = int(base_heads)
        if heads_mode == "half":
            heads = max(1, base_heads // 2)
        elif heads_mode == "half_if_small":
            heads = max(1, base_heads // 2) if int(hidden) < 128 else base_heads
        elif heads_mode == "double_if_fit":
            heads = min(8, max(1, base_heads * 2))
        heads = max(1, min(8, heads))
        hidden = _align_hidden_to_heads(int(hidden), int(heads))
        hidden = _quantize_dim(hidden, multiple=8, minimum=max(64, heads))
        cfg["hidden_size"] = int(hidden)
        cfg["embedding_size"] = int(hidden)
        cfg["heads"] = int(heads)
        inner_ratio = _metric_to_float(prof.get("sasrec", {}).get("inner_ratio"))
        if inner_ratio is None:
            inner_ratio = max(1.0, float(base.get("inner_size", hidden * 2)) / float(hidden))
        cfg["inner_size"] = _quantize_dim(int(round(float(hidden) * float(inner_ratio))), multiple=8, minimum=int(hidden))

    if model_option == "gru4rec":
        delta = int(prof.get("gru4rec", {}).get("layer_delta", 0))
        layers2 = max(1, min(3, int(layers) + delta))
        cfg["layers"] = int(layers2)
        cfg["num_layers"] = int(layers2)

    if model_option == "duorec":
        d = prof.get("duorec", {})
        cfg.update(
            {
                "contrast": str(d.get("contrast", base.get("contrast", "un"))),
                "tau": float(d.get("tau", base.get("tau", 0.30))),
                "lmd": float(d.get("lmd", base.get("lmd", 0.04))),
                "lmd_sem": float(d.get("lmd_sem", base.get("lmd_sem", 0.0))),
                "semantic_sample_max_tries": int(base.get("semantic_sample_max_tries", 2)),
            }
        )
    elif model_option == "difsr":
        d = prof.get("difsr", {})
        cfg.update(
            {
                "attribute_hidden_size": int(cfg["hidden_size"]),
                "fusion_type": str(d.get("fusion_type", base.get("fusion_type", "gate"))),
                "use_attribute_predictor": bool(d.get("use_attribute_predictor", base.get("use_attribute_predictor", True))),
                "lambda_attr": float(d.get("lambda_attr", base.get("lambda_attr", 0.10))),
            }
        )
    elif model_option == "fame":
        d = prof.get("fame", {})
        experts = int(d.get("num_experts", base.get("num_experts", 2)))
        cfg["num_experts"] = max(2, min(6, experts))

    return cfg


def _build_lr_bounds(base_lo: float, base_hi: float, cfg: Dict[str, Any]) -> tuple[float, float]:
    center = math.sqrt(float(base_lo) * float(base_hi)) * float(cfg["lr_mult"])
    span = max(1.05, math.sqrt(float(base_hi) / float(base_lo)) * float(cfg["lr_span_mult"]))
    lo = _clamp(float(center / span), LR_CLAMP_MIN, LR_CLAMP_MAX)
    hi = _clamp(float(center * span), LR_CLAMP_MIN, LR_CLAMP_MAX)
    if hi <= lo:
        hi = min(LR_CLAMP_MAX, max(lo * 1.3, lo + 1e-6))
    return lo, hi


def _build_command(row: Dict[str, Any], gpu_id: str) -> list[str]:
    model = str(row["model_option"]).lower()
    cfg = dict(row["config"])
    dropout = float(cfg["dropout"])
    wd = float(cfg["weight_decay"])

    search = {"learning_rate": [float(row["lr_lo"]), float(row["lr_hi"])]}
    search_type = {"learning_rate": "loguniform"}

    py = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        py,
        "hyperopt_tune.py",
        "--config-name",
        _dataset_config_name(str(row["dataset"])),
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
        f"model={model}",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        "++special_logging=true",
        f"++seed={int(row['runtime_seed'])}",
        f"++search={hydra_literal(search)}",
        f"++search_space_type_overrides={hydra_literal(search_type)}",
        f"++MAX_ITEM_LIST_LENGTH={int(cfg['max_len'])}",
        f"++weight_decay={wd}",
    ]

    fixed_search: Dict[str, Any] = {
        "weight_decay": wd,
        "hidden_size": int(cfg["hidden_size"]),
        "embedding_size": int(cfg["embedding_size"]),
        "n_layers": int(cfg.get("layers", cfg.get("num_layers", 1))),
        "num_layers": int(cfg.get("num_layers", cfg.get("layers", 1))),
        "dropout_ratio": dropout,
        "dropout_prob": dropout,
        "hidden_dropout_prob": dropout,
        "attn_dropout_prob": dropout,
    }
    if "inner_size" in cfg:
        fixed_search["inner_size"] = int(cfg["inner_size"])
    if "heads" in cfg:
        fixed_search["n_heads"] = int(cfg["heads"])
        fixed_search["num_heads"] = int(cfg["heads"])
    if "num_experts" in cfg:
        fixed_search["num_experts"] = int(cfg["num_experts"])

    for key, val in fixed_search.items():
        cmd.append(f"++search.{key}={hydra_literal([val])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    if model == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
        cmd.append(f"++num_layers={int(cfg.get('num_layers', cfg['layers']))}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    cmd.extend(
        [
            f"++hidden_size={int(cfg['hidden_size'])}",
            f"++embedding_size={int(cfg['embedding_size'])}",
            f"++n_layers={int(cfg.get('layers', cfg.get('num_layers', 1)))}",
            f"++num_layers={int(cfg.get('num_layers', cfg.get('layers', 1)))}",
        ]
    )
    if "heads" in cfg:
        cmd.extend([f"++n_heads={int(cfg['heads'])}", f"++num_heads={int(cfg['heads'])}"])
    if "inner_size" in cfg:
        cmd.append(f"++inner_size={int(cfg['inner_size'])}")

    if model == "duorec":
        cmd.extend(
            [
                f"++contrast={cfg['contrast']}",
                f"++tau={float(cfg['tau'])}",
                f"++lmd={float(cfg['lmd'])}",
                f"++lmd_sem={float(cfg['lmd_sem'])}",
                f"++semantic_sample_max_tries={int(cfg['semantic_sample_max_tries'])}",
            ]
        )
        cmd.extend(["++train_batch_size=1024", "++eval_batch_size=1536"])
    elif model == "difsr":
        cmd.extend(
            [
                f"++attribute_hidden_size={int(cfg['attribute_hidden_size'])}",
                f"++fusion_type={cfg['fusion_type']}",
                f"++use_attribute_predictor={'true' if bool(cfg['use_attribute_predictor']) else 'false'}",
                f"++lambda_attr={float(cfg['lambda_attr'])}",
            ]
        )
        cmd.extend(["++train_batch_size=3072", "++eval_batch_size=4096"])
    elif model == "fame":
        cmd.append(f"++num_experts={int(cfg['num_experts'])}")
        cmd.extend(["++train_batch_size=1536", "++eval_batch_size=2304"])
    elif model == "gru4rec":
        cmd.extend(["++train_batch_size=8192", "++eval_batch_size=10240"])
    else:
        cmd.extend(["++train_batch_size=4096", "++eval_batch_size=6144"])

    return cmd


def _dataset_estimated_cost(row: Dict[str, Any]) -> float:
    dataset = str(row["dataset"])
    model = str(row["model_option"]).lower()
    profile = str(row["profile_id"])
    max_evals = max(1, int(row["max_evals"]))
    tune_epochs = max(1, int(row["tune_epochs"]))
    return (
        float(DATASET_COST_WEIGHT.get(dataset, 1.0))
        * float(MODEL_COST_WEIGHT.get(model, 1.0))
        * float(PROFILE_COST_WEIGHT.get(profile, 1.0))
        * (float(max_evals) / 5.0)
        * (float(tune_epochs) / 32.0)
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

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    ds_tag = _sanitize_token(dataset, upper=True)
    model_specs = [m for m in MODEL_SPECS if str(m["model_option"]).lower() in requested_models]

    for m_order, spec in enumerate(model_specs, start=1):
        model_opt = str(spec["model_option"])
        model_label = str(spec["model_label"])
        parents = _load_stageb_parent_candidates(dataset, model_opt, model_label, topk=int(args.b_topk))
        max_evals, tune_epochs, tune_patience = _effective_budget(model_opt, args)

        for parent_order, parent in enumerate(parents, start=1):
            parent_id = str(parent.get("parent_profile_id", "B?"))
            parent_cfg = dict(parent.get("config", {}))
            parent_lo = float(parent.get("lr_lo", 5e-4))
            parent_hi = float(parent.get("lr_hi", 2e-3))
            parent_band = str(parent.get("lr_band_id", "BSEL"))
            parent_phase = str(parent.get("parent_run_phase", ""))

            for p_order, pid in enumerate(PROFILE_ORDER, start=1):
                cfg = _build_profile_config(model_opt, parent_cfg, pid)
                lr_lo, lr_hi = _build_lr_bounds(parent_lo, parent_hi, cfg)
                for seed_id in seeds:
                    run_cursor += 1
                    run_id = (
                        f"SC_{ds_tag}_{_sanitize_token(model_label, upper=True)}_"
                        f"{_sanitize_token(parent_id, upper=True)}_{pid}_S{int(seed_id)}"
                    )
                    run_phase = (
                        f"{PHASE_ID}_SC_D{dataset_order_idx:02d}_M{m_order:02d}_P{parent_order:02d}_"
                        f"{_sanitize_token(model_label, upper=True)}_{_sanitize_token(parent_id, upper=True)}_{pid}_S{int(seed_id)}"
                    )
                    row = {
                        "dataset": dataset,
                        "phase_id": PHASE_ID,
                        "axis_id": "SC",
                        "axis_desc": AXIS_DESC,
                        "setting_id": (
                            f"STAGEC_{_sanitize_token(model_label, upper=True)}_"
                            f"{_sanitize_token(parent_id, upper=True)}_{pid}_S{int(seed_id)}"
                        ),
                        "setting_key": "BASELINE_STAGEC_FOCUS",
                        "setting_desc": (
                            f"BASELINE_STAGEC_FOCUS_{_sanitize_token(model_label, upper=True)}_"
                            f"{_sanitize_token(parent_id, upper=True)}_{pid}_S{int(seed_id)}"
                        ),
                        "profile_id": pid,
                        "parent_profile_id": parent_id,
                        "parent_run_phase": parent_phase,
                        "concept_id": "FOCUS",
                        "detail_id": pid,
                        "lr_band_id": parent_band,
                        "hparam_id": f"{parent_id}_{pid}",
                        "seed_id": int(seed_id),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                        "stage_id": STAGE_ID,
                        "model_option": model_opt,
                        "model_label": model_label,
                        "max_evals": int(max_evals),
                        "tune_epochs": int(tune_epochs),
                        "tune_patience": int(tune_patience),
                        "lr_lo": float(lr_lo),
                        "lr_hi": float(lr_hi),
                        "lr_center": math.sqrt(float(lr_lo) * float(lr_hi)),
                        "config": cfg,
                    }
                    row["estimated_cost"] = _dataset_estimated_cost(row)
                    rows.append(row)
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
        "stage_id": STAGE_ID,
        "dataset": dataset,
        "execution_type": "stageC_focus",
        "model_count": len({str(r.get("model_option", "")).lower() for r in rows}),
        "profile_count": len({str(r.get("profile_id", "")) for r in rows}),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "model": r["model_label"],
                "profile_id": r["profile_id"],
                "parent_profile_id": r.get("parent_profile_id", ""),
                "seed_id": r["seed_id"],
                "lr_lo": r["lr_lo"],
                "lr_hi": r["lr_hi"],
                "config": r["config"],
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _smoke_trim_rows(rows: list[Dict[str, Any]], max_runs: int) -> list[Dict[str, Any]]:
    return rows[: max(1, int(max_runs))]


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
        subprocess.run([py, str(script), "--dataset", str(dataset), "--phase", str(phase)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        "print('[ENV_CHECK] Stage C model registration OK')\n"
    )
    proc = subprocess.run([py, "-c", script], cwd=EXP_DIR, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Model registration check failed:\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def _parse_trial_best_mrr_values(log_path: Path) -> list[float]:
    vals: list[float] = []
    if not log_path.exists():
        return vals
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return vals
    for line in lines:
        m = TRIAL_MRR_RE.match(str(line).strip())
        if not m:
            continue
        v = _metric_to_float(m.group(1))
        if v is not None:
            vals.append(float(v))
    return vals


def _top3_std(values: list[float]) -> float:
    if not values:
        return 0.0
    top = sorted(values, reverse=True)[:3]
    if len(top) <= 1:
        return 0.0
    try:
        return float(statistics.pstdev(top))
    except Exception:
        return 0.0


def _build_stagec_candidates(dataset: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    model_records: Dict[str, list[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        log_path = Path(str(row.get("log_path", "")))
        if not _is_completed_log(log_path):
            continue
        rec = _get_result_row_from_log(log_path, str(row.get("run_phase", "")), retries=2, sleep_sec=0.4)
        if not rec:
            continue
        valid = _metric_to_float(rec.get("best_mrr"))
        test = _metric_to_float(rec.get("test_mrr"))
        n_completed = int(rec.get("n_completed", 0) or 0)
        if n_completed <= 0:
            continue
        if valid is None:
            continue

        trials = _parse_trial_best_mrr_values(log_path)
        std_top3 = _top3_std(trials)
        max_evals = max(1, int(row.get("max_evals", 1) or 1))
        completion_ratio = max(0.0, min(1.0, float(n_completed) / float(max_evals)))
        score = float(valid) - 0.5 * float(std_top3) - 0.01 * (1.0 - float(completion_ratio))

        model_records[str(row["model_label"])].append(
            {
                "profile_id": str(row["profile_id"]),
                "parent_profile_id": str(row.get("parent_profile_id", "")),
                "run_phase": str(row["run_phase"]),
                "run_id": str(row["run_id"]),
                "lr_band_id": str(row["lr_band_id"]),
                "lr_lo": float(row["lr_lo"]),
                "lr_hi": float(row["lr_hi"]),
                "best_lr": _metric_to_float(rec.get("best_lr")),
                "valid_mrr20": float(valid),
                "test_mrr20": None if test is None else float(test),
                "std_top3_trial_valid_mrr20": float(std_top3),
                "completion_ratio": float(completion_ratio),
                "score": float(score),
                "n_completed": int(n_completed),
                "max_evals": int(max_evals),
                "config": dict(row.get("config", {})),
                "result_path": str(rec.get("path", "") or ""),
            }
        )

    models_payload: Dict[str, Any] = {}
    for model_name, recs in model_records.items():
        by_profile: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            combo = f"{rec.get('parent_profile_id', '')}>{rec.get('profile_id', '')}"
            by_profile[combo].append(rec)

        profiles: list[Dict[str, Any]] = []
        for combo_id, items in by_profile.items():
            items_sorted = sorted(items, key=lambda x: (float(x["score"]), float(x["valid_mrr20"])), reverse=True)
            best = items_sorted[0]
            scores = [float(x["score"]) for x in items]
            valids = [float(x["valid_mrr20"]) for x in items]
            tests = [float(x["test_mrr20"]) for x in items if x.get("test_mrr20") is not None]
            parent_id = str(best.get("parent_profile_id", ""))
            pid = str(best.get("profile_id", ""))
            profiles.append(
                {
                    "combo_id": combo_id,
                    "parent_profile_id": parent_id,
                    "profile_id": pid,
                    "profile_name": str(C_PROFILES.get(pid, {}).get("name", "")),
                    "n_runs": len(items),
                    "score_mean": float(sum(scores) / max(1, len(scores))),
                    "score_max": float(max(scores)) if scores else None,
                    "valid_mrr20_best": float(max(valids)) if valids else None,
                    "valid_mrr20_mean": float(sum(valids) / max(1, len(valids))),
                    "test_mrr20_mean": None if not tests else float(sum(tests) / len(tests)),
                    "lr_band_id": str(best["lr_band_id"]),
                    "lr_lo": float(best["lr_lo"]),
                    "lr_hi": float(best["lr_hi"]),
                    "best_lr": best.get("best_lr"),
                    "representative_run_phase": str(best["run_phase"]),
                    "representative_run_id": str(best["run_id"]),
                    "config": dict(best.get("config", {})),
                }
            )

        profiles.sort(key=lambda x: (float(x.get("score_mean", -1e9)), float(x.get("valid_mrr20_best", -1e9) or -1e9)), reverse=True)
        selected = list(profiles[:2])

        models_payload[model_name] = {
            "n_records": len(recs),
            "profiles": profiles,
            "selected_profiles": selected,
            "selection_rule": "Top-2 + Stability",
        }

    return {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "stage_id": STAGE_ID,
        "dataset": str(dataset),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "score_formula": "score = valid_mrr20 - 0.5*std(top3_trial_valid_mrr20) - 0.01*(1-completion_ratio)",
        "selection_rule": "Top-2 + Stability",
        "models": models_payload,
    }


def _write_stagec_candidates(dataset: str, rows: list[Dict[str, Any]]) -> Path:
    payload = _build_stagec_candidates(dataset, rows)
    path = _candidates_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, cmd: list[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"model={row.get('model_label','')} profile_id={row.get('profile_id','')} "
            f"parent_profile_id={row.get('parent_profile_id','')} seed={row.get('seed_id','')}"
        ),
        (
            f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)} "
            f"lr_band={row.get('lr_band_id','')} lr_lo={row.get('lr_lo','')} lr_hi={row.get('lr_hi','')}"
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline Stage C (focus) launcher")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--models", default="sasrec,gru4rec,duorec,difsr,fame")
    p.add_argument("--profiles", default=",".join(PROFILE_ORDER))
    p.add_argument("--b-topk", type=int, default=2)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seeds", default="1")
    p.add_argument("--seed-base", type=int, default=180000)

    p.add_argument("--max-evals-default", type=int, default=6)
    p.add_argument("--max-evals-duorec", type=int, default=4)
    p.add_argument("--max-evals-fame", type=int, default=5)

    p.add_argument("--tune-epochs-default", type=int, default=44)
    p.add_argument("--tune-epochs-duorec", type=int, default=30)
    p.add_argument("--tune-epochs-fame", type=int, default=36)

    p.add_argument("--tune-patience-default", type=int, default=6)
    p.add_argument("--tune-patience-duorec", type=int, default=4)
    p.add_argument("--tune-patience-fame", type=int, default=5)

    p.add_argument("--manifest-out", default="")
    p.add_argument("--resume-from-logs", action="store_true", default=True)
    p.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    p.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    p.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    p.set_defaults(verify_logging=True)

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-max-runs", type=int, default=4)
    return p.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals_default = 1
    args.max_evals_duorec = 1
    args.max_evals_fame = 1
    args.tune_epochs_default = 1
    args.tune_epochs_duorec = 1
    args.tune_epochs_fame = 1
    args.tune_patience_default = 1
    args.tune_patience_duorec = 1
    args.tune_patience_fame = 1
    args.seeds = "1"
    gpus = _parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"
    ds = _parse_csv_strings(args.datasets)
    args.datasets = ds[0] if ds else DEFAULT_DATASETS[0]


def _run_dataset(dataset: str, args: argparse.Namespace, *, dataset_order_idx: int, gpus: list[str]) -> int:
    _validate_session_fixed_files(dataset)
    rows = _build_rows(dataset, args, dataset_order_idx=dataset_order_idx)

    req_profiles = [x.upper() for x in _parse_csv_strings(args.profiles)]
    if req_profiles:
        rows = [r for r in rows if str(r.get("profile_id", "")).upper() in req_profiles]
    if args.smoke_test:
        rows = _smoke_trim_rows(rows, args.smoke_max_runs)

    manifest = _write_manifest(dataset, args, rows)
    summary = _summary_path(dataset)
    _ensure_summary_csv(summary)
    g_valid, g_test, m_valid, m_test = _load_summary_bests(summary)

    print(f"[{PHASE_ID}] dataset={dataset} rows={len(rows)} axis={AXIS} manifest={manifest}")

    for r in rows:
        r["log_path"] = str(_build_log_path(r))

    runnable: list[Dict[str, Any]] = []
    skipped = 0
    for r in rows:
        completed = _is_completed_any_log(r, use_resume=bool(args.resume_from_logs))
        if not completed:
            runnable.append(r)
            continue
        if not args.verify_logging:
            skipped += 1
            continue

        lpath = Path(str(r.get("log_path", "")))
        rec = _get_result_row_from_log(lpath, str(r.get("run_phase", "")), retries=2, sleep_sec=0.2)
        if not rec:
            runnable.append(r)
            continue
        ok, detail = _verify_special_from_result(str(rec.get("path", "")))
        if ok:
            skipped += 1
            continue
        print(f"[resume-check] run={r['run_phase']} special_check_failed -> rerun ({detail})")
        runnable.append(r)

    if skipped > 0:
        print(f"[{PHASE_ID}] skipped {skipped} completed runs for dataset={dataset}")

    if not runnable:
        cpath = _write_stagec_candidates(dataset, rows)
        print(f"[{PHASE_ID}] all done via resume for dataset={dataset} candidates={cpath}")
        _update_baseline_phase_summary(dataset, PHASE_ID)
        return 0

    print(f"[queue] dataset={dataset} tasks_total={len(runnable)} gpus={','.join(gpus)}")

    if args.dry_run:
        bins = _plan_gpu_bins(runnable, gpus)
        for gid in gpus:
            for r in bins[gid]:
                cmd = _build_command(r, gid)
                print(
                    f"[dry-run] dataset={dataset} gpu={gid} run_phase={r['run_phase']} "
                    f"model={r['model_label']} profile={r['profile_id']} "
                    f"lr=[{r['lr_lo']:.3e},{r['lr_hi']:.3e}]"
                )
                print("          " + " ".join(cmd))
        return 0

    q: deque = deque(runnable)
    active: Dict[str, Dict[str, Any]] = {}

    try:
        while True:
            for gid in gpus:
                if gid in active or not q:
                    continue
                r = q.popleft()
                cmd = _build_command(r, gid)
                log = Path(str(r["log_path"]))
                _write_log_preamble(log, r, gid, cmd)
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("HYPEROPT_RESULTS_DIR", str(ARTIFACT_ROOT / "results"))
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                with log.open("a", encoding="utf-8") as fh:
                    proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
                active[gid] = {"proc": proc, "row": r, "log_path": str(log)}
                print(f"[launch] dataset={dataset} gpu={gid} run_phase={r['run_phase']} model={r['model_label']} profile={r['profile_id']}")

            done: list[str] = []
            for gid, slot in active.items():
                proc = slot["proc"]
                rc = proc.poll()
                if rc is None:
                    continue
                done.append(gid)
                r = slot["row"]
                print(f"[done] dataset={dataset} gpu={gid} run_phase={r['run_phase']} rc={rc}")

                run_best = None
                run_test = None
                n_completed = None
                interrupted = None
                result_path = ""
                special_ok = False
                status = "run_complete" if int(rc) == 0 else "run_fail"

                rec = _get_result_row_from_log(Path(str(slot.get("log_path", ""))), str(r["run_phase"]), retries=4, sleep_sec=0.6)
                if rec:
                    run_best = _metric_to_float(rec.get("best_mrr"))
                    run_test = _metric_to_float(rec.get("test_mrr"))
                    n_completed = int(rec.get("n_completed", 0) or 0)
                    interrupted = bool(rec.get("interrupted", False))
                    result_path = str(rec.get("path", "") or "")
                    if result_path:
                        special_ok, detail = _verify_special_from_result(result_path)
                        print(f"[logging-check] run={r['run_phase']} {detail} result={result_path}")
                        _mirror_logging_bundle(r, result_path)
                        enforce_special = bool(args.verify_logging and int(rc) == 0 and int(n_completed) > 0)
                        if enforce_special and not special_ok:
                            raise RuntimeError(
                                f"special logging verification failed: run_phase={r['run_phase']} "
                                f"special_ok={special_ok} result={result_path}"
                            )
                if int(rc) == 0 and n_completed is not None and int(n_completed) <= 0:
                    status = "run_fail_zero_trials"

                if run_best is not None:
                    g_valid = run_best if g_valid is None else max(float(g_valid), float(run_best))
                    model = str(r["model_label"])
                    prev = m_valid.get(model)
                    m_valid[model] = run_best if prev is None else max(float(prev), float(run_best))
                if run_test is not None:
                    g_test = run_test if g_test is None else max(float(g_test), float(run_test))
                    model = str(r["model_label"])
                    prev = m_test.get(model)
                    m_test[model] = run_test if prev is None else max(float(prev), float(run_test))

                row_out = {
                    "global_best_valid_mrr20": "" if g_valid is None else f"{float(g_valid):.6f}",
                    "global_best_test_mrr20": "" if g_test is None else f"{float(g_test):.6f}",
                    "model": r["model_label"],
                    "model_best_valid_mrr20": "" if m_valid.get(str(r["model_label"])) is None else f"{float(m_valid[str(r['model_label'])]):.6f}",
                    "model_best_test_mrr20": "" if m_test.get(str(r["model_label"])) is None else f"{float(m_test[str(r['model_label'])]):.6f}",
                    "run_best_valid_mrr20": "" if run_best is None else f"{float(run_best):.6f}",
                    "run_best_test_mrr20": "" if run_test is None else f"{float(run_test):.6f}",
                    "profile_id": r["profile_id"],
                    "concept_id": "FOCUS",
                    "detail_id": r["profile_id"],
                    "lr_band_id": r["lr_band_id"],
                    "lr_lo": f"{float(r['lr_lo']):.6e}",
                    "lr_hi": f"{float(r['lr_hi']):.6e}",
                    "lr_center": f"{float(r['lr_center']):.6e}",
                    "stage_id": STAGE_ID,
                    "run_phase": r["run_phase"],
                    "run_id": r["run_id"],
                    "dataset": dataset,
                    "hparam_id": r["hparam_id"],
                    "seed_id": r["seed_id"],
                    "gpu_id": gid,
                    "status": status,
                    "n_completed": "" if n_completed is None else int(n_completed),
                    "interrupted": "" if interrupted is None else bool(interrupted),
                    "special_ok": bool(special_ok),
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                row_out["parent_profile_id"] = str(r.get("parent_profile_id", ""))
                row_out["parent_run_phase"] = str(r.get("parent_run_phase", ""))
                _append_summary_row(summary, row_out)
                _update_baseline_phase_summary(dataset, PHASE_ID)

                if int(rc) != 0:
                    raise RuntimeError(f"run failed: dataset={dataset} run_phase={r['run_phase']} rc={rc}")

            for gid in done:
                active.pop(gid, None)

            if not q and not active:
                break
            time.sleep(1)
    except Exception:
        _terminate_active(active)
        raise

    cpath = _write_stagec_candidates(dataset, rows)
    print(f"[{PHASE_ID}] summary updated: {summary}")
    print(f"[{PHASE_ID}] candidates updated: {cpath}")
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

    print(
        f"[config] datasets={','.join(datasets)} gpus={','.join(gpus)} models={','.join(models)} "
        f"profiles={args.profiles} b_topk={args.b_topk} seeds={args.seeds} seed_base={args.seed_base} "
        f"max_evals(default/duorec/fame)={args.max_evals_default}/{args.max_evals_duorec}/{args.max_evals_fame} "
        f"tune_epochs(default/duorec/fame)={args.tune_epochs_default}/{args.tune_epochs_duorec}/{args.tune_epochs_fame} "
        f"tune_patience(default/duorec/fame)={args.tune_patience_default}/{args.tune_patience_duorec}/{args.tune_patience_fame} "
        f"smoke_test={args.smoke_test}"
    )

    _check_runtime_models(models)

    for d_idx, ds in enumerate(datasets):
        rc = _run_dataset(ds, args, dataset_order_idx=d_idx, gpus=gpus)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
