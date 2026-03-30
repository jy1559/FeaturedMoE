#!/usr/bin/env python3
"""Launch baseline Stage A LR scans (anchor-2, core-5)."""

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
AXIS = "StageA_LR_anchor2_core5"
PHASE_ID = "P16"
PHASE_NAME = "STAGEA_LR_ANCHOR2_CORE5"
AXIS_DESC = "stagea_lr_anchor2_core5"
STAGE_ID = "A"

DEFAULT_DATASETS = ["lastfm0.03", "amazon_beauty"]
MODEL_SPECS = [
    {"model_option": "sasrec", "model_label": "SASRec"},
    {"model_option": "gru4rec", "model_label": "GRU4Rec"},
    {"model_option": "duorec", "model_label": "DuoRec"},
    {"model_option": "difsr", "model_label": "DIFSR"},
    {"model_option": "fame", "model_label": "FAME"},
]
MODEL_LABEL_BY_OPTION = {str(m["model_option"]): str(m["model_label"]) for m in MODEL_SPECS}

# Stage A: 6 narrow LR bands (high-LR coverage included).
DEFAULT_LR_BANDS = [
    {"band_id": "LR1", "lr_lo": 2.0e-4, "lr_hi": 7.0e-4},
    {"band_id": "LR2", "lr_lo": 4.0e-4, "lr_hi": 1.2e-3},
    {"band_id": "LR3", "lr_lo": 8.0e-4, "lr_hi": 2.4e-3},
    {"band_id": "LR4", "lr_lo": 1.6e-3, "lr_hi": 4.8e-3},
    {"band_id": "LR5", "lr_lo": 3.2e-3, "lr_hi": 7.5e-3},
    {"band_id": "LR6", "lr_lo": 5.0e-3, "lr_hi": 1.0e-2},
]

# Stage A rules requested by user.
LR_CLAMP_MIN = 8.0e-5
LR_CLAMP_MAX = 1.0e-2
MAX_LEN_DEFAULT = 10

DATASET_CONFIG_MAP = {
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
}
DATASET_SHORT_TAG = {
    "lastfm0.03": "LFM",
    "amazon_beauty": "AB",
}

DATASET_LR_MULT = {
    "lastfm0.03": 0.90,
    "amazon_beauty": 1.20,
}

MODEL_LR_MULT = {
    "sasrec": 1.00,
    "gru4rec": 0.85,
    "duorec": 0.90,
    "difsr": 0.95,
    "fame": 0.80,
}

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

# Stage A fixed hparams (LR-only search inside each run).
MODEL_FIXED = {
    "sasrec": {
        "hidden_size": 128,
        "embedding_size": 128,
        "n_layers": 2,
        "num_layers": 2,
        "n_heads": 4,
        "num_heads": 4,
        "inner_size": 256,
        "max_len": MAX_LEN_DEFAULT,
        "dropout": 0.20,
        "weight_decay": 3e-4,
    },
    "gru4rec": {
        "hidden_size": 160,
        "embedding_size": 160,
        "num_layers": 1,
        "n_layers": 1,
        "max_len": MAX_LEN_DEFAULT,
        "dropout": 0.20,
        "weight_decay": 2e-4,
    },
    "duorec": {
        "hidden_size": 96,
        "embedding_size": 96,
        "n_layers": 1,
        "num_layers": 1,
        "n_heads": 2,
        "num_heads": 2,
        "inner_size": 192,
        "max_len": MAX_LEN_DEFAULT,
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
        "n_layers": 1,
        "num_layers": 1,
        "n_heads": 4,
        "num_heads": 4,
        "inner_size": 256,
        "attribute_hidden_size": 128,
        "max_len": MAX_LEN_DEFAULT,
        "dropout": 0.20,
        "weight_decay": 2e-4,
        "fusion_type": "gate",
        "use_attribute_predictor": True,
        "lambda_attr": 0.10,
    },
    "fame": {
        "hidden_size": 128,
        "embedding_size": 128,
        "n_layers": 1,
        "num_layers": 1,
        "n_heads": 4,
        "num_heads": 4,
        "inner_size": 256,
        "num_experts": 2,
        "max_len": MAX_LEN_DEFAULT,
        "dropout": 0.20,
        "weight_decay": 3e-4,
    },
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


def _fmt_lr_short(value: float) -> str:
    text = f"{float(value):.1e}".replace("+", "")
    # 2.0e-04 -> 2e4
    text = text.replace(".0", "")
    text = text.replace("e-0", "e").replace("e-", "e").replace("e+", "e")
    return text


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(v)))


def _resolve_lr_bands(text: str) -> list[Dict[str, Any]]:
    raw = _parse_csv_strings(text)
    if not raw or (len(raw) == 1 and raw[0].strip().upper() == "AUTO6"):
        out: list[Dict[str, Any]] = []
        for item in DEFAULT_LR_BANDS:
            lo = float(item["lr_lo"])
            hi = float(item["lr_hi"])
            bid = str(item["band_id"])
            pid = f"{bid}_{_fmt_lr_short(lo)}_{_fmt_lr_short(hi)}"
            out.append({"band_id": bid, "profile_id": pid, "lr_lo": lo, "lr_hi": hi})
        return out

    out: list[Dict[str, Any]] = []
    for idx, token in enumerate(raw, start=1):
        t = str(token).strip()
        if ":" not in t:
            raise RuntimeError("Invalid --lr-bands token. Expected 'lo:hi', e.g. 2e-4:7e-4")
        lo_text, hi_text = t.split(":", 1)
        lo = float(lo_text.strip())
        hi = float(hi_text.strip())
        if not (lo > 0.0 and hi > lo):
            raise RuntimeError(f"Invalid lr band bounds: {token}")
        bid = f"LR{idx}"
        pid = f"{bid}_{_sanitize_token(lo_text, upper=False)}_{_sanitize_token(hi_text, upper=False)}"
        out.append({"band_id": bid, "profile_id": pid, "lr_lo": lo, "lr_hi": hi})
    return out


def _dataset_tag(dataset: str) -> str:
    return str(dataset).replace("/", "_")


def _model_dir_name(model_label: str) -> str:
    raw = str(model_label or "").strip()
    if not raw:
        return "unknown_model"
    return raw.replace("/", "_")


def _profile_dir_name(profile_id: str) -> str:
    return _sanitize_token(str(profile_id or ""), upper=False)


def _dataset_short_tag(dataset: str) -> str:
    key = str(dataset or "").strip()
    if key in DATASET_SHORT_TAG:
        return str(DATASET_SHORT_TAG[key])
    return _sanitize_token(key, upper=True)[:10]


def _lr_band_token(lr_band_id: str) -> str:
    return _sanitize_token(str(lr_band_id or ""), upper=False)


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
    return _phase_dataset_dir(dataset) / "stageA_candidates.json"


def _dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip()
    if key not in DATASET_CONFIG_MAP:
        raise RuntimeError(f"Unsupported dataset for Stage A runner: {dataset}")
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
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        lr_val = _metric_to_float(best_params.get("learning_rate"))
        if lr_val is not None:
            return lr_val
    return _metric_to_float(payload.get("best_learning_rate"))


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
            "best_lr": _extract_best_lr(payload),
            "n_completed": int(payload.get("n_completed", 0) or 0),
            "interrupted": bool(payload.get("interrupted", False)),
            "path": str(path.resolve()),
            "mtime": float(path.stat().st_mtime),
        }
        prev = out.get(run_phase)
        if prev is None or float(rec["mtime"]) >= float(prev.get("mtime", 0.0)):
            out[run_phase] = rec
    return out


def _get_result_row(dataset: str, run_phase: str, retries: int = 8, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
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
        "best_lr": _extract_best_lr(payload),
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
        "global_best_valid_mrr20",
        "global_best_test_mrr20",
        "model",
        "model_best_valid_mrr20",
        "model_best_test_mrr20",
        "run_best_valid_mrr20",
        "run_best_test_mrr20",
        "profile_id",
        "concept_id",
        "detail_id",
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
    fieldnames = _summary_fieldnames()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                existing_fields = [f for f in (reader.fieldnames or []) if f]
                existing_rows = list(reader)
        except Exception:
            existing_fields = []
            existing_rows = []
        if existing_fields == fieldnames:
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
                model_valid = _metric_to_float(row.get("model_best_valid_mrr20"))
                model_test = _metric_to_float(row.get("model_best_test_mrr20"))
                global_valid = _metric_to_float(row.get("global_best_valid_mrr20"))
                global_test = _metric_to_float(row.get("global_best_test_mrr20"))

                if run_valid is not None:
                    global_best_valid = run_valid if global_best_valid is None else max(float(global_best_valid), float(run_valid))
                    if model:
                        prev = model_best_valid.get(model)
                        model_best_valid[model] = run_valid if prev is None else max(float(prev), float(run_valid))
                if run_test is not None:
                    global_best_test = run_test if global_best_test is None else max(float(global_best_test), float(run_test))
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
                    global_best_valid = global_valid if global_best_valid is None else max(float(global_best_valid), float(global_valid))
                if global_test is not None:
                    global_best_test = global_test if global_best_test is None else max(float(global_best_test), float(global_test))
    except Exception:
        return None, None, {}, {}
    return global_best_valid, global_best_test, model_best_valid, model_best_test


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path)
    payload = {k: row.get(k, "") for k in _summary_fieldnames()}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_summary_fieldnames())
        writer.writerow(payload)


def _write_log_preamble(log_file: Path, row: Dict[str, Any], gpu_id: str, cmd: list[str]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{PHASE_NAME}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} run_id={row.get('run_id','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"model={row.get('model_label','')} profile_id={row.get('profile_id','')} seed={row.get('seed_id','')}"
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


def _build_log_path(row: Dict[str, Any]) -> Path:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    model_dir = _model_dir_name(str(row["model_label"]))
    model_token = str(row["model_label"])
    lr_token = _lr_band_token(str(row.get("lr_band_id", "")))
    dataset_token = _dataset_short_tag(str(row["dataset"]))
    seed_id = int(row["seed_id"])
    seed_suffix = "" if seed_id == 1 else f"_S{seed_id}"
    filename = f"A_{lr_token}_{dataset_token}_{model_token}{seed_suffix}.log"
    return ds_dir / model_dir / filename


def _legacy_log_paths(row: Dict[str, Any]) -> list[Path]:
    ds_dir = _phase_dataset_dir(str(row["dataset"]))
    profile_id = _profile_dir_name(str(row["profile_id"]))
    model_token = _sanitize_token(str(row["model_label"]), upper=True)
    model_dir = _model_dir_name(str(row["model_label"]))
    seed_id = int(row["seed_id"])
    filename = f"{PHASE_ID}_SA_{AXIS_DESC}_{model_token}_{profile_id}_S{seed_id}.log"
    return [
        ds_dir / profile_id / filename,
        ds_dir / model_dir / profile_id / filename,
    ]


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


def _effective_lr_bounds(dataset: str, model_option: str, lr_lo: float, lr_hi: float) -> tuple[float, float]:
    d_mult = float(DATASET_LR_MULT.get(str(dataset), 1.0))
    m_mult = float(MODEL_LR_MULT.get(str(model_option), 1.0))
    lo = _clamp(float(lr_lo) * d_mult * m_mult, LR_CLAMP_MIN, LR_CLAMP_MAX)
    hi = _clamp(float(lr_hi) * d_mult * m_mult, LR_CLAMP_MIN, LR_CLAMP_MAX)
    if hi <= lo:
        hi = min(LR_CLAMP_MAX, max(lo * 1.3, lo + 1e-6))
    return lo, hi


def _effective_budget(model_option: str, args: argparse.Namespace) -> tuple[int, int, int]:
    model = str(model_option).lower()
    if model == "duorec":
        return int(args.max_evals_duorec), int(args.tune_epochs_duorec), int(args.tune_patience_duorec)
    if model == "fame":
        return int(args.max_evals_fame), int(args.tune_epochs_fame), int(args.tune_patience_fame)
    return int(args.max_evals_default), int(args.tune_epochs_default), int(args.tune_patience_default)


def _model_runtime_resource_overrides(model_option: str) -> list[str]:
    model = str(model_option).lower()
    if model == "sasrec":
        return ["++train_batch_size=4096", "++eval_batch_size=6144"]
    if model == "gru4rec":
        return ["++train_batch_size=8192", "++eval_batch_size=10240"]
    if model == "duorec":
        return ["++train_batch_size=1024", "++eval_batch_size=1536"]
    if model == "difsr":
        return ["++train_batch_size=3072", "++eval_batch_size=4096"]
    if model == "fame":
        return ["++train_batch_size=1536", "++eval_batch_size=2304"]
    return []


def _build_command(row: Dict[str, Any], gpu_id: str) -> list[str]:
    model_option = str(row["model_option"]).lower()
    fixed = dict(MODEL_FIXED[model_option])
    max_evals = int(row["max_evals"])
    tune_epochs = int(row["tune_epochs"])
    tune_patience = int(row["tune_patience"])

    lr_lo = float(row["lr_lo"])
    lr_hi = float(row["lr_hi"])
    dropout = float(fixed["dropout"])
    weight_decay = float(fixed["weight_decay"])

    search_lr_only = {"learning_rate": [lr_lo, lr_hi]}
    search_type_lr_only = {"learning_rate": "loguniform"}

    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        _dataset_config_name(str(row["dataset"])),
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
        f"++MAX_ITEM_LIST_LENGTH={int(fixed['max_len'])}",
    ]

    fixed_search: Dict[str, Any] = {
        "weight_decay": weight_decay,
        "hidden_size": int(fixed["hidden_size"]),
        "embedding_size": int(fixed["embedding_size"]),
        "inner_size": int(fixed.get("inner_size", fixed["hidden_size"] * 2)),
        "n_layers": int(fixed.get("n_layers", fixed.get("num_layers", 1))),
        "num_layers": int(fixed.get("num_layers", fixed.get("n_layers", 1))),
        "n_heads": int(fixed.get("n_heads", fixed.get("num_heads", 1))),
        "num_heads": int(fixed.get("num_heads", fixed.get("n_heads", 1))),
        "dropout_ratio": dropout,
        "dropout_prob": dropout,
        "hidden_dropout_prob": dropout,
        "attn_dropout_prob": dropout,
        "num_experts": int(fixed.get("num_experts", 2)),
    }
    for key, value in fixed_search.items():
        cmd.append(f"++search.{key}={hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")

    if model_option == "gru4rec":
        cmd.append(f"++dropout_prob={dropout}")
    else:
        cmd.append(f"++dropout_ratio={dropout}")

    # Fixed model knobs.
    cmd.extend(
        [
            f"++hidden_size={int(fixed['hidden_size'])}",
            f"++embedding_size={int(fixed['embedding_size'])}",
            f"++n_layers={int(fixed.get('n_layers', fixed.get('num_layers', 1)))}",
            f"++num_layers={int(fixed.get('num_layers', fixed.get('n_layers', 1)))}",
        ]
    )
    if "n_heads" in fixed or "num_heads" in fixed:
        cmd.extend(
            [
                f"++n_heads={int(fixed.get('n_heads', fixed.get('num_heads', 1)))}",
                f"++num_heads={int(fixed.get('num_heads', fixed.get('n_heads', 1)))}",
            ]
        )
    if "inner_size" in fixed:
        cmd.append(f"++inner_size={int(fixed['inner_size'])}")

    if model_option == "duorec":
        cmd.extend(
            [
                f"++contrast={fixed['contrast']}",
                f"++tau={float(fixed['tau'])}",
                f"++lmd={float(fixed['lmd'])}",
                f"++lmd_sem={float(fixed['lmd_sem'])}",
                f"++semantic_sample_max_tries={int(fixed['semantic_sample_max_tries'])}",
            ]
        )
    elif model_option == "difsr":
        cmd.extend(
            [
                f"++attribute_hidden_size={int(fixed['attribute_hidden_size'])}",
                f"++fusion_type={fixed['fusion_type']}",
                f"++use_attribute_predictor={'true' if bool(fixed['use_attribute_predictor']) else 'false'}",
                f"++lambda_attr={float(fixed['lambda_attr'])}",
            ]
        )
    elif model_option == "fame":
        cmd.append(f"++num_experts={int(fixed['num_experts'])}")

    cmd.extend(_model_runtime_resource_overrides(model_option))
    return cmd


def _dataset_estimated_cost(row: Dict[str, Any]) -> float:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"]).lower()
    max_evals = max(1, int(row["max_evals"]))
    tune_epochs = max(1, int(row["tune_epochs"]))
    return (
        float(DATASET_COST_WEIGHT.get(dataset, 1.0))
        * float(MODEL_COST_WEIGHT.get(model_option, 1.0))
        * (float(max_evals) / 5.0)
        * (float(tune_epochs) / 30.0)
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

    lr_bands = _resolve_lr_bands(args.lr_bands)
    if not lr_bands:
        raise RuntimeError("No LR bands resolved")

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    ds_tag = _sanitize_token(dataset, upper=True)
    model_specs = [m for m in MODEL_SPECS if str(m["model_option"]).lower() in requested_models]

    for model_order, spec in enumerate(model_specs, start=1):
        model_label = str(spec["model_label"])
        model_option = str(spec["model_option"])
        max_evals, tune_epochs, tune_patience = _effective_budget(model_option, args)

        for band_order, band in enumerate(lr_bands, start=1):
            band_id = str(band["band_id"])
            profile_id = str(band["profile_id"])
            lo, hi = _effective_lr_bounds(dataset, model_option, float(band["lr_lo"]), float(band["lr_hi"]))

            for seed_id in seeds:
                run_cursor += 1
                run_id = f"SA_{ds_tag}_{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}"
                run_phase = (
                    f"{PHASE_ID}_SA_D{dataset_order_idx:02d}_M{model_order:02d}_"
                    f"{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}"
                )
                row = {
                    "dataset": dataset,
                    "phase_id": PHASE_ID,
                    "axis_id": "SA",
                    "axis_desc": AXIS_DESC,
                    "setting_id": f"STAGEA_{_sanitize_token(model_label, upper=True)}_{profile_id}_S{int(seed_id)}",
                    "setting_key": "BASELINE_STAGEA_LR",
                    "setting_desc": (
                        f"BASELINE_STAGEA_LR_{_sanitize_token(model_label, upper=True)}_"
                        f"{profile_id}_S{int(seed_id)}"
                    ),
                    "profile_id": profile_id,
                    "concept_id": "LR",
                    "detail_id": band_id,
                    "lr_band_id": band_id,
                    "hparam_id": profile_id,
                    "seed_id": int(seed_id),
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "stage": "stageA",
                    "stage_id": STAGE_ID,
                    "model_option": model_option,
                    "model_label": model_label,
                    "band_order": int(band_order),
                    "max_evals": int(max_evals),
                    "tune_epochs": int(tune_epochs),
                    "tune_patience": int(tune_patience),
                    "lr_lo": float(lo),
                    "lr_hi": float(hi),
                    "lr_center": math.sqrt(float(lo) * float(hi)),
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
    lr_bands = sorted({str(r.get("lr_band_id", "")) for r in rows if str(r.get("lr_band_id", ""))})
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "stage_id": STAGE_ID,
        "dataset": dataset,
        "execution_type": "stageA_lr",
        "model_count": len({str(r.get("model_option", "")).lower() for r in rows}),
        "lr_band_count": len(lr_bands),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "lr_clamp": {"min": LR_CLAMP_MIN, "max": LR_CLAMP_MAX},
        "max_len_default": MAX_LEN_DEFAULT,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "model": r["model_label"],
                "lr_band_id": r["lr_band_id"],
                "profile_id": r["profile_id"],
                "seed_id": r["seed_id"],
                "lr_lo": r["lr_lo"],
                "lr_hi": r["lr_hi"],
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
        "print('[ENV_CHECK] Stage A model registration OK')\n"
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


def _build_stagea_candidates(dataset: str, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    model_records: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        log_path = Path(str(row.get("log_path", "")))
        if not _is_completed_log(log_path):
            continue
        rec = _get_result_row_from_log_or_scan(
            dataset=str(dataset),
            run_phase=str(row.get("run_phase", "")),
            log_path=log_path,
            retries=2,
            sleep_sec=0.4,
        )
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
                "run_phase": str(row["run_phase"]),
                "run_id": str(row["run_id"]),
                "profile_id": str(row["profile_id"]),
                "lr_band_id": str(row["lr_band_id"]),
                "band_order": int(row.get("band_order", 0) or 0),
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
                "result_path": str(rec.get("path", "") or ""),
            }
        )

    models_payload: Dict[str, Any] = {}
    n_bands = len(_resolve_lr_bands("AUTO6"))

    for model_name, recs in model_records.items():
        # aggregate per band (seed-aware)
        by_band: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        for rec in recs:
            by_band[int(rec.get("band_order", 0))].append(rec)

        bands: list[Dict[str, Any]] = []
        for band_idx, items in by_band.items():
            items_sorted = sorted(items, key=lambda x: (float(x.get("score", -1e9)), float(x.get("valid_mrr20", -1e9))), reverse=True)
            scores = [float(x.get("score", 0.0)) for x in items]
            valids = [float(x.get("valid_mrr20", 0.0)) for x in items]
            tests = [float(x.get("test_mrr20")) for x in items if x.get("test_mrr20") is not None]
            stds = [float(x.get("std_top3_trial_valid_mrr20", 0.0)) for x in items]
            comps = [float(x.get("completion_ratio", 0.0)) for x in items]
            best_item = items_sorted[0]
            bands.append(
                {
                    "lr_band_id": str(best_item["lr_band_id"]),
                    "band_order": int(band_idx),
                    "profile_id": str(best_item["profile_id"]),
                    "lr_lo": float(best_item["lr_lo"]),
                    "lr_hi": float(best_item["lr_hi"]),
                    "best_lr": best_item.get("best_lr"),
                    "n_runs": len(items),
                    "score_mean": float(sum(scores) / max(1, len(scores))),
                    "score_max": float(max(scores)) if scores else None,
                    "valid_mrr20_mean": float(sum(valids) / max(1, len(valids))),
                    "valid_mrr20_best": float(max(valids)) if valids else None,
                    "test_mrr20_mean": None if not tests else float(sum(tests) / len(tests)),
                    "std_top3_trial_valid_mrr20_mean": float(sum(stds) / max(1, len(stds))),
                    "completion_ratio_mean": float(sum(comps) / max(1, len(comps))),
                    "representative_run_phase": str(best_item["run_phase"]),
                    "representative_run_id": str(best_item["run_id"]),
                }
            )

        bands.sort(
            key=lambda x: (
                float(x.get("score_mean", -1e9)),
                float(x.get("valid_mrr20_best", -1e9) or -1e9),
                -int(x.get("band_order", 0)),
            ),
            reverse=True,
        )

        selected = list(bands[:2])
        selected_orders = {int(x.get("band_order", 0)) for x in selected}
        band_best_by_order = {int(x.get("band_order", 0)): x for x in bands}

        if selected:
            top = selected[0]
            top_order = int(top.get("band_order", 0))
            edge_targets: list[int] = []

            # boundary promotion
            if top_order == 1:
                edge_targets.append(2)
            elif top_order == n_bands:
                edge_targets.append(n_bands - 1)

            # best-lr near boundary promotion
            best_lr = _metric_to_float(top.get("best_lr"))
            lo = float(top.get("lr_lo", 0.0) or 0.0)
            hi = float(top.get("lr_hi", 0.0) or 0.0)
            width = hi - lo
            if best_lr is not None and width > 0.0:
                pos = (float(best_lr) - lo) / width
                if pos <= 0.10:
                    edge_targets.append(top_order - 1)
                elif pos >= 0.90:
                    edge_targets.append(top_order + 1)

            for target in edge_targets:
                if target < 1 or target > n_bands or target in selected_orders:
                    continue
                cand = band_best_by_order.get(target)
                if cand is None:
                    continue
                selected.append(cand)
                selected_orders.add(target)

        selected.sort(key=lambda x: int(x.get("band_order", 0)))

        models_payload[model_name] = {
            "n_records": len(recs),
            "bands": bands,
            "selected_bands": selected,
            "selection_rule": "Top-2 + Stability; boundary/edge neighbor promotion",
        }

    return {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "stage_id": STAGE_ID,
        "dataset": str(dataset),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "score_formula": "score = valid_mrr20 - 0.5*std(top3_trial_valid_mrr20) - 0.01*(1-completion_ratio)",
        "selection_rule": "Top-2 + Stability with boundary/edge neighbor promotion",
        "models": models_payload,
    }


def _write_stagea_candidates(dataset: str, rows: list[Dict[str, Any]]) -> Path:
    payload = _build_stagea_candidates(dataset, rows)
    path = _candidates_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline Stage A LR launcher (anchor-2, core-5)")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default="sasrec,gru4rec,duorec,difsr,fame")
    parser.add_argument("--lr-bands", default="AUTO6")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=160000)

    parser.add_argument("--max-evals-default", type=int, default=6)
    parser.add_argument("--max-evals-duorec", type=int, default=4)
    parser.add_argument("--max-evals-fame", type=int, default=5)

    parser.add_argument("--tune-epochs-default", type=int, default=36)
    parser.add_argument("--tune-epochs-duorec", type=int, default=24)
    parser.add_argument("--tune-epochs-fame", type=int, default=32)

    parser.add_argument("--tune-patience-default", type=int, default=5)
    parser.add_argument("--tune-patience-duorec", type=int, default=3)
    parser.add_argument("--tune-patience-fame", type=int, default=4)

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
    if args.smoke_test:
        rows = _smoke_trim_rows(rows, args.smoke_max_runs)

    manifest_path = _write_manifest(dataset, args, rows)
    summary_path = _summary_path(dataset)
    _ensure_summary_csv(summary_path)
    global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model = _load_summary_bests(summary_path)

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
        candidates_path = _write_stagea_candidates(dataset, rows)
        print(f"[{PHASE_ID}] all runs completed for dataset={dataset}. candidates={candidates_path}")
        _update_baseline_phase_summary(dataset, PHASE_ID)
        return 0

    shared_queue: deque = deque(runnable)
    print(
        f"[queue] dataset={dataset} mode=shared_gpu_queue "
        f"tasks_total={len(shared_queue)} gpus={','.join(gpus)}"
    )

    if args.dry_run:
        gpu_bins = _plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = _build_command(row, gpu_id)
                print(
                    f"[dry-run] dataset={dataset} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} lr_band={row['lr_band_id']} seed={row['seed_id']} "
                    f"lr=[{row['lr_lo']:.3e},{row['lr_hi']:.3e}]"
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
                cmd = _build_command(row, gpu_id)
                log_path = Path(str(row["log_path"]))
                _write_log_preamble(log_path, row, gpu_id, cmd)
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
                    f"model={row['model_label']} lr_band={row['lr_band_id']}"
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
                        enforce_special = bool(args.verify_logging and int(rc) == 0 and int(n_completed) > 0)
                        if enforce_special and not special_ok:
                            raise RuntimeError(
                                f"special logging verification failed: run_phase={row['run_phase']} "
                                f"special_ok={special_ok} result={result_path}"
                            )
                if int(rc) == 0 and n_completed is not None and int(n_completed) <= 0:
                    status = "run_fail_zero_trials"

                if run_best is not None:
                    global_best_valid = run_best if global_best_valid is None else max(float(global_best_valid), float(run_best))
                    model_name = str(row["model_label"])
                    prev_model_best = model_best_valid_by_model.get(model_name)
                    model_best_valid_by_model[model_name] = run_best if prev_model_best is None else max(float(prev_model_best), float(run_best))

                if test_mrr is not None:
                    global_best_test = test_mrr if global_best_test is None else max(float(global_best_test), float(test_mrr))
                    model_name = str(row["model_label"])
                    prev_model_test_best = model_best_test_by_model.get(model_name)
                    model_best_test_by_model[model_name] = test_mrr if prev_model_test_best is None else max(float(prev_model_test_best), float(test_mrr))

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
                    "profile_id": row["profile_id"],
                    "concept_id": row["concept_id"],
                    "detail_id": row["detail_id"],
                    "lr_band_id": row["lr_band_id"],
                    "lr_lo": f"{float(row['lr_lo']):.6e}",
                    "lr_hi": f"{float(row['lr_hi']):.6e}",
                    "lr_center": f"{float(row['lr_center']):.6e}",
                    "stage_id": STAGE_ID,
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

    candidates_path = _write_stagea_candidates(dataset, rows)
    print(f"[{PHASE_ID}] summary updated: {summary_path}")
    print(f"[{PHASE_ID}] candidates updated: {candidates_path}")
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

    _ = _resolve_lr_bands(args.lr_bands)

    print(
        f"[config] datasets={','.join(datasets)} gpus={','.join(gpus)} models={','.join(models)} "
        f"lr_bands={args.lr_bands} seeds={args.seeds} seed_base={args.seed_base} "
        f"max_evals(default/duorec/fame)={args.max_evals_default}/{args.max_evals_duorec}/{args.max_evals_fame} "
        f"tune_epochs(default/duorec/fame)={args.tune_epochs_default}/{args.tune_epochs_duorec}/{args.tune_epochs_fame} "
        f"tune_patience(default/duorec/fame)={args.tune_patience_default}/{args.tune_patience_duorec}/{args.tune_patience_fame} "
        f"smoke_test={args.smoke_test}"
    )

    _check_runtime_models(models)

    for d_idx, dataset in enumerate(datasets):
        rc = _run_dataset(dataset, args, dataset_order_idx=d_idx, gpus=gpus)
        if rc != 0:
            return int(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
