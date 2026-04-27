#!/usr/bin/env python3
"""Focused add-on tuning pipeline for underperforming final_experiment baselines."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import (
    ARTIFACT_ROOT,
    CODE_DIR,
    DEFAULT_DATASETS,
    DEFAULT_MAX_RUN_HOURS,
    DEFAULT_OOM_RETRY_LIMIT,
    MODEL_LABELS,
    NUMERIC_FALLBACKS,
    CHOICE_FALLBACKS,
    STAGE3_SEED_COUNTS,
    build_common_search_entries,
    config_signature,
    dataset_config_name,
    dedupe_keep_order,
    extract_error_tail,
    has_run_status_end_normal,
    hydra_literal,
    load_manifest,
    load_result_payload,
    now_utc,
    normalize_dataset,
    normalize_model,
    narrow_space_from_configs,
    parse_csv_list,
    parse_result_path_from_log,
    parse_result_summary,
    python_bin,
    read_csv_rows,
    read_json,
    result_has_successful_trials,
    safe_float,
    safe_int,
    sanitize_token,
    stage_tune_epochs,
    stage_tune_patience,
    trial_test_mean,
    trial_valid_mean,
    upsert_csv_row,
    write_csv_rows,
    write_json,
)


TRACK = "final_experiment_addtuning"
LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK
RESULT_ROOT = ARTIFACT_ROOT / "results" / TRACK
EXPERIMENTS_DIR = CODE_DIR.parents[1]
HYPEROPT_TUNE_SCRIPT = EXPERIMENTS_DIR / "hyperopt_tune.py"

EXISTING_TRACKS = {
    "final_experiment_addtuning": {
        "log_root": ARTIFACT_ROOT / "logs" / "final_experiment_addtuning",
        "result_root": ARTIFACT_ROOT / "results" / "final_experiment_addtuning",
    },
    "final_experiment": {
        "log_root": ARTIFACT_ROOT / "logs" / "final_experiment",
        "result_root": ARTIFACT_ROOT / "results" / "final_experiment",
    },
    "final_experiment_5models": {
        "log_root": ARTIFACT_ROOT / "logs" / "final_experiment_5models",
        "result_root": ARTIFACT_ROOT / "results" / "results_final_experiment_5models",
    },
}

FOCUS_MODELS = ["bsarec", "difsr", "fame", "duorec", "fdsa", "fearec"]
MODEL_ORDER = ["bsarec", "difsr", "duorec", "fame", "fdsa", "fearec", "gru4rec", "sasrec", "tisasrec"]
MEDIUM_FALLBACK_MODELS = {"duorec", "fame", "fdsa", "fearec"}
MEDIUM_FALLBACK_DATASETS = {"movielens1m", "retail_rocket"}
SUMMARY_JSON = CODE_DIR / "addtuning_current_summary.json"
SUMMARY_CSV = CODE_DIR / "addtuning_current_summary.csv"
SUMMARY_MD = CODE_DIR / "addtuning_current_summary.md"

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

MODEL_PRIORITY_PARAMS = {
    "bsarec": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "bsarec_alpha", "bsarec_c", "MAX_ITEM_LIST_LENGTH"],
    "difsr": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "lambda_attr", "attribute_hidden_size", "fusion_type"],
    "duorec": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "tau", "lmd", "lmd_sem"],
    "fame": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "num_experts", "MAX_ITEM_LIST_LENGTH"],
    "fdsa": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "lambda_attr", "attribute_hidden_size", "MAX_ITEM_LIST_LENGTH"],
    "fearec": ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "tau", "lmd", "lmd_sem"],
}

TIER_POLICY = {
    "light": {
        "stage1_max_evals": 24,
        "stage1_lr_points": 5,
        "stage1_other": 3,
        "stage1_random": 2,
        "stage2_top_k": 2,
        "stage2_max_evals": 10,
        "stage2_lr_points": 4,
        "stage2_other": 3,
        "stage3_top_k": 1,
        "stage3_extra_seeds": 0,
        "stage4_enabled": True,
        "stage4_trigger_ratio": 0.0,
        "stage5_top_k": 1,
    },
    "medium": {
        "stage1_max_evals": 32,
        "stage1_lr_points": 6,
        "stage1_other": 4,
        "stage1_random": 3,
        "stage2_top_k": 2,
        "stage2_max_evals": 12,
        "stage2_lr_points": 5,
        "stage2_other": 4,
        "stage3_top_k": 2,
        "stage3_extra_seeds": 0,
        "stage4_enabled": True,
        "stage4_trigger_ratio": 0.94,
        "stage4_max_evals": 8,
        "stage4_lr_points": 4,
        "stage4_other": 3,
        "stage4_random": 2,
        "stage5_top_k": 1,
    },
    "heavy": {
        "stage1_max_evals": 40,
        "stage1_lr_points": 7,
        "stage1_other": 5,
        "stage1_random": 4,
        "stage2_top_k": 3,
        "stage2_max_evals": 14,
        "stage2_lr_points": 5,
        "stage2_other": 4,
        "stage3_top_k": 2,
        "stage3_extra_seeds": 0,
        "stage4_enabled": True,
        "stage4_trigger_ratio": 0.9,
        "stage4_max_evals": 10,
        "stage4_lr_points": 4,
        "stage4_other": 3,
        "stage4_random": 3,
        "stage5_top_k": 1,
    },
}

STAGE_TUNE_PLAN = {
    "stage1": {"tune_epochs": 40, "tune_patience": 5},
    "stage2": {"tune_epochs": 70, "tune_patience": 7},
    "stage3": {"tune_epochs": 100, "tune_patience": 10},
    "stage4": {"tune_epochs": 85, "tune_patience": 8},
    "stage5": {"tune_epochs": 100, "tune_patience": 10},
}

ADDTUNING_SEED_COUNTS = {
    "beauty": 2,
    "foursquare": 2,
    "KuaiRecLargeStrictPosV2_0.2": 2,
    "retail_rocket": 2,
    "movielens1m": 1,
    "lastfm0.03": 1,
}

DATASET_EVAL_SCALE = {
    "movielens1m": 0.5,
    "lastfm0.03": 0.55,
    "KuaiRecLargeStrictPosV2_0.2": 0.7,
    "retail_rocket": 0.8,
    "foursquare": 0.85,
    "beauty": 1.0,
}

DATASET_STAGE_BUDGET_BOOST = {
    "stage1": {
        "beauty": 1.5,
        "foursquare": 1.2,
    },
    "stage2": {
        "beauty": 1.5,
        "foursquare": 1.2,
    },
}

DATASET_MAX_RUN_HOURS = {
    "movielens1m": 2.0,
    "lastfm0.03": 2.0,
    "KuaiRecLargeStrictPosV2_0.2": 2.0,
    "retail_rocket": 2.0,
    "foursquare": 2.0,
    "beauty": 2.0,
}

APPENDIX_CANDIDATE_OVERRIDES = {
    "weight_decay": [5e-7, 1e-6, 1e-5, 5e-5, 1e-4, 1.5e-4],
    "MAX_ITEM_LIST_LENGTH": [10, 20, 30, 50],
    "hidden_size": [64, 96, 112, 128, 160, 192],
    "embedding_size": [64, 96, 112, 128, 160, 192],
    "inner_size": [128, 192, 224, 256, 320, 384],
    "num_layers": [1, 2, 3, 4],
    "n_layers": [1, 2, 3, 4],
    "num_heads": [1, 2, 4, 8],
    "n_heads": [1, 2, 4, 8],
    "hidden_dropout_prob": [0.10, 0.12, 0.13, 0.15, 0.18, 0.20],
    "dropout_ratio": [0.10, 0.12, 0.13, 0.15, 0.18, 0.20],
    "attn_dropout_prob": [0.06, 0.08, 0.10, 0.12, 0.15, 0.20],
    "dropout_prob": [0.10, 0.15, 0.20, 0.25, 0.30],
    "tau": [0.16, 0.18, 0.20, 0.22, 0.24],
    "lmd": [0.02, 0.03, 0.04, 0.05, 0.06],
    "lmd_sem": [0.0, 0.04, 0.05, 0.08, 0.10, 0.12],
    "bsarec_alpha": [0.35, 0.50, 0.55, 0.70],
    "bsarec_c": [2, 3, 5, 7],
    "attribute_hidden_size": [96, 128, 160, 192],
    "lambda_attr": [0.08, 0.09, 0.10, 0.12, 0.14, 0.15],
    "fusion_type": ["gate", "sum", "concat"],
    "num_experts": [2, 3, 4, 5, 6],
}

STAGE_WIDE_KEYS = [
    "weight_decay",
    "MAX_ITEM_LIST_LENGTH",
    "hidden_size",
    "embedding_size",
    "inner_size",
    "num_layers",
    "n_layers",
    "num_heads",
    "n_heads",
    "hidden_dropout_prob",
    "dropout_ratio",
    "attn_dropout_prob",
    "dropout_prob",
]

STAGE1_SUBSPACE_LABELS = ["core", "arch", "model"]

STAGE4_BAND_POLICY = {
    "light": {"max_evals": 6, "lr_points": 5, "other": 4, "random": 2, "stage5_top_k": 1, "parent_top_k": 2},
    "medium": {"max_evals": 8, "lr_points": 6, "other": 5, "random": 2, "stage5_top_k": 1, "parent_top_k": 3},
    "heavy": {"max_evals": 10, "lr_points": 7, "other": 6, "random": 3, "stage5_top_k": 1, "parent_top_k": 4},
}

DEFAULT_ADDTUNING_MAX_RUN_HOURS = float(DEFAULT_MAX_RUN_HOURS)
DEFAULT_ADDTUNING_OOM_RETRY_LIMIT = int(DEFAULT_OOM_RETRY_LIMIT)

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()
SUMMARY_WRITE_LOCK = threading.Lock()


def stage_summary_path(stage: str) -> Path:
    return LOG_ROOT / stage / "summary.csv"


def stage_manifest_path(stage: str) -> Path:
    return LOG_ROOT / stage / "manifest.json"


def log_path_for_row(stage: str, row: Dict[str, Any]) -> Path:
    dataset = sanitize_token(row.get("dataset", ""), upper=False)
    model = sanitize_token(row.get("model", ""), upper=False)
    family = sanitize_token(row.get("family", "baseline"), upper=False)
    job_id = sanitize_token(row.get("job_id", ""), upper=True)
    return LOG_ROOT / stage / dataset / model / family / f"{job_id}.log"


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def stage_plan(stage: str) -> Dict[str, int]:
    return dict(STAGE_TUNE_PLAN.get(stage, STAGE_TUNE_PLAN["stage3"]))


def eval_scale_for_dataset(dataset: str) -> float:
    return float(DATASET_EVAL_SCALE.get(normalize_dataset(dataset), 1.0))


def max_run_hours_for_dataset(dataset: str, default_hours: float) -> float:
    return float(DATASET_MAX_RUN_HOURS.get(normalize_dataset(dataset), default_hours))


def scaled_eval_budget(dataset: str, base_evals: int, *, minimum: int = 1) -> int:
    scaled = int(round(float(base_evals) * eval_scale_for_dataset(dataset)))
    return max(int(minimum), scaled)


def scaled_stage_eval_budget(stage: str, dataset: str, base_evals: int, *, minimum: int = 1) -> int:
    scaled = scaled_eval_budget(dataset, base_evals, minimum=minimum)
    boost = float(dict(DATASET_STAGE_BUDGET_BOOST.get(stage, {}) or {}).get(normalize_dataset(dataset), 1.0))
    boosted = int(round(float(scaled) * boost))
    return max(int(minimum), boosted)


def scaled_axis_budget(dataset: str, base_count: int, *, minimum: int = 2) -> int:
    scaled = int(round(float(base_count) * eval_scale_for_dataset(dataset)))
    return max(int(minimum), scaled)


def append_candidate_pool(key: str, original_values: Sequence[Any]) -> List[Any]:
    return list(dedupe_keep_order([*list(original_values), *list(APPENDIX_CANDIDATE_OVERRIDES.get(key, []))]))


def stage4_gap_plan(
    *,
    dataset: str,
    model: str,
    state: Dict[str, Any],
    current_score: float,
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    ds = normalize_dataset(dataset)
    best_all = safe_float(dict(summary_payload.get("dataset_best") or {}).get(ds, 0.0), 0.0)
    focus_rows = [row for row in list(summary_payload.get("focus_plan_rows") or []) if normalize_dataset(row.get("dataset", "")) == ds]
    focus_rows = sorted(focus_rows, key=lambda row: safe_float(row.get("test_metric_mean", 0.0), 0.0), reverse=True)
    focus_best = safe_float(focus_rows[0].get("test_metric_mean", 0.0), 0.0) if focus_rows else best_all
    rank = next((idx + 1 for idx, row in enumerate(focus_rows) if normalize_model(row.get("model", "")) == normalize_model(model)), len(focus_rows) + 1)
    ratio_all = current_score / best_all if best_all > 0.0 else 0.0
    ratio_focus = current_score / focus_best if focus_best > 0.0 else ratio_all
    severity = 0
    if str(state.get("selected_stage", "")) != "stage3":
        severity += 1
    if ratio_all < 0.97:
        severity += 1
    if ratio_all < 0.94:
        severity += 1
    if ratio_all < 0.90:
        severity += 1
    if ratio_focus < 0.96:
        severity += 1
    if rank >= 5:
        severity += 1
    if severity <= 1:
        return {"enabled": False, "band": "light", "reason": "close_to_frontier"}
    if severity == 2:
        band = "light"
    elif severity <= 4:
        band = "medium"
    else:
        band = "heavy"
    policy = STAGE4_BAND_POLICY[band]
    return {
        "enabled": True,
        "band": band,
        "severity": severity,
        "ratio_all": ratio_all,
        "ratio_focus": ratio_focus,
        "rank": rank,
        "max_evals": scaled_eval_budget(ds, int(policy["max_evals"]), minimum=2),
        "parent_top_k": int(policy["parent_top_k"]),
        "stage5_top_k": int(policy["stage5_top_k"]),
    }


def resolve_existing_result_path(track: str, row: Dict[str, Any]) -> Path:
    raw = str(row.get("result_path", "")).strip()
    if raw:
        direct = Path(raw)
        if direct.exists():
            return direct
        alt = EXISTING_TRACKS[track]["result_root"] / direct.name
        if alt.exists():
            return alt
    job_id = str(row.get("job_id", "")).strip()
    stage_token = str(row.get("stage", "")).strip().lower()
    fallback_prefix = re.compile(rf"^{re.escape(normalize_dataset(row.get('dataset', '')))}_.*_{re.escape(stage_token.replace('stage', 's'))}_", re.IGNORECASE)
    result_root = Path(EXISTING_TRACKS[track]["result_root"])
    candidates = [path for path in result_root.glob("*.json") if job_id and job_id.lower() in path.name.lower()]
    if not candidates and raw:
        stem = Path(raw).stem
        candidates = [path for path in result_root.glob(f"{stem}*.json")]
    if not candidates:
        candidates = [path for path in result_root.glob("*.json") if fallback_prefix.search(path.name)]
    return candidates[0] if candidates else Path(raw) if raw else Path()


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


def load_base_pair_specs() -> Dict[Tuple[str, str], Dict[str, Any]]:
    payload = load_manifest(CODE_DIR / "space_manifest.json")
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for spec in list(payload.get("pair_specs") or []):
        if str(spec.get("family", "")) != "baseline":
            continue
        dataset = normalize_dataset(spec.get("dataset", ""))
        model = normalize_model(spec.get("model", ""))
        if model not in FOCUS_MODELS:
            continue
        out[(dataset, model)] = dict(spec)
    return out


def top_unique_trials_by_test(payload: Dict[str, Any], *, top_k: int) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    fixed_search = dict(payload.get("fixed_search") or {})
    for idx, trial in enumerate(list(payload.get("trials") or []), start=1):
        if str((trial or {}).get("status", "")).strip().lower() != "ok":
            continue
        params = dict((trial or {}).get("params") or {})
        config = dict(fixed_search)
        config.update(params)
        ranked.append(
            {
                "trial_rank": idx,
                "valid_score": trial_valid_mean(trial or {}),
                "test_score": trial_test_mean(trial or {}),
                "config": config,
                "signature": config_signature(config),
            }
        )
    ranked.sort(key=lambda row: (float(row["test_score"]), float(row["valid_score"])), reverse=True)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in ranked:
        if row["signature"] in seen:
            continue
        seen.add(row["signature"])
        out.append(row)
        if len(out) >= int(top_k):
            break
    return out


def load_stage3_manifest_signatures(log_root: Path) -> Dict[str, str]:
    path = log_root / "stage3" / "manifest.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    return {str(row.get("job_id", "")): str(row.get("config_signature", "")) for row in list(payload.get("rows") or [])}


def group_key_for_stage3(job_id: str) -> str:
    return str(job_id).split("_SEED")[0]


def aggregate_existing_results(datasets: Sequence[str]) -> Dict[str, Any]:
    normalized_datasets = {normalize_dataset(item) for item in datasets}
    stage3_groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    fallback_best: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for track, meta in EXISTING_TRACKS.items():
        log_root = Path(meta["log_root"])
        manifest_signatures = load_stage3_manifest_signatures(log_root)

        for stage in ("stage1", "stage2"):
            summary_path = log_root / stage / "summary.csv"
            if not summary_path.exists():
                continue
            for row in read_csv_rows(summary_path):
                if str(row.get("status", "")).lower() != "ok":
                    continue
                dataset = normalize_dataset(row.get("dataset", ""))
                model = normalize_model(row.get("model", ""))
                if dataset not in normalized_datasets:
                    continue
                key = (dataset, model)
                result_path = resolve_existing_result_path(track, row)
                if not result_path.exists():
                    continue
                payload = load_result_payload(result_path)
                trials = top_unique_trials_by_test(payload, top_k=1)
                if not trials:
                    continue
                cand = {
                    "dataset": dataset,
                    "model": model,
                    "selected_stage": stage,
                    "test_metric_mean": safe_float(row.get("test_score"), 0.0),
                    "valid_metric_mean": safe_float(row.get("valid_score"), 0.0),
                    "seed_count": 1,
                    "track": track,
                    "source": str(row.get("job_id", "")),
                    "config": dict(trials[0]["config"]),
                }
                cur = fallback_best.get(key)
                if cur is None or (cand["test_metric_mean"], cand["valid_metric_mean"]) > (cur["test_metric_mean"], cur["valid_metric_mean"]):
                    fallback_best[key] = cand

        summary_path = log_root / "stage3" / "summary.csv"
        if not summary_path.exists():
            continue
        for row in read_csv_rows(summary_path):
            if str(row.get("status", "")).lower() != "ok":
                continue
            dataset = normalize_dataset(row.get("dataset", ""))
            model = normalize_model(row.get("model", ""))
            if dataset not in normalized_datasets:
                continue
            group_key = group_key_for_stage3(str(row.get("job_id", "")))
            bucket = stage3_groups.setdefault(
                (dataset, model, group_key),
                {
                    "dataset": dataset,
                    "model": model,
                    "selected_stage": "stage3",
                    "track": track,
                    "source": group_key,
                    "test_values": [],
                    "valid_values": [],
                    "seed_count": 0,
                    "config_signature": "",
                },
            )
            bucket["test_values"].append(safe_float(row.get("test_score"), 0.0))
            bucket["valid_values"].append(safe_float(row.get("valid_score"), 0.0))
            bucket["seed_count"] += 1
            signature = manifest_signatures.get(str(row.get("job_id", "")), "")
            if signature and not bucket["config_signature"]:
                bucket["config_signature"] = signature

    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (dataset, model, _group), row in stage3_groups.items():
        config = json.loads(row["config_signature"]) if row.get("config_signature") else {}
        cand = {
            "dataset": dataset,
            "model": model,
            "selected_stage": "stage3",
            "test_metric_mean": mean(row["test_values"]),
            "valid_metric_mean": mean(row["valid_values"]),
            "seed_count": int(row["seed_count"]),
            "track": row["track"],
            "source": row["source"],
            "config": config,
        }
        cur = selected.get((dataset, model))
        if cur is None or (cand["test_metric_mean"], cand["valid_metric_mean"]) > (cur["test_metric_mean"], cur["valid_metric_mean"]):
            selected[(dataset, model)] = cand

    for key, row in fallback_best.items():
        selected.setdefault(key, row)

    rows = sorted(selected.values(), key=lambda item: (item["dataset"], item["model"]))
    dataset_best: Dict[str, float] = defaultdict(float)
    for row in rows:
        dataset_best[row["dataset"]] = max(dataset_best[row["dataset"]], safe_float(row["test_metric_mean"], 0.0))
    return {
        "rows": rows,
        "dataset_best": dict(dataset_best),
    }


def assign_tier(row: Dict[str, Any], dataset_best: Dict[str, float]) -> str:
    best = safe_float(dataset_best.get(row["dataset"], 0.0), 0.0)
    score = safe_float(row.get("test_metric_mean", 0.0), 0.0)
    ratio = score / best if best > 0.0 else 0.0
    stage = str(row.get("selected_stage", ""))
    model = normalize_model(row.get("model", ""))
    dataset = normalize_dataset(row.get("dataset", ""))
    if stage != "stage3":
        base = "heavy" if ratio < 0.9 else "medium"
        if model in MEDIUM_FALLBACK_MODELS and dataset in MEDIUM_FALLBACK_DATASETS:
            return "medium" if base == "light" else base
        return base
    if ratio >= 0.97:
        tier = "light"
    elif ratio >= 0.92:
        tier = "medium"
    else:
        tier = "heavy"
    if model in MEDIUM_FALLBACK_MODELS and dataset in MEDIUM_FALLBACK_DATASETS and tier == "light":
        return "medium"
    return tier


def stage_key_order(stage: str) -> Tuple[int, str]:
    order = {"stage1": 1, "stage2": 2, "stage3": 3, "stage4": 4, "stage5": 5}
    return (order.get(stage, 99), stage)


def collect_runtime_profiles(datasets: Sequence[str]) -> Dict[str, Any]:
    normalized_datasets = {normalize_dataset(item) for item in datasets}
    per_pair: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    per_dataset: Dict[str, List[float]] = defaultdict(list)
    global_values: List[float] = []
    for track, meta in EXISTING_TRACKS.items():
        log_root = Path(meta["log_root"])
        for stage in ("stage1", "stage2", "stage3"):
            summary_path = log_root / stage / "summary.csv"
            if not summary_path.exists():
                continue
            for row in read_csv_rows(summary_path):
                if str(row.get("status", "")).lower() != "ok":
                    continue
                dataset = normalize_dataset(row.get("dataset", ""))
                model = normalize_model(row.get("model", ""))
                if dataset not in normalized_datasets or model not in FOCUS_MODELS:
                    continue
                result_path = resolve_existing_result_path(track, row)
                payload = load_result_payload(result_path) if result_path.exists() else {}
                elapsed = safe_float(row.get("elapsed_sec"), 0.0)
                if elapsed <= 0.0:
                    continue
                source_epochs = safe_int(payload.get("tune_epochs", 0), 0) or int(stage_tune_epochs(stage))
                if stage in {"stage1", "stage2"}:
                    trial_count = max(len(list(payload.get("trials") or [])), safe_int(payload.get("n_completed", 0), 0), 1)
                    sec_per_epoch = elapsed / max(trial_count * source_epochs, 1)
                else:
                    sec_per_epoch = elapsed / max(source_epochs, 1)
                per_pair[(dataset, model)].append(sec_per_epoch)
                per_dataset[dataset].append(sec_per_epoch)
                global_values.append(sec_per_epoch)
    return {
        "per_pair": {key: median(values) for key, values in per_pair.items()},
        "per_dataset": {key: median(values) for key, values in per_dataset.items()},
        "global": median(global_values),
    }


def runtime_profile_for_pair(runtime_profiles: Dict[str, Any], dataset: str, model: str) -> float:
    dataset_key = normalize_dataset(dataset)
    model_key = normalize_model(model)
    return float(
        runtime_profiles["per_pair"].get(
            (dataset_key, model_key),
            runtime_profiles["per_dataset"].get(dataset_key, runtime_profiles.get("global", 0.0)),
        )
    )


def runtime_profiles_to_jsonable(runtime_profiles: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "per_pair": {
            f"{dataset}::{model}": value
            for (dataset, model), value in dict(runtime_profiles.get("per_pair") or {}).items()
        },
        "per_dataset": dict(runtime_profiles.get("per_dataset") or {}),
        "global": float(runtime_profiles.get("global", 0.0) or 0.0),
    }


def estimate_row_runtime_sec(row: Dict[str, Any], runtime_profiles: Dict[str, Any]) -> float:
    sec_per_epoch = runtime_profile_for_pair(runtime_profiles, str(row.get("dataset", "")), str(row.get("model", "")))
    if sec_per_epoch <= 0.0:
        return 0.0
    evals = max(int(row.get("max_evals", 1) or 1), 1)
    epochs = max(int(row.get("tune_epochs", 100) or 100), 1)
    estimate = sec_per_epoch * evals * epochs
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    if max_run_hours > 0.0:
        estimate = min(estimate, max_run_hours * 3600.0)
    return estimate


def stage_tier_runtime_summary(stage_rows: Dict[str, List[Dict[str, Any]]], runtime_profiles: Dict[str, Any], gpu_count: int) -> Dict[str, Any]:
    stage_estimates: List[Dict[str, Any]] = []
    total_gpu_hours = 0.0
    total_wall_hours = 0.0
    for stage in sorted(stage_rows, key=stage_key_order):
        rows = list(stage_rows[stage])
        job_estimates = [estimate_row_runtime_sec(row, runtime_profiles) for row in rows]
        total_sec = sum(job_estimates)
        max_sec = max(job_estimates) if job_estimates else 0.0
        wall_sec = max(total_sec / max(gpu_count, 1), max_sec)
        total_gpu_hours += total_sec / 3600.0
        total_wall_hours += wall_sec / 3600.0
        per_tier_counts = {tier: sum(1 for row in rows if str(row.get("tier", "")) == tier) for tier in ("light", "medium", "heavy")}
        stage_estimates.append(
            {
                "stage": stage,
                "job_count": len(rows),
                "gpu_hours": total_sec / 3600.0,
                "wall_hours_8gpu": wall_sec / 3600.0,
                "max_single_job_hours": max_sec / 3600.0,
                "tier_counts": per_tier_counts,
            }
        )
    return {
        "stages": stage_estimates,
        "total_gpu_hours": total_gpu_hours,
        "total_wall_hours_8gpu": total_wall_hours,
        "gpu_count": gpu_count,
    }


def format_range(values: Sequence[Any]) -> str:
    normalized = [str(value) for value in values if str(value) != ""]
    if not normalized:
        return ""
    unique = list(dedupe_keep_order(normalized))
    if len(unique) == 1:
        return unique[0]
    numeric_values: List[float] = []
    for value in unique:
        try:
            numeric_values.append(float(value))
        except ValueError:
            numeric_values = []
            break
    if numeric_values:
        lower = min(numeric_values)
        upper = max(numeric_values)
        if all(value.is_integer() for value in numeric_values):
            return f"{int(lower)}-{int(upper)}"
        return f"{lower:g}-{upper:g}"
    return f"{unique[0]}-{unique[-1]}"


def build_stage_plan_rows(summary_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    rows = list(summary_payload.get("focus_plan_rows") or [])
    dataset_best = dict(summary_payload.get("dataset_best") or {})
    stage_rows: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ["stage1", "stage2", "stage3", "stage4", "stage5"]}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", ""))
        tier = str(row.get("tier", ""))
        policy = TIER_POLICY[tier]
        stage1_total = scaled_stage_eval_budget("stage1", dataset, int(policy["stage1_max_evals"]), minimum=8)
        for sub_budget in weighted_split_eval_budget(stage1_total, stage1_subspace_weights(), minimum=4):
            stage_rows["stage1"].append(
                {
                    "stage": "stage1",
                    "dataset": dataset,
                    "model": model,
                    "tier": tier,
                    "max_evals": int(sub_budget),
                    "tune_epochs": int(stage_plan("stage1")["tune_epochs"]),
                    "max_run_hours": max_run_hours_for_dataset(dataset, DEFAULT_ADDTUNING_MAX_RUN_HOURS),
                    "lr_points": scaled_axis_budget(dataset, int(policy["stage1_lr_points"]), minimum=3),
                    "other_points": scaled_axis_budget(dataset, int(policy["stage1_other"]), minimum=2),
                    "random_injections": int(policy.get("stage1_random", 0)),
                }
            )
        stage_rows["stage2"].append(
            {
                "stage": "stage2",
                "dataset": dataset,
                "model": model,
                "tier": tier,
                "max_evals": scaled_stage_eval_budget("stage2", dataset, int(policy["stage2_max_evals"]), minimum=6),
                "tune_epochs": int(stage_plan("stage2")["tune_epochs"]),
                "max_run_hours": max_run_hours_for_dataset(dataset, DEFAULT_ADDTUNING_MAX_RUN_HOURS),
                "lr_points": scaled_axis_budget(dataset, int(policy["stage2_lr_points"]), minimum=3),
                "other_points": scaled_axis_budget(dataset, int(policy["stage2_other"]), minimum=2),
                "random_injections": int(policy.get("stage2_random", 0)),
            }
        )
        stage3_top_k = stage3_top_k_for(dataset, tier)
        for _config_rank in range(1, stage3_top_k + 1):
            for _seed in range(1, stage3_seed_count_for_pair(normalize_dataset(dataset), tier) + 1):
                stage_rows["stage3"].append(
                    {
                        "stage": "stage3",
                        "dataset": dataset,
                        "model": model,
                        "tier": tier,
                        "max_evals": 1,
                        "tune_epochs": int(stage_plan("stage3")["tune_epochs"]),
                        "max_run_hours": max_run_hours_for_dataset(dataset, DEFAULT_ADDTUNING_MAX_RUN_HOURS),
                    }
                )
        stage4_plan = stage4_gap_plan(dataset=dataset, model=model, state=row, current_score=safe_float(row.get("test_metric_mean", 0.0), 0.0), summary_payload=summary_payload)
        if bool(stage4_plan.get("enabled", False)):
            stage_rows["stage4"].append(
                {
                    "stage": "stage4",
                    "dataset": dataset,
                    "model": model,
                    "tier": str(stage4_plan["band"]),
                    "max_evals": int(stage4_plan["max_evals"]),
                    "tune_epochs": int(stage_plan("stage4")["tune_epochs"]),
                    "max_run_hours": max_run_hours_for_dataset(dataset, DEFAULT_ADDTUNING_MAX_RUN_HOURS),
                    "lr_points": int(STAGE4_BAND_POLICY[str(stage4_plan['band'])]["lr_points"]),
                    "other_points": int(STAGE4_BAND_POLICY[str(stage4_plan['band'])]["other"]),
                    "random_injections": int(STAGE4_BAND_POLICY[str(stage4_plan['band'])]["random"]),
                }
            )
            for _config_rank in range(1, int(stage4_plan.get("stage5_top_k", 1)) + 1):
                for _seed in range(1, seed_count_for_pair(normalize_dataset(dataset), str(stage4_plan["band"])) + 1):
                    stage_rows["stage5"].append(
                        {
                            "stage": "stage5",
                            "dataset": dataset,
                            "model": model,
                            "tier": str(stage4_plan["band"]),
                            "max_evals": 1,
                            "tune_epochs": int(stage_plan("stage5")["tune_epochs"]),
                            "max_run_hours": max_run_hours_for_dataset(dataset, DEFAULT_ADDTUNING_MAX_RUN_HOURS),
                        }
                    )
    return stage_rows


def build_policy_rows(stage_rows: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stage in sorted(stage_rows, key=stage_key_order):
        plan = stage_plan(stage)
        for tier in ("light", "medium", "heavy"):
            tier_rows = [row for row in stage_rows[stage] if str(row.get("tier", "")) == tier]
            if not tier_rows and stage not in {"stage4", "stage5"}:
                pass
            policy = TIER_POLICY[tier]
            stage_key = stage if stage in {"stage1", "stage2"} else "stage4"
            lr_points = format_range([row.get("lr_points", "") for row in tier_rows]) if tier_rows else str(policy.get(f"{stage_key}_lr_points", ""))
            other = format_range([row.get("other_points", "") for row in tier_rows]) if tier_rows else str(policy.get(f"{stage_key}_other", ""))
            random_count = format_range([row.get("random_injections", 0) for row in tier_rows]) if tier_rows else str(policy.get(f"{stage_key}_random", 0))
            max_evals = format_range([row.get("max_evals", "") for row in tier_rows]) if tier_rows else ("1" if stage in {"stage3", "stage5"} else str(policy.get(f"{stage_key}_max_evals", "")))
            rows.append(
                {
                    "stage": stage,
                    "tier": tier,
                    "job_count": len(tier_rows),
                    "tune_epochs": int(plan["tune_epochs"]),
                    "tune_patience": int(plan["tune_patience"]),
                    "max_evals": max_evals,
                    "lr_points": lr_points,
                    "other_points": other,
                    "random_injections": random_count,
                    "seed_rule": "base+extra" if stage in {"stage3", "stage5"} else "-",
                }
            )
    return rows


def focus_plan_rows(summary_rows: Sequence[Dict[str, Any]], dataset_best: Dict[str, float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in summary_rows:
        model = normalize_model(row.get("model", ""))
        if model not in FOCUS_MODELS:
            continue
        best = safe_float(dataset_best.get(row["dataset"], 0.0), 0.0)
        score = safe_float(row.get("test_metric_mean", 0.0), 0.0)
        out.append(
            {
                **row,
                "tier": assign_tier(row, dataset_best),
                "gap_to_best": max(best - score, 0.0),
                "ratio_to_best": score / best if best > 0.0 else 0.0,
            }
        )
    return sorted(out, key=lambda item: (item["dataset"], item["model"]))


def write_existing_summary_files(aggregate: Dict[str, Any]) -> None:
    rows = list(aggregate["rows"])
    dataset_best = dict(aggregate["dataset_best"])
    focus_rows = focus_plan_rows(rows, dataset_best)
    planning_payload = {"generated_at": now_utc(), "rows": rows, "dataset_best": dataset_best, "focus_plan_rows": focus_rows}
    stage_rows = build_stage_plan_rows(planning_payload)
    runtime_profiles = collect_runtime_profiles(sorted(by_dataset_key for by_dataset_key in dataset_best))
    runtime_summary = stage_tier_runtime_summary(stage_rows, runtime_profiles, gpu_count=8)
    policy_rows = build_policy_rows(stage_rows)

    write_json(
        SUMMARY_JSON,
        {
            "generated_at": now_utc(),
            "rows": rows,
            "dataset_best": dataset_best,
            "focus_plan_rows": focus_rows,
            "stage_rows": stage_rows,
            "policy_rows": policy_rows,
            "runtime_profiles": runtime_profiles_to_jsonable(runtime_profiles),
            "runtime_summary_8gpu": runtime_summary,
        },
    )

    wide_rows: List[Dict[str, Any]] = []
    by_dataset = defaultdict(dict)
    for row in rows:
        by_dataset[row["dataset"]][row["model"]] = row
    for dataset in sorted(by_dataset):
        out = {"dataset": dataset}
        for model in MODEL_ORDER:
            cell = by_dataset[dataset].get(model)
            out[MODEL_LABELS.get(model, model)] = "" if cell is None else f"{safe_float(cell['test_metric_mean'], 0.0):.6f}"
        wide_rows.append(out)
    write_csv_rows(SUMMARY_CSV, wide_rows, ["dataset", *[MODEL_LABELS.get(model, model) for model in MODEL_ORDER]])

    lines = [
        "# Current Final Experiment Summary",
        "",
        "Stage3 seed means are used when available. Otherwise the table uses the best test score found across existing stage1/2 runs, including the current final_experiment_addtuning track.",
        "",
        "| Dataset | " + " | ".join(MODEL_LABELS.get(model, model) for model in MODEL_ORDER) + " |",
        "|---|" + "|".join(["---"] * len(MODEL_ORDER)) + "|",
    ]
    for dataset in sorted(by_dataset):
        parts = [dataset]
        for model in MODEL_ORDER:
            cell = by_dataset[dataset].get(model)
            if cell is None:
                parts.append("")
                continue
            marker = "" if cell["selected_stage"] == "stage3" else "*"
            parts.append(f"{safe_float(cell['test_metric_mean'], 0.0):.6f}{marker}")
        lines.append("| " + " | ".join(parts) + " |")
    lines.extend(
        [
            "",
            "`*` means stage3 was unavailable and the cell uses the best stage1/2 test score.",
            "",
            "## Focused Addtuning Tiers",
            "",
            "| Dataset | Model | Current | Best | Ratio | Tier | Source |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in focus_rows:
        lines.append(
            "| {dataset} | {model} | {score:.6f} | {best:.6f} | {ratio:.3f} | {tier} | {stage}/{source} |".format(
                dataset=row["dataset"],
                model=MODEL_LABELS.get(row["model"], row["model"]),
                score=safe_float(row["test_metric_mean"], 0.0),
                best=safe_float(dataset_best.get(row["dataset"], 0.0), 0.0),
                ratio=safe_float(row["ratio_to_best"], 0.0),
                tier=row["tier"],
                stage=row["selected_stage"],
                source=row["source"],
            )
        )
    lines.extend(
        [
            "",
            "## Stage And Tier Policy",
            "",
            "Epoch/patience increases by stage and final confirmation uses 100/10. Max-evals are front-loaded in stage1 and reduced in later stages.",
            "Stage1 now spends real budget on a few targeted local sweeps around the incumbent. Stage3 is a cheap 1-seed reranking gate over only the top 1--2 configs, and multi-seed confirmation is deferred to the final confirmation stage. Stage4 remains gap-driven, but now gets a meaningfully larger refinement budget when a model is still behind the dataset frontier.",
            "Candidate counts below are per-axis option counts for local search refinement. They are not cartesian combo jobs; actual trials are capped by `max_evals` and the per-job time limit.",
            "",
            "| Stage | Tier | Jobs | Epochs | Patience | Max Evals | LR Points | Other Points | Random Adds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in policy_rows:
        lines.append(
            "| {stage} | {tier} | {jobs} | {epochs} | {patience} | {max_evals} | {lr_points} | {other_points} | {random_adds} |".format(
                stage=row["stage"],
                tier=row["tier"],
                jobs=row["job_count"],
                epochs=row["tune_epochs"],
                patience=row["tune_patience"],
                max_evals=row["max_evals"],
                lr_points=row["lr_points"] if row["lr_points"] != "" else "-",
                other_points=row["other_points"] if row["other_points"] != "" else "-",
                random_adds=row["random_injections"],
            )
        )
    lines.extend(
        [
            "",
            "## Estimated Runtime On 8 GPUs",
            "",
            "Wall time is estimated as `max(total GPU-hours / 8, longest single job)` per stage, using historical per-epoch runtimes from existing stage1/2/3 runs of the same dataset/model when available.",
            "All addtuning jobs inherit the existing safeguards: OOM retries halve train/eval batch size, the time budget stops launching new trials after the current trial finishes and runs final evaluation, and stage workers pull from one global GPU queue rather than dataset-partitioned queues.",
            "",
            "| Stage | Jobs | Light | Medium | Heavy | GPU-hours | Est. Wall-hours | Longest Job |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in runtime_summary["stages"]:
        lines.append(
            "| {stage} | {jobs} | {light} | {medium} | {heavy} | {gpu_hours:.2f} | {wall_hours:.2f} | {max_job:.2f} |".format(
                stage=row["stage"],
                jobs=row["job_count"],
                light=row["tier_counts"].get("light", 0),
                medium=row["tier_counts"].get("medium", 0),
                heavy=row["tier_counts"].get("heavy", 0),
                gpu_hours=row["gpu_hours"],
                wall_hours=row["wall_hours_8gpu"],
                max_job=row["max_single_job_hours"],
            )
        )
    lines.extend(
        [
            "",
            "Total estimated GPU-hours: {gpu_hours:.2f}".format(gpu_hours=runtime_summary["total_gpu_hours"]),
            "Total estimated wall-hours on 8 GPUs: {wall_hours:.2f}".format(wall_hours=runtime_summary["total_wall_hours_8gpu"]),
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inject_candidates(
    base_values: Sequence[Any],
    original_values: Sequence[Any],
    *,
    random_count: int,
    rng: random.Random,
    fallback_key: str,
) -> List[Any]:
    result = list(dedupe_keep_order(list(base_values)))
    selected_tokens = {json.dumps(item, ensure_ascii=True, sort_keys=True) for item in result}
    original = list(dedupe_keep_order(list(original_values)))
    pool = [item for item in original if json.dumps(item, ensure_ascii=True, sort_keys=True) not in selected_tokens]
    rng.shuffle(pool)
    for item in pool[: max(0, int(random_count))]:
        token = json.dumps(item, ensure_ascii=True, sort_keys=True)
        if token not in selected_tokens:
            result.append(item)
            selected_tokens.add(token)
    if fallback_key in NUMERIC_FALLBACKS:
        extra = [item for item in NUMERIC_FALLBACKS[fallback_key] if json.dumps(item, ensure_ascii=True, sort_keys=True) not in selected_tokens]
        rng.shuffle(extra)
        for item in extra[: max(0, int(random_count) - len(pool[: max(0, int(random_count))]))]:
            token = json.dumps(item, ensure_ascii=True, sort_keys=True)
            if token not in selected_tokens:
                result.append(item)
                selected_tokens.add(token)
    if fallback_key in CHOICE_FALLBACKS:
        for item in CHOICE_FALLBACKS[fallback_key]:
            token = json.dumps(item, ensure_ascii=True, sort_keys=True)
            if token not in selected_tokens:
                result.append(item)
                selected_tokens.add(token)
    return result


def split_eval_budget(total: int, buckets: int, *, minimum: int = 2) -> List[int]:
    total_value = max(int(total), 1)
    bucket_count = max(int(buckets), 1)
    if bucket_count == 1:
        return [total_value]
    base = max(int(minimum), 1)
    if total_value <= bucket_count * base:
        values = [base] * bucket_count
        overflow = sum(values) - total_value
        cursor = bucket_count - 1
        while overflow > 0 and cursor >= 0:
            if values[cursor] > 1:
                values[cursor] -= 1
                overflow -= 1
            else:
                cursor -= 1
        return values
    values = [base] * bucket_count
    remainder = total_value - (bucket_count * base)
    cursor = 0
    while remainder > 0:
        values[cursor % bucket_count] += 1
        remainder -= 1
        cursor += 1
    return values


def weighted_split_eval_budget(total: int, weights: Sequence[float], *, minimum: int = 2) -> List[int]:
    if not weights:
        return [max(int(total), 1)]
    total_value = max(int(total), 1)
    base = max(int(minimum), 1)
    bucket_count = len(weights)
    normalized_weights = [max(float(weight), 0.0) for weight in weights]
    weight_sum = sum(normalized_weights) or float(bucket_count)
    shares = [weight / weight_sum for weight in normalized_weights]
    values = [base] * bucket_count
    remaining = max(total_value - (base * bucket_count), 0)
    fractional: List[Tuple[int, float]] = []
    for idx, share in enumerate(shares):
        raw = remaining * share
        extra = int(raw)
        values[idx] += extra
        fractional.append((idx, raw - extra))
    leftover = total_value - sum(values)
    for idx, _fraction in sorted(fractional, key=lambda item: item[1], reverse=True):
        if leftover <= 0:
            break
        values[idx] += 1
        leftover -= 1
    while leftover > 0:
        for idx in range(bucket_count):
            if leftover <= 0:
                break
            values[idx] += 1
            leftover -= 1
    return values


def search_space_combo_count(search_space: Dict[str, Sequence[Any]]) -> int:
    count = 1
    for values in dict(search_space or {}).values():
        if not isinstance(values, (list, tuple, set)):
            value_count = 1
        else:
            value_count = max(len(list(values)), 1)
        count *= value_count
    return int(count)


def trim_values_for_anchor(values: Sequence[Any], anchor_value: Any, *, max_items: int) -> List[Any]:
    ordered = list(dedupe_keep_order(list(values)))
    if len(ordered) <= max(int(max_items), 1):
        return ordered
    if anchor_value is None:
        return ordered[: max(int(max_items), 1)]
    numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ordered)
    if numeric:
        ranked = sorted(
            ordered,
            key=lambda value: (
                abs(safe_float(value, 0.0) - safe_float(anchor_value, 0.0)),
                safe_float(value, 0.0),
            ),
        )
    else:
        anchor_token = json.dumps(anchor_value, ensure_ascii=True, sort_keys=True)
        ranked = sorted(
            ordered,
            key=lambda value: (json.dumps(value, ensure_ascii=True, sort_keys=True) != anchor_token),
        )
    return ranked[: max(int(max_items), 1)]


def subspace_axis_cap(label: str, key: str) -> int:
    if key == "learning_rate":
        return 5
    if label in {"regularization", "dropout", "model"}:
        return 3
    return 2


def stage1_subspace_specs(model: str) -> List[Tuple[str, List[str]]]:
    model_specific = [
        key
        for key in MODEL_PRIORITY_PARAMS.get(model, [])
        if key not in {"learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "weight_decay", "MAX_ITEM_LIST_LENGTH", "dropout_ratio", "dropout_prob"}
    ]
    return [
        ("regularization", ["learning_rate", "weight_decay", "MAX_ITEM_LIST_LENGTH"]),
        ("dropout", ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob", "dropout_ratio", "dropout_prob"]),
        ("architecture", ["learning_rate", "hidden_size", "embedding_size", "inner_size", "attribute_hidden_size"]),
        ("structure", ["learning_rate", "num_layers", "n_layers", "num_heads", "n_heads", *model_specific[:2]]),
    ]


def stage1_subspace_weights() -> List[float]:
    return [0.32, 0.24, 0.24, 0.20]


def stage3_top_k_for(dataset: str, tier: str) -> int:
    top_k = int(TIER_POLICY[tier]["stage3_top_k"])
    if normalize_dataset(dataset) in {"movielens1m", "lastfm0.03"}:
        top_k = min(top_k, 2)
    return max(top_k, 1)


def stage3_seed_count_for_pair(_dataset: str, _tier: str) -> int:
    return 1


def slice_search_space(
    search_space: Dict[str, Sequence[Any]],
    fixed_context: Dict[str, Any],
    *,
    label: str,
    active_keys: Sequence[str],
    anchor_config: Dict[str, Any],
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    selected_keys = set(active_keys)
    sliced_space: Dict[str, List[Any]] = {}
    next_fixed = dict(fixed_context or {})
    for key, values in dict(search_space or {}).items():
        ordered = list(values) if isinstance(values, (list, tuple, set)) else [values]
        if key in selected_keys:
            sliced_space[key] = trim_values_for_anchor(
                ordered,
                dict(anchor_config or {}).get(key),
                max_items=subspace_axis_cap(label, key),
            )
            continue
        if key in next_fixed:
            continue
        anchor_value = dict(anchor_config or {}).get(key)
        if anchor_value is None and ordered:
            anchor_value = ordered[0]
        if anchor_value is not None:
            next_fixed[key] = anchor_value
    return sliced_space, next_fixed


def stage_payload_items_for_pair(
    stage_payloads: Dict[Tuple[str, str, str], Dict[str, Any]],
    dataset: str,
    model: str,
    job_id_prefix: str,
) -> List[Dict[str, Any]]:
    ds = normalize_dataset(dataset)
    mdl = normalize_model(model)
    out: List[Dict[str, Any]] = []
    for (row_dataset, row_model, job_id), payload in stage_payloads.items():
        if row_dataset != ds or row_model != mdl:
            continue
        if not str(job_id).startswith(job_id_prefix):
            continue
        out.append(payload)
    return out


def top_trials_across_payloads(payload_items: Sequence[Dict[str, Any]], *, top_k_per_payload: int, top_k: int) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for item in payload_items:
        for trial in top_unique_trials_by_test(dict(item.get("payload") or {}), top_k=max(int(top_k_per_payload), 1)):
            ranked.append(trial)
    ranked.sort(key=lambda row: (float(row.get("test_score", 0.0)), float(row.get("valid_score", 0.0))), reverse=True)
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for row in ranked:
        signature = str(row.get("signature", ""))
        if signature in seen:
            continue
        seen.add(signature)
        out.append(row)
        if len(out) >= int(top_k):
            break
    return out


def refine_search_space(
    spec: Dict[str, Any],
    parent_configs: Sequence[Dict[str, Any]],
    *,
    model: str,
    tier: str,
    stage: str,
    seed: int,
) -> Dict[str, List[Any]]:
    policy = STAGE4_BAND_POLICY[tier] if stage == "stage4" and tier in STAGE4_BAND_POLICY else TIER_POLICY[tier]
    original_space = dict(spec.get("search_space") or {})
    dataset = normalize_dataset(spec.get("dataset", ""))
    if stage in {"stage1", "stage2"}:
        lr_points = scaled_axis_budget(dataset, int(policy[f"{stage}_lr_points"]), minimum=3)
        max_other = scaled_axis_budget(dataset, int(policy[f"{stage}_other"]), minimum=2)
        random_count = int(policy.get(f"{stage}_random", 0))
    else:
        lr_points = scaled_axis_budget(dataset, int(policy.get("lr_points", 6)), minimum=4)
        max_other = scaled_axis_budget(dataset, int(policy.get("other", 4)), minimum=3)
        random_count = int(policy.get("random", 2))
    refined = narrow_space_from_configs(original_space, list(parent_configs), lr_points=lr_points, max_other=max_other)
    rng = random.Random(seed)
    keys = list(dedupe_keep_order([*MODEL_PRIORITY_PARAMS.get(model, ["learning_rate", "hidden_dropout_prob", "attn_dropout_prob"]), *(STAGE_WIDE_KEYS if stage in {"stage1", "stage4"} else [])]))
    for key in keys:
        original_values = append_candidate_pool(key, list(original_space.get(key) or []))
        if not original_values and stage in {"stage1", "stage4"}:
            anchor = None
            for cfg in parent_configs:
                if key in dict(cfg or {}):
                    anchor = dict(cfg or {}).get(key)
                    break
            if anchor is None:
                anchor = dict(spec.get("fixed_context") or {}).get(key)
            if anchor is not None and key in APPENDIX_CANDIDATE_OVERRIDES:
                original_values = append_candidate_pool(key, [anchor])
        current_values = list(refined.get(key) or [])
        if not current_values and original_values:
            current_values = list(original_values[: max(2, min(len(original_values), max_other))])
        if stage == "stage4" and len(original_values) > len(current_values):
            random_count = max(random_count, 2)
        next_values = inject_candidates(
            current_values,
            original_values,
            random_count=random_count,
            rng=rng,
            fallback_key=key,
        )
        if next_values:
            refined[key] = next_values
    for cfg in parent_configs:
        for key, value in dict(cfg or {}).items():
            if key not in refined:
                continue
            token = json.dumps(value, ensure_ascii=True, sort_keys=True)
            seen = {json.dumps(item, ensure_ascii=True, sort_keys=True) for item in refined[key]}
            if token not in seen:
                refined[key].append(value)
    return refined


def validate_dry_run_row(row: Dict[str, Any], gpu_id: str, search_algo: str) -> Tuple[bool, List[str], List[str]]:
    cmd = build_command(row, gpu_id, search_algo)
    errors: List[str] = []
    warnings: List[str] = []
    python_path = Path(cmd[0])
    if not python_path.exists():
        errors.append(f"python_not_found: {python_path}")
    if not HYPEROPT_TUNE_SCRIPT.exists():
        errors.append(f"hyperopt_tune_missing: {HYPEROPT_TUNE_SCRIPT}")
    combo_count = search_space_combo_count(dict(row.get("search_space") or {}))
    max_evals = max(int(row.get("max_evals", 1) or 1), 1)
    axis_count = len(dict(row.get("search_space") or {}))
    if combo_count > max_evals * 200:
        warnings.append(f"search_space_large: combos={combo_count} max_evals={max_evals} axes={axis_count}")
    return (not errors), errors, warnings


def write_dry_run_log(row: Dict[str, Any], *, stage: str, gpu_id: str, search_algo: str) -> bool:
    log_path = log_path_for_row(stage, row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(row, gpu_id, search_algo)
    ok, errors, warnings = validate_dry_run_row(row, gpu_id, search_algo)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# dry_run=true\n")
        fh.write(f"# job_id={row['job_id']} dataset={row['dataset']} model={row['model']} stage={row['stage']}\n")
        fh.write(f"# cwd={EXPERIMENTS_DIR}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n")
        fh.write(f"# search_axes={len(dict(row.get('search_space') or {}))} combos={search_space_combo_count(dict(row.get('search_space') or {}))} max_evals={int(row.get('max_evals', 0) or 0)}\n")
        for warning in warnings:
            fh.write(f"WARNING: {warning}\n")
        for error in errors:
            fh.write(f"ERROR: {error}\n")
        fh.write(f"# dry_run_validation={'ok' if ok else 'fail'}\n")
    return ok


def run_dry_run_validation(rows: List[Dict[str, Any]], *, stage: str, gpus: List[str], search_algo: str) -> int:
    if not rows:
        return 0
    failures = 0
    gpu_list = list(gpus) or ["0"]
    for idx, row in enumerate(rows):
        gpu_id = str(gpu_list[idx % len(gpu_list)])
        if not write_dry_run_log(row, stage=stage, gpu_id=gpu_id, search_algo=search_algo):
            failures += 1
    return 1 if failures else 0


def build_command(row: Dict[str, Any], gpu_id: str, search_algo: str) -> List[str]:
    search, types = build_common_search_entries(row.get("search_space") or {}, row.get("fixed_context") or {})
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    oom_retry_limit = int(row.get("oom_retry_limit", 0) or 0)
    cmd = [
        python_bin(),
        str(HYPEROPT_TUNE_SCRIPT),
        "--config-name",
        dataset_config_name(str(row["dataset"])),
        "--search-algo",
        str(search_algo),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        str(int(row.get("tune_epochs", 100))),
        "--tune-patience",
        str(int(row.get("tune_patience", 10))),
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
        "family": row.get("family", "baseline"),
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
        f"stage={row.get('stage', '')} dataset={row.get('dataset', '')} "
        f"model={row.get('model', '')} job={row.get('job_id', '')} "
        f"tier={row.get('tier', '')} max_evals={row.get('max_evals', 0)}"
    )
    max_run_hours = float(row.get("max_run_hours", 0.0) or 0.0)
    if max_run_hours > 0.0:
        base += f" max_run_hours={max_run_hours:.3f}"
    oom_retry_limit = int(row.get("oom_retry_limit", 0) or 0)
    if oom_retry_limit > 0:
        base += f" oom_retries={oom_retry_limit}"
    return base


def with_runtime_safeguards(row: Dict[str, Any], *, default_max_run_hours: float, oom_retry_limit: int) -> Dict[str, Any]:
    enriched = dict(row)
    enriched["max_run_hours"] = float(max_run_hours_for_dataset(str(row.get("dataset", "")), default_max_run_hours))
    enriched["oom_retry_limit"] = int(oom_retry_limit)
    return enriched


def run_one_job(row: Dict[str, Any], gpu_id: str, search_algo: str) -> Dict[str, Any]:
    log_path = log_path_for_row(str(row["stage"]), row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_command(row, gpu_id, search_algo)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    rc = 1
    print(f"[addtuning][gpu={gpu_id}] START {describe_job(row)} log={log_path}", flush=True)
    proc: subprocess.Popen[Any] | None = None
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# job_id={row['job_id']} dataset={row['dataset']} model={row['model']} stage={row['stage']}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(EXPERIMENTS_DIR),
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
    payload = load_result_payload(result_path_obj) if result_path_obj is not None else {}
    has_success = result_has_successful_trials(payload)
    status = "ok" if (rc == 0 and has_success and (normal_end or result_path_obj is not None)) else "fail"
    error = "" if status == "ok" else f"rc={rc} tail={extract_error_tail(log_path)}"
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
        f"[addtuning][gpu={gpu_id}] END {describe_job(row)} status={status} "
        f"valid={safe_float(summary.get('valid_score', 0.0), 0.0):.6f} "
        f"test={safe_float(summary.get('test_score', 0.0), 0.0):.6f} elapsed={elapsed:.1f}s",
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
    payload = load_result_payload(result_path_obj)
    if not result_has_successful_trials(payload):
        return None
    return build_summary_row(
        row,
        gpu_id="resume",
        status="ok",
        result_path=str(result_path_obj),
        log_path=log_path,
        elapsed_sec=0.0,
        error="",
    )


def run_jobs(rows: List[Dict[str, Any]], *, stage: str, gpus: List[str], search_algo: str, resume_from_logs: bool, dry_run: bool) -> int:
    summary_path = stage_summary_path(stage)
    install_signal_handlers()
    if dry_run:
        return run_dry_run_validation(rows, stage=stage, gpus=gpus, search_algo=search_algo)
    pending: Queue[Dict[str, Any]] = Queue()
    for row in rows:
        if resume_from_logs:
            resumed = resumed_summary_row(row)
            if resumed is not None:
                with SUMMARY_WRITE_LOCK:
                    upsert_csv_row(summary_path, SUMMARY_FIELDS, resumed, key_fields=("job_id",))
                continue
        pending.put(row)
    if pending.empty():
        return 0
    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))
    error_count = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal error_count
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo)
                with SUMMARY_WRITE_LOCK:
                    upsert_csv_row(summary_path, SUMMARY_FIELDS, summary, key_fields=("job_id",))
                if str(summary.get("status", "")) != "ok":
                    with lock:
                        error_count += 1
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return 1 if error_count or STOP_EVENT.is_set() else 0


def load_stage_payloads(stage: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in read_csv_rows(stage_summary_path(stage)):
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        result_path = Path(str(row.get("result_path", "")).strip())
        if not result_path.exists():
            continue
        payload = load_result_payload(result_path)
        out[(normalize_dataset(row.get("dataset", "")), normalize_model(row.get("model", "")), str(row.get("job_id", "")))] = {
            "summary": row,
            "payload": payload,
            "result_path": result_path,
        }
    return out


def load_stage_manifest_rows(stage: str) -> Dict[str, Dict[str, Any]]:
    path = stage_manifest_path(stage)
    if not path.exists():
        return {}
    payload = read_json(path)
    return {str(row.get("job_id", "")): dict(row) for row in list(payload.get("rows") or [])}


def seed_count_for_pair(dataset: str, tier: str) -> int:
    base = int(ADDTUNING_SEED_COUNTS.get(dataset, STAGE3_SEED_COUNTS.get(dataset, 2)))
    return base + int(TIER_POLICY[tier].get("stage3_extra_seeds", 0))


def make_job_id(stage: str, dataset: str, model: str, suffix: str = "") -> str:
    base = f"ADD_{sanitize_token(stage, upper=True)}_{sanitize_token(dataset, upper=True)}_{sanitize_token(model, upper=True)}"
    return base if not suffix else f"{base}_{suffix}"


def write_stage_manifest(stage: str, rows: List[Dict[str, Any]]) -> Path:
    path = stage_manifest_path(stage)
    payload = {
        "generated_at": now_utc(),
        "track": TRACK,
        "stage": stage,
        "run_count": len(rows),
        "rows": rows,
    }
    write_json(path, payload)
    return path


def current_focus_lookup(summary_payload: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (normalize_dataset(row.get("dataset", "")), normalize_model(row.get("model", ""))): dict(row)
        for row in list(summary_payload.get("focus_plan_rows") or [])
    }


def build_stage1_rows(summary_payload: Dict[str, Any], specs: Dict[Tuple[str, str], Dict[str, Any]], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = current_focus_lookup(summary_payload)
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        ds = normalize_dataset(dataset)
        for model in FOCUS_MODELS:
            spec = specs.get((ds, model))
            state = lookup.get((ds, model))
            if spec is None or state is None:
                continue
            tier = str(state["tier"])
            policy = TIER_POLICY[tier]
            base_seed = 910000 + cursor + 1
            anchor_config = dict(state.get("config") or {})
            refined = refine_search_space(spec, [anchor_config], model=model, tier=tier, stage="stage1", seed=base_seed)
            subspaces = stage1_subspace_specs(model)
            budgets = weighted_split_eval_budget(scaled_stage_eval_budget("stage1", spec["dataset"], int(policy["stage1_max_evals"]), minimum=8), stage1_subspace_weights(), minimum=4)
            for sub_idx, ((label, active_keys), sub_budget) in enumerate(zip(subspaces, budgets), start=1):
                sliced_search, sliced_fixed = slice_search_space(refined, dict(spec.get("fixed_context") or {}), label=label, active_keys=active_keys, anchor_config=anchor_config)
                cursor += 1
                rows.append(with_runtime_safeguards({
                        "stage": "stage1",
                        "run_axis": f"add_stage1_{label}",
                        "dataset": spec["dataset"],
                        "model": model,
                        "family": "baseline",
                        "job_id": make_job_id("stage1", spec["dataset"], model, suffix=label.upper()),
                        "run_phase": make_job_id("stage1", spec["dataset"], model, suffix=label.upper()),
                        "parent_job_id": str(state.get("source", "")),
                        "runtime_seed": 910000 + cursor,
                        "max_evals": int(sub_budget),
                        "search_space": sliced_search,
                        "fixed_context": sliced_fixed,
                        "tier": tier,
                        "subspace_label": label,
                        "tune_epochs": stage_plan("stage1")["tune_epochs"],
                        "tune_patience": stage_plan("stage1")["tune_patience"],
                    }, default_max_run_hours=DEFAULT_ADDTUNING_MAX_RUN_HOURS, oom_retry_limit=DEFAULT_ADDTUNING_OOM_RETRY_LIMIT))
    return rows


def build_stage2_rows(summary_payload: Dict[str, Any], specs: Dict[Tuple[str, str], Dict[str, Any]], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = current_focus_lookup(summary_payload)
    stage1_payloads = load_stage_payloads("stage1")
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        ds = normalize_dataset(dataset)
        for model in FOCUS_MODELS:
            spec = specs.get((ds, model))
            state = lookup.get((ds, model))
            stage1_items = stage_payload_items_for_pair(stage1_payloads, spec["dataset"], model, make_job_id("stage1", spec["dataset"], model)) if spec else []
            if spec is None or state is None or not stage1_items:
                continue
            tier = str(state["tier"])
            policy = TIER_POLICY[tier]
            parents = top_trials_across_payloads(stage1_items, top_k_per_payload=2, top_k=int(policy["stage2_top_k"]))
            if not parents:
                continue
            cursor += 1
            rows.append(with_runtime_safeguards({
                    "stage": "stage2",
                    "run_axis": "add_stage2_promote",
                    "dataset": spec["dataset"],
                    "model": model,
                    "family": "baseline",
                    "job_id": make_job_id("stage2", spec["dataset"], model),
                    "run_phase": make_job_id("stage2", spec["dataset"], model),
                    "parent_job_id": make_job_id("stage1", spec["dataset"], model),
                    "runtime_seed": 920000 + cursor,
                    "max_evals": scaled_stage_eval_budget("stage2", spec["dataset"], int(policy["stage2_max_evals"]), minimum=6),
                    "search_space": refine_search_space(spec, [item["config"] for item in parents], model=model, tier=tier, stage="stage2", seed=920000 + cursor),
                    "fixed_context": dict(spec.get("fixed_context") or {}),
                    "tier": tier,
                    "tune_epochs": stage_plan("stage2")["tune_epochs"],
                    "tune_patience": stage_plan("stage2")["tune_patience"],
                }, default_max_run_hours=DEFAULT_ADDTUNING_MAX_RUN_HOURS, oom_retry_limit=DEFAULT_ADDTUNING_OOM_RETRY_LIMIT))
    return rows


def build_stage3_rows(summary_payload: Dict[str, Any], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = current_focus_lookup(summary_payload)
    stage1_payloads = load_stage_payloads("stage1")
    stage2_payloads = load_stage_payloads("stage2")
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        ds = normalize_dataset(dataset)
        for model in FOCUS_MODELS:
            state = lookup.get((ds, model))
            if state is None:
                continue
            tier = str(state["tier"])
            stage2_item = stage2_payloads.get((ds, model, make_job_id("stage2", dataset, model)))
            stage3_top_k = stage3_top_k_for(dataset, tier)
            trials = top_unique_trials_by_test(stage2_item["payload"], top_k=stage3_top_k) if stage2_item else []
            if not trials:
                stage1_items = stage_payload_items_for_pair(stage1_payloads, dataset, model, make_job_id("stage1", dataset, model))
                trials = top_trials_across_payloads(stage1_items, top_k_per_payload=2, top_k=min(2, stage3_top_k)) if stage1_items else []
            seed_count = stage3_seed_count_for_pair(ds, tier)
            for config_rank, trial in enumerate(trials, start=1):
                for seed_id in range(1, seed_count + 1):
                    cursor += 1
                    config = dict(trial["config"])
                    rows.append(with_runtime_safeguards({
                            "stage": "stage3",
                            "run_axis": "add_stage3_seed_confirm",
                            "dataset": dataset,
                            "model": model,
                            "family": "baseline",
                            "job_id": make_job_id("stage3", dataset, model, suffix=f"C{config_rank}_SEED{seed_id}"),
                            "run_phase": make_job_id("stage3", dataset, model, suffix=f"C{config_rank}_SEED{seed_id}"),
                            "parent_job_id": make_job_id("stage2", dataset, model),
                            "runtime_seed": 930000 + cursor,
                            "max_evals": 1,
                            "search_space": config,
                            "fixed_context": {},
                            "tier": tier,
                            "config_rank": config_rank,
                            "seed_id": seed_id,
                            "config_signature": config_signature(config),
                            "tune_epochs": stage_plan("stage3")["tune_epochs"],
                            "tune_patience": stage_plan("stage3")["tune_patience"],
                        }, default_max_run_hours=DEFAULT_ADDTUNING_MAX_RUN_HOURS, oom_retry_limit=DEFAULT_ADDTUNING_OOM_RETRY_LIMIT))
    return rows


def aggregate_stage3_selected() -> Dict[Tuple[str, str], Dict[str, Any]]:
    manifest_rows = load_stage_manifest_rows("stage3")
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in read_csv_rows(stage_summary_path("stage3")):
        if str(row.get("status", "")).lower() != "ok":
            continue
        dataset = normalize_dataset(row.get("dataset", ""))
        model = normalize_model(row.get("model", ""))
        job_id = str(row.get("job_id", ""))
        config_id = group_key_for_stage3(job_id)
        bucket = grouped.setdefault(
            (dataset, model, config_id),
            {"test_values": [], "valid_values": [], "seed_count": 0, "config_signature": "", "config_id": config_id},
        )
        bucket["test_values"].append(safe_float(row.get("test_score"), 0.0))
        bucket["valid_values"].append(safe_float(row.get("valid_score"), 0.0))
        bucket["seed_count"] += 1
        signature = str(manifest_rows.get(job_id, {}).get("config_signature", ""))
        if signature and not bucket["config_signature"]:
            bucket["config_signature"] = signature
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (dataset, model, _config_id), bucket in grouped.items():
        cand = {
            "dataset": dataset,
            "model": model,
            "config_id": bucket["config_id"],
            "mean_test": mean(bucket["test_values"]),
            "mean_valid": mean(bucket["valid_values"]),
            "seed_count": int(bucket["seed_count"]),
            "config": json.loads(bucket["config_signature"]) if bucket.get("config_signature") else {},
        }
        cur = selected.get((dataset, model))
        if cur is None or (cand["mean_test"], cand["mean_valid"]) > (cur["mean_test"], cur["mean_valid"]):
            selected[(dataset, model)] = cand
    return selected


def build_stage4_rows(summary_payload: Dict[str, Any], specs: Dict[Tuple[str, str], Dict[str, Any]], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = current_focus_lookup(summary_payload)
    dataset_best = dict(summary_payload.get("dataset_best") or {})
    stage2_payloads = load_stage_payloads("stage2")
    stage3_selected = aggregate_stage3_selected()
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        ds = normalize_dataset(dataset)
        for model in FOCUS_MODELS:
            spec = specs.get((ds, model))
            state = lookup.get((ds, model))
            current = stage3_selected.get((ds, model))
            if spec is None or state is None:
                continue
            current_score = safe_float((current or {}).get("mean_test", state.get("test_metric_mean", 0.0)), 0.0)
            stage4_plan = stage4_gap_plan(dataset=dataset, model=model, state=state, current_score=current_score, summary_payload=summary_payload)
            if not bool(stage4_plan.get("enabled", False)):
                continue
            band = str(stage4_plan["band"])
            stage2_item = stage2_payloads.get((ds, model, make_job_id("stage2", dataset, model)))
            parents = top_unique_trials_by_test(stage2_item["payload"], top_k=int(stage4_plan.get("parent_top_k", 2))) if stage2_item else []
            if current and current.get("config"):
                parents = [*parents, {"config": dict(current["config"])}]
            parent_configs = [dict(item["config"]) for item in parents if dict(item.get("config") or {})]
            if not parent_configs:
                parent_configs = [dict(state.get("config") or {})]
            cursor += 1
            rows.append(with_runtime_safeguards({
                    "stage": "stage4",
                    "run_axis": "add_stage4_expansion",
                    "dataset": dataset,
                    "model": model,
                    "family": "baseline",
                    "job_id": make_job_id("stage4", dataset, model),
                    "run_phase": make_job_id("stage4", dataset, model),
                    "parent_job_id": make_job_id("stage3", dataset, model),
                    "runtime_seed": 940000 + cursor,
                    "max_evals": int(stage4_plan["max_evals"]),
                    "search_space": refine_search_space(spec, parent_configs, model=model, tier=band, stage="stage4", seed=940000 + cursor),
                    "fixed_context": dict(spec.get("fixed_context") or {}),
                    "tier": band,
                    "base_tier": str(state["tier"]),
                    "stage4_severity": int(stage4_plan.get("severity", 0)),
                    "tune_epochs": stage_plan("stage4")["tune_epochs"],
                    "tune_patience": stage_plan("stage4")["tune_patience"],
                }, default_max_run_hours=DEFAULT_ADDTUNING_MAX_RUN_HOURS, oom_retry_limit=DEFAULT_ADDTUNING_OOM_RETRY_LIMIT))
    return rows


def build_stage5_rows(summary_payload: Dict[str, Any], datasets: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = current_focus_lookup(summary_payload)
    stage3_selected = aggregate_stage3_selected()
    stage4_payloads = load_stage_payloads("stage4")
    stage4_manifest = load_stage_manifest_rows("stage4")
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        ds = normalize_dataset(dataset)
        for model in FOCUS_MODELS:
            state = lookup.get((ds, model))
            if state is None:
                continue
            stage4_job_id = make_job_id("stage4", dataset, model)
            stage4_meta = dict(stage4_manifest.get(stage4_job_id) or {})
            tier = str(stage4_meta.get("tier", state["tier"]))
            top_k = int(STAGE4_BAND_POLICY.get(tier, {"stage5_top_k": 1}).get("stage5_top_k", 1))
            stage4_item = stage4_payloads.get((ds, model, stage4_job_id))
            trials = top_unique_trials_by_test(stage4_item["payload"], top_k=top_k) if stage4_item else []
            if not trials and (ds, model) in stage3_selected:
                trials = [{"config": dict(stage3_selected[(ds, model)].get("config") or {})}]
            seed_count = seed_count_for_pair(ds, tier)
            for config_rank, trial in enumerate(trials, start=1):
                config = dict(trial.get("config") or {})
                if not config:
                    continue
                for seed_id in range(1, seed_count + 1):
                    cursor += 1
                    rows.append(with_runtime_safeguards({
                            "stage": "stage5",
                            "run_axis": "add_stage5_final_confirm",
                            "dataset": dataset,
                            "model": model,
                            "family": "baseline",
                            "job_id": make_job_id("stage5", dataset, model, suffix=f"C{config_rank}_SEED{seed_id}"),
                            "run_phase": make_job_id("stage5", dataset, model, suffix=f"C{config_rank}_SEED{seed_id}"),
                            "parent_job_id": make_job_id("stage4", dataset, model),
                            "runtime_seed": 950000 + cursor,
                            "max_evals": 1,
                            "search_space": config,
                            "fixed_context": {},
                            "tier": tier,
                            "config_rank": config_rank,
                            "seed_id": seed_id,
                            "config_signature": config_signature(config),
                            "tune_epochs": stage_plan("stage5")["tune_epochs"],
                            "tune_patience": stage_plan("stage5")["tune_patience"],
                        }, default_max_run_hours=DEFAULT_ADDTUNING_MAX_RUN_HOURS, oom_retry_limit=DEFAULT_ADDTUNING_OOM_RETRY_LIMIT))
    return rows


def run_stage(stage: str, rows: List[Dict[str, Any]], *, gpus: List[str], search_algo: str, resume_from_logs: bool, dry_run: bool) -> int:
    manifest_path = write_stage_manifest(stage, rows)
    print(f"[addtuning][{stage}] manifest -> {manifest_path}")
    print(f"[addtuning][{stage}] run_count={len(rows)}")
    if not rows:
        return 0
    return run_jobs(rows, stage=stage, gpus=gpus, search_algo=search_algo, resume_from_logs=resume_from_logs, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused add-on tuning pipeline for selected baseline models")
    parser.add_argument("action", choices=["plan", "stage1", "stage2", "stage3", "stage4", "stage5", "auto"])
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--max-run-hours", type=float, default=DEFAULT_MAX_RUN_HOURS)
    parser.add_argument("--oom-retry-limit", type=int, default=DEFAULT_OOM_RETRY_LIMIT)
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    global DEFAULT_ADDTUNING_MAX_RUN_HOURS, DEFAULT_ADDTUNING_OOM_RETRY_LIMIT
    args = parse_args()
    DEFAULT_ADDTUNING_MAX_RUN_HOURS = float(args.max_run_hours)
    DEFAULT_ADDTUNING_OOM_RETRY_LIMIT = int(args.oom_retry_limit)
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    gpus = parse_csv_list(args.gpus) or ["0"]
    specs = load_base_pair_specs()
    aggregate = aggregate_existing_results(datasets)
    write_existing_summary_files(aggregate)
    summary_payload = read_json(SUMMARY_JSON)

    if args.action == "plan":
        print(f"[addtuning] wrote current summary -> {SUMMARY_MD}")
        return 0

    if args.action == "stage1":
        return run_stage("stage1", build_stage1_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    if args.action == "stage2":
        return run_stage("stage2", build_stage2_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    if args.action == "stage3":
        return run_stage("stage3", build_stage3_rows(summary_payload, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    if args.action == "stage4":
        return run_stage("stage4", build_stage4_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    if args.action == "stage5":
        return run_stage("stage5", build_stage5_rows(summary_payload, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))

    rc = 0
    rc |= run_stage("stage1", build_stage1_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    rc |= run_stage("stage2", build_stage2_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    rc |= run_stage("stage3", build_stage3_rows(summary_payload, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    rc |= run_stage("stage4", build_stage4_rows(summary_payload, specs, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    rc |= run_stage("stage5", build_stage5_rows(summary_payload, datasets), gpus=gpus, search_algo=args.search_algo, resume_from_logs=bool(args.resume_from_logs), dry_run=bool(args.dry_run))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())