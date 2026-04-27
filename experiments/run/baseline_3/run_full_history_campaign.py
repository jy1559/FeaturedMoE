#!/usr/bin/env python3
"""Run the baseline_3 full-history campaign on feature_added_v4 datasets.

Design:
- 6 datasets x 10 models x 4 combos = 240 runs
- combo order is outermost: C1 for all dataset/model pairs, then C2, C3, C4
- each run uses hyperopt with a small search space (learning_rate + weight_decay)
- all runs use session_fixed + full_history_session_targets
- queue is shared, and each GPU pulls the next pending run immediately
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

TRACK = "baseline_3"
DEFAULT_AXIS = "FULL_HISTORY_V4_C4"

LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK
RESULT_ROOT = ARTIFACT_ROOT / "results" / TRACK

DEFAULT_DATASETS = [
    "beauty",
    "KuaiRecLargeStrictPosV2_0.2",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
]

DEFAULT_MODELS = [
    "sasrec",
    "tisasrec",
    "gru4rec",
    "duorec",
    "fearec",
    "bsarec",
    "fame",
    "difsr",
    "fdsa",
    "featured_moe_n3",
]

DATASET_CONFIG_MAP = {
    "kuaireclargestrictposv2_0.2": "tune_kuai_strict_small",
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
    "beauty": "tune_ab",
    "foursquare": "tune_fs",
    "movielens1m": "tune_ml",
    "retail_rocket": "tune_rr",
}

DATASET_ORDER_INDEX = {dataset: idx for idx, dataset in enumerate(DEFAULT_DATASETS)}
MODEL_ORDER_INDEX = {model: idx for idx, model in enumerate(DEFAULT_MODELS)}

BASELINE2_SUMMARY_PATHS = [
    ("PAIR60_V4", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_V4" / "summary.csv"),
    ("PAIR60_V4_REVISED", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_V4_REVISED" / "summary.csv"),
    ("PAIR60_V4_REVISED_LONG12H", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_V4_REVISED_LONG12H" / "summary.csv"),
    ("PAIR60_ADDTUNING", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_ADDTUNING" / "summary.csv"),
    ("PAIR60_ADDTUNING2", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_ADDTUNING2" / "summary.csv"),
    ("PAIR60_ADDTUNING3", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_ADDTUNING3" / "summary.csv"),
    ("PAIR60_ADDTUNING3_2", ARTIFACT_ROOT / "logs" / "baseline_2" / "PAIR60_ADDTUNING3_2" / "summary.csv"),
]

FMOE_SUMMARY_PATH_TEMPLATES = [
    ("fmoe_n4_cross_dataset", ARTIFACT_ROOT / "logs" / "fmoe_n4" / "CrossDataset_A12_Portfolio" / "{dataset}" / "summary.csv"),
    ("fmoe_n4_stage1", ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage1_A12_BroadTemplates" / "{dataset}" / "summary.csv"),
    ("fmoe_n4_stage2", ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage2_A12_MixedTemplates" / "{dataset}" / "summary.csv"),
    ("fmoe_n4_stage3", ARTIFACT_ROOT / "logs" / "fmoe_n4" / "Stage3_A12_SeenFocus" / "{dataset}" / "summary.csv"),
    ("fmoe_n3_final_all", ARTIFACT_ROOT / "logs" / "fmoe_n3" / "Final_all_datasets" / "{dataset}" / "summary.csv"),
    ("fmoe_n3_final_tuning", ARTIFACT_ROOT / "logs" / "fmoe_n3" / "Final_tuning_A12" / "{dataset}" / "summary.csv"),
]

FMOE_LOG_ROOT_TEMPLATES = {
    source_name: Path(str(template)).parent
    for source_name, template in FMOE_SUMMARY_PATH_TEMPLATES
}

FMOE_RESULT_DIRS = [
    ARTIFACT_ROOT / "results" / "fmoe_n4",
    ARTIFACT_ROOT / "results" / "fmoe_n3",
]

ALLOWED_FMOE_RUN_AXES = {
    "crossdataset_a12_portfolio",
    "stage1_a12_broadtemplates",
    "stage2_a12_mixedtemplates",
    "stage3_a12_seenfocus",
    "final_all_datasets",
    "final_tuning_a12",
}

SUMMARY_FIELDS = [
    "dataset",
    "model",
    "model_label",
    "pair_id",
    "combo_id",
    "combo_rank",
    "combo_kind",
    "combo_theme",
    "combo_source",
    "source_axis",
    "source_phase",
    "selection_score",
    "hparam_id",
    "architecture_id",
    "run_phase",
    "run_id",
    "runtime_seed",
    "gpu_id",
    "status",
    "best_valid_mrr20",
    "test_mrr20",
    "seen_test_mrr20",
    "valid_unseen_mrr20",
    "valid_unseen_hit20",
    "test_unseen_mrr20",
    "test_unseen_hit20",
    "valid_main_seen_count",
    "valid_main_unseen_count",
    "test_main_seen_count",
    "test_main_unseen_count",
    "avg_epoch_time_sec",
    "avg_epoch_time_ms",
    "test_inference_time_sec",
    "lr_lo",
    "lr_hi",
    "wd_lo",
    "wd_hi",
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
    "combo_kind",
    "combo_theme",
    "combo_source",
    "source_axis",
    "source_phase",
    "selection_score",
    "hparam_id",
    "architecture_id",
    "lr_lo",
    "lr_hi",
    "wd_lo",
    "wd_hi",
    "params_json",
]

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()


@dataclass
class Candidate:
    params: Dict[str, Any]
    source: str
    source_axis: str
    source_phase: str
    best_valid_mrr20: float
    seen_test_mrr20: float
    overall_test_mrr20: float
    selection_score: float
    hparam_id: str = ""
    architecture_id: str = ""

    @property
    def tie_breaker(self) -> Tuple[float, float, float, float]:
        return (
            float(self.selection_score),
            float(self.best_valid_mrr20),
            float(self.seen_test_mrr20),
            float(self.overall_test_mrr20),
        )


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
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if remove_path and prepend_path is not None and str(prepend_path) in sys.path:
            sys.path.remove(str(prepend_path))


PAIR60 = load_module(
    RUN_DIR / "baseline_2" / "run_pair60_campaign.py",
    "baseline2_pair60_for_full_history",
)

MODEL_SPEC_BY_OPTION = {key: dict(value) for key, value in PAIR60.MODEL_SPEC_BY_OPTION.items()}


def now_utc() -> str:
    return PAIR60.now_utc()


def parse_csv_list(text: str) -> List[str]:
    return PAIR60.parse_csv_list(text)


def parse_csv_ints(text: str) -> List[int]:
    return PAIR60.parse_csv_ints(text)


def safe_float(value: Any, default: float = 0.0) -> float:
    return PAIR60.safe_float(value, default)


def safe_int(value: Any, default: int = 0) -> int:
    return PAIR60.safe_int(value, default)


def sanitize_token(text: str, *, upper: bool = False) -> str:
    return PAIR60.sanitize_token(text, upper=upper)


def read_csv(path: Path) -> List[Dict[str, str]]:
    return PAIR60.read_csv(path)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    PAIR60.write_csv(path, rows, fieldnames)


def dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG_MAP:
        raise KeyError(f"Unknown dataset config mapping: {dataset}")
    return DATASET_CONFIG_MAP[key]


def canonical_result_path(raw_path: str | Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    candidates = [Path(text).expanduser()]
    replacements = [
        ("/workspace/jy1559/FMoE", str(REPO_ROOT)),
        ("/workspace/FMoE", str(REPO_ROOT)),
    ]
    for old, new in replacements:
        if text.startswith(old):
            candidates.append(Path(text.replace(old, new, 1)).expanduser())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


@lru_cache(maxsize=8192)
def read_json_cached(path_text: str) -> Dict[str, Any]:
    path = canonical_result_path(path_text)
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def harmonic_or_min(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return min(a, b)
    return 2.0 * a * b / (a + b)


def extract_selection_metrics(row: Dict[str, Any]) -> Tuple[float, float, float, str]:
    result_path = str(row.get("result_path", "") or "")
    payload = read_json_cached(result_path) if result_path else {}

    best_valid = safe_float(row.get("best_valid_mrr20", row.get("run_best_valid_mrr20", row.get("global_best_valid_mrr20", 0.0))), 0.0)
    overall_test = safe_float(row.get("test_mrr20", 0.0), 0.0)
    seen_test = overall_test

    if payload:
        best_valid = safe_float((payload.get("best_valid_result", {}) or {}).get("mrr@20", best_valid), best_valid)
        overall_test = safe_float((payload.get("test_result", {}) or {}).get("mrr@20", overall_test), overall_test)
        valid_seen_payload = (payload.get("best_valid_special_metrics", {}) or {}).get("overall_seen_target", {}) or {}
        seen_payload = (payload.get("test_special_metrics", {}) or {}).get("overall_seen_target", {}) or {}
        best_valid = safe_float(valid_seen_payload.get("mrr@20", best_valid), best_valid)
        seen_test = safe_float(seen_payload.get("mrr@20", overall_test), overall_test)

    score = harmonic_or_min(best_valid, seen_test)
    return best_valid, seen_test, overall_test, result_path


def normalize_baseline_params(model_option: str, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    model = str(model_option).lower()
    params = dict(raw_params or {})

    if "n_layers" in params and "num_layers" not in params:
        params["num_layers"] = params["n_layers"]
    if "n_heads" in params and "num_heads" not in params:
        params["num_heads"] = params["n_heads"]
    if "dropout_ratio" in params and "dropout" not in params:
        params["dropout"] = params["dropout_ratio"]
    if "dropout_prob" in params and "dropout" not in params:
        params["dropout"] = params["dropout_prob"]
    if "hidden_dropout_prob" in params and "dropout" not in params:
        params["dropout"] = params["hidden_dropout_prob"]

    params.pop("learning_rate", None)
    params.pop("lr_group", None)
    params.pop("fixed_hidden_dropout_prob", None)
    params.pop("fixed_weight_decay", None)

    if model == "gru4rec":
        hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
        return {
            "hidden_size": hidden,
            "embedding_size": hidden,
            "num_layers": safe_int(params.get("num_layers", 1), 1),
            "dropout": safe_float(params.get("dropout", 0.2), 0.2),
            "weight_decay": safe_float(params.get("weight_decay", 1e-4), 1e-4),
            "max_len": safe_int(params.get("max_len", params.get("MAX_ITEM_LIST_LENGTH", 20)), 20),
        }

    if model == "featured_moe_n3":
        emb = safe_int(params.get("embedding_size", params.get("hidden_size", 128)), 128)
        return {
            "embedding_size": emb,
            "hidden_size": emb,
            "num_heads": safe_int(params.get("num_heads", 4), 4),
            "d_ff": safe_int(params.get("d_ff", emb * 2), emb * 2),
            "d_expert_hidden": safe_int(params.get("d_expert_hidden", emb), emb),
            "d_router_hidden": safe_int(params.get("d_router_hidden", max(emb // 2, 32)), max(emb // 2, 32)),
            "d_feat_emb": safe_int(params.get("d_feat_emb", 12), 12),
            "expert_scale": safe_int(params.get("expert_scale", 2), 2),
            "dropout": safe_float(params.get("dropout", params.get("hidden_dropout_prob", 0.15)), 0.15),
            "weight_decay": safe_float(params.get("weight_decay", 1e-6), 1e-6),
            "max_len": safe_int(params.get("max_len", params.get("MAX_ITEM_LIST_LENGTH", 20)), 20),
            "fmoe_v2_layout_id": safe_int(params.get("fmoe_v2_layout_id", 0), 0),
            "router_impl": params.get("router_impl", "learned"),
            "router_use_hidden": bool(params.get("router_use_hidden", True)),
            "router_use_feature": bool(params.get("router_use_feature", True)),
            "fmoe_stage_execution_mode": params.get("fmoe_stage_execution_mode", "serial"),
            "fmoe_architecture_id": str(params.get("fmoe_architecture_id", params.get("architecture_id", "A12"))),
        }

    hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
    normalized = {
        "hidden_size": hidden,
        "embedding_size": safe_int(params.get("embedding_size", hidden), hidden),
        "num_layers": safe_int(params.get("num_layers", 2), 2),
        "num_heads": safe_int(params.get("num_heads", 2), 2),
        "inner_size": safe_int(params.get("inner_size", hidden * 2), hidden * 2),
        "dropout": safe_float(params.get("dropout", 0.15), 0.15),
        "weight_decay": safe_float(params.get("weight_decay", 1e-4), 1e-4),
        "max_len": safe_int(params.get("max_len", params.get("MAX_ITEM_LIST_LENGTH", 20)), 20),
    }

    if model == "tisasrec":
        normalized["time_span"] = safe_int(params.get("time_span", 256), 256)
    if model in {"duorec", "fearec"}:
        if "contrast" in params:
            normalized["contrast"] = str(params.get("contrast"))
        if "tau" in params:
            normalized["tau"] = safe_float(params.get("tau", 0.2), 0.2)
        if "lmd" in params:
            normalized["lmd"] = safe_float(params.get("lmd", 0.04), 0.04)
        if "lmd_sem" in params:
            normalized["lmd_sem"] = safe_float(params.get("lmd_sem", 0.0), 0.0)
        if "global_ratio" in params:
            normalized["global_ratio"] = safe_float(params.get("global_ratio", 1.0), 1.0)
        if "semantic_sample_max_tries" in params:
            normalized["semantic_sample_max_tries"] = safe_int(params.get("semantic_sample_max_tries", 2), 2)
    if model in {"difsr", "fdsa"}:
        normalized["attribute_hidden_size"] = safe_int(params.get("attribute_hidden_size", hidden), hidden)
        if "fusion_type" in params:
            normalized["fusion_type"] = str(params.get("fusion_type"))
        if "use_attribute_predictor" in params:
            normalized["use_attribute_predictor"] = bool(params.get("use_attribute_predictor"))
        if "lambda_attr" in params:
            normalized["lambda_attr"] = safe_float(params.get("lambda_attr", 0.1), 0.1)
    if model == "fdsa":
        normalized["selected_features"] = list(params.get("selected_features", ["category"]))
        normalized["pooling_mode"] = str(params.get("pooling_mode", "mean"))
    if model == "fame":
        normalized["num_experts"] = safe_int(params.get("num_experts", 3), 3)
    return normalized


def normalize_fmoe_result_params(dataset: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    payload = read_json_cached(str(row.get("result_path", "") or ""))
    best_params = dict(payload.get("best_params", {}) or {})
    if not best_params:
        return None

    arch_id = str(row.get("architecture_id", best_params.get("fmoe_architecture_id", "A12")) or "A12")
    params = {
        "embedding_size": safe_int(best_params.get("embedding_size", best_params.get("hidden_size", 128)), 128),
        "hidden_size": safe_int(best_params.get("hidden_size", best_params.get("embedding_size", 128)), 128),
        "num_heads": safe_int(best_params.get("num_heads", 4), 4),
        "d_ff": safe_int(best_params.get("d_ff", 256), 256),
        "d_expert_hidden": safe_int(best_params.get("d_expert_hidden", 128), 128),
        "d_router_hidden": safe_int(best_params.get("d_router_hidden", 64), 64),
        "d_feat_emb": safe_int(best_params.get("d_feat_emb", 12), 12),
        "expert_scale": safe_int(best_params.get("expert_scale", 2), 2),
        "dropout": safe_float(best_params.get("fixed_hidden_dropout_prob", best_params.get("hidden_dropout_prob", 0.15)), 0.15),
        "weight_decay": safe_float(best_params.get("fixed_weight_decay", 1e-6), 1e-6),
        "max_len": safe_int(best_params.get("MAX_ITEM_LIST_LENGTH", best_params.get("max_len", 20)), 20),
        "fmoe_v2_layout_id": safe_int(best_params.get("fmoe_v2_layout_id", 0), 0),
        "router_impl": str(best_params.get("router_impl", "learned")),
        "router_use_hidden": bool(best_params.get("router_use_hidden", True)),
        "router_use_feature": bool(best_params.get("router_use_feature", True)),
        "fmoe_stage_execution_mode": str(best_params.get("fmoe_stage_execution_mode", "serial")),
        "fmoe_architecture_id": arch_id,
    }
    return normalize_baseline_params("featured_moe_n3", params)


@lru_cache(maxsize=2048)
def find_fmoe_log(source_name: str, dataset: str, run_phase: str) -> Path | None:
    template = FMOE_LOG_ROOT_TEMPLATES.get(str(source_name))
    if template is None:
        return None
    root = Path(str(template).format(dataset=dataset))
    if not root.exists():
        return None
    for log_path in root.rglob("*.log"):
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(6):
                    line = fh.readline()
                    if not line:
                        break
                    if f"run_phase={run_phase}" in line:
                        return log_path
        except Exception:
            continue
    return None


def _extract_command_line(log_path: Path) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    for line in lines[:20]:
        if "hyperopt_tune.py" in line:
            return line.strip()
    return ""


def _extract_cli_value(command: str, key: str) -> str:
    import re

    pattern = re.compile(rf"(?:^|\s)\+\+{re.escape(key)}=([^\s]+)")
    match = pattern.search(command)
    return match.group(1).strip() if match else ""


def _extract_search_choice(command: str, key: str) -> List[str]:
    import re

    pattern = re.compile(rf"(?:^|\s)\+\+search\.{re.escape(key)}=\[([^\]]*)\]")
    match = pattern.search(command)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [token.strip().strip('"') for token in raw.split(",") if token.strip()]


def reconstruct_fmoe_params_from_log(source_name: str, dataset: str, row: Dict[str, Any]) -> Dict[str, Any] | None:
    run_phase = str(row.get("run_phase", "") or "").strip()
    if not run_phase:
        return None
    log_path = find_fmoe_log(source_name, dataset, run_phase)
    if log_path is None:
        return None
    command = _extract_command_line(log_path)
    if not command:
        return None

    result_path = canonical_result_path(str(row.get("result_path", "") or ""))
    payload = read_json_cached(str(result_path)) if result_path is not None and result_path.exists() else {}
    best_params = dict(payload.get("best_params", {}) or {})

    def _num(key: str, default: float) -> float:
        token = _extract_cli_value(command, key)
        return safe_float(token.strip('"'), default) if token else float(default)

    weight_choices = _extract_search_choice(command, "weight_decay")
    hidden_dropout_choices = _extract_search_choice(command, "hidden_dropout_prob")

    hidden_dropout = safe_float(best_params.get("hidden_dropout_prob", hidden_dropout_choices[0] if hidden_dropout_choices else 0.15), 0.15)
    weight_decay = safe_float(weight_choices[0] if weight_choices else 1e-6, 1e-6)
    max_len = safe_int(best_params.get("MAX_ITEM_LIST_LENGTH", _num("MAX_ITEM_LIST_LENGTH", 20)), 20)
    expert_scale = safe_int(best_params.get("expert_scale", _num("expert_scale", 2)), 2)
    emb = safe_int(_num("embedding_size", 128), 128)

    params = {
        "embedding_size": emb,
        "hidden_size": emb,
        "num_heads": safe_int(_num("num_heads", 4), 4),
        "d_ff": safe_int(_num("d_ff", emb * 2), emb * 2),
        "d_expert_hidden": safe_int(_num("d_expert_hidden", emb), emb),
        "d_router_hidden": safe_int(_num("d_router_hidden", max(emb // 2, 32)), max(emb // 2, 32)),
        "d_feat_emb": safe_int(_num("d_feat_emb", 12), 12),
        "expert_scale": expert_scale,
        "dropout": hidden_dropout,
        "weight_decay": weight_decay,
        "max_len": max_len,
        "fmoe_v2_layout_id": safe_int(_num("fmoe_v2_layout_id", 0), 0),
        "router_impl": _extract_cli_value(command, "router_impl").strip('"') or "learned",
        "router_use_hidden": (_extract_cli_value(command, "router_use_hidden") or "true").lower() == "true",
        "router_use_feature": (_extract_cli_value(command, "router_use_feature") or "true").lower() == "true",
        "fmoe_stage_execution_mode": _extract_cli_value(command, "fmoe_stage_execution_mode").strip('"') or "serial",
        "fmoe_architecture_id": _extract_cli_value(command, "fmoe_architecture_id").strip('"') or str(row.get("architecture_id", "A12") or "A12"),
    }
    return normalize_baseline_params("featured_moe_n3", params)


def load_baseline2_sources() -> List[Tuple[str, List[Dict[str, str]]]]:
    out: List[Tuple[str, List[Dict[str, str]]]] = []
    for axis_name, path in BASELINE2_SUMMARY_PATHS:
        rows = read_csv(path)
        if rows:
            out.append((axis_name, rows))
    return out


def load_fmoe_sources(dataset: str) -> List[Tuple[str, List[Dict[str, str]]]]:
    out: List[Tuple[str, List[Dict[str, str]]]] = []
    for source_name, template in FMOE_SUMMARY_PATH_TEMPLATES:
        path = Path(str(template).format(dataset=dataset))
        rows = read_csv(path)
        if rows:
            out.append((source_name, rows))
    return out


def candidate_dedup_key(model_option: str, params: Dict[str, Any]) -> str:
    normalized = normalize_baseline_params(model_option, params)
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_existing_baseline_candidates(
    dataset: str,
    model_option: str,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Candidate]:
    out_by_sig: Dict[str, Candidate] = {}
    ds = str(dataset).strip().lower()
    model = str(model_option).strip().lower()
    for axis_name, rows in baseline2_sources:
        for row in rows:
            if str(row.get("dataset", "")).strip().lower() != ds:
                continue
            if str(row.get("model", "")).strip().lower() != model:
                continue
            status = str(row.get("status", "")).strip().lower()
            if status != "ok":
                continue
            params_text = str(row.get("params_json", row.get("base_params_json", "")) or "").strip()
            if not params_text:
                continue
            try:
                params = normalize_baseline_params(model_option, json.loads(params_text))
            except Exception:
                continue
            best_valid, seen_test, overall_test, result_path = extract_selection_metrics(row)
            cand = Candidate(
                params=params,
                source=f"{axis_name}:{row.get('run_phase', '')}",
                source_axis=axis_name,
                source_phase=str(row.get("run_phase", "") or ""),
                best_valid_mrr20=best_valid,
                seen_test_mrr20=seen_test,
                overall_test_mrr20=overall_test,
                selection_score=harmonic_or_min(best_valid, seen_test),
                hparam_id=str(row.get("hparam_id", "") or ""),
                architecture_id=str(row.get("architecture_id", "") or ""),
            )
            sig = candidate_dedup_key(model_option, cand.params)
            prev = out_by_sig.get(sig)
            if prev is None or cand.tie_breaker > prev.tie_breaker:
                out_by_sig[sig] = cand
    return sorted(out_by_sig.values(), key=lambda cand: cand.tie_breaker, reverse=True)


def build_existing_fmoe_candidates(dataset: str) -> List[Candidate]:
    out_by_sig: Dict[str, Candidate] = {}
    dataset_key = str(dataset).strip()

    for result_dir in FMOE_RESULT_DIRS:
        if not result_dir.exists():
            continue
        for result_path in result_dir.glob(f"{dataset_key}_FeaturedMoE_N3*.json"):
            payload = read_json_cached(str(result_path))
            if not payload:
                continue
            if str(payload.get("dataset", "")).strip() != dataset_key:
                continue
            if str(payload.get("model", "")).strip().lower() not in {"featuredmoe_n3", "featured_moe_n3"}:
                continue
            run_axis = str(payload.get("run_axis", "") or "").strip().lower()
            if run_axis not in ALLOWED_FMOE_RUN_AXES:
                continue
            row = {
                "result_path": str(result_path),
                "run_phase": str(payload.get("run_phase", "") or ""),
                "architecture_id": str((payload.get("best_params", {}) or {}).get("fmoe_architecture_id", "") or ""),
            }
            params = normalize_fmoe_result_params(dataset, row)
            if not params:
                continue
            best_valid = safe_float((payload.get("best_valid_result", {}) or {}).get("mrr@20", payload.get("best_mrr@20", 0.0)), 0.0)
            overall_test = safe_float((payload.get("test_result", {}) or {}).get("mrr@20", payload.get("test_mrr@20", 0.0)), 0.0)
            seen_payload = (payload.get("test_special_metrics", {}) or {}).get("overall_seen_target", {}) or {}
            seen_test = safe_float(seen_payload.get("mrr@20", overall_test), overall_test)
            cand = Candidate(
                params=params,
                source=f"{run_axis}:{row['run_phase']}",
                source_axis=run_axis,
                source_phase=row["run_phase"],
                best_valid_mrr20=best_valid,
                seen_test_mrr20=seen_test,
                overall_test_mrr20=overall_test,
                selection_score=harmonic_or_min(best_valid, seen_test),
                hparam_id=str((payload.get("best_params", {}) or {}).get("hparam_id", "") or ""),
                architecture_id=str(params.get("fmoe_architecture_id", row.get("architecture_id", "")) or ""),
            )
            sig = candidate_dedup_key("featured_moe_n3", cand.params)
            prev = out_by_sig.get(sig)
            if prev is None or cand.tie_breaker > prev.tie_breaker:
                out_by_sig[sig] = cand

    for source_name, rows in load_fmoe_sources(dataset):
        for row in rows:
            status = str(row.get("status", "")).strip().lower()
            trigger = str(row.get("trigger", "")).strip().lower()
            if status != "ok" and "run_complete" not in status and "run_complete" not in trigger:
                continue
            result_path = canonical_result_path(str(row.get("result_path", "") or ""))
            if result_path is None or not result_path.exists():
                params = reconstruct_fmoe_params_from_log(source_name, dataset, row)
            else:
                params = normalize_fmoe_result_params(dataset, row)
            if params is None:
                params = reconstruct_fmoe_params_from_log(source_name, dataset, row)
            if not params:
                continue
            best_valid, seen_test, overall_test, _ = extract_selection_metrics(row)
            cand = Candidate(
                params=params,
                source=f"{source_name}:{row.get('run_phase', '')}",
                source_axis=source_name,
                source_phase=str(row.get("run_phase", "") or ""),
                best_valid_mrr20=best_valid,
                seen_test_mrr20=seen_test,
                overall_test_mrr20=overall_test,
                selection_score=harmonic_or_min(best_valid, seen_test),
                hparam_id=str(row.get("hparam_id", "") or ""),
                architecture_id=str(row.get("architecture_id", "") or params.get("fmoe_architecture_id", "")),
            )
            sig = candidate_dedup_key("featured_moe_n3", cand.params)
            prev = out_by_sig.get(sig)
            if prev is None or cand.tie_breaker > prev.tie_breaker:
                out_by_sig[sig] = cand
    return sorted(out_by_sig.values(), key=lambda cand: cand.tie_breaker, reverse=True)


def select_top_unique_candidates(model_option: str, pool: List[Candidate], n: int) -> List[Candidate]:
    seen = set()
    out: List[Candidate] = []
    for cand in pool:
        sig = candidate_dedup_key(model_option, cand.params)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= n:
            break
    return out


def full_history_len_boost(dataset: str) -> int:
    return {
        "beauty": 6,
        "KuaiRecLargeStrictPosV2_0.2": 8,
        "foursquare": 6,
        "retail_rocket": 6,
        "movielens1m": 10,
        "lastfm0.03": 12,
    }.get(str(dataset), 8)


def full_history_len_cap(dataset: str) -> int:
    return {
        "beauty": 28,
        "KuaiRecLargeStrictPosV2_0.2": 40,
        "foursquare": 30,
        "retail_rocket": 30,
        "movielens1m": 40,
        "lastfm0.03": 50,
    }.get(str(dataset), 40)


def build_full_history_adapted_params(dataset: str, model_option: str, base_params: Dict[str, Any]) -> Dict[str, Any]:
    model = str(model_option).lower()
    params = normalize_baseline_params(model_option, base_params)
    params["max_len"] = min(full_history_len_cap(dataset), safe_int(params.get("max_len", 20), 20) + full_history_len_boost(dataset))
    params["weight_decay"] = round(max(safe_float(params.get("weight_decay", 1e-4), 1e-4) * 1.4, 1e-7), 8)

    if "dropout" in params:
        params["dropout"] = round(min(0.32, safe_float(params.get("dropout", 0.15), 0.15) + 0.02), 4)

    if model in {"sasrec", "tisasrec", "bsarec", "difsr", "fdsa", "fame"}:
        if safe_int(params.get("num_layers", 2), 2) >= 3 and dataset in {"beauty", "foursquare", "retail_rocket"}:
            params["num_layers"] = safe_int(params.get("num_layers", 2), 2) - 1

    if model == "gru4rec":
        if dataset in {"lastfm0.03", "movielens1m"}:
            params["num_layers"] = min(3, safe_int(params.get("num_layers", 1), 1) + 1)

    if model in {"duorec", "fearec"}:
        if "tau" in params:
            params["tau"] = round(min(0.8, safe_float(params.get("tau", 0.2), 0.2) * 1.1), 4)
        if str(params.get("contrast", "un")).lower() == "un" and dataset in {"movielens1m", "lastfm0.03"}:
            params["contrast"] = "su"
            params["lmd"] = 0.0
            params["lmd_sem"] = round(max(safe_float(params.get("lmd_sem", 0.06), 0.06), 0.06), 4)

    if model in {"difsr", "fdsa"}:
        params["fusion_type"] = str(params.get("fusion_type", "gate") or "gate")
        params["use_attribute_predictor"] = True
        params["lambda_attr"] = round(min(0.28, safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.03), 4)

    if model == "fame":
        params["num_experts"] = min(6, safe_int(params.get("num_experts", 3), 3) + 1)

    if model == "featured_moe_n3":
        params["d_router_hidden"] = min(160, int(round(safe_int(params.get("d_router_hidden", 64), 64) * 1.25)))
        params["d_feat_emb"] = max(12, safe_int(params.get("d_feat_emb", 12), 12))
        params["expert_scale"] = min(4, safe_int(params.get("expert_scale", 2), 2) + 1)

    return params


def build_challenge_params(dataset: str, model_option: str, top1: Dict[str, Any], top2: Dict[str, Any]) -> Dict[str, Any]:
    params = PAIR60.build_exploration_params(dataset, model_option, top1, top2)
    params = normalize_baseline_params(model_option, params)
    if "max_len" in params:
        top1_len = safe_int(top1.get("max_len", 20), 20)
        params["max_len"] = min(
            full_history_len_cap(dataset),
            max(
                safe_int(params.get("max_len", 20), 20) + max(2, full_history_len_boost(dataset) // 2),
                top1_len + max(2, full_history_len_boost(dataset) // 3),
            ),
        )
    return params


def choose_batch_sizes(dataset: str, model_option: str) -> Tuple[int, int]:
    dataset_name = str(dataset)
    model_name = str(model_option).lower()

    if dataset_name == "beauty":
        if model_name == "featured_moe_n3":
            return 4096, 8192
        return 6144, 12288

    if dataset_name == "KuaiRecLargeStrictPosV2_0.2":
        if model_name == "featured_moe_n3":
            return 3072, 6144
        return 4096, 8192

    if dataset_name == "foursquare":
        return 4096, 8192

    if dataset_name == "retail_rocket":
        return 4096, 8192

    if dataset_name == "movielens1m":
        return 4096, 8192

    if dataset_name == "lastfm0.03":
        if model_name == "featured_moe_n3":
            return 2048, 4096
        return 3072, 6144

    return 4096, 8192


def compute_lr_bounds(dataset: str, model_option: str, combo_rank: int, params: Dict[str, Any]) -> Tuple[float, float]:
    lo, hi = PAIR60.compute_lr_bounds(dataset, model_option, min(int(combo_rank), 2), params)
    if int(combo_rank) == 3:
        lo *= 0.85
        hi *= 0.95
    elif int(combo_rank) == 4:
        lo *= 0.8
        hi *= 1.05
    lo = max(2e-5, float(lo))
    hi = min(8e-3, float(hi))
    if hi <= lo:
        hi = min(8e-3, lo * 2.0)
    return lo, hi


def compute_wd_bounds(model_option: str, combo_kind: str, params: Dict[str, Any]) -> Tuple[float, float]:
    model = str(model_option).lower()
    default_wd = 1e-6 if model == "featured_moe_n3" else 1e-4
    base = max(safe_float(params.get("weight_decay", default_wd), default_wd), 1e-7)
    if combo_kind == "challenge":
        lo = base * 0.4
        hi = base * 2.5
    elif combo_kind == "full_history_adapted":
        lo = base * 0.6
        hi = base * 2.0
    else:
        lo = base * 0.7
        hi = base * 1.7
    lo = max(1e-7, lo)
    hi = min(5e-3, hi)
    if hi <= lo:
        hi = min(5e-3, lo * 2.0)
    return float(lo), float(hi)


def build_pair_combos(
    dataset: str,
    model_spec: Dict[str, str],
    baseline_mod: Any,
    fmoe_mod: Any,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    model_option = str(model_spec["model_option"])
    family = str(model_spec["family"])

    if family == "fmoe":
        existing = build_existing_fmoe_candidates(dataset)
        fallback = [
            Candidate(
                params=normalize_baseline_params(model_option, cand.params),
                source=cand.source,
                source_axis=cand.source.split(":", 1)[0],
                source_phase="",
                best_valid_mrr20=float(cand.score),
                seen_test_mrr20=float(cand.score),
                overall_test_mrr20=float(cand.score),
                selection_score=float(cand.score),
                hparam_id=cand.hparam_id,
                architecture_id=cand.architecture_id,
            )
            for cand in PAIR60.default_fmoe_candidates(dataset, fmoe_mod)
        ]
    else:
        existing = build_existing_baseline_candidates(dataset, model_option, baseline2_sources)
        fallback = [
            Candidate(
                params=normalize_baseline_params(model_option, cand.params),
                source=cand.source,
                source_axis=cand.source.split(":", 1)[0],
                source_phase="",
                best_valid_mrr20=float(cand.score),
                seen_test_mrr20=float(cand.score),
                overall_test_mrr20=float(cand.score),
                selection_score=float(cand.score),
                hparam_id=cand.hparam_id,
                architecture_id=cand.architecture_id,
            )
            for cand in (
                PAIR60.stage_a_candidate_pool(read_csv(PAIR60.BASELINE2_STAGEA_SUMMARY), dataset, model_option)
                + PAIR60.baseline_history_pool(dataset, model_spec, baseline_mod)
                + PAIR60.default_baseline_candidates(dataset, model_option, baseline_mod)
            )
        ]

    pool = select_top_unique_candidates(model_option, existing, 8)
    if len(pool) < 2:
        seen = {candidate_dedup_key(model_option, cand.params) for cand in pool}
        for cand in fallback:
            sig = candidate_dedup_key(model_option, cand.params)
            if sig in seen:
                continue
            pool.append(cand)
            seen.add(sig)
            if len(pool) >= 2:
                break

    if not pool:
        raise RuntimeError(f"No candidate pool for dataset={dataset} model={model_option}")
    if len(pool) == 1:
        pool.append(pool[0])

    top1 = pool[0]
    top2 = next(
        (
            cand
            for cand in pool[1:]
            if str(cand.source_phase or "") != str(top1.source_phase or "")
        ),
        pool[1],
    )
    adapted = build_full_history_adapted_params(dataset, model_option, top1.params)
    challenge = build_challenge_params(dataset, model_option, top1.params, top2.params)

    combos = [
        {
            "combo_id": "C1",
            "combo_rank": 1,
            "combo_kind": "existing_primary",
            "combo_theme": "best_prior_balanced",
            "combo_source": top1.source,
            "source_axis": top1.source_axis,
            "source_phase": top1.source_phase,
            "selection_score": top1.selection_score,
            "hparam_id": top1.hparam_id,
            "architecture_id": top1.architecture_id,
            "params": normalize_baseline_params(model_option, top1.params),
        },
        {
            "combo_id": "C2",
            "combo_rank": 2,
            "combo_kind": "existing_secondary",
            "combo_theme": "best_prior_alternative",
            "combo_source": top2.source,
            "source_axis": top2.source_axis,
            "source_phase": top2.source_phase,
            "selection_score": top2.selection_score,
            "hparam_id": top2.hparam_id,
            "architecture_id": top2.architecture_id,
            "params": normalize_baseline_params(model_option, top2.params),
        },
        {
            "combo_id": "C3",
            "combo_rank": 3,
            "combo_kind": "full_history_adapted",
            "combo_theme": "longer_context_stabilized",
            "combo_source": f"adapted_from:{top1.source}",
            "source_axis": top1.source_axis,
            "source_phase": top1.source_phase,
            "selection_score": top1.selection_score,
            "hparam_id": top1.hparam_id,
            "architecture_id": top1.architecture_id,
            "params": adapted,
        },
        {
            "combo_id": "C4",
            "combo_rank": 4,
            "combo_kind": "challenge",
            "combo_theme": "aggressive_full_history_variant",
            "combo_source": f"challenge_from:{top1.source}|{top2.source}",
            "source_axis": top1.source_axis,
            "source_phase": top1.source_phase,
            "selection_score": max(top1.selection_score, top2.selection_score),
            "hparam_id": top1.hparam_id,
            "architecture_id": top1.architecture_id,
            "params": challenge,
        },
    ]

    seen = set()
    for combo in combos:
        sig = candidate_dedup_key(model_option, combo["params"])
        if sig in seen:
            combo["params"]["weight_decay"] = round(
                max(safe_float(combo["params"].get("weight_decay", 1e-4), 1e-4) * 1.25, 1e-7),
                8,
            )
            sig = candidate_dedup_key(model_option, combo["params"])
        seen.add(sig)
    return combos


def build_matrix_rows(
    datasets: List[str],
    model_options: List[str],
    baseline_mod: Any,
    fmoe_mod: Any,
    baseline2_sources: List[Tuple[str, List[Dict[str, str]]]],
) -> List[Dict[str, Any]]:
    matrix_rows: List[Dict[str, Any]] = []
    pair_cursor = 0
    for dataset in datasets:
        for model_option in model_options:
            pair_cursor += 1
            model_spec = MODEL_SPEC_BY_OPTION[model_option]
            pair_id = f"P{pair_cursor:03d}"
            combos = build_pair_combos(dataset, model_spec, baseline_mod, fmoe_mod, baseline2_sources)
            for combo in combos:
                lr_lo, lr_hi = compute_lr_bounds(dataset, model_option, int(combo["combo_rank"]), combo["params"])
                wd_lo, wd_hi = compute_wd_bounds(model_option, str(combo["combo_kind"]), combo["params"])
                matrix_rows.append(
                    {
                        "dataset": dataset,
                        "model_option": model_option,
                        "model_label": model_spec["model_label"],
                        "pair_id": pair_id,
                        "combo_id": combo["combo_id"],
                        "combo_rank": int(combo["combo_rank"]),
                        "combo_kind": combo["combo_kind"],
                        "combo_theme": combo["combo_theme"],
                        "combo_source": combo["combo_source"],
                        "source_axis": combo["source_axis"],
                        "source_phase": combo["source_phase"],
                        "selection_score": float(combo["selection_score"]),
                        "hparam_id": combo.get("hparam_id", ""),
                        "architecture_id": combo.get("architecture_id", ""),
                        "lr_lo": float(lr_lo),
                        "lr_hi": float(lr_hi),
                        "wd_lo": float(wd_lo),
                        "wd_hi": float(wd_hi),
                        "params_json": json.dumps(combo["params"], ensure_ascii=False, sort_keys=True),
                    }
                )
    return matrix_rows


def build_run_rows(matrix_rows: List[Dict[str, Any]], axis: str, seeds: List[int], runtime_seed_base: int) -> List[Dict[str, Any]]:
    by_pair_combo = sorted(
        matrix_rows,
        key=lambda item: (
            int(item["combo_rank"]),
            DATASET_ORDER_INDEX[str(item["dataset"])],
            MODEL_ORDER_INDEX[str(item["model_option"])],
        ),
    )
    rows: List[Dict[str, Any]] = []
    cursor = 0
    for item in by_pair_combo:
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
    return PAIR60.extract_error_tail(log_path)


def parse_result_path_from_log(log_path: Path) -> Path | None:
    return PAIR60.parse_result_path_from_log(log_path)


def has_run_status_end_normal(log_path: Path) -> bool:
    return PAIR60.has_run_status_end_normal(log_path)


def weight_search_key(model_option: str) -> str:
    return "fixed_weight_decay" if str(model_option).lower() == "featured_moe_n3" else "weight_decay"


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> List[str]:
    dataset = str(row["dataset"])
    model_option = str(row["model_option"])
    params = json.loads(str(row["params_json"]))
    train_batch_size, eval_batch_size = choose_batch_sizes(dataset, model_option)

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
        "feature_mode=full_v4",
        f"train_batch_size={int(train_batch_size)}",
        f"eval_batch_size={int(eval_batch_size)}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        f"++seed={int(row['runtime_seed'])}",
        "++history_input_mode=full_history_session_targets",
        "++history_group_field=user_id",
        "++target_group_field=session_id",
        "++history_eval_policy=strict_train_prefix",
    ]

    cmd.extend(PAIR60.base_runtime_overrides(model_option, params))

    wd_key = weight_search_key(model_option)
    search_dict = {
        "learning_rate": [float(row["lr_lo"]), float(row["lr_hi"])],
        wd_key: [float(row["wd_lo"]), float(row["wd_hi"])],
    }
    search_types = {
        "learning_rate": "loguniform",
        wd_key: "loguniform",
    }
    cmd.append(f"++search={PAIR60.hydra_literal(search_dict)}")
    cmd.append(f"++search_space_type_overrides={PAIR60.hydra_literal(search_types)}")

    fixed = PAIR60.fixed_search_entries(model_option, params)
    for key, value in fixed.items():
        if key == wd_key:
            continue
        cmd.append(f"++search.{key}={PAIR60.hydra_literal([value])}")
        cmd.append(f"++search_space_type_overrides.{key}=choice")
    return cmd


def parse_result_metrics_and_timing(result_path: Path | None) -> Dict[str, Any]:
    metrics = PAIR60.parse_result_metrics(result_path) if result_path is not None else {}
    timing = {
        "seen_test_mrr20": metrics.get("test_mrr20", 0.0),
        "avg_epoch_time_sec": 0.0,
        "avg_epoch_time_ms": 0.0,
        "test_inference_time_sec": 0.0,
    }
    if result_path is not None and result_path.exists():
        payload = read_json_cached(str(result_path))
        valid_seen_payload = (payload.get("best_valid_special_metrics", {}) or {}).get("overall_seen_target", {}) or {}
        seen_payload = (payload.get("test_special_metrics", {}) or {}).get("overall_seen_target", {}) or {}
        metrics["best_valid_mrr20"] = safe_float(valid_seen_payload.get("mrr@20", metrics.get("best_valid_mrr20", 0.0)), metrics.get("best_valid_mrr20", 0.0))
        metrics["test_mrr20"] = safe_float(seen_payload.get("mrr@20", metrics.get("test_mrr20", 0.0)), metrics.get("test_mrr20", 0.0))
        timing["seen_test_mrr20"] = safe_float(seen_payload.get("mrr@20", metrics.get("test_mrr20", 0.0)), metrics.get("test_mrr20", 0.0))
        timing["avg_epoch_time_sec"] = safe_float(payload.get("avg_epoch_time_sec", 0.0), 0.0)
        timing["avg_epoch_time_ms"] = safe_float(payload.get("avg_epoch_time_ms", timing["avg_epoch_time_sec"] * 1000.0), 0.0)
        timing["test_inference_time_sec"] = safe_float(payload.get("test_inference_time_sec", 0.0), 0.0)
    metrics.update(timing)
    return metrics


def append_timing_footer(log_path: Path, metrics: Dict[str, Any]) -> None:
    if not log_path.exists():
        return
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(
                "[FULL_HISTORY_TIMING] "
                f"avg_epoch_sec={safe_float(metrics.get('avg_epoch_time_sec', 0.0), 0.0):.4f} "
                f"avg_epoch_ms={safe_float(metrics.get('avg_epoch_time_ms', 0.0), 0.0):.1f} "
                f"test_inference_sec={safe_float(metrics.get('test_inference_time_sec', 0.0), 0.0):.4f}\n"
            )
    except Exception:
        return


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
        "combo_rank": int(row["combo_rank"]),
        "combo_kind": row["combo_kind"],
        "combo_theme": row["combo_theme"],
        "combo_source": row["combo_source"],
        "source_axis": row.get("source_axis", ""),
        "source_phase": row.get("source_phase", ""),
        "selection_score": float(row.get("selection_score", 0.0) or 0.0),
        "hparam_id": row.get("hparam_id", ""),
        "architecture_id": row.get("architecture_id", ""),
        "run_phase": row["run_phase"],
        "run_id": row["run_id"],
        "runtime_seed": int(row["runtime_seed"]),
        "gpu_id": str(gpu_id),
        "status": status,
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "seen_test_mrr20": metrics.get("seen_test_mrr20", 0.0),
        "valid_unseen_mrr20": metrics.get("valid_unseen_mrr20", 0.0),
        "valid_unseen_hit20": metrics.get("valid_unseen_hit20", 0.0),
        "test_unseen_mrr20": metrics.get("test_unseen_mrr20", 0.0),
        "test_unseen_hit20": metrics.get("test_unseen_hit20", 0.0),
        "valid_main_seen_count": metrics.get("valid_main_seen_count", 0),
        "valid_main_unseen_count": metrics.get("valid_main_unseen_count", 0),
        "test_main_seen_count": metrics.get("test_main_seen_count", 0),
        "test_main_unseen_count": metrics.get("test_main_unseen_count", 0),
        "avg_epoch_time_sec": metrics.get("avg_epoch_time_sec", 0.0),
        "avg_epoch_time_ms": metrics.get("avg_epoch_time_ms", 0.0),
        "test_inference_time_sec": metrics.get("test_inference_time_sec", 0.0),
        "lr_lo": float(row["lr_lo"]),
        "lr_hi": float(row["lr_hi"]),
        "wd_lo": float(row["wd_lo"]),
        "wd_hi": float(row["wd_hi"]),
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
    metrics = parse_result_metrics_and_timing(result_path_obj)
    append_timing_footer(log_path, metrics)
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
    metrics = parse_result_metrics_and_timing(result_path_obj)
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
        safe_float(row.get("seen_test_mrr20", 0.0), 0.0),
        safe_float(row.get("test_mrr20", 0.0), 0.0),
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
                "combo_kind": best.get("combo_kind", ""),
                "best_valid_mrr20": best.get("best_valid_mrr20", 0.0),
                "seen_test_mrr20": best.get("seen_test_mrr20", 0.0),
                "test_mrr20": best.get("test_mrr20", 0.0),
                "avg_epoch_time_sec": best.get("avg_epoch_time_sec", 0.0),
                "test_inference_time_sec": best.get("test_inference_time_sec", 0.0),
                "result_path": best.get("result_path", ""),
            }
        )
    out.sort(key=lambda r: (DATASET_ORDER_INDEX.get(str(r["dataset"]), 999), MODEL_ORDER_INDEX.get(str(r["model"]), 999)))
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
    parser.add_argument("--search-algo", type=str, default="random")

    parser.add_argument("--resume-from-logs", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--slack-progress-step", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    STOP_EVENT.clear()
    install_signal_handlers()

    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    model_options = [m.lower() for m in parse_csv_list(args.models)] or list(DEFAULT_MODELS)
    for model in model_options:
        if model not in MODEL_SPEC_BY_OPTION:
            raise RuntimeError(f"Unsupported model option: {model}")
    gpus = parse_csv_list(args.gpus) or ["0"]
    seeds = parse_csv_ints(args.seeds) or [1]

    baseline_mod, fmoe_mod, slack_mod = PAIR60.load_external_modules()
    SlackProgressNotifier = getattr(slack_mod, "SlackProgressNotifier")

    baseline2_sources = load_baseline2_sources()
    matrix_rows = build_matrix_rows(datasets, model_options, baseline_mod, fmoe_mod, baseline2_sources)

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
                "combo_kind",
                "best_valid_mrr20",
                "seen_test_mrr20",
                "test_mrr20",
                "avg_epoch_time_sec",
                "test_inference_time_sec",
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
            DATASET_ORDER_INDEX.get(str(r.get("dataset", "")), 999),
            MODEL_ORDER_INDEX.get(str(r.get("model", "")), 999),
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
            "combo_kind",
            "best_valid_mrr20",
            "seen_test_mrr20",
            "test_mrr20",
            "avg_epoch_time_sec",
            "test_inference_time_sec",
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