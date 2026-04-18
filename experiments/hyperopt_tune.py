#!/usr/bin/env python3
"""
Hyperopt (TPE) hyperparameter tuner for RecBole sequential recommendation.

Background
----------
Hyperopt implements Tree-structured Parzen Estimators (TPE), a Bayesian
optimisation algorithm.  Unlike grid/random search that evaluates configs
independently, TPE builds a probabilistic model from past trial results and
proposes the *next most promising* config.  This means:

  * Better configs are found with fewer trials (typically 30-60 vs 1000+ grid)
  * Continuous parameters (learning_rate, dropout) are explored efficiently
    in their natural scale (log, uniform) instead of a handful of grid points

Combined with early stopping, bad configs die after ~15 epochs while good
configs train for the full budget, making each trial count.

Search-space design
-------------------
  Common (from tune_*.yaml):
    learning_rate     loguniform [1e-4, 1e-2]
    weight_decay      0 | loguniform [1e-5, 1e-3]
    hidden_dropout    uniform [0.1, 0.5]
    MAX_ITEM_LIST_LEN choice [5, 10, 30]

  Model-specific (from model/*.yaml, auto-merged by Hydra):
    SASRec:  attn_dropout [0.1..0.4], num_layers {1,2,3}
    GRU4Rec: num_layers {1,2}
    BSARec:  num_layers {1,2,3}, num_heads {2,4,8}
    FAME:    num_heads {2,4,8}, num_experts {2,4,8}, num_layers {1,2,3}
    SIGMA:   num_layers {1,2}, state_size {16,32}, remaining_ratio, conv_kernel
    SRGNN:   step {1,2,3,4}
    Caser:   num_filters {8,16,32}, filter_sizes
    FENRec:  num_layers {1,2,3}, cl_weight, cl_temperature
    MSSR:    num_layers {1,2,3}, attribute_hidden_size {32,64,128}
    PAtt:    num_layers {1,2,3}, diversity_gamma

Usage
-----
    # Single model tuning
    python hyperopt_tune.py --config-name tune_ml model=sasrec gpu_id=0

    # More trials for complex model
    python hyperopt_tune.py --config-name tune_ml model=fame gpu_id=0 --max-evals 60

    # Override tuning budget
    python hyperopt_tune.py --config-name tune_rr model=gru4rec --tune-epochs 50

    # With wandb logging
    python hyperopt_tune.py --config-name tune_ml model=sasrec --log-wandb

    # Multiple baseline models on one dataset (shell runner)
    bash run/baseline/tune_by_dataset.sh --dataset movielens1m --models sasrec,gru4rec --gpus 0,1
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
#  Pre-import: CPU thread limits (must precede torch)
# ═══════════════════════════════════════════════════════════════════
import os
import subprocess
import sys

# Parse gpu_id from CLI before torch import and pin visible CUDA device.
_gpu_id = None
for _arg in sys.argv[1:]:
    if _arg.startswith("gpu_id="):
        _gpu_id = _arg.split("=", 1)[1]
        break
if _gpu_id is not None and _gpu_id != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)

for _v in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "4"

import torch
torch.set_num_threads(4)
from tqdm import tqdm

# RecBole 1.2.1 bugfix + custom model registration
import recbole_patch  # noqa: F401

import copy
import gc
import importlib
import io
import json
import gzip
import csv
import shutil
import time
import argparse
import warnings
import traceback
import signal
import atexit
import re
from itertools import product
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from hyperopt import fmin, tpe, rand, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_trainer
from recbole.utils import utils as _rbu
get_model = _rbu.get_model

from hydra_utils import load_hydra_config, configure_eval_sampling, enforce_v4_feature_mode
from omegaconf import OmegaConf

try:
    from models.FeaturedMoE_N3.feature_config import ALL_FEATURE_COLUMNS as _N3_ALL_FEATURE_COLUMNS
except Exception:
    _N3_ALL_FEATURE_COLUMNS = []

_FEATURE_AWARE_MOE_MODELS = {
    "featured_moe",
    "featuredmoe",
    "featured_moe_hgr",
    "featured_moe_hgr_v4",
    "featuredmoe_hgr",
    "featuredmoe_hgr_v4",
    "featuredmoe_hgrv4",
    "featured_moe_v2",
    "featuredmoe_v2",
    "featured_moe_v3",
    "featuredmoe_v3",
    "featured_moe_v4_distillation",
    "featuredmoe_v4_distillation",
    "featured_moe_n",
    "featuredmoe_n",
    "featured_moe_n2",
    "featuredmoe_n2",
    "featured_moe_n3",
    "featuredmoe_n3",
}

_FEATURED_MOE_V2_MODELS = {
    "featured_moe_hgr",
    "featured_moe_hgr_v4",
    "featuredmoe_hgr",
    "featuredmoe_hgr_v4",
    "featuredmoe_hgrv4",
    "featured_moe_v2",
    "featuredmoe_v2",
    "featured_moe_v3",
    "featuredmoe_v3",
    "featured_moe_v4_distillation",
    "featuredmoe_v4_distillation",
    "featured_moe_n",
    "featuredmoe_n",
    "featured_moe_n2",
    "featuredmoe_n2",
    "featured_moe_n3",
    "featuredmoe_n3",
}

_SIDEINFO_SEQ_MODELS = {
    "difsr",
    "fdsa",
}


def _config_get(config_obj, key, default=None):
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    try:
        return config_obj[key]
    except Exception:
        return getattr(config_obj, key, default)


def _build_item_counts_from_loader(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        return None
    inter_feat = getattr(dataset, "inter_feat", None)
    iid_field = getattr(dataset, "iid_field", None)
    n_items = int(getattr(dataset, "item_num", 0))
    if inter_feat is None or iid_field is None or n_items <= 0:
        return None
    item_ids = inter_feat[iid_field]
    if not torch.is_tensor(item_ids):
        item_ids = torch.as_tensor(item_ids)
    item_ids = item_ids.long().clamp(min=0)
    return torch.bincount(item_ids, minlength=n_items)[:n_items].cpu()


def _safe_len(obj) -> int | None:
    try:
        return int(len(obj))
    except Exception:
        return None


def _estimate_eval_target_count(data_loader) -> int | None:
    dataset = getattr(data_loader, "dataset", None)
    if dataset is None:
        return None
    inter_feat = getattr(dataset, "inter_feat", None)
    if inter_feat is not None:
        try:
            return int(len(inter_feat))
        except Exception:
            pass
    return _safe_len(dataset)


def _summarize_epoch_speed(epoch_trace_rows: list[dict]) -> tuple[float | None, float | None]:
    epoch_times = []
    for row in list(epoch_trace_rows or []):
        try:
            value = float(row.get("epoch_time_sec", 0.0) or 0.0)
        except Exception:
            value = 0.0
        if value > 0.0:
            epoch_times.append(value)
    if not epoch_times:
        return None, None
    avg_epoch_time_sec = float(sum(epoch_times) / len(epoch_times))
    avg_epoch_per_hour = 3600.0 / avg_epoch_time_sec if avg_epoch_time_sec > 0.0 else None
    return avg_epoch_time_sec, avg_epoch_per_hour


def _extract_cold_slice_metrics(special_metrics: dict | None) -> dict:
    out = {
        "count": 0,
        "hit@5": 0.0,
        "hit@10": 0.0,
        "hit@20": 0.0,
        "ndcg@5": 0.0,
        "ndcg@10": 0.0,
        "ndcg@20": 0.0,
        "mrr@5": 0.0,
        "mrr@10": 0.0,
        "mrr@20": 0.0,
    }
    if not isinstance(special_metrics, dict):
        return out
    try:
        slices = special_metrics.get("slices", {}) or {}
        pop = slices.get("target_popularity_abs", {}) or {}
        cold = pop.get("cold_0", {}) or {}
        out["count"] = int(cold.get("count", 0) or 0)
        out["hit@5"] = float(cold.get("hit@5", 0.0) or 0.0)
        out["hit@10"] = float(cold.get("hit@10", 0.0) or 0.0)
        out["mrr@20"] = float(cold.get("mrr@20", 0.0) or 0.0)
        out["mrr@5"] = float(cold.get("mrr@5", 0.0) or 0.0)
        out["mrr@10"] = float(cold.get("mrr@10", 0.0) or 0.0)
        out["hit@20"] = float(cold.get("hit@20", 0.0) or 0.0)
        out["ndcg@5"] = float(cold.get("ndcg@5", 0.0) or 0.0)
        out["ndcg@10"] = float(cold.get("ndcg@10", 0.0) or 0.0)
        out["ndcg@20"] = float(cold.get("ndcg@20", 0.0) or 0.0)
    except Exception:
        return out
    return out


def _extract_main_eval_seen_unseen_stats(trainer, split_name: str) -> dict:
    out = {"total_targets": 0, "seen_targets": 0, "unseen_targets": 0, "dropped_eval_rows": 0, "enabled": False}
    stats = getattr(trainer, "_main_eval_unseen_filter_stats", None)
    if not isinstance(stats, dict):
        return out
    rec = stats.get(str(split_name), None)
    if not isinstance(rec, dict):
        return out
    for k in ("total_targets", "seen_targets", "unseen_targets", "dropped_eval_rows"):
        try:
            out[k] = int(rec.get(k, 0) or 0)
        except Exception:
            out[k] = 0
    out["enabled"] = bool(rec.get("enabled", False))
    return out


def _normalize_model_name(raw) -> str:
    """Normalize model key used for family checks.

    Some launchers use tune aliases (e.g. featured_moe_n3_tune) that should share
    the same feature-aware behavior as the base model.
    """
    key = str(raw or "").strip().lower()
    for suffix in ("_tune", "-tune"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    return key


def _sync_model_dimensions(cfg_dict: dict) -> None:
    """Synchronize hidden/embedding size according to model family policy."""
    model_name = _normalize_model_name(cfg_dict.get("model", ""))

    if model_name in {"difsr", "tisasrec"}:
        if "hidden_size" in cfg_dict:
            hid = int(cfg_dict["hidden_size"])
            cfg_dict["embedding_size"] = hid
        elif "embedding_size" in cfg_dict:
            emb = int(cfg_dict["embedding_size"])
            cfg_dict["hidden_size"] = emb
        return

    # v2 uses embedding_size as primary; keep hidden_size aligned for RecBole internals.
    if model_name in _FEATURED_MOE_V2_MODELS:
        if "embedding_size" in cfg_dict:
            emb = int(cfg_dict["embedding_size"])
            cfg_dict["hidden_size"] = emb
            cfg_dict["inner_size"] = emb * 2
        elif "hidden_size" in cfg_dict:
            hid = int(cfg_dict["hidden_size"])
            cfg_dict["embedding_size"] = hid
            cfg_dict["inner_size"] = hid * 2
        return


def _ensure_feature_load_columns(cfg_dict: dict) -> None:
    """Guarantee engineered feature columns are loaded for feature-aware models.

    Some launch paths can accidentally keep a minimal `load_col.inter` and silently drop
    engineered feature fields, which then forces zero-filled feature fallbacks.
    """
    model_name = _normalize_model_name(cfg_dict.get("model", ""))
    is_moe = model_name in _FEATURE_AWARE_MOE_MODELS
    is_sideinfo = model_name in _SIDEINFO_SEQ_MODELS
    if not (is_moe or is_sideinfo):
        return

    feature_mode = str(cfg_dict.get("feature_mode", "")).strip().lower()
    data_path = str(cfg_dict.get("data_path", "")).strip().lower()
    uses_feature_added = ("feature_added" in data_path) or (
        feature_mode in {"full", "full_v2", "full_v3", "full_v4", "feature_added"}
    )
    if not uses_feature_added:
        return

    load_col = cfg_dict.get("load_col")
    if not isinstance(load_col, dict):
        load_col = {}

    inter_cols = load_col.get("inter", [])
    if not isinstance(inter_cols, list):
        inter_cols = []

    mandatory = ["session_id", "item_id", "timestamp"]
    merged = []
    feature_cols = list(_N3_ALL_FEATURE_COLUMNS or []) if is_moe else []
    for col in list(inter_cols) + mandatory + feature_cols:
        name = str(col).strip()
        if name and name not in merged:
            merged.append(name)

    load_col["inter"] = merged
    if is_moe:
        feature_cols = list(_N3_ALL_FEATURE_COLUMNS or [])
        for col in feature_cols:
            name = str(col).strip()
            if name and name not in load_col["inter"]:
                load_col["inter"].append(name)

    if is_sideinfo:
        selected = cfg_dict.get("selected_features")
        if isinstance(selected, str):
            selected = [selected]
        if not isinstance(selected, list):
            selected = []
        selected = [str(x).strip() for x in selected if str(x).strip()]
        if not selected:
            selected = ["category"]
            cfg_dict["selected_features"] = selected

        item_cols = load_col.get("item", [])
        if not isinstance(item_cols, list):
            item_cols = []
        merged_item = []
        for col in list(item_cols) + ["item_id"] + selected:
            name = str(col).strip()
            if name and name not in merged_item:
                merged_item.append(name)
        load_col["item"] = merged_item

    cfg_dict["load_col"] = load_col


def _model_uses_feature_inputs(cfg_dict: dict) -> bool:
    model_name = _normalize_model_name(cfg_dict.get("model", ""))
    if model_name not in _FEATURE_AWARE_MOE_MODELS:
        return False
    feature_mode = str(cfg_dict.get("feature_mode", "")).strip().lower()
    data_path = str(cfg_dict.get("data_path", "")).strip().lower()
    return ("feature_added" in data_path) or (feature_mode in {"full", "full_v2", "full_v3", "full_v4", "feature_added"})

    # Legacy behavior for v1/baseline models.
    if "hidden_size" in cfg_dict:
        hid = int(cfg_dict["hidden_size"])
        cfg_dict["embedding_size"] = hid
        cfg_dict["inner_size"] = hid * 2

# Optional log tee: keep console live, strip tqdm/ANSI from log files.
_log_path = os.environ.get("LOG_FILE", "").strip()
if _log_path:
    log_dir = os.path.dirname(_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    _ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def _clean_for_log(text: str) -> str:
        if "\r" in text and "\n" not in text:
            return ""
        cleaned = text.replace("\r", "")
        cleaned = _ansi_re.sub("", cleaned)
        stripped = cleaned.strip()
        # Drop tqdm/progress-bar snapshots from file logs.
        if ("it/s" in stripped and "%" in stripped and "|" in stripped) or ("GPU RAM:" in stripped):
            return ""
        if cleaned.lstrip().lower().startswith("wandb"):
            return ""
        return cleaned

    class _TeeStream(io.TextIOBase):
        def __init__(self, console, log_file):
            self.console = console
            self.log_file = log_file

        def write(self, text):
            self.console.write(text)
            cleaned = _clean_for_log(text)
            if cleaned:
                self.log_file.write(cleaned)
            return len(text)

        def flush(self):
            self.console.flush()
            self.log_file.flush()

        def isatty(self):
            return self.console.isatty()

    _log_fh = open(_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, _log_fh)
    sys.stderr = _TeeStream(sys.stderr, _log_fh)

# Keep tqdm bars transient on terminal (no persistent stacked bars).
import tqdm as tqdm_module
_original_tqdm = tqdm_module.tqdm


class TqdmWithoutLeave(_original_tqdm):
    def __init__(self, iterable=None, desc=None, total=None, leave=None, **kwargs):
        if leave is None:
            leave = False
        super().__init__(iterable=iterable, desc=desc, total=total, leave=leave, **kwargs)


tqdm_module.tqdm = TqdmWithoutLeave
tqdm_module.std.tqdm = TqdmWithoutLeave

_DATA_BUNDLE_CACHE = OrderedDict()
_TEMP_PATHS_TO_CLEAN: set[str] = set()

_TERMINATION_SIGNAL = None
_RUN_START_TS = datetime.now().isoformat(timespec="seconds")


def _sig_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except Exception:
        return str(signum)


def _on_termination_signal(signum, _frame):
    global _TERMINATION_SIGNAL
    _TERMINATION_SIGNAL = int(signum)
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[RUN_STATUS] TERMINATED signal={_sig_name(signum)}({signum}) pid={os.getpid()} ts={ts}")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    raise SystemExit(128 + int(signum))


def _on_process_exit():
    ts = datetime.now().isoformat(timespec="seconds")
    if _TERMINATION_SIGNAL is None:
        print(f"[RUN_STATUS] END status=normal pid={os.getpid()} start={_RUN_START_TS} end={ts}")
    else:
        print(
            f"[RUN_STATUS] END status=terminated signal={_sig_name(_TERMINATION_SIGNAL)}"
            f"({_TERMINATION_SIGNAL}) pid={os.getpid()} start={_RUN_START_TS} end={ts}"
        )


def _register_temp_path(path: Path | str) -> None:
    try:
        _TEMP_PATHS_TO_CLEAN.add(str(Path(path).resolve()))
    except Exception:
        _TEMP_PATHS_TO_CLEAN.add(str(path))


def _cleanup_temp_path(path: Path | str) -> None:
    target = str(path)
    try:
        Path(target).unlink(missing_ok=True)
    except Exception:
        pass
    _TEMP_PATHS_TO_CLEAN.discard(target)


def _cleanup_registered_temp_paths() -> None:
    for path in list(_TEMP_PATHS_TO_CLEAN):
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass
        finally:
            _TEMP_PATHS_TO_CLEAN.discard(path)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
    try:
        signal.signal(_sig, _on_termination_signal)
    except Exception:
        pass
atexit.register(_cleanup_registered_temp_paths)
atexit.register(_on_process_exit)
print(f"[RUN_STATUS] START pid={os.getpid()} ts={_RUN_START_TS}")


def _strip_dataset_suffix(name: str):
    """Extract dataset stem from RecBole filenames (supports dots in names)."""
    for suffix in (".train.inter", ".valid.inter", ".test.inter", ".inter", ".item"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _resolve_dataset_name_case(cfg_dict):
    """Resolve dataset name to actual on-disk casing."""
    dataset = str(cfg_dict.get("dataset", "")).strip()
    if not dataset:
        return dataset

    data_path = Path(str(cfg_dict.get("data_path", "."))).expanduser()
    direct_dir = data_path / dataset
    if direct_dir.exists():
        return dataset

    direct_files = [
        data_path / f"{dataset}.inter",
        data_path / f"{dataset}.item",
        data_path / f"{dataset}.train.inter",
    ]
    if any(p.exists() for p in direct_files):
        return dataset

    if not data_path.exists():
        return dataset

    target = dataset.lower()
    try:
        for p in data_path.iterdir():
            if p.is_dir() and p.name.lower() == target:
                return p.name

        stems = {}
        for p in data_path.iterdir():
            if not p.is_file():
                continue
            stem = _strip_dataset_suffix(p.name)
            if stem:
                stems.setdefault(stem.lower(), stem)
        if target in stems:
            return stems[target]
    except Exception:
        return dataset

    return dataset


def _canonical_dataset_name(name: str) -> str:
    """Canonical dataset name used when persisting result files."""
    raw = str(name or "").strip()
    if not raw:
        return raw
    mapping = {
        "kuairec0.3": "kuairec0.3",
        "kuairecsmall0.1": "KuaiRecSmall0.1",
        "kuaireclargestrictposv2": "KuaiRecLargeStrictPosV2",
        "kuaireclargestrictposv2_0.2": "KuaiRecLargeStrictPosV2_0.2",
        "lastfm0.03": "lastfm0.03",
    }
    return mapping.get(raw.lower(), raw)


def _is_large_dataset_cache_target(cfg_dict):
    if not bool(cfg_dict.get("large_dataset_cache_policy_enabled", True)):
        return False
    dataset = str(cfg_dict.get("dataset", "")).strip().lower()
    if not dataset:
        return False
    raw = cfg_dict.get(
        "large_dataset_cache_datasets",
        [
            "lastfm",
            "lastfm0.3",
            "kuairec",
            "kuairec0.3",
            "kuaireclargestrictposv2",
            "kuaireclargestrictposv2_0.2",
        ],
    )
    if isinstance(raw, str):
        ds_list = [x.strip().lower() for x in raw.split(",") if x.strip()]
    else:
        ds_list = [str(x).strip().lower() for x in raw if str(x).strip()]
    return dataset in set(ds_list)


def _get_large_dataset_cache_anchor_len(cfg_dict):
    try:
        return int(cfg_dict.get("large_dataset_cache_anchor_len", 50))
    except Exception:
        return 50


def _resolve_cache_source_tag(cfg_dict):
    """Resolve cache source tag as `feature_added` or `basic`."""
    data_path = str(cfg_dict.get("data_path", "")).lower()
    if "feature_added" in data_path:
        return "feature_added"
    if data_path.endswith("/basic") or "/basic/" in data_path:
        return "basic"

    feature_mode = str(cfg_dict.get("feature_mode", "")).lower()
    if feature_mode in ("full", "full_v2", "full_v3", "full_v4", "feature_added"):
        return "feature_added"
    if feature_mode == "basic":
        return "basic"

    model = _normalize_model_name(cfg_dict.get("model", ""))
    if model in _FEATURE_AWARE_MOE_MODELS:
        return "feature_added"
    return "basic"


def _resolve_cache_max_len(cfg_dict):
    max_len = cfg_dict.get("MAX_ITEM_LIST_LENGTH", None)
    if max_len is None:
        max_len = cfg_dict.get("max_seq_length", None)
    if max_len is None:
        max_len = 50
    return max_len


def _make_data_cache_key(cfg_dict: dict) -> str:
    return json.dumps(
        {
            "dataset": cfg_dict.get("dataset"),
            "source_tag": _resolve_cache_source_tag(cfg_dict),
            "MAX_ITEM_LIST_LENGTH": _resolve_cache_max_len(cfg_dict),
            "train_batch_size": _normalize_batch_size_value(cfg_dict.get("train_batch_size")),
            "eval_batch_size": _normalize_batch_size_value(cfg_dict.get("eval_batch_size")),
            "eval_args": cfg_dict.get("eval_args"),
            "data_path": cfg_dict.get("data_path"),
            "load_col": cfg_dict.get("load_col"),
            "field_separator": cfg_dict.get("field_separator"),
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _reset_loader_seed(loader, seed: int):
    try:
        if hasattr(loader, "generator") and loader.generator is not None:
            loader.generator.manual_seed(seed)
    except Exception:
        pass
    try:
        if (
            hasattr(loader, "sampler")
            and hasattr(loader.sampler, "generator")
            and loader.sampler.generator is not None
        ):
            loader.sampler.generator.manual_seed(seed)
    except Exception:
        pass


def _put_data_cache(key: str, bundle, max_entries: int):
    _DATA_BUNDLE_CACHE[key] = bundle
    _DATA_BUNDLE_CACHE.move_to_end(key, last=True)
    while len(_DATA_BUNDLE_CACHE) > max_entries:
        _DATA_BUNDLE_CACHE.popitem(last=False)


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def get_args():
    parser = argparse.ArgumentParser(
        description="Hyperopt (TPE) hyperparameter tuning for RecBole models",
        epilog=(
            "Examples:\n"
            "  python hyperopt_tune.py --config-name tune_ml model=sasrec gpu_id=0\n"
            "  python hyperopt_tune.py --config-name tune_ab model=fame --max-evals 60\n"
            "  python hyperopt_tune.py --config-name tune_rr model=gru4rec --tune-epochs 50\n"
            "  python hyperopt_tune.py --config-name tune_ml model=sigma --log-wandb\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-name", dest="config_name", default="config",
        help="Hydra config name (e.g. tune_ml, tune_ab)",
    )
    parser.add_argument(
        "--max-evals", dest="max_evals", type=int, default=50,
        help="Number of hyperopt trials (default: 50)",
    )
    parser.add_argument(
        "--max-run-hours", dest="max_run_hours", type=float, default=0.0,
        help="Optional wall-clock cap for one hyperopt run. After the current trial finishes, no new trials are started (default: disabled).",
    )
    parser.add_argument(
        "--oom-retry-limit", dest="oom_retry_limit", type=int, default=0,
        help="Retry the same trial on OOM by halving train/eval batch size. Value 2 means original -> 1/2 -> 1/4 (default: disabled).",
    )
    parser.add_argument(
        "--tune-epochs", dest="tune_epochs", type=int, default=None,
        help="Override epoch count for tuning (default: use config value)",
    )
    parser.add_argument(
        "--tune-patience", dest="tune_patience", type=int, default=None,
        help="Override early-stopping patience (default: use config value)",
    )
    parser.add_argument(
        "--log-wandb", dest="log_wandb", action="store_true", default=False,
        help="Enable wandb logging for this tuning session",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for TPE sampler (default: 42)",
    )
    parser.add_argument(
        "--search-algo",
        dest="search_algo",
        choices=("tpe", "random"),
        default="tpe",
        help="Search algorithm for hyperopt fmin (default: tpe)",
    )
    parser.add_argument(
        "--space-yaml", dest="space_yaml", type=str, default=None,
        help="Optional YAML file with {fixed, search} to override tuning space.",
    )
    parser.add_argument(
        "--run-group", dest="run_group", type=str, default=os.environ.get("RUN_GROUP", ""),
        help="Run group label for result partitioning (e.g. baseline, fmoe).",
    )
    parser.add_argument(
        "--run-axis", dest="run_axis", type=str, default=os.environ.get("RUN_AXIS", ""),
        help="Run axis label (e.g. train, hparam, layout, schedule).",
    )
    parser.add_argument(
        "--run-phase", dest="run_phase", type=str, default=os.environ.get("RUN_PHASE", ""),
        help="Optional pipeline phase label (e.g. P0, P1, ...).",
    )
    parser.add_argument(
        "--parent-result", dest="parent_result", type=str, default=os.environ.get("PARENT_RESULT", ""),
        help="Optional parent result JSON path.",
    )

    args, remaining = parser.parse_known_args()

    # Parse key=value overrides from remaining args
    overrides = []
    for tok in remaining:
        if tok == "--":
            continue
        if tok.startswith("max_evals="):
            args.max_evals = int(tok.split("=", 1)[1])
        elif tok.startswith("max_run_hours="):
            args.max_run_hours = float(tok.split("=", 1)[1])
        elif tok.startswith("oom_retry_limit="):
            args.oom_retry_limit = int(tok.split("=", 1)[1])
        elif tok.startswith("tune_epochs="):
            args.tune_epochs = int(tok.split("=", 1)[1])
        elif tok.startswith("tune_patience="):
            args.tune_patience = int(tok.split("=", 1)[1])
        elif tok in ("log_wandb=true", "log_wandb=True"):
            args.log_wandb = True
        elif tok in ("log_wandb=false", "log_wandb=False"):
            args.log_wandb = False
        elif tok.startswith("run_group="):
            args.run_group = tok.split("=", 1)[1]
        elif tok.startswith("run_axis="):
            args.run_axis = tok.split("=", 1)[1]
        elif tok.startswith("run_phase="):
            args.run_phase = tok.split("=", 1)[1]
        elif tok.startswith("search_algo="):
            raw_algo = tok.split("=", 1)[1].strip().lower()
            if raw_algo in {"tpe", "random"}:
                args.search_algo = raw_algo
        elif tok.startswith("parent_result="):
            args.parent_result = tok.split("=", 1)[1]
        else:
            overrides.append(tok)
    args.overrides = overrides
    return args


# ═══════════════════════════════════════════════════════════════════
#  Search-space builder
# ═══════════════════════════════════════════════════════════════════
def _space_type(key: str, type_overrides: dict | None = None) -> str:
    """Determine the hyperopt distribution type for a parameter name."""
    k = key.lower()
    if isinstance(type_overrides, dict):
        override = type_overrides.get(key)
        if override is None:
            override = type_overrides.get(k)
        if override is not None:
            ov = str(override).strip().lower()
            if ov in {"choice", "uniform", "loguniform", "loguniform_zero"}:
                return ov
    if k == "learning_rate":
        return "loguniform"
    if k == "weight_decay":
        return "loguniform_zero"  # 0 + loguniform
    if k == "balance_loss_lambda":
        return "loguniform"
    if "dropout" in k or k.endswith("_prob"):
        return "uniform"
    return "choice"


def split_search_params(search: dict) -> tuple[dict, dict]:
    """Split search block into tuned(len>1) and fixed(singleton/non-list) params."""
    tuned: dict = {}
    fixed: dict = {}
    for key, values in (search or {}).items():
        if isinstance(values, list):
            if len(values) == 0:
                continue
            if len(values) == 1:
                fixed[key] = values[0]
            else:
                tuned[key] = values
        else:
            fixed[key] = values
    return tuned, fixed


def build_hyperopt_space(search: dict, type_overrides: dict | None = None) -> dict:
    """Convert search dict to hyperopt space.

    Mapping:
      learning_rate       → hp.loguniform  (log scale)
      weight_decay        → hp.choice(0, hp.loguniform)
      *dropout* / *_prob  → hp.uniform     (continuous 0–1)
      balance_loss_lambda → hp.loguniform
      everything else     → hp.choice      (categorical / discrete)
    """
    space: dict = {}
    for key, values in search.items():
        if not isinstance(values, list) or len(values) <= 1:
            continue

        all_numeric = all(isinstance(v, (int, float)) for v in values)
        stype = _space_type(key, type_overrides=type_overrides)

        if stype == "loguniform" and all_numeric:
            lo, hi = float(min(values)), float(max(values))
            if lo <= 0 or lo == hi:
                space[key] = hp.choice(key, values)
            else:
                space[key] = hp.loguniform(key, np.log(lo), np.log(hi))

        elif stype == "loguniform_zero" and all_numeric:
            nonzero = sorted(float(v) for v in values if v > 0)
            has_zero = any(v == 0 for v in values)
            if has_zero and len(nonzero) >= 2:
                space[key] = hp.choice(key, [
                    0.0,
                    hp.loguniform(f"{key}_nz", np.log(nonzero[0]), np.log(nonzero[-1])),
                ])
            elif has_zero and len(nonzero) == 1:
                space[key] = hp.choice(key, [0.0, nonzero[0]])
            elif nonzero and nonzero[0] < nonzero[-1]:
                space[key] = hp.loguniform(key, np.log(nonzero[0]), np.log(nonzero[-1]))
            else:
                space[key] = hp.choice(key, values)

        elif stype == "uniform" and all_numeric:
            lo, hi = float(min(values)), float(max(values))
            if lo == hi:
                continue
            space[key] = hp.uniform(key, lo, hi)

        else:
            space[key] = hp.choice(key, values)

    return space


def _is_choice_only_space(tuned_search: dict, type_overrides: dict | None = None) -> bool:
    if not tuned_search:
        return False
    for key, values in tuned_search.items():
        if not isinstance(values, list) or len(values) <= 1:
            return False
        if _space_type(key, type_overrides=type_overrides) != "choice":
            return False
    return True


def _enumerate_choice_combos(tuned_search: dict, keys: list[str]) -> list[dict]:
    value_lists = [list(tuned_search[k]) for k in keys]
    combos: list[dict] = []
    for values in product(*value_lists):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos


def _count_choice_combos(tuned_search: dict, keys: list[str]) -> int:
    total = 1
    for key in keys:
        total *= len(list(tuned_search[key]))
    return total


def _choice_signature(params: dict, keys: list[str]) -> tuple[tuple[str, str], ...]:
    sig: list[tuple[str, str]] = []
    for key in keys:
        value = params.get(key)
        if isinstance(value, float):
            # Normalize float noise so repeated categorical choices hash identically.
            value = round(float(value), 16)
        try:
            token = json.dumps(value, ensure_ascii=True, sort_keys=True)
        except Exception:
            token = repr(value)
        sig.append((key, token))
    return tuple(sig)


def _pick_first_unused_choice_combo(
    combos: list[dict],
    used_signatures: set[tuple[tuple[str, str], ...]],
    keys: list[str],
) -> dict | None:
    for combo in combos:
        sig = _choice_signature(combo, keys)
        if sig not in used_signatures:
            return dict(combo)
    return None


def _hparam_group(key: str) -> str:
    k = str(key).lower()
    if (
        k in {"learning_rate", "weight_decay", "hidden_dropout_prob", "balance_loss_lambda"}
        or "batch_size" in k
        or k in {"epochs", "stopping_step"}
    ):
        return "optimization"
    if (
        k in {"hidden_size", "embedding_size", "num_heads", "num_layers", "inner_size"}
        or k.startswith("d_")
        or k == "expert_scale"
        or "dropout" in k
    ):
        return "model_core"
    if (
        "layout" in k
        or k.startswith("n_pre_")
        or k in {"n_pre_layer", "n_post_layer", "n_total_attn_layers"}
    ):
        return "layout"
    if (
        k.startswith("moe_top_k")
        or k in {"moe_top_k", "moe_top_k_policy", "moe_top_k_ratio", "moe_top_k_min"}
        or "router" in k
        or "routing" in k
        or "macro_session_pooling" in k
    ):
        return "routing_moe"
    if (
        k.startswith("fmoe_schedule")
        or "warmup" in k
        or "temperature_start" in k
        or "alpha_" in k
    ):
        return "schedule"
    if (
        "stage_merge" in k
        or "bundle_" in k
        or "parallel_stage_gate" in k
        or "hir_" in k
        or k.startswith("proto_")
        or "protox_" in k
        or "stage_weight_floor" in k
        or "stage_delta_scale" in k
    ):
        return "merge_hir"
    return "other"


def _print_grouped_params(
    title: str,
    params: dict,
    *,
    tuned: bool = False,
    type_overrides: dict | None = None,
) -> None:
    print(f"{title}: {len(params)}")
    if not params:
        print("  (none)")
        return

    order = ["optimization", "model_core", "layout", "routing_moe", "schedule", "merge_hir", "other"]
    bucket: dict[str, list[str]] = OrderedDict((g, []) for g in order)
    for key in sorted(params.keys()):
        bucket.setdefault(_hparam_group(key), []).append(key)

    for group in order:
        keys = bucket.get(group, [])
        if not keys:
            continue
        print(f"  [{group}]")
        for key in keys:
            if tuned:
                stype = _space_type(key, type_overrides=type_overrides)
                print(f"    {key:30s}  {str(params[key]):40s}  -> {stype}")
            else:
                print(f"    {key:30s}  {str(params[key]):40s}")


def _layout_catalog_from_cfg(cfg: dict) -> list | None:
    def _with_n2_controls(catalog_obj):
        if not isinstance(catalog_obj, list):
            return catalog_obj
        model_name = str(cfg.get("model", "")).lower()
        if "featured_moe_n2" not in model_name and "featuredmoe_n2" not in model_name:
            return catalog_obj
        catalog = copy.deepcopy(catalog_obj)
        known = {str(entry.get("id", "")).strip() for entry in catalog if isinstance(entry, dict)}
        if "L34" not in known:
            catalog.append(
                {
                    "id": "L34",
                    "execution": "serial",
                    "global_pre_layers": 1,
                    "global_post_layers": 0,
                    "stages": {
                        "macro": {"pass_layers": 0, "moe_blocks": 1},
                        "mid": {"pass_layers": 0, "moe_blocks": 0},
                        "micro": {"pass_layers": 0, "moe_blocks": 0},
                    },
                }
            )
        if "L35" not in known:
            catalog.append(
                {
                    "id": "L35",
                    "execution": "serial",
                    "global_pre_layers": 1,
                    "global_post_layers": 0,
                    "stages": {
                        "macro": {"pass_layers": 0, "moe_blocks": 1},
                        "mid": {"pass_layers": 0, "moe_blocks": 1},
                        "micro": {"pass_layers": 0, "moe_blocks": 0},
                    },
                }
            )
        return catalog

    catalog = cfg.get("fmoe_v2_layout_catalog")
    if isinstance(catalog, list):
        return _with_n2_controls(catalog)

    layout_execution = cfg.get("layout_execution")
    if isinstance(layout_execution, dict):
        catalog = layout_execution.get("fmoe_v2_layout_catalog")
        if isinstance(catalog, list):
            return _with_n2_controls(catalog)

    catalog = cfg.get("arch_layout_catalog")
    if isinstance(catalog, list):
        return _with_n2_controls(catalog)
    return None


def _safe_int(v) -> int | None:
    try:
        return int(v)
    except Exception:
        return None


def _format_layout_entry(layout_id, entry) -> str:
    if not isinstance(entry, dict):
        return f"id={layout_id} raw={entry}"

    exec_mode = entry.get("execution", "?")
    pre = entry.get("global_pre_layers", entry.get("n_pre_layer", "?"))
    post = entry.get("global_post_layers", entry.get("n_post_layer", "?"))
    total_layers = _safe_int(pre) or 0
    total_layers += _safe_int(post) or 0
    total_moe = 0
    stage_bits: list[str] = []
    stages = entry.get("stages")
    if isinstance(stages, dict):
        for stage_name in ("macro", "mid", "micro"):
            s = stages.get(stage_name)
            if isinstance(s, dict):
                p = s.get("pass_layers", "?")
                m = s.get("moe_blocks", "?")
                stage_bits.append(f"{stage_name}(pass={p},moe={m})")
                total_layers += (_safe_int(p) or 0) + (_safe_int(m) or 0)
                total_moe += _safe_int(m) or 0
    stage_desc = " | ".join(stage_bits) if stage_bits else "stages=?"
    return (
        f"id={layout_id} exec={exec_mode} pre={pre} post={post} "
        f"total_layers={total_layers} total_moe={total_moe} | {stage_desc}"
    )


def _print_layout_details(cfg: dict, tuned_search: dict, fixed_search: dict) -> None:
    layout_ids: list[int] = []
    for key in ("fmoe_v2_layout_id", "arch_layout_id"):
        if key in fixed_search:
            i = _safe_int(fixed_search[key])
            if i is not None and i not in layout_ids:
                layout_ids.append(i)
        if key in tuned_search and isinstance(tuned_search[key], list):
            for v in tuned_search[key]:
                i = _safe_int(v)
                if i is not None and i not in layout_ids:
                    layout_ids.append(i)

    if not layout_ids:
        return

    layout_ids = sorted(layout_ids)
    catalog = _layout_catalog_from_cfg(cfg)
    exec_fixed = fixed_search.get("fmoe_stage_execution_mode", cfg.get("fmoe_stage_execution_mode", ""))
    exec_tuned = tuned_search.get("fmoe_stage_execution_mode")

    print("Layout details:")
    print(f"  execution(fixed)={exec_fixed}")
    if exec_tuned is not None:
        print(f"  execution(candidates)={exec_tuned}")

    if not isinstance(catalog, list) or not catalog:
        print(f"  layout_ids={layout_ids} (catalog not found in cfg)")
        return

    for lid in layout_ids:
        if 0 <= lid < len(catalog):
            entry = catalog[lid]
            print(f"  {_format_layout_entry(lid, entry)}")
        else:
            print(f"  id={lid} (out-of-range; catalog_size={len(catalog)})")


def _deep_update(dst: dict, src: dict):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value


def load_space_yaml(path: str) -> tuple[dict, dict]:
    """Load stage space YAML.

    Supported forms:
      1) {fixed: {...}, search: {...}}
      2) legacy direct mapping -> treated as `search`.
    """
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = (Path.cwd() / raw).resolve()
    if not raw.exists():
        raise FileNotFoundError(f"space yaml not found: {raw}")

    with raw.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("space yaml root must be a mapping")

    if "fixed" in data or "search" in data:
        fixed = data.get("fixed", {}) or {}
        search = data.get("search", {}) or {}
    else:
        fixed = {}
        search = data

    if not isinstance(fixed, dict):
        raise ValueError("space yaml 'fixed' must be a mapping")
    if not isinstance(search, dict):
        raise ValueError("space yaml 'search' must be a mapping")

    return fixed, search


def _set_nested_value(cfg: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _dropout_target_keys(model_name: str) -> list[str]:
    model_key = str(model_name or "").strip().lower()
    targets = ["hidden_dropout_prob"]
    if model_key in {
        "sasrec",
        "fdsa",
        "bsarec",
        "fenrec",
        "patt",
        "duorec",
        "tisasrec",
        "fearec",
        "difsr",
    }:
        targets.append("attn_dropout_prob")
    return targets


def _apply_runtime_param(cfg: dict, key: str, value):
    key_str = str(key).strip()
    if not key_str:
        return

    if key_str.lower() == "dropout_ratio":
        for target_key in _dropout_target_keys(cfg.get("model", "")):
            _set_nested_value(cfg, target_key, float(value))
        return

    _set_nested_value(cfg, key_str, value)


def _exception_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    cursor: BaseException | None = exc
    seen: set[int] = set()
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        text = str(cursor).strip()
        if text:
            messages.append(text)
        cursor = cursor.__cause__ or cursor.__context__
    return messages


def _is_oom_error(exc: BaseException) -> bool:
    for text in _exception_messages(exc):
        lowered = text.lower()
        if (
            "out of memory" in lowered
            or "cuda error: out of memory" in lowered
            or "cuda out of memory" in lowered
            or "cublas_status_alloc_failed" in lowered
            or "cudnn_status_alloc_failed" in lowered
            or "hip out of memory" in lowered
        ):
            return True
    return False


def _normalize_batch_size_value(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _current_runtime_batch_sizes(cfg: dict) -> tuple[int | None, int | None]:
    train_bs = _normalize_batch_size_value(cfg.get("train_batch_size"))
    eval_bs = _normalize_batch_size_value(cfg.get("eval_batch_size"))
    return train_bs, eval_bs


def _halve_batch_sizes_for_retry(cfg: dict) -> dict | None:
    train_bs, eval_bs = _current_runtime_batch_sizes(cfg)
    anchor = train_bs if train_bs is not None else eval_bs
    if anchor is None:
        return None
    if train_bs is None:
        train_bs = anchor
    if eval_bs is None:
        eval_bs = anchor
    new_train_bs = max(1, int(train_bs) // 2)
    new_eval_bs = max(1, int(eval_bs) // 2)
    if new_train_bs >= int(train_bs) and new_eval_bs >= int(eval_bs):
        return None
    cfg["train_batch_size"] = int(new_train_bs)
    cfg["eval_batch_size"] = int(new_eval_bs)
    return {
        "train_before": int(train_bs),
        "eval_before": int(eval_bs),
        "train_after": int(new_train_bs),
        "eval_after": int(new_eval_bs),
    }


def _safe_slug(raw: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(raw or "").strip())
    text = text.strip("._-")
    return text or "na"


def _best_stage_temp_path(cfg_dict: dict) -> Path:
    tmp_dir = Path(__file__).resolve().parent / "run" / "artifacts" / "tmp" / "best_stage"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataset = _safe_slug(_canonical_dataset_name(cfg_dict.get("dataset", "")))
    model = _safe_slug(str(cfg_dict.get("model", "")))
    phase = _safe_slug(str(cfg_dict.get("run_phase", "")))
    return tmp_dir / f"pid{os.getpid()}_{dataset}_{model}_{phase}.pth"


def _dataset_tag(raw: str) -> str:
    key = str(raw or "").strip().lower()
    mapping = {
        "movielens1m": "ML1",
        "retail_rocket": "ReR",
        "retailrocket": "ReR",
        "foursquare": "FSQ",
        "lastfm0.3": "LF3",
        "lastfm0.03": "LF03",
        "kuairec0.3": "KU3",
        "kuairecsmall0.1": "KU01",
        "kuaireclargestrictposv2": "KUL",
        "kuaireclargestrictposv2_0.2": "KUL02",
        "amazon_beauty": "AMA",
        "amazonbeauty": "AMA",
    }
    if key in mapping:
        return mapping[key]
    slim = re.sub(r"[^a-z0-9]+", "", key).upper()
    if len(slim) >= 3:
        return slim[:3]
    if slim:
        return slim.ljust(3, "X")
    return "UNK"


def _model_tag(raw: str) -> str:
    key = str(raw or "").strip().lower()
    if "featuredmoe_protox" in key or "featured_moe_protox" in key:
        return "FMoEProtoX"
    if "featured_moe_individual" in key or "featuredmoe_individual" in key:
        return "FMoEIndividual"
    if "featuredmoe_hir2" in key or "featured_moe_hir2" in key:
        return "FMoEHiR2"
    if "featured_moe_v2_hir" in key or "featuredmoe_v2_hir" in key:
        return "FMoEv2HiR"
    if "featured_moe_v3" in key or "featuredmoe_v3" in key:
        return "FMoEv3"
    if "featured_moe_v4_distillation" in key or "featuredmoe_v4_distillation" in key:
        return "FMoEv4D"
    if (
        "featured_moe_hgr_v4" in key
        or "featuredmoe_hgr_v4" in key
        or "featuredmoe_hgrv4" in key
        or "featured_moe_hgrv4" in key
    ):
        return "FMoEHGRv4"
    if (
        "featured_moe_hgr_v3" in key
        or "featuredmoe_hgr_v3" in key
        or "featuredmoe_hgrv3" in key
        or "featured_moe_hgrv3" in key
    ):
        return "FMoEHGRv3"
    if "featured_moe_hgr" in key or "featuredmoe_hgr" in key:
        return "FMoEHGR"
    if "featuredmoe_hir" in key or "featured_moe_hir" in key:
        return "FMoEHiR"
    if "featured_moe_v2" in key or "featuredmoe_v2" in key:
        return "FMoEv2"
    if "featured_moe_n2" in key or "featuredmoe_n2" in key:
        return "FMoEN2"
    if "featured_moe_n3" in key or "featuredmoe_n3" in key:
        return "FMoEN3"
    if "featuredmoe" in key or "featured_moe" in key:
        return "FMoE"
    return _safe_slug(raw)


def _phase_bucket(raw: str) -> str:
    return _safe_slug(str(raw or "").split("_", 1)[0]) or "PNA"


def _logs_root() -> Path:
    raw = str(os.environ.get("RUN_LOGS_DIR", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "run" / "artifacts" / "logs"


def _logging_root() -> Path:
    raw = str(os.environ.get("HYPEROPT_LOGGING_DIR", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parent / "run" / "artifacts" / "logging"


def _use_unified_logging_layout() -> bool:
    raw = str(os.environ.get("FMOE_UNIFIED_LOGGING_LAYOUT", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _artifact_mirror_paths(
    result_file: Path,
    dataset: str,
    model: str,
    run_group: str,
    run_axis: str,
    run_phase: str,
) -> dict[str, Path]:
    if _use_unified_logging_layout():
        run_dir = result_file.parent
        return {
            "normal_result": result_file,
            "special_result": run_dir / "special_metrics.json",
            "special_log": run_dir / "special_log.json",
        }

    dataset_dir = _safe_slug(_canonical_dataset_name(dataset))
    model_tag = _model_tag(model)
    axis_tag = _safe_slug(run_axis or "axis")
    phase_bucket = _phase_bucket(run_phase)
    results_group_dir = result_file.parent
    normal_dir = results_group_dir / "normal" / axis_tag / phase_bucket / dataset_dir / model_tag
    special_results_dir = results_group_dir / "special" / axis_tag / phase_bucket / dataset_dir / model_tag
    special_logs_dir = (
        _logs_root()
        / _safe_slug(run_group or "misc")
        / "special"
        / axis_tag
        / phase_bucket
        / dataset_dir
        / model_tag
    )
    special_name = f"{result_file.stem}_special_metrics.json"
    return {
        "normal_result": normal_dir / result_file.name,
        "special_result": special_results_dir / special_name,
        "special_log": special_logs_dir / special_name,
    }


def _write_json_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_ser) + "\n", encoding="utf-8")


def _write_json_gz_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_ser)
        f.write("\n")


def _write_csv_gz_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _ser(v) for k, v in row.items()})


def _write_csv_file(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _ser(v) for k, v in row.items()})


def _write_csv_with_schema(path: Path, rows: list[dict], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = list(fieldnames or [])
    if not fn:
        fn = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fn)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _ser(v) for k, v in row.items()})


def _copy_if_exists(src: str | Path | None, dst: Path) -> str:
    src_path = Path(str(src or "")).expanduser()
    if not src_path.exists() or not src_path.is_file():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() == dst.resolve():
        return str(dst.resolve())
    # In legacy layout, some artifacts are stored as *.json.gz / *.csv.gz.
    # If bundle targets plain extensions, transparently decompress so editors can open them directly.
    if src_path.suffix == ".gz" and dst.suffix in {".json", ".csv"}:
        with gzip.open(src_path, "rb") as src_f:
            dst.write_bytes(src_f.read())
        return str(dst.resolve())
    shutil.copy2(src_path, dst)
    return str(dst.resolve())


def _bundle_run_dir(*, path: Path, model: str, dataset: str, run_group: str, run_phase: str) -> Path:
    phase_bucket = _phase_bucket(run_phase)
    dataset_bucket = _safe_slug(_canonical_dataset_name(dataset))
    group_bucket = _safe_slug(run_group or _model_tag(model) or "model")
    if path.name == "result.json":
        run_id = _safe_slug(path.parent.name)
    else:
        run_id = _safe_slug(path.stem)
    return _logging_root() / group_bucket / dataset_bucket / phase_bucket / run_id


def _write_bundle_summary_md(path: Path, payload: dict) -> None:
    lines = [
        "# Run Logging Summary",
        "",
        f"- model: {payload.get('model', '')}",
        f"- dataset: {payload.get('dataset', '')}",
        f"- run_group: {payload.get('run_group', '')}",
        f"- run_axis: {payload.get('run_axis', '')}",
        f"- run_phase: {payload.get('run_phase', '')}",
        f"- best_mrr@20: {payload.get('best_mrr@20', '')}",
        f"- test_mrr@20: {payload.get('test_mrr@20', '')}",
        f"- test_hr@10: {payload.get('test_hr@10', '')}",
        "",
        "## Artifacts",
        "",
        f"- result_json: {payload.get('result_json', '')}",
        f"- special_metrics: {payload.get('special_metrics_json', '')}",
        f"- special_log: {payload.get('special_log_json', '')}",
        f"- diag_dir: {payload.get('diag_dir', '')}",
        f"- diag_meta: {payload.get('diag_meta_json', '')}",
        f"- diag_tier_a_final: {payload.get('diag_tier_a_final_csv', '')}",
        f"- diag_tier_b_internal: {payload.get('diag_tier_b_internal_csv', '')}",
        f"- diag_viz_manifest: {payload.get('diag_viz_manifest_json', '')}",
        f"- diag_viz_feature_pca: {payload.get('diag_viz_feature_pca_csv_gz', '')}",
        f"- diag_viz_router_input_pca: {payload.get('diag_viz_router_input_pca_csv_gz', '')}",
        f"- diag_viz_group_feature_pca: {payload.get('diag_viz_group_feature_pca_csv_gz', '')}",
        f"- diag_raw_trial_summary: {payload.get('diag_raw_trial_summary_csv', '')}",
        f"- diag_raw_best_valid: {payload.get('diag_raw_best_valid_json', '')}",
        f"- diag_raw_test: {payload.get('diag_raw_test_json', '')}",
        f"- diag_raw_epoch_trace: {payload.get('diag_raw_epoch_trace_csv', '')}",
        f"- feature_ablation: {payload.get('feature_ablation_json', '')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_overview_rows(overview_json_path: str | Path | None) -> list[dict]:
    p = Path(str(overview_json_path or ""))
    if not p.exists() or not p.is_file():
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = []
    for row in list((payload or {}).get("stages", []) or []):
        rows.append(
            {
                "stage": row.get("stage", ""),
                "n_eff": row.get("n_eff", ""),
                "cv_usage": row.get("cv_usage", ""),
                "top1_max_frac": row.get("top1_max_frac", ""),
                "entropy_mean": row.get("entropy_mean", ""),
                "route_jitter_adjacent": row.get("route_jitter_adjacent", ""),
                "route_consistency_knn_js": row.get("route_consistency_knn_js", ""),
                "route_consistency_knn_score": row.get("route_consistency_knn_score", ""),
                "route_consistency_group_knn_js": row.get("route_consistency_group_knn_js", ""),
                "route_consistency_group_knn_score": row.get("route_consistency_group_knn_score", ""),
                "route_consistency_intra_group_knn_mean_js": row.get("route_consistency_intra_group_knn_mean_js", ""),
                "route_consistency_intra_group_knn_mean_score": row.get("route_consistency_intra_group_knn_mean_score", ""),
                "route_consistency_feature_group_knn_tempo_js": row.get("route_consistency_feature_group_knn_tempo_js", ""),
                "route_consistency_feature_group_knn_tempo_score": row.get("route_consistency_feature_group_knn_tempo_score", ""),
                "route_consistency_feature_group_knn_focus_js": row.get("route_consistency_feature_group_knn_focus_js", ""),
                "route_consistency_feature_group_knn_focus_score": row.get("route_consistency_feature_group_knn_focus_score", ""),
                "route_consistency_feature_group_knn_memory_js": row.get("route_consistency_feature_group_knn_memory_js", ""),
                "route_consistency_feature_group_knn_memory_score": row.get("route_consistency_feature_group_knn_memory_score", ""),
                "route_consistency_feature_group_knn_exposure_js": row.get("route_consistency_feature_group_knn_exposure_js", ""),
                "route_consistency_feature_group_knn_exposure_score": row.get("route_consistency_feature_group_knn_exposure_score", ""),
                "route_consistency_feature_group_knn_mean_score": row.get("route_consistency_feature_group_knn_mean_score", ""),
                "family_top_expert_mean_share": row.get("family_top_expert_mean_share", ""),
            }
        )
    return rows


def _write_logging_bundle(
    *,
    result_path: Path,
    data_payload: dict,
    model: str,
    dataset: str,
    run_group: str,
    run_phase: str,
) -> dict:
    bundle_dir = _bundle_run_dir(
        path=result_path,
        model=model,
        dataset=dataset,
        run_group=run_group,
        run_phase=run_phase,
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    def _copy_diag(src_key: str, rel_path: str) -> str:
        return _copy_if_exists(data_payload.get(src_key), bundle_dir / rel_path)

    out = {
        "bundle_dir": str(bundle_dir.resolve()),
        "result_json": _copy_if_exists(result_path, bundle_dir / "result.json"),
        "special_metrics_json": _copy_if_exists(data_payload.get("special_result_file"), bundle_dir / "special_metrics.json"),
        "special_log_json": _copy_if_exists(data_payload.get("special_log_file"), bundle_dir / "special_log.json"),
        "diag_dir": str((bundle_dir / "diag").resolve()),
        "diag_meta_json": _copy_diag("diag_meta_file", "diag/meta.json"),
        "diag_tier_a_final_csv": _copy_diag("diag_tier_a_final_file", "diag/tier_a_final/final_metrics.csv"),
        "diag_tier_b_internal_csv": _copy_diag("diag_tier_b_internal_file", "diag/tier_b_internal/internal_metrics.csv"),
        "diag_viz_manifest_json": _copy_diag("diag_viz_manifest_file", "diag/tier_c_viz/viz_manifest.json"),
        "diag_viz_feature_pca_csv_gz": _copy_diag("diag_viz_feature_pca_file", "diag/tier_c_viz/viz_feature_pca.csv.gz"),
        "diag_viz_router_input_pca_csv_gz": _copy_diag("diag_viz_router_input_pca_file", "diag/tier_c_viz/viz_router_input_pca.csv.gz"),
        "diag_viz_group_feature_pca_csv_gz": _copy_diag("diag_viz_group_feature_pca_file", "diag/tier_c_viz/viz_group_feature_pca.csv.gz"),
        "diag_raw_trial_summary_csv": _copy_diag("diag_raw_trial_summary_file", "diag/raw/trial_summary.csv"),
        "diag_raw_best_valid_json": _copy_diag("diag_raw_best_valid_file", "diag/raw/best_valid_diag.json"),
        "diag_raw_best_valid_overview_json": _copy_diag("diag_raw_best_valid_overview_file", "diag/raw/best_valid_overview.json"),
        "diag_raw_best_valid_overview_md": _copy_diag("diag_raw_best_valid_overview_md_file", "diag/raw/best_valid_overview.md"),
        "diag_raw_early_valid_json": _copy_diag("diag_raw_early_valid_file", "diag/raw/early_valid_diag.json"),
        "diag_raw_test_json": _copy_diag("diag_raw_test_file", "diag/raw/test_diag.json"),
        "diag_raw_collapse_json": _copy_diag("diag_raw_collapse_file", "diag/raw/collapse_diag.json"),
        "diag_raw_epoch_trace_csv": _copy_diag("diag_raw_epoch_trace_file", "diag/raw/epoch_trace.csv"),
        "feature_ablation_json": _copy_if_exists(data_payload.get("feature_ablation_file"), bundle_dir / "feature_ablation.json"),
    }

    overview_rows = _extract_overview_rows(out.get("diag_raw_best_valid_overview_json"))
    if overview_rows:
        _write_csv_file(bundle_dir / "diag" / "raw" / "overview_table.csv", overview_rows)
        out["diag_overview_table_csv"] = str((bundle_dir / "diag" / "raw" / "overview_table.csv").resolve())
    else:
        out["diag_overview_table_csv"] = ""

    summary_payload = {
        "model": data_payload.get("model", model),
        "dataset": data_payload.get("dataset", _canonical_dataset_name(dataset)),
        "run_group": data_payload.get("run_group", run_group),
        "run_axis": data_payload.get("run_axis", ""),
        "run_phase": data_payload.get("run_phase", run_phase),
        "timestamp": data_payload.get("timestamp", ""),
        "max_evals": data_payload.get("max_evals", ""),
        "n_completed": data_payload.get("n_completed", ""),
        "best_mrr@20": data_payload.get("best_mrr@20", ""),
        "test_mrr@20": data_payload.get("test_mrr@20", ""),
        "test_hr@10": data_payload.get("test_hr@10", ""),
        "best_params": data_payload.get("best_params", {}),
        "best_valid_result": data_payload.get("best_valid_result", {}),
        "test_result": data_payload.get("test_result", {}),
        **out,
    }

    trials_rows = []
    for t in list(data_payload.get("trials", []) or []):
        trials_rows.append(
            {
                "trial": t.get("trial", ""),
                "status": t.get("status", ""),
                "mrr@20": t.get("mrr@20", ""),
                "test_mrr@20": t.get("test_mrr@20", ""),
                "test_hr@10": t.get("test_hr@10", ""),
                "epochs_run": t.get("epochs_run", ""),
                "early_stopped": t.get("early_stopped", ""),
            }
        )
    if trials_rows:
        _write_csv_file(bundle_dir / "trials_brief.csv", trials_rows)
        out["trials_brief_csv"] = str((bundle_dir / "trials_brief.csv").resolve())
    else:
        out["trials_brief_csv"] = ""

    analysis_card = {
        "model": summary_payload["model"],
        "dataset": summary_payload["dataset"],
        "run_axis": summary_payload["run_axis"],
        "run_phase": summary_payload["run_phase"],
        "best_mrr@20": summary_payload["best_mrr@20"],
        "test_mrr@20": summary_payload["test_mrr@20"],
        "test_hr@10": summary_payload["test_hr@10"],
        "best_params": summary_payload["best_params"],
        "stage_overview": overview_rows,
        "feature_ablation_available": bool(out.get("feature_ablation_json")),
        "diag_available": bool(out.get("diag_meta_json") or out.get("diag_tier_a_final_csv")),
        "special_available": bool(out.get("special_metrics_json")),
    }
    _write_json_file(bundle_dir / "analysis_card.json", analysis_card)
    out["analysis_card_json"] = str((bundle_dir / "analysis_card.json").resolve())

    summary_payload.update(
        {
            "trials_brief_csv": out.get("trials_brief_csv", ""),
            "analysis_card_json": out.get("analysis_card_json", ""),
            "diag_overview_table_csv": out.get("diag_overview_table_csv", ""),
        }
    )
    _write_json_file(bundle_dir / "run_summary.json", summary_payload)
    _write_bundle_summary_md(bundle_dir / "run_summary.md", summary_payload)
    out["run_summary_json"] = str((bundle_dir / "run_summary.json").resolve())
    out["run_summary_md"] = str((bundle_dir / "run_summary.md").resolve())
    return out


def _diag_artifact_paths(
    result_file: Path,
    dataset: str,
    model: str,
    run_group: str,
    run_axis: str,
    run_phase: str,
) -> dict[str, Path]:
    del dataset, model, run_group, run_axis, run_phase
    diag_root = result_file.parent / "diag"
    return {
        "meta": diag_root / "meta.json",
        "tier_a_final": diag_root / "tier_a_final" / "final_metrics.csv",
        "tier_b_internal": diag_root / "tier_b_internal" / "internal_metrics.csv",
        "tier_c_manifest": diag_root / "tier_c_viz" / "viz_manifest.json",
        "tier_c_feature_pca": diag_root / "tier_c_viz" / "viz_feature_pca.csv.gz",
        "tier_c_router_input_pca": diag_root / "tier_c_viz" / "viz_router_input_pca.csv.gz",
        "tier_c_group_feature_pca": diag_root / "tier_c_viz" / "viz_group_feature_pca.csv.gz",
        "raw_trial_summary": diag_root / "raw" / "trial_summary.csv",
        "raw_best_valid_diag": diag_root / "raw" / "best_valid_diag.json",
        "raw_best_valid_overview": diag_root / "raw" / "best_valid_overview.json",
        "raw_best_valid_overview_md": diag_root / "raw" / "best_valid_overview.md",
        "raw_early_valid_diag": diag_root / "raw" / "early_valid_diag.json",
        "raw_test_diag": diag_root / "raw" / "test_diag.json",
        "raw_collapse_diag": diag_root / "raw" / "collapse_diag.json",
        "raw_epoch_trace": diag_root / "raw" / "epoch_trace.csv",
    }


def _diag_scalar_metrics(diag_payload: dict | None) -> dict:
    if not isinstance(diag_payload, dict):
        return {}
    return dict(diag_payload.get("scalar_metrics", {}) or {})


def _diag_tier_rows(diag_payload: dict | None, *, tier_key: str) -> list[dict]:
    if not isinstance(diag_payload, dict):
        return []
    tiers = dict(diag_payload.get("diag_tiers", {}) or {})
    rows = list(tiers.get(tier_key, []) or [])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(dict(row))
    return out


def _diag_viz_rows(diag_payload: dict | None, *, key: str) -> list[dict]:
    if not isinstance(diag_payload, dict):
        return []
    tiers = dict(diag_payload.get("diag_tiers", {}) or {})
    tier_c = dict(tiers.get("tier_c_viz", {}) or {})
    rows = list(tier_c.get(key, []) or [])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(dict(row))
    return out


def _write_csv_gz_with_schema(path: Path, rows: list[dict], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = list(fieldnames or [])
    if not fn:
        fn = sorted({key for row in rows for key in row.keys()})
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fn)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _ser(v) for k, v in row.items()})


def _metric_name_token(name: str) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "group"
    out = []
    prev_us = False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    token = "".join(out).strip("_")
    return token or "group"


def _build_diag_overview_payload(diag_payload: dict | None) -> dict:
    if not isinstance(diag_payload, dict):
        return {"split": "", "feature_mode": "", "stages": []}

    stages = []
    stage_metrics = dict(diag_payload.get("stage_metrics", {}) or {})
    for stage_key in sorted(stage_metrics.keys()):
        st = dict(stage_metrics.get(stage_key, {}) or {})
        spec = dict(st.get("specialization_summary", {}) or {})
        intra = dict(st.get("route_consistency_intra_group_knn", {}) or {})
        feat_grp = dict(st.get("route_consistency_feature_group_knn", {}) or {})
        feat_group_names = list(feat_grp.get("group_names", []) or [])
        feat_js = list(feat_grp.get("js_by_group", []) or [])
        feat_score = list(feat_grp.get("score_by_group", []) or [])
        feat_map = {}
        for idx, group_name in enumerate(feat_group_names):
            token = _metric_name_token(group_name)
            js_val = float(feat_js[idx]) if idx < len(feat_js) else 0.0
            score_val = float(feat_score[idx]) if idx < len(feat_score) else 0.0
            feat_map[f"route_consistency_feature_group_knn_{token}_js"] = js_val
            feat_map[f"route_consistency_feature_group_knn_{token}_score"] = score_val
        for token in ("tempo", "focus", "memory", "exposure"):
            feat_map.setdefault(f"route_consistency_feature_group_knn_{token}_js", 0.0)
            feat_map.setdefault(f"route_consistency_feature_group_knn_{token}_score", 0.0)
        stages.append(
            {
                "stage": stage_key,
                "n_eff": float(st.get("n_eff", 0.0) or 0.0),
                "cv_usage": float(st.get("cv_usage", 0.0) or 0.0),
                "top1_max_frac": float(st.get("top1_max_frac", 0.0) or 0.0),
                "entropy_mean": float(st.get("entropy_mean", 0.0) or 0.0),
                "route_jitter_adjacent": float(st.get("route_jitter_adjacent", 0.0) or 0.0),
                "route_consistency_knn_js": float(st.get("route_consistency_knn_js", 0.0) or 0.0),
                "route_consistency_knn_score": float(st.get("route_consistency_knn_score", 0.0) or 0.0),
                "route_consistency_group_knn_js": float(st.get("route_consistency_group_knn_js", 0.0) or 0.0),
                "route_consistency_group_knn_score": float(st.get("route_consistency_group_knn_score", 0.0) or 0.0),
                "route_consistency_intra_group_knn_mean_js": float(intra.get("mean_js", 0.0) or 0.0),
                "route_consistency_intra_group_knn_mean_score": float(intra.get("mean_score", 0.0) or 0.0),
                "route_consistency_feature_group_knn_mean_score": float(
                    ((st.get("route_consistency_feature_group_knn") or {}).get("mean_score", 0.0) or 0.0)
                ),
                "family_top_expert_mean_share": float(spec.get("mean_top_expert_share", 0.0) or 0.0),
                **feat_map,
            }
        )

    return {
        "split": str(diag_payload.get("split", "")),
        "feature_mode": str(diag_payload.get("feature_mode", "")),
        "stages": stages,
    }


def _diag_overview_markdown(overview: dict) -> str:
    rows = list(overview.get("stages", []) or [])
    lines = [
        "# Diagnostic Overview",
        "",
        f"- split: {overview.get('split', '')}",
        f"- feature_mode: {overview.get('feature_mode', '')}",
        "",
        "## Base Metrics",
        "",
        "| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {n_eff:.4f} | {cv_usage:.4f} | {top1_max_frac:.4f} | {entropy_mean:.4f} | {route_jitter_adjacent:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## KNN Core",
            "",
            "| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {stage} | {route_consistency_knn_js:.4f} | {route_consistency_knn_score:.4f} | {route_consistency_group_knn_js:.4f} | {route_consistency_group_knn_score:.4f} | {route_consistency_intra_group_knn_mean_js:.4f} | {route_consistency_intra_group_knn_mean_score:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Feature-Group KNN",
            "",
            "| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {stage} | {route_consistency_feature_group_knn_tempo_js:.4f} | {route_consistency_feature_group_knn_tempo_score:.4f} | {route_consistency_feature_group_knn_focus_js:.4f} | {route_consistency_feature_group_knn_focus_score:.4f} | {route_consistency_feature_group_knn_memory_js:.4f} | {route_consistency_feature_group_knn_memory_score:.4f} | {route_consistency_feature_group_knn_exposure_js:.4f} | {route_consistency_feature_group_knn_exposure_score:.4f} | {family_top_expert_mean_share:.4f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _top1_share_from_diag(diag_payload: dict | None) -> dict[str, list[float]]:
    if not isinstance(diag_payload, dict):
        return {}
    out = {}
    for stage_key, stage_payload in dict(diag_payload.get("stage_metrics", {}) or {}).items():
        counts = stage_payload.get("top1_count", []) or []
        total = float(sum(float(v) for v in counts))
        if total <= 0:
            continue
        out[stage_key] = [float(v) / total for v in counts]
    return out


def _compute_route_change_metric(normal_diag: dict | None, perturbed_diag: dict | None) -> float:
    normal = _top1_share_from_diag(normal_diag)
    perturbed = _top1_share_from_diag(perturbed_diag)
    scores = []
    for stage_key, probs in normal.items():
        alt = perturbed.get(stage_key)
        if not alt or len(alt) != len(probs):
            continue
        scores.append(sum(abs(float(a) - float(b)) for a, b in zip(probs, alt)) / 2.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _select_sparse_epoch_trace(rows: list[dict], best_epoch: int | None) -> list[dict]:
    if not rows:
        return []
    keep_epochs = {1, 3, 10, rows[-1].get("epoch")}
    if best_epoch is not None:
        keep_epochs.add(int(best_epoch))
    return [row for row in rows if int(row.get("epoch", 0) or 0) in keep_epochs]


def _build_special_metrics_payload(
    *,
    normalized_trials: list[dict],
    model: str,
    dataset: str,
    run_group: str,
    run_axis: str,
    run_phase: str,
    source_result_file: Path,
) -> dict | None:
    rows = []
    for trial in normalized_trials:
        valid_special = trial.get("valid_special_metrics")
        test_special = trial.get("test_special_metrics")
        if valid_special is None and test_special is None:
            continue
        rows.append(
            {
                "trial": trial.get("trial"),
                "status": trial.get("status"),
                "mrr@20": _ser(trial.get("mrr@20", 0.0)),
                "best_hr@10": _ser(trial.get("best_hr@10", 0.0)),
                "test_mrr@20": _ser(trial.get("test_mrr@20", 0.0)),
                "test_hr@10": _ser(trial.get("test_hr@10", 0.0)),
                "params": trial.get("params", {}),
                "valid_special_metrics": valid_special or {},
                "test_special_metrics": test_special or {},
                "early_valid_special_metrics": trial.get("early_valid_special_metrics") or {},
                "valid_main_eval_filter": trial.get("valid_main_eval_filter") or {},
                "test_main_eval_filter": trial.get("test_main_eval_filter") or {},
                "valid_cold_target_metrics": trial.get("valid_cold_target_metrics") or {},
                "test_cold_target_metrics": trial.get("test_cold_target_metrics") or {},
            }
        )
    if not rows:
        return None

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best_row = max(ok_rows or rows, key=lambda row: float(row.get("mrr@20", 0.0) or 0.0))
    return {
        "model": model,
        "dataset": _canonical_dataset_name(dataset),
        "dataset_raw": dataset,
        "run_group": run_group,
        "run_axis": run_axis,
        "run_phase": run_phase,
        "source_result_file": str(source_result_file.resolve()),
        "best_trial": {
            "trial": best_row.get("trial"),
            "mrr@20": _ser(best_row.get("mrr@20", 0.0)),
            "best_hr@10": _ser(best_row.get("best_hr@10", 0.0)),
            "test_mrr@20": _ser(best_row.get("test_mrr@20", 0.0)),
            "test_hr@10": _ser(best_row.get("test_hr@10", 0.0)),
            "params": best_row.get("params", {}),
        },
        "best_valid_special_metrics": best_row.get("valid_special_metrics", {}) or {},
        "early_valid_special_metrics": best_row.get("early_valid_special_metrics", {}) or {},
        "test_special_metrics": best_row.get("test_special_metrics", {}) or {},
        "best_valid_main_eval_filter": best_row.get("valid_main_eval_filter", {}) or {},
        "test_main_eval_filter": best_row.get("test_main_eval_filter", {}) or {},
        "best_valid_cold_target_metrics": best_row.get("valid_cold_target_metrics", {}) or {},
        "test_cold_target_metrics": best_row.get("test_cold_target_metrics", {}) or {},
        "trials": rows,
    }


def _save_best_stage(model, path: Path) -> None:
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, path)
    del cpu_state
    gc.collect()


def _reapply_cli_overrides(cfg: dict, overrides: list[str]):
    """Re-apply explicit key=value CLI overrides after --space-yaml merge."""
    allowed_top_keys = {
        "MAX_ITEM_LIST_LENGTH",
        "moe_top_k",
        "moe_top_k_policy",
        "moe_top_k_ratio",
        "moe_top_k_min",
        "router_impl",
        "router_impl_by_stage",
        "rule_router",
        "rule_router.variant",
        "rule_router.n_bins",
        "rule_router.feature_per_expert",
        "rule_router.custom_stage_feature_map",
        "rule_router.expert_bias",
        "expert_scale",
        "learning_rate",
        "weight_decay",
        "hidden_dropout_prob",
        "dropout_ratio",
        "balance_loss_lambda",
        "search_space_type_overrides",
        "num_heads",
        "arch_layout_id",
        "fmoe_v2_layout_id",
        "fmoe_stage_execution_mode",
        "fmoe_v2_parallel_stage_gate_top_k",
        "fmoe_v2_parallel_stage_gate_temperature",
        "fmoe_v2_stage_merge_aux_enable",
        "fmoe_v2_stage_merge_aux_lambda_scale",
        "num_layers",
        "n_pre_layer",
        "n_pre_macro",
        "n_pre_mid",
        "n_pre_micro",
        "n_post_layer",
        "stage_moe_repeat_after_pre_layer",
        "stage_merge_mode",
        "group_router_mode",
        "group_top_k",
        "bundle_top_k",
        "parallel_stage_gate_top_k",
        "hir_use_bundle_aux_loss",
        "hir_bundle_aux_lambda_scale",
        "fmoe_schedule_enable",
        "alpha_warmup_until",
        "alpha_warmup_start",
        "alpha_warmup_end",
        "temperature_warmup_until",
        "mid_router_temperature_start",
        "micro_router_temperature_start",
        "moe_top_k_start",
        "moe_top_k_warmup_until",
        "eval_every",
        "feature_encoder_mode",
        "feature_encoder_sinusoidal_features",
        "feature_encoder_sinusoidal_n_freqs",
        "rule_bias_scale",
        "fmoe_special_logging",
        "expert_hidden_by_stage",
        "expert_depth_by_stage",
        "stage_inter_layer_style",
        "arch_state_tag",
        "arch_variant_tag",
        "wave",
        "combo_id",
        "combo_desc",
        "pair_id",
    }
    for token in overrides:
        if "=" not in token:
            continue
        key, raw_val = token.split("=", 1)
        key = key.strip()
        if not key:
            continue
        # Ignore Hydra append/remove operators here; they are already handled by compose.
        if key[0] in "+~":
            continue
        # Avoid re-applying Hydra group selectors (e.g. model=featured_moe_tune).
        if not key.startswith("search.") and key not in allowed_top_keys:
            continue
        try:
            parsed_val = yaml.safe_load(raw_val)
        except Exception:
            parsed_val = raw_val
        _set_nested_value(cfg, key, parsed_val)


def _extract_featured_moe_arch(model, cfg: dict) -> dict:
    """Collect FeaturedMoE architecture/depth fields for trial logging."""
    keys = (
        "arch_layout_id",
        "fmoe_v2_layout_id",
        "fmoe_stage_execution_mode",
        "router_impl",
        "router_impl_by_stage",
        "rule_router_cfg",
        "stage_inter_layer_style",
        "moe_block_variant",
        "router_group_feature_mode",
        "router_use_hidden",
        "router_use_feature",
        "router_feature_proj_dim",
        "router_feature_proj_layers",
        "router_feature_scale",
        "router_hidden_scale",
        "router_group_feature_scale",
        "rule_agreement_lambda",
        "group_coverage_lambda",
        "lr_scheduler_type",
        "lr_scheduler_warmup_ratio",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_plateau_factor",
        "lr_scheduler_plateau_patience",
        "arch_state_tag",
        "arch_variant_tag",
        "n_pre_layer",
        "n_pre_macro",
        "n_pre_mid",
        "n_pre_micro",
        "n_post_layer",
        "n_total_attn_layers",
        "num_layers",
    )
    out = {}
    for key in keys:
        val = getattr(model, key, None)
        if val is None and key in cfg:
            val = cfg.get(key)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                continue
            val = val.item()
        out[key] = _ser(val)
    return out


def _extract_context_fixed(cfg: dict) -> dict:
    """Collect non-search fixed context fields for result metadata."""
    keys = (
        "arch_layout_catalog",
        "arch_layout_id",
        "fmoe_v2_layout_catalog",
        "fmoe_v2_layout_id",
        "fmoe_stage_execution_mode",
        "router_impl",
        "router_impl_by_stage",
        "rule_router",
        "stage_inter_layer_style",
        "moe_block_variant",
        "router_group_feature_mode",
        "router_use_hidden",
        "router_use_feature",
        "router_feature_proj_dim",
        "router_feature_proj_layers",
        "router_feature_scale",
        "router_hidden_scale",
        "router_group_feature_scale",
        "rule_agreement_lambda",
        "group_coverage_lambda",
        "lr_scheduler_type",
        "lr_scheduler_warmup_ratio",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_plateau_factor",
        "lr_scheduler_plateau_patience",
        "arch_state_tag",
        "arch_variant_tag",
        "num_layers",
        "n_pre_layer",
        "n_pre_macro",
        "n_pre_mid",
        "n_pre_micro",
        "n_post_layer",
        "stage_moe_repeat_after_pre_layer",
        "stage_merge_mode",
        "group_router_mode",
        "group_top_k",
        "bundle_top_k",
        "parallel_stage_gate_top_k",
        "fmoe_v2_parallel_stage_gate_top_k",
        "fmoe_v2_parallel_stage_gate_temperature",
        "fmoe_v2_stage_merge_aux_enable",
        "fmoe_v2_stage_merge_aux_lambda_scale",
        "hir_use_bundle_aux_loss",
        "hir_bundle_aux_lambda_scale",
        "fmoe_schedule_enable",
        "alpha_warmup_until",
        "temperature_warmup_until",
        "moe_top_k",
        "moe_top_k_warmup_until",
        "search_space_type_overrides",
        "wave",
        "combo_id",
        "combo_desc",
        "pair_id",
    )
    out = {}
    for key in keys:
        if key not in cfg:
            continue
        out[key] = _ser(cfg.get(key))
    return out


# ═══════════════════════════════════════════════════════════════════
#  Runtime acceleration
# ═══════════════════════════════════════════════════════════════════
def configure_runtime_acceleration(cfg_dict: dict):
    """Apply optional runtime acceleration toggles."""
    use_gpu = bool(cfg_dict.get("use_gpu", True))
    if not use_gpu or not torch.cuda.is_available():
        return

    enable_tf32 = bool(cfg_dict.get("enable_tf32", True))
    try:
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    except Exception:
        pass

    enable_cudnn_benchmark = cfg_dict.get("enable_cudnn_benchmark", None)
    if enable_cudnn_benchmark is None:
        enable_cudnn_benchmark = not bool(cfg_dict.get("reproducibility", True))
    try:
        torch.backends.cudnn.benchmark = bool(enable_cudnn_benchmark)
    except Exception:
        pass


def configure_data_cache(cfg_dict: dict):
    """Enable deterministic dataset/dataloader cache paths for faster tuning startup."""
    if not bool(cfg_dict.get("enable_data_cache", True)):
        return

    dataset = str(cfg_dict.get("dataset", "dataset"))
    source_tag = _resolve_cache_source_tag(cfg_dict)
    max_len = _resolve_cache_max_len(cfg_dict)
    eval_args = cfg_dict.get("eval_args", {}) or {}
    split_cfg = eval_args.get("split", {}) if isinstance(eval_args, dict) else {}
    has_pre_split = bool(cfg_dict.get("benchmark_filename"))
    is_session_split = (isinstance(split_cfg, dict) and "RS" in split_cfg) or has_pre_split
    split_tag = "session" if is_session_split else "inter"
    anchor_len = _get_large_dataset_cache_anchor_len(cfg_dict)
    if _is_large_dataset_cache_target(cfg_dict) and int(max_len) != int(anchor_len):
        # Keep experiment MAX_ITEM_LIST_LENGTH unchanged, but skip disk cache writes.
        cfg_dict["save_dataset"] = False
        cfg_dict["save_dataloaders"] = False
        cfg_dict["cache_dataloaders"] = False
        cfg_dict["enable_session_split_cache"] = False
        cfg_dict["dataset_save_path"] = ""
        cfg_dict["dataloaders_save_path"] = ""
        print(
            f"[CachePolicy] Disk cache OFF for {dataset} len={max_len} "
            f"(anchor len={anchor_len} only)"
        )
        return

    cache_root = Path(cfg_dict.get("data_cache_dir", "saved/recbole_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)

    # Shared cache key by dataset + data-source + split + max length.
    dataset_tag = f"{dataset}_{source_tag}_{split_tag}_len{max_len}"
    cache_dir = cache_root / dataset_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    dl_file = cache_dir / f"{dataset}-for-dataloader.pth"

    cfg_dict["save_dataset"] = True
    cfg_dict["save_dataloaders"] = bool(cfg_dict.get("cache_dataloaders", False))
    cfg_dict["checkpoint_dir"] = str(cache_dir)
    # Use RecBole default cache filenames under checkpoint_dir.
    cfg_dict["dataset_save_path"] = ""
    # Shared dataloader cache path (cache_dataloaders is disabled by default).
    cfg_dict["dataloaders_save_path"] = str(dl_file)


def _trial_artifact_stage_path(cfg_dict: dict, *, trial_num: int, tag: str) -> Path:
    tmp_dir = Path(__file__).resolve().parent / "run" / "artifacts" / "tmp" / "trial_stage"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataset = _safe_slug(_canonical_dataset_name(cfg_dict.get("dataset", "")))
    model = _safe_slug(str(cfg_dict.get("model", "")))
    phase = _safe_slug(str(cfg_dict.get("run_phase", "")))
    tag_slug = _safe_slug(tag)
    return tmp_dir / f"pid{os.getpid()}_{dataset}_{model}_{phase}_T{int(trial_num):03d}_{tag_slug}.pth"


def _copy_stage_file(src: Path, dst: Path | str) -> str:
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    _register_temp_path(target)
    return str(target.resolve())


def _export_stage_file(src: Path, dst: Path | str) -> str:
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return str(target.resolve())


def _run_trainer_eval(trainer, data_loader, *, split_name: str, show_progress: bool, collect_special: bool, collect_diag: bool):
    eval_special = None
    eval_diag = None
    if hasattr(trainer, "_main_eval_unseen_filter_stats"):
        try:
            trainer._main_eval_unseen_filter_stats[str(split_name)] = {
                "total_targets": 0,
                "seen_targets": 0,
                "unseen_targets": 0,
                "dropped_eval_rows": 0,
                "enabled": False,
            }
        except Exception:
            pass
    if collect_special:
        from recbole_patch import begin_special_eval
        begin_special_eval(trainer, data_loader, split_name=split_name)
    if collect_diag:
        from recbole_patch import begin_diagnostic_eval
        begin_diagnostic_eval(trainer, split_name=split_name)
    eval_result = trainer._valid_epoch(data_loader, show_progress=show_progress)
    if collect_special:
        from recbole_patch import end_special_eval
        eval_special = end_special_eval(trainer)
    if collect_diag:
        from recbole_patch import end_diagnostic_eval
        eval_diag = end_diagnostic_eval(trainer)
    if isinstance(eval_result, tuple):
        eval_result = next((x for x in eval_result if isinstance(x, dict)), eval_result[0])
    eval_filter = _extract_main_eval_seen_unseen_stats(trainer, str(split_name))
    return eval_result, eval_special, eval_diag, eval_filter


def _transfer_enabled(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _transfer_cfg(cfg_dict: dict) -> dict:
    raw = cfg_dict.get("transfer", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _transfer_group_router_name(source_architecture: str) -> str:
    arch = str(source_architecture or "").strip().upper()
    if arch == "A10":
        return "router_b"
    if arch == "A12":
        return "router_e"
    raise ValueError(
        f"Unsupported transfer source_architecture={source_architecture!r}. "
        "StageA currently supports A10/A12 only."
    )


def _transfer_prefixes(*, mode: str, source_architecture: str) -> list[str]:
    def _stage_prefix(stage: str) -> str:
        return f"stage_executor.stage_blocks.{stage}."

    def _all_router_prefixes(stage: str) -> list[str]:
        base = _stage_prefix(stage)
        return [
            base + "router_a.",
            base + "router_b.",
            base + "router_c.",
            base + "router_d.",
            base + "router_e.",
        ]

    macro_prefix = _stage_prefix("macro")
    group_router = _transfer_group_router_name(source_architecture)

    if mode == "macro_feature_encoder":
        return [macro_prefix + "feature_encoder."]
    if mode == "macro_group_router":
        return [macro_prefix + group_router + "."]
    if mode in {"macro_group_router_all", "macro_full_router"}:
        return _all_router_prefixes("macro")
    if mode == "all_stage_group_router":
        return [
            _stage_prefix("macro") + group_router + ".",
            _stage_prefix("mid") + group_router + ".",
            _stage_prefix("micro") + group_router + ".",
        ]
    if mode == "all_stage_full_router":
        return _all_router_prefixes("macro") + _all_router_prefixes("mid") + _all_router_prefixes("micro")
    if mode == "all_stage_feature_encoder":
        return [
            _stage_prefix("macro") + "feature_encoder.",
            _stage_prefix("mid") + "feature_encoder.",
            _stage_prefix("micro") + "feature_encoder.",
        ]
    if mode == "macro_encoder_all":
        return [
            macro_prefix + "feature_encoder.",
            macro_prefix + group_router + ".",
            macro_prefix + "router_d.",
        ]
    return []


def _apply_transfer_initialization(model, cfg_dict: dict) -> dict:
    transfer = _transfer_cfg(cfg_dict)
    enabled = _transfer_enabled(transfer.get("enable", False))
    mode = str(transfer.get("mode", "none") or "none").strip().lower()
    if not enabled or mode in {"", "none"}:
        return {"enabled": False, "mode": "none"}

    source_checkpoint = str(transfer.get("source_checkpoint", "") or "").strip()
    if not source_checkpoint:
        raise RuntimeError(f"Transfer is enabled but transfer.source_checkpoint is empty (mode={mode})")
    checkpoint_path = Path(source_checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Transfer checkpoint not found: {checkpoint_path}")

    source_architecture = str(transfer.get("source_architecture", "") or "").strip().upper()
    strict_shape = _transfer_enabled(transfer.get("strict_shape", False))
    source_state = torch.load(checkpoint_path, map_location="cpu")
    target_state = model.state_dict()

    if mode == "full_model":
        if strict_shape:
            missing, unexpected = model.load_state_dict(source_state, strict=True)
            # strict=True should already raise on mismatch; keep a stable report shape.
            return {
                "enabled": True,
                "mode": mode,
                "loaded_tensors": len(source_state),
                "skipped_shape": 0,
                "skipped_missing": 0,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "source_checkpoint": str(checkpoint_path.resolve()),
            }
        matched = {
            key: value
            for key, value in source_state.items()
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
        }
        skipped_shape = sum(
            1
            for key, value in source_state.items()
            if key in target_state and tuple(target_state[key].shape) != tuple(value.shape)
        )
        skipped_missing = sum(1 for key in source_state if key not in target_state)
        merged_state = model.state_dict()
        merged_state.update(matched)
        load_result = model.load_state_dict(merged_state, strict=False)
        report = {
            "enabled": True,
            "mode": mode,
            "loaded_tensors": len(matched),
            "skipped_shape": skipped_shape,
            "skipped_missing": skipped_missing,
            "missing_keys": len(getattr(load_result, "missing_keys", []) or []),
            "unexpected_keys": len(getattr(load_result, "unexpected_keys", []) or []),
            "source_checkpoint": str(checkpoint_path.resolve()),
        }
        print(
            "[transfer-init] "
            f"mode={mode} source_architecture={source_architecture or '-'} "
            f"loaded={report['loaded_tensors']} skipped_shape={report['skipped_shape']} "
            f"skipped_missing={report['skipped_missing']} source={report['source_checkpoint']}"
        )
        return report

    prefixes = _transfer_prefixes(mode=mode, source_architecture=source_architecture)
    if not prefixes:
        raise RuntimeError(
            f"Unsupported transfer.mode={mode!r}. "
            "Expected one of: none, macro_feature_encoder, macro_group_router, macro_group_router_all, "
            "macro_full_router, all_stage_group_router, all_stage_full_router, all_stage_feature_encoder, "
            "macro_encoder_all, full_model."
        )

    matched: dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_missing = 0
    considered = 0
    for key, value in source_state.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            continue
        considered += 1
        target_tensor = target_state.get(key)
        if target_tensor is None:
            skipped_missing += 1
            continue
        if tuple(target_tensor.shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        matched[key] = value

    merged_state = model.state_dict()
    merged_state.update(matched)
    load_result = model.load_state_dict(merged_state, strict=False)
    report = {
        "enabled": True,
        "mode": mode,
        "loaded_tensors": len(matched),
        "considered_tensors": considered,
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
        "missing_keys": len(getattr(load_result, "missing_keys", []) or []),
        "unexpected_keys": len(getattr(load_result, "unexpected_keys", []) or []),
        "source_checkpoint": str(checkpoint_path.resolve()),
        "prefixes": prefixes,
    }
    print(
        "[transfer-init] "
        f"mode={mode} source_architecture={source_architecture or '-'} "
        f"loaded={report['loaded_tensors']}/{report['considered_tensors']} "
        f"skipped_shape={report['skipped_shape']} skipped_missing={report['skipped_missing']} "
        f"source={report['source_checkpoint']}"
    )
    if not matched:
        print(f"[transfer-init][warn] no compatible tensors loaded for mode={mode} prefixes={prefixes}")
    return report


def _collect_feature_ablation_metrics(
    *,
    trainer,
    model,
    valid_data,
    reference_mrr20: float,
    reference_diag: dict | None,
    show_progress: bool,
    enable_global: bool,
    enable_family: bool,
    split_prefix: str,
) -> dict:
    metrics = {}
    if not hasattr(model, "set_feature_ablation_mode"):
        return metrics

    def _eval(mode: str, *, family: str | None = None, split_name: str):
        model.set_feature_ablation_mode(mode, family)
        return _run_trainer_eval(
            trainer,
            valid_data,
            split_name=split_name,
            show_progress=show_progress,
            collect_special=False,
            collect_diag=True,
        )

    if enable_global:
        zero_result, _zero_special, valid_zero_diag, _zero_filter = _eval("zero", split_name=f"{split_prefix}_zero")
        shuffle_result, _shuffle_special, valid_shuffle_diag, _shuffle_filter = _eval("shuffle", split_name=f"{split_prefix}_shuffle")
        metrics.update(
            {
                "feature_zero_delta_mrr": float(reference_mrr20 - float((zero_result or {}).get("mrr@20", 0.0) or 0.0)),
                "feature_shuffle_delta_mrr": float(reference_mrr20 - float((shuffle_result or {}).get("mrr@20", 0.0) or 0.0)),
                "route_change_under_feature_shuffle": _compute_route_change_metric(reference_diag, valid_shuffle_diag),
                "route_change_under_feature_zero": _compute_route_change_metric(reference_diag, valid_zero_diag),
            }
        )

    if enable_family:
        for family_name in list(getattr(model, "feature_family_names", []) or []):
            family_slug = _safe_slug(str(family_name).lower())
            fam_zero_result, _fam_zero_special, fam_zero_diag, _fam_zero_filter = _eval(
                "zero",
                family=family_name,
                split_name=f"{split_prefix}_{family_slug}_zero",
            )
            fam_shuffle_result, _fam_shuffle_special, fam_shuffle_diag, _fam_shuffle_filter = _eval(
                "shuffle",
                family=family_name,
                split_name=f"{split_prefix}_{family_slug}_shuffle",
            )
            metrics.update(
                {
                    f"family_{family_slug}_zero_delta_mrr": float(
                        reference_mrr20 - float((fam_zero_result or {}).get("mrr@20", 0.0) or 0.0)
                    ),
                    f"family_{family_slug}_shuffle_delta_mrr": float(
                        reference_mrr20 - float((fam_shuffle_result or {}).get("mrr@20", 0.0) or 0.0)
                    ),
                    f"family_{family_slug}_route_change_under_zero": _compute_route_change_metric(reference_diag, fam_zero_diag),
                    f"family_{family_slug}_route_change_under_shuffle": _compute_route_change_metric(reference_diag, fam_shuffle_diag),
                }
            )

    model.set_feature_ablation_mode("none")
    return metrics


def _build_eval_runtime(cfg_dict: dict):
    cfg = copy.deepcopy(cfg_dict)
    cfg["log_wandb"] = False
    model_name = _normalize_model_name(cfg.get("model", ""))
    if model_name in _FEATURE_AWARE_MOE_MODELS and "fmoe_special_logging" not in cfg:
        cfg["fmoe_special_logging"] = True
    if model_name in _FEATURE_AWARE_MOE_MODELS and "fmoe_diag_logging" not in cfg:
        cfg["fmoe_diag_logging"] = True
    if (
        model_name in _FEATURE_AWARE_MOE_MODELS
        and "fmoe_feature_ablation_logging" not in cfg
        and _model_uses_feature_inputs(cfg)
    ):
        cfg["fmoe_feature_ablation_logging"] = bool(cfg.get("fmoe_special_logging", True) and cfg.get("fmoe_diag_logging", True))
    cfg["show_progress"] = bool(cfg.get("show_progress", True))
    for drop in ("search", "search_stages", "search_strategy", "max_search"):
        cfg.pop(drop, None)
    cfg["valid_metric"] = "MRR@20"
    _sync_model_dimensions(cfg)
    _ensure_feature_load_columns(cfg)
    if cfg.get("loss_type", "CE").upper() == "CE":
        cfg["train_neg_sample_args"] = None
    configure_runtime_acceleration(cfg)
    configure_data_cache(cfg)

    _argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        config = Config(model=cfg["model"], dataset=cfg["dataset"], config_dict=cfg)
    finally:
        sys.argv = _argv

    warnings.filterwarnings("ignore", category=FutureWarning)
    init_seed(config["seed"], config["reproducibility"])

    cache_key = _make_data_cache_key(cfg)
    use_mem_cache = bool(cfg.get("in_memory_data_cache", True))
    max_mem_cache = max(1, int(cfg.get("max_in_memory_data_cache", 1)))
    if use_mem_cache and cache_key in _DATA_BUNDLE_CACHE:
        dataset, train_data, valid_data, test_data = _DATA_BUNDLE_CACHE[cache_key]
        _DATA_BUNDLE_CACHE.move_to_end(cache_key, last=True)
        _reset_loader_seed(train_data, int(config["seed"]))
        _reset_loader_seed(valid_data, int(config["seed"]))
        _reset_loader_seed(test_data, int(config["seed"]))
    else:
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        if use_mem_cache:
            _put_data_cache(
                cache_key,
                (dataset, train_data, valid_data, test_data),
                max_entries=max_mem_cache,
            )

    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])
    trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_cls(config, model)
    setattr(trainer, "_disable_patch_logging", True)
    trainer._fmoe_special_item_counts_train = _build_item_counts_from_loader(train_data)
    trainer._main_eval_unseen_filter_stats = {}
    trainer._fmoe_special_item_counts_train = _build_item_counts_from_loader(train_data)
    trainer._main_eval_unseen_filter_stats = {}
    return cfg, config, dataset, train_data, valid_data, test_data, model, trainer


def _collect_deferred_combo_best_artifacts(
    *,
    base_cfg: dict,
    sampled_params: dict,
    checkpoint_path: str,
    probe_checkpoint_path: str,
    trial_num: int,
) -> dict:
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {}

    cfg_trial = copy.deepcopy(base_cfg)
    cfg_trial.pop("search", None)
    for k, v in sampled_params.items():
        _apply_runtime_param(cfg_trial, str(k), _ser(v))

    runtime = _build_eval_runtime(cfg_trial)
    cfg_eval, config, dataset, train_data, valid_data, test_data, model, trainer = runtime
    show_progress = bool(cfg_eval.get("show_progress", True))
    model_name = _normalize_model_name(cfg_eval.get("model", ""))
    special_logging_enabled = model_name in _FEATURE_AWARE_MOE_MODELS and bool(
        _config_get(config, "fmoe_special_logging", cfg_eval.get("fmoe_special_logging", True))
    )
    diag_logging_enabled = model_name in _FEATURE_AWARE_MOE_MODELS and bool(
        _config_get(config, "fmoe_diag_logging", cfg_eval.get("fmoe_diag_logging", True))
    )
    feature_ablation_logging_enabled = model_name in _FEATURE_AWARE_MOE_MODELS and bool(
        _config_get(config, "fmoe_feature_ablation_logging", cfg_eval.get("fmoe_feature_ablation_logging", False))
    )
    family_ablation_logging_enabled = model_name in _FEATURE_AWARE_MOE_MODELS and bool(
        _config_get(
            config,
            "fmoe_feature_family_ablation_logging",
            cfg_eval.get("fmoe_feature_family_ablation_logging", feature_ablation_logging_enabled),
        )
    )

    try:
        best_state = torch.load(checkpoint_path, map_location="cpu")
        trainer.model.load_state_dict(best_state)
        del best_state
        gc.collect()

        trainer.model.set_feature_ablation_mode("none")
        best_valid_result, best_valid_special_metrics, best_valid_diag, best_valid_filter = _run_trainer_eval(
            trainer,
            valid_data,
            split_name="best_valid",
            show_progress=show_progress,
            collect_special=special_logging_enabled,
            collect_diag=diag_logging_enabled,
        )
        test_result, test_special_metrics, test_diag, test_filter = _run_trainer_eval(
            trainer,
            test_data,
            split_name="test",
            show_progress=show_progress,
            collect_special=special_logging_enabled,
            collect_diag=diag_logging_enabled,
        )

        feature_ablation_metrics = {}
        if feature_ablation_logging_enabled:
            feature_ablation_metrics = _collect_feature_ablation_metrics(
                trainer=trainer,
                model=trainer.model,
                valid_data=valid_data,
                reference_mrr20=float((best_valid_result or {}).get("mrr@20", 0.0) or 0.0),
                reference_diag=best_valid_diag,
                show_progress=show_progress,
                enable_global=True,
                enable_family=family_ablation_logging_enabled,
                split_prefix="valid",
            )
            if isinstance(best_valid_diag, dict):
                best_valid_diag["feature_ablation"] = feature_ablation_metrics

        early_valid_result = {}
        early_valid_special_metrics = {}
        early_valid_diag = {}
        if probe_checkpoint_path and Path(probe_checkpoint_path).exists():
            probe_state = torch.load(probe_checkpoint_path, map_location="cpu")
            trainer.model.load_state_dict(probe_state)
            del probe_state
            gc.collect()
            trainer.model.set_feature_ablation_mode("none")
            early_valid_result, early_valid_special_metrics, early_valid_diag, early_valid_filter = _run_trainer_eval(
                trainer,
                valid_data,
                split_name="early_valid",
                show_progress=show_progress,
                collect_special=special_logging_enabled,
                collect_diag=diag_logging_enabled,
            )
            if feature_ablation_logging_enabled:
                early_feature_metrics = _collect_feature_ablation_metrics(
                    trainer=trainer,
                    model=trainer.model,
                    valid_data=valid_data,
                    reference_mrr20=float((early_valid_result or {}).get("mrr@20", 0.0) or 0.0),
                    reference_diag=early_valid_diag,
                    show_progress=show_progress,
                    enable_global=True,
                    enable_family=family_ablation_logging_enabled,
                    split_prefix="early_valid",
                )
                feature_ablation_metrics.update({f"early_{k}": v for k, v in early_feature_metrics.items()})
                if isinstance(early_valid_diag, dict):
                    early_valid_diag["feature_ablation"] = early_feature_metrics

        return {
            "valid_result": {k: float(v) for k, v in (best_valid_result or {}).items()},
            "test_result": {k: float(v) for k, v in (test_result or {}).items()},
            "valid_special_metrics": best_valid_special_metrics or {},
            "test_special_metrics": test_special_metrics or {},
            "valid_main_eval_filter": best_valid_filter or {},
            "test_main_eval_filter": test_filter or {},
            "valid_diag": best_valid_diag or {},
            "test_diag": test_diag or {},
            "feature_ablation_metrics": feature_ablation_metrics or {},
            "early_valid_result": {k: float(v) for k, v in (early_valid_result or {}).items()} if early_valid_result else {},
            "early_valid_special_metrics": early_valid_special_metrics or {},
            "early_valid_main_eval_filter": early_valid_filter if 'early_valid_filter' in locals() else {},
            "early_valid_diag": early_valid_diag or {},
            "valid_cold_target_metrics": _extract_cold_slice_metrics(best_valid_special_metrics),
            "test_cold_target_metrics": _extract_cold_slice_metrics(test_special_metrics),
            "artifact_trial_num": int(trial_num),
        }
    finally:
        for obj in (model, trainer, train_data, valid_data, test_data, dataset, config):
            del obj
        gc.collect()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
#  Single-trial training
# ═══════════════════════════════════════════════════════════════════
def train_and_evaluate(cfg_dict: dict, trial_num: int | None = None, progress_cb=None) -> dict:
    """Train one configuration.  Returns best-validation MRR@20 and metrics."""
    cfg = copy.deepcopy(cfg_dict)
    cfg["log_wandb"] = False
    model_name = _normalize_model_name(cfg.get("model", ""))
    if model_name in _FEATURE_AWARE_MOE_MODELS and "fmoe_special_logging" not in cfg:
        cfg["fmoe_special_logging"] = True
    if model_name in _FEATURE_AWARE_MOE_MODELS and "fmoe_diag_logging" not in cfg:
        cfg["fmoe_diag_logging"] = True
    if (
        model_name in _FEATURE_AWARE_MOE_MODELS
        and "fmoe_feature_ablation_logging" not in cfg
        and _model_uses_feature_inputs(cfg)
    ):
        cfg["fmoe_feature_ablation_logging"] = bool(cfg.get("fmoe_special_logging", True) and cfg.get("fmoe_diag_logging", True))
    trial_epoch_log = bool(cfg.get("trial_epoch_log", False))
    cfg["show_progress"] = bool(cfg.get("show_progress", True)) and trial_epoch_log
    for drop in ("search", "search_stages", "search_strategy", "max_search"):
        cfg.pop(drop, None)

    cfg["valid_metric"] = "MRR@20"

    _sync_model_dimensions(cfg)
    _ensure_feature_load_columns(cfg)

    # CE loss → disable negative sampling
    if cfg.get("loss_type", "CE").upper() == "CE":
        cfg["train_neg_sample_args"] = None

    configure_runtime_acceleration(cfg)
    configure_data_cache(cfg)

    # Build RecBole Config (silence CLI arg warnings)
    _argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        config = Config(model=cfg["model"], dataset=cfg["dataset"], config_dict=cfg)
    finally:
        sys.argv = _argv

    warnings.filterwarnings("ignore", category=FutureWarning)
    init_seed(config["seed"], config["reproducibility"])

    cache_key = _make_data_cache_key(cfg)
    use_mem_cache = bool(cfg.get("in_memory_data_cache", True))
    max_mem_cache = max(1, int(cfg.get("max_in_memory_data_cache", 1)))
    data_cache_hit = "miss"

    if use_mem_cache and cache_key in _DATA_BUNDLE_CACHE:
        dataset, train_data, valid_data, test_data = _DATA_BUNDLE_CACHE[cache_key]
        _DATA_BUNDLE_CACHE.move_to_end(cache_key, last=True)
        _reset_loader_seed(train_data, int(config["seed"]))
        _reset_loader_seed(valid_data, int(config["seed"]))
        _reset_loader_seed(test_data, int(config["seed"]))
        t_ds = 0.0
        t_dl = 0.0
        data_cache_hit = "memory"
    else:
        t_ds = time.time()
        dataset = create_dataset(config)
        t_ds = time.time() - t_ds
        t_dl = time.time()
        train_data, valid_data, test_data = data_preparation(config, dataset)
        t_dl = time.time() - t_dl
        if use_mem_cache:
            _put_data_cache(
                cache_key,
                (dataset, train_data, valid_data, test_data),
                max_entries=max_mem_cache,
            )
        data_cache_hit = "disk_or_build"

    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])
    transfer_report = _apply_transfer_initialization(model, cfg)

    trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_cls(config, model)
    setattr(trainer, "_disable_patch_logging", True)
    # Keep train-based seen/unseen reference stable across valid/test evaluation.
    trainer._fmoe_special_item_counts_train = _build_item_counts_from_loader(train_data)
    trainer._main_eval_unseen_filter_stats = {}
    special_logging_enabled = bool(_config_get(config, "special_logging", cfg.get("special_logging", False)))
    diag_logging_enabled = False
    eval_logging_timing = "per_eval"
    feature_ablation_logging_enabled = False
    family_ablation_logging_enabled = False
    artifact_logging_policy = "per_trial"
    probe_snapshot_enabled = False
    if model_name in _FEATURE_AWARE_MOE_MODELS:
        special_logging_enabled = special_logging_enabled or bool(
            _config_get(config, "fmoe_special_logging", cfg.get("fmoe_special_logging", True))
        )
        diag_logging_enabled = bool(
            _config_get(config, "fmoe_diag_logging", cfg.get("fmoe_diag_logging", True))
        )
        eval_logging_timing = str(
            _config_get(config, "fmoe_eval_logging_timing", cfg.get("fmoe_eval_logging_timing", "final_only"))
        ).strip().lower() or "final_only"
        if eval_logging_timing not in {"per_eval", "final_only"}:
            eval_logging_timing = "final_only"
        feature_ablation_logging_enabled = bool(
            _config_get(
                config,
                "fmoe_feature_ablation_logging",
                cfg.get("fmoe_feature_ablation_logging", False),
            )
        )
        family_ablation_logging_enabled = bool(
            _config_get(
                config,
                "fmoe_feature_family_ablation_logging",
                cfg.get("fmoe_feature_family_ablation_logging", feature_ablation_logging_enabled),
            )
        )
        artifact_logging_policy = str(
            _config_get(
                config,
                "fmoe_artifact_logging_policy",
                cfg.get("fmoe_artifact_logging_policy", "combo_best"),
            )
        ).strip().lower() or "combo_best"
        if artifact_logging_policy not in {"per_trial", "combo_best"}:
            artifact_logging_policy = "combo_best"
        probe_snapshot_enabled = bool(
            _config_get(
                config,
                "fmoe_probe_snapshot_enable",
                cfg.get("fmoe_probe_snapshot_enable", True),
            )
        )
    defer_detailed_artifacts = bool(
        model_name in _FEATURE_AWARE_MOE_MODELS
        and artifact_logging_policy != "per_trial"
        and (special_logging_enabled or diag_logging_enabled or feature_ablation_logging_enabled)
    )
    best_valid_special_metrics = None
    test_special_metrics = None
    best_valid_diag = None
    test_diag = None
    valid_zero_diag = None
    valid_shuffle_diag = None
    feature_ablation_metrics = {}
    epoch_trace_rows: list[dict] = []
    best_epoch = None

    # Sampled evaluation for large item sets
    n_items = dataset.item_num
    ev_cfg = cfg.get("eval_sampling", {})
    threshold = ev_cfg.get("auto_full_threshold", 500000)
    sample_num = ev_cfg.get("sample_num", 1000)
    if n_items > threshold:
        item_counts = np.zeros(n_items, dtype=np.int64)
        ids = dataset.inter_feat[dataset.iid_field]
        ids = ids.numpy() if hasattr(ids, "numpy") else ids
        for iid in ids:
            if 0 <= iid < n_items:
                item_counts[iid] += 1
        neg_items = np.argsort(-item_counts)
        neg_items = neg_items[neg_items != 0][:sample_num]
        from recbole_patch import setup_sampled_eval
        setup_sampled_eval(trainer, neg_items, sample_num, n_items, config["device"])

    # Training loop with early stopping + temp best-stage checkpoint for test.
    best_stage_path = _best_stage_temp_path(cfg)
    try:
        best_stage_path.unlink(missing_ok=True)
    except Exception:
        pass
    _register_temp_path(best_stage_path)
    probe_stage_path = _trial_artifact_stage_path(cfg, trial_num=trial_num or 0, tag="probe_local")
    try:
        probe_stage_path.unlink(missing_ok=True)
    except Exception:
        pass
    _register_temp_path(probe_stage_path)
    probe_stage_saved = False
    best_mrr20 = float("-inf")
    best_result: dict = {}
    test_result: dict = {}
    try:
        patience = int(config["stopping_step"])
    except (KeyError, TypeError):
        patience = 8
    no_improve = 0
    max_epochs = int(config["epochs"])
    eval_every = max(1, int(cfg.get("eval_every", 1)))
    show_progress = bool(cfg.get("show_progress", True))
    early_stopped = False
    final_epoch = 0
    lr_scheduler, lr_scheduler_type = _build_lr_scheduler(cfg, trainer, max_epochs=max_epochs)

    def _run_eval(data_loader, *, split_name: str, collect_special: bool = True, collect_diag: bool = True):
        return _run_trainer_eval(
            trainer,
            data_loader,
            split_name=split_name,
            show_progress=show_progress,
            collect_special=bool(collect_special and special_logging_enabled),
            collect_diag=bool(collect_diag and diag_logging_enabled),
        )

    t0 = time.time()
    try:
        for epoch in range(max_epochs):
            epoch_start = time.time()
            final_epoch = epoch + 1
            if progress_cb is not None:
                try:
                    progress_cb(
                        {
                            "trial_num": trial_num,
                            "epoch": epoch + 1,
                            "max_epochs": max_epochs,
                            "train_loss": 0.0,
                            "eval": False,
                            "best_mrr20": float(best_mrr20) if best_mrr20 > -1e8 else 0.0,
                            "patience_used": int(no_improve),
                            "patience_total": int(patience),
                            "state": "epoch_start",
                        }
                    )
                except Exception:
                    pass

            try:
                schedule_model = trainer.model
                if hasattr(schedule_model, "set_schedule_epoch"):
                    schedule_model.set_schedule_epoch(
                        epoch_idx=epoch,
                        max_epochs=max_epochs,
                        log_now=False,
                    )
            except Exception:
                pass

            train_loss = trainer._train_epoch(train_data, epoch_idx=epoch, show_progress=show_progress)
            if isinstance(train_loss, (tuple, list)):
                train_loss = sum(float(x) for x in train_loss)
            train_loss = float(train_loss)
            epoch_lr = _optimizer_current_lr(trainer.optimizer)

            should_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == max_epochs)
            if not should_eval:
                if lr_scheduler is not None and lr_scheduler_type in {"cosine", "warmup_cosine"}:
                    lr_scheduler.step()
                    epoch_lr = _optimizer_current_lr(trainer.optimizer)
                if progress_cb is not None:
                    try:
                        progress_cb(
                            {
                                "trial_num": trial_num,
                                "epoch": epoch + 1,
                                "max_epochs": max_epochs,
                                "train_loss": float(train_loss),
                                "eval": False,
                                "best_mrr20": float(best_mrr20) if best_mrr20 > -1e8 else 0.0,
                                "patience_used": int(no_improve),
                                "patience_total": int(patience),
                                "lr": float(epoch_lr),
                            }
                        )
                    except Exception:
                        pass

                epoch_time = time.time() - epoch_start
                if trial_epoch_log:
                    print(
                        f"    Ep {epoch+1:>3}/{max_epochs:<3}\tSKIP@{eval_every}\t"
                        f"train_loss {train_loss:7.4f}\tlr {epoch_lr:8.2e}\t"
                        f"pat {no_improve:>2}/{patience:<2}\t"
                        f"time {epoch_time:6.2f}s"
                    )
                epoch_trace_rows.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                        "eval": 0,
                        "valid_mrr20": 0.0,
                        "best_mrr20": float(best_mrr20) if best_mrr20 > -1e8 else 0.0,
                        "lr": float(epoch_lr),
                        "patience_used": int(no_improve),
                        "epoch_time_sec": float(epoch_time),
                    }
                )
                continue

            collect_epoch_logging = eval_logging_timing == "per_eval" and not defer_detailed_artifacts
            vr, epoch_valid_special, epoch_valid_diag, _epoch_valid_filter = _run_eval(
                valid_data,
                split_name="valid",
                collect_special=collect_epoch_logging,
                collect_diag=collect_epoch_logging,
            )

            mrr20 = float(vr.get("mrr@20", 0.0))
            if mrr20 > best_mrr20:
                best_mrr20 = mrr20
                best_result = {k: float(v) for k, v in vr.items()}
                if collect_epoch_logging:
                    best_valid_special_metrics = copy.deepcopy(epoch_valid_special)
                    best_valid_diag = copy.deepcopy(epoch_valid_diag)
                _save_best_stage(trainer.model, best_stage_path)
                no_improve = 0
                best_epoch = epoch + 1
            else:
                no_improve += 1
            if probe_snapshot_enabled and not probe_stage_saved:
                _save_best_stage(trainer.model, probe_stage_path)
                probe_stage_saved = True

            if lr_scheduler is not None:
                if lr_scheduler_type == "plateau":
                    lr_scheduler.step(mrr20)
                elif lr_scheduler_type in {"cosine", "warmup_cosine"}:
                    lr_scheduler.step()
                epoch_lr = _optimizer_current_lr(trainer.optimizer)

            if progress_cb is not None:
                try:
                    progress_cb(
                        {
                            "trial_num": trial_num,
                            "epoch": epoch + 1,
                            "max_epochs": max_epochs,
                            "train_loss": float(train_loss),
                            "eval": True,
                            "valid_mrr20": float(mrr20),
                            "best_mrr20": float(best_mrr20),
                            "patience_used": int(no_improve),
                            "patience_total": int(patience),
                            "lr": float(epoch_lr),
                        }
                    )
                except Exception:
                    pass

            epoch_time = time.time() - epoch_start
            if trial_epoch_log:
                best_disp = best_mrr20 if best_mrr20 > -1e8 else 0.0
                print(
                    f"    Ep {epoch+1:>3}/{max_epochs:<3}\tEVAL    \t"
                    f"train_loss {train_loss:7.4f}\tvalid M@20 {mrr20:9.6f}\t"
                    f"best M@20 {best_disp:9.6f}\tlr {epoch_lr:8.2e}\t"
                    f"pat {no_improve:>2}/{patience:<2}\t"
                    f"time {epoch_time:6.2f}s"
                )

            epoch_trace_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "eval": 1,
                    "valid_mrr20": float(mrr20),
                    "best_mrr20": float(best_mrr20),
                    "lr": float(epoch_lr),
                    "patience_used": int(no_improve),
                    "epoch_time_sec": float(epoch_time),
                }
            )

            if no_improve >= patience:
                early_stopped = True
                break

        if not best_result:
            best_result = {"mrr@20": 0.0}
        if best_mrr20 <= -1e8:
            best_mrr20 = float(best_result.get("mrr@20", 0.0) or 0.0)

        if best_stage_path.exists():
            best_state = torch.load(best_stage_path, map_location="cpu")
            trainer.model.load_state_dict(best_state)
            del best_state
            gc.collect()
        else:
            print(f"[WARN] Missing best-stage checkpoint before test: {best_stage_path}")

        collect_final_artifacts = not defer_detailed_artifacts
        best_valid_eval_result, best_valid_special_metrics, best_valid_diag, best_valid_filter = _run_eval(
            valid_data,
            split_name="best_valid",
            collect_special=collect_final_artifacts,
            collect_diag=collect_final_artifacts,
        )
        if best_valid_eval_result:
            best_result = {k: float(v) for k, v in best_valid_eval_result.items()}
            best_mrr20 = float(best_result.get("mrr@20", best_mrr20) or best_mrr20)

        if (
            feature_ablation_logging_enabled
            and not defer_detailed_artifacts
            and hasattr(trainer.model, "set_feature_ablation_mode")
        ):
            feature_ablation_metrics = _collect_feature_ablation_metrics(
                trainer=trainer,
                model=trainer.model,
                valid_data=valid_data,
                reference_mrr20=float(best_mrr20),
                reference_diag=best_valid_diag,
                show_progress=show_progress,
                enable_global=True,
                enable_family=family_ablation_logging_enabled,
                split_prefix="valid",
            )
            if isinstance(best_valid_diag, dict):
                best_valid_diag["feature_ablation"] = feature_ablation_metrics

        test_eval_started = time.time()
        tr, test_special_metrics, test_diag, test_filter = _run_eval(
            test_data,
            split_name="test",
            collect_special=collect_final_artifacts,
            collect_diag=collect_final_artifacts,
        )
        test_eval_time_sec = time.time() - test_eval_started
        test_result = {k: float(v) for k, v in tr.items()}

        avg_epoch_time_sec, avg_epoch_per_hour = _summarize_epoch_speed(epoch_trace_rows)
        test_eval_batches = _safe_len(test_data)
        test_eval_targets = _estimate_eval_target_count(test_data)
        test_eval_batches_per_sec = (
            float(test_eval_batches) / float(test_eval_time_sec)
            if test_eval_batches is not None and test_eval_time_sec > 0.0
            else None
        )
        test_eval_targets_per_sec = (
            float(test_eval_targets) / float(test_eval_time_sec)
            if test_eval_targets is not None and test_eval_time_sec > 0.0
            else None
        )

        elapsed = time.time() - t0

        fmoe_arch = {}
        try:
            model_name = _normalize_model_name(cfg.get("model", ""))
            if model_name in _FEATURE_AWARE_MOE_MODELS:
                fmoe_arch = _extract_featured_moe_arch(model, cfg)
        except Exception:
            fmoe_arch = {}

        artifact_best_checkpoint = ""
        artifact_probe_checkpoint = ""
        if defer_detailed_artifacts:
            staged_best_path = str(cfg.get("__artifact_trial_best_path", "") or "").strip()
            staged_probe_path = str(cfg.get("__artifact_trial_probe_path", "") or "").strip()
            if staged_best_path and best_stage_path.exists():
                artifact_best_checkpoint = _copy_stage_file(best_stage_path, staged_best_path)
            if staged_probe_path and probe_stage_saved and probe_stage_path.exists():
                artifact_probe_checkpoint = _copy_stage_file(probe_stage_path, staged_probe_path)

        return {
            "mrr@20": float(best_mrr20),
            "valid_result": best_result,
            "test_result": test_result,
            "valid_special_metrics": best_valid_special_metrics,
            "test_special_metrics": test_special_metrics,
            "valid_main_eval_filter": best_valid_filter or {},
            "test_main_eval_filter": test_filter or {},
            "valid_cold_target_metrics": _extract_cold_slice_metrics(best_valid_special_metrics),
            "test_cold_target_metrics": _extract_cold_slice_metrics(test_special_metrics),
            "valid_diag": best_valid_diag,
            "test_diag": test_diag,
            "valid_zero_diag": valid_zero_diag,
            "valid_shuffle_diag": valid_shuffle_diag,
            "feature_ablation_metrics": feature_ablation_metrics,
            "epoch_trace": _select_sparse_epoch_trace(epoch_trace_rows, best_epoch),
            "epochs_run": final_epoch,
            "early_stop_epoch": final_epoch,
            "early_stopped": early_stopped,
            "final_lr": _optimizer_current_lr(trainer.optimizer),
            "lr_scheduler_type": lr_scheduler_type,
            "elapsed": elapsed,
            "avg_epoch_time_sec": avg_epoch_time_sec,
            "avg_epoch_per_hour": avg_epoch_per_hour,
            "test_eval_time_sec": float(test_eval_time_sec),
            "test_eval_batches": test_eval_batches,
            "test_eval_targets": test_eval_targets,
            "test_eval_batches_per_sec": test_eval_batches_per_sec,
            "test_eval_targets_per_sec": test_eval_targets_per_sec,
            "fmoe_arch": fmoe_arch,
            "artifact_best_checkpoint": artifact_best_checkpoint,
            "artifact_probe_checkpoint": artifact_probe_checkpoint,
            "artifact_logging_policy": artifact_logging_policy,
            "transfer_report": transfer_report,
        }
    finally:
        _cleanup_temp_path(best_stage_path)
        _cleanup_temp_path(probe_stage_path)
        del model, trainer, train_data, valid_data, test_data, dataset, config
        gc.collect()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
#  Results I/O
# ═══════════════════════════════════════════════════════════════════
def _ser(v):
    """Make a value JSON-serialisable."""
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _save_results(
    path,
    trials_data,
    model,
    dataset,
    args,
    *,
    final_best=None,
    tuned_search=None,
    fixed_search=None,
    context_fixed=None,
    space_yaml=None,
    run_group="",
    run_axis="",
    run_phase="",
    parent_result="",
    interrupted=False,
    interrupted_at=None,
    best_checkpoint_file="",
    probe_checkpoint_file="",
    time_budget_hours=0.0,
    time_budget_reached=False,
    time_budget_reached_at_sec=None,
    stop_reason="",
):
    path = Path(path)
    dataset_canonical = _canonical_dataset_name(dataset)
    normalized_trials = []
    completed_trials = 0
    for trial in trials_data:
        row = dict(trial)
        status = row.get("status")
        if status is None:
            if row.get("error"):
                status = "fail"
            elif isinstance(row.get("mrr@20"), (int, float)):
                status = "ok"
            else:
                status = "unknown"
            row["status"] = status
        if status in ("ok", "fail"):
            completed_trials += 1
        normalized_trials.append(row)

    data = {
        "model": model,
        "dataset": dataset_canonical,
        "dataset_raw": dataset,
        "max_evals": args.max_evals,
        "oom_retry_limit": int(args.oom_retry_limit or 0),
        "tune_epochs": args.tune_epochs,
        "n_completed": completed_trials,
        "n_recorded_trials": len(normalized_trials),
        "timestamp": datetime.now().isoformat(),
        "trials": [
            {
                k: v
                for k, v in trial.items()
                if k
                not in {
                    "valid_special_metrics",
                    "early_valid_result",
                    "early_valid_special_metrics",
                    "test_special_metrics",
                    "valid_diag",
                    "early_valid_diag",
                    "test_diag",
                    "valid_zero_diag",
                    "valid_shuffle_diag",
                    "feature_ablation_metrics",
                    "epoch_trace",
                    "artifact_best_checkpoint",
                    "artifact_probe_checkpoint",
                }
            }
            for trial in normalized_trials
        ],
        "run_group": run_group,
        "run_axis": run_axis,
        "run_phase": run_phase,
        "parent_result": parent_result,
        "interrupted": bool(interrupted),
        "interrupted_at": interrupted_at,
        "time_budget_hours": float(time_budget_hours or 0.0),
        "time_budget_reached": bool(time_budget_reached),
        "time_budget_reached_at_sec": (
            float(time_budget_reached_at_sec)
            if isinstance(time_budget_reached_at_sec, (int, float))
            else None
        ),
        "stop_reason": str(stop_reason or ""),
    }
    if tuned_search is not None:
        data["tuned_search"] = tuned_search
    if fixed_search is not None:
        data["fixed_search"] = fixed_search
    if context_fixed is not None:
        data["context_fixed"] = context_fixed
    if space_yaml is not None:
        data["space_yaml"] = str(space_yaml)
    if final_best is not None:
        data["best_params"] = {k: _ser(v) for k, v in final_best.items()}
    if str(best_checkpoint_file or "").strip():
        data["best_checkpoint_file"] = str(best_checkpoint_file)
    if str(probe_checkpoint_file or "").strip():
        data["probe_checkpoint_file"] = str(probe_checkpoint_file)
    ok = [t for t in normalized_trials if t.get("status") == "ok"]
    special_payload = _build_special_metrics_payload(
        normalized_trials=normalized_trials,
        model=model,
        dataset=dataset,
        run_group=run_group,
        run_axis=run_axis,
        run_phase=run_phase,
        source_result_file=path,
    )
    mirror_paths = _artifact_mirror_paths(path, dataset_canonical, model, run_group, run_axis, run_phase)
    if ok:
        bt = max(ok, key=lambda x: x.get("mrr@20", 0))
        best_valid_result = bt.get("valid_result", {}) or {}
        test_result = bt.get("test_result", {}) or {}
        data["best_mrr@20"] = _ser(bt.get("mrr@20", 0.0))
        data["best_hr@10"] = _ser(best_valid_result.get("hit@10", 0.0))
        data["test_mrr@20"] = _ser(test_result.get("mrr@20", 0.0))
        data["test_hr@10"] = _ser(test_result.get("hit@10", 0.0))
        data["avg_epoch_time_sec"] = _ser(bt.get("avg_epoch_time_sec", 0.0))
        data["avg_epoch_per_hour"] = _ser(bt.get("avg_epoch_per_hour", 0.0))
        data["test_eval_time_sec"] = _ser(bt.get("test_eval_time_sec", 0.0))
        data["test_eval_batches_per_sec"] = _ser(bt.get("test_eval_batches_per_sec", 0.0))
        data["test_eval_targets_per_sec"] = _ser(bt.get("test_eval_targets_per_sec", 0.0))
        data["best_valid_result"] = best_valid_result
        data["test_result"] = test_result
        data["best_valid_special_metrics"] = bt.get("valid_special_metrics") or {}
        data["early_valid_result"] = bt.get("early_valid_result") or {}
        data["early_valid_special_metrics"] = bt.get("early_valid_special_metrics") or {}
        data["test_special_metrics"] = bt.get("test_special_metrics") or {}
        data["best_valid_main_eval_filter"] = bt.get("valid_main_eval_filter") or {}
        data["test_main_eval_filter"] = bt.get("test_main_eval_filter") or {}
        data["best_valid_cold_target_metrics"] = bt.get("valid_cold_target_metrics") or {}
        data["test_cold_target_metrics"] = bt.get("test_cold_target_metrics") or {}
        if bt.get("feature_ablation_metrics"):
            data["feature_ablation_metrics"] = bt.get("feature_ablation_metrics") or {}
    data["normal_result_mirror_file"] = str(mirror_paths["normal_result"].resolve())
    data["special_result_file"] = str(mirror_paths["special_result"].resolve()) if special_payload else ""
    data["special_log_file"] = str(mirror_paths["special_log"].resolve()) if special_payload else ""

    _write_json_file(path, data)
    if mirror_paths["normal_result"].resolve() != path.resolve():
        _write_json_file(mirror_paths["normal_result"], data)
    if special_payload is not None:
        _write_json_file(mirror_paths["special_result"], special_payload)
        if mirror_paths["special_log"].resolve() != mirror_paths["special_result"].resolve():
            _write_json_file(mirror_paths["special_log"], special_payload)

    diag_paths = _diag_artifact_paths(path, dataset_canonical, model, run_group, run_axis, run_phase)
    trial_summary_rows = []
    epoch_trace_rows = []
    for trial in normalized_trials:
        summary_row = {
            "trial": trial.get("trial"),
            "status": trial.get("status"),
            "mrr@20": trial.get("mrr@20"),
            "test_mrr@20": trial.get("test_mrr@20"),
            "test_hr@10": trial.get("test_hr@10"),
            "avg_epoch_time_sec": trial.get("avg_epoch_time_sec"),
            "test_eval_time_sec": trial.get("test_eval_time_sec"),
            "test_eval_targets_per_sec": trial.get("test_eval_targets_per_sec"),
            "epochs_run": trial.get("epochs_run"),
            "early_stopped": trial.get("early_stopped"),
        }
        summary_row.update(_diag_scalar_metrics(trial.get("valid_diag")))
        summary_row.update({f"feature_ablation.{k}": v for k, v in (trial.get("feature_ablation_metrics") or {}).items()})
        trial_summary_rows.append(summary_row)
        for row in list(trial.get("epoch_trace") or []):
            epoch_trace_rows.append({"trial": trial.get("trial"), **row})

    has_any_diag = any(
        bool(trial.get("valid_diag") or trial.get("early_valid_diag") or trial.get("test_diag"))
        for trial in normalized_trials
    )
    if has_any_diag and trial_summary_rows:
        _write_csv_file(diag_paths["raw_trial_summary"], trial_summary_rows)
    if has_any_diag and epoch_trace_rows:
        _write_csv_file(diag_paths["raw_epoch_trace"], epoch_trace_rows)

    tier_a_rows: list[dict] = []
    tier_b_rows: list[dict] = []
    viz_feature_rows: list[dict] = []
    viz_router_input_rows: list[dict] = []
    viz_group_feature_rows: list[dict] = []
    overview_payload = {"split": "", "feature_mode": "", "stages": []}
    collapse_row = min(ok, key=lambda x: float(x.get("mrr@20", 0.0) or 0.0)) if ok else {}

    if ok and bt.get("valid_diag"):
        best_valid_diag_payload = bt.get("valid_diag") or {}
        _write_json_file(diag_paths["raw_best_valid_diag"], best_valid_diag_payload)
        overview_payload = _build_diag_overview_payload(best_valid_diag_payload)
        _write_json_file(diag_paths["raw_best_valid_overview"], overview_payload)
        diag_paths["raw_best_valid_overview_md"].write_text(
            _diag_overview_markdown(overview_payload),
            encoding="utf-8",
        )
        tier_a_rows = _diag_tier_rows(best_valid_diag_payload, tier_key="tier_a_final")
        tier_b_rows = _diag_tier_rows(best_valid_diag_payload, tier_key="tier_b_internal")
        viz_feature_rows = _diag_viz_rows(best_valid_diag_payload, key="viz_feature_pca")
        viz_router_input_rows = _diag_viz_rows(best_valid_diag_payload, key="viz_router_input_pca")
        viz_group_feature_rows = _diag_viz_rows(best_valid_diag_payload, key="viz_group_feature_pca")

    if ok and bt.get("early_valid_diag"):
        _write_json_file(diag_paths["raw_early_valid_diag"], bt.get("early_valid_diag") or {})
    if ok and bt.get("test_diag"):
        _write_json_file(diag_paths["raw_test_diag"], bt.get("test_diag") or {})
    if collapse_row and collapse_row.get("valid_diag"):
        collapse_payload = copy.deepcopy(collapse_row.get("valid_diag") or {})
        if collapse_row.get("feature_ablation_metrics"):
            collapse_payload["feature_ablation"] = collapse_row.get("feature_ablation_metrics") or {}
        _write_json_file(diag_paths["raw_collapse_diag"], collapse_payload)

    tier_fieldnames = [
        "stage_name",
        "stage_key",
        "split",
        "aggregation_level",
        "node_kind",
        "node_name",
        "route_space",
        "support_size",
        "wrapper_name",
        "source_type",
        "temperature",
        "top_k",
        "entropy_norm",
        "n_eff_norm",
        "top1_monopoly_norm",
        "jitter_adj_norm",
        "knn_consistency_score",
        "knn_consistency_js",
        "n_eff",
        "cv_usage",
        "top1_max_frac",
        "entropy_mean",
    ]
    _write_csv_with_schema(diag_paths["tier_a_final"], tier_a_rows, fieldnames=tier_fieldnames)
    _write_csv_with_schema(diag_paths["tier_b_internal"], tier_b_rows, fieldnames=tier_fieldnames)

    viz_manifest = {
        "schema_version": "p8_router_diag_v1",
        "split": str(overview_payload.get("split", "")),
        "feature_mode": str(overview_payload.get("feature_mode", "")),
        "row_counts": {
            "viz_feature_pca": len(viz_feature_rows),
            "viz_router_input_pca": len(viz_router_input_rows),
            "viz_group_feature_pca": len(viz_group_feature_rows),
        },
    }
    _write_json_file(diag_paths["tier_c_manifest"], viz_manifest)
    _write_csv_gz_with_schema(
        diag_paths["tier_c_feature_pca"],
        viz_feature_rows,
        fieldnames=["pc1", "pc2", "stage_name", "aggregation_level", "wrapper_name", "final_top1_group", "final_top1_expert", "final_confidence", "session_length", "group_feature_scores"],
    )
    _write_csv_gz_with_schema(
        diag_paths["tier_c_router_input_pca"],
        viz_router_input_rows,
        fieldnames=["pc1", "pc2", "stage_name", "aggregation_level", "node_name", "top1_label", "confidence", "wrapper_name"],
    )
    _write_csv_gz_with_schema(
        diag_paths["tier_c_group_feature_pca"],
        viz_group_feature_rows,
        fieldnames=["pc1", "pc2", "stage_name", "aggregation_level", "group_name", "final_top1_group", "final_top1_expert", "primitive_top1", "confidence"],
    )

    meta_payload = {
        "schema_version": "p8_router_diag_v1",
        "model": model,
        "dataset": dataset_canonical,
        "run_group": run_group,
        "run_axis": run_axis,
        "run_phase": run_phase,
        "split": str(overview_payload.get("split", "")),
        "feature_mode": str(overview_payload.get("feature_mode", "")),
        "paths": {
            "tier_a_final": str(diag_paths["tier_a_final"].resolve()),
            "tier_b_internal": str(diag_paths["tier_b_internal"].resolve()),
            "tier_c_manifest": str(diag_paths["tier_c_manifest"].resolve()),
            "tier_c_feature_pca": str(diag_paths["tier_c_feature_pca"].resolve()),
            "tier_c_router_input_pca": str(diag_paths["tier_c_router_input_pca"].resolve()),
            "tier_c_group_feature_pca": str(diag_paths["tier_c_group_feature_pca"].resolve()),
            "raw_trial_summary": str(diag_paths["raw_trial_summary"].resolve()) if has_any_diag and trial_summary_rows else "",
            "raw_best_valid_diag": str(diag_paths["raw_best_valid_diag"].resolve()) if ok and bt.get("valid_diag") else "",
            "raw_best_valid_overview": str(diag_paths["raw_best_valid_overview"].resolve()) if ok and bt.get("valid_diag") else "",
            "raw_early_valid_diag": str(diag_paths["raw_early_valid_diag"].resolve()) if ok and bt.get("early_valid_diag") else "",
            "raw_test_diag": str(diag_paths["raw_test_diag"].resolve()) if ok and bt.get("test_diag") else "",
            "raw_collapse_diag": str(diag_paths["raw_collapse_diag"].resolve()) if collapse_row and collapse_row.get("valid_diag") else "",
            "raw_epoch_trace": str(diag_paths["raw_epoch_trace"].resolve()) if has_any_diag and epoch_trace_rows else "",
        },
        "row_counts": {
            "tier_a_final": len(tier_a_rows),
            "tier_b_internal": len(tier_b_rows),
            "viz_feature_pca": len(viz_feature_rows),
            "viz_router_input_pca": len(viz_router_input_rows),
            "viz_group_feature_pca": len(viz_group_feature_rows),
        },
    }
    _write_json_file(diag_paths["meta"], meta_payload)

    feature_ablation_path = ""
    if ok and bt.get("feature_ablation_metrics"):
        p = path.parent / "feature_ablation.json"
        _write_json_file(
            p,
            {
                "best_trial": int(bt.get("trial", 0) or 0),
                "mrr@20": _ser(bt.get("mrr@20", 0.0)),
                "feature_ablation_metrics": bt.get("feature_ablation_metrics") or {},
            },
        )
        feature_ablation_path = str(p.resolve())

    data["diag_dir"] = str((path.parent / "diag").resolve())
    data["diag_meta_file"] = str(diag_paths["meta"].resolve())
    data["diag_tier_a_final_file"] = str(diag_paths["tier_a_final"].resolve())
    data["diag_tier_b_internal_file"] = str(diag_paths["tier_b_internal"].resolve())
    data["diag_viz_manifest_file"] = str(diag_paths["tier_c_manifest"].resolve())
    data["diag_viz_feature_pca_file"] = str(diag_paths["tier_c_feature_pca"].resolve())
    data["diag_viz_router_input_pca_file"] = str(diag_paths["tier_c_router_input_pca"].resolve())
    data["diag_viz_group_feature_pca_file"] = str(diag_paths["tier_c_group_feature_pca"].resolve())
    data["diag_raw_trial_summary_file"] = str(diag_paths["raw_trial_summary"].resolve()) if has_any_diag and trial_summary_rows else ""
    data["diag_raw_best_valid_file"] = str(diag_paths["raw_best_valid_diag"].resolve()) if ok and bt.get("valid_diag") else ""
    data["diag_raw_best_valid_overview_file"] = str(diag_paths["raw_best_valid_overview"].resolve()) if ok and bt.get("valid_diag") else ""
    data["diag_raw_best_valid_overview_md_file"] = str(diag_paths["raw_best_valid_overview_md"].resolve()) if ok and bt.get("valid_diag") else ""
    data["diag_raw_early_valid_file"] = str(diag_paths["raw_early_valid_diag"].resolve()) if ok and bt.get("early_valid_diag") else ""
    data["diag_raw_test_file"] = str(diag_paths["raw_test_diag"].resolve()) if ok and bt.get("test_diag") else ""
    data["diag_raw_collapse_file"] = str(diag_paths["raw_collapse_diag"].resolve()) if collapse_row and collapse_row.get("valid_diag") else ""
    data["diag_raw_epoch_trace_file"] = str(diag_paths["raw_epoch_trace"].resolve()) if has_any_diag and epoch_trace_rows else ""
    data["feature_ablation_file"] = feature_ablation_path

    bundle_payload = _write_logging_bundle(
        result_path=path,
        data_payload=data,
        model=model,
        dataset=dataset,
        run_group=run_group,
        run_phase=run_phase,
    )
    data["logging_bundle_dir"] = bundle_payload.get("bundle_dir", "")
    data["logging_bundle_summary_file"] = bundle_payload.get("run_summary_json", "")

    _write_json_file(path, data)
    if mirror_paths["normal_result"].resolve() != path.resolve():
        _write_json_file(mirror_paths["normal_result"], data)
    return {
        "result_file": str(path.resolve()),
        "normal_result_mirror_file": str(mirror_paths["normal_result"].resolve()),
        "special_result_file": str(mirror_paths["special_result"].resolve()) if special_payload else "",
        "special_log_file": str(mirror_paths["special_log"].resolve()) if special_payload else "",
        "best_checkpoint_file": str(best_checkpoint_file or ""),
        "probe_checkpoint_file": str(probe_checkpoint_file or ""),
        "diag_dir": str((path.parent / "diag").resolve()),
        "diag_meta_file": str(diag_paths["meta"].resolve()),
        "diag_tier_a_final_file": str(diag_paths["tier_a_final"].resolve()),
        "diag_tier_b_internal_file": str(diag_paths["tier_b_internal"].resolve()),
        "diag_viz_manifest_file": str(diag_paths["tier_c_manifest"].resolve()),
        "diag_viz_feature_pca_file": str(diag_paths["tier_c_feature_pca"].resolve()),
        "diag_viz_router_input_pca_file": str(diag_paths["tier_c_router_input_pca"].resolve()),
        "diag_viz_group_feature_pca_file": str(diag_paths["tier_c_group_feature_pca"].resolve()),
        "diag_raw_trial_summary_file": str(diag_paths["raw_trial_summary"].resolve()) if has_any_diag and trial_summary_rows else "",
        "diag_raw_best_valid_file": str(diag_paths["raw_best_valid_diag"].resolve()) if ok and bt.get("valid_diag") else "",
        "diag_raw_best_valid_overview_file": str(diag_paths["raw_best_valid_overview"].resolve()) if ok and bt.get("valid_diag") else "",
        "diag_raw_best_valid_overview_md_file": str(diag_paths["raw_best_valid_overview_md"].resolve()) if ok and bt.get("valid_diag") else "",
        "diag_raw_early_valid_file": str(diag_paths["raw_early_valid_diag"].resolve()) if ok and bt.get("early_valid_diag") else "",
        "diag_raw_test_file": str(diag_paths["raw_test_diag"].resolve()) if ok and bt.get("test_diag") else "",
        "diag_raw_collapse_file": str(diag_paths["raw_collapse_diag"].resolve()) if collapse_row and collapse_row.get("valid_diag") else "",
        "diag_raw_epoch_trace_file": str(diag_paths["raw_epoch_trace"].resolve()) if has_any_diag and epoch_trace_rows else "",
        "feature_ablation_file": feature_ablation_path,
        "logging_bundle_dir": bundle_payload.get("bundle_dir", ""),
        "logging_bundle_summary_file": bundle_payload.get("run_summary_json", ""),
        "logging_bundle_summary_md_file": bundle_payload.get("run_summary_md", ""),
        "logging_bundle_diag_table_file": bundle_payload.get("diag_overview_table_csv", ""),
    }


def _maybe_update_active_tracker_result(result_path) -> None:
    run_id = str(os.environ.get("RUN_ID", "")).strip()
    if not run_id:
        return

    timeline_raw = str(os.environ.get("RUN_TIMELINE_DIR", "")).strip()
    if timeline_raw:
        timeline_dir = Path(timeline_raw).expanduser().resolve()
    else:
        timeline_dir = Path(__file__).resolve().parent / "run" / "artifacts" / "timeline"

    state_path = timeline_dir / "state" / f"{run_id}.json"
    if not state_path.exists():
        return

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(state, dict):
        return

    try:
        state["result_file"] = str(Path(result_path).resolve())
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def _maybe_update_baseline_phase_summary(dataset: str, phase: str) -> None:
    enabled = str(os.environ.get("BASELINE_PHASE_SUMMARY", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    phase_folder = str(os.environ.get("BASELINE_SUMMARY_PHASE", "")).strip()
    if not phase_folder:
        parts = [tok for tok in str(phase or "").split("_") if tok]
        for i, tok in enumerate(parts):
            if tok in {"A", "B"}:
                phase_folder = "_".join(parts[:i]) if i > 0 else parts[0]
                break
        if not phase_folder:
            phase_folder = str(phase or "").strip()
    if not dataset or not phase_folder:
        return

    script_path = Path(__file__).resolve().parent / "run" / "baseline" / "update_phase_summary.py"
    if not script_path.exists():
        return

    try:
        subprocess.run(
            [sys.executable, str(script_path), "--dataset", str(dataset), "--phase", phase_folder],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _maybe_update_fmoe_n_phase_summary(run_group: str, run_axis: str, phase: str) -> None:
    group = str(run_group or "").strip().lower()
    if group not in {"fmoe_n", "fmoe_n2", "fmoe_n3"}:
        return

    env_prefix = "FMOE_N3" if group == "fmoe_n3" else "FMOE_N2" if group == "fmoe_n2" else "FMOE_N"
    enabled = str(os.environ.get("TRACK_PHASE_SUMMARY", "")).strip().lower()
    if not enabled:
        enabled = str(os.environ.get(f"{env_prefix}_PHASE_SUMMARY", "")).strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    phase_folder = str(os.environ.get("TRACK_SUMMARY_PHASE", "")).strip()
    if not phase_folder:
        phase_folder = str(os.environ.get(f"{env_prefix}_SUMMARY_PHASE", "")).strip()
    if not phase_folder:
        phase_folder = str(str(phase or "").split("_", 1)[0]).strip()
    axis_folder = str(os.environ.get("TRACK_SUMMARY_AXIS", "")).strip()
    if not axis_folder:
        axis_folder = str(os.environ.get(f"{env_prefix}_SUMMARY_AXIS", "")).strip()
    axis_folder = axis_folder or str(run_axis or "").strip() or "hparam"
    if not phase_folder:
        return

    script_override = str(os.environ.get("TRACK_SUMMARY_SCRIPT", "")).strip()
    script_path = Path(script_override).expanduser().resolve() if script_override else (
        Path(__file__).resolve().parent / "run" / group / "update_phase_summary.py"
    )
    if not script_path.exists():
        return

    try:
        subprocess.run(
            [sys.executable, str(script_path), "--phase", phase_folder, "--axis", axis_folder],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _format_duration(seconds: float) -> str:
    """Human-readable duration with explicit unit."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.0f}s"


def _optimizer_current_lr(optimizer) -> float:
    try:
        lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    except Exception:
        return 0.0
    return max(lrs) if lrs else 0.0


def _build_lr_scheduler(cfg: dict, trainer, max_epochs: int):
    scheduler_type = str(cfg.get("lr_scheduler_type", "none") or "none").strip().lower()
    if scheduler_type in {"", "none", "off", "false"}:
        return None, "none"

    base_lr = float(getattr(trainer, "learning_rate", cfg.get("learning_rate", 0.0)) or 0.0)
    min_lr_ratio = min(max(float(cfg.get("lr_scheduler_min_lr_ratio", 0.1) or 0.1), 0.0), 1.0)
    eta_min = base_lr * min_lr_ratio
    t_max = max(int(max_epochs), 1)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=t_max,
            eta_min=eta_min,
        )
        return scheduler, scheduler_type

    if scheduler_type == "warmup_cosine":
        warmup_ratio = min(max(float(cfg.get("lr_scheduler_warmup_ratio", 0.1) or 0.0), 0.0), 1.0)
        warmup_epochs = max(int(round(t_max * warmup_ratio)), 1)

        def _warmup_cosine_lambda(epoch_idx: int) -> float:
            step = epoch_idx + 1
            if warmup_epochs > 0 and step <= warmup_epochs:
                return max(step / float(warmup_epochs), 1e-8)
            remain = max(t_max - warmup_epochs, 1)
            progress = min(max((step - warmup_epochs) / float(remain), 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=_warmup_cosine_lambda)
        return scheduler, scheduler_type

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer,
            mode="max",
            factor=float(cfg.get("lr_scheduler_plateau_factor", 0.5) or 0.5),
            patience=max(int(cfg.get("lr_scheduler_plateau_patience", 3) or 0), 0),
            min_lr=eta_min,
        )
        return scheduler, scheduler_type

    return None, "none"


# ═══════════════════════════════════════════════════════════════════
#  Wandb helpers (per-trial run mode)
# ═══════════════════════════════════════════════════════════════════
_wandb_enabled = False
_wandb_trial_open = False
_wandb_live_error_reported = False
_wandb_trial_error_reported = False
_wandb_mod = None
_wandb_import_error = ""


def _get_wandb_module():
    """Import wandb with shadowing diagnostics."""
    global _wandb_mod, _wandb_import_error
    if _wandb_mod is not None:
        return _wandb_mod
    if _wandb_import_error:
        raise RuntimeError(_wandb_import_error)

    try:
        wb = importlib.import_module("wandb")
    except Exception as e:
        _wandb_import_error = (
            "wandb import failed. Install with `pip install wandb` in the active env."
            f" ({e})"
        )
        raise RuntimeError(_wandb_import_error) from e

    if not hasattr(wb, "init"):
        wb_file = getattr(wb, "__file__", None)
        wb_path = list(getattr(wb, "__path__", [])) if hasattr(wb, "__path__") else []
        _wandb_import_error = (
            "wandb module resolved without `init`."
            f" module_file={wb_file}, module_path={wb_path}. "
            "Likely a local `wandb/` directory is shadowing the real package."
        )
        raise RuntimeError(_wandb_import_error)

    _wandb_mod = wb
    return wb


def _wandb_trial_start(model, dataset, args, trial_num, sampled_params, search_space_info):
    """Initialise one wandb run per hyperopt trial."""
    global _wandb_enabled, _wandb_trial_open
    try:
        wandb = _get_wandb_module()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_group = str(getattr(args, "run_group", "") or "").strip().lower()
        run_axis = str(getattr(args, "run_axis", "") or "").strip().lower()
        run_phase = str(getattr(args, "run_phase", "") or "").strip()
        phase_tag = run_phase if run_phase else "PNA"
        run_name = (
            f"tune_{dataset}_{model}_{run_axis or 'axis'}_{phase_tag}_"
            f"T{int(trial_num):03d}_{ts}"
        )
        tags = ["hyperopt", "trial", str(model), str(dataset)]
        if run_group:
            tags.append(run_group)
        if run_axis:
            tags.append(run_axis)
        if run_phase:
            tags.append(run_phase)

        trial_cfg = {
            "tuning_mode": "hyperopt_tpe",
            "trial_num": int(trial_num),
            "model": model,
            "dataset": dataset,
            "max_evals": args.max_evals,
            "tune_epochs": args.tune_epochs,
            "seed": args.seed,
            "run_group": run_group,
            "run_axis": run_axis,
            "run_phase": run_phase,
            "trial_params": {k: _ser(v) for k, v in sampled_params.items()},
            "search_space": search_space_info,
        }
        wandb.init(
            project=args.wandb_project if hasattr(args, "wandb_project") else "2026_FMoE_hyperopt",
            name=run_name,
            config=trial_cfg,
            tags=tags,
            reinit="finish_previous",
        )
        wandb.define_metric("live/*", step_metric="live/epoch")
        wandb.define_metric("trial/*", step_metric="live/epoch")
        _wandb_enabled = True
        _wandb_trial_open = True
        print(f"[Wandb] trial {trial_num} started project={wandb.run.project} name={wandb.run.name}")
    except Exception as e:
        print(f"[WARN] wandb trial init failed (trial={trial_num}): {e}")
        _wandb_enabled = False
        _wandb_trial_open = False


def _wandb_log_trial(trial_num, params, result, *, status="ok", error=""):
    """Log final trial summary to current wandb trial run."""
    global _wandb_trial_error_reported
    if not (_wandb_enabled and _wandb_trial_open):
        return
    try:
        wandb = _get_wandb_module()
        step = int(result.get("epochs_run", 0) or 0) if isinstance(result, dict) else 0
        step = max(1, step)
        payload = {
            "live/epoch": step,
            "trial/trial_num": int(trial_num),
            "trial/status": str(status),
            "trial/mrr@20": float((result or {}).get("mrr@20", 0.0)),
            "trial_status": str(status),
            "mrr@20": float((result or {}).get("mrr@20", 0.0)),
            "trial/epochs_run": int((result or {}).get("epochs_run", 0) or 0),
            "epochs_run": int((result or {}).get("epochs_run", 0) or 0),
            "trial/early_stop_epoch": int((result or {}).get("early_stop_epoch", (result or {}).get("epochs_run", 0)) or 0),
            "trial/early_stopped": int(bool((result or {}).get("early_stopped", False))),
            "trial/elapsed": float((result or {}).get("elapsed", 0.0) or 0.0),
        }
        for k, v in (params or {}).items():
            payload[f"trial/param_{k}"] = _ser(v)
        if error:
            payload["trial/error"] = str(error)
        for mk in ("hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10", "ndcg@20",
                    "mrr@5", "mrr@10", "mrr@20"):
            val = ((result or {}).get("valid_result", {}) or {}).get(mk, 0)
            payload[f"trial/{mk}"] = val
        for k, v in (((result or {}).get("fmoe_arch", {}) or {})).items():
            payload[f"trial/{k}"] = _ser(v)
        wandb.log(payload, step=step)
    except Exception as e:
        if not _wandb_trial_error_reported:
            print(f"[WARN] wandb trial summary log failed: {e}")
            _wandb_trial_error_reported = True


def _wandb_log_live(update: dict):
    """Log live in-trial progress (epoch-level) to current trial run."""
    global _wandb_live_error_reported
    if not (_wandb_enabled and _wandb_trial_open):
        return
    try:
        wandb = _get_wandb_module()
        trial_num = int(update.get("trial_num", 0) or 0)
        epoch = int(update.get("epoch", 0) or 0)
        max_epochs = int(update.get("max_epochs", 0) or 0)
        payload = {
            "live/epoch": epoch,
            "live/trial_num": trial_num,
            "live/max_epochs": max_epochs,
            "live/train_loss": float(update.get("train_loss", 0.0) or 0.0),
            "train_loss": float(update.get("train_loss", 0.0) or 0.0),
            "live/eval": int(bool(update.get("eval", False))),
            "live/best_mrr20": float(update.get("best_mrr20", 0.0) or 0.0),
            "best_mrr20": float(update.get("best_mrr20", 0.0) or 0.0),
            "live/patience_used": int(update.get("patience_used", 0) or 0),
            "live/patience_total": int(update.get("patience_total", 0) or 0),
        }
        if "state" in update:
            payload["live/state"] = str(update.get("state", ""))
        if "valid_mrr20" in update:
            vm = float(update.get("valid_mrr20", 0.0) or 0.0)
            payload["live/valid_mrr20"] = vm
            payload["valid_mrr20"] = vm
            payload["mrr@20"] = vm
        wandb.log(payload, step=max(1, epoch))
    except Exception as e:
        if not _wandb_live_error_reported:
            print(f"[WARN] wandb live log failed: {e}")
            _wandb_live_error_reported = True


def _wandb_trial_finish(extra_summary=None):
    """Close current wandb trial run."""
    global _wandb_trial_open
    if not _wandb_trial_open:
        return
    try:
        wandb = _get_wandb_module()
        run_name = wandb.run.name if wandb.run is not None else ""
        if wandb.run is not None and isinstance(extra_summary, dict):
            for k, v in extra_summary.items():
                wandb.summary[k] = _ser(v)
        wandb.finish()
        if run_name:
            print(f"[Wandb] trial finished name={run_name}")
    except Exception:
        pass
    finally:
        _wandb_trial_open = False


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    config_dir = Path(__file__).parent / "configs"

    # Load and compose config (Hydra)
    cfg = load_hydra_config(config_dir, args.config_name, args.overrides)

    # Eval-sampling policy
    cfg_omega = OmegaConf.create(cfg)
    cfg_omega = configure_eval_sampling(cfg_omega)
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    enforce_v4_feature_mode(cfg)

    # Optional stage-space YAML override (fixed + search).
    if args.space_yaml:
        fixed_from_yaml, search_from_yaml = load_space_yaml(args.space_yaml)
        if fixed_from_yaml:
            _deep_update(cfg, copy.deepcopy(fixed_from_yaml))
        if search_from_yaml:
            cfg["search"] = copy.deepcopy(search_from_yaml)
        # Keep explicit CLI key=value overrides as highest precedence.
        _reapply_cli_overrides(cfg, args.overrides)
        print(
            f"[SpaceYAML] loaded: {args.space_yaml} "
            f"(fixed={len(fixed_from_yaml)}, search={len(search_from_yaml)})"
        )

    # Normalize dataset casing and apply cache-size policy before tuning.
    resolved_dataset = _resolve_dataset_name_case(cfg)
    if resolved_dataset and resolved_dataset != cfg.get("dataset"):
        print(f"[Dataset] normalize dataset name: {cfg.get('dataset')} -> {resolved_dataset}")
        cfg["dataset"] = resolved_dataset

    _ensure_feature_load_columns(cfg)

    # Optional epoch / patience overrides
    if args.tune_epochs is not None:
        cfg["epochs"] = args.tune_epochs
    if args.tune_patience is not None:
        cfg["stopping_step"] = args.tune_patience

    # Wandb toggle: CLI flag or config value
    log_wandb = args.log_wandb or cfg.get("log_wandb", False)
    args.wandb_project = cfg.get("wandb_project_hyperopt", "2026_FMoE_hyperopt")

    model = cfg.get("model", "?")
    dataset = cfg.get("dataset", "?")
    dataset_canonical = _canonical_dataset_name(dataset)
    run_group = str(args.run_group or "").strip().lower()
    run_axis = str(args.run_axis or "").strip().lower()
    run_phase = str(args.run_phase or "").strip()

    # Fail fast before hyperopt loop if model registration/import is broken.
    try:
        _ = get_model(model)
    except Exception as e:
        print(f"[ERROR] model precheck failed for model={model}: {e}")
        print("[HINT] check custom model imports under experiments/models and recbole_patch.py")
        raise

    # Build search space (tuned vs fixed split)
    search = cfg.get("search", {})
    if not search:
        print("[ERROR] No 'search' block in merged config. Nothing to tune.")
        sys.exit(1)

    raw_space_type_overrides = cfg.get("search_space_type_overrides", {}) or {}
    if isinstance(raw_space_type_overrides, dict):
        space_type_overrides = {
            str(k).strip(): str(v).strip().lower()
            for k, v in raw_space_type_overrides.items()
            if str(k).strip()
        }
    else:
        space_type_overrides = {}

    tuned_search, fixed_search = split_search_params(search)
    if fixed_search:
        for k, v in fixed_search.items():
            _apply_runtime_param(cfg, str(k), copy.deepcopy(v))
    context_fixed = _extract_context_fixed(cfg)

    single_run_mode = False
    if tuned_search:
        space = build_hyperopt_space(tuned_search, type_overrides=space_type_overrides)
        if not space:
            single_run_mode = True
            space = {"__single_run__": hp.choice("__single_run__", [0])}
    else:
        single_run_mode = True
        space = {"__single_run__": hp.choice("__single_run__", [0])}

    if single_run_mode and args.max_evals != 1:
        print(
            f"[Info] No tunable params (or converted space empty). "
            f"Switching to single-run mode: max_evals {args.max_evals} -> 1"
        )
        args.max_evals = 1

    unique_choice_enabled = False
    choice_keys: list[str] = []
    choice_combos: list[dict] = []
    used_choice_signatures: set[tuple[tuple[str, str], ...]] = set()
    max_enumerated_choice_combos = 200000
    if _is_choice_only_space(tuned_search, type_overrides=space_type_overrides):
        choice_keys = sorted(tuned_search.keys())
        n_choice_combos = _count_choice_combos(tuned_search, choice_keys)
        if n_choice_combos > 0 and args.max_evals > n_choice_combos:
            print(
                f"[Info] Finite choice-only space has {n_choice_combos} unique combos; "
                f"capping max_evals {args.max_evals} -> {n_choice_combos}"
            )
            args.max_evals = n_choice_combos
        if 0 < n_choice_combos <= max_enumerated_choice_combos:
            choice_combos = _enumerate_choice_combos(tuned_search, choice_keys)
            unique_choice_enabled = True
            print(
                f"[UniqueChoice] enabled: dedupe/remap duplicates across {n_choice_combos} finite combos"
            )
        elif n_choice_combos > max_enumerated_choice_combos:
            print(
                f"[UniqueChoice] disabled: finite choice-only space has {n_choice_combos} combos, "
                f"exceeding enumeration cap {max_enumerated_choice_combos}"
            )

    # Print header
    n_discrete = 1
    for k, vals in tuned_search.items():
        if isinstance(vals, list) and _space_type(k, type_overrides=space_type_overrides) == "choice":
            n_discrete *= len(vals)
    total_pool_est = n_discrete  # continuous params are infinite, show discrete combos

    print(f"\n{'=' * 65}")
    print(f"  Hyperopt  |  {model} x {dataset}")
    print(f"  max_evals={args.max_evals}  epochs={cfg.get('epochs')}  "
          f"patience={cfg.get('stopping_step')}  wandb={'ON' if log_wandb else 'off'}")
    print(f"  search_algo={str(args.search_algo).lower()}")
    if float(args.max_run_hours or 0.0) > 0.0:
        print(f"  max_run_hours={float(args.max_run_hours):.3f}  (stop launching new trials after current trial)")
    if int(args.oom_retry_limit or 0) > 0:
        print(f"  oom_retry_limit={int(args.oom_retry_limit)}  (halve train/eval batch size on OOM)")
    train_bs = cfg.get("train_batch_size", "?")
    eval_bs = cfg.get("eval_batch_size", "?")
    print(f"  batch_size(train/valid/test)={train_bs}/{eval_bs}/{eval_bs}")
    print(f"  track={run_group or '-'}  axis={run_axis or '-'}  phase={run_phase or '-'}")
    if "train_batch_size" in tuned_search:
        print(f"  search.train_batch_size={tuned_search['train_batch_size']}")
    elif "train_batch_size" in fixed_search:
        print(f"  search.train_batch_size=[{fixed_search['train_batch_size']}]")
    if "eval_batch_size" in tuned_search:
        print(f"  search.eval_batch_size={tuned_search['eval_batch_size']}")
    elif "eval_batch_size" in fixed_search:
        print(f"  search.eval_batch_size=[{fixed_search['eval_batch_size']}]")
    print(f"{'=' * 65}")
    tuned_keys = sorted(tuned_search.keys())
    if tuned_keys:
        changed = ", ".join(tuned_keys[:12])
        if len(tuned_keys) > 12:
            changed += f", ... (+{len(tuned_keys) - 12})"
        print(f"Changed knobs: {changed}")
    else:
        print("Changed knobs: (none; single-run mode)")

    print(f"Tuned params(len>1): {len(tuned_search)}  (~{total_pool_est} discrete combos)")
    _print_grouped_params("Tuned groups", tuned_search, tuned=True, type_overrides=space_type_overrides)
    print(f"Fixed params(singleton/non-list): {len(fixed_search)}")
    _print_grouped_params("Fixed groups", fixed_search, tuned=False)
    _print_layout_details(cfg, tuned_search, fixed_search)
    print()

    search_info = {
        "tuned": {k: str(v) for k, v in tuned_search.items()},
        "fixed": {k: str(v) for k, v in fixed_search.items()},
        "space_yaml": str(args.space_yaml) if args.space_yaml else "",
        "space_type_overrides": {k: str(v) for k, v in space_type_overrides.items()},
        "search_algo": str(args.search_algo).lower(),
        "max_run_hours": float(args.max_run_hours or 0.0),
        "oom_retry_limit": int(args.oom_retry_limit or 0),
    }

    # Wandb (per-trial run mode)
    if log_wandb:
        os.environ["WANDB_DISABLED"] = "false"
        print(f"[Wandb] per-trial mode enabled project={args.wandb_project}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("[Wandb] disabled for this hyperopt run")

    # Output dir (artifacts-first; legacy path can still be supplied via env)
    results_root_env = str(os.environ.get("HYPEROPT_RESULTS_DIR", "")).strip()
    if results_root_env:
        results_dir = Path(results_root_env).expanduser()
    else:
        results_dir = Path(__file__).parent / "run" / "artifacts" / "results"
    run_group = str(args.run_group or "").strip().lower()
    if run_group:
        safe_group = re.sub(r"[^a-z0-9._-]+", "_", run_group).strip("._-")
        if safe_group:
            results_dir = results_dir / safe_group
    run_axis = str(args.run_axis or "").strip().lower()
    run_phase = str(args.run_phase or "").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if _use_unified_logging_layout():
        phase_bucket = _phase_bucket(run_phase)
        dataset_bucket = _safe_slug(dataset_canonical)
        model_bucket = _safe_slug(run_group or _model_tag(model) or "model")
        run_folder = f"{_safe_slug(run_phase or 'phase')}_{ts}_pid{os.getpid()}"
        results_dir = _logging_root() / model_bucket / dataset_bucket / phase_bucket / run_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    if _use_unified_logging_layout():
        result_file = results_dir / "result.json"
    else:
        phase_for_file = str(args.run_phase or "").strip().lower()
        phase_slug = re.sub(r"[^a-z0-9._-]+", "_", phase_for_file).strip("._-")
        if phase_slug:
            phase_slug = phase_slug[:80]
        file_parts = [dataset_canonical, model]
        if phase_slug:
            file_parts.append(phase_slug)
        file_parts.extend([ts, f"pid{os.getpid()}"])
        result_file = results_dir / ("_".join(file_parts) + ".json")

    # Future case-eval / re-inference workflows need a stable exported checkpoint path.
    if _normalize_model_name(model) in _FEATURE_AWARE_MOE_MODELS:
        cfg.setdefault("__artifact_combo_best_export_path", str((results_dir / "best_model_state.pth").resolve()))
        cfg.setdefault("__artifact_combo_probe_export_path", str((results_dir / "probe_model_state.pth").resolve()))

    trials = Trials()
    all_trials_data: list[dict] = []
    global_t0 = time.time()
    max_run_seconds = max(0.0, float(args.max_run_hours or 0.0) * 3600.0)
    best_so_far = float("-inf")
    tuned_search_ser = {k: _ser(v) for k, v in tuned_search.items()}
    fixed_search_ser = {k: _ser(v) for k, v in fixed_search.items()}
    context_fixed_ser = {k: _ser(v) for k, v in context_fixed.items()}
    parent_result = str(args.parent_result or "").strip()
    interrupted = False
    interrupted_at = None
    time_budget_reached = False
    time_budget_reached_at_sec = None
    stop_reason = ""
    artifact_paths = {
        "result_file": str(result_file.resolve()),
        "normal_result_mirror_file": "",
        "special_result_file": "",
        "special_log_file": "",
    }
    artifact_logging_policy = str(cfg.get("fmoe_artifact_logging_policy", "combo_best") or "combo_best").strip().lower()
    if artifact_logging_policy not in {"per_trial", "combo_best"}:
        artifact_logging_policy = "combo_best"
    defer_detailed_artifacts = bool(_normalize_model_name(model) in _FEATURE_AWARE_MOE_MODELS and artifact_logging_policy != "per_trial")
    combo_best_artifact_state = {
        "trial": None,
        "mrr20": float("-inf"),
        "params": {},
        "best_checkpoint": "",
        "probe_checkpoint": "",
    }

    # Objective function
    def objective(params):
        nonlocal best_so_far, interrupted, interrupted_at, combo_best_artifact_state
        trial_num = len(all_trials_data) + 1
        sampled_params = {k: v for k, v in params.items() if k != "__single_run__"}

        if unique_choice_enabled and choice_keys:
            proposed_sig = _choice_signature(sampled_params, choice_keys)
            if proposed_sig in used_choice_signatures:
                replacement = _pick_first_unused_choice_combo(choice_combos, used_choice_signatures, choice_keys)
                if replacement is not None:
                    sampled_params = replacement
                    remap_sig = _choice_signature(sampled_params, choice_keys)
                    print(f"[UniqueChoice] remap duplicate suggestion -> {sampled_params}")
                    used_choice_signatures.add(remap_sig)
                else:
                    # Exhausted (should not happen when max_evals is capped); keep proposal.
                    used_choice_signatures.add(proposed_sig)
            else:
                used_choice_signatures.add(proposed_sig)

        # Merge sampled hyperparams into base config
        cfg_trial = copy.deepcopy(cfg)
        cfg_trial.pop("search", None)
        cfg_trial["run_group"] = run_group
        cfg_trial["run_axis"] = run_axis
        cfg_trial["run_phase"] = run_phase
        cfg_trial["parent_result"] = parent_result
        for k, v in sampled_params.items():
            _apply_runtime_param(cfg_trial, str(k), _ser(v))
        if defer_detailed_artifacts:
            cfg_trial["__artifact_trial_best_path"] = str(_trial_artifact_stage_path(cfg_trial, trial_num=trial_num, tag="combo_best"))
            cfg_trial["__artifact_trial_probe_path"] = str(_trial_artifact_stage_path(cfg_trial, trial_num=trial_num, tag="combo_probe"))

        # Compact trial header
        parts = []
        for k, v in sorted(sampled_params.items()):
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        display = ", ".join(parts) if parts else "(fixed-only run)"
        print(f"\n[{trial_num}/{args.max_evals}] {display}")

        if log_wandb:
            _wandb_trial_start(model, dataset, args, trial_num, sampled_params, search_info)
            _wandb_log_live(
                {
                    "trial_num": trial_num,
                    "epoch": 0,
                    "max_epochs": int(cfg_trial.get("epochs", 0) or 0),
                    "train_loss": 0.0,
                    "eval": False,
                    "best_mrr20": 0.0,
                    "patience_used": 0,
                    "patience_total": int(cfg_trial.get("stopping_step", 0) or 0),
                    "state": "trial_started",
                }
            )

        oom_retry_count = 0
        oom_retry_history: list[dict] = []
        effective_train_bs, effective_eval_bs = _current_runtime_batch_sizes(cfg_trial)
        try:
            while True:
                try:
                    result = train_and_evaluate(
                        cfg_trial,
                        trial_num=trial_num,
                        progress_cb=_wandb_log_live if log_wandb else None,
                    )
                    effective_train_bs, effective_eval_bs = _current_runtime_batch_sizes(cfg_trial)
                    break
                except Exception as e:
                    if not _is_oom_error(e):
                        raise
                    if oom_retry_count >= int(args.oom_retry_limit or 0):
                        print(
                            f"[OOM_RETRY] trial={trial_num} exhausted retries "
                            f"after {oom_retry_count} reductions. last_error={str(e)[:240]}",
                            flush=True,
                        )
                        raise
                    reduction = _halve_batch_sizes_for_retry(cfg_trial)
                    if reduction is None:
                        print(
                            f"[OOM_RETRY] trial={trial_num} hit OOM but batch size cannot be reduced further. "
                            f"last_error={str(e)[:240]}",
                            flush=True,
                        )
                        raise
                    oom_retry_count += 1
                    effective_train_bs = int(reduction["train_after"])
                    effective_eval_bs = int(reduction["eval_after"])
                    sampled_params["train_batch_size"] = effective_train_bs
                    sampled_params["eval_batch_size"] = effective_eval_bs
                    retry_note = {
                        "retry_idx": int(oom_retry_count),
                        "train_before": int(reduction["train_before"]),
                        "eval_before": int(reduction["eval_before"]),
                        "train_after": int(reduction["train_after"]),
                        "eval_after": int(reduction["eval_after"]),
                    }
                    oom_retry_history.append(retry_note)
                    print(
                        "[OOM_RETRY] "
                        f"trial={trial_num} retry={oom_retry_count}/{int(args.oom_retry_limit or 0)} "
                        f"train_batch_size {reduction['train_before']} -> {reduction['train_after']} "
                        f"eval_batch_size {reduction['eval_before']} -> {reduction['eval_after']}",
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    time.sleep(0.5)
            mrr20 = result["mrr@20"]
            if mrr20 > best_so_far:
                best_so_far = mrr20
            current_artifact_best = str(result.get("artifact_best_checkpoint", "") or "").strip()
            current_artifact_probe = str(result.get("artifact_probe_checkpoint", "") or "").strip()
            if defer_detailed_artifacts:
                is_combo_best = (
                    combo_best_artifact_state["trial"] is None
                    or float(mrr20) > float(combo_best_artifact_state["mrr20"])
                )
                if is_combo_best:
                    _cleanup_temp_path(combo_best_artifact_state.get("best_checkpoint"))
                    _cleanup_temp_path(combo_best_artifact_state.get("probe_checkpoint"))
                    combo_best_artifact_state = {
                        "trial": int(trial_num),
                        "mrr20": float(mrr20),
                        "params": {k: _ser(v) for k, v in sampled_params.items()},
                        "best_checkpoint": current_artifact_best,
                        "probe_checkpoint": current_artifact_probe,
                    }
                else:
                    _cleanup_temp_path(current_artifact_best)
                    _cleanup_temp_path(current_artifact_probe)
            current_best_hr10 = float((result.get("valid_result", {}) or {}).get("hit@10", 0.0) or 0.0)
            current_test_mrr20 = float((result.get("test_result", {}) or {}).get("mrr@20", 0.0) or 0.0)
            current_test_hr10 = float((result.get("test_result", {}) or {}).get("hit@10", 0.0) or 0.0)
            current_avg_epoch_time_sec = float(result.get("avg_epoch_time_sec", 0.0) or 0.0)
            current_test_eval_time_sec = float(result.get("test_eval_time_sec", 0.0) or 0.0)
            current_test_eval_batches_per_sec = float(result.get("test_eval_batches_per_sec", 0.0) or 0.0)
            current_test_eval_targets_per_sec = float(result.get("test_eval_targets_per_sec", 0.0) or 0.0)
            run_best_trial = {
                "mrr@20": float(mrr20),
                "valid_result": result.get("valid_result", {}) or {},
                "test_result": result.get("test_result", {}) or {},
            }
            for prev in all_trials_data:
                if prev.get("status") != "ok":
                    continue
                prev_mrr = float(prev.get("mrr@20", 0.0) or 0.0)
                if prev_mrr > float(run_best_trial.get("mrr@20", 0.0) or 0.0):
                    run_best_trial = prev
            run_best_valid = (run_best_trial.get("valid_result", {}) or {})
            run_best_test = (run_best_trial.get("test_result", {}) or {})
            run_best_metrics = {
                "best_mrr@20": float(run_best_trial.get("mrr@20", 0.0) or 0.0),
                "best_hr@10": float(run_best_valid.get("hit@10", 0.0) or 0.0),
                "test_mrr@20": float(run_best_test.get("mrr@20", 0.0) or 0.0),
                "test_hr@10": float(run_best_test.get("hit@10", 0.0) or 0.0),
            }
            elapsed_total = time.time() - global_t0
            avg_trial_sec = elapsed_total / max(1, trial_num)
            remaining_trials = max(0, int(args.max_evals) - trial_num)
            eta_sec = avg_trial_sec * remaining_trials
            stop_label = (
                f"early_stop@{result['early_stop_epoch']}"
                if bool(result.get("early_stopped", False))
                else f"full@{result['epochs_run']}"
            )
            print(
                f"  -> MRR@20={mrr20:.6f}  best={best_so_far:.6f}  "
                f"({stop_label}, trial={_format_duration(result['elapsed'])}, "
                f"avg={_format_duration(avg_trial_sec)}/trial, "
                f"total={_format_duration(elapsed_total)}, "
                f"ETA={_format_duration(eta_sec)})"
            )
            print(
                "[TRIAL_METRICS] "
                f"cur_best_mrr20={float(mrr20):.6f} "
                f"cur_best_hr10={current_best_hr10:.6f} "
                f"cur_test_mrr20={current_test_mrr20:.6f} "
                f"cur_test_hr10={current_test_hr10:.6f} "
                f"avg_epoch_time_sec={current_avg_epoch_time_sec:.4f} "
                f"test_eval_time_sec={current_test_eval_time_sec:.4f} "
                f"test_eval_batches_per_sec={current_test_eval_batches_per_sec:.4f} "
                f"test_eval_targets_per_sec={current_test_eval_targets_per_sec:.4f} "
                f"run_best_mrr20={run_best_metrics['best_mrr@20']:.6f} "
                f"run_best_hr10={run_best_metrics['best_hr@10']:.6f} "
                f"run_test_mrr20={run_best_metrics['test_mrr@20']:.6f} "
                f"run_test_hr10={run_best_metrics['test_hr@10']:.6f}"
            )

            trial_record = {
                "trial": trial_num,
                "params": {k: _ser(v) for k, v in sampled_params.items()},
                "mrr@20": float(mrr20),
                "valid_result": result["valid_result"],
                "test_result": result.get("test_result", {}),
                "valid_special_metrics": result.get("valid_special_metrics") or {},
                "test_special_metrics": result.get("test_special_metrics") or {},
                "valid_main_eval_filter": result.get("valid_main_eval_filter") or {},
                "test_main_eval_filter": result.get("test_main_eval_filter") or {},
                "valid_cold_target_metrics": result.get("valid_cold_target_metrics") or {},
                "test_cold_target_metrics": result.get("test_cold_target_metrics") or {},
                "valid_diag": result.get("valid_diag") or {},
                "test_diag": result.get("test_diag") or {},
                "valid_zero_diag": result.get("valid_zero_diag") or {},
                "valid_shuffle_diag": result.get("valid_shuffle_diag") or {},
                "feature_ablation_metrics": result.get("feature_ablation_metrics") or {},
                "epoch_trace": result.get("epoch_trace") or [],
                "best_hr@10": current_best_hr10,
                "test_mrr@20": current_test_mrr20,
                "test_hr@10": current_test_hr10,
                "epochs_run": result["epochs_run"],
                "early_stop_epoch": result.get("early_stop_epoch", result["epochs_run"]),
                "early_stopped": bool(result.get("early_stopped", False)),
                "elapsed": round(result["elapsed"], 1),
                "avg_epoch_time_sec": result.get("avg_epoch_time_sec"),
                "avg_epoch_per_hour": result.get("avg_epoch_per_hour"),
                "test_eval_time_sec": result.get("test_eval_time_sec"),
                "test_eval_batches": result.get("test_eval_batches"),
                "test_eval_targets": result.get("test_eval_targets"),
                "test_eval_batches_per_sec": result.get("test_eval_batches_per_sec"),
                "test_eval_targets_per_sec": result.get("test_eval_targets_per_sec"),
                "oom_retry_count": int(oom_retry_count),
                "oom_retry_history": [dict(item) for item in oom_retry_history],
                "effective_train_batch_size": effective_train_bs,
                "effective_eval_batch_size": effective_eval_bs,
                "status": "ok",
                "artifact_best_checkpoint": current_artifact_best,
                "artifact_probe_checkpoint": current_artifact_probe,
            }
            trial_record.update(_diag_scalar_metrics(trial_record.get("valid_diag")))
            trial_record.update({f"feature_ablation.{k}": v for k, v in (trial_record.get("feature_ablation_metrics") or {}).items()})
            fmoe_arch = result.get("fmoe_arch", {}) or {}
            if fmoe_arch:
                trial_record["fmoe_arch"] = {k: _ser(v) for k, v in fmoe_arch.items()}
                for k in (
                    "n_pre_layer",
                    "n_pre_macro",
                    "n_pre_mid",
                    "n_pre_micro",
                    "n_post_layer",
                    "n_total_attn_layers",
                    "num_layers",
                ):
                    if k in fmoe_arch:
                        trial_record[k] = _ser(fmoe_arch[k])
            all_trials_data.append(trial_record)
            artifact_paths = _save_results(
                result_file,
                all_trials_data,
                model,
                dataset,
                args,
                tuned_search=tuned_search_ser,
                fixed_search=fixed_search_ser,
                context_fixed=context_fixed_ser,
                space_yaml=args.space_yaml,
                run_group=run_group,
                run_axis=run_axis,
                run_phase=run_phase,
                parent_result=parent_result,
                interrupted=interrupted,
                interrupted_at=interrupted_at,
            )
            _maybe_update_active_tracker_result(result_file)
            _maybe_update_baseline_phase_summary(dataset_canonical, run_phase)
            _maybe_update_fmoe_n_phase_summary(run_group, run_axis, run_phase)
            _wandb_log_trial(trial_num, sampled_params, result)
            _wandb_trial_finish(
                {
                    "trial_status": "ok",
                    "trial_num": int(trial_num),
                    "trial_mrr@20": float(mrr20),
                }
            )

            return {"loss": -mrr20, "status": STATUS_OK}

        except KeyboardInterrupt:
            interrupted = True
            interrupted_at = datetime.now().isoformat()
            _cleanup_temp_path(cfg_trial.get("__artifact_trial_best_path"))
            _cleanup_temp_path(cfg_trial.get("__artifact_trial_probe_path"))
            _wandb_log_trial(
                trial_num,
                sampled_params,
                {"mrr@20": 0.0, "epochs_run": 0, "elapsed": 0.0},
                status="interrupted",
                error="KeyboardInterrupt",
            )
            _wandb_trial_finish(
                {
                    "trial_status": "interrupted",
                    "trial_num": int(trial_num),
                }
            )
            all_trials_data.append({
                "trial": trial_num,
                "params": {k: _ser(v) for k, v in sampled_params.items()},
                "oom_retry_count": int(oom_retry_count),
                "oom_retry_history": [dict(item) for item in oom_retry_history],
                "effective_train_batch_size": effective_train_bs,
                "effective_eval_batch_size": effective_eval_bs,
                "status": "interrupted",
                "error": "KeyboardInterrupt",
            })
            artifact_paths = _save_results(
                result_file,
                all_trials_data,
                model,
                dataset,
                args,
                tuned_search=tuned_search_ser,
                fixed_search=fixed_search_ser,
                context_fixed=context_fixed_ser,
                space_yaml=args.space_yaml,
                run_group=run_group,
                run_axis=run_axis,
                run_phase=run_phase,
                parent_result=parent_result,
                interrupted=interrupted,
                interrupted_at=interrupted_at,
            )
            _maybe_update_active_tracker_result(result_file)
            _maybe_update_baseline_phase_summary(dataset_canonical, run_phase)
            _maybe_update_fmoe_n_phase_summary(run_group, run_axis, run_phase)
            raise
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            _cleanup_temp_path(cfg_trial.get("__artifact_trial_best_path"))
            _cleanup_temp_path(cfg_trial.get("__artifact_trial_probe_path"))
            _wandb_log_trial(
                trial_num,
                sampled_params,
                {"mrr@20": 0.0, "epochs_run": 0, "elapsed": 0.0},
                status="fail",
                error=str(e),
            )
            _wandb_trial_finish(
                {
                    "trial_status": "fail",
                    "trial_num": int(trial_num),
                    "trial_error": str(e),
                }
            )
            all_trials_data.append({
                "trial": trial_num,
                "params": {k: _ser(v) for k, v in sampled_params.items()},
                "mrr@20": 0.0,
                "oom_retry_count": int(oom_retry_count),
                "oom_retry_history": [dict(item) for item in oom_retry_history],
                "effective_train_batch_size": effective_train_bs,
                "effective_eval_batch_size": effective_eval_bs,
                "status": "fail",
                "error": str(e),
            })
            artifact_paths = _save_results(
                result_file,
                all_trials_data,
                model,
                dataset,
                args,
                tuned_search=tuned_search_ser,
                fixed_search=fixed_search_ser,
                context_fixed=context_fixed_ser,
                space_yaml=args.space_yaml,
                run_group=run_group,
                run_axis=run_axis,
                run_phase=run_phase,
                parent_result=parent_result,
                interrupted=interrupted,
                interrupted_at=interrupted_at,
            )
            _maybe_update_active_tracker_result(result_file)
            _maybe_update_baseline_phase_summary(dataset_canonical, run_phase)
            _maybe_update_fmoe_n_phase_summary(run_group, run_axis, run_phase)
            return {"loss": 1.0, "status": STATUS_FAIL}

    # Run TPE
    try:
        algo_name = str(args.search_algo).strip().lower()
        algo_fn = rand.suggest if algo_name == "random" else tpe.suggest
        search_rng = np.random.default_rng(args.seed)
        best = {}
        if max_run_seconds > 0.0:
            while len(trials.trials) < int(args.max_evals):
                next_eval_target = min(int(args.max_evals), len(trials.trials) + 1)
                best = fmin(
                    fn=objective,
                    space=space,
                    algo=algo_fn,
                    max_evals=next_eval_target,
                    trials=trials,
                    rstate=search_rng,
                    show_progressbar=False,
                )
                elapsed_total = time.time() - global_t0
                if elapsed_total >= max_run_seconds:
                    time_budget_reached = True
                    time_budget_reached_at_sec = float(elapsed_total)
                    stop_reason = "time_budget_reached"
                    print(
                        "[TIME_BUDGET] "
                        f"reached limit={float(args.max_run_hours):.3f}h "
                        f"after {len(trials.trials)}/{int(args.max_evals)} trials "
                        f"(elapsed={_format_duration(elapsed_total)}). "
                        "Stopping new trials and finalizing current best.",
                        flush=True,
                    )
                    break
        else:
            best = fmin(
                fn=objective,
                space=space,
                algo=algo_fn,
                max_evals=args.max_evals,
                trials=trials,
                rstate=search_rng,
                show_progressbar=False,
            )
        best_params = space_eval(space, best)
        if "__single_run__" in best_params:
            best_params.pop("__single_run__", None)
    except KeyboardInterrupt:
        interrupted = True
        interrupted_at = interrupted_at or datetime.now().isoformat()
        print("\n[WARN] Interrupted by user. Saving latest partial results...")
        best_params = {}
    except Exception as e:
        print(f"\n[WARN] fmin ended with: {e}")
        best_params = {}

    # Best results
    ok_trials = [t for t in all_trials_data if t.get("status") == "ok"]
    best_trial = max(ok_trials, key=lambda x: x.get("mrr@20", 0.0)) if ok_trials else None
    best_mrr = float(best_trial.get("mrr@20", 0.0) or 0.0) if best_trial else 0.0
    best_valid_result = (best_trial or {}).get("valid_result", {}) or {}
    best_test_result = (best_trial or {}).get("test_result", {}) or {}
    best_hr10 = float(best_valid_result.get("hit@10", 0.0) or 0.0)
    test_mrr20 = float(best_test_result.get("mrr@20", 0.0) or 0.0)
    test_hr10 = float(best_test_result.get("hit@10", 0.0) or 0.0)
    best_avg_epoch_time_sec = float((best_trial or {}).get("avg_epoch_time_sec", 0.0) or 0.0)
    best_test_eval_time_sec = float((best_trial or {}).get("test_eval_time_sec", 0.0) or 0.0)
    best_test_eval_batches_per_sec = float((best_trial or {}).get("test_eval_batches_per_sec", 0.0) or 0.0)
    best_test_eval_targets_per_sec = float((best_trial or {}).get("test_eval_targets_per_sec", 0.0) or 0.0)
    if best_trial and isinstance(best_trial.get("params"), dict):
        best_params = dict(best_trial.get("params") or {})
    elif not best_params and ok_trials:
        best_params = max(ok_trials, key=lambda x: x["mrr@20"]).get("params", {})

    should_collect_deferred_artifacts = bool(
        defer_detailed_artifacts
        and best_trial is not None
        and combo_best_artifact_state.get("best_checkpoint")
    )
    if should_collect_deferred_artifacts:
        deferred_payload = _collect_deferred_combo_best_artifacts(
            base_cfg=cfg,
            sampled_params=combo_best_artifact_state.get("params") or {},
            checkpoint_path=str(combo_best_artifact_state.get("best_checkpoint") or ""),
            probe_checkpoint_path=str(combo_best_artifact_state.get("probe_checkpoint") or ""),
            trial_num=int(combo_best_artifact_state.get("trial") or 0),
        )
        if deferred_payload:
            target_trial_num = int(combo_best_artifact_state.get("trial") or 0)
            for row in all_trials_data:
                if int(row.get("trial") or 0) != target_trial_num:
                    continue
                for key in list(row.keys()):
                    if str(key).startswith("feature_ablation."):
                        row.pop(key, None)
                row["valid_result"] = deferred_payload.get("valid_result") or row.get("valid_result", {})
                row["test_result"] = deferred_payload.get("test_result") or row.get("test_result", {})
                row["valid_special_metrics"] = deferred_payload.get("valid_special_metrics") or {}
                row["early_valid_result"] = deferred_payload.get("early_valid_result") or {}
                row["early_valid_special_metrics"] = deferred_payload.get("early_valid_special_metrics") or {}
                row["test_special_metrics"] = deferred_payload.get("test_special_metrics") or {}
                row["valid_main_eval_filter"] = deferred_payload.get("valid_main_eval_filter") or {}
                row["early_valid_main_eval_filter"] = deferred_payload.get("early_valid_main_eval_filter") or {}
                row["test_main_eval_filter"] = deferred_payload.get("test_main_eval_filter") or {}
                row["valid_cold_target_metrics"] = deferred_payload.get("valid_cold_target_metrics") or {}
                row["test_cold_target_metrics"] = deferred_payload.get("test_cold_target_metrics") or {}
                row["valid_diag"] = deferred_payload.get("valid_diag") or {}
                row["early_valid_diag"] = deferred_payload.get("early_valid_diag") or {}
                row["test_diag"] = deferred_payload.get("test_diag") or {}
                row["feature_ablation_metrics"] = deferred_payload.get("feature_ablation_metrics") or {}
                row["best_hr@10"] = float((row.get("valid_result", {}) or {}).get("hit@10", 0.0) or 0.0)
                row["test_mrr@20"] = float((row.get("test_result", {}) or {}).get("mrr@20", 0.0) or 0.0)
                row["test_hr@10"] = float((row.get("test_result", {}) or {}).get("hit@10", 0.0) or 0.0)
                row.update(_diag_scalar_metrics(row.get("valid_diag")))
                row.update({f"feature_ablation.{k}": v for k, v in (row.get("feature_ablation_metrics") or {}).items()})
                break
            best_trial = max(ok_trials, key=lambda x: x.get("mrr@20", 0.0)) if ok_trials else None
            best_valid_result = (best_trial or {}).get("valid_result", {}) or {}
            best_test_result = (best_trial or {}).get("test_result", {}) or {}
            best_hr10 = float(best_valid_result.get("hit@10", 0.0) or 0.0)
            test_mrr20 = float(best_test_result.get("mrr@20", 0.0) or 0.0)
            test_hr10 = float(best_test_result.get("hit@10", 0.0) or 0.0)

            best_avg_epoch_time_sec = float((best_trial or {}).get("avg_epoch_time_sec", 0.0) or 0.0)
            best_test_eval_time_sec = float((best_trial or {}).get("test_eval_time_sec", 0.0) or 0.0)
            best_test_eval_batches_per_sec = float((best_trial or {}).get("test_eval_batches_per_sec", 0.0) or 0.0)
            best_test_eval_targets_per_sec = float((best_trial or {}).get("test_eval_targets_per_sec", 0.0) or 0.0)

    exported_best_checkpoint = ""
    exported_probe_checkpoint = ""
    combo_best_export_path = str(cfg.get("__artifact_combo_best_export_path", "") or "").strip()
    combo_probe_export_path = str(cfg.get("__artifact_combo_probe_export_path", "") or "").strip()
    if combo_best_export_path and combo_best_artifact_state.get("best_checkpoint"):
        best_ckpt_src = Path(str(combo_best_artifact_state.get("best_checkpoint") or ""))
        if best_ckpt_src.exists():
            exported_best_checkpoint = _export_stage_file(best_ckpt_src, combo_best_export_path)
    if combo_probe_export_path and combo_best_artifact_state.get("probe_checkpoint"):
        probe_ckpt_src = Path(str(combo_best_artifact_state.get("probe_checkpoint") or ""))
        if probe_ckpt_src.exists():
            exported_probe_checkpoint = _export_stage_file(probe_ckpt_src, combo_probe_export_path)

    # Summary
    total_time = time.time() - global_t0
    print(f"\n{'=' * 65}")
    print(f"  DONE  |  {model} x {dataset}")
    print(f"  Best MRR@20 = {best_mrr:.6f}  |  {len(ok_trials)}/{args.max_evals} trials OK")
    print(f"  Best HR@10  = {best_hr10:.6f}")
    print(f"  Test MRR@20 = {test_mrr20:.6f}  (best-valid checkpoint)")
    print(f"  Test HR@10  = {test_hr10:.6f}  (best-valid checkpoint)")
    if time_budget_reached:
        print(
            f"  Stop reason = time_budget({float(args.max_run_hours):.3f}h)"
            f" reached at {_format_duration(float(time_budget_reached_at_sec or total_time))}"
        )
    if best_avg_epoch_time_sec > 0.0:
        print(f"  Avg epoch time = {best_avg_epoch_time_sec:.3f} sec/epoch")
    if best_test_eval_time_sec > 0.0:
        print(
            f"  Final test eval = {best_test_eval_time_sec:.3f} sec"
            f" | batches/s={best_test_eval_batches_per_sec:.3f}"
            f" | targets/s={best_test_eval_targets_per_sec:.3f}"
        )
    print(
        "[RUN_METRICS] "
        f"best_valid_mrr20={best_mrr:.6f} "
        f"best_valid_hr10={best_hr10:.6f} "
        f"test_mrr20={test_mrr20:.6f} "
        f"test_hr10={test_hr10:.6f} "
        f"avg_epoch_time_sec={best_avg_epoch_time_sec:.4f} "
        f"test_eval_time_sec={best_test_eval_time_sec:.4f} "
        f"test_eval_batches_per_sec={best_test_eval_batches_per_sec:.4f} "
        f"test_eval_targets_per_sec={best_test_eval_targets_per_sec:.4f}"
    )
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")
    artifact_paths = _save_results(
        result_file,
        all_trials_data,
        model,
        dataset,
        args,
        final_best=best_params,
        tuned_search=tuned_search_ser,
        fixed_search=fixed_search_ser,
        context_fixed=context_fixed_ser,
        space_yaml=args.space_yaml,
        run_group=run_group,
        run_axis=run_axis,
        run_phase=run_phase,
        parent_result=parent_result,
        interrupted=interrupted,
        interrupted_at=interrupted_at,
        best_checkpoint_file=exported_best_checkpoint,
        probe_checkpoint_file=exported_probe_checkpoint,
        time_budget_hours=float(args.max_run_hours or 0.0),
        time_budget_reached=time_budget_reached,
        time_budget_reached_at_sec=time_budget_reached_at_sec,
        stop_reason=stop_reason,
    )
    print(f"  Results -> {result_file}")
    if artifact_paths.get("normal_result_mirror_file"):
        print(f"  Normal mirror -> {artifact_paths['normal_result_mirror_file']}")
    if artifact_paths.get("special_result_file"):
        print(f"  Special result -> {artifact_paths['special_result_file']}")
    if artifact_paths.get("special_log_file"):
        print(f"  Special log -> {artifact_paths['special_log_file']}")
    print(f"{'=' * 65}\n")
    _maybe_update_active_tracker_result(result_file)
    _maybe_update_baseline_phase_summary(dataset_canonical, run_phase)
    _maybe_update_fmoe_n_phase_summary(run_group, run_axis, run_phase)
    _cleanup_temp_path(combo_best_artifact_state.get("best_checkpoint"))
    _cleanup_temp_path(combo_best_artifact_state.get("probe_checkpoint"))


if __name__ == "__main__":
    main()
