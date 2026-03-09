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
import time
import argparse
import warnings
import traceback
import signal
import atexit
import re
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_trainer
from recbole.utils import utils as _rbu
get_model = _rbu.get_model

from hydra_utils import load_hydra_config, configure_eval_sampling
from omegaconf import OmegaConf

_FEATURE_AWARE_MOE_MODELS = {
    "featured_moe",
    "featuredmoe",
    "featured_moe_hir",
    "featuredmoe_hir",
    "featured_moe_hgr",
    "featuredmoe_hgr",
    "featured_moe_hir2",
    "featuredmoe_hir2",
    "featured_moe_protox",
    "featuredmoe_protox",
    "featured_moe_v2",
    "featuredmoe_v2",
}

_FEATURED_MOE_V2_MODELS = {
    "featured_moe_hgr",
    "featuredmoe_hgr",
    "featured_moe_v2",
    "featuredmoe_v2",
    "featured_moe_hir2",
    "featuredmoe_hir2",
    "featured_moe_protox",
    "featuredmoe_protox",
}


def _sync_model_dimensions(cfg_dict: dict) -> None:
    """Synchronize hidden/embedding size according to model family policy."""
    model_name = str(cfg_dict.get("model", "")).strip().lower()

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


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
    try:
        signal.signal(_sig, _on_termination_signal)
    except Exception:
        pass
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
        ["lastfm", "lastfm0.3", "kuairec", "kuairec0.3"],
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
    if feature_mode in ("full", "feature_added"):
        return "feature_added"
    if feature_mode == "basic":
        return "basic"

    model = str(cfg_dict.get("model", "")).lower()
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
    catalog = cfg.get("fmoe_v2_layout_catalog")
    if isinstance(catalog, list):
        return catalog

    layout_execution = cfg.get("layout_execution")
    if isinstance(layout_execution, dict):
        catalog = layout_execution.get("fmoe_v2_layout_catalog")
        if isinstance(catalog, list):
            return catalog

    catalog = cfg.get("arch_layout_catalog")
    if isinstance(catalog, list):
        return catalog
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
        "rule_router.n_bins",
        "rule_router.feature_per_expert",
        "rule_router.custom_stage_feature_map",
        "rule_router.expert_bias",
        "expert_scale",
        "learning_rate",
        "weight_decay",
        "hidden_dropout_prob",
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
    split_tag = "session" if isinstance(split_cfg, dict) and "RS" in split_cfg else "inter"
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


# ═══════════════════════════════════════════════════════════════════
#  Single-trial training
# ═══════════════════════════════════════════════════════════════════
def train_and_evaluate(cfg_dict: dict, trial_num: int | None = None, progress_cb=None) -> dict:
    """Train one configuration.  Returns best-validation MRR@20 and metrics."""
    cfg = copy.deepcopy(cfg_dict)
    cfg["log_wandb"] = False
    trial_epoch_log = bool(cfg.get("trial_epoch_log", False))
    cfg["show_progress"] = bool(cfg.get("show_progress", True)) and trial_epoch_log
    for drop in ("search", "search_stages", "search_strategy", "max_search"):
        cfg.pop(drop, None)

    cfg["valid_metric"] = "MRR@20"

    _sync_model_dimensions(cfg)

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

    if torch.cuda.is_available() and bool(config["use_gpu"]):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        try:
            cur = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(cur)
            print(f"    [Device] CUDA_VISIBLE_DEVICES={vis} | torch_device=cuda:{cur} ({dev_name})")
        except Exception:
            print(f"    [Device] CUDA_VISIBLE_DEVICES={vis} | torch_device={config['device']}")
    else:
        print(f"    [Device] torch_device={config['device']}")

    if bool(cfg.get("enable_data_cache", True)):
        print(
            "    [DataCache] "
            f"save_dataset={config['save_dataset']} "
            f"save_dataloaders={config['save_dataloaders']} | "
            f"checkpoint_dir={config['checkpoint_dir']} | "
            f"dataloaders_save_path={config['dataloaders_save_path']}"
        )

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

    print(
        f"    [DataPrep] cache_hit={data_cache_hit}, "
        f"create_dataset={t_ds:.2f}s, data_preparation={t_dl:.2f}s, "
        f"n_items={dataset.item_num}"
    )

    model_cls = get_model(config["model"])
    model = model_cls(config, train_data.dataset).to(config["device"])

    trainer_cls = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_cls(config, model)
    setattr(trainer, "_disable_patch_logging", True)

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

    # Training loop with early stopping
    best_mrr20 = float("-inf")
    best_result: dict = {}
    try:
        patience = int(config["stopping_step"])
    except (KeyError, TypeError):
        patience = 8
    no_improve = 0
    max_epochs = int(config["epochs"])
    eval_every = max(1, int(cfg.get("eval_every", 1)))
    show_progress = bool(cfg.get("show_progress", True))
    early_stopped = False

    t0 = time.time()
    final_epoch = 0
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
        # Keep epoch-based scheduling behavior consistent with recbole_train.py.
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

        should_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == max_epochs)
        if not should_eval:
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
                        }
                    )
                except Exception:
                    pass

            if trial_epoch_log:
                epoch_time = time.time() - epoch_start
                print(
                    f"    Ep {epoch+1:>3}/{max_epochs:<3}\tSKIP@{eval_every}\t"
                    f"train_loss {train_loss:7.4f}\tpat {no_improve:>2}/{patience:<2}\t"
                    f"time {epoch_time:6.2f}s"
                )
            continue

        vr = trainer._valid_epoch(valid_data, show_progress=show_progress)
        if isinstance(vr, tuple):
            vr = next((x for x in vr if isinstance(x, dict)), vr[0])

        mrr20 = float(vr.get("mrr@20", 0.0))
        if mrr20 > best_mrr20:
            best_mrr20 = mrr20
            best_result = {k: float(v) for k, v in vr.items()}
            no_improve = 0
        else:
            no_improve += 1

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
                    }
                )
            except Exception:
                pass

        if trial_epoch_log:
            epoch_time = time.time() - epoch_start
            best_disp = best_mrr20 if best_mrr20 > -1e8 else 0.0
            print(
                f"    Ep {epoch+1:>3}/{max_epochs:<3}\tEVAL    \t"
                f"train_loss {train_loss:7.4f}\tvalid M@20 {mrr20:7.4f}\t"
                f"best M@20 {best_disp:7.4f}\tpat {no_improve:>2}/{patience:<2}\t"
                f"time {epoch_time:6.2f}s"
            )

        if no_improve >= patience:
            early_stopped = True
            break

    elapsed = time.time() - t0

    fmoe_arch = {}
    try:
        model_name = str(cfg.get("model", "")).lower()
        if model_name in _FEATURE_AWARE_MOE_MODELS:
            fmoe_arch = _extract_featured_moe_arch(model, cfg)
    except Exception:
        fmoe_arch = {}

    # GPU memory cleanup
    del model, trainer, train_data, valid_data, test_data, dataset, config
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mrr@20": best_mrr20,
        "valid_result": best_result,
        "epochs_run": final_epoch,
        "early_stop_epoch": final_epoch,
        "early_stopped": early_stopped,
        "elapsed": elapsed,
        "fmoe_arch": fmoe_arch,
    }


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
):
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
        "tune_epochs": args.tune_epochs,
        "n_completed": completed_trials,
        "n_recorded_trials": len(normalized_trials),
        "timestamp": datetime.now().isoformat(),
        "trials": normalized_trials,
        "run_group": run_group,
        "run_axis": run_axis,
        "run_phase": run_phase,
        "parent_result": parent_result,
        "interrupted": bool(interrupted),
        "interrupted_at": interrupted_at,
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
    ok = [t for t in normalized_trials if t.get("status") == "ok"]
    if ok:
        bt = max(ok, key=lambda x: x.get("mrr@20", 0))
        data["best_mrr@20"] = bt["mrr@20"]
        data["best_valid_result"] = bt.get("valid_result", {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_ser)


def _format_duration(seconds: float) -> str:
    """Human-readable duration with explicit unit."""
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    return f"{seconds:.0f}s"


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
            _set_nested_value(cfg, str(k), copy.deepcopy(v))
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

    # Print header
    n_discrete = 1
    for k, vals in tuned_search.items():
        if isinstance(vals, list) and _space_type(k, type_overrides=space_type_overrides) == "choice":
            n_discrete *= len(vals)
    total_pool_est = n_discrete  # continuous params are infinite, show discrete combos

    print(f"\n{'=' * 65}")
    print(f"  Hyperopt TPE  |  {model} x {dataset}")
    print(f"  max_evals={args.max_evals}  epochs={cfg.get('epochs')}  "
          f"patience={cfg.get('stopping_step')}  wandb={'ON' if log_wandb else 'off'}")
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
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    phase_for_file = str(args.run_phase or "").strip().lower()
    phase_slug = re.sub(r"[^a-z0-9._-]+", "_", phase_for_file).strip("._-")
    if phase_slug:
        phase_slug = phase_slug[:80]
    file_parts = [dataset_canonical, model]
    if phase_slug:
        file_parts.append(phase_slug)
    file_parts.extend([ts, f"pid{os.getpid()}"])
    result_file = results_dir / ("_".join(file_parts) + ".json")

    trials = Trials()
    all_trials_data: list[dict] = []
    global_t0 = time.time()
    best_so_far = float("-inf")
    tuned_search_ser = {k: _ser(v) for k, v in tuned_search.items()}
    fixed_search_ser = {k: _ser(v) for k, v in fixed_search.items()}
    context_fixed_ser = {k: _ser(v) for k, v in context_fixed.items()}
    run_axis = str(args.run_axis or "").strip().lower()
    run_phase = str(args.run_phase or "").strip()
    parent_result = str(args.parent_result or "").strip()
    interrupted = False
    interrupted_at = None

    # Objective function
    def objective(params):
        nonlocal best_so_far, interrupted, interrupted_at
        trial_num = len(all_trials_data) + 1
        sampled_params = {k: v for k, v in params.items() if k != "__single_run__"}

        # Merge sampled hyperparams into base config
        cfg_trial = copy.deepcopy(cfg)
        cfg_trial.pop("search", None)
        for k, v in sampled_params.items():
            _set_nested_value(cfg_trial, str(k), _ser(v))

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

        try:
            result = train_and_evaluate(cfg_trial, trial_num=trial_num, progress_cb=_wandb_log_live if log_wandb else None)
            mrr20 = result["mrr@20"]
            if mrr20 > best_so_far:
                best_so_far = mrr20
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
                f"  -> MRR@20={mrr20:.4f}  best={best_so_far:.4f}  "
                f"({stop_label}, trial={_format_duration(result['elapsed'])}, "
                f"avg={_format_duration(avg_trial_sec)}/trial, "
                f"total={_format_duration(elapsed_total)}, "
                f"ETA={_format_duration(eta_sec)})"
            )

            trial_record = {
                "trial": trial_num,
                "params": {k: _ser(v) for k, v in sampled_params.items()},
                "mrr@20": float(mrr20),
                "valid_result": result["valid_result"],
                "epochs_run": result["epochs_run"],
                "early_stop_epoch": result.get("early_stop_epoch", result["epochs_run"]),
                "early_stopped": bool(result.get("early_stopped", False)),
                "elapsed": round(result["elapsed"], 1),
                "status": "ok",
            }
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
            _save_results(
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
                "status": "interrupted",
                "error": "KeyboardInterrupt",
            })
            _save_results(
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
            raise
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
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
                "status": "fail",
                "error": str(e),
            })
            _save_results(
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
            return {"loss": 1.0, "status": STATUS_FAIL}

    # Run TPE
    try:
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=args.max_evals,
            trials=trials,
            rstate=np.random.default_rng(args.seed),
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
    best_mrr = max((t["mrr@20"] for t in ok_trials), default=0.0)
    if not best_params and ok_trials:
        best_params = max(ok_trials, key=lambda x: x["mrr@20"]).get("params", {})

    # Summary
    total_time = time.time() - global_t0
    print(f"\n{'=' * 65}")
    print(f"  DONE  |  {model} x {dataset}")
    print(f"  Best MRR@20 = {best_mrr:.4f}  |  {len(ok_trials)}/{args.max_evals} trials OK")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  Best params:")
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v:.6g}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  Results -> {result_file}")
    print(f"{'=' * 65}\n")

    _save_results(
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
    )


if __name__ == "__main__":
    main()
