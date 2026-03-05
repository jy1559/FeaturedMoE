#!/usr/bin/env python3
"""
RecBole runner for FMoE experiments with Hydra config composition.

Features:
- Hierarchical config loading via Hydra (base -> model -> dataset -> CLI overrides)
- Grid search via `search` lists in config
- Adaptive evaluation sampling (full vs sampled based on item count)
- RecBole-native training and evaluation
- CPU thread limiting for shared server environments
- Enhanced wandb logging with timing and metrics

Usage:
    python recbole_train.py model=sasrec dataset=kuairec
    python recbole_train.py model=bert4rec dataset=lastfm --search
    python recbole_train.py model=sasrec dataset=kuairec epochs=50 learning_rate=0.0005
"""

from __future__ import annotations

# ============ GPU Selection (MUST be before torch import) ============
import os
import sys

# Parse gpu_id from CLI before torch imports
_gpu_id = None
for arg in sys.argv[1:]:
    if arg.startswith('gpu_id='):
        _gpu_id = arg.split('=')[1]
        break

# IMPORTANT:
# Set CUDA_VISIBLE_DEVICES BEFORE importing torch so physical GPU mapping is
# deterministic (cuda:0 == selected physical GPU).
if _gpu_id is not None and _gpu_id != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)

# ============ CPU Thread Limiting (MUST be before any imports) ============
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

# Apply RecBole 1.2.1 bugfix before importing RecBole
import recbole_patch  # noqa: F401

import sys
import argparse
import time
import warnings
import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_trainer
# Import get_model from recbole_utils to use the patched version
from recbole.utils import utils as recbole_utils
get_model = recbole_utils.get_model
import copy

from hydra_utils import (
    load_hydra_config,
    expand_search_space,
    configure_eval_sampling,
    normalize_search_stages,
    narrow_search_space,
)


# Configure tqdm to not leave progress bars (cleaner output)
from tqdm import tqdm
tqdm.pandas()  # Enable tqdm for pandas

import io
import re

_FEATURE_AWARE_MOE_MODELS = {
    "featured_moe",
    "featuredmoe",
    "featured_moe_hir",
    "featuredmoe_hir",
}

# Optional log tee: keep console clean, strip tqdm/ANSI from log
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

# Monkey-patch tqdm.tqdm to force leave=False for batch-level progress
import tqdm as tqdm_module
original_tqdm = tqdm_module.tqdm

class TqdmWithoutLeave(original_tqdm):
    """Custom tqdm that removes progress bars after completion (leave=False)."""
    def __init__(self, iterable=None, desc=None, total=None, leave=None, **kwargs):
        # Force leave=False to clean up batch progress bars
        if leave is None:
            leave = False
        super().__init__(iterable=iterable, desc=desc, total=total, leave=leave, **kwargs)

# Replace tqdm in all modules
tqdm_module.tqdm = TqdmWithoutLeave
tqdm_module.std.tqdm = TqdmWithoutLeave

_DATA_BUNDLE_CACHE = OrderedDict()


def _strip_dataset_suffix(name: str):
    """Extract dataset stem from RecBole filenames (supports dots in names)."""
    for suffix in (".train.inter", ".valid.inter", ".test.inter", ".inter", ".item"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _resolve_dataset_name_case(cfg_dict):
    """
    Resolve dataset name to actual on-disk casing.

    Example: `kuairec0.3` -> `KuaiRec0.3` when that directory exists.
    """
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


def _make_data_cache_key(cfg_dict):
    """Build a stable cache key for dataset/dataloader reuse."""
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


def configure_data_cache(cfg_dict):
    """Enable deterministic dataset/dataloader cache paths for faster startup."""
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


def get_args():
    parser = argparse.ArgumentParser(
        description="RecBole experiment runner with Hydra config composition",
        epilog="Examples:\n"
               "  python recbole_train.py model=sasrec dataset=kuairec\n"
               "  python recbole_train.py model=bert4rec dataset=lastfm epochs=50\n"
               "  python recbole_train.py model=sasrec dataset=kuairec --search\n"
               "  python recbole_train.py --config-name tune_ml model=sasrec --search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--search", action="store_true", help="Enable grid search")
    parser.add_argument("--config-name", dest="config_name", default="config",
                       help="Root config file name (default: config)")
    parser.add_argument("overrides", nargs="*", 
                       help="Hydra-style overrides: model=sasrec dataset=kuairec epochs=50")
    return parser.parse_args()


def configure_runtime_acceleration(cfg_dict):
    """Apply optional runtime acceleration toggles (safe defaults)."""
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
        # Keep deterministic behavior unless user explicitly opts in.
        enable_cudnn_benchmark = not bool(cfg_dict.get("reproducibility", True))
    try:
        torch.backends.cudnn.benchmark = bool(enable_cudnn_benchmark)
    except Exception:
        pass


def _collect_featured_moe_runtime_config(model) -> dict:
    """Collect runtime-resolved FeaturedMoE/HiR architecture fields for logging/config sync."""
    keys = (
        "arch_layout_id",
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
        if not hasattr(model, key):
            continue
        val = getattr(model, key)
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                continue
            val = val.item()
        out[key] = val
    return out


def run_custom_training(cfg_i, run_name: str, save_model: bool = False, run_logger=None):
    """Custom training loop with explicit wandb logging (RecBole wandb disabled)."""
    cfg_local = cfg_i.copy()
    cfg_local['log_wandb'] = False  # disable RecBole's internal wandb
    
    # Use gpu_id directly (no CUDA_VISIBLE_DEVICES manipulation)
    # RecBole will use torch.device(f'cuda:{gpu_id}')
    # Keep gpu_id as-is from config
    
    # CE loss doesn't support negative sampling
    if cfg_local.get('loss_type', 'CE').upper() == 'CE':
        cfg_local['train_neg_sample_args'] = None

    configure_runtime_acceleration(cfg_local)
    configure_data_cache(cfg_local)

    # Prevent RecBole from parsing CLI args and emitting unused-args warning
    argv_backup = sys.argv
    sys.argv = sys.argv[:1]
    try:
        config = Config(model=cfg_local['model'], dataset=cfg_local['dataset'], config_dict=cfg_local)
    finally:
        sys.argv = argv_backup

    # Suppress noisy pandas FutureWarning from fillna chained assignment
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Suppress torch GradScaler deprecation FutureWarning from RecBole trainer
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*GradScaler.*")
    init_seed(config['seed'], config['reproducibility'])

    if torch.cuda.is_available() and bool(config["use_gpu"]):
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)")
        try:
            cur = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(cur)
            print(f"[Device] CUDA_VISIBLE_DEVICES={vis} | torch_device=cuda:{cur} ({dev_name})")
        except Exception:
            print(f"[Device] CUDA_VISIBLE_DEVICES={vis} | torch_device={config['device']}")
    else:
        print(f"[Device] torch_device={config['device']}")

    if bool(cfg_local.get("enable_data_cache", True)):
        print(
            "[DataCache] "
            f"save_dataset={config['save_dataset']} "
            f"save_dataloaders={config['save_dataloaders']} | "
            f"checkpoint_dir={config['checkpoint_dir']} | "
            f"dataloaders_save_path={config['dataloaders_save_path']}"
        )

    cache_key = _make_data_cache_key(cfg_local)
    use_mem_cache = bool(cfg_local.get("in_memory_data_cache", True))
    max_mem_cache = max(1, int(cfg_local.get("max_in_memory_data_cache", 1)))
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
    """
    # Debug: Log MAX_ITEM_LIST_LENGTH and dataset info
    try:
        max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
    except:
        max_item_list_len = None
    if hasattr(dataset, 'field2seqlen'):
        seq_len = dataset.field2seqlen.get('item_id_list', max_item_list_len)
        print(f"[DEBUG] Sequence length from dataset: {seq_len}")"""
    
    print(
        f"[DataPrep] cache_hit={data_cache_hit}, "
        f"create_dataset={t_ds:.2f}s, data_preparation={t_dl:.2f}s, "
        f"n_items={dataset.item_num}"
    )
    # Debug: show both batch counts and sample counts per split
    """try:
        tr_samples = len(getattr(train_data, 'dataset', []))
        va_samples = len(getattr(valid_data, 'dataset', []))
        te_samples = len(getattr(test_data, 'dataset', []))
        print(
            f"[DEBUG] Train batches: {len(train_data)}, Valid batches: {len(valid_data)}, Test batches: {len(test_data)}\n"
            f"        Train samples: {tr_samples}, Valid samples: {va_samples}, Test samples: {te_samples}"
        )
    except Exception:
        print(f"[DEBUG] Train batches: {len(train_data)}, Valid batches: {len(valid_data)}, Test batches: {len(test_data)}")
    """
    # Get model (custom models already registered in Config initialization)
    model_cls = get_model(config['model'])
    model = model_cls(config, train_data.dataset).to(config['device'])

    # Sync runtime-resolved FeaturedMoE layout/depth fields back to cfg/loggers.
    model_name_l = str(cfg_i.get("model", "")).lower()
    if model_name_l in _FEATURE_AWARE_MOE_MODELS:
        resolved = _collect_featured_moe_runtime_config(model)
        if resolved:
            cfg_i.update(resolved)
            cfg_local.update(resolved)
            try:
                for k, v in resolved.items():
                    config[k] = v
            except Exception:
                pass
            if run_logger is not None and hasattr(run_logger, "update_config"):
                try:
                    run_logger.update_config(cfg_i)
                except Exception:
                    pass
            if cfg_i.get("log_wandb", False):
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.config.update(resolved, allow_val_change=True)
                except Exception:
                    pass
            print(
                "[FMoE Layout] "
                f"layout_id={resolved.get('arch_layout_id')} "
                f"depth=[{resolved.get('n_pre_layer')},"
                f"{resolved.get('n_pre_macro')},"
                f"{resolved.get('n_pre_mid')},"
                f"{resolved.get('n_pre_micro')},"
                f"{resolved.get('n_post_layer')}] "
                f"n_total_attn_layers={resolved.get('n_total_attn_layers')} "
                f"num_layers={resolved.get('num_layers')}"
            )

    trainer_cls = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = trainer_cls(config, model)
    setattr(trainer, '_disable_patch_logging', True)  # bypass patched valid/train hooks

    # Setup sampled evaluation if n_items > threshold
    eval_sampling = cfg_i.get('eval_sampling', {})
    auto_threshold = eval_sampling.get('auto_full_threshold', 500000)
    sample_num = eval_sampling.get('sample_num', 1000)
    n_items = dataset.item_num
    
    if n_items > auto_threshold:
        # Compute popularity-based negative candidates from RecBole dataset
        # This uses internal item IDs (0 to n_items-1)
        import numpy as np
        # Count item occurrences (popularity)
        item_counts = np.zeros(n_items, dtype=np.int64)
        inter_feat = dataset.inter_feat
        item_ids = inter_feat[dataset.iid_field].numpy() if hasattr(inter_feat[dataset.iid_field], 'numpy') else inter_feat[dataset.iid_field]
        for iid in item_ids:
            if 0 <= iid < n_items:
                item_counts[iid] += 1
        
        # Sort by popularity (descending), exclude padding item (0)
        # Item 0 is typically padding in RecBole
        item_pop_order = np.argsort(-item_counts)
        # Filter out item 0 (padding)
        neg_items = item_pop_order[item_pop_order != 0][:sample_num]
        
        # Setup sampled eval
        from recbole_patch import setup_sampled_eval
        setup_sampled_eval(trainer, neg_items, sample_num, n_items, config['device'])
        print(f"[SampledEval] Dataset has {n_items} items > {auto_threshold}, using sampled evaluation with {sample_num} negatives")
        print(f"[SampledEval] Using {len(neg_items)} popularity-based negatives for evaluation")

    valid_metric = str(config['valid_metric']).lower()
    best_metric = float('-inf')
    best_model_state = None
    best_valid_result = None
    # Track best per-metric independently to keep best_* curves monotonic
    best_by_metric = {
        'hit@5': float('-inf'),
        'hit@10': float('-inf'),
        'hit@20': float('-inf'),
        'ndcg@5': float('-inf'),
        'ndcg@10': float('-inf'),
        'ndcg@20': float('-inf'),
        'mrr@5': float('-inf'),
        'mrr@10': float('-inf'),
        'mrr@20': float('-inf'),
    }
    valid_step = 0

    log_wandb = cfg_i.get('log_wandb', False)
    max_epochs = config['epochs']
    eval_every = max(1, int(cfg_i.get("eval_every", 1)))
    # Hide internal tqdm bars; we'll print concise epoch lines instead
    # Show batch-level tqdm bars during training/validation, but make them disappear after each epoch
    train_show = True
    eval_show = True
    # Patience from config (stopping_step)
    try:
        patience = int(config['stopping_step'])
    except Exception:
        patience = 10
    no_improve = 0
    epoch_times = []

    def fmt_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    start_time = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()

        # Epoch-based schedule hook (e.g., FeaturedMoE warm-up controls).
        try:
            schedule_model = trainer.model
            if hasattr(schedule_model, "set_schedule_epoch"):
                schedule_model.set_schedule_epoch(epoch_idx=epoch, max_epochs=max_epochs, log_now=False)
        except Exception:
            pass
        
        # Training phase with progress
        train_loss = trainer._train_epoch(train_data, epoch_idx=epoch, show_progress=train_show)

        do_eval = ((epoch + 1) % eval_every == 0) or (epoch + 1 == max_epochs)
        valid_result = best_valid_result.copy() if isinstance(best_valid_result, dict) else {}
        if do_eval:
            # Validation phase with progress
            valid_result = trainer._valid_epoch(valid_data, show_progress=eval_show)
            if isinstance(valid_result, tuple):
                # RecBole may return (loss, metrics) or (metrics, extra)
                valid_result = next((x for x in valid_result if isinstance(x, dict)), valid_result[0])

            current = valid_result.get(valid_metric, None)
            if current is not None and current > best_metric:
                best_metric = current
                best_valid_result = valid_result.copy()
                best_model_state = {k: v.detach().cpu() for k, v in trainer.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            # Update best per metric (monotonic)
            for mk in list(best_by_metric.keys()):
                val = valid_result.get(mk, float('-inf'))
                if val > best_by_metric[mk]:
                    best_by_metric[mk] = val

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = max_epochs - (epoch + 1)
        eta = avg_epoch * remaining
        display_best_mrr20 = best_by_metric['mrr@20'] if best_by_metric['mrr@20'] > -1e8 else 0.0
        display_valid_mrr20 = valid_result.get('mrr@20', 0.0)
        eval_tag = "EVAL" if do_eval else f"SKIP@{eval_every}"

        # Concise, aligned epoch line (tabs + fixed widths)
        progress_pct = (epoch + 1) / max_epochs * 100
        epoch_line = (
            f"Ep {epoch+1:>3}/{max_epochs:<3}\t"
            f"{eval_tag:<8}\t"
            f"Prog {progress_pct:6.2f}%\t"
            f"ETA {fmt_time(eta)}\t"
            f"train_loss {train_loss:7.4f}\t"
            f"valid M@20 {display_valid_mrr20:7.4f}\t"
            f"best M@20 {display_best_mrr20:7.4f}\t"
            f"pat {no_improve:>2}/{patience:<2}\t"
            f"time {epoch_time:6.2f}s"
        )
        print(epoch_line)

        # FeaturedMoE expert/bucket debug logging (per-epoch summary)
        moe_summary = None
        debug_logging_enabled = bool(getattr(model, "fmoe_debug_logging", False))
        if debug_logging_enabled and hasattr(model, 'get_epoch_log_summary'):
            try:
                moe_summary = model.get_epoch_log_summary()
                if moe_summary.get("n_batches", 0) > 0:
                    from models.FeaturedMoE.logging_utils import MoELogger
                    print(MoELogger.format_summary(moe_summary))
                # Expert analysis (performance + feature bias)
                if moe_summary.get("analysis", {}).get("n_sampled", 0) > 0:
                    from models.FeaturedMoE.analysis_logger import ExpertAnalysisLogger
                    print(ExpertAnalysisLogger.format_summary(moe_summary["analysis"]))
            except Exception:
                pass  # Don't crash on logging failure

        # RunLogger: epoch metrics + expert weights + analysis to outputs/FMoE/
        if run_logger is not None:
            try:
                run_logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_result=valid_result,
                    did_eval=do_eval,
                    epoch_time=epoch_time,
                )
                if moe_summary and moe_summary.get("n_batches", 0) > 0:
                    run_logger.log_expert_weights(epoch=epoch, moe_summary=moe_summary)
                if moe_summary and moe_summary.get("analysis", {}).get("n_sampled", 0) > 0:
                    run_logger.log_analysis(epoch=epoch, analysis=moe_summary["analysis"])
            except Exception:
                pass

        if log_wandb:
            import wandb
            payload = {
                'train/loss': train_loss,
                # timing and steps
                'etc/epoch_time_sec': epoch_time,
                'etc/did_eval': int(do_eval),
                'train_step': epoch,
                'valid_step': valid_step,
            }
            if do_eval:
                payload.update({
                    # raw valid metrics
                    'valid/hit@5': valid_result.get('hit@5', 0),
                    'valid/hit@10': valid_result.get('hit@10', 0),
                    'valid/hit@20': valid_result.get('hit@20', 0),
                    'valid/ndcg@5': valid_result.get('ndcg@5', 0),
                    'valid/ndcg@10': valid_result.get('ndcg@10', 0),
                    'valid/ndcg@20': valid_result.get('ndcg@20', 0),
                    'valid/mrr@5': valid_result.get('mrr@5', 0),
                    'valid/mrr@10': valid_result.get('mrr@10', 0),
                    'valid/mrr@20': valid_result.get('mrr@20', 0),
                    # best-so-far monotonic metrics
                    'valid/best_hit@5': best_by_metric['hit@5'],
                    'valid/best_hit@10': best_by_metric['hit@10'],
                    'valid/best_hit@20': best_by_metric['hit@20'],
                    'valid/best_ndcg@5': best_by_metric['ndcg@5'],
                    'valid/best_ndcg@10': best_by_metric['ndcg@10'],
                    'valid/best_ndcg@20': best_by_metric['ndcg@20'],
                    'valid/best_mrr@5': best_by_metric['mrr@5'],
                    'valid/best_mrr@10': best_by_metric['mrr@10'],
                    'valid/best_mrr@20': best_by_metric['mrr@20'],
                })
            wandb.log(payload, step=epoch)

        if do_eval:
            valid_step += 1

        # Early stopping check
        if do_eval and no_improve >= patience:
            print(f"Early stopping triggered: no improvement for {patience} epochs.")
            break

    # If no improvement ever logged (edge), synthesize from best_by_metric
    if best_valid_result is None:
        best_valid_result = {
            'hit@5': best_by_metric['hit@5'] if best_by_metric['hit@5'] > -1e8 else 0.0,
            'hit@10': best_by_metric['hit@10'] if best_by_metric['hit@10'] > -1e8 else 0.0,
            'hit@20': best_by_metric['hit@20'] if best_by_metric['hit@20'] > -1e8 else 0.0,
            'ndcg@5': best_by_metric['ndcg@5'] if best_by_metric['ndcg@5'] > -1e8 else 0.0,
            'ndcg@10': best_by_metric['ndcg@10'] if best_by_metric['ndcg@10'] > -1e8 else 0.0,
            'ndcg@20': best_by_metric['ndcg@20'] if best_by_metric['ndcg@20'] > -1e8 else 0.0,
            'mrr@5': best_by_metric['mrr@5'] if best_by_metric['mrr@5'] > -1e8 else 0.0,
            'mrr@10': best_by_metric['mrr@10'] if best_by_metric['mrr@10'] > -1e8 else 0.0,
            'mrr@20': best_by_metric['mrr@20'] if best_by_metric['mrr@20'] > -1e8 else 0.0,
        }

    # Load best model for test evaluation
    if best_model_state is not None:
        trainer.model.load_state_dict(best_model_state)

    test_result = trainer._valid_epoch(test_data, show_progress=eval_show)
    if isinstance(test_result, tuple):
        test_result = next((x for x in test_result if isinstance(x, dict)), test_result[0])
    elapsed_time = time.time() - start_time

    result = {
        'best_valid_score': best_metric,
        'best_valid_result': best_valid_result or {},
        'test_result': test_result,
        'elapsed_time': elapsed_time,
    }

    # RunLogger: save final summary
    if run_logger is not None:
        try:
            run_logger.log_final(
                best_valid_result=best_valid_result or {},
                test_result=test_result,
                elapsed_time=elapsed_time,
                extra={
                    "best_valid_score": best_metric,
                    "epochs_run": len(epoch_times),
                    "model": cfg_i.get("model", ""),
                    "dataset": cfg_i.get("dataset", ""),
                },
            )
        except Exception:
            pass

    return result


def main():
    args = get_args()

    config_dir = Path(__file__).parent / "configs"
    
    # Load config with Hydra composition
    try:
        cfg = load_hydra_config(
            config_dir=config_dir,
            config_name=args.config_name,
            overrides=args.overrides,
        )
    except Exception as e:
        print(f"❌ Config error: {e}")
        print("\nHint: Specify model and dataset, e.g.:")
        print("  python recbole_train.py model=sasrec dataset=kuairec")
        sys.exit(1)

    # Configure eval sampling (auto full vs sampled based on item count)
    from omegaconf import DictConfig, OmegaConf
    cfg_omega = OmegaConf.create(cfg)
    cfg_omega = configure_eval_sampling(cfg_omega)
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)

    # Normalize dataset casing against actual files/directories on disk.
    resolved_dataset = _resolve_dataset_name_case(cfg)
    if resolved_dataset and resolved_dataset != cfg.get("dataset"):
        print(f"[Dataset] normalize dataset name: {cfg.get('dataset')} -> {resolved_dataset}")
        cfg["dataset"] = resolved_dataset

    # Resolve model/dataset names
    model = cfg.get("model")
    dataset = cfg.get("dataset")
    
    if not model or not dataset:
        print("❌ Both model and dataset must be specified")
        print(f"  model={model}, dataset={dataset}")
        sys.exit(1)

    def run_single(cfg_i, run_index, total, stage_idx=None, stage_total=None):
        cfg_i = cfg_i.copy()
        cfg_i.pop("search", None)

        if "hidden_size" in cfg_i:
            cfg_i["embedding_size"] = cfg_i["hidden_size"]
            cfg_i["inner_size"] = cfg_i["hidden_size"] * 2

        save_model = cfg_i.pop("saved", False)

        eval_args = cfg_i.get("eval_args", {})
        split_config = eval_args.get("split", {})
        if isinstance(split_config, dict):
            eval_mode = "session" if "RS" in split_config else "inter"
        else:
            eval_mode = "session" if cfg_i.get("benchmark_filename") else "inter"

        dataset_abbrev = {
            'amazon_beauty': 'AMA',
            'foursquare': 'NYC', 
            'kuairec': 'KUA',
            'kuairec0.3': 'KU3',
            'lastfm': 'LFM',
            'lastfm0.3': 'LF3',
            'movielens1m': 'ML1',
            'retail_rocket': 'ReR',
        }.get(dataset.lower(), dataset[:3].upper())

        model_abbrev = {
            'sasrec': 'SAS',
            'gru4rec': 'GRU',
            'narm': 'NRM',
            'stamp': 'STP',
            'bert4rec': 'BRT',
            'srgnn': 'GNN',
            'featuredmoe': 'FME',
            'featured_moe': 'FME',
            'featuredmoe_hir': 'FHR',
            'featured_moe_hir': 'FHR',
        }.get(model.lower(), model[:3].upper())
        code = f"{dataset_abbrev}_{model_abbrev}"

        info_tokens = []
        mode_token = "eS" if eval_mode == "session" else "eI"
        info_tokens.append(mode_token)
        if stage_idx is not None:
            info_tokens.append(f"s{stage_idx+1}")

        defaults = {
            'epochs': 100, 'learning_rate': 0.001, 'hidden_size': 128, 
            'train_batch_size': 4096, 'MAX_ITEM_LIST_LENGTH': 50,
            'weight_decay': 0.0, 'dropout_prob': 0.2
        }
        if cfg_i.get('epochs', defaults['epochs']) != defaults['epochs']:
            info_tokens.append(f"e{cfg_i.get('epochs')}")
        if cfg_i.get('learning_rate', defaults['learning_rate']) != defaults['learning_rate']:
            info_tokens.append(f"lr{cfg_i.get('learning_rate')}")
        if cfg_i.get('hidden_size', defaults['hidden_size']) != defaults['hidden_size']:
            info_tokens.append(f"h{cfg_i.get('hidden_size')}")
        if cfg_i.get('train_batch_size', defaults['train_batch_size']) != defaults['train_batch_size']:
            info_tokens.append(f"b{cfg_i.get('train_batch_size')}")
        if cfg_i.get('MAX_ITEM_LIST_LENGTH', defaults['MAX_ITEM_LIST_LENGTH']) != defaults['MAX_ITEM_LIST_LENGTH']:
            info_tokens.append(f"max{cfg_i.get('MAX_ITEM_LIST_LENGTH')}")
        if cfg_i.get('weight_decay', defaults['weight_decay']) != defaults['weight_decay']:
            info_tokens.append(f"wd{cfg_i.get('weight_decay')}")
        if cfg_i.get('dropout_prob', defaults['dropout_prob']) != defaults['dropout_prob']:
            info_tokens.append(f"dr{cfg_i.get('dropout_prob')}")

        info_tag = "_".join(info_tokens) if info_tokens else "DEF"
        timestamp = datetime.now().strftime("%m%d%H%M%S%f")
        run_name = f"{code}_{info_tag}_{timestamp}"

        label = f"[{run_index+1}/{total}]"
        if stage_total is not None:
            label = f"[S{stage_idx+1}/{stage_total}] {label}"
        print(f"\n{label} {run_name}")

        if cfg_i.get("log_wandb", False):
            try:
                import wandb
                wandb.init(
                    project=cfg_i.get("wandb_project", "FMoE_2026"),
                    name=run_name,
                    config={
                        "model": model,
                        "dataset": dataset,
                        "eval_mode": eval_mode,
                        "stage": stage_idx + 1 if stage_idx is not None else None,
                        **{k: v for k, v in cfg_i.items() if k not in ['log_wandb', 'wandb_project']}
                    },
                    settings=wandb.Settings(allow_val_change=True),
                    reinit=True,
                )

                wandb.define_metric("train/*", step_metric="train_step")
                wandb.define_metric("valid/*", step_metric="valid_step")
                wandb.define_metric("eval/*", step_metric="valid_step")
                wandb.define_metric("test/*", step_metric="valid_step")
                wandb.define_metric("etc/*", step_metric="valid_step")
            except Exception as e:
                print(f"⚠️  wandb init failed: {e}")

        start_time = time.time()

        # Create RunLogger for FeaturedMoE runs
        _run_logger = None
        if model and model.lower() in _FEATURE_AWARE_MOE_MODELS:
            try:
                from models.FeaturedMoE.run_logger import RunLogger
                fmoe_debug_logging = bool(
                    cfg_i.get(
                        "fmoe_debug_logging",
                        cfg_i.get("log_expert_weights", False)
                        or cfg_i.get("log_expert_analysis", False),
                    )
                )
                _run_logger = RunLogger(
                    run_name=run_name,
                    config=cfg_i,
                    debug_logging=fmoe_debug_logging,
                )
                mode_name = "debug_full" if fmoe_debug_logging else "metrics_only"
                print(f"  📁 Logging to {_run_logger.output_path} ({mode_name})")
            except Exception as e:
                print(f"  ⚠️  RunLogger init failed: {e}")

        result = run_custom_training(cfg_i, run_name, save_model, run_logger=_run_logger)
        elapsed_time = result.get('elapsed_time', time.time() - start_time)
        best_valid_result = result.get('best_valid_result', {})
        if isinstance(best_valid_result, tuple):
            best_valid_result = next((x for x in best_valid_result if isinstance(x, dict)), best_valid_result[0])
        test_result = result.get('test_result', {})
        if isinstance(test_result, tuple):
            test_result = next((x for x in test_result if isinstance(x, dict)), test_result[0])

        if cfg_i.get("log_wandb", False):
            try:
                import wandb
                if wandb.run is not None:
                    val_mrr20 = best_valid_result.get('mrr@20', 0)
                    test_mrr20 = test_result.get('mrr@20', 0)
                    generalization_gap = ((val_mrr20 - test_mrr20) / val_mrr20 * 100) if val_mrr20 > 0 else 0

                    wandb.log({
                        "test/HR@5": test_result.get('hit@5', 0),
                        "test/HR@10": test_result.get('hit@10', 0),
                        "test/NDCG@10": test_result.get('ndcg@10', 0),
                        "test/MRR@20": test_result.get('mrr@20', 0),
                        "etc/time_minutes": elapsed_time / 60,
                        "etc/generalization_gap_pct": generalization_gap,
                    })

                    wandb.summary.update({
                        "best_valid_HR@10": best_valid_result.get('hit@10', 0),
                        "best_valid_NDCG@10": best_valid_result.get('ndcg@10', 0),
                        "test_HR@10": test_result.get('hit@10', 0),
                        "test_NDCG@10": test_result.get('ndcg@10', 0),
                        "time_min": elapsed_time / 60,
                    })
                    wandb.finish()
            except Exception as e:
                print(f"⚠️  wandb summary logging failed: {e}")
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.finish()
                except:
                    pass

        print(f"\n📊 Results Summary:")
        print(f"  Valid Best: HR@5={best_valid_result.get('hit@5', 0):.4f}, "
              f"HR@10={best_valid_result.get('hit@10', 0):.4f}, "
              f"MRR@20={best_valid_result.get('mrr@20', 0):.4f}")
        print(f"  Test:       HR@5={test_result.get('hit@5', 0):.4f}, "
              f"HR@10={test_result.get('hit@10', 0):.4f}, "
              f"MRR@20={test_result.get('mrr@20', 0):.4f}")
        print(f"  Time: {elapsed_time/60:.2f} min")

        metric_key = str(cfg_i.get('valid_metric', 'mrr@20')).lower()
        metric_val = best_valid_result.get(metric_key, result.get('best_valid_score', float('-inf')))
        return metric_val

    max_search = cfg.get("max_search", 200)
    stage_defs = normalize_search_stages(cfg, max_search) if args.search else []
    search_strategy = str(cfg.get("search_strategy", "grid")).lower()

    def summarize_stage_drops(
        base_search,
        stage_results,
        min_keep,
        drop_min,
        drop_max,
        drop_gap,
    ):
        print("[StageSummary] drop_min={} drop_max={} min_keep={} drop_gap={}".format(
            drop_min, drop_max, min_keep, drop_gap
        ))

        score_map = []
        for score, cfg_i in stage_results:
            score_map.append((float(score) if score is not None else float("-inf"), cfg_i))

        for key, values in base_search.items():
            if not isinstance(values, list) or len(values) == 0:
                continue
            if len(values) <= min_keep:
                print(f"[StageSummary] {key}: keep {values} (len<=min_keep)")
                continue

            value_scores = {v: [] for v in values}
            for score, cfg_i in score_map:
                if key in cfg_i:
                    v = cfg_i.get(key)
                    if v in value_scores:
                        value_scores[v].append(score)

            mean_scores = {}
            for v in values:
                scores = value_scores.get(v, [])
                mean_scores[v] = sum(scores) / len(scores) if scores else float("-inf")

            ranked = sorted(values, key=lambda v: (mean_scores[v], -values.index(v)), reverse=True)

            max_drop_allowed = min(drop_max, len(values) - min_keep)
            min_drop_allowed = min(drop_min, max_drop_allowed)
            if max_drop_allowed <= 0:
                print(f"[StageSummary] {key}: keep {values} (min_keep)")
                continue

            best_drop = min_drop_allowed
            best_gap = float("-inf")
            for drop_n in range(min_drop_allowed, max_drop_allowed + 1):
                keep_n = len(values) - drop_n
                if keep_n < min_keep or keep_n >= len(ranked):
                    continue
                gap = mean_scores[ranked[keep_n - 1]] - mean_scores[ranked[keep_n]]
                if gap > best_gap:
                    best_gap = gap
                    best_drop = drop_n

            use_drop = best_drop
            reason = "gap={:.4f} >= drop_gap".format(best_gap)
            if best_gap < drop_gap:
                use_drop = min_drop_allowed
                reason = "gap={:.4f} < drop_gap".format(best_gap)

            keep_n = len(values) - use_drop
            keep_set = set(ranked[:keep_n])
            kept = [v for v in values if v in keep_set]
            dropped = [v for v in values if v not in keep_set]
            print(f"[StageSummary] {key}: keep {kept} drop {dropped} ({reason})")

    if args.search and stage_defs:
        base_search = cfg.get("search", {})
        configs = expand_search_space(cfg, max_configs=stage_defs[0]["max_configs"])
        print(f"\n🔍 Staged search: {len(configs)} initial configs, {len(stage_defs)} stages\n")

        candidates = configs
        for stage_idx, stage in enumerate(stage_defs):
            stage_epochs = stage["epochs"]
            stage_max = stage["max_configs"]
            stage_candidates = candidates[:stage_max]
            stage_results = []
            print(f"\n== Stage {stage_idx+1}/{len(stage_defs)}: epochs={stage_epochs}, configs={len(stage_candidates)} ==")

            for i, cfg_i in enumerate(stage_candidates):
                cfg_i = cfg_i.copy()
                cfg_i["epochs"] = stage_epochs
                if stage.get("stopping_step") is not None:
                    cfg_i["stopping_step"] = stage["stopping_step"]
                metric_val = run_single(cfg_i, i, len(stage_candidates), stage_idx, len(stage_defs))
                stage_results.append((metric_val, cfg_i))

            stage_results.sort(key=lambda x: x[0], reverse=True)
            top_k = stage.get("top_k")
            if top_k is None:
                top_ratio = stage.get("top_ratio", 0.25)
                top_k = max(1, int(len(stage_results) * top_ratio))
            top_k = max(1, min(int(top_k), len(stage_results)))
            top_configs = [cfg_i for _, cfg_i in stage_results[:top_k]]

            if search_strategy in ("narrow", "narrowing") and stage_idx < len(stage_defs) - 1:
                window = stage.get("window", None)
                cat_keep = stage.get("cat_keep", None)
                min_keep = stage.get("min_keep", 2)
                drop_min = stage.get("drop_min", 1)
                drop_max = stage.get("drop_max", 2)
                drop_gap = stage.get("drop_gap", 0.002)
                summarize_stage_drops(
                    base_search,
                    stage_results,
                    min_keep=min_keep,
                    drop_min=drop_min,
                    drop_max=drop_max,
                    drop_gap=drop_gap,
                )
                narrowed = narrow_search_space(
                    base_search,
                    stage_results,
                    window=window,
                    cat_keep=cat_keep,
                    min_keep=min_keep,
                    drop_min=drop_min,
                    drop_max=drop_max,
                    drop_gap=drop_gap,
                )
                next_cfg = cfg.copy()
                next_cfg["search"] = narrowed
                next_max = stage_defs[stage_idx + 1]["max_configs"]
                candidates = expand_search_space(next_cfg, max_configs=next_max)
            else:
                candidates = top_configs
    else:
        configs = expand_search_space(cfg, max_configs=max_search) if args.search else [cfg]
        if args.search:
            print(f"\n🔍 Grid search: {len(configs)} configurations\n")

        for i, cfg_i in enumerate(configs):
            run_single(cfg_i, i, len(configs))


if __name__ == "__main__":
    main()
