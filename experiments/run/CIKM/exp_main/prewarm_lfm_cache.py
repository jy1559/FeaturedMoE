#!/usr/bin/env python3
"""Pre-build LastFM RecBole dataset/session split cache for CIKM runs."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

_CIKM_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _CIKM_DIR.parents[2]
_EXP_DIR = _REPO_ROOT / "experiments"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

import recbole_patch  # noqa: F401,E402
from hydra_utils import configure_eval_sampling, enforce_v4_feature_mode, load_hydra_config  # noqa: E402
from hyperopt_tune import (  # noqa: E402
    _ensure_feature_load_columns,
    _resolve_dataset_name_case,
    _sync_model_dimensions,
    configure_data_cache,
    configure_runtime_acceleration,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prewarm LastFM dataset cache")
    p.add_argument("--max-len", type=int, default=10)
    p.add_argument("--train-batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-sample-num", type=int, default=1000)
    p.add_argument(
        "--model",
        default="sasrec",
        help="Hydra model config name to compose before building the cache.",
    )
    p.add_argument(
        "--drop-item-features",
        action="store_true",
        help="Build a no-item-feature cache for models that only need session/item/timestamp.",
    )
    p.add_argument(
        "--light-data",
        action="store_true",
        help="Use Datasets/processed/final_dataset_light instead of the feature-rich final_dataset.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = _EXP_DIR / "configs"
    dataset_root = _REPO_ROOT / "Datasets" / "processed"
    data_dir_name = "final_dataset_light" if args.light_data else "final_dataset"
    final_data_root = dataset_root / data_dir_name
    if args.light_data and args.drop_item_features:
        cache_name = "cikm_light_noitem"
    elif args.light_data:
        cache_name = "cikm_light"
    elif args.drop_item_features:
        cache_name = "cikm_final_noitem"
    else:
        cache_name = "cikm_final"
    cache_root = _EXP_DIR / "saved" / "recbole_cache" / cache_name

    overrides = [
        f"model={args.model}",
        "dataset=lastfm",
        "eval_mode=session_fixed",
        "feature_mode=final",
        f"++dataset_root={dataset_root}",
        f"++data_path={final_data_root}",
        f"++MAX_ITEM_LIST_LENGTH={args.max_len}",
        f"train_batch_size={args.train_batch_size}",
        f"eval_batch_size={args.eval_batch_size}",
        "use_gpu=false",
        "enable_data_cache=true",
        "enable_disk_data_cache=true",
        "enable_session_split_cache=true",
        "in_memory_data_cache=false",
        "cache_dataloaders=false",
        "save_dataloaders=false",
        f"large_dataset_cache_anchor_len={args.max_len}",
        f"data_cache_dir={cache_root}",
        "++eval_sampling.mode=auto",
        "++eval_sampling.auto_full_threshold=100000",
        f"++eval_sampling.sample_num={args.eval_sample_num}",
        "show_progress=false",
        "log_wandb=false",
    ]
    if args.drop_item_features:
        overrides.append("load_col.item=[]")

    cfg = load_hydra_config(config_dir, "tune_lfm_cikm", overrides)
    cfg_omega = configure_eval_sampling(OmegaConf.create(cfg))
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    enforce_v4_feature_mode(cfg)

    resolved_dataset = _resolve_dataset_name_case(cfg)
    if resolved_dataset and resolved_dataset != cfg.get("dataset"):
        cfg["dataset"] = resolved_dataset
    _sync_model_dimensions(cfg)
    _ensure_feature_load_columns(cfg)
    configure_runtime_acceleration(cfg)
    configure_data_cache(cfg)

    print("=================================================================", flush=True)
    print("  LastFM cache prewarm", flush=True)
    print(f"  model config  : {args.model}", flush=True)
    print(f"  model         : {cfg.get('model')}", flush=True)
    print(f"  drop item feat: {args.drop_item_features}", flush=True)
    print(f"  light data    : {args.light_data}", flush=True)
    print(f"  dataset       : {cfg.get('dataset')}", flush=True)
    print(f"  data_path     : {cfg.get('data_path')}", flush=True)
    print(f"  max_len       : {cfg.get('MAX_ITEM_LIST_LENGTH')}", flush=True)
    print(f"  checkpoint_dir: {cfg.get('checkpoint_dir')}", flush=True)
    print("=================================================================", flush=True)

    argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        config = Config(model=cfg["model"], dataset=cfg["dataset"], config_dict=cfg)
    finally:
        sys.argv = argv

    init_seed(config["seed"], config["reproducibility"])
    t0 = time.time()
    dataset = create_dataset(config)
    t_dataset = time.time() - t0
    t1 = time.time()
    train_data, valid_data, test_data = data_preparation(config, dataset)
    t_prepare = time.time() - t1

    print("=================================================================", flush=True)
    print("  LastFM cache prewarm DONE", flush=True)
    print(f"  dataset build : {t_dataset / 60:.1f} min", flush=True)
    print(f"  data prepare  : {t_prepare / 60:.1f} min", flush=True)
    print(f"  train batches : {len(train_data)}", flush=True)
    print(f"  valid batches : {len(valid_data)}", flush=True)
    print(f"  test batches  : {len(test_data)}", flush=True)
    cache_dir = Path(str(cfg.get("checkpoint_dir", "")))
    if cache_dir.exists():
        files = sorted(cache_dir.glob("*"))
        print(f"  cache files   : {len(files)}", flush=True)
        for path in files[:20]:
            print(f"    {path.name}  {path.stat().st_size / (1024 ** 2):.1f} MiB", flush=True)
    print("=================================================================", flush=True)


if __name__ == "__main__":
    main()
