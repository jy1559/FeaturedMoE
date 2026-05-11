#!/usr/bin/env python3
"""CIKM 2026 - stable LastFM full-dataset runner.

LastFM has 547K items, so CE training and full-sort evaluation allocate
batch_size x n_items score tensors. This runner is intentionally sequential
and uses conservative batch sizes plus OOM retry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CIKM_DIR = Path(__file__).resolve().parent.parent
if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))

from common import BASELINE_MODELS, RESULT_ROOT, ROUTE_MODEL, run_jobs_queued  # noqa: E402

DEFAULT_MODELS = [
    "sasrec",
    "gru4rec",
    "tisasrec",
    "duorec",
    "bsarec",
    "fearec",
    "difsr",
    "fame",
    "fdsa",
    ROUTE_MODEL,
]

SUMMARY_CSV = RESULT_ROOT / "lastfm_stable_summary.csv"
RUN_AXIS = "cikm_lfm_stable"
RUN_PHASE = "P0_LFM_STABLE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LastFM full dataset safely, one model at a time")
    p.add_argument("--gpu", default="0", help="Single GPU id to use; sequential by design")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=[*BASELINE_MODELS, ROUTE_MODEL],
        help="Models to run sequentially",
    )
    p.add_argument("--max-evals", type=int, default=1, help="Hyperopt trials per model")
    p.add_argument("--epochs", type=int, default=100, help="Max epochs per trial")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=10, help="Override MAX_ITEM_LIST_LENGTH")
    p.add_argument("--eval-sample-num", type=int, default=1000)
    p.add_argument("--oom-retry-limit", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stable_overrides = [
        f"max_evals={args.max_evals}",
        f"tune_epochs={args.epochs}",
        f"tune_patience={args.patience}",
        f"oom_retry_limit={args.oom_retry_limit}",
        f"train_batch_size={args.train_batch_size}",
        f"eval_batch_size={args.eval_batch_size}",
        "enable_data_cache=false",
        "enable_disk_data_cache=false",
        "in_memory_data_cache=false",
        "cache_dataloaders=false",
        f"++MAX_ITEM_LIST_LENGTH={args.max_len}",
        "++eval_sampling.mode=auto",
        "++eval_sampling.auto_full_threshold=100000",
        f"++eval_sampling.sample_num={args.eval_sample_num}",
        "++log_unseen_target_metrics=false",
    ]

    jobs = [
        {
            "dataset": "lastfm",
            "model": model,
            "extra_overrides": list(stable_overrides),
        }
        for model in args.models
    ]

    print("[main_lfm_stable] sequential LastFM full-dataset run", flush=True)
    print(f"  gpu       : {args.gpu}", flush=True)
    print(f"  models    : {args.models}", flush=True)
    print(f"  max_evals : {args.max_evals}", flush=True)
    print(f"  batch     : train={args.train_batch_size} eval={args.eval_batch_size}", flush=True)
    print(f"  max_len   : {args.max_len}", flush=True)
    print(f"  oom_retry : {args.oom_retry_limit}", flush=True)
    print(f"  summary   : {SUMMARY_CSV}", flush=True)

    run_jobs_queued(
        jobs,
        gpus=[str(args.gpu)],
        summary_path=SUMMARY_CSV,
        run_axis=RUN_AXIS,
        run_phase=RUN_PHASE,
    )

    print(f"\n[main_lfm_stable] DONE. Results -> {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
