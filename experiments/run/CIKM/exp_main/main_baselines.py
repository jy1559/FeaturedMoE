#!/usr/bin/env python3
"""CIKM 2026 – main table baselines (P0).

Runs 9 baselines on KuaiRec then lastfm using narrow hyperopt search.
Results saved to CIKM/results/main_baselines_summary.csv.

Usage:
    python main_baselines.py [--gpus 0] [--datasets KuaiRec lastfm] [--models sasrec gru4rec ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve common.py one directory up
_CIKM_DIR = Path(__file__).resolve().parent.parent
if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))

from common import (  # noqa: E402
    BASELINE_MODELS,
    DATASETS,
    RESULT_ROOT,
    run_jobs_queued,
)

SUMMARY_CSV = RESULT_ROOT / "main_baselines_summary.csv"
RUN_AXIS    = "cikm_main_baselines"
RUN_PHASE   = "P0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CIKM baseline experiments")
    p.add_argument("--gpus",     nargs="+", default=["0"],
                   help="GPU IDs to use (jobs are round-robin distributed)")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS,
                   help="Datasets to run (default: KuaiRec lastfm)")
    p.add_argument("--models",   nargs="+", default=BASELINE_MODELS,
                   choices=BASELINE_MODELS,
                   help="Models to run (default: all 9 baselines)")
    return p.parse_args()


def build_jobs(datasets: list[str], models: list[str]) -> list[dict]:
    """Build job list: KuaiRec first, then lastfm; model order preserved."""
    jobs = []
    for dataset in datasets:
        for model in models:
            jobs.append({"dataset": dataset, "model": model})
    return jobs


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args.datasets, args.models)

    print(f"[main_baselines] {len(jobs)} jobs  gpus={args.gpus}", flush=True)
    print(f"  datasets : {args.datasets}", flush=True)
    print(f"  models   : {args.models}", flush=True)
    print(f"  summary  : {SUMMARY_CSV}", flush=True)

    run_jobs_queued(
        jobs,
        gpus=args.gpus,
        summary_path=SUMMARY_CSV,
        run_axis=RUN_AXIS,
        run_phase=RUN_PHASE,
    )

    print(f"\n[main_baselines] DONE. Results → {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
