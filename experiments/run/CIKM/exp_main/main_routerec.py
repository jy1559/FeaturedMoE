#!/usr/bin/env python3
"""CIKM 2026 – RouteRec (FeaturedMoE_N3) main table runs (P0).

Runs featured_moe_n3 on KuaiRec then lastfm using narrow hyperopt search.
Results saved to CIKM/results/main_routerec_summary.csv.

Usage:
    python main_routerec.py [--gpus 0] [--datasets KuaiRec lastfm]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_CIKM_DIR = Path(__file__).resolve().parent.parent
if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))

from common import (  # noqa: E402
    DATASETS,
    RESULT_ROOT,
    ROUTE_MODEL,
    run_jobs_queued,
)

SUMMARY_CSV = RESULT_ROOT / "main_routerec_summary.csv"
RUN_AXIS    = "cikm_main_routerec"
RUN_PHASE   = "P0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CIKM RouteRec experiments")
    p.add_argument("--gpus",     nargs="+", default=["0"],
                   help="GPU IDs (default: [0])")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   choices=DATASETS,
                   help="Datasets (default: KuaiRec lastfm)")
    return p.parse_args()


def build_jobs(datasets: list[str]) -> list[dict]:
    return [{"dataset": ds, "model": ROUTE_MODEL} for ds in datasets]


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args.datasets)

    print(f"[main_routerec] {len(jobs)} jobs  gpus={args.gpus}", flush=True)
    print(f"  datasets : {args.datasets}", flush=True)
    print(f"  model    : {ROUTE_MODEL}", flush=True)
    print(f"  summary  : {SUMMARY_CSV}", flush=True)

    run_jobs_queued(
        jobs,
        gpus=args.gpus,
        summary_path=SUMMARY_CSV,
        run_axis=RUN_AXIS,
        run_phase=RUN_PHASE,
    )

    print(f"\n[main_routerec] DONE. Results → {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
