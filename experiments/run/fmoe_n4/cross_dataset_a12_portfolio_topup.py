#!/usr/bin/env python3
"""CrossDataset A12 portfolio top-up runner.

Goal:
- Continue after the current 12/12/6/6/4 plan with extra templates prioritized by
  under-explored datasets:
  - amazon_beauty: +4  (total 16)
  - foursquare:    +4  (total 16)
  - movielens1m:   +6  (total 12)
  - retail_rocket: +6  (total 12)
  - lastfm0.03:    +8  (total 12)

Strategy:
- Reuse existing compiled portfolio engine (`cross_dataset_a12_portfolio.py`) for
  compatibility and logging layout.
- Run in 3 sequential groups so quick datasets finish first and long-tail (lastfm)
  runs later while GPUs stay occupied.
- Keep `tpe` search, but adjust budget per group for speed/coverage balance.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


THIS_FILE = Path(__file__).resolve()
FMOE_N4_DIR = THIS_FILE.parent
EXPERIMENTS_DIR = FMOE_N4_DIR.parents[1]
PORTFOLIO_ENTRY = EXPERIMENTS_DIR / "run" / "fmoe_n4" / "cross_dataset_a12_portfolio.py"


@dataclass(frozen=True)
class GroupPlan:
    name: str
    datasets_csv: str
    counts_csv: str
    max_evals: int
    tune_epochs: int
    batch_size: int
    eval_batch_size: int


def _default_python_bin() -> str:
    env_bin = os.environ.get("RUN_PYTHON_BIN")
    if env_bin:
        return env_bin
    venv_bin = Path("/venv/FMoE/bin/python")
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)
    return sys.executable


def _build_plan() -> list[GroupPlan]:
    return [
        GroupPlan(
            name="fast_ab_fs_topup",
            datasets_csv="amazon_beauty,foursquare",
            counts_csv="amazon_beauty:16,foursquare:16",
            max_evals=10,
            tune_epochs=80,
            batch_size=4096,
            eval_batch_size=6144,
        ),
        GroupPlan(
            name="mid_ml_rr_topup",
            datasets_csv="movielens1m,retail_rocket",
            counts_csv="movielens1m:12,retail_rocket:12",
            max_evals=10,
            tune_epochs=100,
            batch_size=4096,
            eval_batch_size=6144,
        ),
        GroupPlan(
            name="tail_lastfm_topup",
            datasets_csv="lastfm0.03",
            counts_csv="lastfm0.03:12",
            max_evals=9,
            tune_epochs=110,
            batch_size=3072,
            eval_batch_size=4096,
        ),
    ]


def _cmd_for_group(plan: GroupPlan, args: argparse.Namespace) -> list[str]:
    cmd = [
        args.python_bin,
        str(PORTFOLIO_ENTRY.relative_to(EXPERIMENTS_DIR)),
        "--datasets",
        plan.datasets_csv,
        "--dataset-template-counts",
        plan.counts_csv,
        "--max-evals",
        str(plan.max_evals),
        "--tune-epochs",
        str(plan.tune_epochs),
        "--batch-size",
        str(plan.batch_size),
        "--eval-batch-size",
        str(plan.eval_batch_size),
        "--search-algo",
        args.search_algo,
        "--gpus",
        args.gpus,
        "--seeds",
        args.seeds,
    ]
    return cmd


def _run_group(cmd: Sequence[str], dry_run: bool) -> int:
    printable = " ".join(cmd)
    print(f"[RUN] {printable}")
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(EXPERIMENTS_DIR), check=False)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run CrossDataset A12 top-up batches")
    ap.add_argument("--python-bin", default=_default_python_bin())
    ap.add_argument("--gpus", default=os.environ.get("N4_GPUS", "0,1,2,3,4,5,6,7"))
    ap.add_argument("--seeds", default=os.environ.get("N4_SEEDS", "1"))
    ap.add_argument("--search-algo", default=os.environ.get("N4_SEARCH_ALGO", "tpe"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not PORTFOLIO_ENTRY.exists():
        print(f"error: missing portfolio entry: {PORTFOLIO_ENTRY}", file=sys.stderr)
        return 1

    print("[TOPUP_PLAN] CrossDataset A12 extra templates: +4/+4/+6/+6/+8")
    print(f"[TOPUP_PLAN] gpus={args.gpus} seeds={args.seeds} search_algo={args.search_algo}")

    for plan in _build_plan():
        print(
            "[GROUP] "
            f"name={plan.name} datasets={plan.datasets_csv} counts={plan.counts_csv} "
            f"max_evals={plan.max_evals} epochs={plan.tune_epochs} "
            f"batch={plan.batch_size}/{plan.eval_batch_size}"
        )
        rc = _run_group(_cmd_for_group(plan, args), dry_run=args.dry_run)
        if rc != 0:
            print(f"[GROUP_FAIL] {plan.name} rc={rc}")
            if not args.continue_on_error:
                return rc

    print("[DONE] CrossDataset A12 top-up plan finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
