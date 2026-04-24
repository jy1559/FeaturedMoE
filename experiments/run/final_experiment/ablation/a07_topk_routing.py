#!/usr/bin/env python3
"""A07 top-k routing regime ablations."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import common_arg_parser, build_train_rows, parse_csv_ints, a07_settings, run_jobs, selected_candidates_from_args, write_manifest  # noqa: E402


def main() -> int:
    parser = common_arg_parser("A07 top-k routing regime ablations", question="a07")
    args = parser.parse_args()
    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="a07",
        candidates=candidates,
        settings=a07_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("a07", rows)
    print(f"[a07] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="a07",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
