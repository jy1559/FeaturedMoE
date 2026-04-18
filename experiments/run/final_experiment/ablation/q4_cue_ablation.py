#!/usr/bin/env python3
"""Q4 cue-family ablations."""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import common_arg_parser, build_train_rows, parse_csv_ints, q4_settings, run_jobs, selected_candidates_from_args, write_manifest  # noqa: E402


def main() -> int:
    parser = common_arg_parser("Q4 cue-family ablations", question="q4")
    args = parser.parse_args()
    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="q4",
        candidates=candidates,
        settings=q4_settings(appendix=bool(args.appendix)),
        seeds=seeds,
        max_evals=args.max_evals,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("q4", rows)
    print(f"[q4] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="q4",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
