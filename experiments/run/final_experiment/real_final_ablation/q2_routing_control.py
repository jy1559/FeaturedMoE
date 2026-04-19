#!/usr/bin/env python3
"""Q2: What should control routing?

Compares five routing/control variants:
  - Shared FFN
  - Hidden Only
  - Feature Fusion Bias
  - Mixed Hidden+Behavior
  - Behavior-Guided
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    common_arg_parser,
    build_train_rows,
    index_path,
    parse_csv_ints,
    q2_settings,
    run_jobs,
    selected_candidates_from_args,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Q2 routing control ablations", question="q2")
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="q2",
        candidates=candidates,
        settings=q2_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("q2", rows)
    print(f"[q2] manifest -> {manifest}")

    rc = run_jobs(
        rows,
        question="q2",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run:
        return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
