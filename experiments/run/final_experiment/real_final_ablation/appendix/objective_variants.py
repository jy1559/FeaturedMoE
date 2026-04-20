#!/usr/bin/env python3
"""Appendix F: objective and regularization variants."""

from __future__ import annotations

from common import (
    build_train_rows,
    common_arg_parser,
    objective_settings,
    parse_csv_ints,
    run_jobs,
    selected_candidates_from_args,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix objective variants", question="objective")
    args = parser.parse_args()
    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="objective",
        candidates=candidates,
        settings=objective_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("objective", rows)
    print(f"[objective] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="objective",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
