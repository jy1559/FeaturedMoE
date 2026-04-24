#!/usr/bin/env python3
"""Appendix E: sparse routing design variants."""

from __future__ import annotations

from common import (
    build_train_rows,
    common_arg_parser,
    parse_csv_ints,
    run_jobs,
    selected_candidates_from_args,
    sparse_settings,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix sparse-routing variants", question="sparse")
    args = parser.parse_args()
    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="sparse",
        candidates=candidates,
        settings=sparse_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("sparse", rows)
    print(f"[sparse] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="sparse",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
