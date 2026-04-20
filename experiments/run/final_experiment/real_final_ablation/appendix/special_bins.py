#!/usr/bin/env python3
"""Appendix C: special-bin analysis with case-eval exports."""

from __future__ import annotations

from common import (
    build_train_rows,
    build_minimal_baseline_rows,
    common_arg_parser,
    parse_csv_ints,
    run_jobs,
    parse_baseline_models,
    selected_candidates_from_args,
    special_bin_settings,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix special-bin analysis", question="special_bins")
    parser.add_argument("--baseline-models", default="sasrec")
    parser.add_argument("--baseline-seeds", default="1")
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="special_bins",
        candidates=candidates,
        settings=special_bin_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    baseline_models = parse_baseline_models(args.baseline_models)
    baseline_seeds = parse_csv_ints(args.baseline_seeds) or [1]
    rows.extend(
        build_minimal_baseline_rows(
            question="special_bins",
            candidates=candidates,
            baseline_models=baseline_models,
            seeds=baseline_seeds,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            smoke_test=bool(args.smoke_test),
            smoke_max_runs=args.smoke_max_runs,
        )
    )
    manifest = write_manifest("special_bins", rows)
    print(f"[special-bins] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="special_bins",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
