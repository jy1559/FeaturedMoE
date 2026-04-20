#!/usr/bin/env python3
"""Appendix K: optional transfer scaffold."""

from __future__ import annotations

from common import (
    build_train_rows,
    common_arg_parser,
    parse_csv_ints,
    run_jobs,
    selected_candidates_from_args,
    transfer_settings,
    write_note_manifest,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix optional transfer", question="transfer")
    parser.add_argument("--run-low-resource", action="store_true")
    args = parser.parse_args()

    if not args.run_low_resource:
        path = write_note_manifest(
            "transfer",
            "Transfer is scaffolded but disabled by default. Re-run with --run-low-resource to populate the appendix slot.",
        )
        print(f"[transfer] placeholder -> {path}")
        return 0

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="transfer",
        candidates=candidates,
        settings=transfer_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("transfer", rows)
    print(f"[transfer] manifest -> {manifest}")
    return run_jobs(
        rows,
        question="transfer",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
