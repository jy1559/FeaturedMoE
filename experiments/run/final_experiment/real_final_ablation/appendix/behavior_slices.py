#!/usr/bin/env python3
"""Appendix I: extended behavioral slices."""

from __future__ import annotations

import subprocess

from common import (
    LOG_ROOT,
    build_postprocess_error_row,
    build_train_rows,
    behavior_slice_settings,
    common_arg_parser,
    find_completed_case_eval_row,
    parse_csv_ints,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix behavior slices", question="behavior_slices")
    parser.add_argument("--postprocess-all", action="store_true", default=True)
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="behavior_slices",
        candidates=candidates,
        settings=behavior_slice_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    if not args.postprocess_only:
        manifest = write_manifest("behavior_slices", rows)
        print(f"[behavior-slices] manifest -> {manifest}")
        rc = run_jobs(
            rows,
            question="behavior_slices",
            gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0 or args.dry_run:
            return rc

    case_rows: list[dict[str, str]] = []
    ok_rows = [row for row in read_summary_rows("behavior_slices") if str(row.get("status", "")).lower() == "ok"]
    for summary_row in ok_rows:
        existing = find_completed_case_eval_row("behavior_slices", summary_row)
        if existing is not None:
            case_rows.append(existing)
            continue
        try:
            bundle = run_case_eval_pipeline(
                question="behavior_slices",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "behavior_slices" / "case_eval" / str(summary_row.get("job_id", "")),
            )
            bundle["status"] = "ok"
            bundle["error"] = ""
        except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
            bundle = build_postprocess_error_row(
                question="behavior_slices",
                source_summary_row=summary_row,
                error=exc,
            )
        case_rows.append(bundle)
    write_index_rows(LOG_ROOT / "behavior_slices" / "behavior_slices_case_eval_index.csv", case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
