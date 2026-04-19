#!/usr/bin/env python3
"""Appendix C: special-bin analysis with case-eval exports."""

from __future__ import annotations

from common import (
    LOG_ROOT,
    build_train_rows,
    common_arg_parser,
    find_completed_case_eval_row,
    parse_csv_ints,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    select_postprocess_rows,
    selected_candidates_from_args,
    special_bin_settings,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix special-bin analysis", question="special_bins")
    parser.add_argument("--postprocess-all", action="store_true")
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
    manifest = write_manifest("special_bins", rows)
    print(f"[special-bins] manifest -> {manifest}")
    rc = run_jobs(
        rows,
        question="special_bins",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run:
        return rc

    ok_rows = [row for row in read_summary_rows("special_bins") if str(row.get("status", "")).lower() == "ok"]
    selected_rows = select_postprocess_rows(ok_rows, postprocess_all=bool(args.postprocess_all))
    case_rows: list[dict[str, str]] = []
    for summary_row in selected_rows:
        existing = find_completed_case_eval_row("special_bins", summary_row)
        if existing is not None:
            case_rows.append(existing)
            continue
        bundle = run_case_eval_pipeline(
            question="special_bins",
            source_summary_row=summary_row,
            output_root=LOG_ROOT / "special_bins" / "case_eval" / str(summary_row.get("job_id", "")),
        )
        bundle["status"] = "ok"
        bundle["error"] = ""
        case_rows.append(bundle)
    write_index_rows(LOG_ROOT / "special_bins" / "special_bins_case_eval_index.csv", case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
