#!/usr/bin/env python3
"""Appendix J: targeted interventions and qualitative cases."""

from __future__ import annotations

from common import (
    LOG_ROOT,
    build_train_rows,
    cases_settings,
    common_arg_parser,
    find_completed_case_eval_row,
    find_completed_intervention_row,
    parse_csv_ints,
    read_summary_rows,
    run_case_eval_pipeline,
    run_interventions,
    run_jobs,
    select_postprocess_rows,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Appendix targeted interventions", question="cases")
    parser.add_argument("--postprocess-all", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="cases",
        candidates=candidates,
        settings=cases_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("cases", rows)
    print(f"[cases] manifest -> {manifest}")
    rc = run_jobs(
        rows,
        question="cases",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run:
        return rc

    ok_rows = [row for row in read_summary_rows("cases") if str(row.get("status", "")).lower() == "ok"]
    selected_rows = select_postprocess_rows(ok_rows, postprocess_all=bool(args.postprocess_all))

    case_rows: list[dict[str, str]] = []
    intervention_rows: list[dict[str, str]] = []
    for summary_row in selected_rows:
        existing_case = find_completed_case_eval_row("cases", summary_row)
        if existing_case is not None:
            case_rows.append(existing_case)
        else:
            bundle = run_case_eval_pipeline(
                question="cases",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "cases" / "case_eval" / str(summary_row.get("job_id", "")),
            )
            bundle["status"] = "ok"
            bundle["error"] = ""
            case_rows.append(bundle)

        existing_intervention = find_completed_intervention_row("cases", summary_row)
        if existing_intervention is not None:
            intervention_rows.append(existing_intervention)
        else:
            intervention_rows.append(
                run_interventions(
                    question="cases",
                    summary_row=summary_row,
                    output_root=LOG_ROOT / "cases" / "interventions" / str(summary_row.get("job_id", "")),
                )
            )

    write_index_rows(LOG_ROOT / "cases" / "cases_case_eval_index.csv", case_rows)
    write_index_rows(LOG_ROOT / "cases" / "cases_intervention_index.csv", intervention_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
