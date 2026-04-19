#!/usr/bin/env python3
"""Appendix H: consolidated routing diagnostics."""

from __future__ import annotations

from common import (
    LOG_ROOT,
    find_completed_case_eval_row,
    read_summary_rows,
    run_case_eval_pipeline,
    select_postprocess_rows,
    write_index_rows,
    write_note_manifest,
)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Appendix routing diagnostics postprocess")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--postprocess-all", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.dry_run:
        write_note_manifest("diagnostics", "Dry-run only: routing diagnostics postprocess was not executed.")
        return 0

    source_questions = ("sparse", "structural", "special_bins", "cases")
    case_rows: list[dict[str, str]] = []
    for question in source_questions:
        ok_rows = [row for row in read_summary_rows(question) if str(row.get("status", "")).lower() == "ok"]
        for summary_row in select_postprocess_rows(ok_rows, postprocess_all=bool(args.postprocess_all)):
            existing = find_completed_case_eval_row("diagnostics", summary_row)
            if existing is not None:
                case_rows.append(existing)
                continue
            bundle = run_case_eval_pipeline(
                question="diagnostics",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "diagnostics" / question / str(summary_row.get("job_id", "")),
                skip_by_group=False,
            )
            bundle["status"] = "ok"
            bundle["error"] = ""
            bundle["source_question"] = question
            case_rows.append(bundle)
    write_index_rows(LOG_ROOT / "diagnostics" / "diagnostics_case_eval_index.csv", case_rows)
    write_note_manifest("diagnostics", "Collected representative case-eval bundles for sparse/structural/behavior appendix diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
