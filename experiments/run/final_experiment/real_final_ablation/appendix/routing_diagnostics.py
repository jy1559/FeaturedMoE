#!/usr/bin/env python3
"""Appendix H: consolidated routing diagnostics."""

from __future__ import annotations

import subprocess

from common import (
    LOG_ROOT,
    build_postprocess_error_row,
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
        if question in {"sparse", "structural"}:
            # Only the representative variants export dedicated checkpoints for post-hoc
            # case-eval. The other rows are still used by the aggregate notebook CSVs, but
            # they should not be scheduled here as they only produce deterministic errors.
            selected_rows = [row for row in ok_rows if str(row.get("checkpoint_file", "")).strip()]
        else:
            checkpoint_rows = [row for row in ok_rows if str(row.get("checkpoint_file", "")).strip()]
            source_pool = checkpoint_rows or ok_rows
            selected_rows = select_postprocess_rows(source_pool, postprocess_all=bool(args.postprocess_all))
        for summary_row in selected_rows:
            existing = find_completed_case_eval_row("diagnostics", summary_row)
            if existing is not None:
                case_rows.append(existing)
                continue
            try:
                bundle = run_case_eval_pipeline(
                    question="diagnostics",
                    source_summary_row=summary_row,
                    output_root=LOG_ROOT / "diagnostics" / question / str(summary_row.get("job_id", "")),
                    skip_by_group=False,
                )
                bundle["status"] = "ok"
                bundle["error"] = ""
                bundle["source_question"] = question
            except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
                bundle = build_postprocess_error_row(
                    question="diagnostics",
                    source_summary_row=summary_row,
                    error=exc,
                    source_question=question,
                )
            case_rows.append(bundle)
    write_index_rows(LOG_ROOT / "diagnostics" / "diagnostics_case_eval_index.csv", case_rows)
    write_note_manifest("diagnostics", "Collected representative case-eval bundles for sparse/structural/behavior appendix diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
