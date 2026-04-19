#!/usr/bin/env python3
"""A08 behavior case studies: train full RouteRec, then run case-eval + diagnostics."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    common_arg_parser,
    build_train_rows,
    find_completed_case_eval_row,
    index_path,
    parse_csv_ints,
    a08_train_settings,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("A08 behavior case studies", question="a08")
    parser.add_argument("--reuse-q2-checkpoints", action="store_true", default=False,
                        help="Skip training; reuse route_rec_full checkpoints from Q2 summary.")
    args = parser.parse_args()

    if not args.reuse_q2_checkpoints:
        candidates = selected_candidates_from_args(args)
        seeds = parse_csv_ints(args.seeds) or [1]
        rows = build_train_rows(
            question="a08",
            candidates=candidates,
            settings=a08_train_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
            smoke_test=bool(args.smoke_test),
            smoke_max_runs=args.smoke_max_runs,
        )
        manifest = write_manifest("a08", rows)
        print(f"[a08] manifest -> {manifest}")
        rc = run_jobs(
            rows,
            question="a08",
            gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0 or args.dry_run:
            return rc
        source_question = "a08"
    else:
        source_question = "q2"
        print("[a08] Reusing Q2 route_rec_full checkpoints for case-eval.")

    case_rows: list[dict[str, str]] = []
    for summary_row in read_summary_rows(source_question):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        if source_question == "q2" and str(summary_row.get("setting_key", "")) != "route_rec_full":
            continue
        existing = find_completed_case_eval_row("a08", summary_row)
        if existing is not None:
            case_rows.append(existing)
            continue
        try:
            bundle = run_case_eval_pipeline(
                question="a08",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "a08" / "case_eval" / str(summary_row.get("job_id", "")),
                skip_by_group=bool(args.case_eval_fast),
            )
            case_rows.append(bundle)
        except Exception as exc:
            case_rows.append(
                {
                    "question": "a08",
                    "dataset": summary_row.get("dataset", ""),
                    "setting_key": summary_row.get("setting_key", ""),
                    "base_rank": summary_row.get("base_rank", ""),
                    "base_tag": summary_row.get("base_tag", ""),
                    "seed_id": summary_row.get("seed_id", ""),
                    "result_path": summary_row.get("result_path", ""),
                    "checkpoint_file": summary_row.get("checkpoint_file", ""),
                    "status": "error",
                    "error": str(exc),
                }
            )
    write_index_rows(index_path("a08", "a08_case_eval_index.csv"), case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
