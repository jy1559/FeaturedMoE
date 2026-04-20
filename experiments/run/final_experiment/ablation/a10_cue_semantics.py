#!/usr/bin/env python3
"""A10 cue semantics and intervention analysis: extended cue profiles + interventions."""

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
    find_completed_intervention_row,
    index_path,
    latest_manifest_under,
    parse_csv_ints,
    python_bin,
    a10_cue_profile_settings,
    a10_intervention_specs,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("A10 cue semantics and intervention analysis", question="a10")
    parser.add_argument("--reuse-q4-checkpoints", action="store_true", default=False,
                        help="Reuse Q4 full-setting checkpoints instead of training new ones.")
    parser.add_argument("--reuse-q5-interventions", action="store_true", default=False,
                        help="Reuse Q5 intervention results for the base set.")
    args = parser.parse_args()

    # --- Part 1: extended cue profile training (or reuse Q4) ---
    if not args.reuse_q4_checkpoints:
        candidates = selected_candidates_from_args(args)
        seeds = parse_csv_ints(args.seeds) or [1]
        rows = build_train_rows(
            question="a10",
            candidates=candidates,
            settings=a10_cue_profile_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
            smoke_test=bool(args.smoke_test),
            smoke_max_runs=args.smoke_max_runs,
        )
        manifest = write_manifest("a10", rows)
        print(f"[a10] manifest -> {manifest}")
        rc = run_jobs(
            rows,
            question="a10",
            gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0 or args.dry_run:
            return rc
        source_question = "a10"
    else:
        source_question = "q4"
        print("[a10] Reusing Q4 checkpoints for cue profile analysis.")

    # --- Part 2: run interventions on full-setting checkpoints ---
    intervention_rows: list[dict[str, str]] = []
    intervention_source = "q5" if args.reuse_q5_interventions else source_question
    for summary_row in read_summary_rows(intervention_source):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        setting_key = str(summary_row.get("setting_key", ""))
        if setting_key not in ("full", "route_rec_full"):
            continue
        job_id = str(summary_row.get("job_id", ""))
        existing = find_completed_intervention_row("a10", summary_row)
        if existing is not None:
            intervention_rows.append(existing)
            continue
        try:
            output_root = LOG_ROOT / "a10" / "interventions" / job_id
            cmd = [
                python_bin(),
                str(CODE_DIR / "eval_checkpoint_interventions.py"),
                "--source-result-json",
                str(summary_row["result_path"]),
                "--checkpoint-file",
                str(summary_row["checkpoint_file"]),
                "--output-root",
                str(output_root),
            ]
            subprocess.run(cmd, check=True, cwd=str(Path(__file__).resolve().parents[4]))
            intervention_manifest = latest_manifest_under(output_root, "intervention_manifest.csv")
            intervention_rows.append(
                {
                    "question": "a10",
                    "dataset": summary_row.get("dataset", ""),
                    "setting_key": setting_key,
                    "base_rank": summary_row.get("base_rank", ""),
                    "base_tag": summary_row.get("base_tag", ""),
                    "seed_id": summary_row.get("seed_id", ""),
                    "result_path": summary_row.get("result_path", ""),
                    "checkpoint_file": summary_row.get("checkpoint_file", ""),
                    "intervention_manifest": str(intervention_manifest),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            intervention_rows.append(
                {
                    "question": "a10",
                    "dataset": summary_row.get("dataset", ""),
                    "setting_key": setting_key,
                    "base_rank": summary_row.get("base_rank", ""),
                    "base_tag": summary_row.get("base_tag", ""),
                    "seed_id": summary_row.get("seed_id", ""),
                    "result_path": summary_row.get("result_path", ""),
                    "checkpoint_file": summary_row.get("checkpoint_file", ""),
                    "status": "error",
                    "error": str(exc),
                }
            )

    write_index_rows(index_path("a10", "a10_intervention_index.csv"), intervention_rows)

    # --- Part 3: case-eval for cue-space profiling ---
    case_rows: list[dict[str, str]] = []
    for summary_row in read_summary_rows(source_question):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        existing = find_completed_case_eval_row("a10", summary_row)
        if existing is not None:
            case_rows.append(existing)
            continue
        try:
            bundle = run_case_eval_pipeline(
                question="a10",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "a10" / "case_eval" / str(summary_row.get("job_id", "")),
                skip_by_group=bool(args.case_eval_fast),
            )
            case_rows.append(bundle)
        except Exception as exc:
            case_rows.append(
                {
                    "question": "a10",
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
    write_index_rows(index_path("a10", "a10_case_eval_index.csv"), case_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
