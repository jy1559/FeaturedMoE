#!/usr/bin/env python3
"""Q5 behavior-regime checkpoint evals and interventions."""

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
    index_path,
    latest_manifest_under,
    parse_csv_ints,
    q5_train_settings,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    python_bin,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Q5 behavior regime + interventions", question="q5")
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="q5",
        candidates=candidates,
        settings=q5_train_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("q5", rows)
    print(f"[q5] manifest -> {manifest}")

    rc = run_jobs(
        rows,
        question="q5",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run:
        return rc

    case_rows: list[dict[str, str]] = []
    intervention_rows: list[dict[str, str]] = []
    for summary_row in read_summary_rows("q5"):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        job_id = str(summary_row.get("job_id", ""))

        try:
            bundle = run_case_eval_pipeline(
                question="q5",
                source_summary_row=summary_row,
                output_root=LOG_ROOT / "q5" / "case_eval" / job_id,
            )
            case_rows.append(bundle)
        except Exception as exc:
            case_rows.append(
                {
                    "question": "q5",
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

        try:
            output_root = LOG_ROOT / "q5" / "interventions" / job_id
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
                    "question": "q5",
                    "dataset": summary_row.get("dataset", ""),
                    "setting_key": summary_row.get("setting_key", ""),
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
                    "question": "q5",
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

    write_index_rows(index_path("q5", "q5_case_eval_index.csv"), case_rows)
    write_index_rows(index_path("q5", "q5_intervention_index.csv"), intervention_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
