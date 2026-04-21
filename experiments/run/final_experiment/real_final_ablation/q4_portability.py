#!/usr/bin/env python3
"""Q4 reduced-cue portability ablations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    REPO_ROOT,
    build_train_rows,
    common_arg_parser,
    find_completed_intervention_row,
    index_path,
    latest_manifest_under,
    parse_csv_ints,
    python_bin,
    q4_portability_settings,
    read_summary_rows,
    resolve_result_artifacts,
    run_jobs,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def _postprocess(args) -> int:
    ok_rows = [row for row in read_summary_rows("q4_portability") if str(row.get("status", "")).lower() == "ok"]
    full_rows = [row for row in ok_rows if str(row.get("setting_key", "")).strip().lower() == "full"]

    selected_rows = full_rows
    if not args.postprocess_all:
        by_dataset: dict[str, list[dict[str, str]]] = {}
        for row in full_rows:
            by_dataset.setdefault(str(row.get("dataset", "")).strip(), []).append(row)

        def _as_int(text: str, default: int = 10**9) -> int:
            try:
                return int(str(text).strip())
            except Exception:
                return default

        picked: list[dict[str, str]] = []
        for dataset, rows_ds in by_dataset.items():
            rows_sorted = sorted(rows_ds, key=lambda r: (_as_int(r.get("base_rank", "")), _as_int(r.get("seed_id", ""))))
            if rows_sorted:
                picked.append(rows_sorted[0])
        selected_rows = picked

    intervention_rows: list[dict[str, str]] = []
    for summary_row in selected_rows:
        job_id = str(summary_row.get("job_id", ""))
        existing_intervention = find_completed_intervention_row("q4_portability", summary_row)
        if existing_intervention is not None:
            intervention_rows.append(existing_intervention)
            continue
        try:
            result_path_obj, _payload, payload_checkpoint = resolve_result_artifacts(
                summary_row,
                require_checkpoint=True,
            )
            output_root = LOG_ROOT / "q4_portability" / "efficacy" / job_id
            cmd = [
                python_bin(),
                str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "real_final_ablation" / "q4_eval_feature_efficacy.py"),
                "--source-result-json",
                str(result_path_obj),
                "--checkpoint-file",
                str(payload_checkpoint),
                "--output-root",
                str(output_root),
            ]
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
            intervention_manifest = latest_manifest_under(output_root, "q4_efficacy_manifest.csv")
            intervention_rows.append(
                {
                    "question": "q4_portability",
                    "dataset": summary_row.get("dataset", ""),
                    "setting_key": summary_row.get("setting_key", ""),
                    "base_rank": summary_row.get("base_rank", ""),
                    "base_tag": summary_row.get("base_tag", ""),
                    "seed_id": summary_row.get("seed_id", ""),
                    "result_path": str(result_path_obj),
                    "checkpoint_file": str(payload_checkpoint),
                    "intervention_manifest": str(intervention_manifest),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            intervention_rows.append(
                {
                    "question": "q4_portability",
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

    write_index_rows(index_path("q4_portability", "q4_efficacy_index.csv"), intervention_rows)
    return 0


def main() -> int:
    parser = common_arg_parser("Q4 reduced-cue portability", question="q4_portability")
    parser.add_argument(
        "--postprocess-all",
        action="store_true",
        help="Run eval-only cue-efficacy for every completed full-cue run (default: only one full run per dataset).",
    )
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="q4_portability",
        candidates=candidates,
        settings=q4_portability_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        lr_mode=args.lr_mode,
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=args.smoke_max_runs,
    )
    manifest = write_manifest("q4_portability", rows)
    print(f"[q4_portability] manifest -> {manifest}")
    rc = run_jobs(
        rows,
        question="q4_portability",
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )
    if rc != 0 or args.dry_run:
        return rc
    return _postprocess(args)


if __name__ == "__main__":
    raise SystemExit(main())
