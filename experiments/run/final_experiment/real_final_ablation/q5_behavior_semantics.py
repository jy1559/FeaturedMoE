#!/usr/bin/env python3
"""Q5 behavioral semantics: final RouteRec plus case/intervention exports."""

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
    find_completed_case_eval_row,
    find_completed_intervention_row,
    index_path,
    latest_manifest_under,
    parse_csv_ints,
    python_bin,
    q5_train_settings,
    read_summary_rows,
    run_case_eval_pipeline,
    run_jobs,
    selected_candidates_from_args,
    write_index_rows,
    write_manifest,
)


def main() -> int:
    parser = common_arg_parser("Q5 behavior semantics", question="q5")
    parser.add_argument(
        "--postprocess-all",
        action="store_true",
        help="Run case-eval/intervention for every completed run (default: only one baseline run per dataset).",
    )
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    seeds = parse_csv_ints(args.seeds) or [1]
    rows = build_train_rows(
        question="q5",
        candidates=candidates,
        settings=q5_train_settings(),
        seeds=seeds,
        max_evals=args.max_evals,
        max_run_hours=args.max_run_hours,
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

    ok_rows = [row for row in read_summary_rows("q5") if str(row.get("status", "")).lower() == "ok"]

    # By default, postprocess only a single baseline run per dataset.
    # This keeps the main-body Q5 artifacts lightweight since the case maps/interventions
    # are qualitative; we don't need to multiply them across seeds/top-k configs.
    selected_rows = ok_rows
    if not args.postprocess_all:
        by_dataset: dict[str, list[dict[str, str]]] = {}
        for row in ok_rows:
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

    case_rows: list[dict[str, str]] = []
    intervention_rows: list[dict[str, str]] = []
    for summary_row in selected_rows:
        job_id = str(summary_row.get("job_id", ""))
        checkpoint_path = str(summary_row.get("checkpoint_file", "") or "").strip()

        existing_case = find_completed_case_eval_row("q5", summary_row)
        if existing_case is not None:
            case_rows.append(existing_case)
        else:
            try:
                bundle = run_case_eval_pipeline(
                    question="q5",
                    source_summary_row=summary_row,
                    output_root=LOG_ROOT / "q5" / "case_eval" / job_id,
                    skip_by_group=bool(args.case_eval_fast),
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

        existing_intervention = find_completed_intervention_row("q5", summary_row)
        if existing_intervention is not None:
            intervention_rows.append(existing_intervention)
        else:
            try:
                output_root = LOG_ROOT / "q5" / "interventions" / job_id
                cmd = [
                    python_bin(),
                    str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "eval_checkpoint_interventions.py"),
                    "--source-result-json",
                    str(summary_row["result_path"]),
                    "--checkpoint-file",
                    str(summary_row["checkpoint_file"]),
                    "--output-root",
                    str(output_root),
                ]
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
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

        case_ok = any(
            str(row.get("status", "")).lower() == "ok"
            and str(row.get("result_path", "") or "") == str(summary_row.get("result_path", "") or "")
            for row in case_rows
        )
        intervention_ok = any(
            str(row.get("status", "")).lower() == "ok"
            and str(row.get("result_path", "") or "") == str(summary_row.get("result_path", "") or "")
            for row in intervention_rows
        )
        if case_ok and intervention_ok:
            _cleanup_checkpoint(checkpoint_path)

    write_index_rows(index_path("q5", "q5_case_eval_index.csv"), case_rows)
    write_index_rows(index_path("q5", "q5_intervention_index.csv"), intervention_rows)
    return 0


def _cleanup_checkpoint(path_str: str) -> None:
    path_text = str(path_str or "").strip()
    if not path_text:
        return
    try:
        path = Path(path_text).expanduser().resolve()
    except Exception:
        return
    try:
        if path.exists():
            path.unlink()
    except Exception as exc:
        print(f"[q5] WARN could not delete checkpoint: {path} ({exc})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
