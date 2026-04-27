#!/usr/bin/env python3
"""A09 boundary conditions: low-data budget curves + lightweight transfer portability."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    EXP_DIR,
    LOG_ROOT,
    REPO_ROOT,
    ROUTE_MODEL,
    ROUTE_MODEL_OVERRIDE,
    TRANSFER_PAIR_SPECS,
    A09_SHARED_HPARAM_PRESETS,
    a09_low_data_settings,
    a09_source_checkpoint_path,
    a09_transfer_settings,
    append_csv_row,
    build_a09_source_rows,
    build_a09_target_rows,
    build_route_command,
    build_summary_row,
    common_arg_parser,
    build_train_rows,
    describe_job,
    ensure_dir,
    extract_error_tail,
    has_run_status_end_normal,
    hydra_literal,
    install_signal_handlers,
    load_result_payload,
    log_path_for_row,
    now_utc,
    parse_csv_ints,
    parse_csv_list,
    parse_result_path_from_log,
    python_bin,
    read_json,
    result_has_successful_trials,
    resumed_summary_row,
    run_jobs,
    run_one_job,
    sanitize_token,
    selected_candidates_from_args,
    summary_path,
    SUMMARY_FIELDS,
    STOP_EVENT,
    validate_session_fixed_files,
    write_manifest,
)


def _run_source_pretrain(rows: list[dict[str, Any]], gpus: list[str], *, search_algo: str, resume_from_logs: bool, dry_run: bool) -> int:
    """Train source checkpoints and export them."""
    install_signal_handlers()
    question = "a09"
    target_summary = summary_path(question)
    target_summary.parent.mkdir(parents=True, exist_ok=True)

    from queue import Empty, Queue
    import threading

    pending: Queue[dict[str, Any]] = Queue()
    for row in rows:
        if resume_from_logs:
            export_path = Path(str(row.get("checkpoint_export_path", "")))
            if export_path.exists():
                print(f"[a09-source] SKIP (export exists) {row.get('job_id')}", flush=True)
                continue
        pending.put(row)

    if dry_run or pending.empty():
        for row in rows:
            print(f"[a09-source][dry-run] {row.get('job_id')} dataset={row.get('dataset')}")
        return 0

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))

    def worker() -> None:
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo=search_algo)
                append_csv_row(target_summary, SUMMARY_FIELDS, summary)
                # Export checkpoint
                if str(summary.get("status", "")) == "ok" and summary.get("checkpoint_file"):
                    export_path = Path(str(row.get("checkpoint_export_path", "")))
                    if export_path and not export_path.exists():
                        export_path.parent.mkdir(parents=True, exist_ok=True)
                        src = Path(str(summary["checkpoint_file"]))
                        if src.exists():
                            shutil.copy2(str(src), str(export_path))
                            print(f"[a09-source] exported checkpoint -> {export_path}", flush=True)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


def _build_transfer_command(row: dict[str, Any], gpu_id: str, *, search_algo: str) -> list[str]:
    """Build hyperopt_tune command with transfer_mode and source_checkpoint overrides."""
    cmd = build_route_command(row, gpu_id, search_algo=search_algo)
    transfer_mode = str(row.get("transfer_mode", "none"))
    if transfer_mode != "none":
        cmd.append(f"++transfer_mode={hydra_literal(transfer_mode)}")
        source_ckpt = str(row.get("source_checkpoint", ""))
        if source_ckpt:
            cmd.append(f"++source_checkpoint={hydra_literal(source_ckpt)}")
    return cmd


def _run_target_transfer(rows: list[dict[str, Any]], gpus: list[str], *, search_algo: str, resume_from_logs: bool, dry_run: bool) -> int:
    """Train target datasets with transfer initialization."""
    install_signal_handlers()
    question = "a09"
    target_summary = LOG_ROOT / question / "transfer_summary.csv"
    target_summary.parent.mkdir(parents=True, exist_ok=True)

    from queue import Empty, Queue
    import threading

    pending: Queue[dict[str, Any]] = Queue()
    for row in rows:
        if resume_from_logs:
            log_p = log_path_for_row("a09", row)
            if has_run_status_end_normal(log_p):
                print(f"[a09-target] SKIP (log ok) {row.get('job_id')}", flush=True)
                continue
        # Verify source checkpoint exists for non-scratch settings
        transfer_mode = str(row.get("transfer_mode", "none"))
        if transfer_mode != "none":
            source_ckpt = Path(str(row.get("source_checkpoint", "")))
            if not source_ckpt.exists():
                print(f"[a09-target] SKIP (no source ckpt) {row.get('job_id')} needs {source_ckpt}", flush=True)
                continue
        pending.put(row)

    if dry_run or pending.empty():
        for row in rows:
            print(f"[a09-target][dry-run] {row.get('job_id')} pair={row.get('pair_id')} mode={row.get('transfer_mode')}")
        return 0

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))

    def worker() -> None:
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo=search_algo)
                append_csv_row(target_summary, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return 0


def main() -> int:
    parser = common_arg_parser("A09 boundary conditions + transfer portability", question="a09")
    parser.add_argument("--mode", default="all", choices=["all", "low_data", "transfer_source", "transfer_target"],
                        help="Which sub-experiment to run.")
    parser.add_argument("--pairs", default="beauty_to_kuairec,kuairec_to_beauty",
                        help="Comma-separated transfer pair IDs.")
    parser.add_argument("--hparam-presets", default="shared_a",
                        help="Comma-separated hparam preset IDs.")
    args = parser.parse_args()

    gpus = [gpu for gpu in str(args.gpus).split(",") if gpu.strip()]
    seeds = parse_csv_ints(args.seeds) or [1]
    mode = str(args.mode)

    # --- Low-data budget curve ---
    if mode in ("all", "low_data"):
        candidates = selected_candidates_from_args(args)
        low_data_rows = build_train_rows(
            question="a09",
            candidates=candidates,
            settings=a09_low_data_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
            smoke_test=bool(args.smoke_test),
            smoke_max_runs=args.smoke_max_runs,
        )
        manifest = write_manifest("a09_low_data", low_data_rows)
        print(f"[a09] low-data manifest -> {manifest}")
        rc = run_jobs(
            low_data_rows,
            question="a09",
            gpus=gpus,
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            print(f"[a09] low-data phase returned rc={rc}")

    # --- Transfer: source pretrain ---
    if mode in ("all", "transfer_source"):
        pairs = parse_csv_list(args.pairs)
        presets = parse_csv_list(args.hparam_presets)
        source_rows = build_a09_source_rows(
            pairs=pairs,
            preset_ids=presets,
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
        )
        print(f"[a09] {len(source_rows)} source pretrain rows")
        rc = _run_source_pretrain(
            source_rows,
            gpus,
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            print(f"[a09] source pretrain returned rc={rc}")

    # --- Transfer: target runs ---
    if mode in ("all", "transfer_target"):
        pairs = parse_csv_list(args.pairs)
        presets = parse_csv_list(args.hparam_presets)
        target_rows = build_a09_target_rows(
            pairs=pairs,
            preset_ids=presets,
            transfer_settings=a09_transfer_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
        )
        print(f"[a09] {len(target_rows)} target transfer rows")
        rc = _run_target_transfer(
            target_rows,
            gpus,
            search_algo=args.search_algo,
            resume_from_logs=bool(args.resume_from_logs),
            dry_run=bool(args.dry_run),
        )
        if rc != 0:
            print(f"[a09] target transfer returned rc={rc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
