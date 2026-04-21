#!/usr/bin/env python3
"""Interleaved Q2 and/or Q3(b) ablation runner.

Runs Q2 routing-control variants and/or the Q3(b) routing-organization variants
for KuaiRec/Foursquare in rank bundles:

  rank 1 -> Q2 rows first, then Q3(b) rows
  rank 2 -> Q2 rows first, then Q3(b) rows
  ...

The queue is global across selected questions so GPUs are shared by one scheduler.
Completed runs are resume-skipped using the existing q2/q3 summary logic.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from queue import Empty, Queue


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    STOP_EVENT,
    SUMMARY_FIELDS,
    _finalize_summary_file,
    _recover_previous_summary_rows,
    append_csv_row,
    build_train_rows,
    common_arg_parser,
    install_signal_handlers,
    now_utc,
    parse_csv_ints,
    q2_settings,
    q3_settings,
    read_summary_rows,
    resumed_summary_row,
    run_one_job,
    selected_candidates_from_args,
    summary_path,
    write_manifest,
)


def _routing_org_settings() -> list[dict]:
    return [setting for setting in q3_settings() if str(setting.get("panel_family", "")).strip() == "routing_org"]


def _rows_for_question(
    *,
    question: str,
    candidates: list,
    settings: list[dict],
    seeds: list[int],
    max_evals: int,
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
    lr_mode: str,
) -> list[dict]:
    rows = build_train_rows(
        question=question,
        candidates=candidates,
        settings=settings,
        seeds=seeds,
        max_evals=max_evals,
        max_run_hours=max_run_hours,
        tune_epochs=tune_epochs,
        tune_patience=tune_patience,
        lr_mode=lr_mode,
        smoke_test=False,
        smoke_max_runs=0,
    )
    rows.sort(
        key=lambda row: (
            int(row.get("base_rank", 0) or 0),
            str(row.get("dataset", "")),
            int(row.get("variant_order", 0) or 0),
            int(row.get("seed_id", 0) or 0),
        )
    )
    return rows


def _interleave_rank_bundles(q2_rows: list[dict], q3_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    known_ranks = sorted(
        {int(row.get("base_rank", 0) or 0) for row in q2_rows}
        | {int(row.get("base_rank", 0) or 0) for row in q3_rows}
    )
    for rank in known_ranks:
        out.extend(row for row in q2_rows if int(row.get("base_rank", 0) or 0) == rank)
        out.extend(row for row in q3_rows if int(row.get("base_rank", 0) or 0) == rank)
    return out


def _write_temp_summary(path: Path, row: dict) -> None:
    append_csv_row(path, SUMMARY_FIELDS, {key: row.get(key, "") for key in SUMMARY_FIELDS})


def run_interleaved_jobs(
    rows: list[dict],
    *,
    gpus: list[str],
    search_algo: str,
    resume_from_logs: bool,
    dry_run: bool,
) -> int:
    install_signal_handlers()
    questions = ("q2", "q3")
    target_summary = {question: summary_path(question) for question in questions}
    temp_summary = {
        question: target_summary[question].with_name(
            f"{target_summary[question].stem}.tmp.interleave.{os.getpid()}{target_summary[question].suffix}"
        )
        for question in questions
    }
    previous_rows: dict[str, list[dict[str, str]]] = {}
    current_rows: dict[str, list[dict]] = {question: [] for question in questions}

    for question in questions:
        target_summary[question].parent.mkdir(parents=True, exist_ok=True)
        if temp_summary[question].exists():
            temp_summary[question].unlink()
        recovered = read_summary_rows(question)
        if not recovered:
            recovered = _recover_previous_summary_rows(question, [row for row in rows if row.get("question") == question])
        previous_rows[question] = recovered

    if dry_run:
        for row in rows:
            planned = {
                **{key: row.get(key, "") for key in SUMMARY_FIELDS},
                "gpu_id": "dry-run",
                "status": "planned",
                "valid_score": "",
                "test_score": "",
                "best_valid_mrr20": "",
                "test_mrr20": "",
                "checkpoint_file": "",
                "result_path": "",
                "elapsed_sec": 0.0,
                "error": "",
                "timestamp_utc": now_utc(),
            }
            _write_temp_summary(temp_summary[str(row["question"])], planned)
        for question in questions:
            _finalize_summary_file(
                target_summary=target_summary[question],
                temp_summary=temp_summary[question],
                previous_summary_rows=previous_rows[question],
                current_rows=[row for row in rows if row.get("question") == question],
            )
        return 0

    pending: Queue[dict] = Queue()
    skipped = 0
    for row in rows:
        question = str(row.get("question", ""))
        if resume_from_logs:
            resumed = resumed_summary_row(row, previous_rows=previous_rows[question])
            if resumed is not None:
                _write_temp_summary(temp_summary[question], resumed)
                skipped += 1
                print(
                    f"[resume] SKIP ({skipped}) question={question} job={row.get('job_id')} "
                    f"rank={row.get('base_rank')} seed={row.get('seed_id')}",
                    flush=True,
                )
                continue
        pending.put(row)

    print(
        f"[interleave] total={len(rows)} skipped(resume)={skipped} pending={pending.qsize()} "
        f"gpus={','.join(gpus)}",
        flush=True,
    )

    if pending.empty():
        for question in questions:
            _finalize_summary_file(
                target_summary=target_summary[question],
                temp_summary=temp_summary[question],
                previous_summary_rows=previous_rows[question],
                current_rows=[row for row in rows if row.get("question") == question],
            )
        return 0

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))

    append_lock = threading.Lock()

    def worker() -> None:
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo=search_algo)
                with append_lock:
                    _write_temp_summary(temp_summary[str(row["question"])], summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for question in questions:
        question_rows = [row for row in rows if row.get("question") == question]
        _finalize_summary_file(
            target_summary=target_summary[question],
            temp_summary=temp_summary[question],
            previous_summary_rows=previous_rows[question],
            current_rows=question_rows,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = common_arg_parser(
        "Interleaved Q2 routing-control and/or Q3(b) routing-organization runner",
        question="q2",
    )
    parser.set_defaults(
        datasets="KuaiRecLargeStrictPosV2_0.2,foursquare",
        top_k_configs=4,
        seeds="1,2,3,4,5",
        max_evals=3,
        max_run_hours=1.0,
        tune_epochs=100,
        tune_patience=10,
    )
    parser.add_argument(
        "--rank-order",
        default="1,2,3,4",
        help="Optional explicit rank order. Only matching base ranks are scheduled.",
    )
    parser.add_argument(
        "--suite",
        default="q3",
        choices=["q2", "q3", "both"],
        help="Which experiment family to schedule. Default: q3 (routing_org only).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    candidates = selected_candidates_from_args(args)
    wanted_ranks = set(parse_csv_ints(args.rank_order))
    if wanted_ranks:
        candidates = [candidate for candidate in candidates if int(candidate.rank) in wanted_ranks]
    if not candidates:
        raise RuntimeError("No base candidates remained after rank filtering.")

    seeds = parse_csv_ints(args.seeds) or [1]
    q2_rows: list[dict] = []
    q3_rows: list[dict] = []
    if args.suite in {"q2", "both"}:
        q2_rows = _rows_for_question(
            question="q2",
            candidates=candidates,
            settings=q2_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
        )
    if args.suite in {"q3", "both"}:
        q3_rows = _rows_for_question(
            question="q3",
            candidates=candidates,
            settings=_routing_org_settings(),
            seeds=seeds,
            max_evals=args.max_evals,
            max_run_hours=args.max_run_hours,
            tune_epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            lr_mode=args.lr_mode,
        )
    ordered_rows = _interleave_rank_bundles(q2_rows, q3_rows)

    if q2_rows:
        write_manifest("q2", q2_rows)
    if q3_rows:
        write_manifest("q3", q3_rows)
    print(
        f"[prepared] suite={args.suite} q2_rows={len(q2_rows)} "
        f"q3_routing_org_rows={len(q3_rows)} total={len(ordered_rows)}",
        flush=True,
    )
    return run_interleaved_jobs(
        ordered_rows,
        gpus=[gpu for gpu in str(args.gpus).split(",") if gpu.strip()],
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
