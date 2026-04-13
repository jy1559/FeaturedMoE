#!/usr/bin/env python3
"""Common utilities for FMoE_N3 wide phase launchers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

from run_phase9_auxloss import (  # noqa: E402
    ARTIFACT_ROOT,
    EXP_DIR,
    _load_result_index,
    _metric_to_float,
)
from slack_progress import SlackProgressNotifier

_TRIAL_METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")
_RUN_METRIC_KV_PATTERN = re.compile(r"([A-Za-z0-9_@./-]+)=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")


def sanitize_token(text: str, *, upper: bool = True) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum():
            out.append(ch.upper() if upper else ch.lower())
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "X"


def _summary_common_fieldnames() -> list[str]:
    return [
        "global_best_valid_mrr20",
        "run_best_valid_mrr20",
        "run_phase",
        "exp_brief",
        "stage",
        "trigger",
        "dataset",
        "seed_id",
        "gpu_id",
        "status",
        "test_mrr20",
        "n_completed",
        "interrupted",
        "special_ok",
        "diag_ok",
        "result_path",
        "timestamp_utc",
    ]


def build_summary_fieldnames(extra_cols: list[str]) -> list[str]:
    base = _summary_common_fieldnames()
    for col in list(extra_cols or []):
        key = str(col).strip()
        if key and key not in base:
            base.append(key)
    return base


def _ensure_summary_csv(path: Path, fieldnames: list[str]) -> None:
    expected = ",".join(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].strip()
        except Exception:
            first = ""
        if first == expected:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}.legacy_{ts}{path.suffix}")
        try:
            path.rename(backup)
        except Exception:
            pass
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


def _append_summary_event(path: Path, fieldnames: list[str], row: Dict[str, Any]) -> None:
    _ensure_summary_csv(path, fieldnames)
    payload = {key: row.get(key, "") for key in fieldnames}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerow(payload)


def _load_summary_state(path: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "global_best_valid": None,
        "run_complete_written": set(),
    }
    if not path.exists():
        return state
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                trigger = str(row.get("trigger", "")).strip().lower()
                run_phase = str(row.get("run_phase", "")).strip()
                if trigger == "run_complete" and run_phase:
                    state["run_complete_written"].add(run_phase)
                g = _metric_to_float(row.get("global_best_valid_mrr20"))
                r = _metric_to_float(row.get("run_best_valid_mrr20"))
                for val in (g, r):
                    if val is None:
                        continue
                    if state["global_best_valid"] is None or val > float(state["global_best_valid"]):
                        state["global_best_valid"] = float(val)
    except Exception:
        return state
    return state


def _summary_format_metric(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{int(ndigits)}f}"


def _summary_row_base(
    *,
    row: Dict[str, Any],
    trigger: str,
    status: str,
    global_best_valid: Optional[float],
    run_best_valid: Optional[float],
    test_mrr20: Optional[float],
    n_completed: Optional[int],
    interrupted: Optional[bool],
    special_ok: Optional[bool],
    diag_ok: Optional[bool],
    result_path: str,
    extra_cols: list[str],
) -> Dict[str, Any]:
    payload = {
        "global_best_valid_mrr20": _summary_format_metric(global_best_valid),
        "run_best_valid_mrr20": _summary_format_metric(run_best_valid),
        "run_phase": row.get("run_phase", ""),
        "exp_brief": row.get("exp_brief", ""),
        "stage": row.get("stage", "wide"),
        "trigger": str(trigger),
        "dataset": row.get("dataset", ""),
        "seed_id": row.get("seed_id", ""),
        "gpu_id": row.get("assigned_gpu", ""),
        "status": str(status),
        "test_mrr20": _summary_format_metric(test_mrr20),
        "n_completed": "" if n_completed is None else int(n_completed),
        "interrupted": "" if interrupted is None else bool(interrupted),
        "special_ok": "" if special_ok is None else bool(special_ok),
        "diag_ok": "" if diag_ok is None else bool(diag_ok),
        "result_path": str(result_path or ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    for col in list(extra_cols or []):
        if col in payload:
            continue
        payload[col] = row.get(col, "")
    return payload


def _parse_trial_metrics_line(line: str) -> Optional[Dict[str, float]]:
    text = str(line or "")
    if "[TRIAL_METRICS]" not in text:
        return None
    metrics: Dict[str, float] = {}
    for key, raw_val in _TRIAL_METRIC_KV_PATTERN.findall(text):
        val = _metric_to_float(raw_val)
        if val is None:
            continue
        metrics[str(key)] = float(val)
    return metrics or None


def _parse_run_metrics_line(line: str) -> Optional[Dict[str, float]]:
    text = str(line or "")
    if "[RUN_METRICS]" not in text:
        return None
    metrics: Dict[str, float] = {}
    for key, raw_val in _RUN_METRIC_KV_PATTERN.findall(text):
        val = _metric_to_float(raw_val)
        if val is None:
            continue
        metrics[str(key)] = float(val)
    return metrics or None


def _scan_log_run_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    latest: Optional[Dict[str, float]] = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parsed = _parse_run_metrics_line(line)
                if parsed:
                    latest = parsed
    except Exception:
        return None
    return latest


def _scan_trial_metric_updates(log_path: Path, start_offset: int) -> tuple[int, list[Dict[str, float]]]:
    updates: list[Dict[str, float]] = []
    new_offset = int(start_offset)
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(max(int(start_offset), 0))
            while True:
                line = fh.readline()
                if not line:
                    break
                parsed = _parse_trial_metrics_line(line)
                if parsed:
                    updates.append(parsed)
            new_offset = fh.tell()
    except Exception:
        return int(start_offset), []
    return new_offset, updates


def _is_completed_log(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        text = str(line).strip()
        if not text:
            continue
        if text == "[RUN_STATUS] END status=normal":
            return True
        return text.startswith("[RUN_STATUS] END status=normal ")
    return False


def _log_path_from_row(*, log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    phase = sanitize_token(phase_id, upper=True)
    axis_id = sanitize_token(str(row.get("axis_id", "A")), upper=True)
    axis_desc = sanitize_token(str(row.get("axis_desc", "axis")), upper=False)
    setting_id = sanitize_token(str(row.get("setting_id", "00")), upper=True)
    setting_desc = sanitize_token(str(row.get("setting_desc", "setting")), upper=True)
    filename = f"{phase}_{axis_id}_{axis_desc}_{setting_id}_{setting_desc}.log"
    return log_dir / filename


def _write_log_preamble(
    *,
    log_file: Path,
    row: Dict[str, Any],
    gpu_id: str,
    args: argparse.Namespace,
    cmd: list[str],
    phase_name: str,
) -> None:
    # Support nested log layouts (e.g., dataset/H1/*.log) safely.
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{phase_name}_SETTING_HEADER]",
        (
            f"run_phase={row.get('run_phase','')} exp_brief={row.get('exp_brief','')} "
            f"phase_id={row.get('phase_id','')} axis_id={row.get('axis_id','')} "
            f"setting_id={row.get('setting_id','')} hparam_id={row.get('hparam_id','')} seed={row.get('seed_id','')}"
        ),
        f"desc={row.get('setting_desc','')}",
        f"dataset={row.get('dataset','')} gpu={gpu_id} order={row.get('assigned_order', 0)}",
        f"max_evals={getattr(args, 'max_evals', '')} tune_epochs={getattr(args, 'tune_epochs', '')} tune_patience={getattr(args, 'tune_patience', '')}",
        f"seed={row.get('runtime_seed','')}",
        "",
        "[COMMAND]",
        f"- {' '.join(cmd)}",
        "",
    ]
    log_file.write_text("\n".join(lines), encoding="utf-8")


def _verify_special_diag_from_result(result_json_path: str) -> tuple[bool, bool, str]:
    path = Path(str(result_json_path or ""))
    if not path.exists():
        return False, False, f"result_missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, False, f"result_parse_error:{exc}"

    special_candidates = [
        payload.get("special_log_file", ""),
        payload.get("special_result_file", ""),
    ]
    diag_candidates = [
        payload.get("diag_raw_trial_summary_file", ""),
        payload.get("diag_tier_a_final_file", ""),
        payload.get("diag_meta_file", ""),
    ]

    special_ok = any(str(p).strip() and Path(str(p)).exists() for p in special_candidates)
    diag_ok = any(str(p).strip() and Path(str(p)).exists() for p in diag_candidates)
    detail = (
        f"special={special_ok} diag={diag_ok} "
        f"special_paths={sum(1 for p in special_candidates if str(p).strip())} "
        f"diag_paths={sum(1 for p in diag_candidates if str(p).strip())}"
    )
    return special_ok, diag_ok, detail


def _extract_valid_mrr_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    candidates = [
        payload.get("best_mrr@20"),
        payload.get("best_mrr"),
    ]
    best_valid_result = payload.get("best_valid_result")
    if isinstance(best_valid_result, dict):
        candidates.append(best_valid_result.get("mrr@20"))
    for value in candidates:
        metric = _metric_to_float(value)
        if metric is not None:
            return metric
    return None


def _extract_test_mrr_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    candidates = [
        payload.get("test_mrr@20"),
        payload.get("test_mrr"),
    ]
    test_result = payload.get("test_result")
    if isinstance(test_result, dict):
        candidates.append(test_result.get("mrr@20"))
    for value in candidates:
        metric = _metric_to_float(value)
        if metric is not None:
            return metric
    return None


def _find_latest_result_row_for_run_phase(dataset: str, axis: str, run_phase: str) -> Optional[Dict[str, Any]]:
    result_root = ARTIFACT_ROOT / "results" / "fmoe_n3"
    if not result_root.exists():
        return None
    latest: Optional[Dict[str, Any]] = None
    latest_mtime = -1.0
    for path in result_root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_axis", "")) != str(axis):
            continue
        if str(payload.get("dataset", "")) != str(dataset):
            continue
        if str(payload.get("run_phase", "")).strip() != str(run_phase):
            continue
        try:
            mtime = float(path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        if mtime < latest_mtime:
            continue
        latest_mtime = mtime
        latest = {
            "run_phase": str(run_phase),
            "best_mrr": _extract_valid_mrr_from_payload(payload),
            "test_mrr": _extract_test_mrr_from_payload(payload),
            "n_completed": int(payload.get("n_completed", 0) or 0),
            "interrupted": bool(payload.get("interrupted", False)),
            "path": str(path),
            "mtime": mtime,
        }
    return latest


def _get_result_row_for_run_phase(dataset: str, axis: str, run_phase: str, retries: int = 8, sleep_sec: float = 1.0) -> Optional[Dict[str, Any]]:
    for _ in range(max(int(retries), 1)):
        index = _load_result_index(dataset, axis)
        row = index.get(str(run_phase))
        if isinstance(row, dict):
            return row
        time.sleep(max(float(sleep_sec), 0.0))
    return None


def _record_trial_new_best_if_any(
    *,
    summary_path: Path,
    fieldnames: list[str],
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    trial_metrics: Dict[str, float],
    extra_cols: list[str],
) -> None:
    run_best = _metric_to_float(trial_metrics.get("run_best_mrr20"))
    if run_best is None:
        run_best = _metric_to_float(trial_metrics.get("cur_best_mrr20"))
    if run_best is None:
        return

    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    if prev_global is not None and run_best <= prev_global:
        return

    new_global = run_best if prev_global is None else max(prev_global, run_best)
    payload = _summary_row_base(
        row=row,
        trigger="trial_new_best",
        status="running",
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=_metric_to_float(trial_metrics.get("run_test_mrr20")),
        n_completed=None,
        interrupted=None,
        special_ok=None,
        diag_ok=None,
        result_path="",
        extra_cols=extra_cols,
    )
    _append_summary_event(summary_path, fieldnames, payload)
    summary_state["global_best_valid"] = float(new_global)


def _record_run_complete_summary(
    *,
    dataset: str,
    axis: str,
    summary_path: Path,
    fieldnames: list[str],
    summary_state: Dict[str, Any],
    row: Dict[str, Any],
    extra_cols: list[str],
    verify_logging: bool,
) -> None:
    run_phase = str(row.get("run_phase", "") or "")
    if not run_phase:
        return
    run_complete_written = summary_state.get("run_complete_written")
    if isinstance(run_complete_written, set) and run_phase in run_complete_written:
        return

    result_row = _get_result_row_for_run_phase(dataset, axis, run_phase, retries=10, sleep_sec=1.0)
    run_best = None
    test_mrr = None
    n_completed = None
    interrupted = None
    result_path = ""
    special_ok = False
    diag_ok = False
    status = "run_complete"

    if isinstance(result_row, dict):
        run_best = _metric_to_float(result_row.get("best_mrr"))
        test_mrr = _metric_to_float(result_row.get("test_mrr"))
        n_completed = int(result_row.get("n_completed", 0) or 0)
        interrupted = bool(result_row.get("interrupted", False))
        result_path = str(result_row.get("path", "") or "")
        if result_path:
            special_ok, diag_ok, detail = _verify_special_diag_from_result(result_path)
            print(f"[wide][logging-check] run={run_phase} {detail} result={result_path}")
            if verify_logging and (not special_ok or not diag_ok):
                latest_row = _find_latest_result_row_for_run_phase(dataset, axis, run_phase)
                latest_path = str((latest_row or {}).get("path", "") or "")
                if latest_path and latest_path != result_path:
                    latest_special_ok, latest_diag_ok, latest_detail = _verify_special_diag_from_result(latest_path)
                    print(f"[wide][logging-check] run={run_phase} fallback_latest {latest_detail} result={latest_path}")
                    if latest_special_ok and latest_diag_ok:
                        result_row = latest_row
                        run_best = _metric_to_float(result_row.get("best_mrr"))
                        test_mrr = _metric_to_float(result_row.get("test_mrr"))
                        n_completed = int(result_row.get("n_completed", 0) or 0)
                        interrupted = bool(result_row.get("interrupted", False))
                        result_path = latest_path
                        special_ok = latest_special_ok
                        diag_ok = latest_diag_ok
            if verify_logging and (not special_ok or not diag_ok):
                raise RuntimeError(
                    f"Logging verification failed for run_phase={run_phase} "
                    f"(special_ok={special_ok}, diag_ok={diag_ok})"
                )
        else:
            status = "run_complete_no_result_path"
    else:
        status = "run_complete_no_result"

    log_metrics = _scan_log_run_metrics(Path(str(row.get("log_path", "") or "")))
    if log_metrics:
        if run_best is None:
            run_best = _metric_to_float(log_metrics.get("best_valid_mrr20"))
        if test_mrr is None:
            test_mrr = _metric_to_float(log_metrics.get("test_mrr20"))

    prev_global = _metric_to_float(summary_state.get("global_best_valid"))
    new_global = prev_global
    if run_best is not None:
        new_global = run_best if prev_global is None else max(prev_global, run_best)

    payload = _summary_row_base(
        row=row,
        trigger="run_complete",
        status=status,
        global_best_valid=new_global,
        run_best_valid=run_best,
        test_mrr20=test_mrr,
        n_completed=n_completed,
        interrupted=interrupted,
        special_ok=special_ok,
        diag_ok=diag_ok,
        result_path=result_path,
        extra_cols=extra_cols,
    )
    _append_summary_event(summary_path, fieldnames, payload)

    if new_global is not None:
        summary_state["global_best_valid"] = float(new_global)
    if not isinstance(run_complete_written, set):
        summary_state["run_complete_written"] = set()
    summary_state["run_complete_written"].add(run_phase)


def launch_wide_rows(
    *,
    rows: list[Dict[str, Any]],
    gpus: list[str],
    args: argparse.Namespace,
    axis: str,
    phase_id: str,
    phase_name: str,
    log_dir: Path,
    summary_path: Path,
    fieldnames: list[str],
    extra_cols: list[str],
    build_command: Callable[[Dict[str, Any], str, argparse.Namespace], list[str]],
    build_log_path: Optional[Callable[[Path, Dict[str, Any], str], Path]] = None,
    verify_logging: bool = True,
    summary_path_for_row: Optional[Callable[[Dict[str, Any]], Path]] = None,
) -> int:
    if not rows:
        print(f"[{phase_id}] no rows to run.")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)

    def _row_summary_path(row: Dict[str, Any]) -> Path:
        if summary_path_for_row is not None:
            return Path(summary_path_for_row(row))
        return summary_path

    summary_states: Dict[str, Dict[str, Any]] = {}

    def _get_summary_state_for(path: Path) -> Dict[str, Any]:
        key = str(path)
        if key not in summary_states:
            path.parent.mkdir(parents=True, exist_ok=True)
            _ensure_summary_csv(path, fieldnames)
            summary_states[key] = _load_summary_state(path)
        return summary_states[key]

    for idx, row in enumerate(rows):
        row["assigned_order"] = idx + 1

    runnable: list[Dict[str, Any]] = []
    skipped_rows: list[Dict[str, Any]] = []
    if build_log_path is None:
        build_log_path = _log_path_from_row
    for row in rows:
        lp = build_log_path(log_dir=log_dir, row=row, phase_id=phase_id)
        row["log_path"] = str(lp)
        sp = _row_summary_path(row)
        row["summary_path"] = str(sp)
        _get_summary_state_for(sp)
        if bool(getattr(args, "resume_from_logs", True)) and _is_completed_log(lp):
            skipped_rows.append(row)
            continue
        runnable.append(row)

    if skipped_rows:
        print(f"[{phase_id}] resume_from_logs=on: skipped {len(skipped_rows)} completed runs by strict log-end marker.")

    if not runnable:
        print(f"[{phase_id}] all runs are already completed by strict log markers.")
        return 0

    if bool(getattr(args, "dry_run", False)):
        rr_idx = 0
        for row in runnable:
            lp = Path(str(row["log_path"]))
            gpu_id = str(gpus[rr_idx % len(gpus)])
            rr_idx += 1
            cmd = build_command(row, gpu_id, args)
            print(
                f"[dry-run] gpu={gpu_id} run_phase={row['run_phase']} "
                f"setting={row.get('setting_id','')} -> {lp}"
            )
            print("          " + " ".join(cmd))
        return 0

    notifier = SlackProgressNotifier(phase_label=phase_name.lower().replace("_", " "), rows=rows)
    notifier.notify_plan(precompleted_rows=skipped_rows)

    pending_queue: deque[Dict[str, Any]] = deque(runnable)

    active: Dict[str, Dict[str, Any]] = {}
    while True:
        for gpu_id in gpus:
            if gpu_id in active:
                continue
            if not pending_queue:
                continue
            row = pending_queue.popleft()
            row["assigned_gpu"] = str(gpu_id)
            lp = Path(str(row["log_path"]))
            cmd = build_command(row, gpu_id, args)
            _write_log_preamble(
                log_file=lp,
                row=row,
                gpu_id=gpu_id,
                args=args,
                cmd=cmd,
                phase_name=phase_name,
            )
            env = dict(os.environ)
            env["HYPEROPT_RESULTS_DIR"] = str(ARTIFACT_ROOT / "results")
            with lp.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
            start_offset = 0
            try:
                start_offset = int(lp.stat().st_size)
            except Exception:
                start_offset = 0
            active[gpu_id] = {
                "proc": proc,
                "row": row,
                "log_path": lp,
                "offset": start_offset,
            }
            print(
                f"[launch] phase={phase_id} gpu={gpu_id} run_phase={row.get('run_phase','')} "
                f"setting={row.get('setting_id','')}"
            )

        done_gpu: list[str] = []
        for gpu_id, slot in active.items():
            proc = slot["proc"]
            row = slot["row"]
            lp = slot["log_path"]
            prev_offset = int(slot.get("offset", 0))
            new_offset, updates = _scan_trial_metric_updates(lp, prev_offset)
            slot["offset"] = int(new_offset)
            for trial_metrics in updates:
                _record_trial_new_best_if_any(
                    summary_path=Path(str(row.get("summary_path", summary_path))),
                    fieldnames=fieldnames,
                    summary_state=_get_summary_state_for(Path(str(row.get("summary_path", summary_path)))),
                    row=row,
                    trial_metrics=trial_metrics,
                    extra_cols=extra_cols,
                )
            rc = proc.poll()
            if rc is None:
                continue
            done_gpu.append(gpu_id)
            print(f"[done] phase={phase_id} gpu={gpu_id} run_phase={row.get('run_phase','')} rc={rc} log={lp}")
            if int(rc) == 0:
                _record_run_complete_summary(
                    dataset=str(row.get("dataset", getattr(args, "dataset", ""))),
                    axis=axis,
                    summary_path=Path(str(row.get("summary_path", summary_path))),
                    fieldnames=fieldnames,
                    summary_state=_get_summary_state_for(Path(str(row.get("summary_path", summary_path)))),
                    row=row,
                    extra_cols=extra_cols,
                    verify_logging=bool(verify_logging),
                )
            notifier.mark_complete(row)

        for gpu_id in done_gpu:
            active.pop(gpu_id, None)

        pending = bool(pending_queue)
        if not pending and not active:
            break
        time.sleep(3)

    updated = sorted(summary_states.keys())
    if len(updated) == 1:
        print(f"[{phase_id}] summary updated: {updated[0]}")
    elif updated:
        print(f"[{phase_id}] summary updated: {len(updated)} files")
    return 0
