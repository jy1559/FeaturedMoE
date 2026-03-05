#!/usr/bin/env python3
"""Simple timeline tracker for run scripts.

Writes:
  - artifacts/timeline/events.jsonl
  - artifacts/timeline/ignored_events.jsonl
  - artifacts/timeline/dashboard.md
  - artifacts/timeline/state/<run_id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_LINE_RE = re.compile(r"Results?\s*->\s*(.+?)\s*$", re.MULTILINE)
BEST_MRR_RE = re.compile(r"Best\s+MRR@20\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
OOM_RE = re.compile(r"CUDA out of memory", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_timeline_dir() -> Path:
    # .../experiments/run/common/experiment_tracker.py -> .../experiments/run/artifacts/timeline
    return Path(__file__).resolve().parents[1] / "artifacts" / "timeline"


def _timeline_dir() -> Path:
    raw = os.environ.get("RUN_TIMELINE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_timeline_dir()


def _repo_root() -> Path:
    # .../experiments/run/common -> repo root
    return Path(__file__).resolve().parents[3]


def _state_dir(base: Path) -> Path:
    return base / "state"


def _events_path(base: Path) -> Path:
    return base / "events.jsonl"


def _ignored_events_path(base: Path) -> Path:
    return base / "ignored_events.jsonl"


def _dashboard_path(base: Path) -> Path:
    return base / "dashboard.md"


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _safe_read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw.strip())
    if not str(p):
        return None
    if p.is_absolute():
        return p

    cwd = Path.cwd()
    cand = (cwd / p).resolve()
    if cand.exists():
        return cand

    repo = _repo_root()
    cand = (repo / p).resolve()
    if cand.exists():
        return cand

    exp = repo / "experiments"
    cand = (exp / p).resolve()
    if cand.exists():
        return cand

    return (cwd / p).resolve()


def _result_from_log(log_path: Path | None) -> Path | None:
    text = _safe_read_text(log_path)
    if not text:
        return None
    found = RESULT_LINE_RE.findall(text)
    if not found:
        return None
    raw = found[-1].strip()
    return _resolve_path(raw)


def _metric_from_result(result_path: Path | None) -> float | None:
    if result_path is None or not result_path.exists():
        return None
    try:
        data = _load_json(result_path)
    except Exception:
        return None

    for key in ("best_mrr@20", "best_mrr20"):
        v = data.get(key)
        if isinstance(v, (int, float)):
            return float(v)

    bvr = data.get("best_valid_result", {})
    if isinstance(bvr, dict):
        v = bvr.get("mrr@20")
        if isinstance(v, (int, float)):
            return float(v)

    trials = data.get("trials", [])
    best: float | None = None
    if isinstance(trials, list):
        for t in trials:
            if not isinstance(t, dict):
                continue
            m = t.get("mrr@20")
            if isinstance(m, (int, float)):
                m = float(m)
                if best is None or m > best:
                    best = m
    return best


def _metric_from_log(log_path: Path | None) -> float | None:
    text = _safe_read_text(log_path)
    if not text:
        return None
    matches = BEST_MRR_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _has_oom(log_path: Path | None) -> bool:
    text = _safe_read_text(log_path)
    if not text:
        return False
    return bool(OOM_RE.search(text))


def _dashboard(base: Path) -> None:
    events = _read_events(_events_path(base))
    ignored = _read_events(_ignored_events_path(base))

    starts = [e for e in events if e.get("event_type") == "start"]
    ends = [e for e in events if e.get("event_type") == "end"]

    lines: list[str] = []
    lines.append("# Experiment Timeline Dashboard")
    lines.append("")
    lines.append(f"- generated_at_utc: {_utc_now()}")
    lines.append(f"- start_events: {len(starts)}")
    lines.append(f"- end_events: {len(ends)}")
    lines.append(f"- ignored_end_events: {len(ignored)}")
    lines.append("")
    lines.append("## Recent End Events")
    lines.append("")
    lines.append("| ts_utc | run_id | track | axis | phase | dataset | model | status | mrr@20 | oom | result_file |")
    lines.append("|---|---|---|---|---|---|---|---|---:|---|---|")

    for row in ends[-100:]:
        ts = str(row.get("ts_utc", row.get("ts", "")))
        run_id = str(row.get("run_id", ""))[:12]
        track = str(row.get("track", ""))
        axis = str(row.get("axis", ""))
        phase = str(row.get("phase", ""))
        dataset = str(row.get("dataset", ""))
        model = str(row.get("model", ""))
        status = str(row.get("status", ""))
        metric = row.get("metric")
        metric_s = "-" if metric is None else f"{float(metric):.6f}"
        oom_s = "yes" if row.get("oom_detected") else "no"
        result_file = str(row.get("result_file", "") or "-")
        lines.append(
            f"| {ts} | {run_id} | {track} | {axis} | {phase} | {dataset} | {model} | {status} | {metric_s} | {oom_s} | {result_file} |"
        )

    if not ends:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - |")

    _dashboard_path(base).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _start(args: argparse.Namespace) -> int:
    base = _timeline_dir()
    base.mkdir(parents=True, exist_ok=True)
    _state_dir(base).mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    ts = _utc_now()
    event = {
        "event_type": "start",
        "run_id": run_id,
        "ts": ts,
        "ts_utc": ts,
        "track": args.track,
        "axis": args.axis,
        "phase": args.phase,
        "dataset": args.dataset,
        "model": args.model,
        "cmd": args.cmd,
        "log_file": args.log_file,
    }
    _append_jsonl(_events_path(base), event)
    (_state_dir(base) / f"{run_id}.json").write_text(
        json.dumps(event, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _dashboard(base)
    print(run_id)
    return 0


def _end(args: argparse.Namespace) -> int:
    base = _timeline_dir()
    base.mkdir(parents=True, exist_ok=True)
    _state_dir(base).mkdir(parents=True, exist_ok=True)

    state_path = _state_dir(base) / f"{args.run_id}.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = _load_json(state_path)
        except Exception:
            state = {}

    track = args.track or state.get("track", "")
    axis = args.axis or state.get("axis", "")
    phase = args.phase or state.get("phase", "")
    dataset = args.dataset or state.get("dataset", "")
    model = args.model or state.get("model", "")
    cmd = args.cmd or state.get("cmd", "")
    log_file_raw = args.log_file or state.get("log_file", "")
    log_file = _resolve_path(log_file_raw)

    result_file = _resolve_path(args.result_file)
    if result_file is None:
        result_file = _result_from_log(log_file)

    result_exists = bool(result_file and result_file.exists())
    oom = _has_oom(log_file)
    include = result_exists or oom

    metric = _metric_from_result(result_file) if result_exists else None
    if metric is None:
        metric = _metric_from_log(log_file)

    ts = _utc_now()
    event = {
        "event_type": "end",
        "run_id": args.run_id,
        "ts": ts,
        "ts_utc": ts,
        "track": track,
        "axis": axis,
        "phase": phase,
        "dataset": dataset,
        "model": model,
        "cmd": cmd,
        "status": args.status,
        "exit_code": args.exit_code,
        "metric": metric,
        "oom_detected": oom,
        "result_file": str(result_file) if result_file else "",
        "log_file": str(log_file) if log_file else "",
    }

    if include:
        _append_jsonl(_events_path(base), event)
    else:
        ignored = dict(event)
        ignored["ignored_reason"] = "no_result_and_no_oom"
        _append_jsonl(_ignored_events_path(base), ignored)

    if state_path.exists():
        try:
            state_path.unlink()
        except Exception:
            pass

    _dashboard(base)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run timeline tracker")
    sub = parser.add_subparsers(dest="cmd_name", required=True)

    p_start = sub.add_parser("start", help="Record run start")
    p_start.add_argument("--run-id", default="")
    p_start.add_argument("--track", required=True)
    p_start.add_argument("--axis", required=True)
    p_start.add_argument("--phase", required=True)
    p_start.add_argument("--dataset", required=True)
    p_start.add_argument("--model", required=True)
    p_start.add_argument("--cmd", required=True)
    p_start.add_argument("--log-file", required=True)
    p_start.set_defaults(func=_start)

    p_end = sub.add_parser("end", help="Record run end")
    p_end.add_argument("--run-id", required=True)
    p_end.add_argument("--track", default="")
    p_end.add_argument("--axis", default="")
    p_end.add_argument("--phase", default="")
    p_end.add_argument("--dataset", default="")
    p_end.add_argument("--model", default="")
    p_end.add_argument("--cmd", default="")
    p_end.add_argument("--log-file", default="")
    p_end.add_argument("--result-file", default="")
    p_end.add_argument("--status", default="unknown")
    p_end.add_argument("--exit-code", type=int, default=0)
    p_end.set_defaults(func=_end)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
