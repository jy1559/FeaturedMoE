#!/usr/bin/env python3
"""Build per-model experiment inventory (CSV + Markdown) from timeline events."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_events_path() -> Path:
    # .../experiments/run/common/model_experiment_report.py -> .../experiments/run/artifacts/timeline/events.jsonl
    return Path(__file__).resolve().parents[1] / "artifacts" / "timeline" / "events.jsonl"


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def _resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    return Path(s).expanduser()


def _best_mrr_from_result(result_file: Path | None) -> float | None:
    if result_file is None or not result_file.exists():
        return None
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    top_best: float | None = None
    for key in ("best_mrr@20", "best_mrr20"):
        v = _to_float(data.get(key))
        if v is not None:
            top_best = v
            break

    bvr_best: float | None = None
    bvr = data.get("best_valid_result")
    if isinstance(bvr, dict):
        bvr_best = _to_float(bvr.get("mrr@20"))

    best_valid_trial: float | None = None
    trials = data.get("trials")
    if isinstance(trials, list):
        for t in trials:
            if not isinstance(t, dict):
                continue
            status = str(t.get("status", "")).strip().lower()
            if status not in {"", "ok", "success"}:
                continue
            m = _to_float(t.get("mrr@20"))
            if m is None:
                continue
            if best_valid_trial is None or m > best_valid_trial:
                best_valid_trial = m

    if best_valid_trial is not None:
        return best_valid_trial
    if bvr_best is not None:
        return bvr_best
    if top_best is not None:
        n_completed = int(_to_float(data.get("n_completed")) or 0)
        if n_completed > 0:
            return top_best
    return None


def _extract_how(cmd: str, axis: str, phase: str) -> str:
    parts: list[str] = [f"{axis}/{phase}"]
    tokens = cmd.split()
    keep_keys = (
        "fmoe_v2_layout_id=",
        "fmoe_stage_execution_mode=",
        "dataset=",
    )
    max_follow = {"--max-evals", "--tune-epochs", "--tune-patience", "--search-profile"}

    i = 0
    extras: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        matched = False
        for k in keep_keys:
            if tok.startswith(k):
                extras.append(tok)
                matched = True
                break
        if not matched and tok in max_follow:
            if i + 1 < len(tokens):
                extras.append(f"{tok}={tokens[i+1]}")
                i += 1
        i += 1

    if extras:
        parts.append(", ".join(extras[:6]))
    return " ; ".join(parts)


@dataclass
class ReportRow:
    run_id: str
    ts_start_utc: str
    ts_end_utc: str
    duration_sec: float | None
    track: str
    axis: str
    phase: str
    dataset: str
    model: str
    status: str
    oom_detected: bool
    best_mrr20: float | None
    how: str
    result_file: str
    log_file: str
    cmd: str

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ts_start_utc": self.ts_start_utc,
            "ts_end_utc": self.ts_end_utc,
            "duration_sec": "" if self.duration_sec is None else f"{self.duration_sec:.3f}",
            "duration_min": "" if self.duration_sec is None else f"{self.duration_sec/60.0:.3f}",
            "track": self.track,
            "axis": self.axis,
            "phase": self.phase,
            "dataset": self.dataset,
            "model": self.model,
            "status": self.status,
            "oom_detected": "yes" if self.oom_detected else "no",
            "best_mrr20": "" if self.best_mrr20 is None else f"{self.best_mrr20:.6f}",
            "how": self.how,
            "result_file": self.result_file,
            "log_file": self.log_file,
            "cmd": self.cmd,
        }


def _build_rows(events: list[dict[str, Any]], track: str, model_prefix: str) -> tuple[list[ReportRow], dict[str, int]]:
    starts = {e.get("run_id"): e for e in events if e.get("event_type") == "start"}
    rows: list[ReportRow] = []
    stats = {
        "matched_end": 0,
        "included": 0,
        "excluded_error": 0,
        "excluded_no_metric": 0,
    }

    m_track = str(track).strip().lower()
    m_model = str(model_prefix).strip().lower()

    for e in events:
        if e.get("event_type") != "end":
            continue

        ev_track = str(e.get("track", "")).strip().lower()
        ev_model = str(e.get("model", "")).strip()
        if ev_track != m_track:
            continue
        if not ev_model.lower().startswith(m_model):
            continue

        stats["matched_end"] += 1
        run_id = str(e.get("run_id", "")).strip()
        start = starts.get(run_id, {})

        status = str(e.get("status", "")).strip().lower()
        oom = bool(e.get("oom_detected"))
        result_file = _resolve_path(e.get("result_file"))
        best_mrr = _best_mrr_from_result(result_file)

        include = False
        if oom:
            include = True
        elif status != "success":
            stats["excluded_error"] += 1
        elif best_mrr is None:
            stats["excluded_no_metric"] += 1
        else:
            include = True

        if not include:
            continue

        ts_start_raw = start.get("ts_utc") or start.get("ts")
        ts_end_raw = e.get("ts_utc") or e.get("ts")
        ts_start = _parse_ts(ts_start_raw)
        ts_end = _parse_ts(ts_end_raw)
        duration_sec: float | None = None
        if ts_start and ts_end:
            duration_sec = max((ts_end - ts_start).total_seconds(), 0.0)

        axis = str(e.get("axis", start.get("axis", "")))
        phase = str(e.get("phase", start.get("phase", "")))
        cmd = str(e.get("cmd", start.get("cmd", "")))

        rows.append(
            ReportRow(
                run_id=run_id,
                ts_start_utc=str(ts_start_raw or ""),
                ts_end_utc=str(ts_end_raw or ""),
                duration_sec=duration_sec,
                track=str(e.get("track", start.get("track", ""))),
                axis=axis,
                phase=phase,
                dataset=str(e.get("dataset", start.get("dataset", ""))),
                model=ev_model,
                status=status or "unknown",
                oom_detected=oom,
                best_mrr20=best_mrr,
                how=_extract_how(cmd, axis, phase),
                result_file=str(result_file) if result_file else "",
                log_file=str(e.get("log_file", start.get("log_file", ""))),
                cmd=cmd,
            )
        )
        stats["included"] += 1

    rows.sort(key=lambda r: r.ts_end_utc, reverse=True)
    return rows, stats


def _write_csv(rows: list[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "ts_start_utc",
        "ts_end_utc",
        "duration_sec",
        "duration_min",
        "track",
        "axis",
        "phase",
        "dataset",
        "model",
        "status",
        "oom_detected",
        "best_mrr20",
        "how",
        "result_file",
        "log_file",
        "cmd",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.as_csv_row())


def _write_md(rows: list[ReportRow], stats: dict[str, int], track: str, model: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    best_by_dataset: dict[str, ReportRow] = {}
    for r in rows:
        if r.best_mrr20 is None:
            continue
        cur = best_by_dataset.get(r.dataset)
        if cur is None or (cur.best_mrr20 is not None and r.best_mrr20 > cur.best_mrr20):
            best_by_dataset[r.dataset] = r

    lines: list[str] = []
    lines.append(f"# {model} Experiment Report")
    lines.append("")
    lines.append(f"- generated_at_utc: {_utc_now()}")
    lines.append(f"- track: {track}")
    lines.append(
        "- include_rule: keep OOM runs, keep successful runs with valid MRR@20, "
        "exclude non-OOM failures and no-metric runs"
    )
    lines.append(f"- matched_end_events: {stats['matched_end']}")
    lines.append(f"- included_runs: {stats['included']}")
    lines.append(f"- excluded_non_oom_error_runs: {stats['excluded_error']}")
    lines.append(f"- excluded_no_metric_runs: {stats['excluded_no_metric']}")
    lines.append("")
    lines.append("## Best By Dataset (MRR@20)")
    lines.append("")
    lines.append("| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |")
    lines.append("|---|---:|---|---|---:|---|---|")
    if best_by_dataset:
        for dataset in sorted(best_by_dataset.keys()):
            r = best_by_dataset[dataset]
            dmin = "-" if r.duration_sec is None else f"{r.duration_sec/60.0:.2f}"
            rid = r.run_id[:12]
            lines.append(
                f"| {dataset} | {r.best_mrr20:.6f} | {r.axis} | {r.phase} | {dmin} | {r.ts_end_utc} | {rid} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Included Runs")
    lines.append("")
    lines.append("| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |")
    lines.append("|---|---|---|---|---|---|---:|---:|---|---|")
    if rows:
        for r in rows:
            mrr = "-" if r.best_mrr20 is None else f"{r.best_mrr20:.6f}"
            dmin = "-" if r.duration_sec is None else f"{r.duration_sec/60.0:.2f}"
            oom = "yes" if r.oom_detected else "no"
            lines.append(
                f"| {r.ts_end_utc} | {r.dataset} | {r.axis} | {r.phase} | {r.status} | {oom} | {mrr} | {dmin} | {r.how} | {r.run_id[:12]} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build per-model experiment report from timeline events.")
    p.add_argument("--track", required=True, help="Track name (e.g. fmoe_v2)")
    p.add_argument("--model", required=True, help="Model prefix to match (e.g. FeaturedMoE_v2)")
    p.add_argument("--model-dir", required=True, help="Model folder path for output files")
    p.add_argument("--events", default=str(_default_events_path()), help="events.jsonl path")
    p.add_argument("--output-csv", default="", help="Override CSV output path")
    p.add_argument("--output-md", default="", help="Override Markdown output path")
    args = p.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve() if args.output_csv else model_dir / "experiment_runs.csv"
    output_md = Path(args.output_md).expanduser().resolve() if args.output_md else model_dir / "experiment_summary.md"

    events = _read_jsonl(events_path)
    rows, stats = _build_rows(events, track=args.track, model_prefix=args.model)

    _write_csv(rows, output_csv)
    _write_md(rows, stats, track=args.track, model=args.model, path=output_md)

    print(f"[OK] wrote {output_csv}")
    print(f"[OK] wrote {output_md}")
    print(
        f"[INFO] matched_end={stats['matched_end']} included={stats['included']} "
        f"excluded_error={stats['excluded_error']} excluded_no_metric={stats['excluded_no_metric']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
