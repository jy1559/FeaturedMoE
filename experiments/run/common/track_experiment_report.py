#!/usr/bin/env python3
"""Build compact per-track experiment overview from timeline events."""

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
    return Path(__file__).resolve().parents[1] / "artifacts" / "timeline" / "events.jsonl"


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
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

    best_direct = _to_float(data.get("best_mrr@20"))
    if best_direct is None:
        best_direct = _to_float(data.get("best_mrr20"))

    bvr = data.get("best_valid_result")
    if isinstance(bvr, dict):
        v = _to_float(bvr.get("mrr@20"))
        if v is not None:
            return v

    best_trial: float | None = None
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
            if best_trial is None or m > best_trial:
                best_trial = m
    if best_trial is not None:
        return best_trial

    n_completed = int(_to_float(data.get("n_completed")) or 0)
    if best_direct is not None and n_completed > 0:
        return best_direct
    return None


def _load_params_from_result(result_file: Path | None) -> dict[str, Any]:
    if result_file is None or not result_file.exists():
        return {}
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, Any] = {}
    for block in (data.get("fixed_search"), data.get("best_params"), data.get("context_fixed")):
        if isinstance(block, dict):
            out.update(block)
    return out


def _short(v: Any, max_len: int = 120) -> str:
    s = str(v)
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _focus_list(raw: str) -> list[str]:
    vals: list[str] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if t:
            vals.append(t)
    return vals


@dataclass
class RunRow:
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
    result_file: str
    log_file: str
    exp_name: str
    exp_desc: str
    exp_focus: str
    params: dict[str, Any]


@dataclass
class ExpSummary:
    dataset: str
    exp_name: str
    exp_desc: str
    axis: str
    n_runs: int
    n_oom: int
    start_utc: str
    end_utc: str
    focus_vars: list[str]
    best_mrr20: float | None
    best_phase: str
    best_model: str
    best_log_file: str
    best_result_file: str
    best_settings: dict[str, Any]
    top_runs: list[dict[str, Any]]

    def csv(self) -> dict[str, Any]:
        top = self.top_runs[:3]
        def _v(i: int, k: str) -> str:
            if i >= len(top):
                return ""
            v = top[i].get(k, "")
            if k == "mrr" and isinstance(v, (int, float)):
                return f"{float(v):.6f}"
            if k == "settings":
                return json.dumps(v, ensure_ascii=False, sort_keys=True)
            return str(v)
        return {
            "dataset": self.dataset,
            "exp_name": self.exp_name,
            "exp_desc": self.exp_desc,
            "axis": self.axis,
            "n_runs": self.n_runs,
            "n_oom": self.n_oom,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "focus_vars": ",".join(self.focus_vars),
            "best_mrr20": "" if self.best_mrr20 is None else f"{self.best_mrr20:.6f}",
            "best_phase": self.best_phase,
            "best_model": self.best_model,
            "best_log_file": self.best_log_file,
            "best_result_file": self.best_result_file,
            "best_settings": json.dumps(self.best_settings, ensure_ascii=False, sort_keys=True),
            "top1_mrr20": _v(0, "mrr"),
            "top1_phase": _v(0, "phase"),
            "top1_log_file": _v(0, "log_file"),
            "top1_settings": _v(0, "settings"),
            "top2_mrr20": _v(1, "mrr"),
            "top2_phase": _v(1, "phase"),
            "top2_log_file": _v(1, "log_file"),
            "top2_settings": _v(1, "settings"),
            "top3_mrr20": _v(2, "mrr"),
            "top3_phase": _v(2, "phase"),
            "top3_log_file": _v(2, "log_file"),
            "top3_settings": _v(2, "settings"),
            "top3_json": json.dumps(self.top_runs[:3], ensure_ascii=False, sort_keys=True),
        }


def _build_rows(events: list[dict[str, Any]], track: str) -> tuple[list[RunRow], dict[str, int]]:
    starts = {e.get("run_id"): e for e in events if e.get("event_type") == "start"}
    out: list[RunRow] = []
    stats = {
        "matched_end": 0,
        "included": 0,
        "excluded_non_oom_error": 0,
        "excluded_no_metric": 0,
    }
    tr = track.strip().lower()

    for e in events:
        if e.get("event_type") != "end":
            continue
        if str(e.get("track", "")).strip().lower() != tr:
            continue
        stats["matched_end"] += 1

        run_id = str(e.get("run_id", "")).strip()
        start = starts.get(run_id, {})
        status = str(e.get("status", "")).strip().lower() or "unknown"
        oom = bool(e.get("oom_detected"))
        result_file = _resolve_path(e.get("result_file"))
        best_mrr = _best_mrr_from_result(result_file)

        include = False
        if oom:
            include = True
        elif status != "success":
            stats["excluded_non_oom_error"] += 1
        elif best_mrr is None:
            stats["excluded_no_metric"] += 1
        else:
            include = True
        if not include:
            continue

        ts_start_raw = str(start.get("ts_utc") or start.get("ts") or "")
        ts_end_raw = str(e.get("ts_utc") or e.get("ts") or "")
        ts_start = _parse_ts(ts_start_raw)
        ts_end = _parse_ts(ts_end_raw)
        duration_sec: float | None = None
        if ts_start and ts_end:
            duration_sec = max((ts_end - ts_start).total_seconds(), 0.0)

        exp_name = str(e.get("exp_name") or start.get("exp_name") or "").strip()
        exp_desc = str(e.get("exp_desc") or start.get("exp_desc") or "").strip()
        exp_focus = str(e.get("exp_focus") or start.get("exp_focus") or "").strip()
        axis = str(e.get("axis") or start.get("axis") or "")
        phase = str(e.get("phase") or start.get("phase") or "")
        if not exp_name:
            phase_root = phase.split("_")[0] if phase else "run"
            exp_name = f"{axis}_{phase_root}" if axis else phase_root

        out.append(
            RunRow(
                run_id=run_id,
                ts_start_utc=ts_start_raw,
                ts_end_utc=ts_end_raw,
                duration_sec=duration_sec,
                track=str(e.get("track", start.get("track", ""))),
                axis=axis,
                phase=phase,
                dataset=str(e.get("dataset", start.get("dataset", ""))),
                model=str(e.get("model", start.get("model", ""))),
                status=status,
                oom_detected=oom,
                best_mrr20=best_mrr,
                result_file=str(result_file) if result_file else "",
                log_file=str(e.get("log_file", start.get("log_file", ""))),
                exp_name=exp_name,
                exp_desc=exp_desc,
                exp_focus=exp_focus,
                params=_load_params_from_result(result_file),
            )
        )
        stats["included"] += 1

    out.sort(key=lambda r: r.ts_end_utc, reverse=True)
    return out, stats


def _aggregate(rows: list[RunRow]) -> list[ExpSummary]:
    grouped: dict[tuple[str, str], list[RunRow]] = {}
    for r in rows:
        grouped.setdefault((r.dataset, r.exp_name), []).append(r)

    out: list[ExpSummary] = []
    for (dataset, exp_name), arr in grouped.items():
        arr_sorted = sorted(arr, key=lambda x: x.ts_end_utc)
        axis = arr_sorted[-1].axis if arr_sorted else ""
        desc = ""
        for x in arr_sorted:
            if x.exp_desc:
                desc = x.exp_desc
                break

        mrr_rows = [x for x in arr_sorted if x.best_mrr20 is not None]

        focus = []
        for x in arr_sorted:
            if x.exp_focus:
                focus = _focus_list(x.exp_focus)
                if focus:
                    break
        if not focus:
            keys = set()
            for x in arr_sorted:
                keys.update(x.params.keys())
            var_keys: list[str] = []
            for k in sorted(keys):
                vals = {json.dumps(x.params.get(k, None), sort_keys=True, ensure_ascii=False) for x in arr_sorted}
                if len(vals) > 1:
                    var_keys.append(k)
            focus = var_keys[:10]

        def _pick_settings(row: RunRow) -> dict[str, Any]:
            picked: dict[str, Any] = {}
            if focus:
                for k in focus:
                    if k in row.params:
                        picked[k] = row.params[k]
            if picked:
                return picked
            keep = [
                "learning_rate",
                "weight_decay",
                "embedding_size",
                "d_feat_emb",
                "d_expert_hidden",
                "d_router_hidden",
                "train_batch_size",
                "eval_batch_size",
                "fmoe_stage_execution_mode",
                "fmoe_v2_layout_id",
            ]
            for k in keep:
                if k in row.params:
                    picked[k] = row.params[k]
            return picked

        ranked = sorted(mrr_rows, key=lambda x: ((x.best_mrr20 or -1.0), x.ts_end_utc), reverse=True)
        best_row = ranked[0] if ranked else None

        top_runs: list[dict[str, Any]] = []
        for idx, row in enumerate(ranked[:3], start=1):
            top_runs.append(
                {
                    "rank": idx,
                    "mrr": row.best_mrr20,
                    "phase": row.phase,
                    "model": row.model,
                    "log_file": row.log_file,
                    "result_file": row.result_file,
                    "settings": _pick_settings(row),
                }
            )
        best_settings = top_runs[0]["settings"] if top_runs else {}

        out.append(
            ExpSummary(
                dataset=dataset,
                exp_name=exp_name,
                exp_desc=desc,
                axis=axis,
                n_runs=len(arr_sorted),
                n_oom=sum(1 for x in arr_sorted if x.oom_detected),
                start_utc=arr_sorted[0].ts_start_utc if arr_sorted else "",
                end_utc=arr_sorted[-1].ts_end_utc if arr_sorted else "",
                focus_vars=focus,
                best_mrr20=best_row.best_mrr20 if best_row else None,
                best_phase=best_row.phase if best_row else "",
                best_model=best_row.model if best_row else "",
                best_log_file=best_row.log_file if best_row else "",
                best_result_file=best_row.result_file if best_row else "",
                best_settings=best_settings,
                top_runs=top_runs,
            )
        )

    out.sort(key=lambda x: (x.dataset, -(x.best_mrr20 or -1.0), x.exp_name))
    return out


def _write_csv(items: list[ExpSummary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "exp_name",
        "exp_desc",
        "axis",
        "n_runs",
        "n_oom",
        "start_utc",
        "end_utc",
        "focus_vars",
        "best_mrr20",
        "best_phase",
        "best_model",
        "best_log_file",
        "best_result_file",
        "best_settings",
        "top1_mrr20",
        "top1_phase",
        "top1_log_file",
        "top1_settings",
        "top2_mrr20",
        "top2_phase",
        "top2_log_file",
        "top2_settings",
        "top3_mrr20",
        "top3_phase",
        "top3_log_file",
        "top3_settings",
        "top3_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            w.writerow(it.csv())


def _settings_line(d: dict[str, Any]) -> str:
    if not d:
        return "-"
    parts = [f"{k}={_short(v, 48)}" for k, v in d.items()]
    return ", ".join(parts)


def _write_md(items: list[ExpSummary], stats: dict[str, int], track: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# {track} Experiment Overview")
    lines.append("")
    lines.append(f"- generated_at_utc: {_utc_now()}")
    lines.append("- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs")
    lines.append(f"- matched_end_events: {stats['matched_end']}")
    lines.append(f"- included_runs: {stats['included']}")
    lines.append(f"- excluded_non_oom_error_runs: {stats['excluded_non_oom_error']}")
    lines.append(f"- excluded_no_metric_runs: {stats['excluded_no_metric']}")
    lines.append(f"- summarized_experiments: {len(items)}")
    lines.append("")

    lines.append("## Experiment Summary Table")
    lines.append("")
    lines.append("| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |")
    lines.append("|---|---|---|---:|---:|---:|---|---|---|---|")
    if items:
        for it in items:
            mrr = "-" if it.best_mrr20 is None else f"{it.best_mrr20:.6f}"
            top3 = "/".join(f"{float(x.get('mrr')):.4f}" for x in it.top_runs[:3] if x.get("mrr") is not None)
            if not top3:
                top3 = "-"
            focus = ", ".join(it.focus_vars) if it.focus_vars else "-"
            lines.append(
                f"| {it.dataset} | {it.exp_name} | {it.axis} | {it.n_runs} | {it.n_oom} | {mrr} | {top3} | {it.best_phase or '-'} | {focus} | {it.best_log_file or '-'} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")

    lines.append("")
    lines.append("## Experiment Notes")
    lines.append("")
    if items:
        for it in items:
            lines.append(f"### {it.dataset} / {it.exp_name}")
            lines.append("")
            desc = it.exp_desc or "(설명 미기록)"
            lines.append(f"- 실험 설명: {desc}")
            lines.append(f"- 실행 규모: runs={it.n_runs}, oom={it.n_oom}, 기간={it.start_utc} ~ {it.end_utc}")
            lines.append(f"- 비교 변수: {', '.join(it.focus_vars) if it.focus_vars else '-'}")
            if it.best_mrr20 is not None:
                lines.append(f"- 최고 성능: MRR@20={it.best_mrr20:.6f} ({it.best_phase}, {it.best_model})")
            else:
                lines.append("- 최고 성능: -")
            lines.append(f"- 최고 설정: {_settings_line(it.best_settings)}")
            lines.append(f"- 최고 로그: {it.best_log_file or '-'}")
            lines.append(f"- 최고 결과 JSON: {it.best_result_file or '-'}")
            lines.append("- Top-3 결과:")
            lines.append("| rank | mrr@20 | phase | settings | log_file |")
            lines.append("|---:|---:|---|---|---|")
            if it.top_runs:
                for r in it.top_runs:
                    mrr_s = "-" if r.get("mrr") is None else f"{float(r['mrr']):.6f}"
                    lines.append(
                        f"| {r.get('rank','-')} | {mrr_s} | {r.get('phase','-')} | {_settings_line(r.get('settings') or {})} | {r.get('log_file','-')} |"
                    )
            else:
                lines.append("| - | - | - | - | - |")
            lines.append("")
    else:
        lines.append("실험 요약 대상이 없습니다.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build compact per-track overview from timeline events.")
    p.add_argument("--track", required=True, help="Track name (e.g. fmoe_v2)")
    p.add_argument("--events", default=str(_default_events_path()), help="events.jsonl path")
    p.add_argument("--output-csv", required=True, help="Output CSV path")
    p.add_argument("--output-md", required=True, help="Output Markdown path")
    args = p.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_md = Path(args.output_md).expanduser().resolve()

    events = _read_jsonl(events_path)
    rows, stats = _build_rows(events=events, track=args.track)
    items = _aggregate(rows)

    _write_csv(items, output_csv)
    _write_md(items, stats, track=args.track, path=output_md)

    print(f"[OK] wrote {output_csv}")
    print(f"[OK] wrote {output_md}")
    print(
        f"[INFO] matched_end={stats['matched_end']} included={stats['included']} "
        f"excluded_non_oom_error={stats['excluded_non_oom_error']} "
        f"excluded_no_metric={stats['excluded_no_metric']} summarized_experiments={len(items)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
