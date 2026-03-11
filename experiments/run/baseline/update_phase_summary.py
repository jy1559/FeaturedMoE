#!/usr/bin/env python3
"""Build per-dataset/per-phase markdown summary for baseline runs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


LOG_NAME_RE = re.compile(r"^(?P<model>[^_]+)_(?P<idx>\d{3})_(?P<desc>.+)\.log$")
LEGACY_LOG_NAME_RE = re.compile(r"^(?P<idx>\d{3})_(?P<model>[^_]+)_(?P<desc>.+)\.log$")
TRIAL_PROGRESS_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+", re.MULTILINE)
METRIC_KEYS = ("best_mrr@20", "best_hr@10", "test_mrr@20", "test_hr@10")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiments_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _logs_root() -> Path:
    raw = os.environ.get("RUN_LOGS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _experiments_root() / "run" / "artifacts" / "logs"


def _results_root() -> Path:
    raw = os.environ.get("HYPEROPT_RESULTS_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _experiments_root() / "run" / "artifacts" / "results"


def _timeline_dir() -> Path:
    raw = os.environ.get("RUN_TIMELINE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _experiments_root() / "run" / "artifacts" / "timeline"


def _events_path() -> Path:
    return _timeline_dir() / "events.jsonl"


def _ignored_events_path() -> Path:
    return _timeline_dir() / "ignored_events.jsonl"


def _state_dir() -> Path:
    return _timeline_dir() / "state"


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


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def _norm_dataset(raw: str) -> str:
    return str(raw or "").strip().lower()


def _phase_folder_from_run_phase(phase: str) -> str:
    tokens = [tok for tok in str(phase or "").split("_") if tok]
    if not tokens:
        return ""
    for i, tok in enumerate(tokens):
        if tok in {"A", "B"}:
            return "_".join(tokens[:i]) if i > 0 else tokens[0]
    return "_".join(tokens)


def _phase_slug(raw: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", str(raw or "").strip().lower()).strip("._-")


def _resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(str(raw).strip())
    if not str(p):
        return None
    return p.expanduser().resolve()


def _canonical_log_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = path
    raw = str(candidate)
    old_seg = f"{os.sep}run{os.sep}logs{os.sep}baseline{os.sep}"
    new_seg = f"{os.sep}run{os.sep}artifacts{os.sep}logs{os.sep}baseline{os.sep}"
    if old_seg in raw and new_seg not in raw:
        candidate = Path(raw.replace(old_seg, new_seg))

    m = LEGACY_LOG_NAME_RE.match(candidate.name)
    if not m:
        return candidate
    new_name = f"{m.group('model')}_{m.group('idx')}_{m.group('desc')}.log"
    return candidate.with_name(new_name)


def _extract_metric_from_trial(trial: dict[str, Any], key: str) -> float | None:
    if key == "best_mrr@20":
        return _to_float(trial.get("mrr@20"))
    if key == "best_hr@10":
        return _to_float(trial.get("best_hr@10")) or _to_float((trial.get("valid_result") or {}).get("hit@10"))
    if key == "test_mrr@20":
        return _to_float(trial.get("test_mrr@20")) or _to_float((trial.get("test_result") or {}).get("mrr@20"))
    if key == "test_hr@10":
        return _to_float(trial.get("test_hr@10")) or _to_float((trial.get("test_result") or {}).get("hit@10"))
    return None


def _extract_metric_from_result(data: dict[str, Any], key: str) -> float | None:
    if key == "best_mrr@20":
        return _to_float(data.get("best_mrr@20")) or _to_float((data.get("best_valid_result") or {}).get("mrr@20"))
    if key == "best_hr@10":
        return _to_float(data.get("best_hr@10")) or _to_float((data.get("best_valid_result") or {}).get("hit@10"))
    if key == "test_mrr@20":
        return _to_float(data.get("test_mrr@20")) or _to_float((data.get("test_result") or {}).get("mrr@20"))
    if key == "test_hr@10":
        return _to_float(data.get("test_hr@10")) or _to_float((data.get("test_result") or {}).get("hit@10"))
    return None


def _current_metrics(data: dict[str, Any]) -> dict[str, float | None]:
    trials = data.get("trials")
    if isinstance(trials, list) and trials:
        last = trials[-1] if isinstance(trials[-1], dict) else {}
        return {key: _extract_metric_from_trial(last, key) for key in METRIC_KEYS}
    return {key: _extract_metric_from_result(data, key) for key in METRIC_KEYS}


def _run_best_metrics(data: dict[str, Any]) -> dict[str, float | None]:
    best = {key: None for key in METRIC_KEYS}
    trials = data.get("trials")
    if isinstance(trials, list):
        for trial in trials:
            if not isinstance(trial, dict):
                continue
            for key in METRIC_KEYS:
                val = _extract_metric_from_trial(trial, key)
                if val is None:
                    continue
                if best[key] is None or val > best[key]:
                    best[key] = val
    for key in METRIC_KEYS:
        if best[key] is None:
            best[key] = _extract_metric_from_result(data, key)
    return best


def _fmt_metric(v: float | None) -> str:
    return "-" if v is None else f"{v:.4f}"


def _metric_equal(lhs: float | None, rhs: float | None, eps: float = 1e-12) -> bool:
    if lhs is None or rhs is None:
        return False
    return abs(float(lhs) - float(rhs)) <= eps


def _row_should_display(row: "Row") -> bool:
    for key in METRIC_KEYS:
        if _metric_equal(row.current.get(key), row.run_best.get(key)):
            return True
    return False


def _trial_progress_from_log(log_path: Path) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "-"
    matches = TRIAL_PROGRESS_RE.findall(text)
    if not matches:
        return "-"
    done, total = matches[-1]
    return f"{int(done)}/{int(total)}"


def _trial_progress(data: dict[str, Any], cmd: str, log_path: Path) -> str:
    completed = data.get("n_completed")
    total = data.get("max_evals")
    if completed is not None and total is not None:
        c = int(_to_float(completed) or 0)
        t = int(_to_float(total) or 0)
        if t > 0:
            return f"{c}/{t}"
    m = re.search(r"--max-evals\s+(\d+)", str(cmd or ""))
    if m:
        current = _trial_progress_from_log(log_path)
        if current != "-":
            return current
        return f"-/{m.group(1)}"
    return _trial_progress_from_log(log_path)


def _result_candidate(dataset: str, model: str, full_phase: str) -> Path | None:
    results_dir = _results_root() / "baseline"
    if not results_dir.exists():
        return None
    dataset_norm = _norm_dataset(dataset)
    phase_norm = _phase_slug(full_phase)
    model_norm = str(model or "").strip().lower()
    candidates: list[Path] = []
    for path in results_dir.glob("*.json"):
        name = path.name.lower()
        if not name.startswith(f"{dataset_norm}_"):
            continue
        if model_norm and f"_{model_norm}_" not in name:
            continue
        if phase_norm and f"_{phase_norm}_" not in name:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def _compute_model_folder_best(rows: list["Row"]) -> dict[str, float | None]:
    best = {key: None for key in METRIC_KEYS}
    for row in rows:
        for key in METRIC_KEYS:
            val = row.run_best.get(key)
            if val is None:
                val = row.current.get(key)
            if val is None:
                continue
            if best[key] is None or val > best[key]:
                best[key] = val
    return best


def _group_rows_by_model(rows: list["Row"]) -> list[tuple[str, list["Row"]]]:
    groups: dict[str, list[Row]] = {}
    order: list[str] = []
    for row in rows:
        model_key = str(row.model or "").strip() or "unknown"
        if model_key not in groups:
            groups[model_key] = []
            order.append(model_key)
        groups[model_key].append(row)
    return [(model, groups[model]) for model in order]


@dataclass
class Row:
    sort_idx: int
    log_name: str
    model: str
    run_phase: str
    status: str
    trials: str
    current: dict[str, float | None]
    run_best: dict[str, float | None]
    folder_best: dict[str, float | None]
    result_name: str


def _build_rows(dataset: str, phase: str) -> list[Row]:
    logs_dir = _logs_root() / "baseline" / dataset / phase
    events = _read_jsonl(_events_path())
    ignored = _read_jsonl(_ignored_events_path())
    ended_run_ids = {str(e.get("run_id", "")) for e in events if e.get("track") == "baseline" and e.get("event_type") == "end"}

    starts_by_log: dict[Path, dict[str, Any]] = {}
    ends_by_log: dict[Path, dict[str, Any]] = {}
    ignored_by_log: dict[Path, dict[str, Any]] = {}

    for row in events:
        if row.get("track") != "baseline":
            continue
        log_path = _canonical_log_path(_resolve_path(row.get("log_file")))
        if log_path is None:
            continue
        row_phase = _phase_folder_from_run_phase(str(row.get("phase", "")))
        if _norm_dataset(row.get("dataset")) != _norm_dataset(dataset) or row_phase != phase:
            continue
        if row.get("event_type") == "start":
            starts_by_log[log_path] = row
        elif row.get("event_type") == "end":
            ends_by_log[log_path] = row

    for row in ignored:
        if row.get("track") != "baseline":
            continue
        log_path = _canonical_log_path(_resolve_path(row.get("log_file")))
        if log_path is None:
            continue
        row_phase = _phase_folder_from_run_phase(str(row.get("phase", "")))
        if _norm_dataset(row.get("dataset")) != _norm_dataset(dataset) or row_phase != phase:
            continue
        ignored_by_log[log_path] = row

    active_state_by_log: dict[Path, dict[str, Any]] = {}
    for path in _state_dir().glob("*.json"):
        data = _read_json(path)
        if data.get("track") != "baseline":
            continue
        if str(data.get("run_id", "")) in ended_run_ids:
            continue
        log_path = _canonical_log_path(_resolve_path(data.get("log_file")))
        if log_path is None:
            continue
        row_phase = _phase_folder_from_run_phase(str(data.get("phase", "")))
        if _norm_dataset(data.get("dataset")) != _norm_dataset(dataset) or row_phase != phase:
            continue
        active_state_by_log[log_path] = data

    rows: list[Row] = []
    if not logs_dir.exists():
        return rows

    for log_path in sorted(logs_dir.glob("*.log"), key=lambda p: p.name):
        m = LOG_NAME_RE.match(log_path.name)
        if not m:
            continue
        start = starts_by_log.get(log_path)
        end = ends_by_log.get(log_path)
        ignored_end = ignored_by_log.get(log_path)
        active = active_state_by_log.get(log_path)

        model = str((end or start or active or {}).get("model") or m.group("model"))
        full_phase = str((end or start or active or {}).get("phase") or phase)
        status = "unknown"
        if end:
            status = str(end.get("status") or "success")
        elif ignored_end:
            status = str(ignored_end.get("status") or "fail")
        elif active or start:
            status = "running"

        result_path = _resolve_path((end or {}).get("result_file"))
        if result_path is None:
            result_path = _result_candidate(dataset, model, full_phase)
        result_data = _read_json(result_path)
        cmd = str((end or start or active or {}).get("cmd") or "")
        trials = _trial_progress(result_data, cmd, log_path)
        current = _current_metrics(result_data)
        run_best = _run_best_metrics(result_data)

        rows.append(
            Row(
                sort_idx=int(m.group("idx")),
                log_name=log_path.name,
                model=str(model),
                run_phase=full_phase,
                status=status,
                trials=trials,
                current=current,
                run_best=run_best,
                folder_best={key: None for key in METRIC_KEYS},
                result_name=result_path.name if result_path else "-",
            )
        )

    rows.sort(key=lambda r: (r.sort_idx, r.log_name))
    for _model, model_rows in _group_rows_by_model(rows):
        model_best = _compute_model_folder_best(model_rows)
        for row in model_rows:
            row.folder_best = dict(model_best)
    return rows


def _write_summary(dataset: str, phase: str, rows: list[Row]) -> Path:
    out_path = _logs_root() / "baseline" / dataset / f"{phase}_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")

    grouped_rows = _group_rows_by_model(rows)
    visible_grouped_rows: list[tuple[str, list[Row], int]] = []
    total_visible = 0
    for model, model_rows in grouped_rows:
        visible_rows = [row for row in model_rows if _row_should_display(row)]
        if not visible_rows:
            continue
        model_best = _compute_model_folder_best(visible_rows)
        for row in visible_rows:
            row.folder_best = dict(model_best)
        visible_grouped_rows.append((model, visible_rows, len(model_rows)))
        total_visible += len(visible_rows)

    lines: list[str] = []
    lines.append(f"# {dataset} {phase} Summary")
    lines.append("")
    lines.append(f"- updated_at_utc: {_utc_now()}")
    lines.append(f"- dataset: {dataset}")
    lines.append(f"- phase_folder: {phase}")
    lines.append(f"- logs_present: {len(rows)}")
    lines.append(f"- rows_shown: {total_visible}")
    lines.append(f"- models_shown: {len(visible_grouped_rows)}")
    lines.append("")
    lines.append("## Metric Layout")
    lines.append("")
    lines.append("- Each metric block is shown as `cur | run | folder`.")
    lines.append("- `folder` means the best `run` value inside the same model table below.")
    lines.append("- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.")
    lines.append("- A row is shown only if the current trial still matches the run-best on at least one metric.")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    if not visible_grouped_rows:
        lines.append("- No visible rows from existing log files.")
    else:
        for model, visible_rows, total_model_rows in visible_grouped_rows:
            folder_best = visible_rows[0].folder_best if visible_rows else {key: None for key in METRIC_KEYS}
            lines.append("<details>")
            lines.append(
                f"<summary><strong>{model}</strong> "
                f"(shown {len(visible_rows)}/{total_model_rows}, "
                f"best_mrr@20={_fmt_metric(folder_best['best_mrr@20'])}, "
                f"best_hr@10={_fmt_metric(folder_best['best_hr@10'])})</summary>"
            )
            lines.append("")
            lines.append(f"- shown_rows: {len(visible_rows)} / total_logs_for_model: {total_model_rows}")
            lines.append(
                f"- folder_best: "
                f"best_mrr@20={_fmt_metric(folder_best['best_mrr@20'])}, "
                f"best_hr@10={_fmt_metric(folder_best['best_hr@10'])}, "
                f"test_mrr@20={_fmt_metric(folder_best['test_mrr@20'])}, "
                f"test_hr@10={_fmt_metric(folder_best['test_hr@10'])}"
            )
            lines.append("")
            lines.append(
                "| experiment | run_phase | status | trials | "
                "best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | "
                "best_hr@10 cur | best_hr@10 run | best_hr@10 folder | "
                "test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | "
                "test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |"
            )
            lines.append(
                "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
            )
            for row in visible_rows:
                lines.append(
                    f"| {row.log_name} | {row.run_phase or '-'} | {row.status or '-'} | {row.trials} | "
                    f"{_fmt_metric(row.current['best_mrr@20'])} | {_fmt_metric(row.run_best['best_mrr@20'])} | {_fmt_metric(row.folder_best['best_mrr@20'])} | "
                    f"{_fmt_metric(row.current['best_hr@10'])} | {_fmt_metric(row.run_best['best_hr@10'])} | {_fmt_metric(row.folder_best['best_hr@10'])} | "
                    f"{_fmt_metric(row.current['test_mrr@20'])} | {_fmt_metric(row.run_best['test_mrr@20'])} | {_fmt_metric(row.folder_best['test_mrr@20'])} | "
                    f"{_fmt_metric(row.current['test_hr@10'])} | {_fmt_metric(row.run_best['test_hr@10'])} | {_fmt_metric(row.folder_best['test_hr@10'])} | "
                    f"{row.result_name} |"
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    if fcntl is None:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path

    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Update baseline dataset/phase markdown summary.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. KuaiRecSmall0.1")
    parser.add_argument("--phase", required=True, help="Phase folder name, e.g. P0 or P0_SMOKE")
    args = parser.parse_args()

    rows = _build_rows(args.dataset, args.phase)
    out_path = _write_summary(args.dataset, args.phase, rows)
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
