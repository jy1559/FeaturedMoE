#!/usr/bin/env python3
"""Collect FMoE/HiR tuning results from JSON first, then log fallback.

Outputs:
  - summary.csv
  - summary.md
  - best_by_dataset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_REPO_ROOT = "/workspace/jy1559/FMoE"
DEFAULT_DATASETS = [
    "movielens1m",
    "retail_rocket",
    "amazon_beauty",
    "foursquare",
    "kuairec0.3",
    "lastfm0.3",
]
KNOWN_GROUPS = {"baseline", "fmoe", "fmoe_hir"}

BEST_LINE_RE = re.compile(
    r"Best\s+([A-Za-z0-9@._-]+)\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE
)
DONE_RE = re.compile(r"\bDONE\s+\|\s+(\S+)\s+x\s+(\S+)", re.IGNORECASE)
HEADER_RE = re.compile(r"Hyperopt TPE\s+\|\s+(\S+)\s+x\s+(\S+)", re.IGNORECASE)
TS_FILE_RE = re.compile(r"(\d{8}_\d{6}(?:_\d{3})?)")


@dataclass
class Record:
    dataset: str
    dataset_raw: str
    model: str
    metric: str
    metric_value: float
    run_group: str
    run_axis: str
    run_phase: str
    timestamp: str
    source_type: str
    source_file: str
    n_completed: int
    n_trials: int
    parent_result: str
    best_params_json: str


def canonical_dataset(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    aliases = {
        "kuairec0.3": "kuairec0.3",
        "kuairec": "kuairec0.3",
        "kuairec-0.3": "kuairec0.3",
        "kuai_rec0.3": "kuairec0.3",
        "retailrocket": "retail_rocket",
        "retail-rocket": "retail_rocket",
        "movielens-1m": "movielens1m",
        "movielens_1m": "movielens1m",
        "ml1m": "movielens1m",
        "lastfm": "lastfm0.3",
    }
    return aliases.get(low, low)


def normalize_metric(metric: str) -> str:
    m = (metric or "").strip().lower()
    return m or "mrr@20"


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items


def parse_timestamp(payload: dict, source: Path) -> str:
    ts = str(payload.get("timestamp", "")).strip()
    if ts:
        return ts
    m = TS_FILE_RE.search(source.name)
    if m:
        stamp = m.group(1)
        for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
            try:
                parsed = datetime.strptime(stamp, fmt)
                return parsed.isoformat()
            except ValueError:
                continue
    return datetime.fromtimestamp(source.stat().st_mtime).isoformat()


def infer_group_from_path(path: Path) -> str:
    for part in path.parts:
        if part in KNOWN_GROUPS:
            return part
    return ""


def infer_group_from_model(model_name: str) -> str:
    m = (model_name or "").lower()
    if "hir" in m:
        return "fmoe_hir"
    if "featuredmoe" in m:
        return "fmoe"
    return ""


def find_json_files(results_roots: Sequence[Path]) -> list[Path]:
    cands: list[Path] = []
    patterns = ["*.json", "baseline/*.json", "fmoe/*.json", "fmoe_hir/*.json"]
    for results_root in results_roots:
        if not results_root.exists():
            continue
        for pat in patterns:
            cands.extend(results_root.glob(pat))
    uniq = sorted({p.resolve() for p in cands})
    return uniq


def metric_from_trials(payload: dict, metric: str) -> float:
    best = float("-inf")
    trials = payload.get("trials", [])
    if not isinstance(trials, list):
        return best
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        value = None
        if metric in trial and isinstance(trial.get(metric), (int, float)):
            value = float(trial[metric])
        else:
            vr = trial.get("valid_result", {})
            if isinstance(vr, dict) and isinstance(vr.get(metric), (int, float)):
                value = float(vr[metric])
        if value is not None and value > best:
            best = value
    return best


def parse_metric(payload: dict, metric: str) -> float:
    best_key = f"best_{metric}"
    if isinstance(payload.get(best_key), (int, float)):
        return float(payload[best_key])

    best_vr = payload.get("best_valid_result", {})
    if isinstance(best_vr, dict) and isinstance(best_vr.get(metric), (int, float)):
        return float(best_vr[metric])

    from_trials = metric_from_trials(payload, metric)
    return from_trials


def to_best_params_json(payload: dict) -> str:
    bp = payload.get("best_params", {})
    if not isinstance(bp, dict):
        return ""
    return json.dumps(bp, sort_keys=True, ensure_ascii=False)


def parse_json_records(
    files: Sequence[Path],
    dataset_filter: set[str],
    metric: str,
) -> list[Record]:
    out: list[Record] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ds_raw = str(payload.get("dataset", "")).strip()
        ds = canonical_dataset(ds_raw)
        if not ds:
            continue
        if dataset_filter and ds not in dataset_filter:
            continue

        metric_value = parse_metric(payload, metric)
        if metric_value == float("-inf"):
            continue

        model_name = str(payload.get("model", "") or "")
        run_group = str(payload.get("run_group", "")).strip().lower()
        if not run_group:
            run_group = infer_group_from_path(path)
        if not run_group:
            run_group = infer_group_from_model(model_name)

        run_axis = str(payload.get("run_axis", "")).strip().lower()
        run_phase = str(payload.get("run_phase", "")).strip()
        n_completed = int(payload.get("n_completed", 0) or 0)
        n_trials = len(payload.get("trials", []) or [])
        parent_result = str(payload.get("parent_result", "") or "")

        out.append(
            Record(
                dataset=ds,
                dataset_raw=ds_raw or ds,
                model=model_name,
                metric=metric,
                metric_value=float(metric_value),
                run_group=run_group,
                run_axis=run_axis,
                run_phase=run_phase,
                timestamp=parse_timestamp(payload, path),
                source_type="json",
                source_file=str(path),
                n_completed=n_completed,
                n_trials=n_trials,
                parent_result=parent_result,
                best_params_json=to_best_params_json(payload),
            )
        )
    return out


def parse_model_dataset_from_log(text: str) -> tuple[str, str]:
    for rx in (DONE_RE, HEADER_RE):
        m = rx.search(text)
        if m:
            return m.group(1), canonical_dataset(m.group(2))
    return "", ""


def parse_best_metric_from_log(text: str, metric: str) -> float:
    metric_key = metric.lower()
    best = float("-inf")
    for m in BEST_LINE_RE.finditer(text):
        key = m.group(1).strip().lower()
        if key == metric_key:
            try:
                value = float(m.group(2))
            except ValueError:
                continue
            if value > best:
                best = value
    return best


def parse_log_fallback_records(
    log_roots: Sequence[Path],
    metric: str,
    need_datasets: set[str],
) -> list[Record]:
    if not need_datasets:
        return []

    out: list[Record] = []
    for log_root in log_roots:
        if not log_root.exists():
            continue
        for path in sorted(log_root.rglob("*.log")):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            model, dataset = parse_model_dataset_from_log(text)
            if not dataset or dataset not in need_datasets:
                continue

            metric_value = parse_best_metric_from_log(text, metric)
            if metric_value == float("-inf"):
                continue

            rel = path.relative_to(log_root)
            parts = rel.parts
            run_group = parts[0] if len(parts) > 0 else ""
            run_axis = parts[1] if len(parts) > 1 else ""
            run_phase = parts[2] if len(parts) > 2 else ""

            out.append(
                Record(
                    dataset=dataset,
                    dataset_raw=dataset,
                    model=model,
                    metric=metric,
                    metric_value=float(metric_value),
                    run_group=run_group,
                    run_axis=run_axis,
                    run_phase=run_phase,
                    timestamp=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    source_type="log_fallback",
                    source_file=str(path),
                    n_completed=0,
                    n_trials=0,
                    parent_result="",
                    best_params_json="",
                )
            )
    return out


def sort_records(records: Iterable[Record]) -> list[Record]:
    def _key(r: Record) -> tuple[str, float, str]:
        return (r.dataset, r.metric_value, r.timestamp)

    return sorted(records, key=_key, reverse=False)


def write_summary_csv(path: Path, records: Sequence[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "dataset_raw",
                "model",
                "metric",
                "metric_value",
                "run_group",
                "run_axis",
                "run_phase",
                "timestamp",
                "source_type",
                "source_file",
                "n_completed",
                "n_trials",
                "parent_result",
                "best_params_json",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.dataset,
                    r.dataset_raw,
                    r.model,
                    r.metric,
                    f"{r.metric_value:.6f}",
                    r.run_group,
                    r.run_axis,
                    r.run_phase,
                    r.timestamp,
                    r.source_type,
                    r.source_file,
                    r.n_completed,
                    r.n_trials,
                    r.parent_result,
                    r.best_params_json,
                ]
            )


def pick_best_by_dataset(records: Sequence[Record], datasets: Sequence[str]) -> dict[str, dict]:
    grouped: dict[str, list[Record]] = {}
    for r in records:
        grouped.setdefault(r.dataset, []).append(r)

    out: dict[str, dict] = {}
    for ds in datasets:
        rows = grouped.get(ds, [])
        if not rows:
            out[ds] = {"status": "missing"}
            continue
        best = max(rows, key=lambda x: x.metric_value)
        out[ds] = {
            "status": "ok",
            "dataset": best.dataset,
            "dataset_raw": best.dataset_raw,
            "model": best.model,
            "metric": best.metric,
            "metric_value": best.metric_value,
            "run_group": best.run_group,
            "run_axis": best.run_axis,
            "run_phase": best.run_phase,
            "timestamp": best.timestamp,
            "source_type": best.source_type,
            "source_file": best.source_file,
            "best_params": json.loads(best.best_params_json) if best.best_params_json else {},
        }
    return out


def write_summary_md(
    path: Path,
    records: Sequence[Record],
    best_by_dataset: dict[str, dict],
    metric: str,
) -> None:
    lines: list[str] = []
    lines.append("# FMoE Result Summary")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat()}")
    lines.append(f"- metric: {metric}")
    lines.append(f"- total_records: {len(records)}")
    lines.append("")
    lines.append("## Best by Dataset")
    lines.append("")
    lines.append(
        "| dataset | model | metric | run_group | run_axis | run_phase | source | timestamp |"
    )
    lines.append("|---|---|---:|---|---|---|---|---|")
    for ds, payload in best_by_dataset.items():
        if payload.get("status") != "ok":
            lines.append(f"| {ds} | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {dataset} | {model} | {metric_value:.4f} | {run_group} | {run_axis} | {run_phase} | {source_type} | {timestamp} |".format(
                **payload
            )
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- artifacts/results JSON을 우선 사용하고, 누락 시 legacy 결과/로그 fallback을 사용한다.")
    lines.append("- 채택 기준은 단일 best score이며 재현성 제약을 기본으로 두지 않는다.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect FMoE experiment results.")
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--metric", default="mrr@20")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    metric = normalize_metric(args.metric)
    datasets_raw = parse_csv_list(args.datasets)
    datasets = [canonical_dataset(x) for x in datasets_raw] if datasets_raw else []
    dataset_filter = set(datasets)

    results_roots = [
        repo_root / "experiments" / "run" / "artifacts" / "results",
        repo_root / "experiments" / "run" / "hyperopt_results",
    ]
    log_roots = [
        repo_root / "experiments" / "run" / "artifacts" / "logs",
        repo_root / "experiments" / "run" / "log",
    ]
    default_out_dir = repo_root / "experiments" / "run" / "artifacts" / "results"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = find_json_files(results_roots)
    json_records = parse_json_records(json_files, dataset_filter, metric)

    json_hit_datasets = {r.dataset for r in json_records}
    need_log_fallback = set(dataset_filter) - json_hit_datasets if dataset_filter else set()
    log_records = parse_log_fallback_records(log_roots, metric, need_log_fallback)

    all_records = json_records + log_records
    all_records = sorted(
        all_records,
        key=lambda r: (r.dataset, -r.metric_value, r.timestamp, r.model),
    )

    ordered_datasets = datasets if datasets else sorted({r.dataset for r in all_records})
    best_by_dataset = pick_best_by_dataset(all_records, ordered_datasets)

    summary_csv = out_dir / "summary.csv"
    summary_md = out_dir / "summary.md"
    best_json = out_dir / "best_by_dataset.json"

    write_summary_csv(summary_csv, all_records)
    write_summary_md(summary_md, all_records, best_by_dataset, metric)
    best_json.write_text(
        json.dumps(best_by_dataset, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] summary.csv: {summary_csv}")
    print(f"[OK] summary.md: {summary_md}")
    print(f"[OK] best_by_dataset.json: {best_json}")
    print(
        "[INFO] records: json={json_count}, log_fallback={log_count}, total={total}".format(
            json_count=len(json_records),
            log_count=len(log_records),
            total=len(all_records),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
