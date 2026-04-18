#!/usr/bin/env python3
"""Flatten final_experiment result JSON files into notebook-friendly CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts" / "results" / "final_experiment"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "writing" / "260418_final_exp_figure" / "data"
NINE_METRICS = ["hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@5", "mrr@10", "mrr@20"]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_mean(metrics: dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    values = [_safe_float(metrics.get(key)) for key in NINE_METRICS if key in metrics]
    if values:
        return float(sum(values) / len(values))
    return _safe_float(metrics.get("mrr@20"))


def _special_block(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get(key) or {}
    return block if isinstance(block, dict) else {}


def _extract_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    valid = payload.get("best_valid_result") or {}
    test = payload.get("test_result") or {}
    valid_special = payload.get("best_valid_special_metrics") or {}
    test_special = payload.get("test_special_metrics") or {}
    valid_seen = _special_block(valid_special, "overall_seen_target")
    test_seen = _special_block(test_special, "overall_seen_target")
    valid_unseen = _special_block(valid_special, "overall_unseen_target")
    test_unseen = _special_block(test_special, "overall_unseen_target")

    row = {
        "result_file": str(path.resolve()),
        "dataset": str(payload.get("dataset", "")),
        "dataset_raw": str(payload.get("dataset_raw", "")),
        "model": str(payload.get("model", "")),
        "run_group": str(payload.get("run_group", "")),
        "run_axis": str(payload.get("run_axis", "")),
        "run_phase": str(payload.get("run_phase", "")),
        "timestamp": str(payload.get("timestamp", "")),
        "n_completed": int(payload.get("n_completed", 0) or 0),
        "max_evals": int(payload.get("max_evals", 0) or 0),
        "selection_rule": "overall_seen_target",
        "best_checkpoint_file": str(payload.get("best_checkpoint_file", "") or ""),
        "special_result_file": str(payload.get("special_result_file", "") or ""),
        "logging_bundle_dir": str(payload.get("logging_bundle_dir", "") or ""),
        "diag_meta_file": str(payload.get("diag_meta_file", "") or ""),
        "diag_tier_a_final_file": str(payload.get("diag_tier_a_final_file", "") or ""),
        "diag_raw_best_valid_file": str(payload.get("diag_raw_best_valid_file", "") or ""),
        "best_valid_mean": _metric_mean(valid),
        "test_mean": _metric_mean(test),
        "best_valid_seen_mean": _metric_mean(valid_seen),
        "test_seen_mean": _metric_mean(test_seen),
        "best_valid_unseen_mean": _metric_mean(valid_unseen),
        "test_unseen_mean": _metric_mean(test_unseen),
    }

    for split_name, block in (
        ("best_valid", valid),
        ("test", test),
        ("best_valid_seen", valid_seen),
        ("test_seen", test_seen),
        ("best_valid_unseen", valid_unseen),
        ("test_unseen", test_unseen),
    ):
        for metric in NINE_METRICS:
            row[f"{split_name}_{metric.replace('@', '')}"] = _safe_float(block.get(metric))
    return row


def _best_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _safe_float(row.get("best_valid_seen_mrr20")),
        _safe_float(row.get("best_valid_mrr20")),
        _safe_float(row.get("test_seen_mrr20")),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export final_experiment result JSON files into flat CSV tables.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--glob", default="*.json", help="Glob pattern under results-root.")
    parser.add_argument("--prefix", default="final_experiment", help="Output filename prefix.")
    args = parser.parse_args()

    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    result_paths = sorted(results_root.rglob(str(args.glob)))
    rows: list[dict[str, Any]] = []
    for path in result_paths:
        if path.name.endswith("_special_metrics.json"):
            continue
        if "normal" in path.parts:
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict) or not payload.get("dataset") or not payload.get("model"):
            continue
        rows.append(_extract_row(path, payload))

    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("run_phase", ""))))
    _write_csv(output_dir / f"{args.prefix}_result_rows.csv", rows)

    best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        current = best_by_pair.get(key)
        if current is None or _best_key(row) > _best_key(current):
            best_by_pair[key] = row
    best_rows = sorted(best_by_pair.values(), key=lambda row: (str(row.get("dataset", "")), str(row.get("model", ""))))
    _write_csv(output_dir / f"{args.prefix}_best_by_dataset_model.csv", best_rows)

    print(f"[DONE] rows={len(rows)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
