#!/usr/bin/env python3
"""Rank FMoE_N4 result files by seen-target MRR for promotion decisions.

This is mainly for the current Stage2 situation where runs may finish with
all-target main metrics in the top-level fields, while seen-target metrics live
inside special_metrics. Stage3 promotion should read the seen-target slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RESULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "results" / "fmoe_n4"


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _nested_metric(block: dict | None, *keys: str) -> float | None:
    current: Any = block or {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _to_float(current)


def _seen_valid_mrr(payload: dict) -> float | None:
    valid_filter = payload.get("best_valid_main_eval_filter", {}) or {}
    if bool(valid_filter.get("enabled", False)):
        return _to_float(payload.get("best_mrr@20")) or _nested_metric(payload.get("best_valid_result"), "mrr@20")
    return (
        _nested_metric(payload.get("best_valid_special_metrics"), "overall_seen_target", "mrr@20")
        or _to_float(payload.get("best_mrr@20"))
        or _nested_metric(payload.get("best_valid_result"), "mrr@20")
    )


def _seen_test_mrr(payload: dict) -> float | None:
    test_filter = payload.get("test_main_eval_filter", {}) or {}
    if bool(test_filter.get("enabled", False)):
        return _to_float(payload.get("test_mrr@20")) or _nested_metric(payload.get("test_result"), "mrr@20")
    return (
        _nested_metric(payload.get("test_special_metrics"), "overall_seen_target", "mrr@20")
        or _to_float(payload.get("test_mrr@20"))
        or _nested_metric(payload.get("test_result"), "mrr@20")
    )


def _result_rows(dataset: str, axis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    axis_norm = str(axis).strip().casefold()
    for path in sorted(RESULT_ROOT.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("dataset", "")) != str(dataset):
            continue
        if axis_norm and str(payload.get("run_axis", "")).strip().casefold() != axis_norm:
            continue
        rows.append(
            {
                "path": path,
                "run_axis": str(payload.get("run_axis", "")),
                "run_phase": str(payload.get("run_phase", "")),
                "best_seen_mrr@20": _seen_valid_mrr(payload),
                "test_seen_mrr@20": _seen_test_mrr(payload),
                "best_main_mrr@20": _to_float(payload.get("best_mrr@20")),
                "test_main_mrr@20": _to_float(payload.get("test_mrr@20")),
                "filter_valid": bool((payload.get("best_valid_main_eval_filter", {}) or {}).get("enabled", False)),
                "filter_test": bool((payload.get("test_main_eval_filter", {}) or {}).get("enabled", False)),
            }
        )
    rows.sort(
        key=lambda row: (
            -1.0 if row.get("best_seen_mrr@20") is None else -float(row["best_seen_mrr@20"]),
            -1.0 if row.get("test_seen_mrr@20") is None else -float(row["test_seen_mrr@20"]),
            str(row.get("run_phase", "")),
        )
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank FMoE_N4 results by seen-target MRR")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--axis", default="Stage2_A12_MixedTemplates")
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    rows = _result_rows(str(args.dataset), str(args.axis))
    if not rows:
        print("No matching result files found.")
        return 1

    topk = max(1, int(args.topk))
    print("rank\trun_phase\tbest_seen_mrr20\ttest_seen_mrr20\tbest_main_mrr20\ttest_main_mrr20\tvalid_filter\ttest_filter\tpath")
    for idx, row in enumerate(rows[:topk], start=1):
        best_seen = row["best_seen_mrr@20"]
        test_seen = row["test_seen_mrr@20"]
        best_main = row["best_main_mrr@20"]
        test_main = row["test_main_mrr@20"]
        print(
            f"{idx}\t{row['run_phase']}\t"
            f"{'' if best_seen is None else f'{best_seen:.6f}'}\t"
            f"{'' if test_seen is None else f'{test_seen:.6f}'}\t"
            f"{'' if best_main is None else f'{best_main:.6f}'}\t"
            f"{'' if test_main is None else f'{test_main:.6f}'}\t"
            f"{int(bool(row['filter_valid']))}\t{int(bool(row['filter_test']))}\t{row['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())