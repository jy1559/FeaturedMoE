#!/usr/bin/env python3
"""Summarize small-baseline round-1 hyperopt results and emit round-2 hints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


TARGET_DATASETS = {"KuaiRecSmall0.1", "lastfm0.03"}
TARGET_MODELS = {"sasrec", "gru4rec", "bsarec", "fame", "fenrec", "patt", "sigma", "srgnn"}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _to_float(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v))
    except Exception:
        return None


def _combo_slug(combo_id: str) -> str:
    return {
        "C1": "a_c1_hi_bs_short",
        "C2": "a_c2_std_bs_mid",
        "C3": "b_c3_long",
        "C4": "b_c4_wide",
    }.get(combo_id, "")


def _phase_parts(run_phase: str) -> dict[str, str]:
    m = re.search(r"^(?P<phase>P\d+?)_(?P<wave>[AB])_(?P<pair>pair\d+)_(?P<combo>C\d+)$", str(run_phase or ""))
    if not m:
        return {"phase": "", "wave": "", "pair_id": "", "combo_id": ""}
    return m.groupdict()


def _extract_lr_bounds(data: dict[str, Any]) -> tuple[float | None, float | None]:
    tuned = data.get("tuned_search")
    if not isinstance(tuned, dict):
        return None, None
    raw = tuned.get("learning_rate")
    if isinstance(raw, list) and raw:
        vals = [_to_float(v) for v in raw]
        vals = [v for v in vals if v is not None]
        if vals:
            return min(vals), max(vals)
    if isinstance(raw, str):
        vals = [_to_float(tok) for tok in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", raw, flags=re.IGNORECASE)]
        vals = [v for v in vals if v is not None]
        if vals:
            return min(vals), max(vals)
    return None, None


def _lr_position(best_lr: float | None, lo: float | None, hi: float | None) -> float | None:
    if best_lr is None or lo is None or hi is None or lo <= 0 or hi <= 0 or hi <= lo:
        return None
    if not (lo <= best_lr <= hi):
        return None
    return (math.log(best_lr) - math.log(lo)) / (math.log(hi) - math.log(lo))


def _suggest_next_range(best_lr: float | None, lo: float | None, hi: float | None) -> tuple[float | None, float | None, bool]:
    pos = _lr_position(best_lr, lo, hi)
    if pos is None or pos <= 0.25 or pos >= 0.75:
        return lo, hi, True
    next_lo = max(lo, best_lr / 3.0)
    next_hi = min(hi, best_lr * 3.0)
    return next_lo, next_hi, False


def _best_trial(data: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    best_mrr = float("-inf")
    for row in data.get("trials", []) or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("status", "")).lower() != "ok":
            continue
        mrr = _to_float(row.get("mrr@20"))
        if mrr is None:
            continue
        if mrr > best_mrr:
            best = row
            best_mrr = mrr
    return best


def _collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        data = _load_json(path)
        if not data:
            continue
        dataset = str(data.get("dataset", "")).strip()
        model = str(data.get("model", "")).strip().lower()
        if dataset not in TARGET_DATASETS or model not in TARGET_MODELS:
            continue

        parts = _phase_parts(str(data.get("run_phase", "")))
        best_trial = _best_trial(data)
        best_lr = _to_float((data.get("best_params") or {}).get("learning_rate"))
        lr_lo, lr_hi = _extract_lr_bounds(data)
        next_lo, next_hi, boundary_hit = _suggest_next_range(best_lr, lr_lo, lr_hi)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "phase": parts["phase"],
                "wave": parts["wave"],
                "pair_id": parts["pair_id"],
                "combo_id": parts["combo_id"],
                "combo_desc": _combo_slug(parts["combo_id"]),
                "best_mrr@20": _to_float(data.get("best_mrr@20")),
                "best_hr@10": _to_float(data.get("best_hr@10")),
                "test_mrr@20": _to_float(data.get("test_mrr@20")),
                "test_hr@10": _to_float(data.get("test_hr@10")),
                "best_lr": best_lr,
                "best_weight_decay": _to_float((data.get("best_params") or {}).get("weight_decay")),
                "best_dropout_ratio": _to_float((data.get("best_params") or {}).get("dropout_ratio")),
                "lr_low": lr_lo,
                "lr_high": lr_hi,
                "lr_position": _lr_position(best_lr, lr_lo, lr_hi),
                "boundary_hit": boundary_hit,
                "next_lr_low": next_lo,
                "next_lr_high": next_hi,
                "n_completed": int(_to_float(data.get("n_completed")) or 0),
                "max_evals": int(_to_float(data.get("max_evals")) or 0),
                "result_path": str(path),
                "best_trial_epochs": int(best_trial.get("epochs_run")) if best_trial.get("epochs_run") is not None else None,
            }
        )
    rows.sort(key=lambda row: (row["dataset"], row["model"], -(row.get("best_mrr@20") or 0.0), row["combo_id"]))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "phase",
        "wave",
        "pair_id",
        "combo_id",
        "combo_desc",
        "best_mrr@20",
        "best_hr@10",
        "test_mrr@20",
        "test_hr@10",
        "best_lr",
        "best_weight_decay",
        "best_dropout_ratio",
        "lr_low",
        "lr_high",
        "lr_position",
        "boundary_hit",
        "next_lr_low",
        "next_lr_high",
        "n_completed",
        "max_evals",
        "best_trial_epochs",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize small baseline round-1 results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "results" / "baseline",
        help="Directory containing baseline hyperopt result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "analysis" / "baseline_small_round1",
        help="Output directory for summary CSV/JSON.",
    )
    args = parser.parse_args()

    rows = _collect_rows(args.results_dir)
    winners: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["dataset"], row["model"])
        if key in seen:
            continue
        winners.append(row)
        seen.add(key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "all_runs.csv", rows)
    _write_csv(args.output_dir / "winners.csv", winners)
    (args.output_dir / "round2_manifest.json").write_text(
        json.dumps(
            {
                "results_dir": str(args.results_dir),
                "n_runs": len(rows),
                "n_winners": len(winners),
                "winners": winners,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[SUMMARY] all_runs={args.output_dir / 'all_runs.csv'}")
    print(f"[SUMMARY] winners={args.output_dir / 'winners.csv'}")
    print(f"[SUMMARY] round2_manifest={args.output_dir / 'round2_manifest.json'}")


if __name__ == "__main__":
    main()
