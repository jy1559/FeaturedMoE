#!/usr/bin/env python3
"""Analyze ablation_2 router-input runs and plot feature-vs-routing similarity."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

DEFAULT_AXIS = "ablation_2_router_input_v3"
DEFAULT_RESULTS_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4")
DEFAULT_OUT_ROOT = Path("/workspace/FeaturedMoE/experiments/run/artifacts/logs/fmoe_n4/ablation") / DEFAULT_AXIS / "analysis"

SETTING_LABEL = {
    "RI-00": "baseline",
    "RI-01": "hidden_only",
    "RI-02": "hidden_plus_feature",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def _collect_runs(results_root: Path, axis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.glob("*.json")):
        if path.name == "meta.json":
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        if str(payload.get("run_axis") or "") != axis:
            continue
        run_phase = str(payload.get("run_phase") or "")
        setting_id = str(payload.get("phase_setting_id") or "")
        seed_id = int(_to_float(payload.get("phase_seed_id"), 0))
        test_mrr = _to_float(payload.get("test_mrr@20"), -1.0)
        best_valid = _to_float(payload.get("best_valid_mrr@20", payload.get("best_valid_score")), -1.0)
        diag_path = Path(str(payload.get("diag_raw_best_valid_file") or "").strip())
        rows.append(
            {
                "run_phase": run_phase,
                "setting_id": setting_id,
                "setting_label": SETTING_LABEL.get(setting_id, setting_id.lower() or "unknown"),
                "seed_id": seed_id,
                "test_mrr20": test_mrr,
                "best_valid_mrr20": best_valid,
                "diag_raw_best_valid_file": str(diag_path) if diag_path else "",
                "result_json": str(path),
            }
        )
    return rows


def _extract_stage_pairs(diag_payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stage_metrics = dict(diag_payload.get("stage_metrics", {}) or {})
    for stage_key, stage_payload_any in stage_metrics.items():
        stage_payload = dict(stage_payload_any or {})
        block = dict(stage_payload.get("feature_route_pair_similarity", {}) or {})
        rows = list(block.get("sample_points", []) or [])
        for row in rows:
            row = dict(row or {})
            row["stage_key"] = str(stage_key)
            out.append(row)
    return out


def analyze(axis: str, results_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    runs = _collect_runs(results_root, axis)
    if not runs:
        raise RuntimeError(f"No result JSONs found for axis={axis} under {results_root}")

    # Save run-level table.
    run_table = out_root / "run_table.csv"
    with run_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_phase",
                "setting_id",
                "setting_label",
                "seed_id",
                "best_valid_mrr20",
                "test_mrr20",
                "diag_raw_best_valid_file",
                "result_json",
            ],
        )
        writer.writeheader()
        for row in sorted(runs, key=lambda r: (r["setting_label"], r["seed_id"], r["run_phase"])):
            writer.writerow(row)

    # Aggregate simple performance summary.
    by_setting: dict[str, list[float]] = {}
    for row in runs:
        by_setting.setdefault(str(row["setting_label"]), []).append(_to_float(row["test_mrr20"], -1.0))

    perf_summary_path = out_root / "performance_summary.csv"
    with perf_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "n", "test_mrr20_mean", "test_mrr20_max"])
        writer.writeheader()
        for setting, vals in sorted(by_setting.items()):
            valid = [v for v in vals if v >= 0]
            writer.writerow(
                {
                    "setting": setting,
                    "n": len(valid),
                    "test_mrr20_mean": (sum(valid) / len(valid)) if valid else -1.0,
                    "test_mrr20_max": max(valid) if valid else -1.0,
                }
            )

    # Build pair-level scatter rows from diag payload.
    pair_rows: list[dict[str, Any]] = []
    for run in runs:
        diag_raw = str(run.get("diag_raw_best_valid_file") or "").strip()
        if not diag_raw:
            continue
        path = Path(diag_raw)
        if not path.is_file():
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        stage_rows = _extract_stage_pairs(payload)
        for row in stage_rows:
            pair_rows.append(
                {
                    "setting_label": run["setting_label"],
                    "run_phase": run["run_phase"],
                    "seed_id": run["seed_id"],
                    "stage_key": row.get("stage_key", ""),
                    "feature_cosine": _to_float(row.get("feature_cosine"), 0.0),
                    "routing_js": _to_float(row.get("routing_js"), 0.0),
                    "routing_similarity": _to_float(row.get("routing_similarity"), 0.0),
                }
            )

    if pair_rows:
        pair_csv = out_root / "feature_vs_routing_pairs.csv"
        with pair_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "setting_label",
                    "run_phase",
                    "seed_id",
                    "stage_key",
                    "feature_cosine",
                    "routing_js",
                    "routing_similarity",
                ],
            )
            writer.writeheader()
            for row in pair_rows:
                writer.writerow(row)

        # Scatter plot.
        colors = {
            "baseline": "#1f77b4",
            "hidden_only": "#ff7f0e",
            "hidden_plus_feature": "#2ca02c",
        }
        plt.figure(figsize=(9, 6))
        for setting in ["baseline", "hidden_only", "hidden_plus_feature"]:
            chunk = [r for r in pair_rows if r["setting_label"] == setting]
            if not chunk:
                continue
            xs = [r["feature_cosine"] for r in chunk]
            ys = [r["routing_similarity"] for r in chunk]
            plt.scatter(xs, ys, s=8, alpha=0.22, c=colors.get(setting, "#888888"), label=setting)
        plt.xlabel("Feature similarity (cosine)")
        plt.ylabel("Routing similarity (exp(-JS))")
        plt.title("Feature vs Routing Similarity (best-valid diag samples)")
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_root / "feature_vs_routing_scatter.png", dpi=180)
        plt.close()

        # Binned curve.
        bins = [-1.0 + i * 0.1 for i in range(21)]
        plt.figure(figsize=(9, 6))
        for setting in ["baseline", "hidden_only", "hidden_plus_feature"]:
            chunk = [r for r in pair_rows if r["setting_label"] == setting]
            if not chunk:
                continue
            sums = [0.0] * 20
            cnts = [0] * 20
            for row in chunk:
                x = float(row["feature_cosine"])
                y = float(row["routing_similarity"])
                b = min(max(int((x + 1.0) / 0.1), 0), 19)
                sums[b] += y
                cnts[b] += 1
            xs = []
            ys = []
            for i in range(20):
                if cnts[i] <= 0:
                    continue
                xs.append(0.5 * (bins[i] + bins[i + 1]))
                ys.append(sums[i] / cnts[i])
            if xs:
                plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=setting, color=colors.get(setting, "#888888"))
        plt.xlabel("Feature similarity (cosine, binned)")
        plt.ylabel("Mean routing similarity")
        plt.title("Routing Similarity by Feature Similarity Bin")
        plt.grid(alpha=0.2)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_root / "feature_vs_routing_curve.png", dpi=180)
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ablation_2 beauty router-input runs")
    parser.add_argument("--axis", default=DEFAULT_AXIS)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    analyze(
        axis=str(args.axis),
        results_root=Path(args.results_root),
        out_root=Path(args.out_root),
    )
    print(f"[analysis] done -> {Path(args.out_root).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
