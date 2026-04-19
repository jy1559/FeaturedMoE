#!/usr/bin/env python3
"""
Update base_candidates.csv with FeaturedMoE_N3 results from:
  - experiments/run/artifacts/results/f/
  - experiments/run/artifacts/results/results_final_experiment_fmoe/
Ranks by seen-target test metric mean (overall_seen_target then overall fallback),
top 8 per dataset.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
ARTIFACT_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts"

RESULT_DIRS = [
    ARTIFACT_ROOT / "results" / "f",
    ARTIFACT_ROOT / "results" / "results_final_experiment_fmoe",
]

# Fallback: used only for datasets with no entries in RESULT_DIRS
FALLBACK_DIRS = [
    ARTIFACT_ROOT / "results" / "fmoe_n4",
]

CSV_PATH = Path(__file__).resolve().parent / "configs" / "base_candidates.csv"

TARGET_MODEL_CLASS = "FeaturedMoE_N3"
TARGET_MODEL = "featured_moe_n3"
TOP_K = 8

DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "beauty",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
    "amazon_beauty",
]

NINE_METRIC_KEYS = [
    "hit@5", "hit@10", "hit@20",
    "ndcg@5", "ndcg@10", "ndcg@20",
    "mrr@5", "mrr@10", "mrr@20",
]

DIR_LABEL = {
    str(ARTIFACT_ROOT / "results" / "f"): "f",
    str(ARTIFACT_ROOT / "results" / "results_final_experiment_fmoe"): "rfmoe",
    str(ARTIFACT_ROOT / "results" / "fmoe_n4"): "fmoe_n4",
}

DATASET_SLUG = {
    "KuaiRecLargeStrictPosV2_0.2": "kuaireclargestrictposv2_0_2",
    "beauty": "beauty",
    "foursquare": "foursquare",
    "retail_rocket": "retail_rocket",
    "movielens1m": "movielens1m",
    "lastfm0.03": "lastfm0_03",
    "amazon_beauty": "amazon_beauty",
}


def safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def metric_mean(metrics: Optional[Dict[str, Any]]) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    values = [safe_float(metrics.get(k), 0.0) for k in NINE_METRIC_KEYS if k in metrics]
    if values:
        return sum(values) / len(values)
    if "mrr@20" in metrics:
        return safe_float(metrics.get("mrr@20"), 0.0)
    return 0.0


def special_seen_mean(special_payload: Optional[Dict[str, Any]]) -> float:
    if not isinstance(special_payload, dict):
        return 0.0
    for key in ("overall_seen_target", "overall"):
        val = metric_mean(special_payload.get(key) or {})
        if val > 0.0:
            return val
    return 0.0


def result_valid_mean(payload: Dict[str, Any]) -> float:
    return max(
        special_seen_mean(payload.get("best_valid_special_metrics") or {}),
        metric_mean(payload.get("best_valid_result") or {}),
        safe_float(payload.get("best_mrr@20"), 0.0),
    )


def result_test_mean(payload: Dict[str, Any]) -> float:
    return max(
        special_seen_mean(payload.get("test_special_metrics") or {}),
        metric_mean(payload.get("test_result") or {}),
        safe_float(payload.get("test_mrr@20"), 0.0),
    )


def scan_fmoe_files(dirs=None) -> Dict[str, Dict[str, Any]]:
    """Return dict: basename -> {path, payload, dir_label}"""
    if dirs is None:
        dirs = RESULT_DIRS
    found: Dict[str, Dict[str, Any]] = {}
    for result_dir in dirs:
        if not result_dir.exists():
            print(f"[WARN] dir not found: {result_dir}", file=sys.stderr)
            continue
        label = DIR_LABEL.get(str(result_dir), result_dir.name)
        for fpath in sorted(result_dir.glob("*.json")):
            name = fpath.name
            if TARGET_MODEL_CLASS not in name:
                continue
            # skip diag/ special/ normal/ subdirs (only .json in root)
            try:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] failed to read {fpath}: {e}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                continue
            model_cls = payload.get("model", "")
            if model_cls != TARGET_MODEL_CLASS:
                continue
            # prefer results_final_experiment_fmoe over f (richer special metrics)
            if name in found and label != "rfmoe":
                continue
            found[name] = {
                "path": fpath,
                "payload": payload,
                "dir_label": label,
                "dataset": payload.get("dataset", ""),
            }
    return found


def build_source_run_phase(payload: Dict[str, Any]) -> str:
    return str(payload.get("run_phase", "")).strip().upper()


def make_tag(dataset: str, rank: int, dir_label: str) -> str:
    slug = DATASET_SLUG.get(dataset, re.sub(r"[^a-z0-9]+", "_", dataset.lower()).strip("_"))
    return f"{slug}_top{rank:02d}_{dir_label}"


def build_notes(rank: int, dir_label: str, payload: Dict[str, Any], test_mean: float) -> str:
    test_mrr20 = safe_float(payload.get("test_mrr@20"), 0.0)
    orig_model = payload.get("model", TARGET_MODEL_CLASS)
    return (
        f"auto top{rank} by test_mean from {dir_label} "
        f"orig_model={orig_model} "
        f"test_mrr20={test_mrr20:.6f}"
    )


def select_top_k(entries: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    """Sort by test_mean desc, then valid_mean desc. Return top k."""
    entries.sort(key=lambda e: (e["test_mean"], e["valid_mean"]), reverse=True)
    return entries[:k]


def main() -> None:
    all_files = scan_fmoe_files()
    print(f"[INFO] found {len(all_files)} FeaturedMoE_N3 JSON files (primary dirs)", file=sys.stderr)

    # Group by dataset from primary dirs
    by_dataset: Dict[str, List[Dict[str, Any]]] = {ds: [] for ds in DATASETS}
    for name, info in all_files.items():
        ds = info["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        payload = info["payload"]
        v_mean = result_valid_mean(payload)
        t_mean = result_test_mean(payload)
        if t_mean <= 0.0:
            continue
        by_dataset[ds].append({
            "name": name,
            "path": info["path"],
            "dir_label": info["dir_label"],
            "payload": payload,
            "valid_mean": v_mean,
            "test_mean": t_mean,
        })

    for ds in DATASETS:
        entries = by_dataset.get(ds, [])
        print(f"[INFO] {ds}: {len(entries)} candidates (primary)", file=sys.stderr)

    # Scan fallback dirs for datasets with no primary entries
    missing_datasets = [ds for ds in DATASETS if not by_dataset.get(ds)]
    if missing_datasets and FALLBACK_DIRS:
        print(f"[INFO] scanning fallback dirs for: {missing_datasets}", file=sys.stderr)
        fallback_files = scan_fmoe_files(FALLBACK_DIRS)
        for name, info in fallback_files.items():
            ds = info["dataset"]
            if ds not in missing_datasets:
                continue
            if ds not in by_dataset:
                by_dataset[ds] = []
            payload = info["payload"]
            v_mean = result_valid_mean(payload)
            t_mean = result_test_mean(payload)
            if t_mean <= 0.0:
                continue
            by_dataset[ds].append({
                "name": name,
                "path": info["path"],
                "dir_label": info["dir_label"],
                "payload": payload,
                "valid_mean": v_mean,
                "test_mean": t_mean,
            })
        for ds in missing_datasets:
            print(f"[INFO] {ds}: {len(by_dataset.get(ds, []))} candidates (fallback)", file=sys.stderr)

    # Build new CSV rows
    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        entries = by_dataset.get(ds, [])
        if not entries:
            print(f"[INFO] {ds}: no data found, skipping", file=sys.stderr)
            continue
        top = select_top_k(entries, TOP_K)
        for rank, entry in enumerate(top, start=1):
            payload = entry["payload"]
            tag = make_tag(ds, rank, entry["dir_label"])
            notes = build_notes(rank, entry["dir_label"], payload, entry["test_mean"])
            run_phase = build_source_run_phase(payload)
            rows.append({
                "dataset": ds,
                "model": TARGET_MODEL,
                "rank": rank,
                "enabled": "true",
                "tag": tag,
                "result_json": str(entry["path"]),
                "notes": notes,
                "config_rank": rank,
                "seed_count": 1,
                "mean_valid_score": f"{entry['valid_mean']:.12f}",
                "mean_test_score": f"{entry['test_mean']:.12f}",
                "source_run_phase": run_phase,
            })

    # Write CSV
    fieldnames = [
        "dataset", "model", "rank", "enabled", "tag", "result_json",
        "notes", "config_rank", "seed_count", "mean_valid_score", "mean_test_score",
        "source_run_phase",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows to {CSV_PATH}", file=sys.stderr)

    # Print summary
    print("\n=== Updated base_candidates.csv ===")
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            print(f"  {ds}: (no entries)")
            continue
        print(f"  {ds}:")
        for r in ds_rows:
            print(f"    rank={r['rank']} test={float(r['mean_test_score']):.6f} valid={float(r['mean_valid_score']):.6f}  {r['tag']}")


if __name__ == "__main__":
    main()
