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
    ARTIFACT_ROOT / "results" / "23090_results",
]

# Fallback: used only for datasets with no entries in RESULT_DIRS
FALLBACK_DIRS = [
    ARTIFACT_ROOT / "results" / "fmoe_n4",
]

CSV_PATH = Path(__file__).resolve().parent / "configs" / "base_candidates.csv"
CSV_TOPK_PATH = Path(__file__).resolve().parent / "configs" / "base_candidates_topk.csv"
CSV_DENSE_PATH = Path(__file__).resolve().parent / "configs" / "base_candidates_dense.csv"

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


def _glob_json_files(root: Path) -> List[Path]:
    # Legacy dirs keep JSON files in root; newer runs may store under nested folders.
    root_files = list(root.glob("*.json"))
    nested_files = list(root.rglob("*.json"))
    if not root_files:
        return nested_files
    # Keep root-first order but include nested-only files too.
    known = {str(p) for p in root_files}
    return root_files + [p for p in nested_files if str(p) not in known]


def scan_fmoe_files(dirs=None) -> Dict[str, Dict[str, Any]]:
    """Return dict: stable_key -> {path, payload, dir_label}"""
    if dirs is None:
        dirs = RESULT_DIRS
    found: Dict[str, Dict[str, Any]] = {}
    for result_dir in dirs:
        if not result_dir.exists():
            print(f"[WARN] dir not found: {result_dir}", file=sys.stderr)
            continue
        label = DIR_LABEL.get(str(result_dir), result_dir.name)
        for fpath in sorted(_glob_json_files(result_dir)):
            name = fpath.name
            if TARGET_MODEL_CLASS not in name:
                continue
            lower_name = name.lower()
            if lower_name.endswith("_special_metrics.json") or "special_metrics" in lower_name:
                continue
            ptext = str(fpath).lower()
            if "/special/" in ptext or "/diag/" in ptext:
                continue
            try:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[WARN] failed to read {fpath}: {e}", file=sys.stderr)
                continue
            if not isinstance(payload, dict):
                continue
            # Keep only full run payloads (not analysis/manifest JSON files).
            trials = payload.get("trials")
            if not isinstance(trials, list) or not trials:
                continue
            model_cls = payload.get("model", "")
            if model_cls != TARGET_MODEL_CLASS:
                continue
            run_phase = str(payload.get("run_phase", "")).strip()
            stable_key = f"{payload.get('dataset','')}::{run_phase}::{name}"
            # Prefer results_final_experiment_fmoe over f when key collides.
            if stable_key in found and label != "rfmoe":
                continue
            found[stable_key] = {
                "path": fpath,
                "payload": payload,
                "dir_label": label,
                "dataset": payload.get("dataset", ""),
            }
    return found


def _all_dense_plain(stage_mode: Any) -> bool:
    if not isinstance(stage_mode, dict):
        return False
    vals = [str(v).strip().lower() for v in stage_mode.values()]
    return bool(vals) and all(v == "dense_plain" for v in vals)


def classify_routing_mode(payload: Dict[str, Any]) -> str:
    run_phase = str(payload.get("run_phase", "")).upper()
    if "TOPK" in run_phase:
        return "topk"

    for obj in (payload.get("fixed_search") or {}, payload.get("context_fixed") or {}):
        if not isinstance(obj, dict):
            continue
        moe_top_k = obj.get("moe_top_k")
        group_top_k = obj.get("group_top_k")
        expert_top_k = obj.get("expert_top_k")
        topk_scope_mode = str(obj.get("topk_scope_mode", "")).strip().lower()
        if safe_float(moe_top_k, 0.0) > 0 or safe_float(group_top_k, 0.0) > 0 or safe_float(expert_top_k, 0.0) > 0:
            return "topk"
        if _all_dense_plain(obj.get("stage_compute_mode")):
            return "dense"
        if topk_scope_mode in {"global_flat", "per_group"} and safe_float(moe_top_k, 0.0) <= 0:
            return "dense"

    # Legacy runs with no explicit top-k metadata are treated as dense/default.
    return "dense"


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
            "routing_mode": classify_routing_mode(payload),
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
                "routing_mode": classify_routing_mode(payload),
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
        entries.sort(key=lambda e: (e["test_mean"], e["valid_mean"]), reverse=True)
        total_top = entries[:TOP_K]
        topk_entries = [e for e in entries if e.get("routing_mode") == "topk"]
        dense_entries = [e for e in entries if e.get("routing_mode") == "dense"]
        topk_rank_map = {id(e): i for i, e in enumerate(topk_entries, start=1)}
        dense_rank_map = {id(e): i for i, e in enumerate(dense_entries, start=1)}

        for rank_total, entry in enumerate(total_top, start=1):
            payload = entry["payload"]
            tag = make_tag(ds, rank_total, entry["dir_label"])
            notes = build_notes(rank_total, entry["dir_label"], payload, entry["test_mean"])
            run_phase = build_source_run_phase(payload)
            rank_topk = topk_rank_map.get(id(entry), "")
            rank_dense = dense_rank_map.get(id(entry), "")
            rows.append({
                "dataset": ds,
                "model": TARGET_MODEL,
                "rank": rank_total,
                "enabled": "true",
                "tag": tag,
                "result_json": str(entry["path"]),
                "notes": notes,
                "config_rank": rank_total,
                "seed_count": 1,
                "mean_valid_score": f"{entry['valid_mean']:.12f}",
                "mean_test_score": f"{entry['test_mean']:.12f}",
                "routing_mode": entry.get("routing_mode", "dense"),
                "rank_total": rank_total,
                "rank_topk": rank_topk,
                "rank_dense": rank_dense,
                "use_for_default": "true",
                "use_for_topk_ablation": "true" if rank_topk else "false",
                "source_run_phase": run_phase,
            })

    # Write CSV
    fieldnames = [
        "dataset", "model", "rank", "enabled", "tag", "result_json",
        "notes", "config_rank", "seed_count", "mean_valid_score", "mean_test_score",
        "routing_mode", "rank_total", "rank_topk", "rank_dense", "use_for_default", "use_for_topk_ablation",
        "source_run_phase",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Companion CSV for top-k-only ablation: rank is remapped to rank_topk.
    topk_rows = []
    for row in rows:
        rk = str(row.get("rank_topk", "")).strip()
        if not rk:
            continue
        r2 = dict(row)
        r2["rank"] = int(rk)
        r2["config_rank"] = int(rk)
        topk_rows.append(r2)
    topk_rows.sort(key=lambda r: (str(r["dataset"]), int(r["rank"])))
    with CSV_TOPK_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topk_rows)

    # Companion CSV for dense-only reference: rank is remapped to rank_dense.
    dense_rows = []
    for row in rows:
        rk = str(row.get("rank_dense", "")).strip()
        if not rk:
            continue
        r2 = dict(row)
        r2["rank"] = int(rk)
        r2["config_rank"] = int(rk)
        dense_rows.append(r2)
    dense_rows.sort(key=lambda r: (str(r["dataset"]), int(r["rank"])))
    with CSV_DENSE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dense_rows)

    print(f"[OK] wrote {len(rows)} rows to {CSV_PATH}", file=sys.stderr)
    print(f"[OK] wrote {len(topk_rows)} rows to {CSV_TOPK_PATH}", file=sys.stderr)
    print(f"[OK] wrote {len(dense_rows)} rows to {CSV_DENSE_PATH}", file=sys.stderr)

    # Print summary
    print("\n=== Updated base_candidates.csv ===")
    for ds in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        if not ds_rows:
            print(f"  {ds}: (no entries)")
            continue
        print(f"  {ds}:")
        for r in ds_rows:
            print(
                f"    rank={r['rank']} mode={r['routing_mode']}"
                f" topk_rank={r['rank_topk'] or '-'} dense_rank={r['rank_dense'] or '-'}"
                f" test={float(r['mean_test_score']):.6f} valid={float(r['mean_valid_score']):.6f}  {r['tag']}"
            )


if __name__ == "__main__":
    main()
