#!/usr/bin/env python3
"""Refresh A05 appendix CSVs from q5 multi-seed outputs.

Selects the best base candidate per dataset by mean overall_seen_target MRR@20
across completed train seeds, then exports all selected-seed intervention rows and
case-routing rows for the A05 notebook/figure pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
Q5_LOG_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts" / "logs" / "final_experiment_ablation" / "q5"
DATA_DIR = ROOT / "data"
INTERVENTION_CSV = DATA_DIR / "appendix_intervention_summary.csv"
CASE_PROFILE_CSV = DATA_DIR / "appendix_case_routing_profile.csv"
REPORT_JSON = DATA_DIR / "appendix_a05_q5_selection.json"

CASE_KEEP = {"original", "memory_plus", "tempo_plus", "focus_plus"}
INTERVENTION_MAP = {
    "full": ("full", "Full cues"),
    "feature_zero_all": ("feature_zero_all", "Zero all cues"),
    "feature_zero_tempo": ("zero_tempo", "Zero Tempo"),
    "feature_zero_focus": ("zero_focus", "Zero Focus"),
    "feature_zero_memory": ("zero_memory", "Zero Memory"),
    "feature_zero_exposure": ("zero_exposure", "Zero Exposure"),
    "feature_shuffle_tempo": ("shuffle_tempo", "Shuffle Tempo"),
    "feature_shuffle_focus": ("shuffle_focus", "Shuffle Focus"),
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    if preferred_fields:
        for field in preferred_fields:
            if field not in fieldnames:
                fieldnames.append(field)
    for row in rows:
        for field in row.keys():
            if field not in fieldnames:
                fieldnames.append(field)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_intervention(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    return INTERVENTION_MAP.get(text, (text, text))


def _result_seen_mrr20(result_path: Path) -> float:
    payload = _read_json(result_path)
    block = ((payload.get("test_special_metrics") or {}).get("overall_seen_target") or {})
    if isinstance(block, dict) and "mrr@20" in block:
        return _safe_float(block.get("mrr@20"))
    return _safe_float(payload.get("test_seen_target_mrr20"))


def _select_best_bases(summary_rows: list[dict[str, str]], datasets: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    result_cache: dict[str, float] = {}
    for row in summary_rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        dataset = str(row.get("dataset", "")).strip()
        if dataset not in datasets:
            continue
        result_path = Path(str(row.get("result_path", "")).strip())
        if not result_path.exists():
            continue
        base_tag = str(row.get("base_tag", "")).strip()
        if not base_tag:
            continue
        result_key = str(result_path)
        if result_key not in result_cache:
            result_cache[result_key] = _result_seen_mrr20(result_path)
        group_key = f"{dataset}::{base_tag}"
        slot = grouped.setdefault(
            group_key,
            {
                "dataset": dataset,
                "base_tag": base_tag,
                "base_rank": _safe_float(row.get("base_rank"), 999.0),
                "result_paths": [],
                "scores": [],
                "seed_ids": [],
            },
        )
        slot["result_paths"].append(result_key)
        slot["scores"].append(result_cache[result_key])
        slot["seed_ids"].append(str(row.get("seed_id", "")).strip())

    best_by_dataset: dict[str, str] = {}
    report: dict[str, Any] = {"selected": {}, "candidates": defaultdict(list)}
    for payload in grouped.values():
        dataset = str(payload["dataset"])
        mean_score = sum(payload["scores"]) / max(len(payload["scores"]), 1)
        candidate_info = {
            "base_tag": payload["base_tag"],
            "base_rank": int(payload["base_rank"]),
            "mean_test_seen_mrr20": mean_score,
            "seed_count": len(payload["scores"]),
            "seed_ids": payload["seed_ids"],
        }
        report["candidates"][dataset].append(candidate_info)
        current_tag = best_by_dataset.get(dataset)
        if current_tag is None:
            best_by_dataset[dataset] = str(payload["base_tag"])
            continue
        current = next(item for item in grouped.values() if item["dataset"] == dataset and item["base_tag"] == current_tag)
        current_mean = sum(current["scores"]) / max(len(current["scores"]), 1)
        current_rank = float(current["base_rank"])
        new_rank = float(payload["base_rank"])
        if mean_score > current_mean or (abs(mean_score - current_mean) <= 1e-12 and new_rank < current_rank):
            best_by_dataset[dataset] = str(payload["base_tag"])

    for dataset, base_tag in best_by_dataset.items():
        chosen = next(item for item in grouped.values() if item["dataset"] == dataset and item["base_tag"] == base_tag)
        report["selected"][dataset] = {
            "base_tag": base_tag,
            "base_rank": int(chosen["base_rank"]),
            "mean_test_seen_mrr20": sum(chosen["scores"]) / max(len(chosen["scores"]), 1),
            "seed_count": len(chosen["scores"]),
            "seed_ids": chosen["seed_ids"],
            "result_paths": chosen["result_paths"],
        }
    return best_by_dataset, report


def _selected_result_paths(summary_rows: list[dict[str, str]], datasets: set[str], best_by_dataset: dict[str, str]) -> set[str]:
    selected: set[str] = set()
    for row in summary_rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        dataset = str(row.get("dataset", "")).strip()
        if dataset not in datasets:
            continue
        if str(row.get("base_tag", "")).strip() != best_by_dataset.get(dataset):
            continue
        result_path = str(row.get("result_path", "")).strip()
        if result_path:
            selected.add(result_path)
    return selected


def _collect_interventions(index_rows: list[dict[str, str]], selected_results: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in index_rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        if str(row.get("result_path", "")).strip() not in selected_results:
            continue
        manifest_path = Path(str(row.get("intervention_manifest", "")).strip())
        for manifest_row in _read_csv(manifest_path):
            if str(manifest_row.get("status", "")).strip().lower() != "ok":
                continue
            intervention_name, intervention_label = _normalize_intervention(str(manifest_row.get("intervention", "")))
            payload = _read_json(Path(str(manifest_row.get("result_file", "")).strip()))
            seen_block = ((payload.get("test_special_metrics") or {}).get("overall_seen_target") or {})
            merged = dict(row)
            merged.update(manifest_row)
            merged["intervention"] = intervention_name
            merged["intervention_label"] = intervention_label
            merged["test_mrr20"] = _safe_float((payload.get("test_result") or {}).get("mrr@20", manifest_row.get("test_mrr20")))
            merged["test_seen_mrr20"] = _safe_float(seen_block.get("mrr@20", manifest_row.get("test_seen_mrr20")))
            out.append(merged)
    return out


def _collect_case_profiles(index_rows: list[dict[str, str]], selected_results: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in index_rows:
        if str(row.get("status", "")).strip().lower() != "ok":
            continue
        if str(row.get("result_path", "")).strip() not in selected_results:
            continue
        export_dir = Path(str(row.get("case_eval_export_dir", "")).strip())
        for table_row in _read_csv(export_dir / "case_eval_routing_profile.csv"):
            if str(table_row.get("eval_split", "")).strip().lower() != "test":
                continue
            group = str(table_row.get("group", "")).strip()
            if group not in CASE_KEEP:
                continue
            merged = dict(row)
            merged.update(table_row)
            out.append(merged)
    return out


def _merge_dataset_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, Any]], datasets: set[str]) -> list[dict[str, Any]]:
    kept = [row for row in existing_rows if str(row.get("dataset", "")).strip() not in datasets]
    return kept + new_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Update A05 appendix data from q5 outputs.")
    parser.add_argument("--datasets", default="KuaiRecLargeStrictPosV2_0.2,foursquare")
    args = parser.parse_args()

    datasets = {token.strip() for token in str(args.datasets).split(",") if token.strip()}
    summary_rows = _read_csv(Q5_LOG_ROOT / "summary.csv")
    intervention_index_rows = _read_csv(Q5_LOG_ROOT / "q5_intervention_index.csv")
    case_index_rows = _read_csv(Q5_LOG_ROOT / "q5_case_eval_index.csv")
    if not summary_rows:
        raise FileNotFoundError(f"Missing q5 summary at {Q5_LOG_ROOT / 'summary.csv'}")
    if not intervention_index_rows:
        raise FileNotFoundError(f"Missing q5 intervention index at {Q5_LOG_ROOT / 'q5_intervention_index.csv'}")
    if not case_index_rows:
        raise FileNotFoundError(f"Missing q5 case-eval index at {Q5_LOG_ROOT / 'q5_case_eval_index.csv'}")

    best_by_dataset, report = _select_best_bases(summary_rows, datasets)
    missing = datasets - set(best_by_dataset)
    if missing:
        raise RuntimeError(f"No completed q5 runs found for datasets: {sorted(missing)}")

    selected_results = _selected_result_paths(summary_rows, datasets, best_by_dataset)
    new_interventions = _collect_interventions(intervention_index_rows, selected_results)
    new_case_profiles = _collect_case_profiles(case_index_rows, selected_results)
    if not new_interventions:
        raise RuntimeError("No selected intervention rows were found in q5 outputs.")
    if not new_case_profiles:
        raise RuntimeError("No selected case-profile rows were found in q5 outputs.")

    merged_interventions = _merge_dataset_rows(_read_csv(INTERVENTION_CSV), new_interventions, datasets)
    merged_case_profiles = _merge_dataset_rows(_read_csv(CASE_PROFILE_CSV), new_case_profiles, datasets)

    _write_csv(
        INTERVENTION_CSV,
        merged_interventions,
        preferred_fields=[
            "question",
            "dataset",
            "setting_key",
            "setting_label",
            "base_rank",
            "base_tag",
            "seed_id",
            "result_path",
            "checkpoint_file",
            "intervention_manifest",
            "status",
            "error",
            "timestamp_utc",
            "source_result_json",
            "model",
            "intervention",
            "intervention_label",
            "intervention_group",
            "target_family",
            "logging_dir",
            "result_file",
            "special_metrics_file",
            "router_diag_file",
            "best_valid_mrr20",
            "test_mrr20",
            "best_valid_seen_mrr20",
            "test_seen_mrr20",
        ],
    )
    _write_csv(
        CASE_PROFILE_CSV,
        merged_case_profiles,
        preferred_fields=[
            "question",
            "dataset",
            "setting_key",
            "setting_label",
            "base_rank",
            "base_tag",
            "seed_id",
            "selection_rule",
            "result_path",
            "checkpoint_file",
            "case_eval_manifest",
            "case_eval_export_dir",
            "status",
            "error",
            "model",
            "eval_split",
            "scope",
            "group",
            "stage_name",
            "routed_family",
            "usage_share",
        ],
    )

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    for dataset, selected in sorted(report["selected"].items()):
        print(
            f"[selected] dataset={dataset} base_tag={selected['base_tag']} "
            f"base_rank={selected['base_rank']} mean_test_seen_mrr20={selected['mean_test_seen_mrr20']:.6f} "
            f"seeds={selected['seed_ids']}"
        )
    print(f"[updated] {INTERVENTION_CSV}")
    print(f"[updated] {CASE_PROFILE_CSV}")
    print(f"[report] {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())