#!/usr/bin/env python3
"""Appendix A/B: dataset, protocol, and full ranking result export."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import LOG_ROOT, notebook_data_dir, load_result_payload, load_selected_space_metadata, now_utc, result_has_successful_trials, selected_candidates_from_args, validate_session_fixed_files, write_csv_rows, write_manifest


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_dataset_stats(dataset: str) -> dict[str, Any]:
    dataset_dir = Path("/workspace/FeaturedMoE/Datasets/processed/feature_added_v4") / dataset
    inter_paths = {
        "train": dataset_dir / f"{dataset}.train.inter",
        "valid": dataset_dir / f"{dataset}.valid.inter",
        "test": dataset_dir / f"{dataset}.test.inter",
    }
    rows = 0
    sessions: set[str] = set()
    items: set[str] = set()
    total_len = 0
    for path in inter_paths.values():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            header = None
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                if idx == 0:
                    header = {name.split(":")[0]: pos for pos, name in enumerate(cols)}
                    continue
                if header is None:
                    continue
                rows += 1
                session_id = cols[header.get("session_id", 0)]
                item_id = cols[header.get("item_id", 0)]
                sessions.add(session_id)
                items.add(item_id)
        session_lengths: dict[str, int] = {}
        with open(path, "r", encoding="utf-8") as f:
            header = None
            for idx, line in enumerate(f):
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                if idx == 0:
                    header = {name.split(":")[0]: pos for pos, name in enumerate(cols)}
                    continue
                if header is None:
                    continue
                session_id = cols[header.get("session_id", 0)]
                session_lengths[session_id] = session_lengths.get(session_id, 0) + 1
        total_len += sum(session_lengths.values())
    avg_len = (float(total_len) / float(max(len(sessions), 1))) if sessions else 0.0
    return {
        "dataset": dataset,
        "interactions": rows,
        "sessions": len(sessions),
        "items": len(items),
        "avg_session_len": round(avg_len, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export appendix dataset/full-result tables.")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--top-k-configs", type=int, default=1)
    parser.add_argument("--base-csv", default="/workspace/FeaturedMoE/experiments/run/final_experiment/ablation/configs/base_candidates.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-tag", default="")
    args, _unknown = parser.parse_known_args()

    candidates = selected_candidates_from_args(args)
    datasets = sorted({candidate.dataset for candidate in candidates})
    for dataset in datasets:
        validate_session_fixed_files(dataset)

    manifest_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    dataset_rows = [_read_dataset_stats(dataset) for dataset in datasets]
    for candidate in candidates:
        payload = load_result_payload(candidate.result_json)
        if not result_has_successful_trials(payload):
            continue
        valid = payload.get("best_valid_result") or {}
        test = payload.get("test_result") or {}
        manifest_rows.append(
            {
                "dataset": candidate.dataset,
                "model": candidate.model,
                "base_rank": candidate.rank,
                "base_tag": candidate.tag,
                "result_json": str(candidate.result_json),
                "checkpoint_file": candidate.checkpoint_file,
            }
        )
        for split_name, metrics in (("valid", valid), ("test", test)):
            for metric_name, value in metrics.items():
                result_rows.append(
                    {
                        "dataset": candidate.dataset,
                        "model": candidate.model,
                        "base_rank": candidate.rank,
                        "base_tag": candidate.tag,
                        "split": split_name,
                        "metric": metric_name,
                        "value": _safe_float(value),
                        "result_json": str(candidate.result_json),
                    }
                )

    meta = load_selected_space_metadata()
    appendix_root = LOG_ROOT / "full_results"
    appendix_root.mkdir(parents=True, exist_ok=True)
    write_csv_rows(appendix_root / "selected_run_index.csv", manifest_rows)
    write_csv_rows(appendix_root / "dataset_stats.csv", dataset_rows)
    write_csv_rows(appendix_root / "full_results_long.csv", result_rows)
    if meta["selected_configs_csv"].exists():
        import csv

        with open(meta["selected_configs_csv"], "r", newline="", encoding="utf-8") as f:
            selected_rows = list(csv.DictReader(f))
        write_csv_rows(appendix_root / "selected_configs.csv", selected_rows)
    if meta["tuning_space_csv"].exists():
        import csv

        with open(meta["tuning_space_csv"], "r", newline="", encoding="utf-8") as f:
            tuning_rows = list(csv.DictReader(f))
        write_csv_rows(appendix_root / "tuning_space.csv", tuning_rows)

    notebook_dir = notebook_data_dir()
    write_csv_rows(notebook_dir / "appendix_dataset_stats.csv", dataset_rows)
    write_csv_rows(notebook_dir / "appendix_full_results_long.csv", result_rows)
    write_csv_rows(notebook_dir / "appendix_selected_runs.csv", manifest_rows)

    write_manifest(
        "full_results",
        [
            {
                "question": "full_results",
                "generated_at": now_utc(),
                "dataset_count": len(dataset_rows),
                "run_count": len(manifest_rows),
                "dry_run": bool(args.dry_run),
            }
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
