#!/usr/bin/env python3
"""AI502 transfer 결과를 baseline 대비 gain 중심으로 요약한다."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = THIS_DIR / "artifacts"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
RESULTS_DIR = ARTIFACT_DIR / "hyperopt_results"
ANALYSIS_DIR = ARTIFACT_DIR / "analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def iter_json_results(root: Path) -> list[dict[str, Any]]:
    out = []
    root = root.resolve()
    result_root = root / "ai502_transfer" if (root / "ai502_transfer").exists() else root
    for path in sorted(result_root.glob("*.json")):
        if path.name.endswith("_special_metrics.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "run_phase" not in payload:
            continue
        payload["_result_path"] = str(path)
        out.append(payload)
    return out


def latest_manifest_rows() -> list[dict[str, Any]]:
    by_job_key: dict[str, dict[str, Any]] = {}
    for path in sorted(MANIFEST_DIR.glob("*.csv")):
        for row in read_csv(path):
            parsed = dict(row)
            try:
                parsed["lr_values"] = json.loads(parsed.get("lr_values") or "[]")
            except Exception:
                parsed["lr_values"] = []
            by_job_key[str(parsed.get("job_key", ""))] = parsed
    return list(by_job_key.values())


def metric(payload: dict[str, Any], name: str = "test_mrr@20") -> float:
    value = payload.get(name)
    if value is None and isinstance(payload.get("test_result"), dict):
        value = payload["test_result"].get("mrr@20")
    try:
        return float(value)
    except Exception:
        return 0.0


def build_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("run_phase") or ""): row for row in results if row.get("run_phase")}


def aggregate(rows: list[dict[str, Any]], value_key: str, group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row.get(value_key, 0.0))
        except Exception:
            continue
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(value)
    out = []
    for key, values in sorted(groups.items()):
        item = {group_keys[i]: key[i] for i in range(len(group_keys))}
        item.update(n=len(values), mean=mean(values), min=min(values), max=max(values))
        out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="AI502 transfer 결과 요약")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(ANALYSIS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_rows = latest_manifest_rows()
    result_index = build_index(iter_json_results(Path(args.results_dir)))

    native_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    enriched: list[dict[str, Any]] = []
    for row in manifest_rows:
        result = result_index.get(row.get("run_phase", ""))
        if not result:
            continue
        item = dict(row)
        item["result_path"] = result.get("_result_path", "")
        item["test_mrr20"] = metric(result)
        item["valid_mrr20"] = float(result.get("best_mrr@20") or 0.0)
        transfer_report = result.get("transfer_report") or {}
        freeze_report = result.get("freeze_report") or {}
        init_delta = transfer_report.get("init_delta_from_target") or {}
        train_delta = transfer_report.get("train_delta_from_init") or {}
        item["transfer_report"] = json.dumps(transfer_report, ensure_ascii=False, sort_keys=True)
        item["freeze_report"] = json.dumps(freeze_report, ensure_ascii=False, sort_keys=True)
        item["loaded_tensors"] = transfer_report.get("loaded_tensors", "")
        item["init_changed_tensors"] = init_delta.get("changed_tensors", "")
        item["train_changed_tensors"] = train_delta.get("changed_tensors", "")
        item["prefix_resolution"] = transfer_report.get("prefix_resolution", "")
        item["trainable_loaded_tensors"] = transfer_report.get("trainable_loaded_tensors", "")
        enriched.append(item)
        if row.get("phase") == "native":
            native_by_key[(row.get("target_dataset", row.get("dataset", "")), row.get("hparam_id", ""), row.get("seed", ""))] = item

    native_rows = [row for row in enriched if row.get("phase") == "native"]
    transfer_rows = [row for row in enriched if row.get("phase") in {"init", "freeze", "multihop", "multihop_bridge"}]
    for row in transfer_rows:
        base = native_by_key.get((row.get("target_dataset", row.get("dataset", "")), row.get("hparam_id", ""), row.get("seed", "")))
        row["baseline_test_mrr20"] = base.get("test_mrr20", "") if base else ""
        try:
            row["gain_vs_native"] = float(row["test_mrr20"]) - float(row["baseline_test_mrr20"])
        except Exception:
            row["gain_vs_native"] = ""

    write_csv(
        out_dir / "native_scratch_mean.csv",
        aggregate(native_rows, "test_mrr20", ["target_dataset", "hparam_id"]),
        ["target_dataset", "hparam_id", "n", "mean", "min", "max"],
    )
    write_csv(
        out_dir / "pair_transfer_gain.csv",
        aggregate([r for r in transfer_rows if r.get("phase") == "init" and r.get("gain_vs_native") != ""], "gain_vs_native", ["pair_id", "target_dataset", "transfer_mode"]),
        ["pair_id", "target_dataset", "transfer_mode", "n", "mean", "min", "max"],
    )

    gain_rows = read_csv(out_dir / "pair_transfer_gain.csv")
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gain_rows:
        by_target[row["target_dataset"]].append(row)
    top_rows = []
    for target, items in by_target.items():
        ranked = sorted(items, key=lambda r: float(r.get("mean") or 0.0), reverse=True)
        for rank, row in enumerate(ranked[:2], start=1):
            top_rows.append({"target_dataset": target, "rank": rank, **row})
    write_csv(
        out_dir / "target_top2_modes.csv",
        top_rows,
        ["target_dataset", "rank", "pair_id", "transfer_mode", "n", "mean", "min", "max"],
    )
    write_csv(
        out_dir / "freeze_gain_loss.csv",
        aggregate([r for r in transfer_rows if r.get("phase") == "freeze" and r.get("gain_vs_native") != ""], "gain_vs_native", ["pair_id", "target_dataset", "transfer_mode", "freeze_policy"]),
        ["pair_id", "target_dataset", "transfer_mode", "freeze_policy", "n", "mean", "min", "max"],
    )
    write_csv(
        out_dir / "multihop_direct_vs_sequential.csv",
        aggregate([r for r in transfer_rows if r.get("phase") == "multihop" and r.get("gain_vs_native") != ""], "gain_vs_native", ["triplet_id", "comparison_role", "target_dataset", "transfer_mode"]),
        ["triplet_id", "comparison_role", "target_dataset", "transfer_mode", "n", "mean", "min", "max"],
    )
    write_csv(
        out_dir / "transfer_qc.csv",
        transfer_rows,
        [
            "phase",
            "run_phase",
            "pair_id",
            "triplet_id",
            "target_dataset",
            "hparam_id",
            "seed",
            "transfer_mode",
            "freeze_policy",
            "loaded_tensors",
            "init_changed_tensors",
            "train_changed_tensors",
            "trainable_loaded_tensors",
            "prefix_resolution",
            "test_mrr20",
            "baseline_test_mrr20",
            "gain_vs_native",
            "result_path",
        ],
    )

    print(f"[summary] native={len(native_rows)} transfer={len(transfer_rows)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
