#!/usr/bin/env python3
"""Export appendix outputs into notebook-friendly CSV/JSON bundles."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DATA_ROOT, LOG_ROOT, read_csv_rows, write_csv_rows, dump_json, notebook_data_dir, now_utc


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _special_block(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get(key) or {}
    return block if isinstance(block, dict) else {}


def _summary_to_result_row(summary_row: dict[str, str]) -> dict[str, Any] | None:
    result_path = str(summary_row.get("result_path", "")).strip()
    if not result_path or not Path(result_path).exists():
        return None
    payload = _load_json(result_path)
    valid = payload.get("best_valid_result") or {}
    test = payload.get("test_result") or {}
    valid_seen = _special_block(payload.get("best_valid_special_metrics") or {}, "overall_seen_target")
    test_seen = _special_block(payload.get("test_special_metrics") or {}, "overall_seen_target")
    row = dict(summary_row)
    row.update(
        {
            "best_valid_mrr20": _safe_float(valid.get("mrr@20")),
            "test_mrr20": _safe_float(test.get("mrr@20")),
            "best_valid_ndcg20": _safe_float(valid.get("ndcg@20")),
            "test_ndcg20": _safe_float(test.get("ndcg@20")),
            "best_valid_hit10": _safe_float(valid.get("hit@10")),
            "test_hit10": _safe_float(test.get("hit@10")),
            "best_valid_seen_mrr20": _safe_float(valid_seen.get("mrr@20", payload.get("valid_seen_target_mrr20"))),
            "test_seen_mrr20": _safe_float(test_seen.get("mrr@20", payload.get("test_seen_target_mrr20"))),
        }
    )
    return row


def _collect_summary_results(question: str, dataset_filter: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in read_csv_rows(LOG_ROOT / question / "summary.csv"):
        if str(row.get("status", "")).lower() != "ok":
            continue
        if dataset_filter and str(row.get("dataset", "")) not in dataset_filter:
            continue
        result_row = _summary_to_result_row(row)
        if result_row is not None:
            result_row["question"] = question
            out.append(result_row)
    return out


def _router_diag_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in result_rows:
        result_path = str(row.get("result_path", "")).strip()
        if not result_path:
            continue
        payload = _load_json(result_path)
        diag_meta = str(payload.get("diag_meta_file", "") or "").strip()
        if not diag_meta or not Path(diag_meta).exists():
            continue
        diag = _load_json(diag_meta)
        stage_metrics = ((diag.get("test") or {}).get("stage_metrics") or {})
        for stage_key, stage_payload in stage_metrics.items():
            group_routing = (stage_payload or {}).get("group_routing") or {}
            out.append(
                {
                    "dataset": row.get("dataset", ""),
                    "question": row.get("question", ""),
                    "setting_key": row.get("setting_key", ""),
                    "setting_label": row.get("setting_label", ""),
                    "variant_label": row.get("variant_label", ""),
                    "base_rank": row.get("base_rank", ""),
                    "seed_id": row.get("seed_id", ""),
                    "stage_key": stage_key,
                    "group_entropy_mean": _safe_float(group_routing.get("group_entropy_mean")),
                    "group_n_eff": _safe_float(group_routing.get("group_n_eff")),
                    "group_top1_max_frac": _safe_float(group_routing.get("group_top1_max_frac")),
                }
            )
    return out


def _concat_case_tables(index_csv: Path, dataset_filter: set[str], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_row in read_csv_rows(index_csv):
        if str(index_row.get("status", "")).lower() != "ok":
            continue
        if dataset_filter and str(index_row.get("dataset", "")) not in dataset_filter:
            continue
        export_dir = str(index_row.get("case_eval_export_dir", "")).strip()
        if not export_dir:
            continue
        for row in read_csv_rows(Path(export_dir) / filename):
            merged = dict(index_row)
            merged.update(row)
            rows.append(merged)
    return rows


def _intervention_rows(index_csv: Path, dataset_filter: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index_row in read_csv_rows(index_csv):
        if str(index_row.get("status", "")).lower() != "ok":
            continue
        if dataset_filter and str(index_row.get("dataset", "")) not in dataset_filter:
            continue
        manifest = str(index_row.get("intervention_manifest", "")).strip()
        if not manifest:
            continue
        for row in read_csv_rows(Path(manifest)):
            if str(row.get("status", "")).lower() != "ok":
                continue
            payload = _load_json(str(row.get("result_file", "")).strip()) if str(row.get("result_file", "")).strip() else {}
            test_seen = _special_block(payload.get("test_special_metrics") or {}, "overall_seen_target")
            out.append(
                {
                    **index_row,
                    **row,
                    "test_mrr20": _safe_float((payload.get("test_result") or {}).get("mrr@20", row.get("test_mrr20"))),
                    "test_seen_mrr20": _safe_float(test_seen.get("mrr@20", row.get("test_seen_mrr20"))),
                }
            )
    return out


def _special_metric_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in result_rows:
        payload = _load_json(str(row.get("result_path", "")))
        for split_name, block in (("valid", payload.get("best_valid_special_metrics") or {}), ("test", payload.get("test_special_metrics") or {})):
            for metric_block_name, metric_block in block.items():
                if not isinstance(metric_block, dict):
                    continue
                if metric_block_name.startswith("overall"):
                    continue
                for metric_name, value in metric_block.items():
                    out.append(
                        {
                            "dataset": row.get("dataset", ""),
                            "question": row.get("question", ""),
                            "setting_key": row.get("setting_key", ""),
                            "split": split_name,
                            "special_block": metric_block_name,
                            "metric": metric_name,
                            "value": _safe_float(value),
                        }
                    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export appendix notebook bundles.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT))
    parser.add_argument("--datasets", default="")
    args, _unknown = parser.parse_known_args()

    dataset_filter = {token.strip() for token in str(args.datasets).split(",") if token.strip()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_data_dir()

    full_results = read_csv_rows(LOG_ROOT / "full_results" / "full_results_long.csv")
    dataset_stats = read_csv_rows(LOG_ROOT / "full_results" / "dataset_stats.csv")
    structural_rows = _collect_summary_results("structural", dataset_filter)
    sparse_rows = _collect_summary_results("sparse", dataset_filter)
    objective_rows = _collect_summary_results("objective", dataset_filter)
    cost_rows = read_csv_rows(LOG_ROOT / "cost" / "summary.csv")
    behavior_rows = _collect_summary_results("behavior_slices", dataset_filter)
    cases_rows = _collect_summary_results("cases", dataset_filter)

    sparse_diag_rows = _router_diag_rows(sparse_rows)
    structural_diag_rows = _router_diag_rows(structural_rows)
    behavior_case_quality = _concat_case_tables(LOG_ROOT / "behavior_slices" / "behavior_slices_case_eval_index.csv", dataset_filter, "case_quality_summary.csv")
    behavior_case_profile = _concat_case_tables(LOG_ROOT / "behavior_slices" / "behavior_slices_case_eval_index.csv", dataset_filter, "case_feature_profiles.csv")
    special_bins_case = _concat_case_tables(LOG_ROOT / "special_bins" / "special_bins_case_eval_index.csv", dataset_filter, "case_quality_summary.csv")
    diagnostic_case_profile = _concat_case_tables(LOG_ROOT / "diagnostics" / "diagnostics_case_eval_index.csv", dataset_filter, "case_eval_routing_profile.csv")
    qualitative_cases = _concat_case_tables(LOG_ROOT / "cases" / "cases_case_eval_index.csv", dataset_filter, "case_eval_routing_profile.csv")
    intervention_rows = _intervention_rows(LOG_ROOT / "cases" / "cases_intervention_index.csv", dataset_filter)
    objective_special = _special_metric_rows(objective_rows)
    transfer_rows = read_csv_rows(LOG_ROOT / "transfer" / "summary.csv")

    if dataset_filter:
        dataset_stats = [row for row in dataset_stats if str(row.get("dataset", "")) in dataset_filter]
        full_results = [row for row in full_results if str(row.get("dataset", "")) in dataset_filter]
        cost_rows = [row for row in cost_rows if str(row.get("dataset", "")) in dataset_filter]
        transfer_rows = [row for row in transfer_rows if str(row.get("dataset", "")) in dataset_filter]

    write_csv_rows(output_dir / "appendix_dataset_stats.csv", dataset_stats, ["dataset", "interactions", "sessions", "items", "avg_session_len"])
    write_csv_rows(output_dir / "appendix_full_results_long.csv", full_results, ["dataset", "model", "base_rank", "base_tag", "split", "metric", "value", "result_json"])
    write_csv_rows(output_dir / "appendix_structural_variants.csv", structural_rows, ["dataset", "variant_label", "variant_group", "variant_order", "test_ndcg20", "test_hit10", "base_rank", "seed_id"])
    write_csv_rows(output_dir / "appendix_sparse_tradeoff.csv", sparse_rows, ["dataset", "setting_key", "setting_label", "variant_label", "test_seen_mrr20", "test_ndcg20", "base_rank", "seed_id"])
    write_csv_rows(output_dir / "appendix_sparse_diagnostics.csv", sparse_diag_rows, ["dataset", "question", "setting_key", "setting_label", "variant_label", "stage_key", "group_entropy_mean", "group_n_eff", "group_top1_max_frac"])
    write_csv_rows(output_dir / "appendix_objective_variants.csv", objective_rows, ["dataset", "setting_key", "setting_label", "variant_label", "test_seen_mrr20", "test_ndcg20", "base_rank", "seed_id"])
    write_csv_rows(output_dir / "appendix_objective_special_metrics.csv", objective_special, ["dataset", "question", "setting_key", "split", "special_block", "metric", "value"])
    write_csv_rows(output_dir / "appendix_cost_summary.csv", cost_rows, ["question", "dataset_scope", "dataset", "model_name", "model", "status", "benchmark_epochs", "build_sec", "train_sec", "infer_sec", "total_params", "active_params", "train_time_ratio", "infer_time_ratio"])
    write_csv_rows(output_dir / "appendix_routing_diagnostics.csv", structural_diag_rows + sparse_diag_rows, ["dataset", "question", "setting_key", "setting_label", "variant_label", "stage_key", "group_entropy_mean", "group_n_eff", "group_top1_max_frac"])
    write_csv_rows(output_dir / "appendix_special_bins.csv", special_bins_case, ["dataset", "group", "metric", "split", "best_valid_seen_mrr20", "test_seen_mrr20"])
    write_csv_rows(output_dir / "appendix_behavior_slice_quality.csv", behavior_case_quality, ["dataset", "group", "eval_split", "test_seen_mrr20", "route_concentration"])
    write_csv_rows(output_dir / "appendix_behavior_slice_profiles.csv", behavior_case_profile, ["dataset", "group", "feature_name", "feature_value"])
    write_csv_rows(output_dir / "appendix_case_routing_profile.csv", qualitative_cases, ["dataset", "group", "stage_name", "routed_family", "usage_share"])
    write_csv_rows(output_dir / "appendix_diagnostic_case_profile.csv", diagnostic_case_profile, ["dataset", "group", "stage_name", "routed_family", "usage_share"])
    write_csv_rows(output_dir / "appendix_intervention_summary.csv", intervention_rows, ["dataset", "intervention", "intervention_label", "target_family", "test_mrr20", "test_seen_mrr20"])
    write_csv_rows(output_dir / "appendix_transfer_summary.csv", transfer_rows, ["dataset", "setting_key", "status"])

    run_index: list[dict[str, Any]] = []
    for question, rows in (
        ("structural", structural_rows),
        ("sparse", sparse_rows),
        ("objective", objective_rows),
        ("behavior_slices", behavior_rows),
        ("cases", cases_rows),
        ("cost", cost_rows),
    ):
        for row in rows:
            run_index.append({"question": question, **row})
    write_csv_rows(output_dir / "appendix_run_index.csv", run_index)
    dump_json(
        output_dir / "appendix_manifest.json",
        {
            "generated_at": now_utc(),
            "datasets": sorted(dataset_filter) if dataset_filter else [],
            "files": sorted(path.name for path in output_dir.glob("appendix_*")),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
