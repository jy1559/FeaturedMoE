#!/usr/bin/env python3
"""Export real-final Q2~Q5 outputs into notebook-friendly CSV/JSON bundles."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import DATA_ROOT, LOG_ROOT, now_utc  # noqa: E402


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames_hint: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames and fieldnames_hint:
        fieldnames = list(fieldnames_hint)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _metric_mean(metrics: dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    keys = ["hit@5", "hit@10", "hit@20", "ndcg@5", "ndcg@10", "ndcg@20", "mrr@5", "mrr@10", "mrr@20"]
    values = [_safe_float(metrics.get(key)) for key in keys if key in metrics]
    return float(sum(values) / len(values)) if values else _safe_float(metrics.get("mrr@20"))


def _special_block(payload: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    block = payload.get(key) or {}
    return block if isinstance(block, dict) else {}


def _summary_to_result_row(summary_row: dict[str, str]) -> dict[str, Any] | None:
    result_path = str(summary_row.get("result_path", "")).strip()
    if not result_path:
        return None
    payload = _load_json(result_path)
    valid = payload.get("best_valid_result") or {}
    test = payload.get("test_result") or {}
    valid_special = payload.get("best_valid_special_metrics") or {}
    test_special = payload.get("test_special_metrics") or {}
    valid_seen = _special_block(valid_special, "overall_seen_target")
    test_seen = _special_block(test_special, "overall_seen_target")
    row = dict(summary_row)
    row.update(
        {
            "selection_rule": "overall_seen_target",
            "best_valid_mean": _metric_mean(valid),
            "test_mean": _metric_mean(test),
            "best_valid_seen_mean": _metric_mean(valid_seen),
            "test_seen_mean": _metric_mean(test_seen),
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


def _collect_summary_rows(question: str, dataset_filter: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _read_csv(LOG_ROOT / question / "summary.csv"):
        if str(row.get("status", "")).lower() != "ok":
            continue
        if dataset_filter and str(row.get("dataset", "")) not in dataset_filter:
            continue
        result_row = _summary_to_result_row(row)
        if result_row is not None:
            out.append(result_row)
    return out


def _concat_case_export(index_csv: Path, filename: str, dataset_filter: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_row in _read_csv(index_csv):
        if dataset_filter and str(index_row.get("dataset", "")) not in dataset_filter:
            continue
        export_dir = str(index_row.get("case_eval_export_dir", "")).strip()
        if not export_dir:
            continue
        target = Path(export_dir) / filename
        for row in _read_csv(target):
            merged = dict(index_row)
            merged.update(row)
            rows.append(merged)
    return rows


def _collapse_q3(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    temporal: list[dict[str, Any]] = []
    routing_org: list[dict[str, Any]] = []
    grouped_temporal: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        panel_family = str(row.get("panel_family", ""))
        if panel_family == "routing_org":
            routing_org.append(row)
        elif panel_family == "temporal_decomp":
            key = (
                str(row.get("dataset", "")),
                str(row.get("base_rank", "")),
                str(row.get("seed_id", "")),
                str(row.get("variant_group", "")),
            )
            grouped_temporal[key].append(row)
    reduced: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, base_rank, seed_id, variant_group), group_rows in grouped_temporal.items():
        picked = max(group_rows, key=lambda item: _safe_float(item.get("best_valid_seen_mrr20", item.get("best_valid_mrr20", 0.0))))
        reduced[(dataset, base_rank, seed_id)].append(picked)
    for key in sorted(reduced):
        temporal.extend(sorted(reduced[key], key=lambda item: int(item.get("variant_order", 0) or 0)))
    routing_org.sort(key=lambda item: (str(item.get("dataset", "")), int(item.get("variant_order", 0) or 0), str(item.get("base_rank", "")), str(item.get("seed_id", ""))))
    return temporal, routing_org


def _map_case_name(group: str) -> tuple[str, str]:
    mapping = {
        "memory_plus": ("Repeat-heavy", "repeat_heavy"),
        "tempo_plus": ("Fast exploratory", "fast_exploratory"),
        "focus_plus": ("Narrow-focus", "narrow_focus"),
    }
    return mapping.get(str(group), ("", ""))


def _build_q5_case_heatmap(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in case_rows:
        case_name, case_type = _map_case_name(str(row.get("group", "")))
        if not case_name:
            continue
        if str(row.get("eval_split", "")).lower() != "test":
            continue
        out.append(
            {
                "dataset": row.get("dataset", ""),
                "case_name": case_name,
                "case_type": case_type,
                "group_name": row.get("routed_family", ""),
                "stage": row.get("stage_name", ""),
                "expert_rank_or_slot": "group_total",
                "selected_mass": _safe_float(row.get("usage_share")),
                "short_description": case_name,
                "base_rank": row.get("base_rank", ""),
                "seed_id": row.get("seed_id", ""),
                "setting_key": row.get("setting_key", ""),
            }
        )
    return out


def _build_q5_intervention_summary(index_csv: Path, dataset_filter: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index_row in _read_csv(index_csv):
        if dataset_filter and str(index_row.get("dataset", "")) not in dataset_filter:
            continue
        manifest_path = str(index_row.get("intervention_manifest", "")).strip()
        if not manifest_path:
            continue
        for row in _read_csv(Path(manifest_path)):
            if str(row.get("status", "")).lower() != "ok":
                continue
            result_file = str(row.get("result_file", "")).strip()
            payload = _load_json(result_file) if result_file else {}
            out.append(
                {
                    "dataset": index_row.get("dataset", ""),
                    "base_rank": index_row.get("base_rank", ""),
                    "seed_id": index_row.get("seed_id", ""),
                    "intervention": row.get("intervention", ""),
                    "intervention_label": row.get("intervention_label", ""),
                    "intervention_group": row.get("intervention_group", ""),
                    "target_family": row.get("target_family", ""),
                    "best_valid_mrr20": _safe_float((payload.get("best_valid_result") or {}).get("mrr@20", row.get("best_valid_mrr20"))),
                    "test_mrr20": _safe_float((payload.get("test_result") or {}).get("mrr@20", row.get("test_mrr20"))),
                    "best_valid_seen_mrr20": _safe_float(row.get("best_valid_seen_mrr20")),
                    "test_seen_mrr20": _safe_float(row.get("test_seen_mrr20")),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export real-final Q2~Q5 outputs.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT))
    parser.add_argument("--datasets", default="")
    args, _unknown = parser.parse_known_args()

    dataset_filter = {token.strip() for token in str(args.datasets).split(",") if token.strip()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q2_rows = _collect_summary_rows("q2", dataset_filter)
    q3_rows = _collect_summary_rows("q3", dataset_filter)
    q4_rows = _read_csv(LOG_ROOT / "q4" / "summary.csv")
    q5_rows = _collect_summary_rows("q5", dataset_filter)

    q5_case_profile = _concat_case_export(LOG_ROOT / "q5" / "q5_case_eval_index.csv", "case_eval_routing_profile.csv", dataset_filter)
    q3_temporal, q3_routing_org = _collapse_q3(q3_rows)
    q5_case_heatmap = _build_q5_case_heatmap(q5_case_profile)
    q5_intervention_summary = _build_q5_intervention_summary(LOG_ROOT / "q5" / "q5_intervention_index.csv", dataset_filter)

    _write_csv(output_dir / "q2_quality.csv", q2_rows, ["dataset", "variant_label", "test_ndcg20", "test_hit10", "base_rank", "seed_id"])
    _write_csv(output_dir / "q3_temporal_decomp.csv", q3_temporal, ["dataset", "variant_label", "variant_group", "variant_order", "test_ndcg20", "test_hit10", "base_rank", "seed_id"])
    _write_csv(output_dir / "q3_routing_org.csv", q3_routing_org, ["dataset", "variant_label", "variant_group", "variant_order", "test_ndcg20", "test_hit10", "base_rank", "seed_id"])
    _write_csv(output_dir / "q4_efficiency_table.csv", q4_rows, ["dataset_scope", "dataset", "model_name", "total_params", "active_params", "train_time_ratio", "infer_time_ratio", "status"])
    _write_csv(output_dir / "q5_case_heatmap.csv", q5_case_heatmap, ["dataset", "case_name", "case_type", "stage", "group_name", "expert_rank_or_slot", "selected_mass", "base_rank", "seed_id"])
    _write_csv(output_dir / "q5_intervention_summary.csv", q5_intervention_summary, ["dataset", "intervention", "intervention_label", "target_family", "test_mrr20", "test_seen_mrr20", "base_rank", "seed_id"])

    run_index = []
    for question, rows in (("q2", q2_rows), ("q3", q3_rows), ("q4", q4_rows), ("q5", q5_rows)):
        for row in rows:
            run_index.append({"question": question, **row})
    _write_csv(output_dir / "q_suite_run_index.csv", run_index, ["question", "dataset", "setting_key", "model_name", "status"])

    with open(output_dir / "q_suite_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": now_utc(),
                "dataset_filter": sorted(dataset_filter),
                "counts": {
                    "q2_quality": len(q2_rows),
                    "q3_temporal_decomp": len(q3_temporal),
                    "q3_routing_org": len(q3_routing_org),
                    "q4_efficiency_table": len(q4_rows),
                    "q5_case_heatmap": len(q5_case_heatmap),
                    "q5_intervention_summary": len(q5_intervention_summary),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[export] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
