#!/usr/bin/env python3
"""Export Q2~Q5 ablation outputs into notebook-friendly CSV/JSON bundles."""

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

from common import DATA_ROOT, LOG_ROOT, RESULT_ROOT, SUMMARY_FIELDS  # noqa: E402

# When set, only rows whose "dataset" field matches are exported.
_DATASETS_FILTER: set[str] = set()


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
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
    valid_unseen = _special_block(valid_special, "overall_unseen_target")
    test_unseen = _special_block(test_special, "overall_unseen_target")
    row = dict(summary_row)
    row.update(
        {
            "selection_rule": "overall_seen_target",
            "best_checkpoint_file": str(payload.get("best_checkpoint_file", "") or ""),
            "special_result_file": str(payload.get("special_result_file", "") or ""),
            "logging_bundle_dir": str(payload.get("logging_bundle_dir", "") or ""),
            "diag_meta_file": str(payload.get("diag_meta_file", "") or ""),
            "best_valid_mean": _metric_mean(valid),
            "test_mean": _metric_mean(test),
            "best_valid_seen_mean": _metric_mean(valid_seen),
            "test_seen_mean": _metric_mean(test_seen),
            "best_valid_unseen_mean": _metric_mean(valid_unseen),
            "test_unseen_mean": _metric_mean(test_unseen),
            "best_valid_seen_mrr20": _safe_float(valid_seen.get("mrr@20", payload.get("valid_seen_target_mrr20"))),
            "test_seen_mrr20": _safe_float(test_seen.get("mrr@20", payload.get("test_seen_target_mrr20"))),
            "best_valid_unseen_mrr20": _safe_float(valid_unseen.get("mrr@20")),
            "test_unseen_mrr20": _safe_float(test_unseen.get("mrr@20")),
            "best_valid_mrr20": _safe_float(valid.get("mrr@20")),
            "test_mrr20": _safe_float(test.get("mrr@20")),
            "best_valid_ndcg20": _safe_float(valid.get("ndcg@20")),
            "test_ndcg20": _safe_float(test.get("ndcg@20")),
            "best_valid_hit10": _safe_float(valid.get("hit@10")),
            "test_hit10": _safe_float(test.get("hit@10")),
        }
    )
    return row


def _collect_summary_result_rows(question: str) -> list[dict[str, Any]]:
    rows = []
    for summary_row in _read_csv(LOG_ROOT / question / "summary.csv"):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        if _DATASETS_FILTER and str(summary_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        row = _summary_to_result_row(summary_row)
        if row is not None:
            row["question"] = question
            rows.append(row)
    rows.sort(key=lambda row: (str(row.get("dataset", "")), str(row.get("setting_key", "")), int(row.get("base_rank", 0) or 0), int(row.get("seed_id", 0) or 0)))
    return rows


def _concat_case_export(index_csv: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index_row in _read_csv(index_csv):
        if _DATASETS_FILTER and str(index_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        export_dir = str(index_row.get("case_eval_export_dir", "")).strip()
        if not export_dir:
            continue
        target_path = Path(export_dir) / filename
        for row in _read_csv(target_path):
            merged = dict(index_row)
            merged.update(row)
            rows.append(merged)
    return rows


def _intervention_performance_rows(index_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    performance_rows: list[dict[str, Any]] = []
    route_rows_raw: list[dict[str, Any]] = []
    for index_row in _read_csv(index_csv):
        if _DATASETS_FILTER and str(index_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        manifest_path = str(index_row.get("intervention_manifest", "")).strip()
        if not manifest_path:
            continue
        manifest_rows = _read_csv(Path(manifest_path))
        full_router_payloads: dict[tuple[str, str], dict[str, Any]] = {}
        full_stage_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        raw_rows: list[dict[str, Any]] = []
        for row in manifest_rows:
            if str(row.get("status", "")).lower() != "ok":
                continue
            result_file = str(row.get("result_file", "")).strip()
            payload = _load_json(result_file) if result_file else {}
            special_payload = _load_json(row["special_metrics_file"]) if str(row.get("special_metrics_file", "")).strip() and Path(str(row["special_metrics_file"])).exists() else {}
            valid_seen = (special_payload.get("valid_overall_seen_target") or {}).get("mrr@20", payload.get("valid_seen_target_mrr20", 0.0))
            test_seen = (special_payload.get("test_overall_seen_target") or {}).get("mrr@20", payload.get("test_seen_target_mrr20", 0.0))
            perf = {
                **index_row,
                **row,
                "best_valid_mrr20": _safe_float((payload.get("best_valid_result") or {}).get("mrr@20", row.get("best_valid_mrr20"))),
                "test_mrr20": _safe_float((payload.get("test_result") or {}).get("mrr@20", row.get("test_mrr20"))),
                "best_valid_seen_mrr20": _safe_float(valid_seen),
                "test_seen_mrr20": _safe_float(test_seen),
            }
            performance_rows.append(perf)

            router_file = str(row.get("router_diag_file", "")).strip()
            if router_file and Path(router_file).exists():
                router_payload = _load_json(router_file)
                for split_name in ("valid", "test"):
                    stage_metrics = ((router_payload.get(split_name) or {}).get("stage_metrics") or {})
                    for stage_key, stage_payload in stage_metrics.items():
                        group_routing = stage_payload.get("group_routing") or {}
                        group_names = list(group_routing.get("group_names") or [])
                        group_share = list(group_routing.get("group_share") or [])
                        for family, share in zip(group_names, group_share):
                            raw_entry = {
                                **index_row,
                                **row,
                                "eval_split": split_name,
                                "stage_key": stage_key,
                                "stage_name": str(stage_key).split("@", 1)[0],
                                "expert_group": family,
                                "mass": _safe_float(share),
                                "group_entropy_mean": _safe_float(group_routing.get("group_entropy_mean")),
                                "group_n_eff": _safe_float(group_routing.get("group_n_eff")),
                                "route_concentration": 1.0 - _safe_float(group_routing.get("group_entropy_mean")),
                            }
                            raw_rows.append(raw_entry)
                            if str(row.get("intervention", "")) == "full":
                                full_stage_lookup[(str(row.get("source_result_json", "")), split_name, str(stage_key), str(family))] = raw_entry
        for raw in raw_rows:
            baseline = full_stage_lookup.get((str(raw.get("source_result_json", "")), str(raw.get("eval_split", "")), str(raw.get("stage_key", "")), str(raw.get("expert_group", ""))))
            if baseline is None:
                continue
            if str(raw.get("intervention", "")) == "full":
                continue
            route_rows_raw.append(
                {
                    **raw,
                    "delta_mass": _safe_float(raw.get("mass")) - _safe_float(baseline.get("mass")),
                    "delta_group_entropy": _safe_float(raw.get("group_entropy_mean")) - _safe_float(baseline.get("group_entropy_mean")),
                    "delta_group_n_eff": _safe_float(raw.get("group_n_eff")) - _safe_float(baseline.get("group_n_eff")),
                    "delta_route_concentration": _safe_float(raw.get("route_concentration")) - _safe_float(baseline.get("route_concentration")),
                }
            )
    return performance_rows, route_rows_raw


def _best_dense_vs_staged_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "")), str(row.get("base_rank", "")), str(row.get("seed_id", "")))].append(row)

    selected: list[dict[str, Any]] = []
    for _, group_rows in grouped.items():
        full = [row for row in group_rows if str(row.get("setting_key", "")) == "full_three_stage"]
        dense = [row for row in group_rows if str(row.get("setting_key", "")) == "dense_full_only"]
        single = [row for row in group_rows if str(row.get("setting_key", "")).startswith("single_stage_")]
        two_stage = [row for row in group_rows if str(row.get("setting_key", "")) in {"remove_macro", "remove_mid", "remove_micro"}]
        if full:
            selected.extend(full[:1])
        if dense:
            selected.extend(dense[:1])
        if single:
            selected.append(max(single, key=lambda row: _safe_float(row.get("best_valid_seen_mrr20"))))
        if two_stage:
            selected.append(max(two_stage, key=lambda row: _safe_float(row.get("best_valid_seen_mrr20"))))
    return selected


def main() -> int:
    global _DATASETS_FILTER
    parser = argparse.ArgumentParser(description="Export Q2~Q5 ablation outputs.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT))
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter (empty = all).")
    args, _unknown = parser.parse_known_args()
    _DATASETS_FILTER = {d.strip() for d in str(args.datasets).split(",") if d.strip()}
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q2_rows = _collect_summary_result_rows("q2")
    q3_rows = _collect_summary_result_rows("q3")
    q4_rows = _collect_summary_result_rows("q4")
    q5_rows = _collect_summary_result_rows("q5")

    _write_csv(output_dir / "q2_quality.csv", q2_rows)
    _write_csv(output_dir / "q3_all_runs.csv", q3_rows)
    _write_csv(output_dir / "q3_stage_ablation.csv", [row for row in q3_rows if str(row.get("setting_key", "")) in {"full_three_stage", "remove_macro", "remove_mid", "remove_micro"}])
    _write_csv(output_dir / "q3_dense_vs_staged.csv", _best_dense_vs_staged_rows(q3_rows))
    _write_csv(output_dir / "q4_all_runs.csv", q4_rows)
    _write_csv(output_dir / "q4_cue_ablation.csv", [row for row in q4_rows if str(row.get("setting_key", "")) in {"full", "remove_category", "remove_time", "sequence_only"}])
    _write_csv(output_dir / "q4_routing_footprint.csv", q4_rows)

    q2_case_perf = _concat_case_export(LOG_ROOT / "q2" / "q2_case_eval_index.csv", "case_eval_performance.csv")
    q2_case_stage = _concat_case_export(LOG_ROOT / "q2" / "q2_case_eval_index.csv", "case_eval_stage_summary.csv")
    q2_case_profile = _concat_case_export(LOG_ROOT / "q2" / "q2_case_eval_index.csv", "case_eval_routing_profile.csv")
    _write_csv(output_dir / "q2_route_rec_case_performance.csv", q2_case_perf)
    _write_csv(output_dir / "q2_route_rec_case_stage_summary.csv", q2_case_stage)
    _write_csv(output_dir / "q2_route_rec_case_routing_profile.csv", q2_case_profile)

    q5_case_perf = _concat_case_export(LOG_ROOT / "q5" / "q5_case_eval_index.csv", "case_eval_performance.csv")
    q5_case_profile = _concat_case_export(LOG_ROOT / "q5" / "q5_case_eval_index.csv", "case_eval_routing_profile.csv")
    _write_csv(output_dir / "q5_behavior_case_performance.csv", q5_case_perf)
    _write_csv(output_dir / "q5_behavior_case_routing_profile.csv", q5_case_profile)

    q5_intervention_perf, q5_route_shift = _intervention_performance_rows(LOG_ROOT / "q5" / "q5_intervention_index.csv")
    _write_csv(output_dir / "q5_intervention_quality.csv", q5_intervention_perf)
    _write_csv(output_dir / "q5_intervention_route_shift.csv", q5_route_shift)

    run_index = []
    for question in ("q2", "q3", "q4", "q5"):
        run_index.extend(_read_csv(LOG_ROOT / question / "summary.csv"))
    _write_csv(output_dir / "q_suite_run_index.csv", run_index)

    bundles = {
        "q2": {"quality_rows": len(q2_rows), "case_rows": len(q2_case_perf)},
        "q3": {"all_rows": len(q3_rows)},
        "q4": {"all_rows": len(q4_rows)},
        "q5": {"train_rows": len(q5_rows), "case_rows": len(q5_case_perf), "intervention_rows": len(q5_intervention_perf)},
    }
    with open(output_dir / "q2_bundle.json", "w", encoding="utf-8") as f:
        json.dump({"rows": len(q2_rows), "case_rows": len(q2_case_perf)}, f, indent=2, ensure_ascii=False)
    with open(output_dir / "q3_bundle.json", "w", encoding="utf-8") as f:
        json.dump({"rows": len(q3_rows)}, f, indent=2, ensure_ascii=False)
    with open(output_dir / "q4_bundle.json", "w", encoding="utf-8") as f:
        json.dump({"rows": len(q4_rows)}, f, indent=2, ensure_ascii=False)
    with open(output_dir / "q5_bundle.json", "w", encoding="utf-8") as f:
        json.dump({"rows": len(q5_rows), "case_rows": len(q5_case_perf), "intervention_rows": len(q5_intervention_perf)}, f, indent=2, ensure_ascii=False)
    with open(output_dir / "q_suite_manifest.json", "w", encoding="utf-8") as f:
        json.dump(bundles, f, indent=2, ensure_ascii=False)
    print(f"[DONE] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
