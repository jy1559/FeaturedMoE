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


def _find_router_diag_from_case_eval(index_row: dict[str, Any]) -> Path | None:
    manifest_path = Path(str(index_row.get("case_eval_manifest", "")).strip())
    if manifest_path.exists():
        run_dir = manifest_path.parent
        logging_dir = run_dir.parent / "logging" / run_dir.name
        candidate = logging_dir / "router_diag.json"
        if candidate.exists():
            return candidate
        fallback = sorted(logging_dir.glob("**/router_diag.json"))
        if fallback:
            return fallback[0]
    export_dir = Path(str(index_row.get("case_eval_export_dir", "")).strip())
    if export_dir.exists():
        candidates = sorted(export_dir.parent.glob("**/router_diag.json"))
        if candidates:
            return candidates[0]
    return None


def _router_diag_rows_from_case_eval(index_csv: Path, dataset_filter: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index_row in read_csv_rows(index_csv):
        if str(index_row.get("status", "")).lower() != "ok":
            continue
        dataset = str(index_row.get("dataset", ""))
        if dataset_filter and dataset not in dataset_filter:
            continue
        diag_path = _find_router_diag_from_case_eval(index_row)
        if diag_path is None or not diag_path.exists():
            continue
        diag = _load_json(diag_path)
        stage_metrics = ((diag.get("test") or {}).get("stage_metrics") or {})
        for stage_key, stage_payload in stage_metrics.items():
            group_routing = (stage_payload or {}).get("group_routing") or {}
            out.append(
                {
                    "dataset": dataset,
                    "question": str(index_row.get("source_question", index_row.get("question", ""))),
                    "setting_key": index_row.get("setting_key", ""),
                    "setting_label": index_row.get("setting_label", ""),
                    "variant_label": index_row.get("setting_label", ""),
                    "base_rank": index_row.get("base_rank", ""),
                    "seed_id": index_row.get("seed_id", ""),
                    "stage_key": str(stage_key).split("@", 1)[0],
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


def _display_model_name(raw: str) -> str:
    name = str(raw or "").strip().lower()
    if name in {"featured_moe_n3", "featured_moe_n3_tune", "routerec", "routerec_full"}:
        return "RouteRec"
    if name == "sasrec":
        return "SASRec"
    if name == "fame":
        return "FAME"
    return str(raw or "")


def _special_bins_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pop_group_map = {
        "<=5": "tail (1-5)",
        "6-20": "mid (6-20)",
        "21-100": "head (21+)",
        ">100": "head (21+)",
        "rare_1_5": "tail (1-5)",
        "21_100": "head (21+)",
        "101+": "head (21+)",
    }
    for row in result_rows:
        payload = _load_json(str(row.get("result_path", "")))
        slices = ((payload.get("test_special_metrics") or {}).get("slices") or {})
        model_label = _display_model_name(str(row.get("model", "")))
        for source_key in ("session_len", "session_len_legacy"):
            block = slices.get(source_key) or {}
            if not isinstance(block, dict):
                continue
            for group, metric_row in block.items():
                out.append(
                    {
                        "dataset": row.get("dataset", ""),
                        "model": model_label,
                        "bin_type": "session",
                        "group": str(group),
                        "test_seen_mrr20": _safe_float((metric_row or {}).get("mrr@20")),
                    }
                )
            if block:
                break
        for source_key in ("target_popularity_abs_legacy", "target_popularity_abs"):
            block = slices.get(source_key) or {}
            if not isinstance(block, dict):
                continue
            for group, metric_row in block.items():
                mapped_group = pop_group_map.get(str(group), str(group))
                out.append(
                    {
                        "dataset": row.get("dataset", ""),
                        "model": model_label,
                        "bin_type": "freq",
                        "group": mapped_group,
                        "test_seen_mrr20": _safe_float((metric_row or {}).get("mrr@20")),
                    }
                )
            if block:
                break
    return out


def _behavior_slice_rows(
    quality_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    score_by_key: dict[tuple[str, str], float] = {}
    for row in quality_rows:
        if str(row.get("eval_split", "test")).strip().lower() != "test":
            continue
        score_by_key[(str(row.get("dataset", "")), str(row.get("group", "")))] = _safe_float(
            row.get("test_seen_mrr20")
        )
    conc_acc: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in profile_rows:
        if str(row.get("eval_split", "test")).strip().lower() != "test":
            continue
        dataset = str(row.get("dataset", ""))
        group = str(row.get("group", ""))
        family = (group.rsplit("_", 1)[0] if "_" in group else group).strip().lower()
        routed_family = str(row.get("routed_family", "")).strip().lower()
        if routed_family != family:
            continue
        conc_acc[(dataset, group)].append(_safe_float(row.get("usage_share")))
    out: list[dict[str, Any]] = []
    for key, score in score_by_key.items():
        concentration_vals = conc_acc.get(key, [])
        out.append(
            {
                "dataset": key[0],
                "model": "RouteRec",
                "group": key[1].replace("exposure_plus", "exploration_plus"),
                "test_seen_mrr20": score,
                "route_concentration": sum(concentration_vals) / len(concentration_vals) if concentration_vals else 0.0,
            }
        )
    return out


def _normalize_intervention_name(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    mapping = {
        "full": ("full", "Full cues"),
        "feature_zero_all": ("feature_zero_all", "Zero all cues"),
        "feature_zero_tempo": ("zero_tempo", "Zero Tempo"),
        "feature_zero_focus": ("zero_focus", "Zero Focus"),
        "feature_zero_memory": ("zero_memory", "Zero Memory"),
        "feature_zero_exposure": ("zero_exposure", "Zero Exposure"),
        "feature_shuffle_tempo": ("shuffle_tempo", "Shuffle Tempo"),
        "feature_shuffle_focus": ("shuffle_focus", "Shuffle Focus"),
    }
    return mapping.get(text, (text, text))


def _main_metric_lookup(full_results: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    metric_priority: dict[tuple[str, str], tuple[int, float]] = {}
    for row in full_results:
        model = str(row.get("model", "")).strip().lower()
        if model not in {"featured_moe_n3", "routerec"}:
            continue
        split = str(row.get("split", "")).strip().lower()
        if split != "test":
            continue
        dataset = str(row.get("dataset", "")).strip()
        metric = str(row.get("metric", "")).strip().lower()
        try:
            base_rank = int(float(row.get("base_rank", 999)))
        except Exception:
            base_rank = 999
        value = _safe_float(row.get("value"))
        key = (dataset, metric)
        current = metric_priority.get(key)
        if current is None or base_rank < current[0]:
            metric_priority[key] = (base_rank, value)
    return {key: value for key, (_rank, value) in metric_priority.items()}


def _apply_main_metric_overrides(
    rows: list[dict[str, Any]],
    lookup: dict[tuple[str, str], float],
    selected_keys: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        patched = dict(row)
        if str(row.get("setting_key", "")).strip() in selected_keys:
            dataset = str(row.get("dataset", "")).strip()
            test_seen = lookup.get((dataset, "mrr@20"))
            test_ndcg20 = lookup.get((dataset, "ndcg@20"))
            test_hit10 = lookup.get((dataset, "hit@10"))
            if test_seen is not None:
                patched["test_seen_mrr20"] = test_seen
                patched["test_mrr20"] = test_seen
            if test_ndcg20 is not None:
                patched["test_ndcg20"] = test_ndcg20
            if test_hit10 is not None:
                patched["test_hit10"] = test_hit10
        out.append(patched)
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

    main_metric_map = _main_metric_lookup(full_results)
    structural_rows = _apply_main_metric_overrides(
        structural_rows,
        main_metric_map,
        {"hierarchical_sparse", "final_three_stage"},
    )
    sparse_rows = _apply_main_metric_overrides(
        sparse_rows,
        main_metric_map,
        {"group_top2"},
    )
    objective_rows = _apply_main_metric_overrides(
        objective_rows,
        main_metric_map,
        {"full_objective"},
    )

    diag_rows = _router_diag_rows_from_case_eval(LOG_ROOT / "diagnostics" / "diagnostics_case_eval_index.csv", dataset_filter)
    sparse_diag_rows = [row for row in diag_rows if str(row.get("question", "")).strip() == "sparse"]
    structural_diag_rows = [row for row in diag_rows if str(row.get("question", "")).strip() == "structural"]
    behavior_case_quality = _concat_case_tables(LOG_ROOT / "behavior_slices" / "behavior_slices_case_eval_index.csv", dataset_filter, "case_eval_performance.csv")
    behavior_case_profile = _concat_case_tables(LOG_ROOT / "behavior_slices" / "behavior_slices_case_eval_index.csv", dataset_filter, "case_eval_routing_profile.csv")
    diagnostic_case_profile = _concat_case_tables(LOG_ROOT / "diagnostics" / "diagnostics_case_eval_index.csv", dataset_filter, "case_eval_routing_profile.csv")
    qualitative_cases = _concat_case_tables(LOG_ROOT / "cases" / "cases_case_eval_index.csv", dataset_filter, "case_eval_routing_profile.csv")
    intervention_rows = _intervention_rows(LOG_ROOT / "cases" / "cases_intervention_index.csv", dataset_filter)
    objective_special = _special_metric_rows(objective_rows)
    transfer_rows = read_csv_rows(LOG_ROOT / "transfer" / "summary.csv")
    special_bins_rows = _special_bins_rows(_collect_summary_results("special_bins", dataset_filter))
    behavior_slice_quality_rows = _behavior_slice_rows(behavior_case_quality, behavior_case_profile)
    normalized_intervention_rows: list[dict[str, Any]] = []
    for row in intervention_rows:
        intervention_name, intervention_label = _normalize_intervention_name(str(row.get("intervention", "")))
        normalized_intervention_rows.append(
            {
                **row,
                "intervention": intervention_name,
                "intervention_label": intervention_label,
            }
        )

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
    write_csv_rows(output_dir / "appendix_special_bins.csv", special_bins_rows, ["dataset", "model", "bin_type", "group", "test_seen_mrr20"])
    write_csv_rows(output_dir / "appendix_behavior_slice_quality.csv", behavior_slice_quality_rows, ["dataset", "model", "group", "test_seen_mrr20", "route_concentration"])
    write_csv_rows(output_dir / "appendix_behavior_slice_profiles.csv", behavior_case_profile, ["dataset", "group", "stage_name", "routed_family", "usage_share"])
    write_csv_rows(
        output_dir / "appendix_case_routing_profile.csv",
        [
            {
                **row,
                "group": str(row.get("group", "")).replace("exposure_plus", "exploration_plus"),
            }
            for row in qualitative_cases
            if str(row.get("eval_split", "test")).strip().lower() == "test"
        ],
        ["dataset", "group", "stage_name", "routed_family", "usage_share"],
    )
    write_csv_rows(output_dir / "appendix_diagnostic_case_profile.csv", diagnostic_case_profile, ["dataset", "group", "stage_name", "routed_family", "usage_share"])
    write_csv_rows(output_dir / "appendix_intervention_summary.csv", normalized_intervention_rows, ["dataset", "intervention", "intervention_label", "target_family", "test_mrr20", "test_seen_mrr20"])
    write_csv_rows(
        output_dir / "appendix_transfer_summary.csv",
        transfer_rows,
        ["dataset", "setting_key", "setting_label", "data_fraction", "route_mrr20", "baseline_mrr20", "status"],
    )

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
