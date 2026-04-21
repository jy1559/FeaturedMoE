#!/usr/bin/env python3
"""Export case-eval manifest + diagnostics into flat CSV tables for Q2/Q5 figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "writing" / "260418_final_exp_figure" / "data"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_family_polarity(group: str) -> tuple[str, str]:
    text = str(group or "")
    if "_" not in text:
        return text, ""
    family, polarity = text.rsplit("_", 1)
    return family, polarity


def _special_row(base: dict[str, Any], special_payload: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    valid_seen = special_payload.get("valid_overall_seen_target") or {}
    test_seen = special_payload.get("test_overall_seen_target") or {}
    valid_unseen = special_payload.get("valid_overall_unseen_target") or {}
    test_unseen = special_payload.get("test_overall_unseen_target") or {}
    valid_main = special_payload.get("valid") or {}
    test_main = special_payload.get("test") or {}
    for prefix, block in (
        ("best_valid", valid_main),
        ("test", test_main),
        ("best_valid_seen", valid_seen),
        ("test_seen", test_seen),
        ("best_valid_unseen", valid_unseen),
        ("test_unseen", test_unseen),
    ):
        for metric in ("hit@10", "ndcg@20", "mrr@20"):
            row[f"{prefix}_{metric.replace('@', '')}"] = _safe_float(block.get(metric))
    return row


def _flatten_scalar_metrics(base: dict[str, Any], split_name: str, scalar_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, metric_value in sorted((scalar_metrics or {}).items()):
        rows.append(
            {
                **base,
                "eval_split": split_name,
                "metric_name": str(metric_name),
                "metric_value": _safe_float(metric_value),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export case-eval manifest and router diagnostics into flat CSV tables.")
    parser.add_argument("--manifest", required=True, help="Path to case_eval_manifest.csv")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    rows = _read_manifest(manifest_path)

    performance_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    heatmap_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []

    for row in rows:
        group = str(row.get("group", ""))
        selected_family, selected_polarity = _group_family_polarity(group)
        base = {
            "timestamp_utc": str(row.get("timestamp_utc", "")),
            "source_run_dir": str(row.get("source_run_dir", "")),
            "source_result_json": str(row.get("source_result_json", "")),
            "checkpoint_file": str(row.get("checkpoint_file", "")),
            "dataset": str(row.get("dataset", "")),
            "model": str(row.get("model", "")),
            "scope": str(row.get("scope", "")),
            "tier": str(row.get("tier", "")),
            "group": group,
            "selected_family": selected_family,
            "selected_polarity": selected_polarity,
            "label": str(row.get("label", "")),
            "data_path": str(row.get("data_path", "")),
            "logging_dir": str(row.get("logging_dir", "")),
            "metrics_summary_file": str(row.get("metrics_summary_file", "")),
            "special_metrics_file": str(row.get("special_metrics_file", "")),
            "router_diag_file": str(row.get("router_diag_file", "")),
            "status": str(row.get("status", "")),
            "error": str(row.get("error", "")),
        }

        perf_row = dict(base)
        for key in (
            "best_valid_mrr20",
            "test_mrr20",
            "best_valid_ndcg20",
            "test_ndcg20",
            "best_valid_hr10",
            "test_hr10",
            "best_valid_seen_mrr20",
            "test_seen_mrr20",
        ):
            perf_row[key] = _safe_float(row.get(key))

        special_path = str(row.get("special_metrics_file", "") or "").strip()
        if special_path and Path(special_path).exists():
            perf_row = _special_row(perf_row, _load_json(special_path))
        performance_rows.append(perf_row)

        router_path = str(row.get("router_diag_file", "") or "").strip()
        if not router_path or not Path(router_path).exists():
            continue
        router_payload = _load_json(router_path)
        for split_name in ("valid", "test"):
            diag = router_payload.get(split_name) or {}
            scalar_rows.extend(_flatten_scalar_metrics(base, split_name, diag.get("scalar_metrics") or {}))
            stage_metrics = diag.get("stage_metrics") or {}
            for stage_key, stage_payload in sorted(stage_metrics.items()):
                group_routing = stage_payload.get("group_routing") or {}
                group_names = list(group_routing.get("group_names") or [])
                group_share = list(group_routing.get("group_share") or [])
                expert_names = list(stage_payload.get("expert_names") or [])
                usage_sum = list(stage_payload.get("usage_sum") or [])
                usage_share = list(stage_payload.get("usage_share") or [])
                top1_count = list(stage_payload.get("top1_count") or [])
                feature_heatmap = stage_payload.get("feature_family_expert_heatmap") or {}
                heatmap_family_names = list(feature_heatmap.get("family_names") or [])
                heatmap_values = list(feature_heatmap.get("values") or [])
                stage_base = {
                    **base,
                    "eval_split": split_name,
                    "stage_key": str(stage_key),
                    "stage_name": str(stage_key).split("@", 1)[0],
                }
                stage_rows.append(
                    {
                        **stage_base,
                        "n_eff": _safe_float(stage_payload.get("n_eff")),
                        "cv_usage": _safe_float(stage_payload.get("cv_usage")),
                        "dead_expert_frac": _safe_float(stage_payload.get("dead_expert_frac")),
                        "top1_max_frac": _safe_float(stage_payload.get("top1_max_frac")),
                        "entropy_mean": _safe_float(stage_payload.get("entropy_mean")),
                        "group_n_eff": _safe_float(group_routing.get("group_n_eff")),
                        "group_cv_usage": _safe_float(group_routing.get("group_cv_usage")),
                        "group_top1_max_frac": _safe_float(group_routing.get("group_top1_max_frac")),
                        "group_entropy_mean": _safe_float(group_routing.get("group_entropy_mean")),
                        "factored_group_entropy_mean": _safe_float(group_routing.get("factored_group_entropy_mean")),
                        "route_consistency_knn_score": _safe_float(stage_payload.get("route_consistency_knn_score")),
                        "route_consistency_knn_js": _safe_float(stage_payload.get("route_consistency_knn_js")),
                        "route_consistency_group_knn_score": _safe_float(stage_payload.get("route_consistency_group_knn_score")),
                        "route_consistency_group_knn_js": _safe_float(stage_payload.get("route_consistency_group_knn_js")),
                        "feature_group_consistency_mean_score": _safe_float(
                            ((stage_payload.get("route_consistency_feature_group_knn") or {}).get("mean_score"))
                        ),
                        "feature_group_consistency_mean_js": _safe_float(
                            ((stage_payload.get("route_consistency_feature_group_knn") or {}).get("mean_js"))
                        ),
                        "family_top_share_mean": _safe_float(
                            ((stage_payload.get("specialization_summary") or {}).get("mean_top_expert_share"))
                        ),
                        "expert_similarity_mean": _safe_float(stage_payload.get("expert_similarity_mean")),
                        "expert_similarity_max": _safe_float(stage_payload.get("expert_similarity_max")),
                        "family_count": len(group_names),
                        "expert_count": len(expert_names),
                    }
                )
                for routed_family, share in zip(group_names, group_share):
                    profile_rows.append(
                        {
                            **stage_base,
                            "routed_family": str(routed_family),
                            "usage_share": _safe_float(share),
                            "selected_vs_routed_match": int(str(routed_family) == selected_family),
                            "group_n_eff": _safe_float(group_routing.get("group_n_eff")),
                            "group_entropy_mean": _safe_float(group_routing.get("group_entropy_mean")),
                            "group_top1_max_frac": _safe_float(group_routing.get("group_top1_max_frac")),
                            "feature_group_consistency_mean_score": _safe_float(
                                ((stage_payload.get("route_consistency_feature_group_knn") or {}).get("mean_score"))
                            ),
                        }
                    )
                for expert_idx, expert_name in enumerate(expert_names):
                    expert_family = str(expert_name).split("_", 1)[0].lower()
                    expert_rows.append(
                        {
                            **stage_base,
                            "expert_index": int(expert_idx),
                            "expert_name": str(expert_name),
                            "expert_family": expert_family,
                            "usage_sum": _safe_float(usage_sum[expert_idx] if expert_idx < len(usage_sum) else 0.0),
                            "usage_share": _safe_float(usage_share[expert_idx] if expert_idx < len(usage_share) else 0.0),
                            "top1_count": _safe_float(top1_count[expert_idx] if expert_idx < len(top1_count) else 0.0),
                            "selected_vs_expert_family_match": int(expert_family == str(selected_family).lower()),
                        }
                    )
                for family_name, family_values in zip(heatmap_family_names, heatmap_values):
                    family_total = sum(_safe_float(value) for value in family_values)
                    for expert_idx, raw_value in enumerate(family_values):
                        expert_name = str(expert_names[expert_idx]) if expert_idx < len(expert_names) else f"expert_{expert_idx}"
                        heatmap_rows.append(
                            {
                                **stage_base,
                                "source_family": str(family_name).lower(),
                                "expert_index": int(expert_idx),
                                "expert_name": expert_name,
                                "raw_value": _safe_float(raw_value),
                                "family_share": _safe_float(_safe_float(raw_value) / family_total if family_total > 0 else 0.0),
                            }
                        )

    performance_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("label", ""))))
    stage_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("eval_split", "")), str(r.get("stage_key", ""))))
    profile_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("eval_split", "")), str(r.get("stage_key", "")), str(r.get("routed_family", ""))))
    expert_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("eval_split", "")), str(r.get("stage_key", "")), int(r.get("expert_index", 0))))
    heatmap_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("eval_split", "")), str(r.get("stage_key", "")), str(r.get("source_family", "")), int(r.get("expert_index", 0))))
    scalar_rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("tier", "")), str(r.get("group", "")), str(r.get("eval_split", "")), str(r.get("metric_name", ""))))

    _write_csv(output_dir / "case_eval_performance.csv", performance_rows)
    _write_csv(output_dir / "case_eval_stage_summary.csv", stage_rows)
    _write_csv(output_dir / "case_eval_routing_profile.csv", profile_rows)
    _write_csv(output_dir / "case_eval_expert_profile.csv", expert_rows)
    _write_csv(output_dir / "case_eval_family_expert_heatmap.csv", heatmap_rows)
    _write_csv(output_dir / "case_eval_router_diag_scalars.csv", scalar_rows)
    print(
        f"[DONE] performance={len(performance_rows)} stage_rows={len(stage_rows)} "
        f"profile_rows={len(profile_rows)} expert_rows={len(expert_rows)} "
        f"heatmap_rows={len(heatmap_rows)} scalar_rows={len(scalar_rows)} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
