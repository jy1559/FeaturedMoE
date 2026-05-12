#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "experiments" / "run" / "artifacts"
DOCS_DIR = ROOT / "docs"
ANCHOR_CATALOG = DOCS_DIR / "artifact_anchor_catalog.csv"

STORAGE_OVERVIEW_CSV = DOCS_DIR / "artifact_storage_overview.csv"
TOP_HPARAMS_CSV = DOCS_DIR / "artifact_top_hparams.csv"
DIAG_INVENTORY_CSV = DOCS_DIR / "artifact_diagnostic_inventory.csv"
CLEANUP_PLAN_JSON = DOCS_DIR / "artifact_cleanup_plan.json"
CLEANUP_SUMMARY_MD = DOCS_DIR / "artifact_cleanup_summary.md"
TRENDS_MD = DOCS_DIR / "artifact_trends.md"

HEAVY_CHILDREN = {
    "experiments",
    "logging",
    "logs",
    "quarantine",
    "results",
    "results_old",
    "tmp",
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _storage_action(scope_path: str) -> tuple[str, str]:
    if scope_path in {"analysis", "inventory", "reports", "timeline"}:
        return "keep", "Small summary directories; keep as lightweight provenance."
    if scope_path == "tmp":
        return "delete_now_if_idle", "Temporary workspace; mostly checkpoint spill and scratch outputs."
    if scope_path.startswith("tmp/best_stage"):
        return "delete_now_if_idle", "Resume checkpoints only; large and not needed after summarizing anchors."
    if scope_path == "results":
        return "keep_subset_only", "Keep only current-doc-backed subsets or regenerateable summaries."
    if scope_path in {"results/final_experiment", "results/results_final_experiment_fmoe"}:
        return "summarized_then_delete", "Covered by overall top-hparam CSV and trend summaries."
    if scope_path.startswith("results_old"):
        return "index_then_delete", "Legacy result copies; retain only compact inventories."
    if scope_path == "logs":
        return "keep_subset_only", "Retain only representative diagnostic subsets after indexing."
    if scope_path.startswith("logs/real_final_ablation"):
        return "keep_diag_index_only", "Case-eval diagnostics are useful, but full raw heatmaps are bulky."
    if scope_path.startswith("logs/final_experiment"):
        return "summarized_then_delete", "Run logs back the final sweep, but high-value information is already in best-config summaries."
    if scope_path == "logging":
        return "index_then_delete", "Massive raw logging cache; keep only representative diagnostic pointers."
    if scope_path.startswith("logging/"):
        return "index_then_delete", "Nested raw logging directories are archival rather than launch-time inputs."
    if scope_path == "quarantine":
        return "review_then_delete", "Likely redundant historical quarantine; spot-check before purge."
    if scope_path == "experiments":
        return "review_then_delete", "Artifact-side experiment snapshots should be reviewed once against docs outputs."
    return "review", "Needs a one-time spot check before deciding."


def _signal_kind(parts: tuple[str, ...]) -> str | None:
    lowered = [part.lower() for part in parts]
    if any("diag" in part for part in lowered):
        return "diag"
    if any("special" in part for part in lowered):
        return "special"
    return None


def _signal_prefix(parts: tuple[str, ...]) -> str:
    directory_parts = parts[:-1] or parts
    end = min(len(directory_parts), 4)
    return "/".join(directory_parts[:end])


def _diag_action(prefix: str) -> tuple[str, str]:
    if prefix.startswith("results/baseline/diag"):
        return "keep_csv_json_only", "Diagnostic metrics and manifests are useful; raw tree can be collapsed to inventories."
    if prefix.startswith("results/baseline/special"):
        return "keep_metric_subset", "Special metrics JSON may be useful, but keep only representative subsets."
    if prefix.startswith("logs/real_final_ablation"):
        return "keep_examples_only", "Preserve only a few representative case_eval outputs per question or trial family."
    if prefix.startswith("logging/") or prefix.startswith("logs/"):
        return "index_then_delete", "Prefer path-level inventories over keeping the full raw directory."
    return "review", "Needs manual review if it is still referenced by downstream writing."


def _scan_artifacts() -> dict[str, Any]:
    storage: dict[tuple[int, str], dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    diag_groups: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kind": None, "files": 0, "bytes": 0, "sample_file": None, "largest_file": None, "largest_file_bytes": 0}
    )
    largest_files: list[tuple[int, str]] = []
    total_files = 0
    total_bytes = 0

    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        try:
            size = path.stat().st_size
        except OSError:
            continue

        rel = path.relative_to(ARTIFACT_ROOT)
        rel_text = rel.as_posix()
        parts = rel.parts

        total_files += 1
        total_bytes += size
        largest_files.append((size, rel_text))

        keys = [(1, parts[0])]
        if parts[0] in HEAVY_CHILDREN and len(parts) >= 2:
            keys.append((2, "/".join(parts[:2])))
        if parts[0] == "tmp" and len(parts) >= 3:
            keys.append((3, "/".join(parts[:3])))

        for depth, scope_path in keys:
            row = storage[(depth, scope_path)]
            row["files"] += 1
            row["bytes"] += size

        kind = _signal_kind(parts)
        if kind is not None:
            prefix = _signal_prefix(parts)
            row = diag_groups[prefix]
            row["kind"] = kind
            row["files"] += 1
            row["bytes"] += size
            if row["sample_file"] is None:
                row["sample_file"] = rel_text
            if size > int(row["largest_file_bytes"]):
                row["largest_file"] = rel_text
                row["largest_file_bytes"] = size

    largest_files.sort(reverse=True)
    return {
        "storage": storage,
        "diag_groups": diag_groups,
        "largest_files": largest_files[:25],
        "total_files": total_files,
        "total_bytes": total_bytes,
    }


def _build_storage_rows(scan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_bytes = max(int(scan["total_bytes"]), 1)
    for (depth, scope_path), stats in sorted(
        scan["storage"].items(), key=lambda item: (item[0][0], -int(item[1]["bytes"]), item[0][1])
    ):
        action, note = _storage_action(scope_path)
        size_bytes = int(stats["bytes"])
        rows.append(
            {
                "depth": depth,
                "scope_path": scope_path,
                "file_count": int(stats["files"]),
                "total_bytes": size_bytes,
                "total_size_human": _format_bytes(size_bytes),
                "share_of_artifacts_pct": f"{(100.0 * size_bytes / total_bytes):.2f}",
                "suggested_action": action,
                "notes": note,
            }
        )
    return rows


def _load_anchor_rows() -> list[dict[str, str]]:
    with ANCHOR_CATALOG.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sort_key_for_score(row: dict[str, str]) -> tuple[float, float, float, int]:
    return (
        _safe_float(row.get("mean_test_mrr20")) or -1.0,
        _safe_float(row.get("mean_valid_mrr20")) or -1.0,
        _safe_float(row.get("mean_test_hr10")) or -1.0,
        int(row.get("source_file_count") or 0),
    )


def _pair_keep_label(spread: float, runner_up_gap: float) -> str:
    if spread >= 0.03:
        return "keep_best_plus_runner_up"
    if runner_up_gap <= 0.002:
        return "keep_best_plus_close_backup"
    return "keep_best_only"


def _compact_numeric_list(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return min(values), max(values)


def _build_top_hparam_rows(anchor_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not anchor_rows:
        return [], {}

    original_fields = list(anchor_rows[0].keys())
    by_pair: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in anchor_rows:
        key = (row.get("family", ""), row.get("dataset_target", ""), row.get("model", ""))
        by_pair[key].append(row)

    top_rows: list[dict[str, Any]] = []
    dataset_winners: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_stats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sensitive_pairs: list[dict[str, Any]] = []
    route_configs: list[dict[str, Any]] = []

    for (family, dataset_target, model), items in sorted(by_pair.items()):
        ranked = sorted(items, key=_sort_key_for_score, reverse=True)
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        best_score = _safe_float(best.get("mean_test_mrr20")) or 0.0
        runner_score = _safe_float(runner.get("mean_test_mrr20")) if runner else None
        worst_score = _safe_float(ranked[-1].get("mean_test_mrr20")) or best_score
        runner_up_gap = best_score - runner_score if runner_score is not None else 0.0
        spread = best_score - worst_score

        row = dict(best)
        row.update(
            {
                "candidate_count_for_pair": len(ranked),
                "overall_pair_rank": 1,
                "runner_up_stage": "" if runner is None else runner.get("stage", ""),
                "runner_up_mrr20": "" if runner_score is None else f"{runner_score:.10f}",
                "runner_up_gap_mrr20": f"{runner_up_gap:.10f}",
                "pair_mrr20_spread": f"{spread:.10f}",
                "pair_keep_recommendation": _pair_keep_label(spread, runner_up_gap),
            }
        )
        top_rows.append(row)

        dataset_winners[dataset_target].append(
            {
                "family": family,
                "model": model,
                "mean_test_mrr20": best_score,
                "source_stage": best.get("stage", ""),
            }
        )
        model_stats[model].append(
            {
                "dataset_target": dataset_target,
                "family": family,
                "mean_test_mrr20": best_score,
                "pair_mrr20_spread": spread,
                "MAX_ITEM_LIST_LENGTH": best.get("MAX_ITEM_LIST_LENGTH", ""),
                "learning_rate": _safe_float(best.get("learning_rate")),
            }
        )
        sensitive_pairs.append(
            {
                "family": family,
                "dataset_target": dataset_target,
                "model": model,
                "pair_mrr20_spread": spread,
                "best_stage": best.get("stage", ""),
                "best_mrr20": best_score,
                "runner_up_stage": "" if runner is None else runner.get("stage", ""),
                "runner_up_mrr20": runner_score,
            }
        )
        if family == "route":
            route_configs.append(
                {
                    "dataset_target": dataset_target,
                    "model": model,
                    "mean_test_mrr20": best_score,
                    "MAX_ITEM_LIST_LENGTH": best.get("MAX_ITEM_LIST_LENGTH", ""),
                    "learning_rate": best.get("learning_rate", ""),
                    "hidden_size": best.get("hidden_size", ""),
                    "d_feat_emb": best.get("d_feat_emb", ""),
                    "d_router_hidden": best.get("d_router_hidden", ""),
                    "d_ff": best.get("d_ff", ""),
                    "d_expert_hidden": best.get("d_expert_hidden", ""),
                    "expert_scale": best.get("expert_scale", ""),
                    "route_consistency_lambda": best.get("route_consistency_lambda", ""),
                    "stage_feature_dropout_prob": best.get("stage_feature_dropout_prob", ""),
                    "z_loss_lambda": best.get("z_loss_lambda", ""),
                    "router_impl": best.get("router_impl", ""),
                }
            )

    for rows in dataset_winners.values():
        rows.sort(key=lambda item: item["mean_test_mrr20"], reverse=True)

    model_summary = []
    for model, items in sorted(model_stats.items()):
        score_values = [item["mean_test_mrr20"] for item in items]
        spread_values = [item["pair_mrr20_spread"] for item in items]
        lr_values = [item["learning_rate"] for item in items if item["learning_rate"] is not None]
        max_item_counts = Counter(item["MAX_ITEM_LIST_LENGTH"] for item in items if item["MAX_ITEM_LIST_LENGTH"])
        lr_min, lr_max = _compact_numeric_list(lr_values)
        model_summary.append(
            {
                "model": model,
                "pair_count": len(items),
                "avg_best_mrr20": sum(score_values) / len(score_values),
                "avg_pair_spread": sum(spread_values) / len(spread_values),
                "common_max_item_list_length": max_item_counts.most_common(3),
                "learning_rate_min": lr_min,
                "learning_rate_max": lr_max,
            }
        )

    trend_summary = {
        "generated_from_pairs": len(top_rows),
        "dataset_winners": {
            dataset: {
                "winner": rows[0],
                "runner_up": rows[1] if len(rows) > 1 else None,
                "margin_vs_runner_up": None if len(rows) < 2 else rows[0]["mean_test_mrr20"] - rows[1]["mean_test_mrr20"],
            }
            for dataset, rows in sorted(dataset_winners.items())
        },
        "model_summary": model_summary,
        "most_sensitive_pairs": sorted(sensitive_pairs, key=lambda item: item["pair_mrr20_spread"], reverse=True)[:15],
        "route_top_configs": sorted(route_configs, key=lambda item: item["dataset_target"]),
    }

    fieldnames = [
        "candidate_count_for_pair",
        "overall_pair_rank",
        "runner_up_stage",
        "runner_up_mrr20",
        "runner_up_gap_mrr20",
        "pair_mrr20_spread",
        "pair_keep_recommendation",
    ] + original_fields
    trend_summary["fieldnames"] = fieldnames
    return top_rows, trend_summary


def _build_trends_markdown(top_rows: list[dict[str, Any]], trend_summary: dict[str, Any]) -> str:
    lines = [
        "# Artifact Trends",
        "",
        f"- generated_at_utc: {_utc_now()}",
        f"- source_catalog: {ANCHOR_CATALOG}",
        f"- pair_count: {len(top_rows)}",
        f"- summary_csv: {TOP_HPARAMS_CSV}",
        "",
        "## Best By Dataset",
        "",
        "| dataset | winner | family | mrr@20 | runner_up | margin |",
        "|---|---|---|---:|---|---:|",
    ]
    for dataset, payload in sorted(trend_summary["dataset_winners"].items()):
        winner = payload["winner"]
        runner = payload["runner_up"]
        margin = payload["margin_vs_runner_up"]
        lines.append(
            f"| {dataset} | {winner['model']} | {winner['family']} | {winner['mean_test_mrr20']:.4f} | "
            f"{'' if runner is None else runner['model']} | {'' if margin is None else f'{margin:.4f}'} |"
        )

    lines.extend(
        [
            "",
            "## Stable Signals",
            "",
            "- `featured_moe_n3` top configs stay in a tight learning-rate band around `5e-4` to `7.2e-4`, always keep `router_impl=learned`, and mostly stay at `expert_scale=4~5` with small `stage_feature_dropout_prob`.",
            "- `difsr` and `fdsa` best rows consistently keep `category` as the selected feature. `fdsa` also locks to `pooling_mode=mean` in every top row.",
            "- `sasrec` and `bsarec` best rows frequently prefer shorter `MAX_ITEM_LIST_LENGTH` values around `10~20`, while `fdsa` and several `fame` winners skew longer at `30`.",
            "",
            "## Sensitive Pairs",
            "",
            "Pairs with a large gap between the best and worst retained candidates are the ones where future retuning is still likely to matter.",
            "",
            "| dataset | model | family | spread | best_mrr20 | runner_up_mrr20 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in trend_summary["most_sensitive_pairs"][:10]:
        runner_up_text = "" if item["runner_up_mrr20"] is None else f"{item['runner_up_mrr20']:.4f}"
        lines.append(
            f"| {item['dataset_target']} | {item['model']} | {item['family']} | {item['pair_mrr20_spread']:.4f} | "
            f"{item['best_mrr20']:.4f} | {runner_up_text} |"
        )

    lines.extend(
        [
            "",
            "## Model-Level Takeaways",
            "",
        ]
    )
    for item in trend_summary["model_summary"]:
        max_item_text = ", ".join(f"{value}:{count}" for value, count in item["common_max_item_list_length"])
        lr_min = item["learning_rate_min"]
        lr_max = item["learning_rate_max"]
        lr_text = "n/a" if lr_min is None or lr_max is None else f"{lr_min:.6f}~{lr_max:.6f}"
        lines.append(
            f"- `{item['model']}`: avg best `mrr@20={item['avg_best_mrr20']:.4f}`, avg spread `{item['avg_pair_spread']:.4f}`, common `MAX_ITEM_LIST_LENGTH` `{max_item_text or 'n/a'}`, learning-rate band `{lr_text}`."
        )

    if trend_summary["route_top_configs"]:
        lines.extend(
            [
                "",
                "## Route Snapshot",
                "",
                "| dataset | mrr@20 | max_item | lr | hidden | feat_emb | router_hidden | expert_scale | route_lambda | feat_dropout | z_loss |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in trend_summary["route_top_configs"]:
            lines.append(
                f"| {item['dataset_target']} | {item['mean_test_mrr20']:.4f} | {item['MAX_ITEM_LIST_LENGTH']} | {item['learning_rate']} | {item['hidden_size']} | "
                f"{item['d_feat_emb']} | {item['d_router_hidden']} | {item['expert_scale']} | {item['route_consistency_lambda']} | {item['stage_feature_dropout_prob']} | {item['z_loss_lambda']} |"
            )

    return "\n".join(lines) + "\n"


def _build_diag_rows(scan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prefix, stats in sorted(scan["diag_groups"].items(), key=lambda item: (-int(item[1]["bytes"]), item[0])):
        action, note = _diag_action(prefix)
        size_bytes = int(stats["bytes"])
        rows.append(
            {
                "signal_prefix": prefix,
                "signal_kind": stats["kind"],
                "file_count": int(stats["files"]),
                "total_bytes": size_bytes,
                "total_size_human": _format_bytes(size_bytes),
                "largest_file_bytes": int(stats["largest_file_bytes"]),
                "largest_file_size_human": _format_bytes(int(stats["largest_file_bytes"])),
                "largest_file": stats["largest_file"] or "",
                "sample_file": stats["sample_file"] or "",
                "suggested_action": action,
                "notes": note,
            }
        )
    return rows


def _recommendation_groups(storage_rows: list[dict[str, Any]], diag_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    recommendations = {
        "delete_now_if_idle": [],
        "summarized_then_delete": [],
        "index_then_delete": [],
        "keep_subset_only": [],
        "keep": [],
        "review_then_delete": [],
        "keep_diag_index_only": [],
    }

    for row in storage_rows:
        action = row["suggested_action"]
        if action not in recommendations:
            continue
        recommendations[action].append(
            {
                "path": row["scope_path"],
                "bytes": row["total_bytes"],
                "size_human": row["total_size_human"],
                "notes": row["notes"],
            }
        )

    if diag_rows:
        recommendations["keep_diag_index_only"].append(
            {
                "path": "docs/artifact_diagnostic_inventory.csv",
                "bytes": 0,
                "size_human": "0B",
                "notes": f"Indexes {len(diag_rows)} diagnostic or special prefixes so raw trees can be pruned.",
            }
        )

    for items in recommendations.values():
        items.sort(key=lambda item: item["bytes"], reverse=True)
    return recommendations


def _build_summary_markdown(
    scan: dict[str, Any],
    storage_rows: list[dict[str, Any]],
    trend_summary: dict[str, Any],
    diag_rows: list[dict[str, Any]],
    recommendations: dict[str, list[dict[str, Any]]],
) -> str:
    top_storage = [row for row in storage_rows if row["depth"] == 1][:8]
    top_diag = diag_rows[:8]
    top_winners = list(sorted(trend_summary["dataset_winners"].items()))[:6]

    lines = [
        "# Artifact Cleanup Summary",
        "",
        f"- generated_at_utc: {_utc_now()}",
        f"- artifact_root: {ARTIFACT_ROOT}",
        f"- total_files: {scan['total_files']}",
        f"- total_size: {_format_bytes(int(scan['total_bytes']))}",
        "",
        "## Largest Artifact Areas",
        "",
        "| scope_path | size | files | action |",
        "|---|---:|---:|---|",
    ]
    for row in top_storage:
        lines.append(
            f"| {row['scope_path']} | {row['total_size_human']} | {row['file_count']} | {row['suggested_action']} |"
        )

    lines.extend(
        [
            "",
            "## Best-Config Snapshot",
            "",
            f"- summarized_pairs: {trend_summary['generated_from_pairs']}",
            f"- most sensitive pair: {trend_summary['most_sensitive_pairs'][0]['dataset_target']} / {trend_summary['most_sensitive_pairs'][0]['model']} / spread={trend_summary['most_sensitive_pairs'][0]['pair_mrr20_spread']:.4f}"
            if trend_summary.get("most_sensitive_pairs")
            else "- most sensitive pair: unavailable",
            "",
            "| dataset | winner | family | mrr@20 |",
            "|---|---|---|---:|",
        ]
    )
    for dataset, payload in top_winners:
        winner = payload["winner"]
        lines.append(f"| {dataset} | {winner['model']} | {winner['family']} | {winner['mean_test_mrr20']:.4f} |")

    lines.extend(
        [
            "",
            "## Diagnostic Or Special Prefixes",
            "",
            "| signal_prefix | size | files | action |",
            "|---|---:|---:|---|",
        ]
    )
    for row in top_diag:
        lines.append(
            f"| {row['signal_prefix']} | {row['total_size_human']} | {row['file_count']} | {row['suggested_action']} |"
        )

    lines.extend(
        [
            "",
            "## Recommended Cleanup Order",
            "",
            "1. Delete `tmp` and checkpoint-heavy spill only after confirming no resume job is active.",
            "2. Keep docs outputs plus small provenance folders (`inventory`, `timeline`, `analysis`, `reports`).",
            "3. Remove `results/final_experiment` and `results/results_final_experiment_fmoe` only after spot-checking the new best-config CSV and trend note.",
            "4. Prune `logging` and bulky diagnostic trees by representative subset, not by wholesale retention.",
            "",
            "## Structured Outputs",
            "",
            f"- {STORAGE_OVERVIEW_CSV.name}",
            f"- {TOP_HPARAMS_CSV.name}",
            f"- {TRENDS_MD.name}",
            f"- {DIAG_INVENTORY_CSV.name}",
            f"- {CLEANUP_PLAN_JSON.name}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    scan = _scan_artifacts()
    storage_rows = _build_storage_rows(scan)
    anchor_rows = _load_anchor_rows()
    top_rows, trend_summary = _build_top_hparam_rows(anchor_rows)
    diag_rows = _build_diag_rows(scan)
    recommendations = _recommendation_groups(storage_rows, diag_rows)

    _write_csv(
        STORAGE_OVERVIEW_CSV,
        storage_rows,
        [
            "depth",
            "scope_path",
            "file_count",
            "total_bytes",
            "total_size_human",
            "share_of_artifacts_pct",
            "suggested_action",
            "notes",
        ],
    )
    _write_csv(
        TOP_HPARAMS_CSV,
        top_rows,
        trend_summary["fieldnames"],
    )
    _write_csv(
        DIAG_INVENTORY_CSV,
        diag_rows,
        [
            "signal_prefix",
            "signal_kind",
            "file_count",
            "total_bytes",
            "total_size_human",
            "largest_file_bytes",
            "largest_file_size_human",
            "largest_file",
            "sample_file",
            "suggested_action",
            "notes",
        ],
    )
    TRENDS_MD.write_text(_build_trends_markdown(top_rows, trend_summary), encoding="utf-8")

    plan_payload = {
        "generated_at_utc": _utc_now(),
        "artifact_root": str(ARTIFACT_ROOT),
        "anchor_catalog": str(ANCHOR_CATALOG),
        "outputs": {
            "storage_overview_csv": str(STORAGE_OVERVIEW_CSV),
            "top_hparams_csv": str(TOP_HPARAMS_CSV),
            "trends_md": str(TRENDS_MD),
            "diagnostic_inventory_csv": str(DIAG_INVENTORY_CSV),
            "cleanup_summary_md": str(CLEANUP_SUMMARY_MD),
        },
        "artifact_totals": {
            "files": int(scan["total_files"]),
            "bytes": int(scan["total_bytes"]),
            "size_human": _format_bytes(int(scan["total_bytes"])),
        },
        "largest_files": [
            {"path": path, "bytes": size, "size_human": _format_bytes(size)} for size, path in scan["largest_files"]
        ],
        "trend_summary": trend_summary,
        "recommendations": recommendations,
    }
    _write_json(CLEANUP_PLAN_JSON, plan_payload)
    CLEANUP_SUMMARY_MD.write_text(
        _build_summary_markdown(scan, storage_rows, trend_summary, diag_rows, recommendations), encoding="utf-8"
    )

    print(f"wrote {STORAGE_OVERVIEW_CSV}")
    print(f"wrote {TOP_HPARAMS_CSV}")
    print(f"wrote {TRENDS_MD}")
    print(f"wrote {DIAG_INVENTORY_CSV}")
    print(f"wrote {CLEANUP_PLAN_JSON}")
    print(f"wrote {CLEANUP_SUMMARY_MD}")


if __name__ == "__main__":
    main()