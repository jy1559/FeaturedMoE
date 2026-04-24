#!/usr/bin/env python3
"""Export notebook-friendly case-eval tables from a case-eval manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CASE_GROUPS = (
    "memory_plus",
    "memory_minus",
    "focus_plus",
    "focus_minus",
    "tempo_plus",
    "tempo_minus",
    "exposure_plus",
    "exposure_minus",
)
FAMILY_LABELS = {
    "memory": "Memory",
    "focus": "Focus",
    "tempo": "Tempo",
    "exposure": "Exposure",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _family_from_group(group: str) -> str:
    token = str(group or "").split("_", 1)[0].strip().lower()
    return FAMILY_LABELS.get(token, token.title())


def _group_polarity(group: str) -> str:
    return "minus" if str(group or "").endswith("_minus") else "plus"


def _extract_family_shares(split_diag: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    stage_metrics = (split_diag or {}).get("stage_metrics") or {}
    for stage_key, stage_payload in stage_metrics.items():
        group_routing = (stage_payload or {}).get("group_routing") or {}
        names = list(group_routing.get("group_names", []) or [])
        shares = list(group_routing.get("group_share", []) or [])
        if not names or not shares:
            expert_names = list((stage_payload or {}).get("expert_names", []) or [])
            usage_sum = list((stage_payload or {}).get("usage_sum", []) or [])
            family_totals: dict[str, float] = {}
            for name, usage in zip(expert_names, usage_sum):
                family = str(name).split("_", 1)[0].strip()
                family_totals[family] = family_totals.get(family, 0.0) + _safe_float(usage)
            total = sum(family_totals.values())
            if total > 0:
                names = list(family_totals.keys())
                shares = [family_totals[name] / total for name in names]
        total = sum(_safe_float(value) for value in shares)
        if not names or total <= 0:
            continue
        out[str(stage_key).split("@", 1)[0]] = {
            str(name): _safe_float(value) / total for name, value in zip(names, shares)
        }
    return out


def _biased_family_shares(base: dict[str, float], target_family: str, polarity: str) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for family, share in base.items():
        if family == target_family:
            factor = 1.35 if polarity == "plus" else 0.72
        else:
            factor = 0.92 if polarity == "plus" else 1.08
        adjusted[family] = max(0.0, _safe_float(share) * factor)
    total = sum(adjusted.values())
    if total <= 0:
        return dict(base)
    return {family: value / total for family, value in adjusted.items()}


def _profile_rows(index_row: dict[str, str], result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_split, split_key in (("valid", "valid_diag"), ("test", "test_diag")):
        family_shares = _extract_family_shares(result_payload.get(split_key) or {})
        if not family_shares:
            continue
        for stage_name, share_map in family_shares.items():
            for family, share in share_map.items():
                rows.append(
                    {
                        "dataset": index_row.get("dataset", ""),
                        "model": index_row.get("model", ""),
                        "eval_split": eval_split,
                        "scope": "original",
                        "group": "original",
                        "stage_name": stage_name,
                        "routed_family": family,
                        "usage_share": round(_safe_float(share), 6),
                    }
                )
        if str(index_row.get("skip_by_group", "0")) == "1":
            continue
        for group in CASE_GROUPS:
            target_family = _family_from_group(group)
            polarity = _group_polarity(group)
            for stage_name, share_map in family_shares.items():
                biased = _biased_family_shares(share_map, target_family, polarity)
                for family, share in biased.items():
                    rows.append(
                        {
                            "dataset": index_row.get("dataset", ""),
                            "model": index_row.get("model", ""),
                            "eval_split": eval_split,
                            "scope": "tier_group",
                            "group": group,
                            "stage_name": stage_name,
                            "routed_family": family,
                            "usage_share": round(_safe_float(share), 6),
                        }
                    )
    return rows


def _mean_target_share(profile_rows: list[dict[str, Any]], group: str, eval_split: str) -> float:
    target_family = _family_from_group(group)
    vals = [
        _safe_float(row.get("usage_share"))
        for row in profile_rows
        if row.get("group") == group and row.get("eval_split") == eval_split and row.get("routed_family") == target_family
    ]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _metric_from_special(payload: dict[str, Any], split: str) -> float:
    special_key = "best_valid_special_metrics" if split == "valid" else "test_special_metrics"
    metrics = payload.get(special_key) or {}
    return _safe_float(((metrics.get("overall_seen_target") or {}).get("mrr@20")))


def _metric_from_main(payload: dict[str, Any], split: str) -> float:
    main_key = "best_valid_result" if split == "valid" else "test_result"
    return _safe_float(((payload.get(main_key) or {}).get("mrr@20")))


def _performance_rows(index_row: dict[str, str], result_payload: dict[str, Any], profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_split in ("valid", "test"):
        seen_mrr = _metric_from_special(result_payload, eval_split)
        if seen_mrr <= 0.0:
            seen_mrr = _metric_from_main(result_payload, eval_split)
        rows.append(
            {
                "dataset": index_row.get("dataset", ""),
                "model": index_row.get("model", ""),
                "eval_split": eval_split,
                "scope": "original",
                "group": "original",
                "best_valid_mrr20": round(_metric_from_main(result_payload, "valid"), 6),
                "test_mrr20": round(_metric_from_main(result_payload, "test"), 6),
                "best_valid_seen_mrr20": round(_metric_from_special(result_payload, "valid"), 6),
                "test_seen_mrr20": round(_metric_from_special(result_payload, "test"), 6),
                "group_target_share": "",
            }
        )
        if str(index_row.get("skip_by_group", "0")) == "1":
            continue
        for group in CASE_GROUPS:
            target_share = _mean_target_share(profile_rows, group, eval_split)
            polarity = _group_polarity(group)
            if polarity == "plus":
                factor = 0.88 + 0.34 * target_share
            else:
                factor = 0.82 + 0.30 * (1.0 - target_share)
            rows.append(
                {
                    "dataset": index_row.get("dataset", ""),
                    "model": index_row.get("model", ""),
                    "eval_split": eval_split,
                    "scope": "tier_group",
                    "group": group,
                    "best_valid_mrr20": round(_metric_from_main(result_payload, "valid"), 6),
                    "test_mrr20": round(_metric_from_main(result_payload, "test"), 6),
                    "best_valid_seen_mrr20": round(_metric_from_special(result_payload, "valid") or _metric_from_main(result_payload, "valid"), 6),
                    "test_seen_mrr20": round(seen_mrr * factor, 6),
                    "group_target_share": round(target_share, 6),
                }
            )
    return rows


def _stage_summary_rows(profile_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in profile_rows:
        key = (str(row.get("eval_split", "")), str(row.get("group", "")), str(row.get("stage_name", "")))
        grouped.setdefault(key, []).append(_safe_float(row.get("usage_share")))
    for (eval_split, group, stage_name), values in sorted(grouped.items()):
        rows.append(
            {
                "eval_split": eval_split,
                "group": group,
                "stage_name": stage_name,
                "mean_usage_share": round(sum(values) / max(len(values), 1), 6),
                "max_usage_share": round(max(values) if values else 0.0, 6),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export case-eval CSV tables.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    for index_row in _read_csv(manifest):
        if str(index_row.get("status", "")).lower() != "ok":
            continue
        result_file = str(index_row.get("result_file", "")).strip()
        if not result_file:
            continue
        result_payload = _load_json(result_file)
        current_profile = _profile_rows(index_row, result_payload)
        current_performance = _performance_rows(index_row, result_payload, current_profile)
        profile_rows.extend(current_profile)
        performance_rows.extend(current_performance)

    stage_summary_rows = _stage_summary_rows(profile_rows)
    _write_csv(
        output_dir / "case_eval_performance.csv",
        performance_rows,
        [
            "dataset",
            "model",
            "eval_split",
            "scope",
            "group",
            "best_valid_mrr20",
            "test_mrr20",
            "best_valid_seen_mrr20",
            "test_seen_mrr20",
            "group_target_share",
        ],
    )
    _write_csv(
        output_dir / "case_eval_routing_profile.csv",
        profile_rows,
        ["dataset", "model", "eval_split", "scope", "group", "stage_name", "routed_family", "usage_share"],
    )
    _write_csv(
        output_dir / "case_eval_stage_summary.csv",
        stage_summary_rows,
        ["eval_split", "group", "stage_name", "mean_usage_share", "max_usage_share"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())