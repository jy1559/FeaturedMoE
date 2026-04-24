#!/usr/bin/env python3
"""Export A06~A10 ablation outputs into notebook-friendly CSV bundles."""

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

from common import DATA_ROOT, LOG_ROOT, RESULT_ROOT  # noqa: E402

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
    if not rows:
        return
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
    return float(sum(values) / len(values)) if values else 0.0


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
    valid_special = payload.get("best_valid_special_metrics") or {}
    test_special = payload.get("test_special_metrics") or {}
    valid_seen = _special_block(valid_special, "overall_seen_target")
    test_seen = _special_block(test_special, "overall_seen_target")
    row = dict(summary_row)
    row.update(
        {
            "best_valid_mean": _metric_mean(valid),
            "test_mean": _metric_mean(test),
            "best_valid_seen_mean": _metric_mean(valid_seen),
            "test_seen_mean": _metric_mean(test_seen),
            "best_valid_seen_mrr20": _safe_float(valid_seen.get("mrr@20")),
            "test_seen_mrr20": _safe_float(test_seen.get("mrr@20")),
            "best_valid_mrr20": _safe_float(valid.get("mrr@20")),
            "test_mrr20": _safe_float(test.get("mrr@20")),
            "test_seen_hit10": _safe_float(test_seen.get("hit@10")),
            "test_seen_ndcg20": _safe_float(test_seen.get("ndcg@20")),
        }
    )
    return row


def _collect_results(question: str) -> list[dict[str, Any]]:
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
    rows.sort(key=lambda r: (str(r.get("dataset", "")), str(r.get("setting_key", "")), int(r.get("base_rank", 0) or 0)))
    return rows


def _router_diag_from_result(result_path: str) -> dict[str, Any]:
    """Extract router diagnostics from a result payload's diag files."""
    payload = _load_json(result_path)
    diag_meta = str(payload.get("diag_meta_file", "") or "").strip()
    if not diag_meta or not Path(diag_meta).exists():
        return {}
    return _load_json(diag_meta)


def export_a06(output_dir: Path) -> None:
    """Export A06 semantic and scope variants."""
    rows = _collect_results("a06")
    semantic_keys = {"full_semantic", "reduced_family", "shuffled_family", "flat_random"}
    scope_keys = {"original_scope", "identical_scope", "scope_swap", "extra_attn"}

    semantic_rows = [r for r in rows if r.get("setting_key") in semantic_keys]
    scope_rows = [r for r in rows if r.get("setting_key") in scope_keys]

    _write_csv(output_dir / "A06_semantic_variants.csv", semantic_rows)
    _write_csv(output_dir / "A06_scope_layout_variants.csv", scope_rows)
    print(f"[export-a06] {len(semantic_rows)} semantic + {len(scope_rows)} scope rows")


def export_a07(output_dir: Path) -> None:
    """Export A07 top-k routing results + router diagnostics."""
    rows = _collect_results("a07")
    tradeoff_rows = []
    sweep_rows = []
    for r in rows:
        setting = str(r.get("setting_key", ""))
        # Extract router diagnostics for active-expert counts
        result_path = str(r.get("result_path", "")).strip()
        diag = {}
        if result_path and Path(result_path).exists():
            try:
                diag = _router_diag_from_result(result_path)
            except Exception:
                pass
        test_diag = diag.get("test", {}) if isinstance(diag, dict) else {}
        stage_metrics = test_diag.get("stage_metrics", {}) if isinstance(test_diag, dict) else {}
        avg_n_eff = 0.0
        n_stages = 0
        for stage_key, sm in stage_metrics.items():
            gr = sm.get("group_routing", {}) if isinstance(sm, dict) else {}
            n_eff = _safe_float(gr.get("group_n_eff"))
            if n_eff > 0:
                avg_n_eff += n_eff
                n_stages += 1
        if n_stages > 0:
            avg_n_eff /= n_stages

        entry = {
            **r,
            "avg_n_eff": avg_n_eff,
        }
        tradeoff_rows.append(entry)
        sweep_rows.append(entry)

    _write_csv(output_dir / "A07_topk_tradeoff.csv", tradeoff_rows)
    _write_csv(output_dir / "A07_topk_sweep.csv", sweep_rows)
    print(f"[export-a07] {len(tradeoff_rows)} top-k rows")


def export_a08(output_dir: Path) -> None:
    """Export A08 behavior case study results."""
    index_rows = _read_csv(LOG_ROOT / "a08" / "a08_case_eval_index.csv")
    slice_quality_rows = []
    slice_alignment_rows = []
    slice_profile_rows = []

    for idx_row in index_rows:
        if str(idx_row.get("status", "")).lower() != "ok":
            continue
        if _DATASETS_FILTER and str(idx_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        export_dir = str(idx_row.get("case_eval_export_dir", "")).strip()
        if not export_dir or not Path(export_dir).exists():
            continue
        # Read exported case tables
        for csv_name, target_list in [
            ("case_quality_summary.csv", slice_quality_rows),
            ("case_alignment.csv", slice_alignment_rows),
            ("case_feature_profiles.csv", slice_profile_rows),
        ]:
            target_path = Path(export_dir) / csv_name
            if target_path.exists():
                for row in _read_csv(target_path):
                    merged = dict(idx_row)
                    merged.update(row)
                    target_list.append(merged)

    _write_csv(output_dir / "A08_dataset_slice_quality.csv", slice_quality_rows)
    _write_csv(output_dir / "A08_slice_alignment.csv", slice_alignment_rows)
    _write_csv(output_dir / "A08_slice_feature_profiles.csv", slice_profile_rows)
    print(f"[export-a08] quality={len(slice_quality_rows)} align={len(slice_alignment_rows)} profiles={len(slice_profile_rows)}")


def export_a09(output_dir: Path) -> None:
    """Export A09 low-data and transfer results."""
    # Low-data results from main summary
    all_rows = _collect_results("a09")
    low_data_rows = [r for r in all_rows if str(r.get("setting_key", "")).startswith("data_frac_")]
    _write_csv(output_dir / "A09_low_resource_transfer.csv", low_data_rows)

    # Transfer results from transfer_summary
    transfer_summary = LOG_ROOT / "a09" / "transfer_summary.csv"
    transfer_rows = []
    for summary_row in _read_csv(transfer_summary):
        if str(summary_row.get("status", "")).lower() != "ok":
            continue
        if _DATASETS_FILTER and str(summary_row.get("target_dataset", summary_row.get("dataset", ""))).split("_to_")[-1] not in _DATASETS_FILTER and str(summary_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        row = _summary_to_result_row(summary_row)
        if row is not None:
            transfer_rows.append(row)
    _write_csv(output_dir / "A09_transfer_variants.csv", transfer_rows)
    print(f"[export-a09] low_data={len(low_data_rows)} transfer={len(transfer_rows)}")


def export_a10(output_dir: Path) -> None:
    """Export A10 cue semantics: cue profiles + intervention results."""
    all_rows = _collect_results("a10")
    _write_csv(output_dir / "A10_cue_profile_variants.csv", all_rows)

    # Intervention results
    intervention_index = _read_csv(LOG_ROOT / "a10" / "a10_intervention_index.csv")
    intervention_perf_rows = []
    intervention_shift_rows = []

    for idx_row in intervention_index:
        if str(idx_row.get("status", "")).lower() != "ok":
            continue
        if _DATASETS_FILTER and str(idx_row.get("dataset", "")) not in _DATASETS_FILTER:
            continue
        manifest_path = str(idx_row.get("intervention_manifest", "")).strip()
        if not manifest_path or not Path(manifest_path).exists():
            continue
        for mrow in _read_csv(Path(manifest_path)):
            if str(mrow.get("status", "")).lower() != "ok":
                continue
            result_file = str(mrow.get("result_file", "")).strip()
            if not result_file or not Path(result_file).exists():
                continue
            payload = _load_json(result_file)
            test = payload.get("test_result") or {}
            test_special = payload.get("test_special_metrics") or {}
            test_seen = _special_block(test_special, "overall_seen_target")
            perf = {
                **idx_row,
                **mrow,
                "test_seen_mrr20": _safe_float(test_seen.get("mrr@20")),
                "test_mrr20": _safe_float(test.get("mrr@20")),
            }
            intervention_perf_rows.append(perf)

            # Router shift
            router_file = str(mrow.get("router_diag_file", "")).strip()
            if router_file and Path(router_file).exists():
                try:
                    router_payload = _load_json(router_file)
                    for split_name in ("test",):
                        stage_metrics = ((router_payload.get(split_name) or {}).get("stage_metrics") or {})
                        for stage_key, sm in stage_metrics.items():
                            gr = sm.get("group_routing", {}) if isinstance(sm, dict) else {}
                            group_names = list(gr.get("group_names") or [])
                            group_share = list(gr.get("group_share") or [])
                            for family, share in zip(group_names, group_share):
                                intervention_shift_rows.append(
                                    {
                                        **idx_row,
                                        **mrow,
                                        "eval_split": split_name,
                                        "stage_key": stage_key,
                                        "expert_group": family,
                                        "mass": _safe_float(share),
                                    }
                                )
                except Exception:
                    pass

    _write_csv(output_dir / "A10_intervention_scores.csv", intervention_perf_rows)
    _write_csv(output_dir / "A10_intervention_shift.csv", intervention_shift_rows)
    print(f"[export-a10] profiles={len(all_rows)} interventions={len(intervention_perf_rows)} shifts={len(intervention_shift_rows)}")


def main() -> int:
    global _DATASETS_FILTER
    parser = argparse.ArgumentParser(description="Export A06~A10 appendix ablation bundles.")
    parser.add_argument("--output-dir", default=str(DATA_ROOT))
    parser.add_argument("--questions", default="a06,a07,a08,a09,a10")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter (empty = all).")
    args, _unknown = parser.parse_known_args()
    _DATASETS_FILTER = {d.strip() for d in str(args.datasets).split(",") if d.strip()}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = {q.strip() for q in str(args.questions).split(",") if q.strip()}

    exporters = {
        "a06": export_a06,
        "a07": export_a07,
        "a08": export_a08,
        "a09": export_a09,
        "a10": export_a10,
    }
    for q, fn in exporters.items():
        if q in questions:
            print(f"[export] {q}...")
            fn(output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
