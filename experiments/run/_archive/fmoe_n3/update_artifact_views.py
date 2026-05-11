#!/usr/bin/env python3
"""Build compact artifact-level summaries for FMoE_N3 axes."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/workspace/jy1559/FMoE")
ARTIFACT_ROOT = REPO_ROOT / "experiments" / "run" / "artifacts"
RESULT_ROOT = ARTIFACT_ROOT / "results" / "fmoe_n3"
LOG_ROOT = ARTIFACT_ROOT / "logs" / "fmoe_n3"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_artifact_path(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    fixed = text.replace("/artifacts/logs/fmoe_n3/fmoe_n3/", "/artifacts/logs/fmoe_n3/")
    return fixed


def _flatten(prefix: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        col = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(col, value))
        elif isinstance(value, list):
            out[col] = json.dumps(value, ensure_ascii=False)
        else:
            out[col] = value
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["note"]
        rows = [{"note": "no rows"}]
    else:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _latest_and_best_runs(axis: str) -> tuple[dict[tuple[str, str, str], dict], dict[tuple[str, str, str], dict], dict[tuple[str, str, str], int]]:
    latest: dict[tuple[str, str, str], dict] = {}
    best: dict[tuple[str, str, str], dict] = {}
    counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for path in sorted(RESULT_ROOT.glob("*.json")):
        payload = _load_json(path)
        if not payload:
            continue
        if str(payload.get("run_group", "")) != "fmoe_n3":
            continue
        if str(payload.get("run_axis", "")) != axis:
            continue
        combo_id = str(payload.get("run_phase", "") or "").strip()
        dataset = str(payload.get("dataset", "") or "").strip()
        model = str(payload.get("model", "") or "").strip()
        if not combo_id or not dataset or not model:
            continue
        key = (combo_id, dataset, model)
        counts[key] += 1
        row = {
            "path": path,
            "mtime": float(path.stat().st_mtime),
            "payload": payload,
        }
        if key not in latest or row["mtime"] > latest[key]["mtime"]:
            latest[key] = row
        current_best = _safe_float(payload.get("best_mrr@20"))
        prev_best = _safe_float((best.get(key) or {}).get("payload", {}).get("best_mrr@20"))
        if key not in best or (current_best is not None and (prev_best is None or current_best > prev_best)):
            best[key] = row

    return latest, best, counts


def build_special_summary(axis: str) -> Path:
    latest, best, counts = _latest_and_best_runs(axis)
    rows: list[dict[str, Any]] = []
    for key in sorted(latest.keys()):
        combo_id, dataset, model = key
        payload = latest[key]["payload"]
        best_payload = best[key]["payload"]
        row = {
            "combo_id": combo_id,
            "dataset": dataset,
            "model": model,
            "run_count": counts[key],
            "latest_result_file": str(latest[key]["path"].resolve()),
            "latest_special_log_file": _normalize_artifact_path(payload.get("special_log_file", "")),
            "latest_special_result_file": _normalize_artifact_path(payload.get("special_result_file", "")),
            "latest_best_mrr20": payload.get("best_mrr@20", ""),
            "latest_test_mrr20": payload.get("test_mrr@20", ""),
            "best_mrr20_seen": best_payload.get("best_mrr@20", ""),
            "best_run_result_file": str(best[key]["path"].resolve()),
        }
        row.update(_flatten("valid.overall", ((payload.get("best_valid_special_metrics") or {}).get("overall") or {})))
        row.update(_flatten("early_valid.overall", ((payload.get("early_valid_special_metrics") or {}).get("overall") or {})))
        row.update(_flatten("test.overall", ((payload.get("test_special_metrics") or {}).get("overall") or {})))
        rows.append(row)

    out_path = LOG_ROOT / axis / f"{axis}_special_summary.csv"
    _write_csv(out_path, rows)
    return out_path


def build_feature_ablation_summary(axis: str) -> Path:
    latest, best, counts = _latest_and_best_runs(axis)
    rows: list[dict[str, Any]] = []
    for key in sorted(latest.keys()):
        combo_id, dataset, model = key
        payload = latest[key]["payload"]
        best_payload = best[key]["payload"]
        row = {
            "combo_id": combo_id,
            "dataset": dataset,
            "model": model,
            "run_count": counts[key],
            "latest_result_file": str(latest[key]["path"].resolve()),
            "latest_best_mrr20": payload.get("best_mrr@20", ""),
            "latest_test_mrr20": payload.get("test_mrr@20", ""),
            "best_mrr20_seen": best_payload.get("best_mrr@20", ""),
            "best_run_result_file": str(best[key]["path"].resolve()),
        }
        row.update(payload.get("feature_ablation_metrics") or {})
        rows.append(row)

    out_path = LOG_ROOT / axis / f"{axis}_feature_ablation_summary.csv"
    _write_csv(out_path, rows)
    return out_path


def build_diag_summary(axis: str) -> Path:
    diag_root = RESULT_ROOT / "diag" / axis
    rows: list[dict[str, Any]] = []
    if diag_root.exists():
        for trial_summary_path in sorted(diag_root.glob("*/*/*/trial_summary.csv")):
            combo_id = trial_summary_path.parts[-4]
            dataset = trial_summary_path.parts[-3]
            model = trial_summary_path.parts[-2]
            with trial_summary_path.open(encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                trial_rows = list(reader)
            if not trial_rows:
                continue
            best_row = max(trial_rows, key=lambda row: _safe_float(row.get("mrr@20")) or float("-inf"))
            base_dir = trial_summary_path.parent
            row = {
                "combo_id": combo_id,
                "dataset": dataset,
                "model": model,
                "trial_count": len(trial_rows),
                "trial_summary_file": str(trial_summary_path.resolve()),
                "best_valid_diag_file": str((base_dir / "best_valid_diag.json.gz").resolve()) if (base_dir / "best_valid_diag.json.gz").exists() else "",
                "best_valid_overview_file": str((base_dir / "best_valid_overview.json").resolve()) if (base_dir / "best_valid_overview.json").exists() else "",
                "best_valid_overview_md_file": str((base_dir / "best_valid_overview.md").resolve()) if (base_dir / "best_valid_overview.md").exists() else "",
                "early_valid_diag_file": str((base_dir / "early_valid_diag.json.gz").resolve()) if (base_dir / "early_valid_diag.json.gz").exists() else "",
                "test_diag_file": str((base_dir / "test_diag.json.gz").resolve()) if (base_dir / "test_diag.json.gz").exists() else "",
                "collapse_diag_file": str((base_dir / "collapse_diag.json.gz").resolve()) if (base_dir / "collapse_diag.json.gz").exists() else "",
                "epoch_trace_file": str((base_dir / "epoch_trace.csv.gz").resolve()) if (base_dir / "epoch_trace.csv.gz").exists() else "",
            }
            row.update(best_row)
            rows.append(row)

    out_path = LOG_ROOT / axis / f"{axis}_diag_summary.csv"
    _write_csv(out_path, rows)
    return out_path


def build_diag_readable_summary(axis: str) -> Path:
    diag_root = RESULT_ROOT / "diag" / axis
    rows: list[dict[str, Any]] = []
    if diag_root.exists():
        for trial_summary_path in sorted(diag_root.glob("*/*/*/trial_summary.csv")):
            combo_id = trial_summary_path.parts[-4]
            dataset = trial_summary_path.parts[-3]
            model = trial_summary_path.parts[-2]
            with trial_summary_path.open(encoding="utf-8", newline="") as fp:
                trial_rows = list(csv.DictReader(fp))
            if not trial_rows:
                continue

            best_row = max(trial_rows, key=lambda row: _safe_float(row.get("mrr@20")) or float("-inf"))

            stage_prefixes = sorted(
                {
                    key.rsplit(".", 1)[0]
                    for key in best_row.keys()
                    if "." in key and key.rsplit(".", 1)[-1] in {
                        "route_jitter_adjacent",
                        "route_consistency_knn_js",
                        "route_consistency_knn_score",
                        "n_eff",
                        "cv_usage",
                        "top1_max_frac",
                        "entropy_mean",
                    }
                }
            )

            row: dict[str, Any] = {
                "combo_id": combo_id,
                "dataset": dataset,
                "model": model,
                "trial": best_row.get("trial", ""),
                "best_mrr@20": best_row.get("mrr@20", ""),
                "best_test_mrr@20": best_row.get("test_mrr@20", ""),
                "best_test_hr@10": best_row.get("test_hr@10", ""),
            }

            for prefix in stage_prefixes:
                for metric in (
                    "n_eff",
                    "cv_usage",
                    "top1_max_frac",
                    "entropy_mean",
                    "route_jitter_adjacent",
                    "route_consistency_knn_js",
                    "route_consistency_knn_score",
                ):
                    key = f"{prefix}.{metric}"
                    if key in best_row:
                        row[key] = best_row.get(key, "")

            rows.append(row)

    out_path = LOG_ROOT / axis / f"{axis}_diag_readable_summary.csv"
    _write_csv(out_path, rows)
    return out_path


def build_artifact_views(axis: str) -> dict[str, Path]:
    return {
        "special_summary": build_special_summary(axis),
        "feature_ablation_summary": build_feature_ablation_summary(axis),
        "diag_summary": build_diag_summary(axis),
        "diag_readable_summary": build_diag_readable_summary(axis),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build compact FMoE_N3 artifact summaries.")
    parser.add_argument("--axis", default="core_ablation_v2")
    args = parser.parse_args()
    paths = build_artifact_views(args.axis)
    for key, path in paths.items():
        print(f"[{key}] {path}")


if __name__ == "__main__":
    main()
