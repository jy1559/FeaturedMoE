#!/usr/bin/env python3
"""Shared helpers for the real-final appendix experiment stack."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
APPENDIX_CODE_DIR = Path(__file__).resolve().parent
APPENDIX_WRITING_ROOT = REPO_ROOT / "writing" / "260419_real_final_exp" / "appendix"
APPENDIX_DATA_ROOT = APPENDIX_WRITING_ROOT / "data"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_module(
    "real_final_ablation_base_common",
    REPO_ROOT / "experiments" / "run" / "final_experiment" / "real_final_ablation" / "common.py",
)
_LEGACY = _load_module(
    "real_final_ablation_legacy_common",
    REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "common.py",
)

_BASE.TRACK = "real_final_ablation/appendix"
_BASE.RESULT_ROOT = _BASE.ARTIFACT_ROOT / "results" / _BASE.TRACK
_BASE.LOG_ROOT = _BASE.ARTIFACT_ROOT / "logs" / _BASE.TRACK
_BASE.DATA_ROOT = APPENDIX_DATA_ROOT
_BASE.DEFAULT_BASE_CSV = (
    REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "configs" / "base_candidates.csv"
)
_BASE.QUESTION_AXIS.update(
    {
        "full_results": "appendix_full_results",
        "special_bins": "appendix_special_bins",
        "structural": "appendix_structural",
        "sparse": "appendix_sparse",
        "objective": "appendix_objective",
        "cost": "appendix_cost",
        "diagnostics": "appendix_diagnostics",
        "behavior_slices": "appendix_behavior_slices",
        "cases": "appendix_cases",
        "transfer": "appendix_transfer",
    }
)


def _requires_case_eval(question: str, summary_row: dict[str, Any]) -> bool:
    question_key = str(question or "").strip().lower()
    setting_key = str(summary_row.get("setting_key", "") or "").strip().lower()
    if question_key in {"special_bins", "behavior_slices", "cases", "diagnostics"}:
        return setting_key == "behavior_guided"
    if question_key == "structural":
        return setting_key in {"final_three_stage", "full_semantic", "original_scope"}
    if question_key == "sparse":
        return setting_key in {"group_dense", "group_top2", "global_top4"}
    return False


def _requires_intervention(question: str, summary_row: dict[str, Any]) -> bool:
    question_key = str(question or "").strip().lower()
    setting_key = str(summary_row.get("setting_key", "") or "").strip().lower()
    return question_key == "cases" and setting_key == "behavior_guided"


_BASE._requires_case_eval = _requires_case_eval
_BASE._requires_intervention = _requires_intervention


REPO_ROOT = _BASE.REPO_ROOT
LOG_ROOT = _BASE.LOG_ROOT
RESULT_ROOT = _BASE.RESULT_ROOT
DATA_ROOT = APPENDIX_DATA_ROOT
DEFAULT_DATASETS = list(_BASE.DEFAULT_DATASETS)
DEFAULT_MODELS = list(_BASE.DEFAULT_MODELS)
ROUTE_MODEL = _BASE.ROUTE_MODEL
ROUTE_MODEL_CLASS = _BASE.ROUTE_MODEL_CLASS
ROUTE_MODEL_OVERRIDE = _BASE.ROUTE_MODEL_OVERRIDE
SUMMARY_FIELDS = list(_BASE.SUMMARY_FIELDS)

ensure_dir = _BASE.ensure_dir
python_bin = _BASE.python_bin
parse_csv_list = _BASE.parse_csv_list
parse_csv_ints = _BASE.parse_csv_ints
read_json = _BASE.read_json
write_json = _BASE.write_json
append_csv_row = _BASE.append_csv_row
now_utc = _BASE.now_utc
canonical_stage_maps = _BASE.canonical_stage_maps
validate_session_fixed_files = _BASE.validate_session_fixed_files
common_arg_parser = _BASE.common_arg_parser
selected_candidates_from_args = _BASE.selected_candidates_from_args
build_train_rows = _BASE.build_train_rows
run_jobs = _BASE.run_jobs
read_summary_rows = _BASE.read_summary_rows
write_manifest = _BASE.write_manifest
index_path = _BASE.index_path
summary_path = _BASE.summary_path
manifest_path = _BASE.manifest_path
write_index_rows = _BASE.write_index_rows
latest_manifest_under = _BASE.latest_manifest_under
run_case_eval_pipeline = _BASE.run_case_eval_pipeline
find_completed_case_eval_row = _BASE.find_completed_case_eval_row
find_completed_intervention_row = _BASE.find_completed_intervention_row
load_result_payload = _BASE.load_result_payload
result_summary_from_payload = _BASE.result_summary_from_payload
result_has_successful_trials = _BASE.result_has_successful_trials
build_eval_config_from_result_payload = _BASE.build_eval_config_from_result_payload


def _dedupe_settings(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for setting in group:
            key = str(setting.get("setting_key", "")).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(deepcopy(setting))
    return out


def full_results_settings() -> list[dict[str, Any]]:
    return [
        {
            "setting_key": "behavior_guided",
            "setting_label": "Behavior-Guided",
            "variant_label": "Behavior-guided",
            "variant_order": 1,
            "overrides": canonical_stage_maps(),
        }
    ]


def special_bin_settings() -> list[dict[str, Any]]:
    return full_results_settings()


def structural_settings() -> list[dict[str, Any]]:
    return _dedupe_settings(_BASE.q3_settings(), _LEGACY.a06_settings())


def sparse_settings() -> list[dict[str, Any]]:
    pretty = {
        "dense_global": ("Dense full mixture", 1),
        "group_dense": ("Dense per group", 2),
        "global_top8": ("Flat sparse top-8", 3),
        "global_top4": ("Flat sparse top-4", 4),
        "group_top2": ("Top-3 groups, Top-2 experts", 5),
        "group_top1": ("Top-3 groups, Top-1 expert", 6),
        "global_top2": ("Flat sparse top-2", 7),
    }
    settings: list[dict[str, Any]] = []
    for item in _LEGACY.a07_settings():
        row = deepcopy(item)
        label, order = pretty.get(str(row.get("setting_key", "")), (str(row.get("setting_label", "")), 99))
        row["variant_label"] = label
        row["variant_order"] = order
        row.setdefault("variant_group", "sparse")
        row.setdefault("panel_family", "sparse")
        settings.append(row)
    return settings


def objective_settings() -> list[dict[str, Any]]:
    full = canonical_stage_maps()
    return [
        {
            "setting_key": "full_objective",
            "setting_label": "Full objective",
            "variant_label": "Full objective",
            "variant_order": 1,
            "overrides": deepcopy(full),
        },
        {
            "setting_key": "no_auxiliary",
            "setting_label": "No Auxiliary Loss",
            "variant_label": "No auxiliary loss",
            "variant_order": 2,
            "overrides": {**deepcopy(full), "route_consistency_lambda": 0.0, "z_loss_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "setting_key": "consistency_only",
            "setting_label": "Consistency Only",
            "variant_label": "Consistency only",
            "variant_order": 3,
            "overrides": {**deepcopy(full), "z_loss_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "setting_key": "zloss_only",
            "setting_label": "Z-Loss Only",
            "variant_label": "z-loss only",
            "variant_order": 4,
            "overrides": {**deepcopy(full), "route_consistency_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "setting_key": "balance_only",
            "setting_label": "Balance Only",
            "variant_label": "Balance only",
            "variant_order": 5,
            "overrides": {**deepcopy(full), "route_consistency_lambda": 0.0, "z_loss_lambda": 0.0},
        },
        {
            "setting_key": "consistency_plus_zloss",
            "setting_label": "Consistency + z-Loss",
            "variant_label": "Consistency + z-loss",
            "variant_order": 6,
            "overrides": {**deepcopy(full), "balance_loss_lambda": 0.0},
        },
    ]


def behavior_slice_settings() -> list[dict[str, Any]]:
    return full_results_settings()


def cases_settings() -> list[dict[str, Any]]:
    return full_results_settings()


def transfer_settings() -> list[dict[str, Any]]:
    return _LEGACY.a09_low_data_settings()


def extended_intervention_specs() -> list[dict[str, Any]]:
    return _LEGACY.a10_intervention_specs()


def select_postprocess_rows(ok_rows: list[dict[str, str]], *, postprocess_all: bool) -> list[dict[str, str]]:
    if postprocess_all:
        return ok_rows
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in ok_rows:
        by_dataset.setdefault(str(row.get("dataset", "")).strip(), []).append(row)

    def _as_int(value: str, default: int = 10**9) -> int:
        try:
            return int(str(value).strip())
        except Exception:
            return default

    selected: list[dict[str, str]] = []
    for dataset_rows in by_dataset.values():
        picked = sorted(
            dataset_rows,
            key=lambda row: (_as_int(row.get("base_rank", "")), _as_int(row.get("seed_id", ""))),
        )
        if picked:
            selected.append(picked[0])
    return selected


def run_interventions(
    *,
    question: str,
    summary_row: dict[str, str],
    output_root: Path,
) -> dict[str, Any]:
    cmd = [
        python_bin(),
        str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "eval_checkpoint_interventions.py"),
        "--source-result-json",
        str(summary_row["result_path"]),
        "--checkpoint-file",
        str(summary_row["checkpoint_file"]),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    manifest = latest_manifest_under(output_root, "intervention_manifest.csv")
    return {
        "question": question,
        "dataset": summary_row.get("dataset", ""),
        "setting_key": summary_row.get("setting_key", ""),
        "setting_label": summary_row.get("setting_label", ""),
        "base_rank": summary_row.get("base_rank", ""),
        "base_tag": summary_row.get("base_tag", ""),
        "seed_id": summary_row.get("seed_id", ""),
        "result_path": summary_row.get("result_path", ""),
        "checkpoint_file": summary_row.get("checkpoint_file", ""),
        "intervention_manifest": str(manifest),
        "status": "ok",
        "error": "",
    }


def write_note_manifest(question: str, note: str, *, extra: dict[str, Any] | None = None) -> Path:
    payload = {"generated_at": now_utc(), "question": question, "note": note}
    if extra:
        payload.update(extra)
    path = manifest_path(question)
    write_json(path, payload)
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    import csv

    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames_hint: list[str] | None = None) -> None:
    import csv

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


def load_selected_space_metadata() -> dict[str, Path]:
    base = REPO_ROOT / "experiments" / "run" / "final_experiment"
    return {
        "selected_configs_csv": base / "selected_configs.csv",
        "selected_configs_json": base / "selected_configs.json",
        "tuning_space_csv": base / "tuning_space.csv",
    }


def notebook_data_dir() -> Path:
    return ensure_dir(APPENDIX_DATA_ROOT)


def notebook_root() -> Path:
    return APPENDIX_WRITING_ROOT


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
