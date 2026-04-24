#!/usr/bin/env python3
"""Shared helpers for the real-final appendix experiment stack."""

from __future__ import annotations

import argparse
import csv
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
_FINAL = _load_module(
    "real_final_ablation_final_common",
    REPO_ROOT / "experiments" / "run" / "final_experiment" / "common.py",
)

_BASE.TRACK = "real_final_ablation/appendix"
_BASE.RESULT_ROOT = _BASE.ARTIFACT_ROOT / "results" / _BASE.TRACK
_BASE.LOG_ROOT = _BASE.ARTIFACT_ROOT / "logs" / _BASE.TRACK
_BASE.DATA_ROOT = APPENDIX_DATA_ROOT
_FINAL.TRACK = _BASE.TRACK
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
_route_command_impl = _BASE.build_route_command


BASELINE_COMPARE_MODELS = ("sasrec",)
BASELINE_MODEL_LABELS = {
    "sasrec": "SASRec",
    "fame": "FAME",
}


def parse_baseline_models(raw: str) -> list[str]:
    models = [str(token).strip().lower() for token in str(raw or "").split(",") if str(token).strip()]
    out: list[str] = []
    for model in models:
        if model not in BASELINE_MODEL_LABELS:
            raise ValueError(f"Unsupported appendix baseline model: {model}")
        if model not in out:
            out.append(model)
    return out


def _load_tuning_space_rows() -> list[dict[str, str]]:
    path = load_selected_space_metadata()["tuning_space_csv"]
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_values_json(raw: str) -> list[Any]:
    try:
        values = json.loads(str(raw))
    except Exception:
        return []
    return values if isinstance(values, list) else []


def _nearest_option(options: list[Any], target: Any, *, default: Any) -> Any:
    numeric: list[float] = []
    cast_map: dict[float, Any] = {}
    for value in options:
        try:
            num = float(value)
        except Exception:
            continue
        numeric.append(num)
        cast_map[num] = value
    if not numeric:
        return default
    try:
        target_num = float(target)
    except Exception:
        target_num = float(numeric[len(numeric) // 2])
    best = min(numeric, key=lambda value: (abs(value - target_num), value))
    return cast_map[best]


def _median_option(options: list[Any], default: Any) -> Any:
    usable = list(options)
    if not usable:
        return default
    return usable[len(usable) // 2]


def _baseline_space_lookup(dataset: str, model: str) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in _load_tuning_space_rows():
        if str(row.get("dataset", "")).strip() != str(dataset).strip():
            continue
        if str(row.get("model", "")).strip().lower() != str(model).strip().lower():
            continue
        values = _parse_values_json(str(row.get("values_json", "")))
        if values:
            out[str(row.get("param", "")).strip()] = values
    return out


def _baseline_capacity_anchor(candidate: _BASE.BaseCandidate) -> dict[str, Any]:
    cfg = dict(candidate.base_config or {})
    hidden = int(cfg.get("hidden_size", cfg.get("embedding_size", 128)) or 128)
    embed = int(cfg.get("embedding_size", cfg.get("hidden_size", 128)) or 128)
    max_len = int(cfg.get("MAX_ITEM_LIST_LENGTH", 20) or 20)
    dropout = float(
        cfg.get(
            "hidden_dropout_prob",
            cfg.get("attn_dropout_prob", cfg.get("dropout_ratio", 0.1)),
        )
        or 0.1
    )
    return {
        "hidden_size": hidden,
        "embedding_size": embed,
        "MAX_ITEM_LIST_LENGTH": max_len,
        "dropout": max(0.05, min(0.2, dropout)),
    }


def build_minimal_baseline_rows(
    *,
    question: str,
    candidates: list[_BASE.BaseCandidate],
    baseline_models: list[str],
    seeds: list[int],
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
    smoke_test: bool,
    smoke_max_runs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for candidate in candidates:
        anchor = _baseline_capacity_anchor(candidate)
        for model in baseline_models:
            space = _baseline_space_lookup(candidate.dataset, model)
            hidden = int(_nearest_option(space.get("hidden_size", []), anchor["hidden_size"], default=anchor["hidden_size"]))
            embedding = int(_nearest_option(space.get("embedding_size", [hidden]), anchor["embedding_size"], default=hidden))
            max_len = int(_nearest_option(space.get("MAX_ITEM_LIST_LENGTH", []), anchor["MAX_ITEM_LIST_LENGTH"], default=anchor["MAX_ITEM_LIST_LENGTH"]))
            if "num_heads" in space:
                heads = int(_nearest_option(space["num_heads"], max(1, hidden // 48), default=1))
            else:
                heads = 1
            if "num_layers" in space:
                num_layers = int(_nearest_option(space["num_layers"], 2 if hidden <= 128 else 3, default=2))
            elif "n_layers" in space:
                num_layers = int(_nearest_option(space["n_layers"], 2 if hidden <= 128 else 3, default=2))
            else:
                num_layers = 2
            learning_rate = float(_median_option(space.get("learning_rate", []), 1e-3))
            weight_decay = float(_median_option(space.get("weight_decay", []), 1e-4))
            hidden_dropout = float(_nearest_option(space.get("hidden_dropout_prob", []), anchor["dropout"], default=anchor["dropout"]))
            attn_dropout = float(_nearest_option(space.get("attn_dropout_prob", []), anchor["dropout"], default=anchor["dropout"]))
            for seed in seeds:
                cursor += 1
                label = BASELINE_MODEL_LABELS.get(model, model.upper())
                rows.append(
                    {
                        "question": question,
                        "stage": "stage1",
                        "run_axis": _BASE.QUESTION_AXIS[question],
                        "dataset": candidate.dataset,
                        "model": model,
                        "family": "baseline",
                        "panel_family": "baseline_compare",
                        "variant_group": "baseline_compare",
                        "variant_label": label,
                        "variant_order": 100 + len(rows),
                        "setting_key": f"{question}_{model}",
                        "setting_label": label,
                        "base_rank": int(candidate.rank),
                        "base_tag": candidate.tag,
                        "base_result_json": str(candidate.result_json),
                        "seed_id": int(seed),
                        "runtime_seed": int(980000 + cursor),
                        "job_id": f"{question.upper()}_{_BASE.sanitize_token(candidate.dataset, upper=True)}_{model.upper()}_R{int(candidate.rank):02d}_S{int(seed)}",
                        "run_phase": f"{question.upper()}_{_BASE.sanitize_token(candidate.dataset, upper=True)}_{model.upper()}_R{int(candidate.rank):02d}_S{int(seed)}",
                        "search_space": {"learning_rate": [learning_rate]},
                        "fixed_context": {
                            "MAX_ITEM_LIST_LENGTH": int(max_len),
                            "hidden_size": int(hidden),
                            "embedding_size": int(embedding),
                            "num_layers": int(num_layers),
                            "num_heads": int(heads),
                            "hidden_dropout_prob": float(hidden_dropout),
                            "attn_dropout_prob": float(attn_dropout),
                            "weight_decay": float(weight_decay),
                        },
                        "max_evals": 1,
                        "max_run_hours": float(max_run_hours),
                        "tune_epochs": int(tune_epochs),
                        "tune_patience": int(tune_patience),
                        "selection_rule": "overall_seen_target",
                    }
                )
    if smoke_test:
        rows = rows[: max(1, int(smoke_max_runs))]
    return rows


def _build_mixed_command(row: dict[str, Any], gpu_id: str, *, search_algo: str) -> list[str]:
    if str(row.get("family", "")).strip().lower() == "baseline":
        return _FINAL.build_baseline_command(row, gpu_id, search_algo)
    return _route_command_impl(row, gpu_id, search_algo=search_algo)


_BASE.build_route_command = _build_mixed_command


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
    """Appendix D structural ablations with variant_group/variant_label aligned to A02 notebook.

    variant_group="temporal"  → stage-layout / temporal-role variants (A02 panel b)
    variant_group="cue_org"   → semantic-grouping / cue-organisation variants (A02 panel a)
    """
    # --- temporal group: stage-layout and temporal-role variants ---
    temporal_remap = {
        "final_three_stage":       ("Final 3-stage",            "temporal", 1),
        "single_view_macro":       ("Single-view (macro only)", "temporal", 2),
        "two_view_remove_micro":   ("Two-view (macro+mid)",     "temporal", 3),
        "two_view_remove_macro":   ("Local-first ordering",     "temporal", 4),
        "two_view_remove_mid":     ("Global-late ordering",     "temporal", 5),
        "identical_scope":         ("Identical scope",          "temporal", 6),
        "scope_swap":              ("Scope swap",               "temporal", 7),
    }
    # --- cue_org group: semantic-grouping and routing-organisation variants ---
    cue_org_remap = {
        "full_semantic":           ("Family Prior Intact",          "cue_org", 1),
        "reduced_family":          ("Fewer Semantic Groups",        "cue_org", 2),
        "shuffled_family":         ("Groups Shuffled",              "cue_org", 3),
        "flat_random":             ("Flat Scalar Bag",              "cue_org", 4),
        "flat_dense":              ("Flat Dense (routing)",         "cue_org", 5),
        "flat_sparse":             ("Flat Sparse (routing)",        "cue_org", 6),
        "hierarchical_dense":      ("Hierarchical Dense (routing)", "cue_org", 7),
        "hierarchical_sparse":     ("Hierarchical Sparse (main)",   "cue_org", 8),
    }
    remap = {**temporal_remap, **cue_org_remap}
    base_settings = _dedupe_settings(_BASE.q3_settings(), _LEGACY.a06_settings())
    out: list[dict[str, Any]] = []
    for s in base_settings:
        row = deepcopy(s)
        key = str(row.get("setting_key", "")).strip()
        if key in remap:
            label, group, order = remap[key]
            row["variant_label"] = label
            row["variant_group"] = group
            row["variant_order"] = order
        else:
            # Fall back: keep existing values; assign group based on panel_family
            pf = str(row.get("panel_family", "")).strip()
            if "routing_org" in pf:
                row.setdefault("variant_group", "cue_org")
            else:
                row.setdefault("variant_group", "temporal")
        out.append(row)
    return out


def sparse_settings() -> list[dict[str, Any]]:
    """Appendix E sparse routing variants with variant_labels aligned to A03 notebook."""
    # Mapping a07 setting_key → (variant_label, variant_order, active_experts)
    # active_experts assumes expert_scale=3, G=4, so total=12.
    label_map = {
        "dense_global":  ("Dense full mixture",       1, 12),
        "group_dense":   ("Dense per group",           2, 12),
        "global_top8":   ("Flat sparse top-8",         3,  8),
        "global_top4":   ("Flat sparse top-6",         4,  6),   # top-4 ≈ top-6 active; closest to notebook label
        "group_top2":    ("Top-3gr Top-2ex — main",    5,  6),   # 3 groups × 2 experts = 6 active
        "group_top1":    ("Top-2gr Top-1ex (2 act.)",  6,  2),
        "global_top2":   ("Flat sparse top-2",         7,  2),
    }
    settings: list[dict[str, Any]] = []
    for item in _LEGACY.a07_settings():
        row = deepcopy(item)
        sk = str(row.get("setting_key", "")).strip()
        if sk in label_map:
            label, order, n_active = label_map[sk]
            row["variant_label"] = label
            row["variant_order"] = order
            row["active_experts"] = n_active
        else:
            row.setdefault("variant_label", str(row.get("setting_label", sk)))
            row.setdefault("variant_order", 99)
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
            "variant_label": "KNN consistency only",
            "variant_order": 3,
            "overrides": {**deepcopy(full), "z_loss_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "setting_key": "zloss_only",
            "setting_label": "Z-Loss Only",
            "variant_label": "Z-loss only",
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
    result_path_obj, _payload, payload_checkpoint = _BASE.resolve_result_artifacts(
        summary_row, require_checkpoint=True
    )
    checkpoint_file = str(summary_row.get("checkpoint_file", "") or "").strip()
    if not _BASE._checkpoint_path_is_usable(checkpoint_file):
        checkpoint_file = payload_checkpoint
    cmd = [
        python_bin(),
        str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "eval_checkpoint_interventions.py"),
        "--source-result-json",
        str(result_path_obj),
        "--checkpoint-file",
        str(Path(checkpoint_file).expanduser().resolve()),
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
        "result_path": str(result_path_obj),
        "checkpoint_file": str(Path(checkpoint_file).expanduser().resolve()),
        "intervention_manifest": str(manifest),
        "status": "ok",
        "error": "",
    }


def build_postprocess_error_row(
    *,
    question: str,
    source_summary_row: dict[str, str],
    error: Exception,
    source_question: str | None = None,
) -> dict[str, str]:
    row = {
        "question": question,
        "dataset": str(source_summary_row.get("dataset", "") or ""),
        "setting_key": str(source_summary_row.get("setting_key", "") or ""),
        "setting_label": str(source_summary_row.get("setting_label", "") or ""),
        "base_rank": str(source_summary_row.get("base_rank", "") or ""),
        "base_tag": str(source_summary_row.get("base_tag", "") or ""),
        "seed_id": str(source_summary_row.get("seed_id", "") or ""),
        "selection_rule": str(source_summary_row.get("selection_rule", "overall_seen_target") or "overall_seen_target"),
        "result_path": str(source_summary_row.get("result_path", "") or ""),
        "checkpoint_file": str(source_summary_row.get("checkpoint_file", "") or ""),
        "status": "error",
        "error": str(error),
    }
    if source_question:
        row["source_question"] = str(source_question)
    return row


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
