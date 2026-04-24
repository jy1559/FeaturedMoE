#!/usr/bin/env python3
"""Shared helpers for the real_final_ablation pipeline (Q2~Q5 main-body experiments)."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"
WRITING_ROOT = REPO_ROOT / "writing" / "260419_real_final_exp"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

FINAL_EXPERIMENT_DIR = RUN_DIR / "final_experiment"
_FINAL_COMMON_SPEC = importlib.util.spec_from_file_location(
    "final_experiment_common", FINAL_EXPERIMENT_DIR / "common.py"
)
if _FINAL_COMMON_SPEC is None or _FINAL_COMMON_SPEC.loader is None:
    raise RuntimeError(f"Failed to load final_experiment common.py from {FINAL_EXPERIMENT_DIR}")
_FINAL_COMMON = importlib.util.module_from_spec(_FINAL_COMMON_SPEC)
_FINAL_COMMON_SPEC.loader.exec_module(_FINAL_COMMON)

append_csv_row = _FINAL_COMMON.append_csv_row
extract_error_tail = _FINAL_COMMON.extract_error_tail
has_run_status_end_normal = _FINAL_COMMON.has_run_status_end_normal
hydra_literal = _FINAL_COMMON.hydra_literal
load_result_payload = _FINAL_COMMON.load_result_payload
now_utc = _FINAL_COMMON.now_utc
parse_result_path_from_log = _FINAL_COMMON.parse_result_path_from_log
python_bin = _FINAL_COMMON.python_bin
read_json = _FINAL_COMMON.read_json
result_has_successful_trials = _FINAL_COMMON.result_has_successful_trials
result_summary_from_payload = _FINAL_COMMON.result_summary_from_payload
sanitize_token = _FINAL_COMMON.sanitize_token
validate_session_fixed_files = _FINAL_COMMON.validate_session_fixed_files
write_json = _FINAL_COMMON.write_json
from hydra_utils import configure_eval_sampling, enforce_v4_feature_mode, load_hydra_config  # noqa: E402


TRACK = "real_final_ablation"
RESULT_ROOT = ARTIFACT_ROOT / "results" / TRACK
LOG_ROOT = ARTIFACT_ROOT / "logs" / TRACK
DATA_ROOT = WRITING_ROOT / "data"
DEFAULT_BASE_CSV = RUN_DIR / "final_experiment" / "ablation" / "configs" / "base_candidates.csv"
DEFAULT_FEATURE_DATA_ROOT = str((REPO_ROOT / "Datasets" / "processed" / "feature_added_v4").resolve())

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "beauty",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
]
DEFAULT_MODELS = ["featured_moe_n3"]
ROUTE_MODEL = "featured_moe_n3"
ROUTE_MODEL_CLASS = "FeaturedMoE_N3"
ROUTE_MODEL_OVERRIDE = "featured_moe_n3_tune"

QUESTION_AXIS = {
    "q2": "q2_routing_control",
    "q3": "q3_design_justification",
    "q4": "q4_efficiency",
    "q5": "q5_behavior_semantics",
}

SUMMARY_FIELDS = [
    "question",
    "stage",
    "dataset",
    "model",
    "family",
    "panel_family",
    "variant_group",
    "variant_label",
    "variant_order",
    "job_id",
    "parent_job_id",
    "run_phase",
    "setting_key",
    "setting_label",
    "base_rank",
    "base_tag",
    "seed_id",
    "runtime_seed",
    "gpu_id",
    "status",
    "selection_rule",
    "valid_score",
    "test_score",
    "best_valid_mrr20",
    "test_mrr20",
    "checkpoint_file",
    "result_path",
    "base_result_json",
    "log_path",
    "elapsed_sec",
    "error",
    "timestamp_utc",
]

STOP_EVENT = threading.Event()
ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()
ACTIVE_PROCESS_LOCK = threading.Lock()

FULL_STAGE_LAYOUT = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
STAGE_NAMES = ("macro", "mid", "micro")
FAMILY_NAMES = ("memory", "focus", "tempo", "exposure")
CASE_GROUPS = [
    "memory_plus",
    "memory_minus",
    "focus_plus",
    "focus_minus",
    "tempo_plus",
    "tempo_minus",
    "exposure_plus",
    "exposure_minus",
]


@dataclass
class BaseCandidate:
    dataset: str
    model: str
    rank: int
    tag: str
    notes: str
    result_json: Path
    payload: dict[str, Any]
    base_config: dict[str, Any]
    checkpoint_file: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _requires_case_eval(question: str, summary_row: dict[str, Any]) -> bool:
    question_key = str(question or "").strip().lower()
    setting_key = str(summary_row.get("setting_key", "") or "").strip().lower()
    if question_key == "q5":
        return setting_key == "behavior_guided"
    return False


def _requires_intervention(question: str, summary_row: dict[str, Any]) -> bool:
    question_key = str(question or "").strip().lower()
    setting_key = str(summary_row.get("setting_key", "") or "").strip().lower()
    if question_key == "q5":
        return setting_key == "behavior_guided"
    return False


def _postprocess_requirements_for_summary(question: str, summary_row: dict[str, Any]) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    if _requires_case_eval(question, summary_row):
        requirements.append((f"{question}_case_eval_index.csv", "case_eval_manifest"))
    if _requires_intervention(question, summary_row):
        requirements.append((f"{question}_intervention_index.csv", "intervention_manifest"))
    return requirements


def _question_logging_policy(question: str) -> dict[str, bool]:
    question_key = str(question or "").strip().lower()
    if question_key == "q2":
        return {
            "special_logging": True,
            "log_unseen_target_metrics": True,
            "fmoe_diag_logging": True,
            "fmoe_special_logging": True,
            "fmoe_feature_family_ablation_logging": True,
        }
    if question_key == "q3":
        return {
            "special_logging": True,
            "log_unseen_target_metrics": True,
            "fmoe_diag_logging": False,
            "fmoe_special_logging": False,
            "fmoe_feature_family_ablation_logging": False,
        }
    if question_key == "q5":
        return {
            "special_logging": False,
            "log_unseen_target_metrics": False,
            "fmoe_diag_logging": False,
            "fmoe_special_logging": False,
            "fmoe_feature_family_ablation_logging": False,
        }
    return {
        "special_logging": False,
        "log_unseen_target_metrics": False,
        "fmoe_diag_logging": False,
        "fmoe_special_logging": False,
        "fmoe_feature_family_ablation_logging": False,
    }


def _should_export_final_checkpoint(row: dict[str, Any]) -> bool:
    question = str(row.get("question", "") or "").strip().lower()
    return _requires_case_eval(question, row) or _requires_intervention(question, row)


def _checkpoint_export_path_for_row(row: dict[str, Any]) -> Path:
    job_id = sanitize_token(str(row.get("job_id", row.get("run_phase", "run")) or "run"))
    return (RESULT_ROOT / "checkpoints" / f"{job_id}_best_model_state.pth").resolve()


def _row_identity_matches(index_row: dict[str, str], summary_row: dict[str, Any]) -> bool:
    result_path = str(summary_row.get("result_path", "") or "").strip()
    index_result_path = str(index_row.get("result_path", "") or "").strip()
    if result_path and index_result_path:
        return result_path == index_result_path
    keys = ("dataset", "setting_key", "base_rank", "seed_id")
    return all(
        str(index_row.get(key, "") or "").strip() == str(summary_row.get(key, "") or "").strip()
        for key in keys
    )


def _load_index_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _is_shared_result_checkpoint(path_str: str) -> bool:
    path_text = str(path_str or "").strip()
    if not path_text:
        return False
    try:
        path = Path(path_text).expanduser().resolve()
    except Exception:
        return False
    shared_paths = {
        (RESULT_ROOT / "best_model_state.pth").resolve(),
        (RESULT_ROOT / "probe_model_state.pth").resolve(),
    }
    return path in shared_paths


def _payload_checkpoint_file(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("best_checkpoint_file", "") or "").strip()


def _checkpoint_path_is_usable(path_str: str) -> bool:
    path_text = str(path_str or "").strip()
    if not path_text or _is_shared_result_checkpoint(path_text):
        return False
    try:
        return Path(path_text).expanduser().resolve().exists()
    except Exception:
        return False


def _payload_has_usable_checkpoint(payload: dict[str, Any] | None) -> bool:
    return _checkpoint_path_is_usable(_payload_checkpoint_file(payload))


def _find_best_result_payload_for_run_phase(run_phase: str) -> Path | None:
    run_phase_text = str(run_phase or "").strip()
    if not run_phase_text:
        return None
    candidates: list[tuple[int, float, Path]] = []
    for result_path in RESULT_ROOT.glob("*.json"):
        try:
            payload = read_json(result_path)
        except Exception:
            continue
        if str(payload.get("run_phase", "") or "").strip() != run_phase_text:
            continue
        if not result_has_successful_trials(payload):
            continue
        usable_checkpoint = 1 if _payload_has_usable_checkpoint(payload) else 0
        candidates.append((usable_checkpoint, result_path.stat().st_mtime, result_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def resolve_result_artifacts(
    source_row: dict[str, Any],
    *,
    require_checkpoint: bool = False,
) -> tuple[Path, dict[str, Any], str]:
    result_path_text = str(source_row.get("result_path", "") or "").strip()
    run_phase = str(source_row.get("run_phase", "") or source_row.get("job_id", "") or "").strip()

    candidates: list[Path] = []
    if result_path_text:
        candidates.append(Path(result_path_text).expanduser())
    fallback = _find_best_result_payload_for_run_phase(run_phase)
    if fallback is not None and fallback not in candidates:
        candidates.append(fallback)

    best_result_path: Path | None = None
    best_payload: dict[str, Any] = {}
    best_checkpoint = ""
    best_rank: tuple[int, int, float] | None = None

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = read_json(candidate)
        except Exception:
            continue
        if not result_has_successful_trials(payload):
            continue
        checkpoint_file = _payload_checkpoint_file(payload)
        checkpoint_usable = 1 if _checkpoint_path_is_usable(checkpoint_file) else 0
        checkpoint_present = 1 if checkpoint_file else 0
        rank = (checkpoint_usable, checkpoint_present, candidate.stat().st_mtime)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_result_path = candidate.resolve()
            best_payload = payload
            best_checkpoint = checkpoint_file

    source_checkpoint = str(source_row.get("checkpoint_file", "") or "").strip()
    if _checkpoint_path_is_usable(source_checkpoint):
        best_checkpoint = source_checkpoint

    if best_result_path is None:
        raise RuntimeError(f"Missing result_path for run_phase={run_phase or '<unknown>'}")
    if require_checkpoint and not best_checkpoint:
        raise RuntimeError(f"Missing exported checkpoint for {best_result_path}")
    return best_result_path, best_payload, best_checkpoint


def _manifest_rows_succeeded(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    if not rows:
        return False
    if "status" not in rows[0]:
        return True
    saw_ok = False
    for row in rows:
        status = str(row.get("status", "") or "").strip().lower()
        if status == "error":
            return False
        if status == "ok":
            saw_ok = True
    return saw_ok


def _postprocess_manifest_exists(index_csv: Path, manifest_field: str, summary_row: dict[str, Any]) -> bool:
    return find_completed_postprocess_row(index_csv, manifest_field, summary_row) is not None


def find_completed_postprocess_row(
    index_csv: Path, manifest_field: str, summary_row: dict[str, Any]
) -> dict[str, str] | None:
    for index_row in _load_index_rows(index_csv):
        if not _row_identity_matches(index_row, summary_row):
            continue
        if str(index_row.get("status", "") or "").strip().lower() == "error":
            continue
        if _is_shared_result_checkpoint(str(index_row.get("checkpoint_file", "") or "")):
            continue
        manifest_path = str(index_row.get(manifest_field, "") or "").strip()
        if not manifest_path:
            continue
        manifest_file = Path(manifest_path).expanduser()
        if _manifest_rows_succeeded(manifest_file):
            return index_row
    return None


def find_completed_case_eval_row(question: str, summary_row: dict[str, Any]) -> dict[str, str] | None:
    requirements = _postprocess_requirements_for_summary(question, summary_row)
    for index_name, manifest_field in requirements:
        if manifest_field != "case_eval_manifest":
            continue
        return find_completed_postprocess_row(index_path(question, index_name), manifest_field, summary_row)
    return None


def find_completed_intervention_row(question: str, summary_row: dict[str, Any]) -> dict[str, str] | None:
    requirements = _postprocess_requirements_for_summary(question, summary_row)
    for index_name, manifest_field in requirements:
        if manifest_field != "intervention_manifest":
            continue
        return find_completed_postprocess_row(index_path(question, index_name), manifest_field, summary_row)
    return None


def postprocess_complete_for_summary(question: str, summary_row: dict[str, Any]) -> bool:
    requirements = _postprocess_requirements_for_summary(question, summary_row)
    if not requirements:
        return True
    for index_name, manifest_field in requirements:
        if not _postprocess_manifest_exists(index_path(question, index_name), manifest_field, summary_row):
            return False
    return True


def parse_csv_list(text: str) -> list[str]:
    return [token.strip() for token in str(text or "").split(",") if token.strip()]


def parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for token in parse_csv_list(text):
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def canonical_stage_maps() -> dict[str, Any]:
    return {
        "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
        "stage_compute_mode": {"macro": "moe", "mid": "moe", "micro": "moe"},
        "stage_router_mode": {"macro": "learned", "mid": "learned", "micro": "learned"},
        "stage_router_source": {"macro": "feature", "mid": "feature", "micro": "feature"},
        "stage_feature_injection": {"macro": "none", "mid": "none", "micro": "none"},
        "router_use_hidden": False,
        "router_use_feature": True,
        "expert_use_hidden": True,
        "expert_use_feature": True,
        "layer_layout": list(FULL_STAGE_LAYOUT),
        "feature_perturb_mode": "none",
        "feature_perturb_apply": "none",
        "feature_perturb_family": [],
        "feature_perturb_keywords": [],
        "feature_perturb_shift": 1,
    }


def stage_layout_for(enabled: Iterable[str]) -> list[str]:
    enabled_set = {str(token).strip().lower() for token in enabled}
    layout: list[str] = []
    if enabled_set:
        layout.append("attn")
    if "macro" in enabled_set:
        layout.append("macro_ffn")
    if "mid" in enabled_set:
        layout.append("mid_ffn")
    if "mid" in enabled_set or "micro" in enabled_set:
        layout.append("attn")
    if "micro" in enabled_set:
        layout.append("micro_ffn")
    return layout or ["attn"]


# ---------------------------------------------------------------------------
# Q2: What should control routing?
# Four routing-source variants:
#   1. RouteRec Full (behavior-guided, both hidden + feature)
#   2. Shared FFN (no routing at all — dense baseline)
#   3. Router Hidden Only (routing but feature-free)
#   4. Router Feature Only (routing but hidden-free)
# ---------------------------------------------------------------------------

def q2_settings() -> list[dict[str, Any]]:
    full = canonical_stage_maps()
    return [
        {
            "setting_key": "shared_ffn",
            "setting_label": "Shared FFN",
            "variant_label": "Shared FFN",
            "variant_order": 1,
            "overrides": {
                **deepcopy(full),
                "stage_compute_mode": {stage: "dense_plain" for stage in STAGE_NAMES},
                "stage_router_mode": {stage: "none" for stage in STAGE_NAMES},
                "stage_router_source": {stage: "hidden" for stage in STAGE_NAMES},
                "stage_feature_injection": {stage: "none" for stage in STAGE_NAMES},
                "router_use_hidden": False,
                "router_use_feature": False,
                "expert_use_feature": False,
            },
        },
        {
            "setting_key": "hidden_only",
            "setting_label": "Hidden Only",
            "variant_label": "Hidden only",
            "variant_order": 2,
            "overrides": {
                **deepcopy(full),
                "router_use_hidden": True,
                "router_use_feature": False,
                "stage_router_source": {stage: "hidden" for stage in STAGE_NAMES},
            },
        },
        {
            "setting_key": "feature_fusion_bias",
            "setting_label": "Fusion Bias",
            "variant_label": "Fusion bias",
            "variant_order": 3,
            "overrides": {
                **deepcopy(full),
                "stage_compute_mode": {stage: "dense_plain" for stage in STAGE_NAMES},
                "stage_router_mode": {stage: "none" for stage in STAGE_NAMES},
                "stage_router_source": {stage: "hidden" for stage in STAGE_NAMES},
                "stage_feature_injection": {stage: "gated_bias" for stage in STAGE_NAMES},
                "router_use_hidden": False,
                "router_use_feature": False,
                "expert_use_feature": True,
            },
        },
        {
            "setting_key": "mixed_hidden_behavior",
            "setting_label": "Mixed Hidden+Behavior",
            "variant_label": "Mixed",
            "variant_order": 4,
            "overrides": {
                **deepcopy(full),
                "router_use_hidden": True,
                "router_use_feature": True,
                "stage_router_source": {stage: "both" for stage in STAGE_NAMES},
            },
        },
        {
            "setting_key": "behavior_guided",
            "setting_label": "Behavior-Guided",
            "variant_label": "Behavior-guided",
            "variant_order": 5,
            "overrides": deepcopy(full),
        },
    ]


# ---------------------------------------------------------------------------
# Q3: Why is RouteRec's three-part design effective?
# Tests three axes:
#   (a) Router composition: stage presence (macro/mid/micro), dense vs sparse
#   (b) Stage semantics: temporal scope assignments
#   (c) Cue design: unstructured vs structured vs shuffled vs no-feature
# ---------------------------------------------------------------------------

def q3_settings() -> list[dict[str, Any]]:
    full = canonical_stage_maps()
    settings: list[dict[str, Any]] = [
        {
            "setting_key": "single_view_macro",
            "setting_label": "Single View (Macro)",
            "panel_family": "temporal_decomp",
            "variant_group": "single_view",
            "variant_label": "Single-view",
            "variant_order": 1,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["macro"]),
                "stage_compute_mode": {"macro": "moe", "mid": "none", "micro": "none"},
                "stage_router_mode": {"macro": "learned", "mid": "none", "micro": "none"},
            },
        },
        {
            "setting_key": "single_view_mid",
            "setting_label": "Single View (Mid)",
            "panel_family": "temporal_decomp",
            "variant_group": "single_view",
            "variant_label": "Single-view",
            "variant_order": 1,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["mid"]),
                "stage_compute_mode": {"macro": "none", "mid": "moe", "micro": "none"},
                "stage_router_mode": {"macro": "none", "mid": "learned", "micro": "none"},
            },
        },
        {
            "setting_key": "single_view_micro",
            "setting_label": "Single View (Micro)",
            "panel_family": "temporal_decomp",
            "variant_group": "single_view",
            "variant_label": "Single-view",
            "variant_order": 1,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["micro"]),
                "stage_compute_mode": {"macro": "none", "mid": "none", "micro": "moe"},
                "stage_router_mode": {"macro": "none", "mid": "none", "micro": "learned"},
            },
        },
        {
            "setting_key": "two_view_remove_macro",
            "setting_label": "Best Two-View Candidate (No Macro)",
            "panel_family": "temporal_decomp",
            "variant_group": "best_two_view",
            "variant_label": "Best 2-view",
            "variant_order": 2,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["mid", "micro"]),
                "stage_compute_mode": {"macro": "none", "mid": "moe", "micro": "moe"},
                "stage_router_mode": {"macro": "none", "mid": "learned", "micro": "learned"},
            },
        },
        {
            "setting_key": "two_view_remove_mid",
            "setting_label": "Best Two-View Candidate (No Mid)",
            "panel_family": "temporal_decomp",
            "variant_group": "best_two_view",
            "variant_label": "Best 2-view",
            "variant_order": 2,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["macro", "micro"]),
                "stage_compute_mode": {"macro": "moe", "mid": "none", "micro": "moe"},
                "stage_router_mode": {"macro": "learned", "mid": "none", "micro": "learned"},
            },
        },
        {
            "setting_key": "two_view_remove_micro",
            "setting_label": "Best Two-View Candidate (No Micro)",
            "panel_family": "temporal_decomp",
            "variant_group": "best_two_view",
            "variant_label": "Best 2-view",
            "variant_order": 2,
            "overrides": {
                **deepcopy(full),
                "layer_layout": stage_layout_for(["macro", "mid"]),
                "stage_compute_mode": {"macro": "moe", "mid": "moe", "micro": "none"},
                "stage_router_mode": {"macro": "learned", "mid": "learned", "micro": "none"},
            },
        },
        {
            "setting_key": "final_three_stage",
            "setting_label": "Final Three-Stage",
            "panel_family": "temporal_decomp",
            "variant_group": "final_three_stage",
            "variant_label": "Final 3-stage",
            "variant_order": 3,
            "overrides": deepcopy(full),
        },
        {
            "setting_key": "flat_sparse",
            "setting_label": "Flat Sparse",
            "panel_family": "routing_org",
            "variant_group": "flat_sparse",
            "variant_label": "Flat sparse",
            "variant_order": 1,
            "overrides": {
                **deepcopy(full),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 6,
                "group_top_k": 0,
                "expert_top_k": 0,
            },
        },
        {
            "setting_key": "flat_dense",
            "setting_label": "Flat Dense",
            "panel_family": "routing_org",
            "variant_group": "flat_dense",
            "variant_label": "Flat dense",
            "variant_order": 2,
            "overrides": {
                **deepcopy(full),
                "topk_scope_mode": "global_flat",
                "moe_top_k": 0,
                "group_top_k": 0,
                "expert_top_k": 0,
            },
        },
        {
            "setting_key": "hierarchical_sparse",
            "setting_label": "Hierarchical Sparse",
            "panel_family": "routing_org",
            "variant_group": "hierarchical_sparse",
            "variant_label": "Hierarchical sparse",
            "variant_order": 3,
            "overrides": {
                **deepcopy(full),
                "topk_scope_mode": "per_group",
                "group_top_k": 3,
                "expert_top_k": 2,
                "moe_top_k": 0,
            },
        },
        {
            "setting_key": "hierarchical_dense",
            "setting_label": "Hierarchical Dense",
            "panel_family": "routing_org",
            "variant_group": "hierarchical_dense",
            "variant_label": "Hierarchical dense",
            "variant_order": 4,
            "overrides": {
                **deepcopy(full),
                "topk_scope_mode": "per_group",
                "group_top_k": 0,
                "expert_top_k": 0,
                "moe_top_k": 0,
            },
        },
    ]
    return settings


# ---------------------------------------------------------------------------
# Q4: Are lightweight cues sufficient in practice?
# Availability-oriented cue reduction:
#   Full → Remove Category → Remove Time → Sequence Only
# ---------------------------------------------------------------------------

def q4_portability_settings() -> list[dict[str, Any]]:
    full = canonical_stage_maps()
    return [
        {
            "setting_key": "full",
            "setting_label": "Full Cues",
            "overrides": deepcopy(full),
        },
        {
            "setting_key": "remove_category",
            "setting_label": "Remove Category",
            "overrides": {
                **deepcopy(full),
                "stage_feature_drop_keywords": ["cat", "theme", "overlap", "mismatch"],
            },
        },
        {
            "setting_key": "remove_time",
            "setting_label": "Remove Time",
            "overrides": {
                **deepcopy(full),
                "stage_feature_drop_keywords": ["gap", "pace", "int", "age", "delta"],
            },
        },
        {
            "setting_key": "sequence_only",
            "setting_label": "Sequence Only",
            "overrides": {
                **deepcopy(full),
                "router_use_feature": False,
                "expert_use_feature": False,
                "stage_router_source": {stage: "hidden" for stage in STAGE_NAMES},
            },
        },
    ]


# ---------------------------------------------------------------------------
# Q5: Do routing patterns align with behavioral regimes?
# Train one RouteRec Full checkpoint; post-hoc case eval + interventions.
# ---------------------------------------------------------------------------

def q5_train_settings() -> list[dict[str, Any]]:
    return [
        {
            "setting_key": "behavior_guided",
            "setting_label": "Behavior-Guided",
            "overrides": canonical_stage_maps(),
        }
    ]


def q5_intervention_specs() -> list[dict[str, Any]]:
    out = [
        {
            "intervention": "full",
            "label": "Full",
            "overrides": {},
        },
        {
            "intervention": "feature_zero_all",
            "label": "Feature Zero All",
            "overrides": {"feature_perturb_mode": "zero", "feature_perturb_apply": "eval"},
        },
        {
            "intervention": "feature_shuffle_all",
            "label": "Feature Shuffle All",
            "overrides": {"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval"},
        },
        {
            "intervention": "shuffle_all",
            "label": "Shuffle All",
            "overrides": {"feature_perturb_mode": "global_permute", "feature_perturb_apply": "eval"},
        },
        {
            "intervention": "repeat_flatten",
            "label": "Repeat Flatten",
            "overrides": {
                "feature_perturb_mode": "zero",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["memory"],
            },
        },
        {
            "intervention": "switch_boost",
            "label": "Switch Boost",
            "overrides": {
                "feature_perturb_mode": "family_permute",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["focus"],
            },
        },
        {
            "intervention": "tempo_compress",
            "label": "Tempo Compress",
            "overrides": {
                "feature_perturb_mode": "position_shift",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["tempo"],
                "feature_perturb_shift": 1,
            },
        },
        {
            "intervention": "popularity_spike",
            "label": "Popularity Spike",
            "overrides": {
                "feature_perturb_mode": "family_permute",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["exposure"],
            },
        },
    ]
    for family in FAMILY_NAMES:
        out.append(
            {
                "intervention": f"feature_zero_{family}",
                "label": f"Feature Zero {family}",
                "group": "feature_zero_by_family",
                "target_family": family,
                "overrides": {
                    "feature_perturb_mode": "zero",
                    "feature_perturb_apply": "eval",
                    "feature_perturb_family": [family],
                },
            }
        )
        out.append(
            {
                "intervention": f"feature_shuffle_{family}",
                "label": f"Feature Shuffle {family}",
                "group": "feature_shuffle_by_family",
                "target_family": family,
                "overrides": {
                    "feature_perturb_mode": "shuffle",
                    "feature_perturb_apply": "eval",
                    "feature_perturb_family": [family],
                },
            }
        )
    return out


def _normalize_model_name(name: str) -> str:
    text = str(name or "").strip().lower()
    if text in {"featured_moe_n3", "featuredmoe_n3", "featured_moe_n3_tune", "fmoe_n3", "routerec", "routerec_full"}:
        return ROUTE_MODEL
    return text


def _load_candidate_rows(base_csv: Path) -> list[dict[str, str]]:
    if not base_csv.exists():
        raise FileNotFoundError(f"Base candidate CSV not found: {base_csv}")
    with open(base_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Base candidate CSV is empty: {base_csv}")
    required = {"dataset", "model", "rank", "enabled", "tag", "result_json", "notes"}
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise RuntimeError(f"Base candidate CSV missing columns: {missing}")
    return rows


def _merge_base_config(payload: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("context_fixed", "fixed_search", "best_params"):
        block = payload.get(key) or {}
        if isinstance(block, dict):
            merged.update(deepcopy(block))
    return merged


def build_eval_config_from_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = str(payload.get("dataset", "") or payload.get("dataset_raw", "")).strip()
    if not dataset:
        raise RuntimeError("Result payload is missing dataset.")
    cfg = load_hydra_config(
        config_dir=EXP_DIR / "configs",
        config_name="config",
        overrides=[f"model={ROUTE_MODEL_OVERRIDE}", f"dataset={dataset}"],
    )
    cfg_omega = OmegaConf.create(cfg)
    cfg_omega = configure_eval_sampling(cfg_omega)
    cfg = OmegaConf.to_container(cfg_omega, resolve=True)
    enforce_v4_feature_mode(cfg)
    cfg["dataset"] = dataset
    cfg["model"] = str(cfg.get("model", ROUTE_MODEL_CLASS))
    cfg["eval_mode"] = "session_fixed"
    cfg["feature_mode"] = "full_v4"
    cfg["data_path"] = DEFAULT_FEATURE_DATA_ROOT
    cfg["log_wandb"] = False
    cfg["show_progress"] = False
    cfg["special_logging"] = True
    cfg["exclude_unseen_target_from_main_eval"] = True
    cfg["log_unseen_target_metrics"] = True
    cfg["fmoe_diag_logging"] = True
    cfg["fmoe_special_logging"] = True
    cfg["fmoe_feature_family_ablation_logging"] = True
    cfg["fmoe_feature_ablation_logging"] = False
    cfg["fmoe_eval_logging_timing"] = "final_only"
    cfg.update(_merge_base_config(payload))
    # Re-assert session_fixed protocol after payload merge (prevents RS-split leakage).
    cfg["benchmark_filename"] = ["train", "valid", "test"]
    cfg["eval_args"] = {"group_by": "user", "order": "TO", "mode": "full"}
    cfg["eval_sampling"] = {"mode": "full", "auto_full_threshold": 999999999}
    cfg["history_input_mode"] = "session_only"
    cfg["history_eval_policy"] = "strict_train_prefix"
    cfg["target_group_field"] = "session_id"
    cfg["history_group_field"] = "user_id"
    cfg["exclude_unseen_target_from_main_eval"] = True
    cfg["log_unseen_target_metrics"] = True
    return cfg


def load_base_candidates(
    base_csv: Path, *, datasets: list[str], models: list[str], top_k_configs: int
) -> list[BaseCandidate]:
    rows = _load_candidate_rows(base_csv)
    wanted_datasets = {str(dataset).strip() for dataset in datasets}
    wanted_models = {_normalize_model_name(model) for model in models}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).strip()
        model = _normalize_model_name(row.get("model", ""))
        enabled = str(row.get("enabled", "")).strip().lower()
        if enabled not in {"1", "true", "yes", "y"}:
            continue
        if wanted_datasets and dataset not in wanted_datasets:
            continue
        if wanted_models and model not in wanted_models:
            continue
        grouped.setdefault((dataset, model), []).append(row)

    selected: list[BaseCandidate] = []
    for key, group_rows in sorted(grouped.items()):
        group_rows.sort(key=lambda row: int(row.get("rank", 10**9)))
        seen_ranks: set[int] = set()
        picked = 0
        for row in group_rows:
            rank = int(row["rank"])
            if rank in seen_ranks:
                raise RuntimeError(f"Duplicate rank={rank} for dataset={key[0]} model={key[1]} in {base_csv}")
            seen_ranks.add(rank)
            result_json = Path(str(row["result_json"])).expanduser().resolve()
            payload = read_json(result_json)
            checkpoint_file = str(payload.get("best_checkpoint_file", "") or "").strip()
            selected.append(
                BaseCandidate(
                    dataset=key[0],
                    model=key[1],
                    rank=rank,
                    tag=str(row.get("tag", "")).strip(),
                    notes=str(row.get("notes", "")).strip(),
                    result_json=result_json,
                    payload=payload,
                    base_config=build_eval_config_from_result_payload(payload),
                    checkpoint_file=checkpoint_file,
                )
            )
            picked += 1
            if picked >= max(int(top_k_configs), 1):
                break
    if not selected:
        raise RuntimeError(f"No enabled base candidates matched datasets={datasets} models={models}")
    return selected


def lr_search_spec(base_lr: float, *, lr_mode: str = "narrow_loguniform") -> tuple[list[float], str]:
    lr = float(base_lr if base_lr and base_lr > 0 else 1e-3)
    if lr_mode != "narrow_loguniform":
        return [lr], "choice"
    lo = max(lr * 0.67, 1e-6)
    hi = max(lr * 1.5, lo * 1.05)
    return [lo, hi], "loguniform"


def build_search_entries(
    search_space: dict[str, Any], fixed_context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    search: dict[str, Any] = {}
    types: dict[str, str] = {}
    for key, value in dict(search_space or {}).items():
        search[key] = value if isinstance(value, list) else [value]
        types[key] = "choice"
    for key, value in dict(fixed_context or {}).items():
        if key in search:
            continue
        search[key] = [value]
        types[key] = "choice"
    return search, types


def log_path_for_row(stage: str, row: dict[str, Any]) -> Path:
    dataset = sanitize_token(row.get("dataset", ""), upper=False)
    model = sanitize_token(row.get("model", ""), upper=False)
    family = sanitize_token(row.get("family", ""), upper=False)
    job_id = sanitize_token(row.get("job_id", ""), upper=True)
    return LOG_ROOT / stage / dataset / model / family / f"{job_id}.log"


def summary_path(question: str) -> Path:
    return LOG_ROOT / question / "summary.csv"


def manifest_path(question: str) -> Path:
    return LOG_ROOT / question / "manifest.json"


def index_path(question: str, suffix: str) -> Path:
    return LOG_ROOT / question / suffix


def build_route_row(
    *,
    question: str,
    candidate: BaseCandidate,
    setting: dict[str, Any],
    seed: int,
    runtime_seed: int,
    max_evals: int,
    max_run_hours: float,
    tune_epochs: int,
    tune_patience: int,
    lr_mode: str,
) -> dict[str, Any]:
    base_cfg = deepcopy(candidate.base_config)
    base_cfg.update(deepcopy(setting.get("overrides") or {}))
    base_lr = float(base_cfg.get("learning_rate", 1e-3) or 1e-3)
    search_values, lr_type = lr_search_spec(base_lr, lr_mode=lr_mode)
    fixed_context = deepcopy(base_cfg)
    # Strip session-fixed protocol keys so they don't leak into search block overrides.
    for reserved_key in (
        "benchmark_filename",
        "eval_args",
        "eval_mode",
        "eval_sampling",
        "feature_mode",
        "data_path",
        "history_input_mode",
        "history_eval_policy",
        "target_group_field",
        "history_group_field",
        "exclude_unseen_target_from_main_eval",
        "log_unseen_target_metrics",
    ):
        fixed_context.pop(reserved_key, None)
    fixed_context.pop("learning_rate", None)
    setting_key = str(setting["setting_key"])
    return {
        "question": question,
        "stage": question,
        "run_axis": QUESTION_AXIS[question],
        "dataset": candidate.dataset,
        "model": ROUTE_MODEL,
        "family": "route",
        "panel_family": str(setting.get("panel_family", "")),
        "variant_group": str(setting.get("variant_group", setting_key)),
        "setting_key": setting_key,
        "setting_label": str(setting["setting_label"]),
        "variant_label": str(setting.get("variant_label", setting["setting_label"])),
        "variant_order": int(setting.get("variant_order", 0) or 0),
        "base_rank": int(candidate.rank),
        "base_tag": candidate.tag,
        "base_result_json": str(candidate.result_json),
        "seed_id": int(seed),
        "runtime_seed": int(runtime_seed),
        "job_id": (
            f"{question.upper()}_{sanitize_token(candidate.dataset, upper=True)}_"
            f"{sanitize_token(setting_key, upper=True)}_R{int(candidate.rank):02d}_S{int(seed)}"
        ),
        "run_phase": (
            f"{question.upper()}_{sanitize_token(candidate.dataset, upper=True)}_"
            f"{sanitize_token(setting_key, upper=True)}_R{int(candidate.rank):02d}_S{int(seed)}"
        ),
        "search_space": {"learning_rate": search_values},
        "search_space_types": {"learning_rate": lr_type},
        "fixed_context": fixed_context,
        "max_evals": int(max_evals),
        "max_run_hours": float(max_run_hours),
        "tune_epochs": int(tune_epochs),
        "tune_patience": int(tune_patience),
        "selection_rule": "overall_seen_target",
    }


def build_route_command(row: dict[str, Any], gpu_id: str, *, search_algo: str) -> list[str]:
    search, types = build_search_entries(row.get("search_space") or {}, row.get("fixed_context") or {})
    for key, stype in dict(row.get("search_space_types") or {}).items():
        types[key] = str(stype)
    logging_policy = _question_logging_policy(str(row.get("question", "")))
    should_export_checkpoint = _should_export_final_checkpoint(row)
    cmd = [
        python_bin(),
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(search_algo),
        "--max-evals",
        str(int(row["max_evals"])),
        "--max-run-hours",
        str(float(row.get("max_run_hours") or 1.0)),
        "--tune-epochs",
        str(int(row["tune_epochs"])),
        "--tune-patience",
        str(int(row["tune_patience"])),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        str(row["run_axis"]),
        "--run-phase",
        str(row["run_phase"]),
        f"model={ROUTE_MODEL_OVERRIDE}",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        "feature_mode=full_v4",
        "++benchmark_filename=[train,valid,test]",
        "++eval_args.group_by=user",
        "++eval_args.order=TO",
        "++eval_args.mode=full",
        "++history_input_mode=session_only",
        "++history_eval_policy=strict_train_prefix",
        "++target_group_field=session_id",
        "++history_group_field=user_id",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        f"++special_logging={'true' if logging_policy['special_logging'] else 'false'}",
        "++exclude_unseen_target_from_main_eval=true",
        f"++log_unseen_target_metrics={'true' if logging_policy['log_unseen_target_metrics'] else 'false'}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        "enable_tf32=true",
        "reproducibility=false",
        "++enable_cudnn_benchmark=true",
        "fmoe_debug_logging=false",
        f"fmoe_diag_logging={'true' if logging_policy['fmoe_diag_logging'] else 'false'}",
        f"fmoe_special_logging={'true' if logging_policy['fmoe_special_logging'] else 'false'}",
        f"fmoe_feature_family_ablation_logging={'true' if logging_policy['fmoe_feature_family_ablation_logging'] else 'false'}",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        "enable_data_cache=false",
        "enable_session_split_cache=false",
        "max_in_memory_data_cache=4",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_phase={hydra_literal(str(row['question']).upper())}",
        f"++phase_run_type={hydra_literal(str(row['question']))}",
        f"++phase_axis_id={hydra_literal(str(row['run_axis']))}",
        f"++phase_setting_id={hydra_literal(str(row.get('setting_key', '')))}",
        f"++phase_hparam_id={hydra_literal(str(row.get('base_rank', '')))}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(str(row['job_id']))}",
        f"++artifact_export_final_checkpoint={hydra_literal(should_export_checkpoint)}",
    ]
    if should_export_checkpoint:
        cmd.append(
            f"++__artifact_combo_best_export_path={hydra_literal(str(_checkpoint_export_path_for_row(row)))}"
        )
    for key, values in search.items():
        cmd.append(f"++search.{key}={hydra_literal(values)}")
        cmd.append(f"++search_space_type_overrides.{key}={types[key]}")
    return cmd


def _active_proc_add(proc: subprocess.Popen[Any]) -> None:
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.add(proc)


def _active_proc_remove(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None:
        return
    with ACTIVE_PROCESS_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def _terminate_process(proc: subprocess.Popen[Any], sig_num: int) -> None:
    try:
        if proc.poll() is not None:
            return
        if int(sig_num) == int(signal.SIGKILL):
            proc.kill()
        else:
            proc.terminate()
    except Exception:
        return


def install_signal_handlers() -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        STOP_EVENT.set()
        with ACTIVE_PROCESS_LOCK:
            procs = list(ACTIVE_PROCESSES)
        for proc in procs:
            _terminate_process(proc, signal.SIGTERM)
        time.sleep(0.2)
        for proc in procs:
            _terminate_process(proc, signal.SIGKILL)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def build_summary_row(
    row: dict[str, Any],
    *,
    gpu_id: str,
    status: str,
    result_path: str,
    log_path: Path,
    elapsed_sec: float,
    error: str,
) -> dict[str, Any]:
    payload = load_result_payload(Path(result_path)) if result_path else {}
    metrics = result_summary_from_payload(payload)
    return {
        "question": row.get("question", ""),
        "stage": row.get("stage", ""),
        "dataset": row.get("dataset", ""),
        "model": row.get("model", ""),
        "family": row.get("family", ""),
        "panel_family": row.get("panel_family", ""),
        "variant_group": row.get("variant_group", ""),
        "variant_label": row.get("variant_label", row.get("setting_label", "")),
        "variant_order": row.get("variant_order", ""),
        "job_id": row.get("job_id", ""),
        "parent_job_id": row.get("parent_job_id", ""),
        "run_phase": row.get("run_phase", ""),
        "setting_key": row.get("setting_key", ""),
        "setting_label": row.get("setting_label", ""),
        "base_rank": row.get("base_rank", ""),
        "base_tag": row.get("base_tag", ""),
        "seed_id": int(row.get("seed_id", 0) or 0),
        "runtime_seed": int(row.get("runtime_seed", 0) or 0),
        "gpu_id": str(gpu_id),
        "status": status,
        "selection_rule": row.get("selection_rule", "overall_seen_target"),
        "valid_score": metrics.get("valid_score", 0.0),
        "test_score": metrics.get("test_score", 0.0),
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "checkpoint_file": str(payload.get("best_checkpoint_file", "") or ""),
        "result_path": str(result_path or ""),
        "base_result_json": row.get("base_result_json", ""),
        "log_path": str(log_path),
        "elapsed_sec": float(elapsed_sec),
        "error": str(error or ""),
        "timestamp_utc": now_utc(),
    }


def _summary_row_matches_job(summary_row: dict[str, Any], row: dict[str, Any]) -> bool:
    job_id = str(row.get("job_id", "") or "").strip()
    summary_job_id = str(summary_row.get("job_id", "") or "").strip()
    if job_id and summary_job_id:
        return job_id == summary_job_id
    keys = ("question", "dataset", "setting_key", "base_rank", "seed_id")
    return all(
        str(summary_row.get(key, "") or "").strip() == str(row.get(key, "") or "").strip()
        for key in keys
    )


def _summary_row_is_resumable(question: str, summary_row: dict[str, Any]) -> bool:
    if str(summary_row.get("status", "") or "").strip().lower() != "ok":
        return False
    result_path = str(summary_row.get("result_path", "") or "").strip()
    if not result_path:
        return False
    payload = load_result_payload(Path(result_path))
    if not result_has_successful_trials(payload):
        return False
    return postprocess_complete_for_summary(question, summary_row)


def _find_previous_summary_row(
    question: str, row: dict[str, Any], previous_rows: list[dict[str, str]]
) -> dict[str, Any] | None:
    for summary_row in previous_rows:
        if not _summary_row_matches_job(summary_row, row):
            continue
        if _summary_row_is_resumable(question, summary_row):
            resumed = dict(summary_row)
            resumed["gpu_id"] = "resume"
            resumed["timestamp_utc"] = now_utc()
            return resumed
    return None


def _find_result_payload_for_row(row: dict[str, Any]) -> Path | None:
    run_phase = str(row.get("job_id", "") or row.get("run_phase", "") or "").strip()
    return _find_best_result_payload_for_run_phase(run_phase)


def _preserved_summary_rows(
    previous_rows: list[dict[str, str]],
    current_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preserved: list[dict[str, Any]] = []
    for summary_row in previous_rows:
        if any(_summary_row_matches_job(summary_row, row) for row in current_rows):
            continue
        preserved.append(dict(summary_row))
    return preserved


def _write_summary_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for row in rows:
        append_csv_row(
            path,
            SUMMARY_FIELDS,
            {key: row.get(key, "") for key in SUMMARY_FIELDS},
        )


def _finalize_summary_file(
    *,
    target_summary: Path,
    temp_summary: Path,
    previous_summary_rows: list[dict[str, str]],
    current_rows: list[dict[str, Any]],
) -> None:
    merged_rows = _preserved_summary_rows(previous_summary_rows, current_rows)
    if temp_summary.exists():
        merged_rows.extend(_load_index_rows(temp_summary))
        temp_summary.unlink()
    _write_summary_rows(target_summary, merged_rows)


def _recover_previous_summary_rows(question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for row in rows:
        result_path_obj = _find_result_payload_for_row(row)
        if result_path_obj is None:
            continue
        payload = load_result_payload(result_path_obj)
        if not result_has_successful_trials(payload):
            continue
        recovered.append(
            build_summary_row(
                row,
                gpu_id="recovered",
                status="ok",
                result_path=str(result_path_obj),
                log_path=log_path_for_row(str(row.get("stage", question)), row),
                elapsed_sec=0.0,
                error="",
            )
        )
    return recovered


def resumed_summary_row(
    row: dict[str, Any], previous_rows: list[dict[str, str]] | None = None
) -> dict[str, Any] | None:
    question = str(row.get("question", "") or row.get("stage", "") or "")
    if previous_rows:
        previous = _find_previous_summary_row(question, row, previous_rows)
        if previous is not None:
            return previous

    log_path = log_path_for_row(str(row["stage"]), row)
    if not has_run_status_end_normal(log_path):
        result_path_obj = _find_result_payload_for_row(row)
        if result_path_obj is None:
            return None
    else:
        result_path_obj = parse_result_path_from_log(log_path)
        if result_path_obj is None:
            result_path_obj = _find_result_payload_for_row(row)
            if result_path_obj is None:
                return None
    payload = load_result_payload(result_path_obj)
    if not result_has_successful_trials(payload):
        return None
    summary = build_summary_row(
        row,
        gpu_id="resume",
        status="ok",
        result_path=str(result_path_obj),
        log_path=log_path,
        elapsed_sec=0.0,
        error="",
    )
    if not postprocess_complete_for_summary(
        str(row.get("question", "") or row.get("stage", "")), summary
    ):
        return None
    return summary


def describe_job(row: dict[str, Any]) -> str:
    return (
        f"question={row.get('question')} dataset={row.get('dataset')} job={row.get('job_id')} "
        f"setting={row.get('setting_key')} rank={row.get('base_rank')} seed={row.get('seed_id')} "
        f"max_evals={row.get('max_evals')}"
    )


def run_one_job(row: dict[str, Any], gpu_id: str, *, search_algo: str) -> dict[str, Any]:
    stage = str(row["stage"])
    log_path = log_path_for_row(stage, row)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_route_command(row, gpu_id, search_algo=search_algo)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    rc = 1
    proc: subprocess.Popen[Any] | None = None
    print(f"[launch][gpu={gpu_id}] START {describe_job(row)} log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(EXP_DIR), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True
        )
        _active_proc_add(proc)
        try:
            while True:
                if STOP_EVENT.is_set() and proc.poll() is None:
                    _terminate_process(proc, signal.SIGTERM)
                    time.sleep(0.3)
                    if proc.poll() is None:
                        _terminate_process(proc, signal.SIGKILL)
                polled = proc.poll()
                if polled is not None:
                    rc = int(polled)
                    break
                time.sleep(0.2)
        finally:
            _active_proc_remove(proc)
    elapsed = time.time() - start
    result_path_obj = parse_result_path_from_log(log_path)
    payload = load_result_payload(result_path_obj) if result_path_obj is not None else {}
    normal_end = has_run_status_end_normal(log_path)
    success = result_has_successful_trials(payload)
    status = "ok" if (rc == 0 and success and (normal_end or result_path_obj is not None)) else "fail"
    if status == "ok":
        error = ""
    elif rc == 0 and normal_end and not success:
        error = "completed_without_successful_trial"
    else:
        error = f"rc={rc} tail={extract_error_tail(log_path)}"
    summary = build_summary_row(
        row,
        gpu_id=gpu_id,
        status=status,
        result_path="" if result_path_obj is None else str(result_path_obj),
        log_path=log_path,
        elapsed_sec=elapsed,
        error=error,
    )
    print(
        f"[launch][gpu={gpu_id}] END {describe_job(row)} status={status} "
        f"valid={summary.get('valid_score', 0.0):.6f} test={summary.get('test_score', 0.0):.6f}",
        flush=True,
    )
    return summary


def run_jobs(
    rows: list[dict[str, Any]],
    *,
    question: str,
    gpus: list[str],
    search_algo: str,
    resume_from_logs: bool,
    dry_run: bool,
) -> int:
    install_signal_handlers()
    target_summary = summary_path(question)
    target_summary.parent.mkdir(parents=True, exist_ok=True)
    previous_summary_rows = read_summary_rows(question)
    if not previous_summary_rows:
        previous_summary_rows = _recover_previous_summary_rows(question, rows)
    temp_summary = target_summary.with_name(
        f"{target_summary.stem}.tmp.{os.getpid()}{target_summary.suffix}"
    )
    if temp_summary.exists():
        temp_summary.unlink()

    if dry_run:
        for row in rows:
            append_csv_row(
                temp_summary,
                SUMMARY_FIELDS,
                {key: row.get(key, "") for key in SUMMARY_FIELDS}
                | {
                    "gpu_id": "dry-run",
                    "status": "planned",
                    "valid_score": "",
                    "test_score": "",
                    "best_valid_mrr20": "",
                    "test_mrr20": "",
                    "checkpoint_file": "",
                    "result_path": "",
                    "log_path": str(log_path_for_row(question, row)),
                    "elapsed_sec": 0.0,
                    "error": "",
                    "timestamp_utc": now_utc(),
                },
            )
        _finalize_summary_file(
            target_summary=target_summary,
            temp_summary=temp_summary,
            previous_summary_rows=previous_summary_rows,
            current_rows=rows,
        )
        return 0

    pending: Queue[dict[str, Any]] = Queue()
    n_total = len(rows)
    n_skipped = 0
    for row in rows:
        if resume_from_logs:
            resumed = resumed_summary_row(row, previous_rows=previous_summary_rows)
            if resumed is not None:
                append_csv_row(temp_summary, SUMMARY_FIELDS, resumed)
                n_skipped += 1
                print(
                    f"[resume] SKIP ({n_skipped}) {describe_job(row)}  reason=existing_successful_result",
                    flush=True,
                )
                continue
        pending.put(row)

    n_pending = pending.qsize()
    print(
        f"[run_jobs] question={question}  total={n_total}  "
        f"skipped(resume)={n_skipped}  pending={n_pending}"
        + ("  [resume_from_logs=OFF → all run]" if not resume_from_logs else ""),
        flush=True,
    )

    if pending.empty():
        _finalize_summary_file(
            target_summary=target_summary,
            temp_summary=temp_summary,
            previous_summary_rows=previous_summary_rows,
            current_rows=rows,
        )
        return 0

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(str(gpu))

    def worker() -> None:
        while not STOP_EVENT.is_set():
            try:
                row = pending.get_nowait()
            except Empty:
                return
            gpu_id = gpu_queue.get()
            try:
                summary = run_one_job(row, gpu_id, search_algo=search_algo)
                append_csv_row(temp_summary, SUMMARY_FIELDS, summary)
            finally:
                gpu_queue.put(gpu_id)
                pending.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, len(gpus)))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    _finalize_summary_file(
        target_summary=target_summary,
        temp_summary=temp_summary,
        previous_summary_rows=previous_summary_rows,
        current_rows=rows,
    )
    return 0


def read_summary_rows(question: str) -> list[dict[str, str]]:
    path = summary_path(question)
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(question: str, rows: list[dict[str, Any]]) -> Path:
    path = manifest_path(question)
    payload = {"generated_at": now_utc(), "question": question, "run_count": len(rows), "rows": rows}
    write_json(path, payload)
    return path


def common_arg_parser(description: str, *, question: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--top-k-configs", type=int, default=1)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--base-csv", default=str(DEFAULT_BASE_CSV))
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument(
        "--max-run-hours",
        type=float,
        default=1.0,
        help="Wall-clock cap per job in hours.",
    )
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--lr-mode", default="narrow_loguniform")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--case-eval-fast",
        action="store_true",
        help="Skip by-group case-eval subsets to reduce eval time.",
    )
    parser.set_defaults(question=question)
    return parser


def resolve_base_csv_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [Path.cwd(), DEFAULT_BASE_CSV.parent, REPO_ROOT]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()


def selected_candidates_from_args(args: argparse.Namespace) -> list[BaseCandidate]:
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) or list(DEFAULT_MODELS)
    for dataset in datasets:
        validate_session_fixed_files(dataset)
    return load_base_candidates(
        resolve_base_csv_path(args.base_csv),
        datasets=datasets,
        models=models,
        top_k_configs=int(args.top_k_configs),
    )


def build_train_rows(
    *,
    question: str,
    candidates: list[BaseCandidate],
    settings: list[dict[str, Any]],
    seeds: list[int],
    max_evals: int,
    max_run_hours: float = 1.0,
    tune_epochs: int,
    tune_patience: int,
    lr_mode: str,
    smoke_test: bool,
    smoke_max_runs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for candidate in candidates:
        for setting in settings:
            for seed in seeds:
                cursor += 1
                rows.append(
                    build_route_row(
                        question=question,
                        candidate=candidate,
                        setting=setting,
                        seed=seed,
                        runtime_seed=960000 + cursor,
                        max_evals=max_evals,
                        max_run_hours=max_run_hours,
                        tune_epochs=tune_epochs,
                        tune_patience=tune_patience,
                        lr_mode=lr_mode,
                    )
                )
    if smoke_test:
        rows = rows[: max(1, int(smoke_max_runs))]
    return rows


def materialize_eval_config(source_result_json: str, output_dir: Path) -> Path:
    payload = read_json(Path(source_result_json).expanduser().resolve())
    cfg = build_eval_config_from_result_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return path


def latest_manifest_under(root: Path, filename: str) -> Path:
    matches = sorted(root.rglob(filename), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    return matches[-1]


def run_case_eval_pipeline(
    *,
    question: str,
    source_summary_row: dict[str, str],
    output_root: Path,
    skip_original: bool = False,
    skip_by_group: bool = False,
) -> dict[str, Any]:
    result_path_obj, payload, payload_checkpoint = resolve_result_artifacts(
        source_summary_row, require_checkpoint=True
    )
    result_path = str(result_path_obj)
    checkpoint_file = str(source_summary_row.get("checkpoint_file", "") or "").strip()
    if not _checkpoint_path_is_usable(checkpoint_file):
        checkpoint_file = payload_checkpoint
    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    source_dir = ensure_dir(output_root / "source")
    config_json = materialize_eval_config(result_path, source_dir)
    bundle_root = ensure_dir(output_root / "case_eval")
    bundle_name = f"{question}_{sanitize_token(source_summary_row.get('dataset',''))}_{sanitize_token(source_summary_row.get('setting_key',''))}_R{sanitize_token(source_summary_row.get('base_rank',''))}_S{sanitize_token(source_summary_row.get('seed_id',''))}"
    cmd = [
        python_bin(),
        str(REPO_ROOT / "experiments" / "run" / "fmoe_n4" / "eval_checkpoint_case_subsets.py"),
        "--config-json",
        str(config_json),
        "--checkpoint-file",
        str(checkpoint_path),
        "--source-result-json",
        str(Path(result_path).resolve()),
        "--output-root",
        str(bundle_root),
        "--bundle-name",
        bundle_name,
        "--resume",
    ]
    if skip_original:
        cmd.append("--skip-original")
    if skip_by_group:
        cmd.append("--skip-by-group")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    manifest = (bundle_root / bundle_name / "case_eval_manifest.csv").expanduser().resolve()
    if not manifest.exists():
        manifest = latest_manifest_under(bundle_root, "case_eval_manifest.csv")
    export_dir = ensure_dir(output_root / "tables")
    export_cmd = [
        python_bin(),
        str(REPO_ROOT / "experiments" / "run" / "fmoe_n4" / "export_case_eval_tables.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(export_dir),
    ]
    subprocess.run(export_cmd, check=True, cwd=str(REPO_ROOT))
    return {
        "question": question,
        "dataset": source_summary_row.get("dataset", ""),
        "setting_key": source_summary_row.get("setting_key", ""),
        "setting_label": source_summary_row.get("setting_label", ""),
        "base_rank": source_summary_row.get("base_rank", ""),
        "base_tag": source_summary_row.get("base_tag", ""),
        "seed_id": source_summary_row.get("seed_id", ""),
        "selection_rule": source_summary_row.get("selection_rule", "overall_seen_target"),
        "result_path": result_path,
        "checkpoint_file": str(checkpoint_path),
        "case_eval_manifest": str(manifest),
        "case_eval_export_dir": str(export_dir),
    }


def write_index_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
