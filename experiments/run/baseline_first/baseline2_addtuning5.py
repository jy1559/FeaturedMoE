#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import baseline2_addtuning as stage1
import baseline2_addtuning3_2 as stage32
import baseline2_addtuning4 as stage4
import run_pair60_campaign as pair60


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
ARTIFACT_ROOT = EXP_DIR / "run" / "artifacts"
TRACK = "baseline_2"
AXIS = "PAIR60_ADDTUNING5"

RESULTS_ROOT = ARTIFACT_ROOT / "results" / TRACK
OUTPUT_ROOT = ARTIFACT_ROOT / "logs" / TRACK / AXIS
SPACE_ROOT = OUTPUT_ROOT / "spaces"
SESSION_LOG_ROOT = OUTPUT_ROOT / "session_logs"
PLAN_CSV = OUTPUT_ROOT / "plan.csv"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
MANIFEST_JSON = OUTPUT_ROOT / "manifest.json"

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
SUCCESS_MIN_ELAPSED_SEC = 120.0

FATAL_LOG_TOKENS = (
    "Traceback (most recent call last):",
    "ModuleNotFoundError:",
    "No module named 'torch'",
    'No module named "torch"',
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: CUDA",
    "Killed",
)

SUMMARY_SOURCES = [
    *stage4.SUMMARY_SOURCES,
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_ADDTUNING4" / "summary.csv",
]

TARGETS = [
    ("foursquare", "gru4rec", "high"),
    ("foursquare", "bsarec", "critical"),
    ("foursquare", "fame", "critical"),
    ("foursquare", "difsr", "critical"),
    ("foursquare", "fearec", "high"),
    ("movielens1m", "gru4rec", "high"),
    ("movielens1m", "difsr", "high"),
    ("movielens1m", "duorec", "critical"),
    ("movielens1m", "fearec", "critical"),
    ("lastfm0.03", "gru4rec", "critical"),
    ("lastfm0.03", "fame", "high"),
    ("lastfm0.03", "bsarec", "high"),
    ("lastfm0.03", "duorec", "high"),
    ("lastfm0.03", "fearec", "critical"),
]

PRIORITY_RANK = {"critical": 0, "high": 1}

DATASET_BASE_MAX_EVALS = {
    "foursquare": 14,
    "movielens1m": 11,
    "lastfm0.03": 9,
}

DATASET_BASE_COMBOS = {
    "foursquare": 5,
    "movielens1m": 4,
    "lastfm0.03": 4,
}

MODEL_MAX_EVAL_ADJUST = {
    "gru4rec": 1,
    "bsarec": 1,
    "fame": 1,
    "difsr": 1,
    "fearec": 0,
    "duorec": 0,
}

MODEL_COMBO_BONUS = {
    "gru4rec": 1,
    "bsarec": 1,
    "fame": 1,
    "difsr": 1,
    "fearec": 1,
    "duorec": 0,
}

MODEL_BATCH_OVERRIDES: dict[str, dict[str, int]] = {}

MODEL_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "gru4rec": [
        {"name": "stable_short_anchor", "anchor": "short_best", "scale": 0.95, "max_len": 10, "layers": 1, "profile": "gru_reset", "lr_mode": "narrow_mid", "seq_mode": "short"},
        {"name": "stable_mid_anchor", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gru_deep", "lr_mode": "narrow_mid", "seq_mode": "mid"},
        {"name": "stable_long_anchor", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "gru_reset", "lr_mode": "narrow_high", "seq_mode": "long"},
        {"name": "aggr_wide_mid", "anchor": "balanced_best", "scale": 1.15, "max_len": 18, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.92},
        {"name": "aggr_huge_long", "anchor": "valid_best", "scale": 1.25, "max_len": 24, "layers": 4, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": -0.02, "wd_mult": 0.88},
        {"name": "aggr_len32_probe", "anchor": "best_test", "scale": 1.05, "max_len": 32, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    ],
    "fearec": [
        {"name": "stable_mid_sem", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "narrow_mid", "seq_mode": "mid", "param_overrides": {"tau": 0.18, "lmd": 0.035, "lmd_sem": 0.08}},
        {"name": "stable_short_sem", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "lr_mode": "narrow_mid", "seq_mode": "short", "param_overrides": {"tau": 0.16, "lmd": 0.03, "lmd_sem": 0.05}},
        {"name": "stable_long_sem", "anchor": "valid_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "narrow_high", "seq_mode": "long", "param_overrides": {"tau": 0.2, "lmd": 0.04, "lmd_sem": 0.1}},
        {"name": "aggr_wide_sem", "anchor": "best_test", "scale": 1.15, "max_len": 20, "layers": 3, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9, "param_overrides": {"tau": 0.22, "lmd": 0.05, "lmd_sem": 0.12}},
        {"name": "aggr_long_sem", "anchor": "valid_best", "scale": 1.1, "max_len": 24, "layers": 3, "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide", "dropout_delta": -0.02, "wd_mult": 0.88, "param_overrides": {"tau": 0.24, "lmd": 0.055, "lmd_sem": 0.14}},
    ],
    "duorec": [
        {"name": "stable_mid_contrast", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "narrow_mid", "seq_mode": "mid", "param_overrides": {"tau": 0.18, "lmd": 0.04, "lmd_sem": 0.0}},
        {"name": "stable_short_contrast", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "lr_mode": "narrow_mid", "seq_mode": "short", "param_overrides": {"tau": 0.16, "lmd": 0.03, "lmd_sem": 0.0}},
        {"name": "stable_long_contrast", "anchor": "valid_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "narrow_high", "seq_mode": "long", "param_overrides": {"tau": 0.2, "lmd": 0.045, "lmd_sem": 0.0}},
        {"name": "aggr_wide_contrast", "anchor": "best_test", "scale": 1.15, "max_len": 20, "layers": 3, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9, "param_overrides": {"tau": 0.22, "lmd": 0.055, "lmd_sem": 0.0}},
        {"name": "aggr_long_contrast", "anchor": "valid_best", "scale": 1.1, "max_len": 24, "layers": 3, "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide", "dropout_delta": -0.02, "wd_mult": 0.88, "param_overrides": {"tau": 0.24, "lmd": 0.06, "lmd_sem": 0.0}},
    ],
    "bsarec": [
        {"name": "stable_alpha_hi_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "narrow_mid", "seq_mode": "mid"},
        {"name": "stable_alpha_mid_short", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "profile": "alpha_mid_c3", "lr_mode": "narrow_mid", "seq_mode": "short"},
        {"name": "stable_alpha_hi_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 1, "profile": "alpha_hi_c2", "lr_mode": "narrow_high", "seq_mode": "long"},
        {"name": "aggr_alpha_peak_mid", "anchor": "valid_best", "scale": 1.15, "max_len": 18, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9},
        {"name": "aggr_alpha_peak_long", "anchor": "valid_best", "scale": 1.2, "max_len": 24, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.88},
    ],
    "fame": [
        {"name": "stable_experts4_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "experts4", "lr_mode": "narrow_mid", "seq_mode": "mid"},
        {"name": "stable_experts3_short", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "profile": "experts3", "lr_mode": "narrow_mid", "seq_mode": "short"},
        {"name": "stable_experts4_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "experts4", "lr_mode": "narrow_high", "seq_mode": "long"},
        {"name": "aggr_experts6_mid", "anchor": "valid_best", "scale": 1.12, "max_len": 18, "layers": 3, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
        {"name": "aggr_experts6_long", "anchor": "valid_best", "scale": 1.18, "max_len": 22, "layers": 3, "profile": "experts6", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": 0.01, "wd_mult": 1.05},
    ],
    "difsr": [
        {"name": "stable_gate_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gate_midattr", "lr_mode": "narrow_mid", "seq_mode": "mid"},
        {"name": "stable_concat_mid", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "concat_midattr", "lr_mode": "narrow_mid", "seq_mode": "mid"},
        {"name": "stable_gate_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "gate_highattr", "lr_mode": "narrow_high", "seq_mode": "long"},
        {"name": "aggr_gate_high_wide", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "profile": "gate_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": 0.0, "wd_mult": 0.95},
        {"name": "aggr_concat_long", "anchor": "valid_best", "scale": 1.1, "max_len": 24, "layers": 3, "profile": "concat_highattr", "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide", "dropout_delta": 0.0, "wd_mult": 0.92},
    ],
}


def target_specs() -> list[stage1.TargetSpec]:
    return [stage1.TargetSpec(dataset, model, priority) for dataset, model, priority in TARGETS]


def load_history_rows() -> dict[tuple[str, str], list[stage32.stage2.HistSeed]]:
    return stage32.load_summary_rows(SUMMARY_SOURCES)


def apply_param_overrides(model: str, params: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key, value in overrides.items():
        out[key] = value
    return stage32.normalize_params(model, out)


def apply_model_runtime_overrides(model: str, params: dict[str, Any]) -> dict[str, Any]:
    overrides = MODEL_BATCH_OVERRIDES.get(model)
    if not overrides:
        return stage32.normalize_params(model, params)
    return apply_param_overrides(model, params, overrides)


def build_candidate(model: str, anchor_params: dict[str, Any], spec: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    hidden = stage1.safe_int(anchor_params.get("hidden_size", anchor_params.get("embedding_size", 128)), 128)
    params = stage32.adjust_arch(
        model,
        anchor_params,
        hidden=stage32.scale_dim(hidden, spec.get("scale", 1.0)),
        layers=int(spec.get("layers", stage1.safe_int(anchor_params.get("num_layers", 2), 2))),
        max_len=int(spec.get("max_len", stage1.safe_int(anchor_params.get("max_len", 16), 16))),
        head_mode=str(spec.get("head_mode", "base")),
        dropout_delta=float(spec.get("dropout_delta", 0.0)),
        weight_decay_mult=float(spec.get("wd_mult", 1.0)),
    )
    profile = str(spec.get("profile", "")).strip()
    if profile:
        params = stage32.apply_model_profile(model, params, profile)
    param_overrides = spec.get("param_overrides") or {}
    if param_overrides:
        params = apply_param_overrides(model, params, dict(param_overrides))
    params = apply_model_runtime_overrides(model, params)
    meta = {
        "lr_mode": str(spec.get("lr_mode", "wide_mid")),
        "seq_mode": str(spec.get("seq_mode", "mid")),
        "profile": profile,
    }
    return str(spec["name"]), stage32.normalize_params(model, params), meta


def base_lr_band(model: str, lr_mode: str) -> tuple[float, float]:
    narrow_bands = {
        "gru4rec": {
            "narrow_mid": (6e-5, 7e-4),
            "narrow_high": (1e-4, 1.0e-3),
        },
        "fearec": {
            "narrow_mid": (8e-5, 8e-4),
            "narrow_high": (1.5e-4, 1.2e-3),
        },
        "duorec": {
            "narrow_mid": (8e-5, 8e-4),
            "narrow_high": (1.5e-4, 1.2e-3),
        },
        "bsarec": {
            "narrow_mid": (1e-4, 1.0e-3),
            "narrow_high": (1.5e-4, 1.5e-3),
        },
        "fame": {
            "narrow_mid": (8e-5, 8e-4),
            "narrow_high": (1.5e-4, 1.2e-3),
        },
        "difsr": {
            "narrow_mid": (1.5e-3, 5.0e-3),
            "narrow_high": (2.5e-3, 7.0e-3),
        },
    }
    if model in narrow_bands and lr_mode in narrow_bands[model]:
        return narrow_bands[model][lr_mode]
    return stage4.base_lr_band(model, lr_mode)


def build_search_block(model: str, combo_kind: str, params: dict[str, Any], meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    overrides: dict[str, str] = {"learning_rate": "loguniform"}
    lr_lo, lr_hi = base_lr_band(model, meta["lr_mode"])
    search: dict[str, Any] = {"learning_rate": [round(lr_lo, 8), round(lr_hi, 8)]}

    if combo_kind.startswith("stable_"):
        return search, overrides

    max_len = stage1.safe_int(params.get("max_len", 16), 16)
    if meta["seq_mode"] == "short":
        seq_values = [max_len, max_len + 2, max_len + 4]
    elif meta["seq_mode"] == "long":
        seq_values = [max(10, max_len - 4), max_len, max_len + 4]
    else:
        seq_values = [max(8, max_len - 2), max_len, max_len + 2]

    if model == "gru4rec":
        search["MAX_ITEM_LIST_LENGTH"] = sorted({stage32.nearest_seq(value) for value in seq_values})
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"
    elif model in {"fearec", "duorec"}:
        tau = stage1.safe_float(params.get("tau", 0.2), 0.2)
        search["tau"] = stage32.search_choices_around(tau, [tau - 0.03, tau, tau + 0.03], 0.05, 0.4)
        overrides["tau"] = "choice"
    elif model == "bsarec":
        alpha = stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5)
        search["bsarec_alpha"] = stage32.search_choices_around(alpha, [alpha - 0.1, alpha, alpha + 0.1], 0.1, 0.95)
        overrides["bsarec_alpha"] = "choice"
    elif model == "fame":
        experts = stage1.safe_int(params.get("num_experts", 3), 3)
        search["num_experts"] = sorted({max(2, experts - 1), experts, min(6, experts + 1)})
        overrides["num_experts"] = "choice"
    elif model == "difsr":
        search["fusion_type"] = ["gate", "concat"]
        overrides["fusion_type"] = "choice"

    return search, overrides


def combo_count_for_target(target: stage1.TargetSpec, est_runtime_sec: float) -> int:
    base = DATASET_BASE_COMBOS.get(target.dataset, 4) + MODEL_COMBO_BONUS.get(target.model, 0)
    if est_runtime_sec < 180:
        base += 1
    elif est_runtime_sec > 900:
        base -= 1
    return stage1.clamp_int(base, 4, 6)


def max_evals_for_target(target: stage1.TargetSpec, est_runtime_sec: float) -> int:
    base = DATASET_BASE_MAX_EVALS.get(target.dataset, 10) + MODEL_MAX_EVAL_ADJUST.get(target.model, 0)
    if est_runtime_sec > 1200:
        base -= 2
    elif est_runtime_sec > 600:
        base -= 1
    elif est_runtime_sec < 180:
        base += 1
    return stage1.clamp_int(base, 8, 16)


def build_combo_specs(
    target: stage1.TargetSpec,
    rows: list[stage32.stage2.HistSeed],
    est_runtime_sec: float,
    combo_seed_base: int,
) -> list[stage1.ComboSpec]:
    SPACE_ROOT.mkdir(parents=True, exist_ok=True)
    anchors = stage32.choose_append_anchors(target.model, rows)
    if not anchors:
        return []

    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for spec in MODEL_FAMILIES.get(target.model, []):
        anchor_key = str(spec.get("anchor", "best_test"))
        anchor_params = anchors.get(anchor_key)
        if anchor_params is None:
            continue
        candidates.append(build_candidate(target.model, anchor_params, spec))

    uniq = stage32.dedupe_candidates(target.model, candidates, stage32.history_signatures(target.model, rows))

    specs: list[stage1.ComboSpec] = []
    dataset_tag = stage1.sanitize_token(target.dataset)
    model_tag = stage1.sanitize_token(target.model)

    for idx, (combo_kind, params, meta) in enumerate(uniq[: combo_count_for_target(target, est_runtime_sec)], start=1):
        combo_id = f"K{idx}"
        search, type_overrides = build_search_block(target.model, combo_kind, params, meta)
        fixed = stage1.make_fixed_block(target.model, params)
        fixed["search_space_type_overrides"] = type_overrides
        space_yaml = SPACE_ROOT / f"{dataset_tag}_{model_tag}_{combo_id}.yaml"
        stage32.write_space_yaml(space_yaml, fixed, search)
        specs.append(
            stage1.ComboSpec(
                dataset=target.dataset,
                model=target.model,
                combo_id=combo_id,
                combo_kind=combo_kind,
                priority=target.priority,
                est_runtime_sec=est_runtime_sec,
                max_evals=max_evals_for_target(target, est_runtime_sec),
                seed=combo_seed_base + idx,
                base_params=params,
                search_space=search,
                fixed_space=fixed,
                space_yaml=space_yaml,
                run_phase=f"BASELINE2_ADDTUNE5_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
            )
        )
    return specs


def build_command(spec: stage1.ComboSpec, gpu_id: str, python_bin: str, search_algo: str) -> list[str]:
    return [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        pair60.dataset_config_name(spec.dataset),
        "--space-yaml",
        str(spec.space_yaml),
        "--max-evals",
        str(int(spec.max_evals)),
        "--tune-epochs",
        str(DEFAULT_EPOCHS),
        "--tune-patience",
        str(DEFAULT_PATIENCE),
        "--search-algo",
        str(search_algo),
        "--seed",
        str(int(spec.seed)),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS.lower(),
        "--run-phase",
        spec.run_phase,
        f"model={spec.model}",
        f"dataset={spec.dataset}",
        "eval_mode=session_fixed",
        "feature_mode=full_v4",
        "++history_input_mode=session_only",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        f"++seed={int(spec.seed)}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
    ]


def plan_rows(specs: list[stage1.ComboSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "combo_id": spec.combo_id,
                "combo_kind": spec.combo_kind,
                "priority": spec.priority,
                "est_runtime_sec": round(spec.est_runtime_sec, 1),
                "runtime_class": stage1.runtime_class(spec.est_runtime_sec),
                "max_evals": spec.max_evals,
                "epochs": DEFAULT_EPOCHS,
                "patience": DEFAULT_PATIENCE,
                "seed": spec.seed,
                "run_phase": spec.run_phase,
                "space_yaml": str(spec.space_yaml),
                "base_params_json": json.dumps(spec.base_params, ensure_ascii=True, sort_keys=True),
                "search_json": json.dumps(spec.search_space, ensure_ascii=True, sort_keys=True),
            }
        )
    return rows


def has_completion_markers(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if "[RUN_METRICS]" not in text:
        return False
    if "Results ->" not in text:
        return False
    return True


def has_fatal_log_tokens(log_path: Path) -> bool:
    if not log_path.exists():
        return True
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    return any(token in text for token in FATAL_LOG_TOKENS)


def load_result_payload(result_path: Path) -> dict[str, Any] | None:
    if not result_path.exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def result_payload_is_complete(spec: stage1.ComboSpec, result_path: Path) -> bool:
    payload = load_result_payload(result_path)
    if payload is None:
        return False
    if str(payload.get("run_phase", "")).strip() != spec.run_phase:
        return False
    if str(payload.get("dataset", "")).strip() != spec.dataset:
        return False
    if str(payload.get("model", "")).strip().lower() != spec.model.lower():
        return False
    if bool(payload.get("interrupted")):
        return False

    trials = payload.get("trials", []) or []
    if not isinstance(trials, list) or not trials:
        return False

    n_completed = stage1.safe_int(payload.get("n_completed", 0), 0)
    n_recorded_trials = stage1.safe_int(payload.get("n_recorded_trials", len(trials)), len(trials))
    payload_max_evals = stage1.safe_int(payload.get("max_evals", 0), 0)
    if payload_max_evals <= 0:
        return False
    if n_completed != payload_max_evals or n_recorded_trials != payload_max_evals:
        return False
    if len(trials) < payload_max_evals:
        return False
    if any(str(trial.get("status", "")).strip().lower() != "ok" for trial in trials[:payload_max_evals]):
        return False

    if not (payload.get("best_valid_result") or {}):
        return False
    if not (payload.get("test_result") or {}):
        return False
    return True


def log_matches_result_path(log_path: Path, result_path: Path) -> bool:
    parsed = pair60.parse_result_path_from_log(log_path)
    if parsed is None:
        return False
    try:
        return parsed.resolve() == result_path.resolve()
    except Exception:
        return str(parsed) == str(result_path)


def log_is_reusable_success(log_path: Path, result_path: Path) -> bool:
    if not log_path.exists() or not result_path.exists():
        return False
    if not pair60.has_run_status_end_normal(log_path):
        return False
    if not has_completion_markers(log_path):
        return False
    if has_fatal_log_tokens(log_path):
        return False
    if not log_matches_result_path(log_path, result_path):
        return False
    return True


def iter_phase_log_candidates(spec: stage1.ComboSpec, row: dict[str, str] | None) -> list[Path]:
    seen: set[str] = set()
    candidates: list[Path] = []

    def add(path_text: str) -> None:
        text = str(path_text or "").strip()
        if not text:
            return
        path = Path(text)
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if row is not None:
        add(str(row.get("log_path", "")))
    for path in sorted(SESSION_LOG_ROOT.glob(f"{spec.run_phase}*.log"), key=lambda item: item.stat().st_mtime, reverse=True):
        add(str(path))
    return candidates


def find_reusable_artifact(spec: stage1.ComboSpec, row: dict[str, str] | None) -> tuple[Path, Path] | None:
    preferred_result_path: Path | None = None
    if row is not None:
        result_text = str(row.get("result_path", "")).strip()
        if result_text:
            preferred_result_path = Path(result_text)

    for log_path in iter_phase_log_candidates(spec, row):
        result_path = pair60.parse_result_path_from_log(log_path)
        if result_path is None and preferred_result_path is not None:
            result_path = preferred_result_path
        if result_path is None:
            result_path = stage1.find_result_path(spec.run_phase, spec.dataset, spec.model)
        if result_path is None:
            continue
        if not result_payload_is_complete(spec, result_path):
            continue
        if not log_is_reusable_success(log_path, result_path):
            continue
        return log_path, result_path
    return None


def is_reusable_success_row(row: dict[str, str]) -> bool:
    if str(row.get("status", "")).strip().lower() != "ok":
        return False

    elapsed_sec = stage1.safe_float(row.get("elapsed_sec", 0.0), 0.0)
    if elapsed_sec < SUCCESS_MIN_ELAPSED_SEC:
        return False

    log_path_text = str(row.get("log_path", "")).strip()
    result_path_text = str(row.get("result_path", "")).strip()
    if not log_path_text or not result_path_text:
        return False

    log_path = Path(log_path_text)
    result_path = Path(result_path_text)
    if not result_path.exists():
        return False
    if not pair60.has_run_status_end_normal(log_path):
        return False
    if not has_completion_markers(log_path):
        return False
    if has_fatal_log_tokens(log_path):
        return False
    return True


def next_attempt_log_path(run_phase: str) -> Path:
    base_path = SESSION_LOG_ROOT / f"{run_phase}.log"
    if not base_path.exists():
        return base_path
    suffix = time.strftime("%Y%m%d_%H%M%S")
    return SESSION_LOG_ROOT / f"{run_phase}__retry_{suffix}_{time.time_ns()}.log"


def run_one(spec: stage1.ComboSpec, gpu_id: str, python_bin: str, search_algo: str) -> dict[str, Any]:
    SESSION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = next_attempt_log_path(spec.run_phase)
    cmd = build_command(spec, gpu_id, python_bin, search_algo)
    started = time.time()
    status = "failed"
    error = ""
    proc: subprocess.Popen[Any] | None = None

    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# CMD\n")
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), stdout=handle, stderr=subprocess.STDOUT, text=True)
            return_code = proc.wait()
            if return_code == 0:
                status = "ok"
            else:
                status = f"exit_{return_code}"
                error = f"return_code={return_code}"
        except Exception as exc:
            status = "spawn_error"
            error = str(exc)
        finally:
            if proc is not None and proc.poll() is None:
                proc.kill()

    elapsed_sec = time.time() - started
    result_path = stage1.find_result_path(spec.run_phase, spec.dataset, spec.model)
    if result_path is None and status == "ok":
        status = "missing_result"
        error = "result_json_not_found"
    elif status == "ok" and not result_payload_is_complete(spec, result_path):
        status = "incomplete_result"
        error = "result_payload_incomplete_or_failed_trials"
    elif status == "ok" and not pair60.has_run_status_end_normal(log_path):
        status = "missing_normal_end"
        error = "run_status_end_not_normal"
    elif status == "ok" and not has_completion_markers(log_path):
        status = "missing_completion_markers"
        error = "run_metrics_or_result_marker_missing"
    elif status == "ok" and has_fatal_log_tokens(log_path):
        status = "fatal_log_error"
        error = "fatal_error_detected_in_log"
    elif status == "ok" and not log_matches_result_path(log_path, result_path):
        status = "log_result_mismatch"
        error = "log_result_path_mismatch"
    if not error and status != "ok":
        error = pair60.extract_error_tail(log_path)
    return stage1.build_summary_payload(spec, gpu_id, status, result_path, log_path, elapsed_sec, error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session-only v4 follow-up for weak/high-priority baseline_2 models on foursquare, movielens1m, and lastfm0.03.")
    parser.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"))
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--search-algo", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--resume", dest="run_mode", action="store_const", const="resume")
    parser.add_argument("--fresh", dest="run_mode", action="store_const", const="fresh")
    parser.add_argument("--no-resume", dest="run_mode", action="store_const", const="fresh")
    parser.set_defaults(run_mode="auto")
    return parser.parse_args()


def prepare_output_root(run_mode: str, dry_run: bool) -> bool:
    if dry_run:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return False

    if run_mode == "resume":
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return True

    if run_mode == "auto":
        if SUMMARY_CSV.exists():
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return True
        if not OUTPUT_ROOT.exists():
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return False
        try:
            has_existing_files = any(OUTPUT_ROOT.iterdir())
        except FileNotFoundError:
            has_existing_files = False
        if not has_existing_files:
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return False
    else:
        if not OUTPUT_ROOT.exists():
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
            return False

    backup_root = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_ROOT.rename(backup_root)
    print(f"[baseline2_addtuning5] archived previous output to {backup_root}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return False


def build_all_specs(args: argparse.Namespace) -> list[stage1.ComboSpec]:
    history = load_history_rows()
    runtimes = stage1.median_runtime_by_target(history)
    filters = stage1.parse_target_filter(args.targets)

    specs: list[stage1.ComboSpec] = []
    combo_seed_base = 2026042600
    for idx, target in enumerate(target_specs()):
        if filters and (target.dataset, target.model) not in filters:
            continue
        rows = history.get((target.dataset, target.model), [])
        if not rows:
            continue
        est_runtime_sec = runtimes.get((target.dataset, target.model), 600.0)
        specs.extend(build_combo_specs(target, rows, est_runtime_sec, combo_seed_base + idx * 100))

    specs.sort(key=lambda spec: (PRIORITY_RANK.get(spec.priority, 9), spec.est_runtime_sec, spec.model, spec.combo_id))
    if args.limit_jobs and args.limit_jobs > 0:
        specs = specs[: int(args.limit_jobs)]
    return specs


def main() -> None:
    args = parse_args()
    gpus = stage1.parse_csv_list(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified. Use --gpus 0,1,...")

    resume_mode = prepare_output_root(args.run_mode, args.dry_run)
    specs = build_all_specs(args)

    plan = plan_rows(specs)
    stage1.write_csv(
        PLAN_CSV,
        plan,
        [
            "dataset",
            "model",
            "combo_id",
            "combo_kind",
            "priority",
            "est_runtime_sec",
            "runtime_class",
            "max_evals",
            "epochs",
            "patience",
            "seed",
            "run_phase",
            "space_yaml",
            "base_params_json",
            "search_json",
        ],
    )

    manifest = {
        "created_at": stage1.now_utc(),
        "gpus": gpus,
        "python_bin": str(args.python_bin),
        "job_count": len(specs),
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "plan_csv": str(PLAN_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "space_root": str(SPACE_ROOT),
        "session_log_root": str(SESSION_LOG_ROOT),
        "dry_run": bool(args.dry_run),
        "run_mode": args.run_mode,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[baseline2_addtuning5] python={args.python_bin}")
    print(f"[baseline2_addtuning5] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
    if args.dry_run:
        for row in plan[: min(32, len(plan))]:
            print(
                f"  {row['dataset']} {row['model']} {row['combo_id']} kind={row['combo_kind']} "
                f"runtime={row['est_runtime_sec']}s max_evals={row['max_evals']}"
            )
        return

    summary_fields = [
        "dataset",
        "model",
        "combo_id",
        "combo_kind",
        "priority",
        "run_phase",
        "gpu_id",
        "status",
        "best_valid_mrr20",
        "test_mrr20",
        "valid_unseen_mrr20",
        "test_unseen_mrr20",
        "test_main_seen_count",
        "test_main_unseen_count",
        "est_runtime_sec",
        "elapsed_sec",
        "max_evals",
        "epochs",
        "patience",
        "seed",
        "result_path",
        "log_path",
        "space_yaml",
        "error",
        "timestamp_utc",
        "base_params_json",
    ]

    existing = stage1.read_existing_summary(SUMMARY_CSV) if resume_mode else {}
    reusable_count = 0
    remaining: list[stage1.ComboSpec] = []
    for spec in specs:
        reusable = find_reusable_artifact(spec, existing.get(spec.run_phase)) if resume_mode else None
        if reusable is not None:
            reusable_count += 1
            continue
        remaining.append(spec)
    print(f"[baseline2_addtuning5] reusable_completed={reusable_count} remaining={len(remaining)}")
    if not remaining:
        print(f"[baseline2_addtuning5] nothing to run; summary={SUMMARY_CSV}")
        return

    job_queue: Queue[stage1.ComboSpec] = Queue()
    for spec in remaining:
        job_queue.put(spec)

    write_lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                spec = job_queue.get_nowait()
            except Empty:
                return
            print(
                f"[baseline2_addtuning5] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                stage1.append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning5] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
                f"gpu={gpu_id} status={row['status']} elapsed={row['elapsed_sec']}"
            )
            job_queue.task_done()

    threads: list[threading.Thread] = []
    for gpu_id in gpus:
        thread = threading.Thread(target=worker, args=(gpu_id,), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    print(f"[baseline2_addtuning5] completed summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()