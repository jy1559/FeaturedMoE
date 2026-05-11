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
import run_pair60_campaign as pair60


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
ARTIFACT_ROOT = EXP_DIR / "run" / "artifacts"
TRACK = "baseline_2"
AXIS = "PAIR60_ADDTUNING4"

RESULTS_ROOT = ARTIFACT_ROOT / "results" / TRACK
OUTPUT_ROOT = ARTIFACT_ROOT / "logs" / TRACK / AXIS
SPACE_ROOT = OUTPUT_ROOT / "spaces"
SESSION_LOG_ROOT = OUTPUT_ROOT / "session_logs"
PLAN_CSV = OUTPUT_ROOT / "plan.csv"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
MANIFEST_JSON = OUTPUT_ROOT / "manifest.json"

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 7

SUMMARY_SOURCES = [
    *stage32.SUMMARY_SOURCES,
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_ADDTUNING3_2" / "summary.csv",
]

TARGETS = [
    ("retail_rocket", "gru4rec", "high"),
    ("retail_rocket", "tisasrec", "high"),
    ("retail_rocket", "fearec", "high"),
    ("retail_rocket", "duorec", "high"),
    ("retail_rocket", "bsarec", "high"),
    ("retail_rocket", "fame", "high"),
    ("retail_rocket", "difsr", "high"),
    ("retail_rocket", "fdsa", "high"),
]

PRIORITY_RANK = {"high": 0}

MODEL_MAX_EVALS = {
    "gru4rec": 10,
    "tisasrec": 10,
    "fearec": 10,
    "duorec": 10,
    "bsarec": 10,
    "fame": 9,
    "difsr": 11,
    "fdsa": 10,
}

MODEL_BATCH_OVERRIDES: dict[str, dict[str, int]] = {
    "fdsa": {"train_batch_size": 1024, "eval_batch_size": 2048},
}

MODEL_FAMILIES: dict[str, list[dict[str, Any]]] = {
    "gru4rec": [
        {"name": "moderate_short_1l", "anchor": "short_best", "scale": 0.95, "max_len": 10, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "short"},
        {"name": "moderate_mid_2l", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "mid"},
        {"name": "moderate_long_1l", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "long"},
        {"name": "challenge_wide_3l", "anchor": "best_test", "scale": 1.15, "max_len": 18, "layers": 3, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.9},
    ],
    "tisasrec": [
        {"name": "moderate_short_1l", "anchor": "short_best", "scale": 0.9, "max_len": 10, "layers": 1, "lr_mode": "wide_mid", "seq_mode": "short", "head_mode": "base"},
        {"name": "moderate_mid_2l", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "mid", "head_mode": "base"},
        {"name": "moderate_long_2l", "anchor": "valid_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "long", "head_mode": "base"},
        {"name": "challenge_deep_wide", "anchor": "best_test", "scale": 1.15, "max_len": 24, "layers": 3, "lr_mode": "wide_high", "seq_mode": "long", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9},
    ],
    "fearec": [
        {"name": "moderate_sem_mid", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "mid", "param_overrides": {"tau": 0.18, "lmd": 0.035, "lmd_sem": 0.08}},
        {"name": "moderate_short_reset", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "lr_mode": "wide_mid", "seq_mode": "short", "param_overrides": {"tau": 0.16, "lmd": 0.03, "lmd_sem": 0.05}},
        {"name": "moderate_long_stable", "anchor": "valid_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "long", "param_overrides": {"tau": 0.2, "lmd": 0.04, "lmd_sem": 0.1}},
        {"name": "challenge_wide_sem", "anchor": "best_test", "scale": 1.15, "max_len": 20, "layers": 3, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9, "param_overrides": {"tau": 0.22, "lmd": 0.05, "lmd_sem": 0.12}},
    ],
    "duorec": [
        {"name": "moderate_mid_contrast", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "mid", "param_overrides": {"tau": 0.18, "lmd": 0.04, "lmd_sem": 0.0}},
        {"name": "moderate_short_reset", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "lr_mode": "wide_mid", "seq_mode": "short", "param_overrides": {"tau": 0.16, "lmd": 0.03, "lmd_sem": 0.0}},
        {"name": "moderate_long_stable", "anchor": "valid_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "long", "param_overrides": {"tau": 0.2, "lmd": 0.045, "lmd_sem": 0.0}},
        {"name": "challenge_wide_contrast", "anchor": "best_test", "scale": 1.15, "max_len": 20, "layers": 3, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9, "param_overrides": {"tau": 0.22, "lmd": 0.055, "lmd_sem": 0.0}},
    ],
    "bsarec": [
        {"name": "moderate_alpha_hi_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "mid"},
        {"name": "moderate_alpha_mid_short", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "profile": "alpha_mid_c3", "lr_mode": "wide_mid", "seq_mode": "short"},
        {"name": "moderate_alpha_hi_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 1, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "long"},
        {"name": "challenge_alpha_peak_wide", "anchor": "valid_best", "scale": 1.15, "max_len": 18, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9},
    ],
    "fame": [
        {"name": "moderate_experts4_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "mid"},
        {"name": "moderate_experts3_short", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "profile": "experts3", "lr_mode": "wide_mid", "seq_mode": "short"},
        {"name": "moderate_experts4_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "long"},
        {"name": "challenge_experts6_long", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.01, "wd_mult": 1.05},
    ],
    "difsr": [
        {"name": "moderate_gate_mid", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gate_midattr", "lr_mode": "wide_mid", "seq_mode": "mid"},
        {"name": "moderate_concat_mid", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "concat_midattr", "lr_mode": "wide_mid", "seq_mode": "mid"},
        {"name": "moderate_gate_long", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "gate_highattr", "lr_mode": "wide_mid", "seq_mode": "long"},
        {"name": "challenge_highattr_wide", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "profile": "gate_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": 0.0, "wd_mult": 0.95},
    ],
    "fdsa": [
        {"name": "moderate_mid_attr", "anchor": "best_test", "scale": 1.0, "max_len": 16, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "mid", "param_overrides": {"attribute_hidden_size": 160, "lambda_attr": 0.12, "selected_features": ["category"], "pooling_mode": "mean"}},
        {"name": "moderate_short_attr", "anchor": "short_best", "scale": 0.95, "max_len": 12, "layers": 1, "lr_mode": "wide_mid", "seq_mode": "short", "param_overrides": {"attribute_hidden_size": 128, "lambda_attr": 0.1, "selected_features": ["category"], "pooling_mode": "mean"}},
        {"name": "moderate_long_attr", "anchor": "long_best", "scale": 1.0, "max_len": 20, "layers": 2, "lr_mode": "wide_mid", "seq_mode": "long", "param_overrides": {"attribute_hidden_size": 192, "lambda_attr": 0.14, "selected_features": ["category"], "pooling_mode": "mean"}},
        {"name": "challenge_wide_attr", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide", "dropout_delta": -0.01, "wd_mult": 0.9, "param_overrides": {"attribute_hidden_size": 224, "lambda_attr": 0.16, "selected_features": ["category"], "pooling_mode": "mean"}},
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
    extra_bands = {
        "tisasrec": {
            "reset_low": (5e-5, 1.5e-3),
            "wide_mid": (1e-4, 4e-3),
            "wide_high": (3e-4, 8e-3),
            "ultrawide": (5e-5, 1.2e-2),
        },
        "fearec": {
            "reset_low": (5e-5, 1.2e-3),
            "wide_mid": (1e-4, 3.5e-3),
            "wide_high": (3e-4, 8e-3),
            "ultrawide": (5e-5, 1.0e-2),
        },
        "duorec": {
            "reset_low": (5e-5, 1.2e-3),
            "wide_mid": (1e-4, 3.5e-3),
            "wide_high": (3e-4, 8e-3),
            "ultrawide": (5e-5, 1.0e-2),
        },
        "fdsa": {
            "reset_low": (5e-4, 3e-3),
            "wide_mid": (1e-3, 6e-3),
            "wide_high": (3e-3, 1.2e-2),
            "ultrawide": (1e-3, 1.8e-2),
        },
    }
    if model in extra_bands:
        return extra_bands[model][lr_mode]
    return stage32.base_lr_band(model, lr_mode)


def build_search_block(model: str, combo_kind: str, params: dict[str, Any], meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    overrides: dict[str, str] = {"learning_rate": "loguniform"}
    search: dict[str, Any] = {}

    lr_lo, lr_hi = base_lr_band(model, meta["lr_mode"])
    search["learning_rate"] = [round(lr_lo, 8), round(lr_hi, 8)]

    max_len = stage1.safe_int(params.get("max_len", 16), 16)
    tuned_axes = 0

    if any(token in combo_kind for token in ["short", "long", "mid"]):
        if meta["seq_mode"] == "short":
            seq_values = [max_len, max_len + 2, max_len + 4]
        elif meta["seq_mode"] == "long":
            seq_values = [max(10, max_len - 4), max_len, max_len + 4]
        else:
            seq_values = [max(8, max_len - 2), max_len, max_len + 2]
        search["MAX_ITEM_LIST_LENGTH"] = sorted({stage32.nearest_seq(value) for value in seq_values})
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"
        tuned_axes += 1
    else:
        wd = stage1.safe_float(params.get("weight_decay", 1e-4), 1e-4)
        search["weight_decay"] = sorted(
            {
                round(stage1.clamp_float(wd * 0.6, 1e-6, 5e-2), 8),
                round(stage1.clamp_float(wd, 1e-6, 5e-2), 8),
                round(stage1.clamp_float(wd * 1.8, 1e-6, 5e-2), 8),
            }
        )
        overrides["weight_decay"] = "choice"
        tuned_axes += 1

    if model in {"tisasrec", "gru4rec"} and tuned_axes < 2:
        layers = stage1.safe_int(params.get("num_layers", 2), 2)
        search["num_layers"] = sorted({1, 2, 3, 4, layers})
        overrides["num_layers"] = "choice"
        tuned_axes += 1
    elif model in {"fearec", "duorec"} and tuned_axes < 2:
        tau = stage1.safe_float(params.get("tau", 0.2), 0.2)
        search["tau"] = stage32.search_choices_around(tau, [tau - 0.04, tau, tau + 0.04], 0.05, 0.4)
        overrides["tau"] = "choice"
        tuned_axes += 1
    elif model == "fdsa" and tuned_axes < 2:
        attr_hidden = stage1.safe_int(params.get("attribute_hidden_size", 128), 128)
        search["attribute_hidden_size"] = sorted({96, attr_hidden, stage32.nearest_dim(attr_hidden * 1.2), 224})
        overrides["attribute_hidden_size"] = "choice"
        tuned_axes += 1

    if model == "bsarec" and tuned_axes < 2:
        search["bsarec_alpha"] = stage32.search_choices_around(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5), [0.35, 0.55, 0.78, 0.9], 0.1, 0.95)
        overrides["bsarec_alpha"] = "choice"
    elif model == "fame" and tuned_axes < 2:
        experts = stage1.safe_int(params.get("num_experts", 3), 3)
        search["num_experts"] = sorted({2, experts, 4, 6})
        overrides["num_experts"] = "choice"
    elif model == "difsr" and tuned_axes < 2:
        search["fusion_type"] = ["gate", "concat"]
        overrides["fusion_type"] = "choice"
    elif model == "fdsa" and tuned_axes < 2:
        lambda_attr = stage1.safe_float(params.get("lambda_attr", 0.1), 0.1)
        search["lambda_attr"] = stage32.search_choices_around(lambda_attr, [lambda_attr - 0.04, lambda_attr, lambda_attr + 0.04], 0.0, 0.3)
        overrides["lambda_attr"] = "choice"

    return search, overrides


def combo_count_for_target(_target: stage1.TargetSpec, _est_runtime_sec: float) -> int:
    return 4


def max_evals_for_target(target: stage1.TargetSpec, est_runtime_sec: float) -> int:
    base = MODEL_MAX_EVALS.get(target.model, 10)
    if est_runtime_sec > 240:
        base -= 1
    if est_runtime_sec < 90:
        base += 1
    return stage1.clamp_int(base, 8, 12)


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
                run_phase=f"BASELINE2_ADDTUNE4_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
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


def run_one(spec: stage1.ComboSpec, gpu_id: str, python_bin: str, search_algo: str) -> dict[str, Any]:
    SESSION_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = SESSION_LOG_ROOT / f"{spec.run_phase}.log"
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
    if not error and status != "ok":
        error = pair60.extract_error_tail(log_path)
    return stage1.build_summary_payload(spec, gpu_id, status, result_path, log_path, elapsed_sec, error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail_rocket local follow-up for 8 non-SASRec/non-RouteRec baseline_2 models.")
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
    print(f"[baseline2_addtuning4] archived previous output to {backup_root}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return False


def build_all_specs(args: argparse.Namespace) -> list[stage1.ComboSpec]:
    history = load_history_rows()
    runtimes = stage1.median_runtime_by_target(history)
    filters = stage1.parse_target_filter(args.targets)

    specs: list[stage1.ComboSpec] = []
    combo_seed_base = 2026042400
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

    print(f"[baseline2_addtuning4] python={args.python_bin}")
    print(f"[baseline2_addtuning4] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
    if args.dry_run:
        for row in plan[: min(24, len(plan))]:
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
    remaining = [
        spec
        for spec in specs
        if spec.run_phase not in existing or str(existing[spec.run_phase].get("status", "")).strip().lower() != "ok"
    ]
    if not remaining:
        print(f"[baseline2_addtuning4] nothing to run; summary={SUMMARY_CSV}")
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
                f"[baseline2_addtuning4] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                stage1.append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning4] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
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

    print(f"[baseline2_addtuning4] completed summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()