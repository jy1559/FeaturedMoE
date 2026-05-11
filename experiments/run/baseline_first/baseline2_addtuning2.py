#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import baseline2_addtuning as stage1
import run_pair60_campaign as pair60


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
ARTIFACT_ROOT = EXP_DIR / "run" / "artifacts"
TRACK = "baseline_2"
AXIS = "PAIR60_ADDTUNING2"

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
    *stage1.SUMMARY_SOURCES,
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_ADDTUNING" / "summary.csv",
]

TARGETS = [
    ("beauty", "duorec", "critical"),
    ("beauty", "fearec", "critical"),
    ("beauty", "fame", "critical"),
    ("beauty", "bsarec", "critical"),
    ("beauty", "difsr", "critical"),
    ("foursquare", "duorec", "high"),
    ("foursquare", "difsr", "high"),
    ("foursquare", "bsarec", "high"),
    ("foursquare", "fame", "high"),
    ("retail_rocket", "fdsa", "high"),
    ("retail_rocket", "bsarec", "high"),
    ("retail_rocket", "fame", "high"),
    ("KuaiRecLargeStrictPosV2_0.2", "fdsa", "high"),
]

OPTIONAL_TARGETS = [
    ("beauty", "gru4rec", "optional"),
    ("foursquare", "gru4rec", "optional"),
]

PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "optional": 2,
}

FAST_DATASET_BONUS = {
    "beauty": -3,
    "retail_rocket": -1,
    "foursquare": 0,
    "KuaiRecLargeStrictPosV2_0.2": 1,
}


@dataclass(frozen=True)
class HistSeed:
    dataset: str
    model: str
    axis: str
    run_phase: str
    best_valid_mrr20: float
    test_mrr20: float
    test_main_seen_count: int
    elapsed_sec: float
    lr_lo: float
    lr_hi: float
    params: dict[str, Any]
    selection_score: float


@dataclass(frozen=True)
class TargetStats:
    best_test_mrr20: float
    best_valid_mrr20: float
    ratio_to_dataset_best: float
    valid_test_gap: float
    seen_count: int


def target_specs(include_optional: bool) -> list[stage1.TargetSpec]:
    items = [stage1.TargetSpec(dataset, model, priority) for dataset, model, priority in TARGETS]
    if include_optional:
        items.extend(stage1.TargetSpec(dataset, model, priority) for dataset, model, priority in OPTIONAL_TARGETS)
    return items


def choose_params_json(row: dict[str, str]) -> str:
    for key in ("base_params_json", "params_json"):
        text = str(row.get(key, "")).strip()
        if text:
            return text
    return ""


def selection_score(dataset: str, best_valid: float, test_mrr20: float, seen_count: int) -> float:
    gap = max(best_valid - test_mrr20, 0.0)
    gap_weight = 0.95 if dataset == "beauty" else 0.6
    seen_bonus = 0.002 if seen_count > 0 else -0.02
    return test_mrr20 + (0.35 * best_valid) - (gap_weight * gap) + seen_bonus


def load_history_rows() -> dict[tuple[str, str], list[HistSeed]]:
    grouped: dict[tuple[str, str], list[HistSeed]] = {}
    for summary_path in SUMMARY_SOURCES:
        if not summary_path.exists():
            continue
        axis = summary_path.parent.name
        for row in stage1.read_csv(summary_path):
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            dataset = str(row.get("dataset", "")).strip()
            model = str(row.get("model", "")).strip().lower()
            params_text = choose_params_json(row)
            if not dataset or not model or not params_text:
                continue
            try:
                params = json.loads(params_text)
            except Exception:
                continue
            best_valid = stage1.safe_float(row.get("best_valid_mrr20", 0.0), 0.0)
            test_mrr20 = stage1.safe_float(row.get("test_mrr20", 0.0), 0.0)
            seen_count = stage1.safe_int(row.get("test_main_seen_count", 0), 0)
            elapsed_sec = stage1.safe_float(row.get("elapsed_sec", row.get("est_runtime_sec", 0.0)), 0.0)
            seed = HistSeed(
                dataset=dataset,
                model=model,
                axis=axis,
                run_phase=str(row.get("run_phase", "")).strip(),
                best_valid_mrr20=best_valid,
                test_mrr20=test_mrr20,
                test_main_seen_count=seen_count,
                elapsed_sec=elapsed_sec,
                lr_lo=stage1.safe_float(row.get("lr_lo", 0.0), 0.0),
                lr_hi=stage1.safe_float(row.get("lr_hi", 0.0), 0.0),
                params=params,
                selection_score=selection_score(dataset, best_valid, test_mrr20, seen_count),
            )
            grouped.setdefault((dataset, model), []).append(seed)

    for key, rows in grouped.items():
        rows.sort(key=lambda item: (item.selection_score, item.test_mrr20, item.best_valid_mrr20), reverse=True)
    return grouped


def normalize_params(model: str, params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    hidden = stage1.safe_int(out.get("hidden_size", out.get("embedding_size", 128)), 128)
    embed = stage1.safe_int(out.get("embedding_size", hidden), hidden)
    layers = stage1.safe_int(out.get("num_layers", out.get("n_layers", 2)), 2)
    heads = stage1.safe_int(out.get("num_heads", out.get("n_heads", 4)), 4)

    out["hidden_size"] = hidden
    out["embedding_size"] = embed
    out["inner_size"] = stage1.safe_int(out.get("inner_size", hidden * 2), hidden * 2)
    out["max_len"] = stage1.safe_int(out.get("max_len", 20), 20)
    out["dropout"] = stage1.safe_float(out.get("dropout", 0.15), 0.15)
    out["weight_decay"] = stage1.safe_float(out.get("weight_decay", 1e-4), 1e-4)

    if model in {"duorec", "fearec", "fdsa", "tisasrec"}:
        out["n_layers"] = layers
        out["n_heads"] = heads
    out["num_layers"] = layers
    out["num_heads"] = heads

    if model in {"duorec", "fearec"}:
        out["contrast"] = out.get("contrast", "un")
        out["tau"] = stage1.safe_float(out.get("tau", 0.2), 0.2)
        out["lmd"] = stage1.safe_float(out.get("lmd", 0.04), 0.04)
        out["lmd_sem"] = stage1.safe_float(out.get("lmd_sem", 0.0), 0.0)
        out["semantic_sample_max_tries"] = stage1.safe_int(out.get("semantic_sample_max_tries", 2), 2)
    if model == "fearec":
        out["global_ratio"] = stage1.safe_float(out.get("global_ratio", 0.85), 0.85)
    if model == "bsarec":
        out["bsarec_alpha"] = stage1.safe_float(out.get("bsarec_alpha", 0.5), 0.5)
        out["bsarec_c"] = stage1.safe_int(out.get("bsarec_c", 3), 3)
    if model == "fame":
        out["num_experts"] = stage1.safe_int(out.get("num_experts", 3), 3)
    if model in {"difsr", "fdsa"}:
        out["attribute_hidden_size"] = stage1.safe_int(out.get("attribute_hidden_size", hidden), hidden)
        out["fusion_type"] = str(out.get("fusion_type", "sum")).lower()
        out["lambda_attr"] = stage1.safe_float(out.get("lambda_attr", 0.1), 0.1)
        out["use_attribute_predictor"] = bool(out.get("use_attribute_predictor", True))
    if model == "fdsa":
        out["selected_features"] = out.get("selected_features", ["category"])
        out["pooling_mode"] = out.get("pooling_mode", "mean")
    return out


def summarize_target(rows: list[HistSeed], dataset_best_test: float) -> TargetStats:
    best_test = max(rows, key=lambda item: (item.test_mrr20, item.selection_score, item.best_valid_mrr20))
    best_valid = max(rows, key=lambda item: (item.best_valid_mrr20, item.test_mrr20, item.selection_score))
    ratio = (best_test.test_mrr20 / dataset_best_test) if dataset_best_test > 0 else 0.0
    return TargetStats(
        best_test_mrr20=best_test.test_mrr20,
        best_valid_mrr20=best_valid.best_valid_mrr20,
        ratio_to_dataset_best=ratio,
        valid_test_gap=max(best_valid.best_valid_mrr20 - best_test.test_mrr20, 0.0),
        seen_count=best_test.test_main_seen_count,
    )


def combo_count_for_target(target: stage1.TargetSpec, est_runtime_sec: float, stats: TargetStats) -> int:
    speed = stage1.runtime_class(est_runtime_sec)
    if target.dataset == "beauty":
        if speed in {"very_fast", "fast"}:
            base = 6
        if speed == "medium":
            base = 5
        else:
            base = 4
        if stats.ratio_to_dataset_best < 0.4:
            base += 2
        elif stats.ratio_to_dataset_best < 0.8:
            base += 1
        if stats.valid_test_gap > 0.025:
            base += 1
        return stage1.clamp_int(base, 4, 8)
    if speed == "very_fast":
        base = 5
    if speed == "fast":
        base = 4
    if speed == "medium":
        base = 3
    elif speed == "slow":
        base = 2
    else:
        base = 2
    if stats.ratio_to_dataset_best < 0.35 and est_runtime_sec <= 900:
        base += 2
    elif stats.ratio_to_dataset_best < 0.7 and est_runtime_sec <= 1200:
        base += 1
    elif stats.ratio_to_dataset_best >= 0.95 and stats.valid_test_gap <= 0.01:
        base -= 1
    return stage1.clamp_int(base, 2, 6)


def max_evals_for_target(target: stage1.TargetSpec, est_runtime_sec: float, stats: TargetStats) -> int:
    speed = stage1.runtime_class(est_runtime_sec)
    base = {
        "very_fast": 12,
        "fast": 10,
        "medium": 8,
        "slow": 6,
        "very_slow": 4,
    }[speed]
    if target.dataset == "beauty":
        base += 3 if speed in {"very_fast", "fast", "medium"} else 2
    if target.model in {"duorec", "fearec", "fame"} and target.dataset == "beauty":
        base += 1
    if stats.ratio_to_dataset_best < 0.4 and est_runtime_sec <= 400:
        base += 2
    elif stats.ratio_to_dataset_best < 0.8 and est_runtime_sec <= 900:
        base += 1
    if stats.valid_test_gap > 0.025 and est_runtime_sec <= 1200:
        base += 1
    if stats.ratio_to_dataset_best >= 0.95 and stats.valid_test_gap <= 0.01:
        base -= 1
    if target.dataset == "retail_rocket" and est_runtime_sec > 1000:
        base -= 1
    return stage1.clamp_int(base, 4, 16)


def shift_common(params: dict[str, Any], *, dropout_delta: float, weight_decay_mult: float, max_len_delta: int) -> dict[str, Any]:
    out = dict(params)
    out["dropout"] = round(stage1.clamp_float(stage1.safe_float(out.get("dropout", 0.15), 0.15) + dropout_delta, 0.05, 0.35), 4)
    wd = stage1.safe_float(out.get("weight_decay", 1e-4), 1e-4)
    out["weight_decay"] = round(stage1.clamp_float(max(wd, 1e-6) * weight_decay_mult, 1e-6, 5e-3), 8)
    out["max_len"] = stage1.clamp_int(stage1.safe_int(out.get("max_len", 20), 20) + max_len_delta, 10, 100)
    return out


def make_aggressive_variant(dataset: str, model: str, base: dict[str, Any], est_runtime_sec: float) -> dict[str, Any]:
    params = shift_common(base, dropout_delta=-0.015 if dataset != "beauty" else -0.005, weight_decay_mult=0.75, max_len_delta=5 if est_runtime_sec <= 600 else 2)
    layers = stage1.safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)

    if model in {"duorec", "fearec"}:
        params["contrast"] = "su" if dataset == "beauty" else "us_x"
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * 1.15, 0.05, 0.5), 4)
        params["lmd_sem"] = round(stage1.clamp_float(stage1.safe_float(params.get("lmd_sem", 0.0), 0.0) + 0.02, 0.0, 0.2), 4)
        if model == "fearec":
            params["global_ratio"] = round(stage1.clamp_float(stage1.safe_float(params.get("global_ratio", 0.85), 0.85) * 0.95, 0.6, 1.1), 4)
    elif model == "fame":
        params["num_experts"] = stage1.clamp_int(stage1.safe_int(params.get("num_experts", 3), 3) + (1 if dataset == "beauty" else 0), 2, 6)
        params["num_layers"] = stage1.clamp_int(layers - (1 if dataset == "beauty" else 0), 1, 3)
    elif model == "bsarec":
        params["bsarec_alpha"] = round(stage1.clamp_float(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5) + 0.12, 0.2, 0.9), 4)
        params["bsarec_c"] = 2 if dataset == "beauty" else 3
    elif model in {"difsr", "fdsa"}:
        params["fusion_type"] = "gate" if str(params.get("fusion_type", "sum")).lower() != "gate" else "concat"
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.03, 0.0, 0.3), 4)
        params["use_attribute_predictor"] = True
        if dataset == "beauty":
            params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    return normalize_params(model, params)


def make_conservative_variant(dataset: str, model: str, base: dict[str, Any]) -> dict[str, Any]:
    params = shift_common(base, dropout_delta=0.03 if dataset == "beauty" else 0.02, weight_decay_mult=1.8, max_len_delta=-6 if dataset == "beauty" else -3)
    layers = stage1.safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)

    if model in {"duorec", "fearec"}:
        params["contrast"] = "un"
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * 0.9, 0.05, 0.5), 4)
        params["lmd_sem"] = round(stage1.clamp_float(max(0.02, stage1.safe_float(params.get("lmd_sem", 0.0), 0.0)), 0.0, 0.2), 4)
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
        if model == "fearec":
            params["global_ratio"] = round(stage1.clamp_float(stage1.safe_float(params.get("global_ratio", 0.85), 0.85) * 0.9, 0.6, 1.1), 4)
    elif model == "fame":
        params["num_experts"] = stage1.clamp_int(stage1.safe_int(params.get("num_experts", 3), 3) - 1, 2, 6)
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    elif model == "bsarec":
        params["bsarec_alpha"] = round(stage1.clamp_float(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5) - 0.1, 0.2, 0.9), 4)
        params["bsarec_c"] = 4
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    elif model in {"difsr", "fdsa"}:
        params["fusion_type"] = "gate"
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.015, 0.0, 0.3), 4)
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    return normalize_params(model, params)


def make_shock_variant(dataset: str, model: str, base: dict[str, Any], peer: dict[str, Any]) -> dict[str, Any]:
    params = normalize_params(model, pair60.build_exploration_params(dataset, model, dict(base), dict(peer)))
    if dataset == "beauty":
        params = shift_common(params, dropout_delta=0.02, weight_decay_mult=1.5, max_len_delta=-4)
    else:
        params = shift_common(params, dropout_delta=0.0, weight_decay_mult=1.1, max_len_delta=2)

    if model in {"duorec", "fearec"}:
        params["contrast"] = "us_x" if dataset == "beauty" else "su"
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * 1.25, 0.05, 0.5), 4)
        params["lmd"] = round(stage1.clamp_float(max(0.02, stage1.safe_float(params.get("lmd", 0.04), 0.04)), 0.0, 0.2), 4)
    elif model == "fame":
        params["num_experts"] = stage1.clamp_int(stage1.safe_int(params.get("num_experts", 3), 3) + 1, 2, 6)
        params["num_layers"] = 1 if dataset == "beauty" else stage1.safe_int(params.get("num_layers", 2), 2)
    elif model == "bsarec":
        params["bsarec_alpha"] = 0.7 if dataset == "beauty" else 0.6
        params["bsarec_c"] = 2
    elif model in {"difsr", "fdsa"}:
        params["fusion_type"] = "concat"
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.05, 0.0, 0.3), 4)
        params["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(params.get("attribute_hidden_size", params.get("hidden_size", 128)), 128), 96, 192)
    return normalize_params(model, params)


def make_low_reg_variant(dataset: str, model: str, base: dict[str, Any], est_runtime_sec: float) -> dict[str, Any]:
    params = shift_common(base, dropout_delta=-0.03 if dataset == "beauty" else -0.02, weight_decay_mult=0.5, max_len_delta=6 if est_runtime_sec <= 500 else 3)
    layers = stage1.safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)
    if model in {"duorec", "fearec"}:
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * 1.2, 0.05, 0.5), 4)
        params["lmd_sem"] = round(stage1.clamp_float(stage1.safe_float(params.get("lmd_sem", 0.0), 0.0) + 0.03, 0.0, 0.2), 4)
        params["contrast"] = "su"
        params["num_layers"] = stage1.clamp_int(max(layers, 2), 1, 3)
    elif model == "fame":
        params["num_experts"] = stage1.clamp_int(stage1.safe_int(params.get("num_experts", 3), 3) + 1, 2, 6)
        params["num_layers"] = stage1.clamp_int(max(layers, 2), 1, 3)
    elif model == "bsarec":
        params["bsarec_alpha"] = round(stage1.clamp_float(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5) + 0.16, 0.2, 0.9), 4)
        params["bsarec_c"] = 2
    elif model in {"difsr", "fdsa"}:
        params["fusion_type"] = "concat"
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.04, 0.0, 0.3), 4)
        params["num_layers"] = stage1.clamp_int(max(layers, 2), 1, 3)
    return normalize_params(model, params)


def make_high_reg_variant(dataset: str, model: str, base: dict[str, Any]) -> dict[str, Any]:
    params = shift_common(base, dropout_delta=0.045 if dataset == "beauty" else 0.03, weight_decay_mult=2.4, max_len_delta=-8 if dataset == "beauty" else -4)
    layers = stage1.safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)
    if model in {"duorec", "fearec"}:
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * 0.85, 0.05, 0.5), 4)
        params["contrast"] = "un"
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    elif model == "fame":
        params["num_experts"] = 2
        params["num_layers"] = 1
    elif model == "bsarec":
        params["bsarec_alpha"] = round(stage1.clamp_float(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5) - 0.16, 0.2, 0.9), 4)
        params["bsarec_c"] = 5
        params["num_layers"] = stage1.clamp_int(layers - 1, 1, 3)
    elif model in {"difsr", "fdsa"}:
        params["fusion_type"] = "gate"
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.02, 0.0, 0.3), 4)
        params["num_layers"] = 1
    return normalize_params(model, params)


def make_micro_variant(dataset: str, model: str, base: dict[str, Any], direction: str) -> dict[str, Any]:
    delta = 0.01 if direction == "up" else -0.01
    weight_mult = 0.85 if direction == "up" else 1.25
    len_delta = 2 if direction == "up" else -2
    params = shift_common(base, dropout_delta=delta, weight_decay_mult=weight_mult, max_len_delta=len_delta)
    if model in {"duorec", "fearec"}:
        params["tau"] = round(stage1.clamp_float(stage1.safe_float(params.get("tau", 0.2), 0.2) * (1.08 if direction == "up" else 0.92), 0.05, 0.5), 4)
    elif model == "fame":
        params["num_experts"] = stage1.clamp_int(stage1.safe_int(params.get("num_experts", 3), 3) + (1 if direction == "up" else 0), 2, 6)
    elif model == "bsarec":
        params["bsarec_alpha"] = round(stage1.clamp_float(stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5) + (0.05 if direction == "up" else -0.05), 0.2, 0.9), 4)
    elif model in {"difsr", "fdsa"}:
        params["lambda_attr"] = round(stage1.clamp_float(stage1.safe_float(params.get("lambda_attr", 0.1), 0.1) + (0.02 if direction == "up" else -0.01), 0.0, 0.3), 4)
    return normalize_params(model, params)


def dedupe_candidates(candidates: list[tuple[str, dict[str, Any], HistSeed | None]]) -> list[tuple[str, dict[str, Any], HistSeed | None]]:
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any], HistSeed | None]] = []
    for combo_kind, params, row in candidates:
        normalized = normalize_params(row.model if row else "", params) if row else dict(params)
        sig = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((combo_kind, normalized, row))
    return out


def choose_seed_rows(rows: list[HistSeed]) -> tuple[HistSeed, HistSeed]:
    best_test = max(rows, key=lambda item: (item.test_mrr20, item.selection_score, item.best_valid_mrr20))
    best_balanced = rows[0]
    return best_test, best_balanced


def build_search_block(dataset: str, model: str, params: dict[str, Any], row: HistSeed | None, combo_kind: str, stats: TargetStats) -> tuple[dict[str, Any], dict[str, str]]:
    overrides: dict[str, str] = {"learning_rate": "choice"}
    search: dict[str, Any] = {}

    lr_lo, lr_hi, lr_center = stage1.center_lr(row, params, dataset, model)
    if dataset == "beauty":
        lr_factors = [0.4, 0.55, 0.72, 0.88, 1.0, 1.15, 1.35, 1.65, 2.0]
    elif stats.ratio_to_dataset_best >= 0.95 and stats.valid_test_gap <= 0.015:
        lr_factors = [0.72, 0.88, 1.0, 1.15, 1.35]
    else:
        lr_factors = [0.55, 0.72, 0.88, 1.0, 1.15, 1.35, 1.6]
    search["learning_rate"] = stage1.around_float(lr_center, lr_factors, max(1e-5, lr_lo * 0.85), min(2e-2, lr_hi * 1.2), digits=8)

    wd = stage1.safe_float(params.get("weight_decay", 1e-4), 1e-4)
    wd_choices = sorted(
        set(
            [
                round(stage1.clamp_float(max(wd, 1e-6) * 0.8, 1e-6, 5e-3), 8),
                round(stage1.clamp_float(max(wd, 1e-6) * 1.4, 1e-6, 5e-3), 8),
            ]
        )
    )
    if dataset == "beauty" or combo_kind in {"conservative", "shock", "high_reg", "low_reg"}:
        search["weight_decay"] = wd_choices
        overrides["weight_decay"] = "choice"

    max_len = stage1.safe_int(params.get("max_len", 20), 20)
    if dataset == "beauty":
        delta = -4 if combo_kind == "conservative" else 4
        max_len_choices = sorted(set([max_len, stage1.clamp_int(max_len + delta, 10, 80)]))
        if len(max_len_choices) > 1:
            search["MAX_ITEM_LIST_LENGTH"] = max_len_choices
            overrides["MAX_ITEM_LIST_LENGTH"] = "choice"

    if model in {"duorec", "fearec"}:
        tau = stage1.safe_float(params.get("tau", 0.2), 0.2)
        tau_choices = sorted(
            set(
                [
                    round(tau, 4),
                    round(stage1.clamp_float(tau * (0.9 if combo_kind == "conservative" else 1.15), 0.05, 0.5), 4),
                ]
            )
        )
        if len(tau_choices) > 1:
            search["tau"] = tau_choices
            overrides["tau"] = "choice"
        if combo_kind in {"shock", "aggressive", "low_reg"}:
            contrast_choices = sorted(set([str(params.get("contrast", "un")), "un", "su"]))
            if len(contrast_choices) > 1:
                search["contrast"] = contrast_choices[:2]
                overrides["contrast"] = "choice"
    elif model == "fame":
        experts = stage1.safe_int(params.get("num_experts", 3), 3)
        expert_choices = sorted(set([experts, stage1.clamp_int(experts + (1 if combo_kind != "conservative" else -1), 2, 6)]))
        if len(expert_choices) > 1:
            search["num_experts"] = expert_choices
            overrides["num_experts"] = "choice"
    elif model == "bsarec":
        alpha = stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5)
        alpha_choices = sorted(set([round(alpha, 4), round(stage1.clamp_float(alpha + (0.08 if combo_kind != "conservative" else -0.08), 0.2, 0.9), 4)]))
        if len(alpha_choices) > 1:
            search["bsarec_alpha"] = alpha_choices
            overrides["bsarec_alpha"] = "choice"
    elif model in {"difsr", "fdsa"}:
        lambda_attr = stage1.safe_float(params.get("lambda_attr", 0.1), 0.1)
        lambda_choices = sorted(set([round(lambda_attr, 4), round(stage1.clamp_float(lambda_attr + 0.025, 0.0, 0.3), 4)]))
        if len(lambda_choices) > 1:
            search["lambda_attr"] = lambda_choices
            overrides["lambda_attr"] = "choice"
        if combo_kind in {"shock", "aggressive", "low_reg"}:
            fusion_choices = sorted(set([str(params.get("fusion_type", "sum")), "gate", "concat"]))
            if len(fusion_choices) > 1:
                search["fusion_type"] = fusion_choices[:2]
                overrides["fusion_type"] = "choice"

    return search, overrides


def build_combo_specs(target: stage1.TargetSpec, rows: list[HistSeed], est_runtime_sec: float, combo_seed_base: int, stats: TargetStats) -> list[stage1.ComboSpec]:
    desired_count = combo_count_for_target(target, est_runtime_sec, stats)
    best_test, best_balanced = choose_seed_rows(rows)

    base_test = normalize_params(target.model, best_test.params)
    base_balanced = normalize_params(target.model, best_balanced.params)

    weak_target = stats.ratio_to_dataset_best < 0.75
    very_weak_target = stats.ratio_to_dataset_best < 0.4
    overfit_target = stats.valid_test_gap > 0.02

    candidates: list[tuple[str, dict[str, Any], HistSeed | None]] = [("test_best", base_test, best_test)]

    if not very_weak_target:
        candidates.append(("balanced", base_balanced, best_balanced))
    else:
        candidates.append(("low_reg", make_low_reg_variant(target.dataset, target.model, base_test, est_runtime_sec), best_test))

    candidates.append(("aggressive", make_aggressive_variant(target.dataset, target.model, base_test, est_runtime_sec), best_test))
    candidates.append(("conservative", make_conservative_variant(target.dataset, target.model, base_balanced), best_balanced))

    if weak_target or overfit_target or target.dataset == "beauty":
        candidates.append(("shock", make_shock_variant(target.dataset, target.model, base_test, base_balanced), best_test))
    if very_weak_target:
        candidates.append(("high_reg", make_high_reg_variant(target.dataset, target.model, base_balanced), best_balanced))
        candidates.append(("alt_shock", make_shock_variant(target.dataset, target.model, base_balanced, base_test), best_balanced))
    if stats.ratio_to_dataset_best >= 0.9:
        candidates.append(("micro_up", make_micro_variant(target.dataset, target.model, base_test, "up"), best_test))
    if stats.valid_test_gap >= 0.015 or very_weak_target:
        candidates.append(("micro_down", make_micro_variant(target.dataset, target.model, base_balanced, "down"), best_balanced))

    uniq = dedupe_candidates(candidates)
    specs: list[stage1.ComboSpec] = []
    SPACE_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, (combo_kind, params, anchor_row) in enumerate(uniq[:desired_count], start=1):
        combo_id = f"K{idx}"
        search, type_overrides = build_search_block(target.dataset, target.model, params, anchor_row, combo_kind, stats)
        fixed = stage1.make_fixed_block(target.model, params)
        fixed["search_space_type_overrides"] = type_overrides

        dataset_tag = stage1.sanitize_token(target.dataset)
        model_tag = stage1.sanitize_token(target.model)
        space_yaml = SPACE_ROOT / f"{dataset_tag}_{model_tag}_{combo_id}.yaml"
        space_yaml.write_text(json.dumps({"fixed": fixed, "search": search}, indent=2) + "\n", encoding="utf-8")

        specs.append(
            stage1.ComboSpec(
                dataset=target.dataset,
                model=target.model,
                combo_id=combo_id,
                combo_kind=combo_kind,
                priority=target.priority,
                est_runtime_sec=est_runtime_sec,
                max_evals=max_evals_for_target(target, est_runtime_sec, stats),
                seed=combo_seed_base + idx,
                base_params=params,
                search_space=search,
                fixed_space=fixed,
                space_yaml=space_yaml,
                run_phase=f"BASELINE2_ADDTUNE2_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
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
            proc = subprocess.Popen(
                cmd,
                cwd=str(EXP_DIR),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
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
    parser = argparse.ArgumentParser(description="Stage-2 targeted baseline_2 hyperopt tuning focused on seen-target test lift.")
    parser.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"))
    parser.add_argument("--include-optional", action="store_true", default=False)
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--search-algo", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def build_all_specs(args: argparse.Namespace) -> list[stage1.ComboSpec]:
    history = load_history_rows()
    runtimes = stage1.median_runtime_by_target(history)
    filters = stage1.parse_target_filter(args.targets)
    targets = target_specs(args.include_optional)
    dataset_best_test: dict[str, float] = {}
    for (dataset, _model), rows in history.items():
        best = max((row.test_mrr20 for row in rows), default=0.0)
        dataset_best_test[dataset] = max(dataset_best_test.get(dataset, 0.0), best)

    specs: list[stage1.ComboSpec] = []
    combo_seed_base = 2026041600
    for idx, target in enumerate(targets):
        if filters and (target.dataset, target.model) not in filters:
            continue
        rows = history.get((target.dataset, target.model), [])
        if not rows:
            continue
        est_runtime_sec = runtimes.get((target.dataset, target.model), 600.0)
        stats = summarize_target(rows, dataset_best_test.get(target.dataset, 0.0))
        specs.extend(build_combo_specs(target, rows, est_runtime_sec, combo_seed_base + idx * 20, stats))

    specs.sort(
        key=lambda spec: (
            PRIORITY_RANK.get(spec.priority, 9),
            FAST_DATASET_BONUS.get(spec.dataset, 9),
            spec.est_runtime_sec,
            spec.dataset,
            spec.model,
            spec.combo_id,
        )
    )
    if args.limit_jobs and args.limit_jobs > 0:
        specs = specs[: int(args.limit_jobs)]
    return specs


def main() -> None:
    args = parse_args()
    gpus = stage1.parse_csv_list(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified. Use --gpus 0,1,...")

    specs = build_all_specs(args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

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
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[baseline2_addtuning2] python={args.python_bin}")
    print(f"[baseline2_addtuning2] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
    if args.dry_run:
        for row in plan[: min(16, len(plan))]:
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

    existing = stage1.read_existing_summary(SUMMARY_CSV) if args.resume else {}
    remaining = [spec for spec in specs if spec.run_phase not in existing or str(existing[spec.run_phase].get("status", "")).strip().lower() != "ok"]
    if not remaining:
        print(f"[baseline2_addtuning2] nothing to run; summary={SUMMARY_CSV}")
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
                f"[baseline2_addtuning2] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                stage1.append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning2] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
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

    print(f"[baseline2_addtuning2] complete summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()