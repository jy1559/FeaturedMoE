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
import baseline2_addtuning2 as stage2
import run_pair60_campaign as pair60


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
ARTIFACT_ROOT = EXP_DIR / "run" / "artifacts"
TRACK = "baseline_2"
AXIS = "PAIR60_ADDTUNING3"

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
    *stage2.SUMMARY_SOURCES,
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_ADDTUNING2" / "summary.csv",
]

TARGETS = [
    ("beauty", "difsr", "critical"),
    ("beauty", "fame", "critical"),
    ("beauty", "bsarec", "critical"),
    ("beauty", "gru4rec", "critical"),
    ("beauty", "tisasrec", "critical"),
    ("beauty", "duorec", "high"),
    ("beauty", "fearec", "high"),
    ("beauty", "sasrec", "normal"),
    ("beauty", "fdsa", "normal"),
]

PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "normal": 2,
}

FAST_DATASET_BONUS = {
    "beauty": -3,
}

MODEL_COMBO_BUDGET = {
    "difsr": 12,
    "fame": 12,
    "bsarec": 12,
    "gru4rec": 10,
    "tisasrec": 10,
    "duorec": 9,
    "fearec": 9,
    "sasrec": 8,
    "fdsa": 8,
}

MODEL_EVAL_BONUS = {
    "difsr": 1,
    "fame": 1,
    "bsarec": 1,
    "gru4rec": 1,
    "tisasrec": 1,
    "duorec": 0,
    "fearec": 0,
    "sasrec": 0,
    "fdsa": 0,
}

DIM_GRID = [64, 80, 96, 112, 128, 160, 192, 224, 256]


def target_specs() -> list[stage1.TargetSpec]:
    return [stage1.TargetSpec(dataset, model, priority) for dataset, model, priority in TARGETS]


def choose_params_json(row: dict[str, str]) -> str:
    return stage2.choose_params_json(row)


def selection_score(dataset: str, best_valid: float, test_mrr20: float, seen_count: int) -> float:
    return stage2.selection_score(dataset, best_valid, test_mrr20, seen_count)


def load_history_rows() -> dict[tuple[str, str], list[stage2.HistSeed]]:
    grouped: dict[tuple[str, str], list[stage2.HistSeed]] = {}
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
            seed = stage2.HistSeed(
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
    return stage2.normalize_params(model, params)


def summarize_target(rows: list[stage2.HistSeed], dataset_best_test: float) -> stage2.TargetStats:
    return stage2.summarize_target(rows, dataset_best_test)


def nearest_dim(value: float) -> int:
    clipped = stage1.clamp_float(float(value), DIM_GRID[0], DIM_GRID[-1])
    return min(DIM_GRID, key=lambda candidate: abs(candidate - clipped))


def scale_dim(base: int, scale: float) -> int:
    return nearest_dim(base * scale)


def history_signatures(model: str, rows: list[stage2.HistSeed]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        out.add(json.dumps(normalize_params(model, row.params), sort_keys=True, ensure_ascii=True))
    return out


def adjust_arch(
    model: str,
    base: dict[str, Any],
    *,
    hidden: int | None = None,
    layers: int | None = None,
    heads: int | None = None,
    max_len: int | None = None,
    dropout_delta: float = 0.0,
    weight_decay_mult: float = 1.0,
) -> dict[str, Any]:
    params = dict(base)
    if hidden is not None:
        params["hidden_size"] = hidden
        params["embedding_size"] = hidden
        params["inner_size"] = hidden * 2
        if model in {"difsr", "fdsa"}:
            params["attribute_hidden_size"] = min(hidden, 192)
    if layers is not None:
        params["num_layers"] = stage1.clamp_int(layers, 1, 4)
        params["n_layers"] = stage1.clamp_int(layers, 1, 4)
    if heads is not None and model != "gru4rec":
        params["num_heads"] = stage1.clamp_int(heads, 2, 8)
        params["n_heads"] = stage1.clamp_int(heads, 2, 8)
    if max_len is not None:
        params["max_len"] = stage1.clamp_int(max_len, 10, 40)
    params["dropout"] = round(stage1.clamp_float(stage1.safe_float(params.get("dropout", 0.15), 0.15) + dropout_delta, 0.03, 0.4), 4)
    wd = stage1.safe_float(params.get("weight_decay", 1e-4), 1e-4)
    params["weight_decay"] = round(stage1.clamp_float(max(wd, 1e-6) * weight_decay_mult, 1e-6, 1e-2), 8)
    if model == "tisasrec":
        max_len_val = stage1.safe_int(params.get("max_len", 20), 20)
        if max_len_val <= 12:
            params["time_span"] = 64
        elif max_len_val <= 16:
            params["time_span"] = 128
        else:
            params["time_span"] = 256
    return normalize_params(model, params)


def decorate_variant(model: str, params: dict[str, Any], flavor: str) -> dict[str, Any]:
    out = dict(params)
    layers = stage1.safe_int(out.get("num_layers", out.get("n_layers", 2)), 2)
    max_len = stage1.safe_int(out.get("max_len", 20), 20)

    if model in {"duorec", "fearec"}:
        if flavor == "short_attack":
            out["contrast"] = "su"
            out["tau"] = round(stage1.clamp_float(stage1.safe_float(out.get("tau", 0.2), 0.2) * 1.4, 0.05, 0.6), 4)
            out["lmd"] = round(stage1.clamp_float(max(0.02, stage1.safe_float(out.get("lmd", 0.0), 0.0)), 0.0, 0.2), 4)
            out["lmd_sem"] = round(stage1.clamp_float(stage1.safe_float(out.get("lmd_sem", 0.0), 0.0) + 0.04, 0.0, 0.2), 4)
        elif flavor == "short_reg":
            out["contrast"] = "un"
            out["tau"] = round(stage1.clamp_float(stage1.safe_float(out.get("tau", 0.2), 0.2) * 0.8, 0.05, 0.6), 4)
            out["lmd_sem"] = round(stage1.clamp_float(max(0.02, stage1.safe_float(out.get("lmd_sem", 0.0), 0.0)), 0.0, 0.2), 4)
        elif flavor == "shock":
            out["contrast"] = "us_x"
            out["tau"] = round(stage1.clamp_float(stage1.safe_float(out.get("tau", 0.2), 0.2) * 1.7, 0.05, 0.7), 4)
            out["lmd"] = round(stage1.clamp_float(max(0.04, stage1.safe_float(out.get("lmd", 0.02), 0.02) + 0.03), 0.0, 0.2), 4)
            out["lmd_sem"] = round(stage1.clamp_float(stage1.safe_float(out.get("lmd_sem", 0.0), 0.0) + 0.06, 0.0, 0.2), 4)
        if model == "fearec":
            ratio = stage1.safe_float(out.get("global_ratio", 0.85), 0.85)
            if flavor == "short_attack":
                out["global_ratio"] = round(stage1.clamp_float(ratio * 0.85, 0.5, 1.1), 4)
            elif flavor == "short_reg":
                out["global_ratio"] = round(stage1.clamp_float(ratio * 0.8, 0.5, 1.1), 4)
            elif flavor == "shock":
                out["global_ratio"] = round(stage1.clamp_float(ratio * 0.92, 0.5, 1.1), 4)
    elif model == "fame":
        experts = stage1.safe_int(out.get("num_experts", 3), 3)
        if flavor == "short_attack":
            out["num_experts"] = stage1.clamp_int(experts + 1, 2, 6)
            out["num_layers"] = 1 if max_len <= 12 else stage1.clamp_int(layers, 1, 3)
        elif flavor == "short_reg":
            out["num_experts"] = 2
            out["num_layers"] = 1
        elif flavor == "shock":
            out["num_experts"] = stage1.clamp_int(experts + 2, 2, 6)
            out["num_layers"] = 1
    elif model == "bsarec":
        alpha = stage1.safe_float(out.get("bsarec_alpha", 0.5), 0.5)
        if flavor == "short_attack":
            out["bsarec_alpha"] = round(stage1.clamp_float(alpha + 0.18, 0.2, 0.9), 4)
            out["bsarec_c"] = 2
        elif flavor == "short_reg":
            out["bsarec_alpha"] = round(stage1.clamp_float(alpha - 0.15, 0.2, 0.9), 4)
            out["bsarec_c"] = 5
        elif flavor == "shock":
            out["bsarec_alpha"] = 0.78
            out["bsarec_c"] = 2
    elif model in {"difsr", "fdsa"}:
        lambda_attr = stage1.safe_float(out.get("lambda_attr", 0.1), 0.1)
        if flavor == "short_attack":
            out["fusion_type"] = "concat"
            out["lambda_attr"] = round(stage1.clamp_float(lambda_attr + 0.06, 0.0, 0.3), 4)
        elif flavor == "short_reg":
            out["fusion_type"] = "gate"
            out["lambda_attr"] = round(stage1.clamp_float(lambda_attr + 0.02, 0.0, 0.3), 4)
            out["num_layers"] = 1
        elif flavor == "shock":
            out["fusion_type"] = "concat" if str(out.get("fusion_type", "sum")).lower() != "concat" else "gate"
            out["lambda_attr"] = round(stage1.clamp_float(max(lambda_attr, 0.16) + 0.05, 0.0, 0.3), 4)
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("attribute_hidden_size", 128), 128) + 32, 64, 224)
        out["use_attribute_predictor"] = True
    elif model == "gru4rec":
        if flavor == "short_attack":
            out["num_layers"] = stage1.clamp_int(max(1, layers - 1), 1, 4)
        elif flavor == "short_reg":
            out["num_layers"] = 1
        elif flavor == "shock":
            out["num_layers"] = stage1.clamp_int(min(4, layers + 1), 1, 4)
    elif model == "tisasrec":
        if flavor == "short_attack":
            out["time_span"] = 64
        elif flavor == "short_reg":
            out["time_span"] = 64
            out["n_layers"] = 1
            out["num_layers"] = 1
        elif flavor == "shock":
            out["time_span"] = 128
    return normalize_params(model, out)


def make_exploration_variant(dataset: str, model: str, base: dict[str, Any], peer: dict[str, Any], *, max_len: int, scale: float) -> dict[str, Any]:
    try:
        params = pair60.build_exploration_params(dataset, model, dict(base), dict(peer))
    except Exception:
        params = dict(base)
    hidden = stage1.safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
    params = adjust_arch(
        model,
        params,
        hidden=scale_dim(hidden, scale),
        layers=stage1.safe_int(params.get("num_layers", params.get("n_layers", 2)), 2),
        heads=stage1.safe_int(params.get("num_heads", params.get("n_heads", 4)), 4),
        max_len=max_len,
        dropout_delta=0.01,
        weight_decay_mult=1.3,
    )
    return decorate_variant(model, params, "shock")


def dedupe_candidates(
    model: str,
    candidates: list[tuple[str, dict[str, Any], stage2.HistSeed | None]],
    history_sigs: set[str],
) -> list[tuple[str, dict[str, Any], stage2.HistSeed | None]]:
    seen: set[str] = set(history_sigs)
    out: list[tuple[str, dict[str, Any], stage2.HistSeed | None]] = []
    for combo_kind, params, row in candidates:
        normalized = normalize_params(model, params)
        sig = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((combo_kind, normalized, row))
    return out


def combo_count_for_target(target: stage1.TargetSpec, est_runtime_sec: float, stats: stage2.TargetStats) -> int:
    base = MODEL_COMBO_BUDGET.get(target.model, 8)
    if stats.ratio_to_dataset_best < 0.08:
        base += 1
    if stats.valid_test_gap > 0.03:
        base += 1
    if est_runtime_sec > 700:
        base -= 1
    return stage1.clamp_int(base, 8, 12)


def max_evals_for_target(target: stage1.TargetSpec, est_runtime_sec: float, stats: stage2.TargetStats) -> int:
    speed = stage1.runtime_class(est_runtime_sec)
    base = {
        "very_fast": 14,
        "fast": 12,
        "medium": 10,
        "slow": 8,
        "very_slow": 8,
    }[speed]
    base += MODEL_EVAL_BONUS.get(target.model, 0)
    if stats.ratio_to_dataset_best < 0.08:
        base += 1
    if stats.ratio_to_dataset_best > 0.65 and target.model in {"duorec", "fearec", "sasrec", "fdsa"}:
        base -= 1
    return stage1.clamp_int(base, 8, 16)


def build_search_block(
    dataset: str,
    model: str,
    params: dict[str, Any],
    row: stage2.HistSeed | None,
    combo_kind: str,
    stats: stage2.TargetStats,
) -> tuple[dict[str, Any], dict[str, str]]:
    overrides: dict[str, str] = {"learning_rate": "loguniform"}
    search: dict[str, Any] = {}

    lr_lo, lr_hi, lr_center = stage1.center_lr(row, params, dataset, model)
    lr_floor = max(2e-4, lr_lo * 0.8, lr_center * 0.45)
    lr_ceil = min(8e-3, max(lr_hi * 2.2, lr_center * 3.0, lr_floor * 3.5))
    if model in {"difsr", "fdsa"}:
        if combo_kind in {"gate_large", "concat_large", "wide_attr", "wide_decay", "large_soft", "large_decay"}:
            lr_floor = max(4e-4, lr_center * 0.75, lr_hi * 0.55)
            lr_ceil = min(8e-3, max(lr_center * 6.0, lr_hi * 4.0, 3e-3))
        elif combo_kind in {"gate_mid", "mid_soft", "mid_decay", "shallow_gate"}:
            lr_floor = max(2.5e-4, lr_center * 0.55)
            lr_ceil = min(6e-3, max(lr_center * 4.2, lr_hi * 2.8, 2e-3))
        else:
            lr_floor = max(2e-4, lr_center * 0.45)
            lr_ceil = min(4e-3, max(lr_center * 3.0, lr_hi * 2.0, 1.6e-3))
    elif model == "bsarec":
        if combo_kind in {"wide_decay", "alpha_high", "large_soft", "large_decay"}:
            lr_floor = max(3e-4, lr_center * 0.65)
            lr_ceil = min(8e-3, max(lr_center * 5.0, lr_hi * 3.5, 2.5e-3))
        else:
            lr_floor = max(2e-4, lr_center * 0.5)
            lr_ceil = min(5e-3, max(lr_center * 3.8, lr_hi * 2.4, 1.8e-3))
    elif model == "fame":
        if combo_kind in {"experts_high", "wide_decay", "large_soft", "large_decay"}:
            lr_floor = max(3e-4, lr_center * 0.65)
            lr_ceil = min(8e-3, max(lr_center * 4.8, lr_hi * 3.0, 2.4e-3))
        else:
            lr_floor = max(2e-4, lr_center * 0.5)
            lr_ceil = min(5e-3, max(lr_center * 3.2, lr_hi * 2.2, 1.8e-3))
    elif model == "gru4rec":
        if combo_kind in {"gru_mid", "gru_deep", "large_soft", "large_decay", "wide_decay"}:
            lr_floor = max(3e-4, lr_center * 0.6)
            lr_ceil = min(6e-3, max(lr_center * 4.5, lr_hi * 3.0, 2.2e-3))
        else:
            lr_floor = max(2e-4, lr_center * 0.45)
            lr_ceil = min(4e-3, max(lr_center * 3.2, lr_hi * 2.3, 1.6e-3))
    elif model in {"duorec", "fearec", "sasrec", "tisasrec"}:
        if combo_kind in {"large_soft", "large_decay", "wide_decay", "wide_global", "tau_high", "time_mid"}:
            lr_floor = max(3e-4, lr_center * 0.6)
            lr_ceil = min(6e-3, max(lr_center * 4.6, lr_hi * 3.0, 2.2e-3))
        else:
            lr_floor = max(2e-4, lr_center * 0.45)
            lr_ceil = min(4e-3, max(lr_center * 3.2, lr_hi * 2.2, 1.6e-3))
    else:
        lr_floor = max(lr_floor, lr_center * 0.45)
        lr_ceil = min(lr_ceil, max(lr_center * 3.0, lr_floor * 3.0, 1.6e-3))
    if stats.ratio_to_dataset_best < 0.08:
        lr_floor = max(lr_floor, 3e-4)
        lr_ceil = min(8e-3, max(lr_ceil, 4e-3))
    if lr_floor >= lr_ceil:
        lr_ceil = min(8e-3, max(lr_floor * 1.8, lr_floor + 1e-4))
    search["learning_rate"] = [round(lr_floor, 8), round(lr_ceil, 8)]

    wd = stage1.safe_float(params.get("weight_decay", 1e-4), 1e-4)
    max_len = stage1.safe_int(params.get("max_len", 20), 20)
    if combo_kind in {"stable_short", "short_dense", "shallow_gate", "shallow_mix", "time_short", "tau_short", "gru_short", "sas_short"}:
        search["MAX_ITEM_LIST_LENGTH"] = sorted(set([stage1.clamp_int(max_len, 10, 20), stage1.clamp_int(max_len + 2, 10, 20)]))
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"
    else:
        search["weight_decay"] = sorted(
            set(
                [
                    round(stage1.clamp_float(max(wd, 1e-6) * 0.6, 1e-6, 1e-2), 8),
                    round(stage1.clamp_float(max(wd, 1e-6), 1e-6, 1e-2), 8),
                    round(stage1.clamp_float(max(wd, 1e-6) * 1.8, 1e-6, 1e-2), 8),
                ]
            )
        )
        overrides["weight_decay"] = "choice"

    if model in {"duorec", "fearec"}:
        tau = stage1.safe_float(params.get("tau", 0.2), 0.2)
        search["tau"] = sorted(set([round(value, 4) for value in [tau * 0.82, tau, tau * 1.18] if 0.05 <= value <= 0.6]))
        overrides["tau"] = "choice"
        if model == "fearec":
            ratio = stage1.safe_float(params.get("global_ratio", 0.85), 0.85)
            if combo_kind in {"global_high", "global_low", "wide_global"}:
                search["global_ratio"] = sorted(set([round(stage1.clamp_float(ratio * 0.9, 0.5, 1.1), 4), round(ratio, 4), round(stage1.clamp_float(ratio * 1.05, 0.5, 1.1), 4)]))
                overrides["global_ratio"] = "choice"
    elif model == "fame":
        experts = stage1.safe_int(params.get("num_experts", 3), 3)
        search["num_experts"] = sorted(set([stage1.clamp_int(experts - 1, 2, 6), experts, stage1.clamp_int(experts + 1, 2, 6)]))
        overrides["num_experts"] = "choice"
    elif model == "bsarec":
        alpha = stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5)
        search["bsarec_alpha"] = sorted(set([round(stage1.clamp_float(alpha - 0.08, 0.2, 0.9), 4), round(alpha, 4), round(stage1.clamp_float(alpha + 0.08, 0.2, 0.9), 4)]))
        overrides["bsarec_alpha"] = "choice"
    elif model in {"difsr", "fdsa"}:
        lambda_attr = stage1.safe_float(params.get("lambda_attr", 0.1), 0.1)
        search["lambda_attr"] = sorted(set([round(stage1.clamp_float(lambda_attr - 0.03, 0.0, 0.3), 4), round(lambda_attr, 4), round(stage1.clamp_float(lambda_attr + 0.03, 0.0, 0.3), 4)]))
        overrides["lambda_attr"] = "choice"
        if combo_kind in {"gate_large", "concat_large", "gate_mid", "concat_mid", "shallow_gate", "wide_attr"}:
            search["fusion_type"] = sorted(set([str(params.get("fusion_type", "sum")).lower(), "gate", "concat"]))[:3]
            overrides["fusion_type"] = "choice"
    elif model == "tisasrec":
        span = stage1.safe_int(params.get("time_span", 128), 128)
        search["time_span"] = sorted(set([max(64, span // 2), span, min(512, span * 2)]))
        overrides["time_span"] = "choice"
    elif model == "gru4rec":
        layers = stage1.safe_int(params.get("num_layers", 1), 1)
        search["num_layers"] = sorted(set([max(1, layers - 1), layers, min(3, layers + 1)]))
        overrides["num_layers"] = "choice"
    elif model in {"sasrec"}:
        layers = stage1.safe_int(params.get("num_layers", 2), 2)
        search["num_layers"] = sorted(set([max(1, layers - 1), layers, min(4, layers + 1)]))
        overrides["num_layers"] = "choice"

    return search, overrides


def scenario_variants(
    target: stage1.TargetSpec,
    base_test: dict[str, Any],
    base_balanced: dict[str, Any],
    base_valid: dict[str, Any],
    best_test: stage2.HistSeed,
    best_balanced: stage2.HistSeed,
    best_valid: stage2.HistSeed,
) -> list[tuple[str, dict[str, Any], stage2.HistSeed | None]]:
    model = target.model
    hidden = stage1.safe_int(base_test.get("hidden_size", base_test.get("embedding_size", 128)), 128)
    layers = stage1.safe_int(base_test.get("num_layers", base_test.get("n_layers", 2)), 2)
    heads = stage1.safe_int(base_test.get("num_heads", base_test.get("n_heads", 4)), 4)

    variants: list[tuple[str, dict[str, Any], stage2.HistSeed | None]] = [
        ("anchor_test", base_test, best_test),
        ("anchor_balanced", base_balanced, best_balanced),
        ("anchor_valid", base_valid, best_valid),
        (
            "stable_short",
            decorate_variant(
                model,
                adjust_arch(model, base_test, hidden=scale_dim(hidden, 0.95), layers=max(1, layers - 1), heads=max(2, min(heads, 4)), max_len=12, dropout_delta=0.02, weight_decay_mult=1.4),
                "short_reg",
            ),
            best_test,
        ),
        (
            "mid_soft",
            decorate_variant(
                model,
                adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.05), layers=max(1, layers), heads=max(2, heads), max_len=14, dropout_delta=-0.01, weight_decay_mult=0.9),
                "short_attack",
            ),
            best_balanced,
        ),
        (
            "mid_decay",
            decorate_variant(
                model,
                adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=max(2, heads), max_len=14, dropout_delta=0.01, weight_decay_mult=1.3),
                "short_reg",
            ),
            best_valid,
        ),
        (
            "large_soft",
            decorate_variant(
                model,
                adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.2), layers=min(4, layers + 1), heads=4 if model == "gru4rec" else max(4, heads), max_len=16, dropout_delta=-0.01, weight_decay_mult=0.85),
                "shock",
            ),
            best_valid,
        ),
        (
            "large_decay",
            decorate_variant(
                model,
                adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.3), layers=min(4, layers + 1), heads=4 if model == "gru4rec" else max(4, heads), max_len=16, dropout_delta=0.0, weight_decay_mult=1.15),
                "shock",
            ),
            best_valid,
        ),
        (
            "shallow_mix",
            decorate_variant(
                model,
                adjust_arch(model, base_test, hidden=scale_dim(hidden, 1.05), layers=1, heads=max(2, min(heads, 4)), max_len=12, dropout_delta=0.015, weight_decay_mult=1.1),
                "short_reg",
            ),
            best_test,
        ),
        ("wide_attr", make_exploration_variant(target.dataset, model, base_test, base_balanced, max_len=14, scale=1.1), best_test),
        ("wide_decay", make_exploration_variant(target.dataset, model, base_balanced, base_valid, max_len=16, scale=1.32), best_balanced),
        (
            "reset_mild",
            decorate_variant(
                model,
                adjust_arch(model, base_valid, hidden=scale_dim(hidden, 0.95), layers=1, heads=max(2, min(heads, 4)), max_len=14, dropout_delta=0.04, weight_decay_mult=1.8),
                "short_reg",
            ),
            best_valid,
        ),
    ]

    if model in {"difsr", "fdsa"}:
        variants.extend(
            [
                (
                    "gate_large",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.2), layers=max(1, layers), heads=max(4, heads), max_len=16, dropout_delta=-0.01, weight_decay_mult=0.95),
                        "short_attack",
                    ),
                    best_valid,
                ),
                (
                    "concat_large",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.15), layers=max(1, layers), heads=max(4, heads), max_len=16, dropout_delta=-0.015, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_balanced,
                ),
                (
                    "gate_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=max(4, min(heads, 4)), max_len=14, dropout_delta=0.0, weight_decay_mult=1.0),
                        "short_reg",
                    ),
                    best_test,
                ),
                (
                    "shallow_gate",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.05), layers=1, heads=max(2, min(heads, 4)), max_len=12, dropout_delta=0.02, weight_decay_mult=1.2),
                        "short_reg",
                    ),
                    best_valid,
                ),
            ]
        )
    elif model == "bsarec":
        variants.extend(
            [
                (
                    "alpha_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 1.05), layers=max(1, layers), heads=max(2, heads), max_len=14, dropout_delta=-0.005, weight_decay_mult=0.95),
                        "short_attack",
                    ),
                    best_test,
                ),
                (
                    "alpha_high",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.15), layers=max(1, layers), heads=max(4, heads), max_len=16, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_valid,
                ),
                (
                    "alpha_decay",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=max(2, heads), max_len=14, dropout_delta=0.01, weight_decay_mult=1.3),
                        "short_reg",
                    ),
                    best_balanced,
                ),
            ]
        )
    elif model == "fame":
        variants.extend(
            [
                (
                    "experts_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=heads, max_len=14, dropout_delta=0.0, weight_decay_mult=1.0),
                        "short_reg",
                    ),
                    best_test,
                ),
                (
                    "experts_high",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.15), layers=max(1, layers), heads=heads, max_len=16, dropout_delta=-0.01, weight_decay_mult=0.85),
                        "shock",
                    ),
                    best_valid,
                ),
                (
                    "experts_short",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 0.95), layers=1, heads=heads, max_len=12, dropout_delta=0.02, weight_decay_mult=1.25),
                        "short_reg",
                    ),
                    best_balanced,
                ),
            ]
        )
    elif model == "gru4rec":
        variants.extend(
            [
                (
                    "gru_short",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 0.95), layers=1, max_len=12, dropout_delta=0.02, weight_decay_mult=1.2),
                        "short_reg",
                    ),
                    best_test,
                ),
                (
                    "gru_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.05), layers=min(3, layers + 1), max_len=14, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_balanced,
                ),
                (
                    "gru_deep",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.15), layers=min(3, layers + 1), max_len=16, dropout_delta=-0.02, weight_decay_mult=0.8),
                        "shock",
                    ),
                    best_valid,
                ),
            ]
        )
    elif model == "tisasrec":
        variants.extend(
            [
                (
                    "time_short",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 0.9), layers=max(1, layers - 1), heads=2, max_len=12, dropout_delta=0.02, weight_decay_mult=1.3),
                        "short_reg",
                    ),
                    best_test,
                ),
                (
                    "time_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.05), layers=min(4, layers), heads=max(4, heads), max_len=14, dropout_delta=-0.01, weight_decay_mult=0.95),
                        "shock",
                    ),
                    best_balanced,
                ),
                (
                    "time_wide",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.1), layers=min(4, layers + 1), heads=8, max_len=16, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_valid,
                ),
            ]
        )
    elif model == "fearec":
        variants.extend(
            [
                (
                    "global_low",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 0.95), layers=max(1, layers), heads=heads, max_len=14, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "short_attack",
                    ),
                    best_test,
                ),
                (
                    "global_high",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.08), layers=max(1, layers), heads=max(2, heads), max_len=16, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_valid,
                ),
                (
                    "wide_global",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=heads, max_len=14, dropout_delta=0.0, weight_decay_mult=1.0),
                        "short_attack",
                    ),
                    best_balanced,
                ),
            ]
        )
    elif model == "duorec":
        variants.extend(
            [
                (
                    "tau_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 1.0), layers=max(1, layers), heads=heads, max_len=14, dropout_delta=-0.01, weight_decay_mult=0.9),
                        "shock",
                    ),
                    best_test,
                ),
                (
                    "tau_short",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 0.9), layers=max(1, layers - 1), heads=2, max_len=12, dropout_delta=0.02, weight_decay_mult=1.2),
                        "short_reg",
                    ),
                    best_valid,
                ),
                (
                    "tau_high",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.05), layers=max(1, layers), heads=heads, max_len=16, dropout_delta=-0.01, weight_decay_mult=0.92),
                        "shock",
                    ),
                    best_balanced,
                ),
            ]
        )
    elif model == "sasrec":
        variants.extend(
            [
                (
                    "sas_short",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_test, hidden=scale_dim(hidden, 0.95), layers=max(1, layers - 1), heads=2, max_len=12, dropout_delta=0.02, weight_decay_mult=1.3),
                        "short_reg",
                    ),
                    best_test,
                ),
                (
                    "sas_mid",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_balanced, hidden=scale_dim(hidden, 1.05), layers=max(1, layers), heads=max(4, heads), max_len=14, dropout_delta=-0.01, weight_decay_mult=0.95),
                        "shock",
                    ),
                    best_balanced,
                ),
                (
                    "sas_large",
                    decorate_variant(
                        model,
                        adjust_arch(model, base_valid, hidden=scale_dim(hidden, 1.2), layers=min(4, layers + 1), heads=8, max_len=16, dropout_delta=-0.02, weight_decay_mult=0.85),
                        "shock",
                    ),
                    best_valid,
                ),
            ]
        )

    return variants


def build_combo_specs(
    target: stage1.TargetSpec,
    rows: list[stage2.HistSeed],
    est_runtime_sec: float,
    combo_seed_base: int,
    stats: stage2.TargetStats,
) -> list[stage1.ComboSpec]:
    desired_count = combo_count_for_target(target, est_runtime_sec, stats)
    best_test, best_balanced = stage2.choose_seed_rows(rows)
    best_valid = max(rows, key=lambda item: (item.best_valid_mrr20, item.test_mrr20, item.selection_score))

    base_test = normalize_params(target.model, best_test.params)
    base_balanced = normalize_params(target.model, best_balanced.params)
    base_valid = normalize_params(target.model, best_valid.params)

    candidates = scenario_variants(target, base_test, base_balanced, base_valid, best_test, best_balanced, best_valid)

    uniq = dedupe_candidates(target.model, candidates, history_signatures(target.model, rows))
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
                run_phase=f"BASELINE2_ADDTUNE3_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
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
    parser = argparse.ArgumentParser(description="Stage-3 aggressive beauty-centric baseline_2 hyperopt tuning.")
    parser.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"))
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
    targets = target_specs()
    dataset_best_test: dict[str, float] = {}
    for (dataset, _model), rows in history.items():
        best = max((row.test_mrr20 for row in rows), default=0.0)
        dataset_best_test[dataset] = max(dataset_best_test.get(dataset, 0.0), best)

    specs: list[stage1.ComboSpec] = []
    combo_seed_base = 2026041700
    for idx, target in enumerate(targets):
        if filters and (target.dataset, target.model) not in filters:
            continue
        rows = history.get((target.dataset, target.model), [])
        if not rows:
            continue
        est_runtime_sec = runtimes.get((target.dataset, target.model), 600.0)
        stats = summarize_target(rows, dataset_best_test.get(target.dataset, 0.0))
        specs.extend(build_combo_specs(target, rows, est_runtime_sec, combo_seed_base + idx * 30, stats))

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

    print(f"[baseline2_addtuning3] python={args.python_bin}")
    print(f"[baseline2_addtuning3] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
    if args.dry_run:
        for row in plan[: min(20, len(plan))]:
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
        print(f"[baseline2_addtuning3] nothing to run; summary={SUMMARY_CSV}")
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
                f"[baseline2_addtuning3] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                stage1.append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning3] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
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

    print(f"[baseline2_addtuning3] complete summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()