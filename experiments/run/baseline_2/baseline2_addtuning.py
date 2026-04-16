#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import run_pair60_campaign as pair60


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
ARTIFACT_ROOT = EXP_DIR / "run" / "artifacts"
TRACK = "baseline_2"
AXIS = "PAIR60_ADDTUNING"

SUMMARY_SOURCES = [
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_V4" / "summary.csv",
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_V4_REVISED" / "summary.csv",
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_V4_REVISED_LONG12H" / "summary.csv",
]

RESULTS_ROOT = ARTIFACT_ROOT / "results" / TRACK
OUTPUT_ROOT = ARTIFACT_ROOT / "logs" / TRACK / AXIS
SPACE_ROOT = OUTPUT_ROOT / "spaces"
SESSION_LOG_ROOT = OUTPUT_ROOT / "session_logs"
PLAN_CSV = OUTPUT_ROOT / "plan.csv"
SUMMARY_CSV = OUTPUT_ROOT / "summary.csv"
MANIFEST_JSON = OUTPUT_ROOT / "manifest.json"

DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 7

NONCONVENTIONAL_TARGETS = [
    ("beauty", "bsarec", "critical"),
    ("beauty", "fame", "critical"),
    ("beauty", "difsr", "critical"),
    ("beauty", "fearec", "high"),
    ("beauty", "duorec", "high"),
    ("retail_rocket", "fdsa", "critical"),
    ("retail_rocket", "bsarec", "high"),
    ("retail_rocket", "fame", "high"),
    ("foursquare", "bsarec", "critical"),
    ("foursquare", "fame", "critical"),
    ("foursquare", "difsr", "critical"),
    ("foursquare", "fearec", "high"),
    ("foursquare", "duorec", "high"),
    ("foursquare", "fdsa", "high"),
    ("movielens1m", "fearec", "high"),
    ("movielens1m", "duorec", "high"),
    ("KuaiRecLargeStrictPosV2_0.2", "fdsa", "critical"),
    ("KuaiRecLargeStrictPosV2_0.2", "bsarec", "high"),
    ("KuaiRecLargeStrictPosV2_0.2", "fame", "high"),
]

OPTIONAL_CONVENTIONAL_TARGETS = [
    ("beauty", "gru4rec", "optional"),
    ("beauty", "tisasrec", "optional"),
    ("foursquare", "gru4rec", "optional"),
    ("lastfm0.03", "gru4rec", "optional"),
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec", "optional"),
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec", "optional"),
]

PRIORITY_RANK = {
    "critical": 0,
    "high": 1,
    "optional": 2,
}

FAST_DATASET_BONUS = {
    "beauty": -2,
    "retail_rocket": -1,
    "foursquare": 0,
    "KuaiRecLargeStrictPosV2_0.2": 1,
    "movielens1m": 2,
    "lastfm0.03": 3,
}


@dataclass(frozen=True)
class TargetSpec:
    dataset: str
    model: str
    priority: str


@dataclass
class HistRow:
    dataset: str
    model: str
    axis: str
    run_phase: str
    best_valid_mrr20: float
    elapsed_sec: float
    lr_lo: float
    lr_hi: float
    params: dict[str, Any]


@dataclass
class ComboSpec:
    dataset: str
    model: str
    combo_id: str
    combo_kind: str
    priority: str
    est_runtime_sec: float
    max_evals: int
    seed: int
    base_params: dict[str, Any]
    search_space: dict[str, Any]
    fixed_space: dict[str, Any]
    space_yaml: Path
    run_phase: str


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_summary_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def sanitize_token(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "NA"


def target_specs(include_conventional: bool) -> list[TargetSpec]:
    base = [TargetSpec(dataset, model, priority) for dataset, model, priority in NONCONVENTIONAL_TARGETS]
    if include_conventional:
        base.extend(TargetSpec(dataset, model, priority) for dataset, model, priority in OPTIONAL_CONVENTIONAL_TARGETS)
    return base


def parse_target_filter(raw_items: list[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in raw_items:
        token = str(item).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"target filter must be dataset:model, got: {token}")
        dataset, model = token.split(":", 1)
        out.add((dataset.strip(), model.strip().lower()))
    return out


def load_history_rows() -> dict[tuple[str, str], list[HistRow]]:
    grouped: dict[tuple[str, str], list[HistRow]] = {}
    for summary_path in SUMMARY_SOURCES:
        if not summary_path.exists():
            continue
        axis = summary_path.parent.name
        for row in read_csv(summary_path):
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            dataset = str(row.get("dataset", "")).strip()
            model = str(row.get("model", "")).strip().lower()
            params_text = str(row.get("params_json", "")).strip()
            if not dataset or not model or not params_text:
                continue
            try:
                params = json.loads(params_text)
            except Exception:
                continue
            hist = HistRow(
                dataset=dataset,
                model=model,
                axis=axis,
                run_phase=str(row.get("run_phase", "")).strip(),
                best_valid_mrr20=safe_float(row.get("best_valid_mrr20", 0.0), 0.0),
                elapsed_sec=safe_float(row.get("elapsed_sec", 0.0), 0.0),
                lr_lo=safe_float(row.get("lr_lo", 0.0), 0.0),
                lr_hi=safe_float(row.get("lr_hi", 0.0), 0.0),
                params=params,
            )
            grouped.setdefault((dataset, model), []).append(hist)

    for key, rows in grouped.items():
        rows.sort(key=lambda item: (item.best_valid_mrr20, -item.elapsed_sec), reverse=True)
    return grouped


def median_runtime_by_target(history: dict[tuple[str, str], list[HistRow]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for key, rows in history.items():
        secs = sorted(item.elapsed_sec for item in rows if item.elapsed_sec > 0)
        if not secs:
            continue
        mid = len(secs) // 2
        if len(secs) % 2 == 1:
            out[key] = secs[mid]
        else:
            out[key] = 0.5 * (secs[mid - 1] + secs[mid])
    return out


def dedupe_hist_rows(rows: list[HistRow]) -> list[HistRow]:
    seen: set[str] = set()
    out: list[HistRow] = []
    for row in rows:
        sig = json.dumps(row.params, sort_keys=True, ensure_ascii=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(row)
    return out


def runtime_class(est_runtime_sec: float) -> str:
    if est_runtime_sec <= 120:
        return "very_fast"
    if est_runtime_sec <= 450:
        return "fast"
    if est_runtime_sec <= 900:
        return "medium"
    if est_runtime_sec <= 1800:
        return "slow"
    return "very_slow"


def max_evals_for_runtime(est_runtime_sec: float) -> int:
    if est_runtime_sec <= 120:
        return 8
    if est_runtime_sec <= 450:
        return 6
    if est_runtime_sec <= 900:
        return 5
    if est_runtime_sec <= 1800:
        return 4
    return 3


def around_float(base: float, factors: list[float], lo: float, hi: float, *, digits: int = 6) -> list[float]:
    values = []
    for factor in factors:
        values.append(round(clamp_float(base * factor, lo, hi), digits))
    values = sorted(set(values))
    if len(values) == 1:
        values = [values[0], values[0]]
    return values


def around_int(base: int, candidates: list[int], lo: int, hi: int) -> list[int]:
    values = sorted(set(clamp_int(candidate, lo, hi) for candidate in candidates))
    if len(values) == 1:
        values = [values[0], values[0]]
    return values


def head_choices(base: int) -> list[int]:
    options = [2, 4, 8]
    if base not in options:
        options.append(base)
    options = sorted(set(options))
    if base <= 2:
        return [2, 4]
    if base >= 8:
        return [4, 8]
    return [2, 4, 8]


def layer_choices(base: int) -> list[int]:
    return sorted(set(clamp_int(candidate, 1, 4) for candidate in [base - 1, base, base + 1]))


def center_lr(row: HistRow | None, params: dict[str, Any], dataset: str, model: str) -> tuple[float, float, float]:
    if row and row.lr_lo > 0 and row.lr_hi > 0 and row.lr_hi >= row.lr_lo:
        lo = row.lr_lo
        hi = row.lr_hi
    else:
        lo, hi = pair60.DATASET_LR_BASE.get(dataset, (5e-5, 5e-3))
        center = math.sqrt(lo * hi) * float(pair60.MODEL_LR_MULT.get(model, 1.0))
        ratio = float(pair60.MODEL_LR_BAND_RATIO.get(model, 4.0))
        lo = center / math.sqrt(ratio)
        hi = center * math.sqrt(ratio)
    center = math.sqrt(lo * hi)
    if "learning_rate" in params:
        base_lr = safe_float(params.get("learning_rate", center), center)
        center = clamp_float(base_lr, lo, hi)
    return lo, hi, center


def make_fixed_block(model: str, params: dict[str, Any]) -> dict[str, Any]:
    fixed = {
        "MAX_ITEM_LIST_LENGTH": safe_int(params.get("max_len", 20), 20),
        "weight_decay": safe_float(params.get("weight_decay", 1e-4), 1e-4),
    }

    hidden = safe_int(params.get("hidden_size", params.get("embedding_size", 128)), 128)
    embed = safe_int(params.get("embedding_size", hidden), hidden)
    layers = safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)
    heads = safe_int(params.get("num_heads", params.get("n_heads", 4)), 4)
    dropout = safe_float(params.get("dropout", 0.15), 0.15)

    if model in {"duorec", "fearec", "fdsa", "tisasrec"}:
        fixed.update(
            {
                "hidden_size": hidden,
                "embedding_size": embed,
                "n_layers": layers,
                "n_heads": heads,
                "inner_size": safe_int(params.get("inner_size", hidden * 2), hidden * 2),
                "dropout_ratio": dropout,
            }
        )
    elif model in {"sasrec", "bsarec", "fame", "difsr"}:
        fixed.update(
            {
                "hidden_size": hidden,
                "embedding_size": embed,
                "num_layers": layers,
                "num_heads": heads,
                "inner_size": safe_int(params.get("inner_size", hidden * 2), hidden * 2),
                "dropout_ratio": dropout,
            }
        )
    elif model == "gru4rec":
        fixed.update(
            {
                "hidden_size": hidden,
                "embedding_size": embed,
                "num_layers": layers,
                "dropout_prob": dropout,
            }
        )

    if model == "tisasrec":
        fixed["time_span"] = safe_int(params.get("time_span", 256), 256)
    if model in {"duorec", "fearec"}:
        fixed["contrast"] = params.get("contrast", "un")
        fixed["tau"] = safe_float(params.get("tau", 0.2), 0.2)
        fixed["lmd"] = safe_float(params.get("lmd", 0.04), 0.04)
        fixed["lmd_sem"] = safe_float(params.get("lmd_sem", 0.0), 0.0)
        if model == "fearec":
            fixed["global_ratio"] = safe_float(params.get("global_ratio", 0.85), 0.85)
    if model == "bsarec":
        fixed["bsarec_alpha"] = safe_float(params.get("bsarec_alpha", 0.5), 0.5)
        fixed["bsarec_c"] = safe_int(params.get("bsarec_c", 3), 3)
    if model == "fame":
        fixed["num_experts"] = safe_int(params.get("num_experts", 4), 4)
    if model in {"difsr", "fdsa"}:
        fixed["attribute_hidden_size"] = safe_int(params.get("attribute_hidden_size", hidden), hidden)
        fixed["fusion_type"] = params.get("fusion_type", "sum")
        fixed["use_attribute_predictor"] = bool(params.get("use_attribute_predictor", True))
        fixed["lambda_attr"] = safe_float(params.get("lambda_attr", 0.1), 0.1)
    if model == "fdsa":
        fixed["selected_features"] = params.get("selected_features", ["category"])
        fixed["pooling_mode"] = params.get("pooling_mode", "mean")

    return fixed


def build_search_block(dataset: str, model: str, params: dict[str, Any], row: HistRow | None, est_runtime_sec: float) -> tuple[dict[str, Any], dict[str, str]]:
    search: dict[str, Any] = {}
    overrides: dict[str, str] = {
        "learning_rate": "loguniform",
        "weight_decay": "loguniform_zero",
    }

    lr_lo, lr_hi, lr_center = center_lr(row, params, dataset, model)
    search["learning_rate"] = around_float(lr_center, [0.55, 0.8, 1.0, 1.25, 1.8], lr_lo, lr_hi, digits=8)

    wd_center = safe_float(params.get("weight_decay", 1e-4), 1e-4)
    search["weight_decay"] = sorted(set([0.0] + around_float(max(wd_center, 1e-6), [0.5, 1.0, 2.0], 1e-6, 5e-3, digits=8)))

    max_len = safe_int(params.get("max_len", 20), 20)
    if est_runtime_sec <= 450:
        search["MAX_ITEM_LIST_LENGTH"] = around_int(max_len, [max_len - 10, max_len, max_len + 10], 10, 100)
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"
    else:
        search["MAX_ITEM_LIST_LENGTH"] = around_int(max_len, [max_len - 5, max_len], 10, 100)
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"

    dropout = safe_float(params.get("dropout", 0.15), 0.15)
    if model == "gru4rec":
        search["dropout_prob"] = around_float(dropout, [0.8, 1.0, 1.2], 0.05, 0.4, digits=4)
        overrides["dropout_prob"] = "choice"
    else:
        search["dropout_ratio"] = around_float(dropout, [0.75, 1.0, 1.25], 0.05, 0.35, digits=4)
        overrides["dropout_ratio"] = "choice"

    layers = safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)
    heads = safe_int(params.get("num_heads", params.get("n_heads", 4)), 4)
    speed = runtime_class(est_runtime_sec)

    if model in {"duorec", "fearec", "fdsa", "tisasrec"}:
        search["n_layers"] = layer_choices(layers) if speed in {"very_fast", "fast", "medium"} else [layers]
        overrides["n_layers"] = "choice"
        if model != "tisasrec" or speed in {"very_fast", "fast", "medium"}:
            search["n_heads"] = head_choices(heads) if speed in {"very_fast", "fast", "medium"} else [heads]
            overrides["n_heads"] = "choice"
    elif model in {"sasrec", "bsarec", "fame", "difsr", "gru4rec"}:
        search["num_layers"] = layer_choices(layers) if speed in {"very_fast", "fast", "medium"} else [layers]
        overrides["num_layers"] = "choice"
        if model in {"sasrec", "bsarec", "fame", "difsr"}:
            search["num_heads"] = head_choices(heads) if speed in {"very_fast", "fast", "medium"} else [heads]
            overrides["num_heads"] = "choice"

    if model == "tisasrec":
        span = safe_int(params.get("time_span", 256), 256)
        candidates = [span]
        if speed in {"very_fast", "fast", "medium"}:
            candidates.extend([max(64, span // 2), min(1024, span * 2)])
        search["time_span"] = sorted(set(candidates))
        overrides["time_span"] = "choice"

    if model in {"duorec", "fearec"}:
        search["tau"] = around_float(safe_float(params.get("tau", 0.2), 0.2), [0.8, 1.0, 1.25], 0.05, 1.0, digits=4)
        search["lmd"] = around_float(safe_float(params.get("lmd", 0.04), 0.04), [0.5, 1.0, 1.5], 0.0, 0.2, digits=4)
        search["lmd_sem"] = around_float(safe_float(params.get("lmd_sem", 0.0), 0.0) + 0.02, [0.0, 1.0, 2.0], 0.0, 0.2, digits=4)
        search["contrast"] = sorted(set([params.get("contrast", "un"), "un", "us_x"]))
        overrides.update({"tau": "choice", "lmd": "choice", "lmd_sem": "choice", "contrast": "choice"})
        if model == "fearec":
            search["global_ratio"] = around_float(safe_float(params.get("global_ratio", 0.85), 0.85), [0.9, 1.0, 1.1], 0.5, 1.2, digits=4)
            overrides["global_ratio"] = "choice"

    if model == "bsarec":
        search["bsarec_alpha"] = [0.3, 0.5, 0.7]
        search["bsarec_c"] = [2, 3, 4]
        overrides["bsarec_alpha"] = "choice"
        overrides["bsarec_c"] = "choice"

    if model == "fame":
        experts = safe_int(params.get("num_experts", 4), 4)
        search["num_experts"] = sorted(set(clamp_int(candidate, 2, 8) for candidate in [experts - 1, experts, experts + 1]))
        overrides["num_experts"] = "choice"

    if model in {"difsr", "fdsa"}:
        attr_hidden = safe_int(params.get("attribute_hidden_size", params.get("hidden_size", 128)), 128)
        if speed in {"very_fast", "fast", "medium"}:
            search["attribute_hidden_size"] = sorted(set(clamp_int(candidate, 64, 256) for candidate in [attr_hidden - 32, attr_hidden, attr_hidden + 32]))
            overrides["attribute_hidden_size"] = "choice"
        search["fusion_type"] = sorted(set([params.get("fusion_type", "sum"), "sum", "gate"]))
        search["lambda_attr"] = around_float(safe_float(params.get("lambda_attr", 0.1), 0.1), [0.6, 1.0, 1.4], 0.0, 0.3, digits=4)
        overrides["fusion_type"] = "choice"
        overrides["lambda_attr"] = "choice"

    for key, values in list(search.items()):
        if not isinstance(values, list):
            values = [values]
        deduped = []
        seen = set()
        for value in values:
            frozen = json.dumps(value, sort_keys=True, ensure_ascii=True)
            if frozen in seen:
                continue
            seen.add(frozen)
            deduped.append(value)
        if len(deduped) <= 1:
            search[key] = deduped[:1]
        else:
            search[key] = deduped

    return search, overrides


def build_speed_regularized_params(dataset: str, model: str, base: dict[str, Any], est_runtime_sec: float) -> dict[str, Any]:
    params = dict(base)
    max_len = safe_int(params.get("max_len", 20), 20)
    layers = safe_int(params.get("num_layers", params.get("n_layers", 2)), 2)
    heads = safe_int(params.get("num_heads", params.get("n_heads", 4)), 4)
    dropout = safe_float(params.get("dropout", 0.15), 0.15)

    if est_runtime_sec > 1800:
        params["max_len"] = clamp_int(max_len - 5, 10, 100)
        params["dropout"] = round(clamp_float(dropout + 0.03, 0.05, 0.35), 4)
        if model in {"fearec", "duorec", "fdsa", "fame", "bsarec", "difsr", "sasrec"}:
            params["num_layers"] = clamp_int(layers - 1, 1, 4)
            params["n_layers"] = params["num_layers"]
    else:
        params["max_len"] = clamp_int(max_len + 5, 10, 100)
        params["dropout"] = round(clamp_float(dropout - 0.02, 0.05, 0.35), 4)

    if model in {"bsarec", "fame", "difsr", "sasrec", "fearec", "duorec", "fdsa", "tisasrec"}:
        if est_runtime_sec > 2500:
            params["num_heads"] = 2 if heads > 2 else heads
            params["n_heads"] = params["num_heads"]
        else:
            params["num_heads"] = 4 if heads < 4 else heads
            params["n_heads"] = params["num_heads"]

    if model == "fame":
        params["num_experts"] = clamp_int(safe_int(params.get("num_experts", 4), 4) - (1 if est_runtime_sec > 2500 else 0), 2, 8)
    if model in {"duorec", "fearec"}:
        params["tau"] = round(clamp_float(safe_float(params.get("tau", 0.2), 0.2) * 0.9, 0.05, 1.0), 4)
    if model in {"difsr", "fdsa"}:
        params["fusion_type"] = "gate" if str(params.get("fusion_type", "sum")).lower() == "sum" else str(params.get("fusion_type", "sum")).lower()
        params["lambda_attr"] = round(clamp_float(safe_float(params.get("lambda_attr", 0.1), 0.1) + 0.03, 0.0, 0.3), 4)

    return params


def ensure_four_combos(target: TargetSpec, rows: list[HistRow], est_runtime_sec: float) -> list[tuple[str, str, dict[str, Any], HistRow | None]]:
    rows = dedupe_hist_rows(rows)
    if not rows:
        raise RuntimeError(f"No history rows for {target.dataset}/{target.model}")

    first = rows[0]
    second = rows[1] if len(rows) > 1 else rows[0]
    combos: list[tuple[str, str, dict[str, Any], HistRow | None]] = [
        ("K1", "hist_best", dict(first.params), first),
        ("K2", "hist_alt", dict(second.params), second),
        ("K3", "hist_explore", pair60.build_exploration_params(target.dataset, target.model, dict(first.params), dict(second.params)), first),
        ("K4", "hist_speed", build_speed_regularized_params(target.dataset, target.model, dict(second.params), est_runtime_sec), second),
    ]

    seen: set[str] = set()
    uniq: list[tuple[str, str, dict[str, Any], HistRow | None]] = []
    for combo_id, combo_kind, params, row in combos:
        sig = json.dumps(params, sort_keys=True, ensure_ascii=True)
        if sig in seen:
            params = build_speed_regularized_params(target.dataset, target.model, params, est_runtime_sec + len(uniq) * 100)
            sig = json.dumps(params, sort_keys=True, ensure_ascii=True)
        seen.add(sig)
        uniq.append((combo_id, combo_kind, params, row))
    return uniq[:4]


def build_combo_specs(target: TargetSpec, rows: list[HistRow], est_runtime_sec: float, combo_seed_base: int) -> list[ComboSpec]:
    SPACE_ROOT.mkdir(parents=True, exist_ok=True)
    combos = ensure_four_combos(target, rows, est_runtime_sec)
    specs: list[ComboSpec] = []
    for idx, (combo_id, combo_kind, params, anchor_row) in enumerate(combos, start=1):
        search, type_overrides = build_search_block(target.dataset, target.model, params, anchor_row, est_runtime_sec)
        fixed = make_fixed_block(target.model, params)
        fixed["search_space_type_overrides"] = type_overrides

        model_tag = sanitize_token(target.model)
        dataset_tag = sanitize_token(target.dataset)
        space_yaml = SPACE_ROOT / f"{dataset_tag}_{model_tag}_{combo_id}.yaml"
        with space_yaml.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"fixed": fixed, "search": search}, indent=2, ensure_ascii=True))
            handle.write("\n")

        seed = combo_seed_base + idx
        run_phase = f"BASELINE2_ADDTUNE_{sanitize_token(target.dataset).upper()}_{sanitize_token(target.model).upper()}_{combo_id}"

        specs.append(
            ComboSpec(
                dataset=target.dataset,
                model=target.model,
                combo_id=combo_id,
                combo_kind=combo_kind,
                priority=target.priority,
                est_runtime_sec=est_runtime_sec,
                max_evals=max_evals_for_runtime(est_runtime_sec),
                seed=seed,
                base_params=params,
                search_space=search,
                fixed_space=fixed,
                space_yaml=space_yaml,
                run_phase=run_phase,
            )
        )
    return specs


def build_command(spec: ComboSpec, gpu_id: str, python_bin: str, search_algo: str) -> list[str]:
    cmd = [
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
    return cmd


def plan_rows(specs: list[ComboSpec]) -> list[dict[str, Any]]:
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
                "runtime_class": runtime_class(spec.est_runtime_sec),
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


def read_existing_summary(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        phase = str(row.get("run_phase", "")).strip()
        if phase:
            out[phase] = row
    return out


def find_result_path(run_phase: str, dataset: str, model: str) -> Path | None:
    latest: tuple[float, Path] | None = None
    for result_path in RESULTS_ROOT.glob("*.json"):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_phase", "")).strip() != run_phase:
            continue
        if str(payload.get("dataset", "")).strip() != dataset:
            continue
        if str(payload.get("model", "")).strip().lower() != model.lower():
            continue
        mtime = result_path.stat().st_mtime
        if latest is None or mtime > latest[0]:
            latest = (mtime, result_path)
    return latest[1] if latest else None


def build_summary_payload(spec: ComboSpec, gpu_id: str, status: str, result_path: Path | None, log_path: Path, elapsed_sec: float, error: str) -> dict[str, Any]:
    metrics = pair60.parse_result_metrics(result_path) if result_path else {}
    return {
        "dataset": spec.dataset,
        "model": spec.model,
        "combo_id": spec.combo_id,
        "combo_kind": spec.combo_kind,
        "priority": spec.priority,
        "run_phase": spec.run_phase,
        "gpu_id": gpu_id,
        "status": status,
        "best_valid_mrr20": metrics.get("best_valid_mrr20", ""),
        "test_mrr20": metrics.get("test_mrr20", ""),
        "valid_unseen_mrr20": metrics.get("valid_unseen_mrr20", ""),
        "test_unseen_mrr20": metrics.get("test_unseen_mrr20", ""),
        "test_main_seen_count": metrics.get("test_main_seen_count", ""),
        "test_main_unseen_count": metrics.get("test_main_unseen_count", ""),
        "est_runtime_sec": round(spec.est_runtime_sec, 1),
        "elapsed_sec": round(elapsed_sec, 3),
        "max_evals": spec.max_evals,
        "epochs": DEFAULT_EPOCHS,
        "patience": DEFAULT_PATIENCE,
        "seed": spec.seed,
        "result_path": "" if result_path is None else str(result_path),
        "log_path": str(log_path),
        "space_yaml": str(spec.space_yaml),
        "error": error,
        "timestamp_utc": now_utc(),
        "base_params_json": json.dumps(spec.base_params, ensure_ascii=True, sort_keys=True),
    }


def run_one(spec: ComboSpec, gpu_id: str, python_bin: str, search_algo: str) -> dict[str, Any]:
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
    result_path = find_result_path(spec.run_phase, spec.dataset, spec.model)
    if result_path is None and status == "ok":
        status = "missing_result"
        error = "result_json_not_found"
    if not error and status != "ok":
        error = pair60.extract_error_tail(log_path)
    return build_summary_payload(spec, gpu_id, status, result_path, log_path, elapsed_sec, error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline_2 targeted additional hyperopt tuning for weak baselines.")
    parser.add_argument("--gpus", type=str, default=os.environ.get("GPUS", "0"))
    parser.add_argument("--include-conventional", action="store_true", default=False)
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--search-algo", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--limit-jobs", type=int, default=0)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def build_all_specs(args: argparse.Namespace) -> list[ComboSpec]:
    history = load_history_rows()
    runtimes = median_runtime_by_target(history)
    filters = parse_target_filter(args.targets)
    targets = target_specs(args.include_conventional)

    specs: list[ComboSpec] = []
    combo_seed_base = 20260416
    for idx, target in enumerate(targets):
        if filters and (target.dataset, target.model) not in filters:
            continue
        rows = history.get((target.dataset, target.model), [])
        if not rows:
            continue
        est_runtime_sec = runtimes.get((target.dataset, target.model), 600.0)
        specs.extend(build_combo_specs(target, rows, est_runtime_sec, combo_seed_base + idx * 20))

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
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs specified. Use --gpus 0,1,...")

    specs = build_all_specs(args)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    plan = plan_rows(specs)
    write_csv(
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
        "created_at": now_utc(),
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

    print(f"[baseline2_addtuning] python={args.python_bin}")
    print(f"[baseline2_addtuning] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
    if args.dry_run:
        for row in plan[: min(12, len(plan))]:
            print(
                f"  {row['dataset']} {row['model']} {row['combo_id']} priority={row['priority']} "
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

    existing = read_existing_summary(SUMMARY_CSV) if args.resume else {}
    remaining = [spec for spec in specs if spec.run_phase not in existing or str(existing[spec.run_phase].get("status", "")).strip().lower() != "ok"]
    if not remaining:
        print(f"[baseline2_addtuning] nothing to run; summary={SUMMARY_CSV}")
        return

    job_queue: Queue[ComboSpec] = Queue()
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
                f"[baseline2_addtuning] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
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

    print(f"[baseline2_addtuning] complete summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()