#!/usr/bin/env python3
"""Baseline_2 / FMoE_N4 staged tuning runner.

Stages:
- A: 30 structural candidates x LR grid, quick run, promote top-12
- B: hyperopt on top-12, promote top-4
- C: mutate top-4 -> 12, re-evaluate, promote top-2
- D: mutate top-2 -> 6, deep tuning, select final top-1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
RUN_DIR = EXP_DIR / "run"
ARTIFACT_ROOT = RUN_DIR / "artifacts"

DEFAULT_DATASETS = [
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "amazon_beauty",
    "beauty",
    "foursquare",
    "movielens1m",
    "retail_rocket",
]

DATASET_CONFIG_MAP = {
    "kuaireclargestrictposv2_0.2": "tune_kuai_strict_small",
    "lastfm0.03": "tune_lfm_small",
    "amazon_beauty": "tune_ab",
    "beauty": "tune_ab",
    "foursquare": "tune_fs",
    "movielens1m": "tune_ml",
    "retail_rocket": "tune_rr",
}

MODEL_LABELS = {
    "sasrec": "SASRec",
    "tisasrec": "TiSASRec",
    "gru4rec": "GRU4Rec",
    "fdsa": "FDSA",
    "featured_moe_n3": "FeaturedMoE_N3",
}

DEFAULT_BASELINE_MODELS = ["sasrec", "tisasrec", "gru4rec"]
DEFAULT_FMOE_MODELS = ["featured_moe_n3"]
STAGE_A_STRUCT_COUNT_DEFAULT = 30

DATASET_SPEED_RANK = {
    "amazon_beauty": 1,
    "beauty": 2,
    "foursquare": 3,
    "retail_rocket": 4,
    "kuaireclargestrictposv2_0.2": 5,
    "lastfm0.03": 6,
    "movielens1m": 7,
}

MODEL_SPEED_RANK = {
    "gru4rec": 1,
    "sasrec": 2,
    "tisasrec": 3,
    "fdsa": 4,
    "featured_moe_n3": 5,
}

SUMMARY_FIELDS = [
    "stage",
    "dataset",
    "model",
    "model_label",
    "candidate_id",
    "parent_candidate_id",
    "run_phase",
    "run_id",
    "runtime_seed",
    "gpu_id",
    "status",
    "best_valid_mrr20",
    "test_mrr20",
    "valid_unseen_mrr20",
    "valid_unseen_hit20",
    "test_unseen_mrr20",
    "test_unseen_hit20",
    "valid_main_seen_count",
    "valid_main_unseen_count",
    "test_main_seen_count",
    "test_main_unseen_count",
    "result_path",
    "log_path",
    "elapsed_sec",
    "error",
    "timestamp_utc",
    "params_json",
]

STAGE_A_LR_GRID = [8e-5, 2e-4, 4e-4, 6e-4, 8e-4, 3e-3]
RUN_STATUS_END_NORMAL_RE = re.compile(r"\[RUN_STATUS\]\s*END\s+status=normal\b", re.IGNORECASE)


def _format_lr_token(lr: float) -> str:
    s = f"{float(lr):.0e}".lower().replace("+0", "").replace("+", "")
    return s


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "x"


def to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def hydra_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        if isinstance(v, float):
            return f"{float(v):.12g}"
        return str(int(v))
    if isinstance(v, str):
        return f"'{v}'"
    if isinstance(v, list):
        return "[" + ",".join(hydra_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        items = []
        for k, val in v.items():
            items.append(f"{k}:{hydra_literal(val)}")
        return "{" + ",".join(items) + "}"
    return json.dumps(v, ensure_ascii=False)


def stage_budget(profile: str, stage: str) -> Dict[str, Any]:
    profile = str(profile).lower()
    stage = str(stage).upper()
    if profile not in {"balanced", "fast", "deep"}:
        profile = "balanced"
    table = {
        "balanced": {
            "A": {"max_evals": 1, "epochs": 18, "patience": 4, "algo": "random"},
            "B": {"max_evals": 30, "epochs": 50, "patience": 5, "algo": "tpe"},
            "C": {"max_evals": 20, "epochs": 80, "patience": 8, "algo": "tpe"},
            "D": {"max_evals": 12, "epochs": 100, "patience": 10, "algo": "tpe"},
        },
        "fast": {
            "A": {"max_evals": 1, "epochs": 12, "patience": 3, "algo": "random"},
            "B": {"max_evals": 20, "epochs": 45, "patience": 5, "algo": "tpe"},
            "C": {"max_evals": 14, "epochs": 70, "patience": 7, "algo": "tpe"},
            "D": {"max_evals": 8, "epochs": 90, "patience": 9, "algo": "tpe"},
        },
        "deep": {
            "A": {"max_evals": 1, "epochs": 24, "patience": 5, "algo": "random"},
            "B": {"max_evals": 40, "epochs": 50, "patience": 5, "algo": "tpe"},
            "C": {"max_evals": 30, "epochs": 80, "patience": 8, "algo": "tpe"},
            "D": {"max_evals": 20, "epochs": 100, "patience": 10, "algo": "tpe"},
        },
    }
    return dict(table[profile][stage])


def dataset_config_name(dataset: str) -> str:
    key = str(dataset).strip().lower()
    if key not in DATASET_CONFIG_MAP:
        raise KeyError(f"Unknown dataset config mapping: {dataset}")
    return DATASET_CONFIG_MAP[key]


def metrics_from_special(special_payload: Dict[str, Any]) -> Dict[str, float]:
    out = {
        "unseen_mrr20": 0.0,
        "unseen_hit20": 0.0,
        "unseen_count": 0.0,
        "seen_count": 0.0,
    }
    if not isinstance(special_payload, dict):
        return out
    overall = special_payload.get("overall", {}) or {}
    slices = special_payload.get("slices", {}) or {}
    pop = slices.get("target_popularity_abs", {}) or {}
    cold = pop.get("cold_0", {}) or {}
    total = float(overall.get("count", 0.0) or 0.0)
    cold_count = float(cold.get("count", 0.0) or 0.0)
    out["unseen_mrr20"] = float(cold.get("mrr@20", 0.0) or 0.0)
    out["unseen_hit20"] = float(cold.get("hit@20", 0.0) or 0.0)
    out["unseen_count"] = cold_count
    out["seen_count"] = max(0.0, total - cold_count)
    return out


def parse_result_metrics(result_path: Path) -> Dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    best_valid_result = payload.get("best_valid_result", {}) or {}
    test_result = payload.get("test_result", {}) or {}
    best_valid_special = payload.get("best_valid_special_metrics", {}) or {}
    test_special = payload.get("test_special_metrics", {}) or {}

    valid_filter = payload.get("best_valid_main_eval_filter", {}) or {}
    test_filter = payload.get("test_main_eval_filter", {}) or {}
    valid_cold = payload.get("best_valid_cold_target_metrics", {}) or {}
    test_cold = payload.get("test_cold_target_metrics", {}) or {}

    if not valid_cold:
        v = metrics_from_special(best_valid_special)
        valid_cold = {"mrr@20": v["unseen_mrr20"], "hit@20": v["unseen_hit20"], "count": v["unseen_count"]}
    if not test_cold:
        t = metrics_from_special(test_special)
        test_cold = {"mrr@20": t["unseen_mrr20"], "hit@20": t["unseen_hit20"], "count": t["unseen_count"]}

    if not valid_filter:
        v = metrics_from_special(best_valid_special)
        valid_filter = {
            "seen_targets": int(v["seen_count"]),
            "unseen_targets": int(v["unseen_count"]),
        }
    if not test_filter:
        t = metrics_from_special(test_special)
        test_filter = {
            "seen_targets": int(t["seen_count"]),
            "unseen_targets": int(t["unseen_count"]),
        }

    return {
        "best_valid_mrr20": float(best_valid_result.get("mrr@20", payload.get("best_mrr@20", 0.0)) or 0.0),
        "test_mrr20": float(test_result.get("mrr@20", payload.get("test_mrr@20", 0.0)) or 0.0),
        "valid_unseen_mrr20": float(valid_cold.get("mrr@20", 0.0) or 0.0),
        "valid_unseen_hit20": float(valid_cold.get("hit@20", 0.0) or 0.0),
        "test_unseen_mrr20": float(test_cold.get("mrr@20", 0.0) or 0.0),
        "test_unseen_hit20": float(test_cold.get("hit@20", 0.0) or 0.0),
        "valid_main_seen_count": int(valid_filter.get("seen_targets", 0) or 0),
        "valid_main_unseen_count": int(valid_filter.get("unseen_targets", 0) or 0),
        "test_main_seen_count": int(test_filter.get("seen_targets", 0) or 0),
        "test_main_unseen_count": int(test_filter.get("unseen_targets", 0) or 0),
    }


def read_existing_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = str(row.get("run_phase", "")).strip()
                if key:
                    out[key] = dict(row)
    except Exception:
        return {}
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(row.get("best_valid_mrr20", -1e9) or -1e9),
        float(row.get("test_mrr20", -1e9) or -1e9),
        float(row.get("test_unseen_mrr20", -1e9) or -1e9),
    )


def topk_by_group(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).strip() != "ok":
            continue
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        items_sorted = sorted(items, key=rank_key, reverse=True)
        out.extend(items_sorted[:k])
    return out


def top1_by_group_candidate_mean(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Select top-1 candidate per (dataset, model) using mean score across seeds."""
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        gk = (str(row.get("dataset", "")), str(row.get("model", "")))
        ck = str(row.get("candidate_id", ""))
        grouped.setdefault(gk, {}).setdefault(ck, []).append(row)

    out: List[Dict[str, Any]] = []
    for gk, cand_map in grouped.items():
        best_ck = None
        best_score = (-1e9, -1e9, -1e9)
        best_rows: List[Dict[str, Any]] = []
        for ck, items in cand_map.items():
            if not items:
                continue
            mean_valid = sum(float(r.get("best_valid_mrr20", 0.0) or 0.0) for r in items) / len(items)
            mean_test = sum(float(r.get("test_mrr20", 0.0) or 0.0) for r in items) / len(items)
            mean_cold = sum(float(r.get("test_unseen_mrr20", 0.0) or 0.0) for r in items) / len(items)
            score = (mean_valid, mean_test, mean_cold)
            if score > best_score:
                best_score = score
                best_ck = ck
                best_rows = items
        if best_ck is None or not best_rows:
            continue
        rep = sorted(best_rows, key=rank_key, reverse=True)[0]
        out.append(rep)
    return out


@dataclass
class Candidate:
    candidate_id: str
    parent_candidate_id: str
    model: str
    dataset: str
    params: Dict[str, Any]


def _model_priors(model: str) -> Dict[str, List[Any]]:
    m = str(model).lower()
    if m == "sasrec":
        return {
            "max_len": [20, 50, 100, 200],
            "num_layers": [1, 2, 3],
            "num_heads": [1, 2, 4],
            "hidden_size": [96, 128, 160, 192],
            "dropout": [0.10, 0.15, 0.20, 0.25],
            "weight_decay": [1e-6, 1e-5, 1e-4, 5e-4],
            "learning_rate": [8e-5, 2e-4, 4e-4, 8e-4, 1.2e-3],
        }
    if m == "tisasrec":
        return {
            "max_len": [20, 50, 100, 200],
            "num_layers": [1, 2, 3],
            "num_heads": [1, 2, 4],
            "hidden_size": [96, 128, 160, 192],
            "time_span": [64, 256, 1024, 4096],
            "dropout": [0.10, 0.15, 0.20, 0.25],
            "weight_decay": [1e-6, 1e-5, 1e-4, 5e-4],
            "learning_rate": [8e-5, 2e-4, 4e-4, 8e-4, 1.2e-3],
        }
    if m == "gru4rec":
        return {
            "max_len": [20, 50, 100, 200],
            "num_layers": [1, 2, 3],
            "hidden_size": [96, 128, 160, 192, 256],
            "dropout": [0.10, 0.20, 0.30],
            "weight_decay": [1e-6, 1e-5, 1e-4, 5e-4],
            "learning_rate": [8e-5, 2e-4, 4e-4, 8e-4, 1.2e-3],
        }
    if m == "fdsa":
        return {
            "max_len": [20, 50, 100, 200],
            "num_layers": [1, 2, 3],
            "num_heads": [1, 2, 4],
            "hidden_size": [96, 128, 160, 192],
            "dropout": [0.10, 0.15, 0.20, 0.25],
            "weight_decay": [1e-6, 1e-5, 1e-4, 5e-4],
            "learning_rate": [8e-5, 2e-4, 4e-4, 8e-4, 1.2e-3],
        }
    if m == "featured_moe_n3":
        return {
            "max_len": [20, 30, 40, 50],
            "embedding_size": [96, 128, 160, 192],
            "d_ff": [192, 256, 320, 384],
            "d_expert_hidden": [96, 128, 160, 192],
            "d_router_hidden": [48, 64, 80, 96],
            "dropout": [0.10, 0.15, 0.20],
            "weight_decay": [5e-7, 1e-6, 2e-6, 4e-6],
            "learning_rate": [8e-5, 1e-4, 2e-4, 4e-4, 8e-4],
        }
    raise KeyError(f"Unsupported model for staged runner: {model}")


def _core_distance(model: str, p: Dict[str, Any]) -> float:
    m = str(model).lower()
    if m in {"sasrec", "tisasrec", "fdsa"}:
        d = 0.0
        d += abs(int(p["max_len"]) - 50) / 50.0
        d += abs(int(p["num_layers"]) - 2) * 0.6
        d += abs(int(p["num_heads"]) - 2) * 0.4
        d += abs(int(p["hidden_size"]) - 128) / 64.0
        if m == "tisasrec":
            d += abs(math.log2(max(int(p["time_span"]), 1)) - math.log2(256)) * 0.2
        return d
    if m == "gru4rec":
        d = 0.0
        d += abs(int(p["max_len"]) - 50) / 50.0
        d += abs(int(p["num_layers"]) - 1) * 0.7
        d += abs(int(p["hidden_size"]) - 128) / 64.0
        return d
    d = 0.0
    d += abs(int(p["max_len"]) - 30) / 20.0
    d += abs(int(p["embedding_size"]) - 128) / 64.0
    d += abs(int(p["d_expert_hidden"]) - 128) / 64.0
    return d


def generate_stage_a_candidates(dataset: str, model: str, n: int = STAGE_A_STRUCT_COUNT_DEFAULT) -> List[Candidate]:
    pri = _model_priors(model)
    candidates: List[Dict[str, Any]] = []
    m = str(model).lower()

    if m == "sasrec":
        for max_len in pri["max_len"]:
            for layers in pri["num_layers"]:
                for heads in pri["num_heads"]:
                    for hidden in pri["hidden_size"]:
                        if hidden % heads != 0:
                            continue
                        candidates.append(
                            {
                                "max_len": int(max_len),
                                "num_layers": int(layers),
                                "num_heads": int(heads),
                                "hidden_size": int(hidden),
                                "inner_size": int(hidden) * 2,
                            }
                        )
    elif m == "tisasrec":
        for max_len in pri["max_len"]:
            for layers in pri["num_layers"]:
                for heads in pri["num_heads"]:
                    for hidden in pri["hidden_size"]:
                        for span in pri["time_span"]:
                            if hidden % heads != 0:
                                continue
                            candidates.append(
                                {
                                    "max_len": int(max_len),
                                    "num_layers": int(layers),
                                    "num_heads": int(heads),
                                    "hidden_size": int(hidden),
                                    "inner_size": int(hidden) * 2,
                                    "time_span": int(span),
                                }
                            )
    elif m == "gru4rec":
        for max_len in pri["max_len"]:
            for layers in pri["num_layers"]:
                for hidden in pri["hidden_size"]:
                    candidates.append(
                        {
                            "max_len": int(max_len),
                            "num_layers": int(layers),
                            "hidden_size": int(hidden),
                            "inner_size": int(hidden) * 2,
                        }
                    )
    elif m == "fdsa":
        for max_len in pri["max_len"]:
            for layers in pri["num_layers"]:
                for heads in pri["num_heads"]:
                    for hidden in pri["hidden_size"]:
                        if hidden % heads != 0:
                            continue
                        candidates.append(
                            {
                                "max_len": int(max_len),
                                "num_layers": int(layers),
                                "num_heads": int(heads),
                                "hidden_size": int(hidden),
                                "inner_size": int(hidden) * 2,
                            }
                        )
    else:
        for max_len in pri["max_len"]:
            for emb in pri["embedding_size"]:
                for dff in pri["d_ff"]:
                    for dex in pri["d_expert_hidden"]:
                        for dr in pri["d_router_hidden"]:
                            candidates.append(
                                {
                                    "max_len": int(max_len),
                                    "embedding_size": int(emb),
                                    "d_ff": int(dff),
                                    "d_expert_hidden": int(dex),
                                    "d_router_hidden": int(dr),
                                }
                            )

    candidates = sorted(candidates, key=lambda p: _core_distance(model, p))
    selected = candidates[: max(n, 1)]

    out: List[Candidate] = []
    for idx, p in enumerate(selected, start=1):
        base = dict(p)
        base["dropout"] = float(pri["dropout"][(idx - 1) % len(pri["dropout"])])
        base["weight_decay"] = float(pri["weight_decay"][(idx - 1) % len(pri["weight_decay"])])
        # Stage A: keep structural candidates, but evaluate each with LR grid.
        for lidx, lr in enumerate(STAGE_A_LR_GRID, start=1):
            q = dict(base)
            q["learning_rate"] = float(lr)
            lr_token = _format_lr_token(float(lr))
            q["lr_group"] = f"lr_{lr_token}"
            out.append(
                Candidate(
                    candidate_id=f"A{idx:03d}_L{lidx:02d}",
                    parent_candidate_id=f"A{idx:03d}",
                    model=model,
                    dataset=dataset,
                    params=q,
                )
            )
    return out


def _neighbor(value: int, ladder: List[int], step: int) -> int:
    ladder = sorted(set(int(x) for x in ladder))
    if int(value) not in ladder:
        ladder.append(int(value))
        ladder = sorted(set(ladder))
    i = ladder.index(int(value))
    j = min(max(i + int(step), 0), len(ladder) - 1)
    return int(ladder[j])


def _mutate_candidate(parent: Candidate, *, stage: str, variant_idx: int) -> Candidate:
    pri = _model_priors(parent.model)
    p = dict(parent.params)
    m = str(parent.model).lower()
    stage = str(stage).upper()

    mode = int(variant_idx)
    direction = 1 if (mode % 2 == 0) else -1

    # Continuous tweaks (both C/D)
    if stage == "C":
        p["learning_rate"] = float(max(1e-5, min(5e-2, float(p.get("learning_rate", 5e-4)) * (1.25 if direction > 0 else 0.8))))
        p["dropout"] = float(max(0.05, min(0.45, float(p.get("dropout", 0.2)) + (0.03 if direction > 0 else -0.03))))
        p["weight_decay"] = float(max(1e-7, min(1e-2, float(p.get("weight_decay", 1e-5)) * (1.4 if direction > 0 else 0.7))))
    else:
        p["learning_rate"] = float(max(1e-5, min(5e-2, float(p.get("learning_rate", 5e-4)) * (1.15 if direction > 0 else 0.87))))
        p["dropout"] = float(max(0.05, min(0.45, float(p.get("dropout", 0.2)) + (0.02 if direction > 0 else -0.02))))
        p["weight_decay"] = float(max(1e-7, min(1e-2, float(p.get("weight_decay", 1e-5)) * (1.2 if direction > 0 else 0.83))))

    # Structural tweaks by model family (C/D 모두 적용)
    if m in {"sasrec", "tisasrec", "fdsa"}:
        p["max_len"] = _neighbor(int(p.get("max_len", 50)), [int(x) for x in pri["max_len"]], step=direction)
        p["num_layers"] = _neighbor(int(p.get("num_layers", 2)), [int(x) for x in pri["num_layers"]], step=(-direction if mode % 3 == 2 else direction))
        p["num_heads"] = _neighbor(int(p.get("num_heads", 2)), [int(x) for x in pri["num_heads"]], step=(direction if mode % 3 == 0 else -direction))
        hidden = _neighbor(int(p.get("hidden_size", 128)), [int(x) for x in pri["hidden_size"]], step=(direction if mode % 3 != 1 else -direction))
        heads = int(max(1, p["num_heads"]))
        # Keep transformer dimension valid.
        if hidden % heads != 0:
            valid_h = [int(h) for h in pri["hidden_size"] if int(h) % heads == 0]
            if valid_h:
                hidden = min(valid_h, key=lambda x: abs(x - hidden))
        p["hidden_size"] = int(hidden)
        p["inner_size"] = int(p["hidden_size"]) * 2
        if m == "tisasrec":
            p["time_span"] = _neighbor(
                int(p.get("time_span", 256)),
                [int(x) for x in pri.get("time_span", [256])],
                step=(direction if mode % 3 != 0 else -direction),
            )
    elif m == "gru4rec":
        p["max_len"] = _neighbor(int(p.get("max_len", 50)), [int(x) for x in pri["max_len"]], step=direction)
        p["num_layers"] = _neighbor(int(p.get("num_layers", 1)), [int(x) for x in pri["num_layers"]], step=(-direction if mode % 3 == 1 else direction))
        p["hidden_size"] = _neighbor(int(p.get("hidden_size", 128)), [int(x) for x in pri["hidden_size"]], step=(direction if mode % 3 != 2 else -direction))
        p["inner_size"] = int(p["hidden_size"]) * 2
    elif m == "featured_moe_n3":
        p["max_len"] = _neighbor(int(p.get("max_len", 30)), [int(x) for x in pri["max_len"]], step=direction)
        p["embedding_size"] = _neighbor(
            int(p.get("embedding_size", 128)),
            [int(x) for x in pri["embedding_size"]],
            step=(direction if mode % 3 != 1 else -direction),
        )
        p["d_ff"] = _neighbor(int(p.get("d_ff", 256)), [int(x) for x in pri["d_ff"]], step=(direction if mode % 3 == 0 else -direction))
        p["d_expert_hidden"] = _neighbor(
            int(p.get("d_expert_hidden", 128)),
            [int(x) for x in pri["d_expert_hidden"]],
            step=(direction if mode % 3 != 2 else -direction),
        )
        p["d_router_hidden"] = _neighbor(
            int(p.get("d_router_hidden", 64)),
            [int(x) for x in pri["d_router_hidden"]],
            step=(-direction if mode % 3 == 0 else direction),
        )

    return Candidate(
        candidate_id="",
        parent_candidate_id=str(parent.candidate_id),
        model=parent.model,
        dataset=parent.dataset,
        params=p,
    )


def mutate_candidates(parents: List[Candidate], *, stage: str, per_parent: int) -> List[Candidate]:
    out: List[Candidate] = []
    cursor = 0
    for parent in parents:
        for k in range(per_parent):
            cursor += 1
            c = _mutate_candidate(parent, stage=stage, variant_idx=k)
            c.candidate_id = f"{stage}{cursor:03d}"
            out.append(c)
    return out


def extract_candidates_from_rows(rows: List[Dict[str, Any]]) -> List[Candidate]:
    out: List[Candidate] = []
    for row in rows:
        params_text = str(row.get("params_json", "")).strip()
        if not params_text:
            continue
        try:
            params = json.loads(params_text)
        except Exception:
            continue
        out.append(
            Candidate(
                candidate_id=str(row.get("candidate_id", "")),
                parent_candidate_id=str(row.get("parent_candidate_id", "")),
                model=str(row.get("model", "")),
                dataset=str(row.get("dataset", "")),
                params=params,
            )
        )
    return out


def parse_result_path_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    pat = re.compile(r"Results\s*->\s*(.+)$")
    for line in reversed(lines):
        m = pat.search(line.strip())
        if m:
            return Path(m.group(1).strip()).expanduser()
    return None


def has_run_status_end_normal(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        token = str(line).strip()
        if not token:
            continue
        return bool(RUN_STATUS_END_NORMAL_RE.search(token))
    return False


def build_run_tokens(*, axis: str, stage: str, dataset: str, model: str, candidate_id: str, run_seed: int) -> tuple[str, str]:
    ds_tag = sanitize(dataset).upper()
    m_tag = sanitize(model).upper()
    run_phase = f"{axis}_{stage}_D{ds_tag}_M{m_tag}_{candidate_id}_S{int(run_seed)}"
    run_id = f"{stage}_{ds_tag}_{m_tag}_{candidate_id}_S{int(run_seed)}"
    return run_phase, run_id


def resolve_log_path(*, axis_root: Path, stage: str, candidate: Candidate, run_phase: str) -> Path:
    model_stage_root = axis_root / candidate.dataset / candidate.model / f"stage{stage}"
    if str(stage).upper() == "A" and str(candidate.params.get("lr_group", "")).strip():
        log_dir = model_stage_root / "lr_groups" / str(candidate.params.get("lr_group")) / "logs"
    else:
        log_dir = model_stage_root / "logs"
    return log_dir / f"{run_phase}.log"


def build_resumed_row_from_log(
    *,
    stage: str,
    candidate: Candidate,
    run_seed: int,
    gpu_id: str,
    axis: str,
    axis_root: Path,
) -> Dict[str, Any] | None:
    run_phase, run_id = build_run_tokens(
        axis=axis,
        stage=stage,
        dataset=candidate.dataset,
        model=candidate.model,
        candidate_id=candidate.candidate_id,
        run_seed=run_seed,
    )
    log_path = resolve_log_path(axis_root=axis_root, stage=stage, candidate=candidate, run_phase=run_phase)
    if not has_run_status_end_normal(log_path):
        return None

    result_path = parse_result_path_from_log(log_path)
    metrics = parse_result_metrics(result_path) if result_path is not None else {}
    return {
        "stage": stage,
        "dataset": candidate.dataset,
        "model": candidate.model,
        "model_label": MODEL_LABELS.get(candidate.model, candidate.model),
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_candidate_id,
        "run_phase": run_phase,
        "run_id": run_id,
        "runtime_seed": int(run_seed),
        "gpu_id": str(gpu_id),
        "status": "ok",
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "valid_unseen_mrr20": metrics.get("valid_unseen_mrr20", 0.0),
        "valid_unseen_hit20": metrics.get("valid_unseen_hit20", 0.0),
        "test_unseen_mrr20": metrics.get("test_unseen_mrr20", 0.0),
        "test_unseen_hit20": metrics.get("test_unseen_hit20", 0.0),
        "valid_main_seen_count": metrics.get("valid_main_seen_count", 0),
        "valid_main_unseen_count": metrics.get("valid_main_unseen_count", 0),
        "test_main_seen_count": metrics.get("test_main_seen_count", 0),
        "test_main_unseen_count": metrics.get("test_main_unseen_count", 0),
        "result_path": "" if result_path is None else str(result_path),
        "log_path": str(log_path),
        "timestamp_utc": now_utc(),
        "params_json": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        "elapsed_sec": 0.0,
    }


def _base_runtime_overrides(model: str, params: Dict[str, Any]) -> List[str]:
    m = str(model).lower()
    o = [
        f"++MAX_ITEM_LIST_LENGTH={int(params.get('max_len', 50))}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
    ]

    if m in {"sasrec", "tisasrec", "fdsa"}:
        hidden = int(params.get("hidden_size", 128))
        layers = int(params.get("num_layers", 2))
        heads = int(params.get("num_heads", 2))
        inner = int(params.get("inner_size", hidden * 2))
        dropout = float(params.get("dropout", 0.2))
        o.extend(
            [
                f"++hidden_size={hidden}",
                f"++embedding_size={hidden}",
                f"++n_layers={layers}",
                f"++num_layers={layers}",
                f"++n_heads={heads}",
                f"++num_heads={heads}",
                f"++inner_size={inner}",
                f"++dropout_ratio={dropout}",
            ]
        )
        if m == "tisasrec":
            o.append(f"++time_span={int(params.get('time_span', 256))}")
        if m == "fdsa":
            o.extend(
                [
                    "++selected_features=['category']",
                    "++pooling_mode=mean",
                ]
            )
    elif m == "gru4rec":
        hidden = int(params.get("hidden_size", 128))
        layers = int(params.get("num_layers", 1))
        dropout = float(params.get("dropout", 0.2))
        o.extend(
            [
                f"++hidden_size={hidden}",
                f"++embedding_size={hidden}",
                f"++num_layers={layers}",
                f"++dropout_prob={dropout}",
            ]
        )
    elif m == "featured_moe_n3":
        emb = int(params.get("embedding_size", 128))
        o.extend(
            [
                f"++embedding_size={emb}",
                f"++hidden_size={emb}",
                f"++d_ff={int(params.get('d_ff', emb * 2))}",
                f"++d_expert_hidden={int(params.get('d_expert_hidden', emb))}",
                f"++d_router_hidden={int(params.get('d_router_hidden', max(emb // 2, 32)))}",
                f"++fixed_hidden_dropout_prob={float(params.get('dropout', 0.15))}",
                f"++fixed_weight_decay={float(params.get('weight_decay', 1e-6))}",
            ]
        )
    return o


def _search_space_entries(stage: str, model: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    stage = str(stage).upper()
    m = str(model).lower()
    search: Dict[str, Any] = {}
    types: Dict[str, str] = {}

    # Always keep structural keys fixed per candidate.
    fixed = {
        "MAX_ITEM_LIST_LENGTH": [int(params.get("max_len", 50))],
        "weight_decay": [float(params.get("weight_decay", 1e-5))],
    }
    if m in {"sasrec", "tisasrec", "fdsa"}:
        fixed.update(
            {
                "hidden_size": [int(params.get("hidden_size", 128))],
                "embedding_size": [int(params.get("hidden_size", 128))],
                "n_layers": [int(params.get("num_layers", 2))],
                "num_layers": [int(params.get("num_layers", 2))],
                "n_heads": [int(params.get("num_heads", 2))],
                "num_heads": [int(params.get("num_heads", 2))],
                "inner_size": [int(params.get("inner_size", int(params.get("hidden_size", 128)) * 2))],
            }
        )
        if m == "tisasrec":
            fixed["time_span"] = [int(params.get("time_span", 256))]
        if m == "fdsa":
            fixed["selected_features"] = [["category"]]
            fixed["pooling_mode"] = ["mean"]
    elif m == "gru4rec":
        fixed.update(
            {
                "hidden_size": [int(params.get("hidden_size", 128))],
                "embedding_size": [int(params.get("hidden_size", 128))],
                "num_layers": [int(params.get("num_layers", 1))],
            }
        )
    elif m == "featured_moe_n3":
        fixed.update(
            {
                "embedding_size": [int(params.get("embedding_size", 128))],
                "hidden_size": [int(params.get("embedding_size", 128))],
                "d_ff": [int(params.get("d_ff", 256))],
                "d_expert_hidden": [int(params.get("d_expert_hidden", 128))],
                "d_router_hidden": [int(params.get("d_router_hidden", 64))],
                "fixed_weight_decay": [float(params.get("weight_decay", 1e-6))],
                "fixed_hidden_dropout_prob": [float(params.get("dropout", 0.15))],
            }
        )

    for k, v in fixed.items():
        search[k] = v
        types[k] = "choice"

    lr = float(params.get("learning_rate", 5e-4))
    wd = float(params.get("weight_decay", 1e-5))
    dr = float(params.get("dropout", 0.2))

    if stage == "A":
        search["learning_rate"] = [lr]
        types["learning_rate"] = "choice"
        if m in {"sasrec", "tisasrec", "fdsa"}:
            search["dropout_ratio"] = [dr]
            search["hidden_dropout_prob"] = [dr]
            search["attn_dropout_prob"] = [dr]
            types["dropout_ratio"] = "choice"
            types["hidden_dropout_prob"] = "choice"
            types["attn_dropout_prob"] = "choice"
        elif m == "gru4rec":
            search["dropout_prob"] = [dr]
            # Keep Stage A as true single-run screening by pinning this as well.
            search["hidden_dropout_prob"] = [dr]
            types["dropout_prob"] = "choice"
            types["hidden_dropout_prob"] = "choice"
    else:
        lr_lo = max(1e-6, lr * (0.25 if stage in {"B", "C"} else 0.5))
        lr_hi = min(5e-2, lr * (4.0 if stage in {"B", "C"} else 2.0))
        wd_lo = max(1e-8, wd * (0.1 if stage in {"B", "C"} else 0.4))
        wd_hi = min(1e-2, wd * (10.0 if stage in {"B", "C"} else 2.5))
        dr_lo = max(0.05, dr - (0.12 if stage in {"B", "C"} else 0.08))
        dr_hi = min(0.5, dr + (0.12 if stage in {"B", "C"} else 0.08))

        search["learning_rate"] = [lr_lo, lr_hi]
        types["learning_rate"] = "loguniform"
        search["weight_decay"] = [wd_lo, wd_hi]
        types["weight_decay"] = "loguniform"
        if m in {"sasrec", "tisasrec", "fdsa"}:
            search["dropout_ratio"] = [dr_lo, dr_hi]
            search["hidden_dropout_prob"] = [dr_lo, dr_hi]
            search["attn_dropout_prob"] = [dr_lo, dr_hi]
            types["dropout_ratio"] = "uniform"
            types["hidden_dropout_prob"] = "uniform"
            types["attn_dropout_prob"] = "uniform"
        elif m == "gru4rec":
            search["dropout_prob"] = [dr_lo, dr_hi]
            types["dropout_prob"] = "uniform"
        elif m == "featured_moe_n3":
            search["fixed_hidden_dropout_prob"] = [dr_lo, dr_hi]
            types["fixed_hidden_dropout_prob"] = "uniform"
            search["fixed_weight_decay"] = [wd_lo, wd_hi]
            types["fixed_weight_decay"] = "loguniform"

    return search, types


def build_command(
    *,
    stage: str,
    track: str,
    axis: str,
    candidate: Candidate,
    run_phase: str,
    run_seed: int,
    gpu_id: str,
    budget: Dict[str, Any],
) -> List[str]:
    model = str(candidate.model).lower()
    dataset = str(candidate.dataset)
    config_name = dataset_config_name(dataset)
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    search, types = _search_space_entries(stage, model, candidate.params)

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        config_name,
        "--max-evals",
        str(int(budget["max_evals"])),
        "--tune-epochs",
        str(int(budget["epochs"])),
        "--tune-patience",
        str(int(budget["patience"])),
        "--search-algo",
        str(budget["algo"]),
        "--seed",
        str(int(run_seed)),
        "--run-group",
        str(track),
        "--run-axis",
        str(axis),
        "--run-phase",
        str(run_phase),
        f"model={model}",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=full_v4",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
    ]
    cmd.extend(_base_runtime_overrides(model, candidate.params))
    cmd.append(f"++seed={int(run_seed)}")

    for k, v in search.items():
        cmd.append(f"++search.{k}={hydra_literal(v)}")
    for k, v in types.items():
        cmd.append(f"++search_space_type_overrides.{k}={v}")
    return cmd


def run_one(
    *,
    stage: str,
    candidate: Candidate,
    run_seed: int,
    gpu_id: str,
    budget: Dict[str, Any],
    track: str,
    axis: str,
    axis_root: Path,
) -> Dict[str, Any]:
    run_phase, run_id = build_run_tokens(
        axis=axis,
        stage=stage,
        dataset=candidate.dataset,
        model=candidate.model,
        candidate_id=candidate.candidate_id,
        run_seed=run_seed,
    )
    log_path = resolve_log_path(axis_root=axis_root, stage=stage, candidate=candidate, run_phase=run_phase)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        stage=stage,
        track=track,
        axis=axis,
        candidate=candidate,
        run_phase=run_phase,
        run_seed=run_seed,
        gpu_id=gpu_id,
        budget=budget,
    )
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    start = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# stage={stage} dataset={candidate.dataset} model={candidate.model} candidate={candidate.candidate_id}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=str(EXP_DIR), env=env, stdout=fh, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    status = "ok" if int(rc) == 0 else "fail"
    result_path = parse_result_path_from_log(log_path)
    metrics = {}
    if status == "ok" and result_path is not None:
        metrics = parse_result_metrics(result_path)
    row = {
        "stage": stage,
        "dataset": candidate.dataset,
        "model": candidate.model,
        "model_label": MODEL_LABELS.get(candidate.model, candidate.model),
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_candidate_id,
        "run_phase": run_phase,
        "run_id": run_id,
        "runtime_seed": int(run_seed),
        "gpu_id": str(gpu_id),
        "status": status,
        "best_valid_mrr20": metrics.get("best_valid_mrr20", 0.0),
        "test_mrr20": metrics.get("test_mrr20", 0.0),
        "valid_unseen_mrr20": metrics.get("valid_unseen_mrr20", 0.0),
        "valid_unseen_hit20": metrics.get("valid_unseen_hit20", 0.0),
        "test_unseen_mrr20": metrics.get("test_unseen_mrr20", 0.0),
        "test_unseen_hit20": metrics.get("test_unseen_hit20", 0.0),
        "valid_main_seen_count": metrics.get("valid_main_seen_count", 0),
        "valid_main_unseen_count": metrics.get("valid_main_unseen_count", 0),
        "test_main_seen_count": metrics.get("test_main_seen_count", 0),
        "test_main_unseen_count": metrics.get("test_main_unseen_count", 0),
        "result_path": "" if result_path is None else str(result_path),
        "log_path": str(log_path),
        "timestamp_utc": now_utc(),
        "params_json": json.dumps(candidate.params, ensure_ascii=False, sort_keys=True),
        "elapsed_sec": round(float(elapsed), 1),
    }
    if status != "ok":
        row["error"] = f"return_code={int(rc)}"
    return row


def build_stage_candidates(
    *,
    stage: str,
    dataset: str,
    model: str,
    prev_promoted: List[Candidate],
) -> List[Candidate]:
    s = str(stage).upper()
    if s == "A":
        return generate_stage_a_candidates(dataset, model, n=STAGE_A_STRUCT_COUNT_DEFAULT)
    if s == "B":
        # A -> B: top-12 promoted (runtime optimization)
        return prev_promoted[:12]
    if s == "C":
        # B top-4 -> C mutated 12 candidates
        return mutate_candidates(prev_promoted[:4], stage="C", per_parent=3)
    if s == "D":
        # C top-2 -> D mutated 6 candidates
        return mutate_candidates(prev_promoted[:2], stage="D", per_parent=3)
    raise ValueError(f"Unknown stage: {stage}")


def stage_output_paths(track: str, axis: str, stage: str) -> Dict[str, Path]:
    stage_u = str(stage).upper()
    axis_root = ARTIFACT_ROOT / "logs" / str(track) / str(axis)
    stage_root = axis_root / "stages" / f"stage{stage_u}"
    return {
        "axis_root": axis_root,
        "stage_root": stage_root,
        "summary_csv": stage_root / "summary.csv",
        "promotion_csv": stage_root / "promotion.csv",
        "leaderboard_csv": stage_root / "leaderboard.csv",
        "manifest_json": stage_root / "manifest.json",
    }


def _stage_sort_key(stage: str) -> int:
    s = str(stage).upper()
    if s == "A":
        return 1
    if s == "B":
        return 2
    if s == "C":
        return 3
    if s == "D":
        return 4
    return 99


def update_dataset_rollup(axis_root: Path, rows: List[Dict[str, Any]], stage: str) -> None:
    ok_rows = [r for r in rows if str(r.get("status", "")) == "ok"]
    by_ds_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in ok_rows:
        key = (str(r.get("dataset", "")), str(r.get("model", "")))
        by_ds_model.setdefault(key, []).append(r)

    for (dataset, model), items in by_ds_model.items():
        best = sorted(items, key=rank_key, reverse=True)[0]
        ds_root = axis_root / dataset
        ds_root.mkdir(parents=True, exist_ok=True)
        summary_path = ds_root / "summary.csv"

        existing: List[Dict[str, str]] = []
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8", newline="") as fh:
                    existing = list(csv.DictReader(fh))
            except Exception:
                existing = []

        prev_cum_valid = 0.0
        prev_cum_test = 0.0
        for row in existing:
            if str(row.get("model", "")) != model:
                continue
            try:
                prev_cum_valid = max(prev_cum_valid, float(row.get("cum_best_valid_mrr20", 0.0) or 0.0))
            except Exception:
                pass
            try:
                prev_cum_test = max(prev_cum_test, float(row.get("cum_best_test_mrr20", 0.0) or 0.0))
            except Exception:
                pass

        stage_best_valid = float(best.get("best_valid_mrr20", 0.0) or 0.0)
        stage_best_test = float(best.get("test_mrr20", 0.0) or 0.0)
        new_row = {
            "stage": str(stage).upper(),
            "model": model,
            "best_candidate_id": str(best.get("candidate_id", "")),
            "best_valid_mrr20": stage_best_valid,
            "best_test_mrr20": stage_best_test,
            "cum_best_valid_mrr20": max(prev_cum_valid, stage_best_valid),
            "cum_best_test_mrr20": max(prev_cum_test, stage_best_test),
            "last_update_utc": now_utc(),
        }

        existing.append({k: str(v) for k, v in new_row.items()})
        existing.sort(key=lambda r: (str(r.get("model", "")), _stage_sort_key(str(r.get("stage", "")))))
        write_csv(
            summary_path,
            existing,
            [
                "stage",
                "model",
                "best_candidate_id",
                "best_valid_mrr20",
                "best_test_mrr20",
                "cum_best_valid_mrr20",
                "cum_best_test_mrr20",
                "last_update_utc",
            ],
        )


def _promote_n_for_stage(stage: str) -> int:
    s = str(stage).upper()
    if s == "A":
        return 12
    if s == "B":
        return 4
    if s == "C":
        return 2
    if s == "D":
        return 1
    return 1


def build_leaderboard(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        key = (str(row.get("dataset", "")), str(row.get("model", "")))
        grouped.setdefault(key, []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, model), items in grouped.items():
        best = sorted(items, key=rank_key, reverse=True)[0]
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "model_label": best.get("model_label", ""),
                "best_candidate_id": best.get("candidate_id", ""),
                "best_valid_mrr20": best.get("best_valid_mrr20", 0.0),
                "test_mrr20": best.get("test_mrr20", 0.0),
                "test_unseen_mrr20": best.get("test_unseen_mrr20", 0.0),
                "test_main_seen_count": best.get("test_main_seen_count", 0),
                "test_main_unseen_count": best.get("test_main_unseen_count", 0),
                "result_path": best.get("result_path", ""),
            }
        )
    out.sort(key=lambda r: (str(r["dataset"]), str(r["model"])))
    return out


def load_previous_promotions(track: str, axis: str, prev_stage: str) -> List[Dict[str, Any]]:
    paths = stage_output_paths(track, axis, prev_stage)
    promo = paths["promotion_csv"]
    if not promo.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with promo.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                out.append(dict(row))
    except Exception:
        return []
    return out


def resolve_stage_inputs(track: str, axis: str, stage: str, dataset: str, model: str) -> List[Candidate]:
    stage = str(stage).upper()
    if stage == "A":
        return []
    prev_stage = {"B": "A", "C": "B", "D": "C"}[stage]
    prev_rows = load_previous_promotions(track, axis, prev_stage)
    selected = [r for r in prev_rows if str(r.get("dataset", "")) == dataset and str(r.get("model", "")) == model]
    return extract_candidates_from_rows(selected)


def run_stage(args: argparse.Namespace, stage: str) -> Dict[str, Any]:
    stage = str(stage).upper()
    if stage not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid stage: {stage}")
    datasets = parse_csv_list(args.datasets) if str(args.datasets).strip() else list(DEFAULT_DATASETS)
    if args.track == "baseline_2":
        default_models = list(DEFAULT_BASELINE_MODELS)
    else:
        default_models = list(DEFAULT_FMOE_MODELS)
    models = parse_csv_list(args.models) if str(args.models).strip() else default_models
    for m in models:
        if m not in MODEL_LABELS:
            raise ValueError(f"Unsupported model for staged runner: {m}")

    if bool(args.fast_first):
        datasets = sorted(datasets, key=lambda d: (DATASET_SPEED_RANK.get(str(d).lower(), 999), str(d).lower()))
        models = sorted(models, key=lambda m: (MODEL_SPEED_RANK.get(str(m).lower(), 999), str(m).lower()))
    gpus = parse_csv_list(args.gpus) if str(args.gpus).strip() else ["0"]
    budget = stage_budget(args.budget_profile, stage)
    paths = stage_output_paths(args.track, args.axis, stage)
    stage_root = paths["stage_root"]
    axis_root = paths["axis_root"]
    stage_root.mkdir(parents=True, exist_ok=True)

    existing = read_existing_summary(paths["summary_csv"]) if args.resume_from_logs else {}

    all_rows: List[Dict[str, Any]] = []
    pending_jobs: List[Dict[str, Any]] = []
    run_idx = 0
    for ds in datasets:
        for model in models:
            prev_promoted = resolve_stage_inputs(args.track, args.axis, stage, ds, model)
            if stage != "A" and not prev_promoted:
                print(f"[skip] stage={stage} dataset={ds} model={model} (no previous promotions)")
                continue
            candidates = build_stage_candidates(
                stage=stage,
                dataset=ds,
                model=model,
                prev_promoted=prev_promoted,
            )
            if stage == "B":
                candidates = candidates[:12]
            elif stage == "C":
                candidates = candidates[:12]
            elif stage == "D":
                candidates = candidates[:6]

            stage_seed = int(args.runtime_seed)
            if stage == "D" and str(args.final_seeds).strip():
                seeds = [int(x) for x in parse_csv_list(args.final_seeds)]
            else:
                seeds = [stage_seed]

            for cand in candidates:
                for seed in seeds:
                    run_idx += 1
                    gpu_id = gpus[(run_idx - 1) % len(gpus)]
                    run_phase_key, _ = build_run_tokens(
                        axis=args.axis,
                        stage=stage,
                        dataset=ds,
                        model=model,
                        candidate_id=cand.candidate_id,
                        run_seed=seed,
                    )
                    if args.resume_from_logs and run_phase_key in existing and str(existing[run_phase_key].get("status", "")) == "ok":
                        row = dict(existing[run_phase_key])
                        all_rows.append(row)
                        continue
                    if args.resume_from_logs:
                        resumed = build_resumed_row_from_log(
                            stage=stage,
                            candidate=cand,
                            run_seed=int(seed),
                            gpu_id=str(gpu_id),
                            axis=args.axis,
                            axis_root=axis_root,
                        )
                        if resumed is not None:
                            print(
                                f"[resume-skip] stage={stage} dataset={ds} model={model} "
                                f"candidate={cand.candidate_id} seed={int(seed)}"
                            )
                            all_rows.append(resumed)
                            continue
                    pending_jobs.append(
                        {
                            "stage": stage,
                            "candidate": cand,
                            "seed": int(seed),
                            "gpu_id": str(gpu_id),
                            "budget": budget,
                            "track": args.track,
                            "axis": args.axis,
                            "axis_root": axis_root,
                            "dataset": ds,
                            "model": model,
                        }
                    )

    # Run pending jobs in parallel across GPUs.
    if pending_jobs:
        print(f"[parallel] stage={stage} launching {len(pending_jobs)} jobs on GPUs={','.join(gpus)}")
        job_queue: Queue = Queue()
        for job in pending_jobs:
            job_queue.put(job)

        row_lock = threading.Lock()

        def _gpu_worker(gpu_id: str) -> None:
            while True:
                try:
                    job = job_queue.get_nowait()
                except Empty:
                    return
                cand = job["candidate"]
                seed = int(job["seed"])
                # Force worker GPU to avoid accidental mismatch from queued hint.
                job["gpu_id"] = str(gpu_id)
                print(
                    f"[run] stage={stage} dataset={job['dataset']} model={job['model']} "
                    f"candidate={cand.candidate_id} seed={seed} gpu={gpu_id}"
                )
                try:
                    row = run_one(
                        stage=job["stage"],
                        candidate=cand,
                        run_seed=seed,
                        gpu_id=str(gpu_id),
                        budget=job["budget"],
                        track=job["track"],
                        axis=job["axis"],
                        axis_root=job["axis_root"],
                    )
                except Exception as e:
                    row = {
                        "stage": stage,
                        "dataset": job["dataset"],
                        "model": job["model"],
                        "model_label": MODEL_LABELS.get(job["model"], job["model"]),
                        "candidate_id": cand.candidate_id,
                        "parent_candidate_id": cand.parent_candidate_id,
                        "run_phase": "",
                        "run_id": "",
                        "runtime_seed": seed,
                        "gpu_id": str(gpu_id),
                        "status": "fail",
                        "best_valid_mrr20": 0.0,
                        "test_mrr20": 0.0,
                        "valid_unseen_mrr20": 0.0,
                        "valid_unseen_hit20": 0.0,
                        "test_unseen_mrr20": 0.0,
                        "test_unseen_hit20": 0.0,
                        "valid_main_seen_count": 0,
                        "valid_main_unseen_count": 0,
                        "test_main_seen_count": 0,
                        "test_main_unseen_count": 0,
                        "result_path": "",
                        "log_path": "",
                        "elapsed_sec": 0.0,
                        "error": f"worker_exception={e}",
                        "timestamp_utc": now_utc(),
                        "params_json": json.dumps(cand.params, ensure_ascii=False, sort_keys=True),
                    }
                with row_lock:
                    all_rows.append(row)
                job_queue.task_done()

        workers: List[threading.Thread] = []
        for gpu_id in gpus:
            t = threading.Thread(target=_gpu_worker, args=(str(gpu_id),), daemon=False)
            t.start()
            workers.append(t)
        for t in workers:
            t.join()

    promote_k = _promote_n_for_stage(stage)
    if stage == "D" and len(parse_csv_list(args.final_seeds)) > 1:
        promoted = top1_by_group_candidate_mean(all_rows)
    else:
        promoted = topk_by_group(all_rows, k=promote_k)
    leaderboard = build_leaderboard(all_rows)
    manifest = {
        "track": args.track,
        "axis": args.axis,
        "stage": stage,
        "budget_profile": args.budget_profile,
        "budget": budget,
        "datasets": datasets,
        "models": models,
        "gpus": gpus,
        "runtime_seed": int(args.runtime_seed),
        "final_seeds": parse_csv_list(args.final_seeds),
        "n_rows": len(all_rows),
        "n_ok": sum(1 for r in all_rows if str(r.get("status", "")) == "ok"),
        "n_promoted": len(promoted),
        "timestamp_utc": now_utc(),
    }

    write_csv(paths["summary_csv"], all_rows, SUMMARY_FIELDS)
    write_csv(paths["promotion_csv"], promoted, SUMMARY_FIELDS)
    write_csv(
        paths["leaderboard_csv"],
        leaderboard,
        [
            "dataset",
            "model",
            "model_label",
            "best_candidate_id",
            "best_valid_mrr20",
            "test_mrr20",
            "test_unseen_mrr20",
            "test_main_seen_count",
            "test_main_unseen_count",
            "result_path",
        ],
    )
    paths["manifest_json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_dataset_rollup(axis_root, all_rows, stage)
    print(f"[stage-done] stage={stage} rows={len(all_rows)} promoted={len(promoted)} summary={paths['summary_csv']}")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=str, default="A", choices=["A", "B", "C", "D"])
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--track", type=str, default="baseline_2")
    p.add_argument("--axis", type=str, default="ABCD_v1")
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--models", type=str, default="")
    p.add_argument("--gpus", type=str, default="0")
    p.add_argument("--runtime-seed", type=int, default=1)
    p.add_argument("--final-seeds", type=str, default="1,2,3")
    p.add_argument("--budget-profile", type=str, default="balanced", choices=["balanced", "fast", "deep"])
    p.add_argument("--resume-from-logs", action="store_true")
    p.add_argument("--fast-first", dest="fast_first", action="store_true", default=True)
    p.add_argument("--no-fast-first", dest="fast_first", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_all:
        for s in ["A", "B", "C", "D"]:
            print(f"\n[run-all] stage={s}")
            run_stage(args, s)
        print("\n[done] all stages completed")
        return
    run_stage(args, args.stage)


if __name__ == "__main__":
    main()
