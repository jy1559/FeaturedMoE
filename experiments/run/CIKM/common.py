#!/usr/bin/env python3
"""Shared infrastructure for CIKM 2026 RouteRec experiments.

All experiments use:
  feature_mode = final   → Datasets/processed/final_dataset/
  eval_mode    = session_fixed
  epochs = 100, patience = 10
  hyperopt max_evals = 5  (narrow lr/wd search around sampled-dataset best)
"""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[3]
EXP_DIR        = REPO_ROOT / "experiments"
RUN_DIR        = EXP_DIR / "run"
CIKM_DIR       = RUN_DIR / "CIKM"
ARTIFACT_ROOT  = RUN_DIR / "artifacts"
DATASET_ROOT   = REPO_ROOT / "Datasets" / "processed"
FINAL_DATA_ROOT = DATASET_ROOT / "final_dataset"
LIGHT_DATA_ROOT = DATASET_ROOT / "final_dataset_light"

TRACK = "cikm_main"
LOG_ROOT    = CIKM_DIR / "logs"
RESULT_ROOT = CIKM_DIR / "results"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# ── constants ─────────────────────────────────────────────────────────────────
PYTHON_BIN = "/venv/FMoE/bin/python"

DATASETS = ["KuaiRec", "lastfm"]

BASELINE_MODELS = [
    "sasrec", "gru4rec", "tisasrec",
    "duorec", "bsarec", "fearec",
    "difsr", "fame", "fdsa",
]
ROUTE_MODEL = "featured_moe_n3"
ALL_MODELS  = [*BASELINE_MODELS, ROUTE_MODEL]

TUNE_EPOCHS   = 100
TUNE_PATIENCE = 10
MAX_EVALS     = 5
SEARCH_ALGO   = "tpe"

# ── per-dataset tune config name ───────────────────────────────────────────────
DATASET_TUNE_CONFIG = {
    "KuaiRec": "tune_kuai_cikm",
    "lastfm":  "tune_lfm_cikm",
}

# ── narrow search spaces ───────────────────────────────────────────────────────
# Derived from best configs on sampled datasets (KuaiRecLargeStrictPosV2_0.2 and lastfm0.03).
# Format: {param: [value, ...]}  – lists are passed as ++search.param=[...] to hyperopt_tune.py
# Only lr and wd are tuned; all other params are fixed (best_fixed).

NARROW_SEARCH: dict[str, dict[str, dict[str, list[Any]]]] = {
    "KuaiRec": {
        "sasrec":  {"learning_rate": [8e-4, 1.2e-3, 2e-3, 3e-3, 5e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "gru4rec": {"learning_rate": [5e-4, 8e-4, 1.2e-3, 2e-3, 3e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "tisasrec":{"learning_rate": [5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "duorec":  {"learning_rate": [3e-4, 5e-4, 8e-4, 1.2e-3, 2e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "bsarec":  {"learning_rate": [5e-4, 8e-4, 1.2e-3, 2e-3, 3e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "fearec":  {"learning_rate": [2e-4, 4e-4, 6e-4, 1e-3, 2e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "difsr":   {"learning_rate": [2e-4, 4e-4, 6e-4, 1e-3, 2e-3],
                    "weight_decay":  [0.0, 1e-6, 1e-5, 1e-4]},
        "fame":    {"learning_rate": [4e-4, 8e-4, 1.2e-3, 2e-3, 3e-3],
                    "weight_decay":  [0.0, 5e-5, 2e-4]},
        "fdsa":    {"learning_rate": [1e-4, 3e-4, 5e-4, 8e-4, 1.5e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "featured_moe_n3": {
            "learning_rate": [3e-4, 6e-4, 1e-3, 1.5e-3, 2.5e-3],
            "weight_decay":  [0.0, 1e-6, 5e-6, 1e-5],
        },
    },
    "lastfm": {
        "sasrec":  {"learning_rate": [3e-4, 5e-4, 8e-4, 1.2e-3, 2e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "gru4rec": {"learning_rate": [4e-4, 8e-4, 1.2e-3, 2e-3, 3e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "tisasrec":{"learning_rate": [3e-4, 5e-4, 8e-4, 1.2e-3, 2e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "duorec":  {"learning_rate": [2e-4, 4e-4, 6e-4, 1e-3, 1.5e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "bsarec":  {"learning_rate": [5e-5, 1e-4, 2e-4, 4e-4, 8e-4],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "fearec":  {"learning_rate": [1e-4, 2e-4, 3e-4, 5e-4, 8e-4],
                    "weight_decay":  [0.0, 1e-5, 1e-4]},
        "difsr":   {"learning_rate": [1e-4, 2e-4, 4e-4, 6e-4, 1e-3],
                    "weight_decay":  [0.0, 1e-6, 1e-5, 1e-4]},
        "fame":    {"learning_rate": [5e-5, 1e-4, 2e-4, 4e-4, 8e-4],
                    "weight_decay":  [0.0, 1e-6, 1e-5]},
        "fdsa":    {"learning_rate": [2e-4, 4e-4, 6e-4, 1e-3, 1.5e-3],
                    "weight_decay":  [0.0, 5e-5, 1e-4]},
        "featured_moe_n3": {
            "learning_rate": [2e-4, 4e-4, 8e-4, 1.2e-3, 2e-3],
            "weight_decay":  [0.0, 1e-6, 5e-6, 1e-5],
        },
    },
    # foursquare: anchors from docs/hparams.md (foursquare featured_moe_n3 raw route row)
    "foursquare": {
        "featured_moe_n3": {
            "learning_rate": [2.5e-4, 5e-4, 7.5e-4, 1.2e-3, 2e-3],
            "weight_decay":  [0.0, 5e-7, 1e-6, 5e-6],
        },
    },
}

# ── fixed params (non-tuned) per dataset/model ────────────────────────────────
# Includes arch params (num_layers, n_heads, etc.) that appear in model YAML search spaces.
# These are fixed to best values from sampled-dataset experiments (selected_configs.csv).
# Models not found in selected_configs use reasonable defaults for full-dataset scale.

FIXED_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "KuaiRec": {
        # num_layers=2 default; attn_dropout_prob fixed (in sasrec search space)
        "sasrec":   {"MAX_ITEM_LIST_LENGTH": 30, "hidden_dropout_prob": 0.20, "attn_dropout_prob": 0.20,
                     "num_layers": 2},
        # gru4rec: num_layers=1 (best from sampled KuaiRec)
        "gru4rec":  {"MAX_ITEM_LIST_LENGTH": 10, "dropout_prob": 0.20,
                     "num_layers": 1},
        # tisasrec: n_layers, n_heads, time_span all in search space
        "tisasrec": {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.15,
                     "n_layers": 2, "n_heads": 4, "time_span": 256},
        # duorec: n_layers=3, n_heads=4 (best from sampled KuaiRec)
        "duorec":   {"MAX_ITEM_LIST_LENGTH": 20, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12,
                     "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.0,
                     "n_layers": 3, "n_heads": 4},
        # bsarec: num_layers, num_heads in search space
        "bsarec":   {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "bsarec_alpha": 0.55, "bsarec_c": 3,
                     "num_layers": 2, "num_heads": 4},
        # fearec: n_layers=2, n_heads=4 (best from sampled KuaiRec)
        "fearec":   {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.12,
                     "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.08,
                     "n_layers": 2, "n_heads": 4},
        # difsr: num_layers, num_heads, fusion_type in search space
        "difsr":    {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "attribute_hidden_size": 160, "lambda_attr": 0.1,
                     "num_layers": 2, "num_heads": 4, "fusion_type": "gate"},
        # fame: num_experts=4, num_heads=8, num_layers=3 (best from sampled KuaiRec)
        "fame":     {"MAX_ITEM_LIST_LENGTH": 20, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "num_experts": 4, "num_heads": 8, "num_layers": 3},
        # fdsa: n_layers=2, n_heads=4 (best from sampled KuaiRec)
        "fdsa":     {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "attribute_hidden_size": 160, "lambda_attr": 0.12,
                     "n_layers": 2, "n_heads": 4},
        # featured_moe_n3: no model-level search params; arch fixed via n3 tune config
        "featured_moe_n3": {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10},
    },
    "lastfm": {
        "sasrec":   {"MAX_ITEM_LIST_LENGTH": 30, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "num_layers": 2},
        # gru4rec: num_layers=2 (best from sampled lastfm)
        "gru4rec":  {"MAX_ITEM_LIST_LENGTH": 10, "dropout_prob": 0.15,
                     "num_layers": 2},
        "tisasrec": {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.12,
                     "n_layers": 2, "n_heads": 4, "time_span": 256},
        # duorec: n_layers=1, n_heads=1 (best from sampled lastfm)
        "duorec":   {"MAX_ITEM_LIST_LENGTH": 30, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.12,
                     "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.0,
                     "n_layers": 1, "n_heads": 1},
        "bsarec":   {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "bsarec_alpha": 0.55, "bsarec_c": 3,
                     "num_layers": 2, "num_heads": 4},
        # fearec: n_layers=3, n_heads=2 (best from sampled lastfm)
        "fearec":   {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.15, "attn_dropout_prob": 0.12,
                     "tau": 0.2, "lmd": 0.04, "lmd_sem": 0.08,
                     "n_layers": 3, "n_heads": 2},
        "difsr":    {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10,
                     "attribute_hidden_size": 160, "lambda_attr": 0.1,
                     "num_layers": 2, "num_heads": 4, "fusion_type": "gate"},
        # fame: num_layers=3, num_heads=4, num_experts=4 (best from sampled lastfm, bumped experts to 4)
        "fame":     {"MAX_ITEM_LIST_LENGTH": 50, "hidden_dropout_prob": 0.12, "attn_dropout_prob": 0.10,
                     "num_experts": 4, "num_heads": 4, "num_layers": 3},
        # fdsa: n_layers=3, n_heads=2 (best from sampled lastfm)
        "fdsa":     {"MAX_ITEM_LIST_LENGTH": 30, "hidden_dropout_prob": 0.20, "attn_dropout_prob": 0.15,
                     "attribute_hidden_size": 160, "lambda_attr": 0.12,
                     "n_layers": 3, "n_heads": 2},
        "featured_moe_n3": {"MAX_ITEM_LIST_LENGTH": 10, "hidden_dropout_prob": 0.10, "attn_dropout_prob": 0.10},
    },
    # foursquare: anchors from docs/hparams.md (foursquare raw route row)
    # featured_moe_n3: MAX_ITEM_LIST_LENGTH=10, hidden_size=224, lr=~5e-4, wd=5e-7
    "foursquare": {
        "featured_moe_n3": {
            "MAX_ITEM_LIST_LENGTH": 10,
            "hidden_dropout_prob": 0.16,
            "attn_dropout_prob":   0.10,
            "hidden_size":         224,
            "d_ff":                448,
            "d_expert_hidden":     224,
            "d_router_hidden":     128,
            "d_feat_emb":          16,
            "expert_scale":        4,
            "route_consistency_lambda": 5e-4,
            "stage_feature_dropout_prob": 0.03,
        },
    },
}

# ── result summary fields ──────────────────────────────────────────────────────
SUMMARY_FIELDS = [
    "dataset", "model", "job_id", "gpu_id",
    "status", "valid_mrr20", "test_mrr20",
    "valid_hr10", "test_hr10", "valid_ndcg10", "test_ndcg10",
    "result_path", "log_path", "elapsed_sec", "error", "timestamp_utc",
]

# ── process registry (for Ctrl-C handling) ─────────────────────────────────────
_STOP    = threading.Event()
_PROCS: set[subprocess.Popen[Any]] = set()
_LOCK   = threading.Lock()


def _register(proc: subprocess.Popen[Any]) -> None:
    with _LOCK:
        _PROCS.add(proc)


def _deregister(proc: subprocess.Popen[Any]) -> None:
    with _LOCK:
        _PROCS.discard(proc)


def _kill_all(sig: int = signal.SIGTERM) -> None:
    with _LOCK:
        procs = list(_PROCS)
    for p in procs:
        try:
            if p.poll() is None:
                p.send_signal(sig)
        except Exception:
            pass


def install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:
        print(f"\n[CIKM] caught signal {signum}, stopping...", flush=True)
        _STOP.set()
        _kill_all(signal.SIGTERM)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


# ── helper ─────────────────────────────────────────────────────────────────────
def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hydra_list(values: list[Any]) -> str:
    """Format a list for Hydra CLI: [a,b,c] style."""
    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)
    return "[" + ",".join(_fmt(v) for v in values) + "]"


def _override_int(overrides: list[str], key: str, default: int) -> int:
    prefix = f"{key}="
    for item in reversed(overrides):
        if item.startswith(prefix):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return default
    return default


def sanitize(s: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def read_result(result_path: Path) -> dict[str, Any]:
    if not result_path.exists():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Extract best valid and test metrics from hyperopt result payload."""
    if not payload:
        return {}
    best_valid = payload.get("best_valid_result") or {}
    test       = payload.get("test_result")       or {}
    return {
        "valid_mrr20":  float(best_valid.get("mrr@20", 0.0)),
        "test_mrr20":   float(test.get("mrr@20", 0.0)),
        "valid_hr10":   float(best_valid.get("hit@10", 0.0)),
        "test_hr10":    float(test.get("hit@10", 0.0)),
        "valid_ndcg10": float(best_valid.get("ndcg@10", 0.0)),
        "test_ndcg10":  float(test.get("ndcg@10", 0.0)),
    }


# ── job builder ────────────────────────────────────────────────────────────────

def build_command(
    *,
    dataset: str,
    model: str,
    gpu_id: str,
    job_id: str,
    run_axis: str = "cikm_main",
    run_phase: str = "P0",
    extra_overrides: list[str] | None = None,
) -> list[str]:
    """Build hyperopt_tune.py command for a single (dataset, model) job."""
    cfg_name  = DATASET_TUNE_CONFIG[dataset]
    search    = NARROW_SEARCH[dataset].get(model, {})
    fixed     = FIXED_PARAMS[dataset].get(model, {})
    effective_overrides = list(extra_overrides or [])
    cli_max_evals = _override_int(effective_overrides, "max_evals", MAX_EVALS)
    cli_epochs = _override_int(effective_overrides, "tune_epochs", TUNE_EPOCHS)
    cli_patience = _override_int(effective_overrides, "tune_patience", TUNE_PATIENCE)

    # Route model uses main config, baselines use tune config
    is_route = model == ROUTE_MODEL
    if is_route:
        model_arg = "featured_moe_n3_tune"
        cfg_arg   = "config"
    else:
        model_arg = model
        cfg_arg   = cfg_name

    cmd: list[str] = [
        PYTHON_BIN,
        "hyperopt_tune.py",
        "--config-name", cfg_arg,
        "--search-algo", SEARCH_ALGO,
        "--max-evals",   str(cli_max_evals),
        "--tune-epochs", str(cli_epochs),
        "--tune-patience", str(cli_patience),
        "--seed",        "42",
        "--run-group",   "cikm",
        "--run-axis",    run_axis,
        "--run-phase",   run_phase,
        f"model={model_arg}",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=final",
        f"++dataset_root={DATASET_ROOT}",
        f"++data_path={FINAL_DATA_ROOT}",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        f"++seed=42",
    ]

    if dataset == "lastfm":
        cmd += [
            "++eval_sampling.mode=auto",
            "++eval_sampling.auto_full_threshold=100000",
            "++eval_sampling.sample_num=1000",
        ]
    else:
        cmd += [
            "++eval_sampling.mode=full",
            "++eval_sampling.auto_full_threshold=999999999",
        ]

    # Route model: override config to use correct dataset/feature settings
    if is_route:
        cmd += [
            f"++dataset={dataset}",
        ]

    # Fixed non-tuned params.
    # For each param, also fix the hyperopt search space to a singleton so that
    # params defined as lists in model YAMLs (e.g. search.num_layers: [1,2,3])
    # are not re-sampled.  Hydra ++ override collapses the search-space list to
    # a scalar, which hyperopt_tune.py treats as "Fixed params(singleton/non-list)".
    _SKIP_SEARCH_FIX = {"MAX_ITEM_LIST_LENGTH"}  # pure config overrides, not in model search
    for k, v in fixed.items():
        if isinstance(v, str):
            cmd.append(f"++{k}={v}")
            if k not in _SKIP_SEARCH_FIX:
                cmd.append(f"++search.{k}={v}")
        elif isinstance(v, bool):
            bval = "true" if v else "false"
            cmd.append(f"++{k}={bval}")
            if k not in _SKIP_SEARCH_FIX:
                cmd.append(f"++search.{k}={bval}")
        else:
            cmd.append(f"++{k}={v}")
            if k not in _SKIP_SEARCH_FIX:
                cmd.append(f"++search.{k}={v}")

    # Narrow search space
    for param, values in search.items():
        cmd.append(f"++search.{param}={hydra_list(values)}")
        # type overrides: lr → loguniform, wd → loguniform_zero
        if param == "learning_rate":
            cmd.append(f"++search_space_type_overrides.learning_rate=loguniform")
        elif param == "weight_decay":
            cmd.append(f"++search_space_type_overrides.weight_decay=loguniform_zero")

    if effective_overrides:
        cmd.extend(effective_overrides)

    return cmd


# ── single job runner ──────────────────────────────────────────────────────────

def run_job(
    *,
    dataset: str,
    model: str,
    gpu_id: str,
    summary_path: Path,
    run_axis: str = "cikm_main",
    run_phase: str = "P0",
    extra_overrides: list[str] | None = None,
) -> dict[str, Any]:
    job_id = f"{dataset}_{model}"
    log_dir = LOG_ROOT / run_axis
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sanitize(job_id)}.log"

    cmd = build_command(
        dataset=dataset, model=model, gpu_id=gpu_id,
        job_id=job_id, run_axis=run_axis, run_phase=run_phase,
        extra_overrides=extra_overrides,
    )

    print(f"[START] {dataset} / {model}  gpu={gpu_id}", flush=True)
    print(f"  cmd: {' '.join(cmd[:8])} ...", flush=True)
    start = time.time()

    result_path_str = ""
    error_msg = ""
    status = "error"

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# job_id={job_id}  dataset={dataset}  model={model}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc = subprocess.Popen(
            cmd, cwd=str(EXP_DIR),
            stdout=fh, stderr=subprocess.STDOUT, text=True, env=env,
        )
        _register(proc)
        try:
            while True:
                if _STOP.is_set() and proc.poll() is None:
                    proc.terminate()
                    time.sleep(1)
                    if proc.poll() is None:
                        proc.kill()
                rc = proc.poll()
                if rc is not None:
                    break
                time.sleep(5)
        finally:
            _deregister(proc)

    elapsed = time.time() - start

    # parse result path from log
    # hyperopt_tune.py prints:  "  Results -> /path/to/result.json"
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        import re
        patterns = [
            r"Results\s*->\s*(.+\.json)",           # "  Results -> /path/result.json"
            r"result saved to[:\s]+(.+\.json)",      # legacy
            r"\"result_path\":\s*\"([^\"]+\.json)\"",# JSON inline
        ]
        for pat in patterns:
            m = re.search(pat, log_text, re.IGNORECASE)
            if m:
                result_path_str = m.group(1).strip()
                break
    except Exception:
        pass

    payload = read_result(Path(result_path_str)) if result_path_str else {}
    metrics = parse_metrics(payload)

    if metrics.get("test_mrr20", 0) > 0:
        status = "ok"
    elif proc.returncode == 0:
        status = "completed"
    else:
        error_msg = f"rc={proc.returncode}"

    row = {
        "dataset": dataset,
        "model": model,
        "job_id": job_id,
        "gpu_id": str(gpu_id),
        "status": status,
        **metrics,
        "result_path": result_path_str,
        "log_path": str(log_path),
        "elapsed_sec": round(elapsed, 1),
        "error": error_msg,
        "timestamp_utc": now_utc(),
    }
    append_csv(summary_path, SUMMARY_FIELDS, row)

    mrr = metrics.get("test_mrr20", 0)
    print(f"[DONE]  {dataset} / {model}  test_mrr20={mrr:.4f}  elapsed={elapsed/60:.1f}min", flush=True)
    return row


# ── multi-GPU queue runner ─────────────────────────────────────────────────────

def run_jobs_queued(
    jobs: list[dict[str, Any]],
    *,
    gpus: list[str],
    summary_path: Path,
    run_axis: str = "cikm_main",
    run_phase: str = "P0",
) -> None:
    """Run jobs using a GPU queue (parallel if multiple GPUs, sequential if 1)."""
    install_signal_handlers()

    pending: Queue[dict[str, Any]] = Queue()
    for j in jobs:
        pending.put(j)

    gpu_q: Queue[str] = Queue()
    for g in gpus:
        gpu_q.put(str(g))

    def _worker() -> None:
        while not _STOP.is_set():
            try:
                job = pending.get_nowait()
            except Empty:
                return
            gpu = gpu_q.get()
            try:
                run_job(
                    dataset=job["dataset"], model=job["model"],
                    gpu_id=gpu, summary_path=summary_path,
                    run_axis=run_axis, run_phase=run_phase,
                    extra_overrides=job.get("extra_overrides"),
                )
            finally:
                gpu_q.put(gpu)
                pending.task_done()

    threads = [threading.Thread(target=_worker, daemon=True) for _ in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_jobs_resource_aware(
    jobs: list[dict[str, Any]],
    *,
    gpus: list[str],
    summary_path: Path,
    run_axis: str = "cikm_main",
    run_phase: str = "P0",
    dataset_parallel_limits: dict[str, int] | None = None,
) -> None:
    """Run jobs with GPU reuse plus optional per-dataset concurrency caps."""
    install_signal_handlers()
    limits = {str(k): max(1, int(v)) for k, v in (dataset_parallel_limits or {}).items()}
    pending = list(jobs)
    available_gpus = [str(g) for g in gpus]
    active_by_dataset: dict[str, int] = {}
    active_threads: list[threading.Thread] = []
    active_total = 0
    cond = threading.Condition()

    def _can_launch(job: dict[str, Any]) -> bool:
        dataset = str(job["dataset"])
        limit = limits.get(dataset)
        if limit is None:
            return True
        return active_by_dataset.get(dataset, 0) < limit

    def _launch(job: dict[str, Any], gpu: str) -> None:
        nonlocal active_total
        dataset = str(job["dataset"])
        try:
            run_job(
                dataset=dataset,
                model=job["model"],
                gpu_id=gpu,
                summary_path=summary_path,
                run_axis=run_axis,
                run_phase=run_phase,
                extra_overrides=job.get("extra_overrides"),
            )
        finally:
            with cond:
                active_by_dataset[dataset] = max(0, active_by_dataset.get(dataset, 1) - 1)
                active_total -= 1
                available_gpus.append(gpu)
                cond.notify_all()

    with cond:
        while (pending or active_total > 0) and not _STOP.is_set():
            launched = False
            while available_gpus and pending and not _STOP.is_set():
                pick_idx = next((idx for idx, job in enumerate(pending) if _can_launch(job)), None)
                if pick_idx is None:
                    break
                job = pending.pop(pick_idx)
                gpu = available_gpus.pop(0)
                dataset = str(job["dataset"])
                active_by_dataset[dataset] = active_by_dataset.get(dataset, 0) + 1
                active_total += 1
                t = threading.Thread(target=_launch, args=(job, gpu), daemon=False)
                active_threads.append(t)
                t.start()
                launched = True
            if not launched:
                cond.wait(timeout=5)

    for t in active_threads:
        t.join()
