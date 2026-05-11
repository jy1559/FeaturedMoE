#!/usr/bin/env python3
"""Run SASRec once per dataset with best P0 hparams + checkpoint export.

Produces one checkpoint per dataset (KuaiRec + lastfm) for cue-tier baseline
comparison.  Reads best lr/wd from P0 main_baselines_summary.csv;
falls back to FIXED narrow-range midpoint if P0 is not yet done.

Checkpoints land in:
  experiments/run/artifacts/results/cikm/cue_tier_baseline/
    KuaiRec_SASRec_cue_baseline_<ts>_best_model_state.pth
    lastfm_SASRec_cue_baseline_<ts>_best_model_state.pth

Usage:
    python run_sasrec_cue_baseline.py [--gpu 0] [--datasets KuaiRec lastfm]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

_CIKM_DIR = Path(__file__).resolve().parents[1]
_EXP_DIR  = _CIKM_DIR.parent.parent
if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))

from common import (  # noqa: E402
    DATASETS, FIXED_PARAMS, NARROW_SEARCH,
    PYTHON_BIN, RESULT_ROOT, TUNE_EPOCHS, TUNE_PATIENCE,
)

LOG_DIR = _CIKM_DIR / "logs" / "cue_baseline"

# Fallback lr/wd if P0 results not available (midpoints of narrow ranges)
FALLBACK_LR = {"KuaiRec": 1.2e-3, "lastfm": 5e-4}
FALLBACK_WD = {"KuaiRec": 0.0,    "lastfm": 0.0}


def _best_sasrec_hparams(dataset: str) -> tuple[float, float]:
    """Return (lr, wd) from P0 best result, or fallback."""
    summary = RESULT_ROOT / "main_baselines_summary.csv"
    if summary.exists():
        for row in csv.DictReader(summary.open()):
            if row.get("dataset") == dataset and row.get("model") == "sasrec":
                rpath = row.get("result_path", "")
                if rpath and Path(rpath).exists():
                    try:
                        data = json.loads(Path(rpath).read_text())
                        bp = data.get("best_params", {}) or {}
                        lr = float(bp.get("learning_rate", 0) or 0)
                        wd = float(bp.get("weight_decay", 0) or 0)
                        if lr > 0:
                            return lr, wd
                    except Exception:
                        pass
    return FALLBACK_LR[dataset], FALLBACK_WD[dataset]


def run_dataset(dataset: str, gpu_id: str) -> None:
    lr, wd = _best_sasrec_hparams(dataset)
    fixed = FIXED_PARAMS[dataset]["sasrec"]

    print(f"\n[cue_baseline] SASRec / {dataset}  lr={lr:.2e} wd={wd:.2e}  gpu={gpu_id}", flush=True)

    cfg_name = "tune_kuai_cikm" if dataset == "KuaiRec" else "tune_lfm_cikm"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sasrec_{dataset}.log"

    cmd = [
        PYTHON_BIN, "hyperopt_tune.py",
        "--config-name", cfg_name,
        "--search-algo", "tpe",
        "--max-evals", "1",        # single run with fixed hparams
        "--tune-epochs", str(TUNE_EPOCHS),
        "--tune-patience", str(TUNE_PATIENCE),
        "--seed", "42",
        "--run-group", "cikm",
        "--run-axis", "cikm_cue_baseline",
        "--run-phase", "P1-cue",
        "model=sasrec",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=final",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++exclude_unseen_target_from_main_eval=true",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        "++seed=42",
        # Fix lr/wd to P0 best (singleton search → treated as fixed by hyperopt)
        f"++search.learning_rate={lr}",
        f"++search.weight_decay={wd}",
        # Export checkpoint so cue_tier_eval.py can load it
        "++artifact_export_all_checkpoints=true",
    ]

    if dataset == "lastfm":
        cmd += [
            "++eval_sampling.mode=auto",
            "++eval_sampling.auto_full_threshold=100000",
            "++eval_sampling.sample_num=1000",
        ]

    for k, v in fixed.items():
        cmd.append(f"++{k}={v}")
        if k not in ("MAX_ITEM_LIST_LENGTH",):
            cmd.append(f"++search.{k}={v}")

    print(f"  log → {log_path}", flush=True)
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(_EXP_DIR), stdout=fh, stderr=subprocess.STDOUT, text=True)

    if proc.returncode == 0:
        print(f"  [DONE] SASRec/{dataset} finished OK", flush=True)
    else:
        print(f"  [ERROR] SASRec/{dataset} rc={proc.returncode}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run SASRec with checkpoint for cue-tier eval")
    p.add_argument("--gpu", default="0")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = p.parse_args()

    for ds in args.datasets:
        run_dataset(ds, args.gpu)

    print("\n[cue_baseline] Done. Run cue_tier_eval.py to evaluate on tier splits.", flush=True)


if __name__ == "__main__":
    main()
