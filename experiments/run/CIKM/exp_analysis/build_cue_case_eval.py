#!/usr/bin/env python3
"""Build cue-tier case-eval splits for full KuaiRec and lastfm (final_dataset).

Splits sessions into 8 behavioral tier groups:
  memory_plus/minus   – high/low item-repetition and familiarity
  focus_plus/minus    – narrow/broad category focus
  tempo_plus/minus    – dense/sparse interaction pace
  exposure_plus/minus – high/low popularity preference

Output: Datasets/processed/cikm_case_eval/pure/{tier}/{dataset}/
  {dataset}.train.inter  (symlink to full train)
  {dataset}.valid.inter  (symlink to full valid)
  {dataset}.test.inter   (filtered test sessions for this tier)
  {dataset}.item         (symlink to full item file)

Usage:
    python build_cue_case_eval.py [--datasets KuaiRec lastfm]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
FINAL_DATASET_ROOT = REPO_ROOT / "Datasets" / "processed" / "final_dataset"
CASE_EVAL_ROOT     = REPO_ROOT / "Datasets" / "processed" / "cikm_case_eval" / "pure"

DATASETS = ["KuaiRec", "lastfm"]

FAMILIES = ["memory", "focus", "tempo", "exposure"]
GROUPS = [
    "memory_plus",  "memory_minus",
    "focus_plus",   "focus_minus",
    "tempo_plus",   "tempo_minus",
    "exposure_plus","exposure_minus",
]

# Tier thresholds: core_q = quantile threshold for "extreme" session score
# contam_max_q = max allowed contamination from other families (for "pure" tier)
PURE_CONFIG = {
    "memory_plus":   {"core_q": 0.90, "contam_max_q": 0.85},
    "memory_minus":  {"core_q": 0.90, "contam_max_q": 0.90},
    "focus_plus":    {"core_q": 0.90, "contam_max_q": 0.90},
    "focus_minus":   {"core_q": 0.90, "contam_max_q": 0.90},
    "tempo_plus":    {"core_q": 0.90, "contam_max_q": 0.90},
    "tempo_minus":   {"core_q": 0.90, "contam_max_q": 0.90},
    "exposure_plus": {"core_q": 0.88, "contam_max_q": 0.92},
    "exposure_minus":{"core_q": 0.88, "contam_max_q": 0.92},
}


def _clip01(values: pd.Series) -> np.ndarray:
    return np.clip(values.to_numpy(dtype=float), 0.0, 1.0)


def _safe_mean(*parts) -> np.ndarray:
    return np.vstack([_clip01(p) for p in parts]).mean(axis=0)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def compute_family_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute memory/focus/tempo/exposure plus-scores per row from 64-feature inter file."""
    memory_plus = _safe_mean(
        df["mid_repeat_r:float"],
        1.0 - df["mid_item_uniq_r:float"],
        df["mid_max_run_i:float"],
        df["mic_suffix_recons_r:float"],
        df["mic_is_recons:float"],
    )
    focus_plus = _safe_mean(
        df["mid_cat_top1:float"],
        1.0 - df["mid_cat_ent:float"],
        1.0 - df["mid_cat_switch_r:float"],
        1.0 - df["mic_suffix_cat_ent:float"],
        1.0 - df["mic_last_cat_mismatch_r:float"],
    )
    tempo_plus = _safe_mean(
        1.0 - df["mid_int_mean:float"],
        1.0 - df["mic_gap_mean:float"],
        1.0 - df["mic_last_gap:float"],
    )
    exposure_plus = _safe_mean(
        df["mid_pop_mean:float"],
        df["mic_last_pop:float"],
        1.0 - df["mid_pop_std:float"],
    )
    return pd.DataFrame({
        "session_id": df["session_id:token"].astype(str),
        "item_id":    df["item_id:token"].astype(str),
        "timestamp":  df["timestamp:float"].astype(float),
        "memory_plus":   memory_plus,
        "focus_plus":    focus_plus,
        "tempo_plus":    tempo_plus,
        "exposure_plus": exposure_plus,
    })


def build_tier_mask(frame: pd.DataFrame, group_name: str) -> np.ndarray:
    """Return boolean mask for sessions belonging to this tier group."""
    parts = group_name.split("_")
    family, polarity = "_".join(parts[:-1]), parts[-1]
    family_vals = frame[f"{family}_plus"].to_numpy()
    core = family_vals if polarity == "plus" else (1.0 - family_vals)

    # Contamination: how extreme are the OTHER families (should be low for "pure" tier)
    contam_parts = []
    for other in FAMILIES:
        if other == family:
            continue
        other_plus = frame[f"{other}_plus"].to_numpy()
        contam_parts.append(2.0 * np.abs(other_plus - 0.5))
    contamination = np.vstack(contam_parts).max(axis=0)

    cfg = PURE_CONFIG[group_name]
    core_cut   = np.quantile(core, float(cfg["core_q"]))
    contam_cut = np.quantile(contamination, float(cfg["contam_max_q"]))
    return (core >= core_cut) & (contamination <= contam_cut)


def process_dataset(dataset: str) -> None:
    src_dir = FINAL_DATASET_ROOT / dataset
    if not src_dir.exists():
        print(f"  [SKIP] {dataset}: final_dataset not found at {src_dir}")
        return

    print(f"  Loading test.inter for {dataset}...", flush=True)
    test_inter = src_dir / f"{dataset}.test.inter"
    df = pd.read_csv(test_inter, sep="\t", low_memory=False)
    df["session_id:token"] = df["session_id:token"].astype(str)
    df["item_id:token"]    = df["item_id:token"].astype(str)
    df["timestamp:float"]  = df["timestamp:float"].astype(float)

    # Compute family scores per row
    frame = compute_family_scores(df)
    # Aggregate to session level (take mean per session for tier assignment)
    sess_scores = frame.groupby("session_id")[
        ["memory_plus", "focus_plus", "tempo_plus", "exposure_plus"]
    ].mean().reset_index()
    # Attach session scores back to frame for mask computation
    frame = frame.merge(
        sess_scores.rename(columns={f: f"{f}_sess" for f in ["memory_plus","focus_plus","tempo_plus","exposure_plus"]}),
        on="session_id", how="left"
    )
    # Use session-level scores for tier assignment
    for f in FAMILIES:
        frame[f"{f}_plus"] = frame[f"{f}_plus_sess"]

    print(f"  n_sessions={frame['session_id'].nunique()} n_rows={len(df)}", flush=True)

    for group_name in GROUPS:
        mask = build_tier_mask(frame, group_name)
        tier_sessions = set(frame.loc[mask, "session_id"].unique())
        n_sessions = len(tier_sessions)

        out_dir = CASE_EVAL_ROOT / group_name / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        # Symlinks for train / valid / item files
        for fname in [f"{dataset}.train.inter", f"{dataset}.valid.inter", f"{dataset}.item"]:
            src = src_dir / fname
            if src.exists():
                _symlink(src, out_dir / fname)

        # Filter test.inter to tier sessions
        filtered = df[df["session_id:token"].isin(tier_sessions)]
        out_test = out_dir / f"{dataset}.test.inter"
        filtered.to_csv(out_test, sep="\t", index=False)

        pct = 100.0 * n_sessions / frame["session_id"].nunique()
        print(f"    {group_name:20s}: {n_sessions:5d} sessions ({pct:.1f}%)  → {out_test}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build cue-tier case eval splits")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args()

    CASE_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {CASE_EVAL_ROOT}", flush=True)

    for dataset in args.datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"Building cue-tier splits for {dataset}", flush=True)
        process_dataset(dataset)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
