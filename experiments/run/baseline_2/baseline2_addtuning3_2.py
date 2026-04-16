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
AXIS = "PAIR60_ADDTUNING3_2"

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
    ARTIFACT_ROOT / "logs" / TRACK / "PAIR60_ADDTUNING3" / "summary.csv",
]

TARGETS = [
    ("beauty", "bsarec", "critical"),
    ("beauty", "fame", "critical"),
    ("beauty", "difsr", "critical"),
    ("beauty", "gru4rec", "critical"),
]

PRIORITY_RANK = {"critical": 0}

MODEL_COMBO_BUDGET = {
    "bsarec": 40,
    "fame": 40,
    "difsr": 40,
    "gru4rec": 40,
}

MODEL_APPEND_BUDGET = {
    "bsarec": 16,
    "fame": 16,
    "difsr": 16,
    "gru4rec": 16,
}

MODEL_MAX_EVALS = {
    "bsarec": 11,
    "fame": 10,
    "difsr": 11,
    "gru4rec": 12,
}

DIM_GRID = [48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
SEQ_GRID = [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50]

GENERIC_FAMILIES = [
    {"name": "tiny_ultrashort", "anchor": "test", "scale": 0.45, "max_len": 8, "layers": 1, "dropout_delta": 0.06, "wd_mult": 2.3, "lr_mode": "reset_low", "seq_mode": "ultrashort", "head_mode": "narrow"},
    {"name": "tiny_short", "anchor": "test", "scale": 0.55, "max_len": 10, "layers": 1, "dropout_delta": 0.05, "wd_mult": 2.0, "lr_mode": "reset_low", "seq_mode": "short", "head_mode": "narrow"},
    {"name": "small_short", "anchor": "balanced", "scale": 0.7, "max_len": 10, "layers": 1, "dropout_delta": 0.04, "wd_mult": 1.7, "lr_mode": "wide_mid", "seq_mode": "short", "head_mode": "narrow"},
    {"name": "small_mid", "anchor": "balanced", "scale": 0.75, "max_len": 14, "layers": 2, "dropout_delta": 0.02, "wd_mult": 1.4, "lr_mode": "wide_mid", "seq_mode": "mid", "head_mode": "narrow"},
    {"name": "base_short", "anchor": "test", "scale": 1.0, "max_len": 10, "layers": 1, "dropout_delta": 0.02, "wd_mult": 1.3, "lr_mode": "wide_mid", "seq_mode": "short", "head_mode": "base"},
    {"name": "base_mid", "anchor": "balanced", "scale": 1.0, "max_len": 14, "layers": 2, "dropout_delta": 0.0, "wd_mult": 1.0, "lr_mode": "wide_mid", "seq_mode": "mid", "head_mode": "base"},
    {"name": "base_long", "anchor": "valid", "scale": 1.0, "max_len": 20, "layers": 2, "dropout_delta": -0.01, "wd_mult": 0.9, "lr_mode": "wide_high", "seq_mode": "long", "head_mode": "base"},
    {"name": "wide_short", "anchor": "balanced", "scale": 1.2, "max_len": 10, "layers": 2, "dropout_delta": 0.01, "wd_mult": 1.1, "lr_mode": "wide_high", "seq_mode": "short", "head_mode": "wide"},
    {"name": "wide_mid", "anchor": "balanced", "scale": 1.25, "max_len": 16, "layers": 2, "dropout_delta": -0.01, "wd_mult": 0.85, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide"},
    {"name": "wide_long", "anchor": "valid", "scale": 1.25, "max_len": 24, "layers": 2, "dropout_delta": -0.02, "wd_mult": 0.8, "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide"},
    {"name": "huge_mid", "anchor": "valid", "scale": 1.45, "max_len": 16, "layers": 3, "dropout_delta": -0.02, "wd_mult": 0.75, "lr_mode": "ultrawide", "seq_mode": "mid", "head_mode": "wide"},
    {"name": "huge_long", "anchor": "valid", "scale": 1.55, "max_len": 24, "layers": 3, "dropout_delta": -0.03, "wd_mult": 0.7, "lr_mode": "ultrawide", "seq_mode": "long", "head_mode": "wide"},
    {"name": "deep_narrow", "anchor": "test", "scale": 0.85, "max_len": 18, "layers": 3, "dropout_delta": 0.03, "wd_mult": 1.4, "lr_mode": "wide_mid", "seq_mode": "mid", "head_mode": "narrow"},
    {"name": "deep_wide", "anchor": "valid", "scale": 1.3, "max_len": 20, "layers": 4, "dropout_delta": 0.0, "wd_mult": 1.0, "lr_mode": "wide_high", "seq_mode": "long", "head_mode": "wide"},
    {"name": "shallow_long", "anchor": "balanced", "scale": 1.05, "max_len": 24, "layers": 1, "dropout_delta": 0.02, "wd_mult": 1.25, "lr_mode": "wide_mid", "seq_mode": "long", "head_mode": "base"},
    {"name": "reg_heavy", "anchor": "test", "scale": 1.0, "max_len": 14, "layers": 2, "dropout_delta": 0.08, "wd_mult": 3.0, "lr_mode": "reset_low", "seq_mode": "mid", "head_mode": "base"},
    {"name": "reg_light", "anchor": "valid", "scale": 1.1, "max_len": 16, "layers": 2, "dropout_delta": -0.05, "wd_mult": 0.45, "lr_mode": "wide_high", "seq_mode": "mid", "head_mode": "wide"},
]

BSAREC_SPECIFIC = [
    {"name": "alpha_lo_c8", "anchor": "test", "scale": 0.75, "max_len": 12, "layers": 1, "profile": "alpha_lo_c8", "lr_mode": "reset_low", "seq_mode": "short"},
    {"name": "alpha_lo_c5", "anchor": "balanced", "scale": 0.9, "max_len": 14, "layers": 2, "profile": "alpha_lo_c5", "lr_mode": "wide_mid", "seq_mode": "mid"},
    {"name": "alpha_mid_c3", "anchor": "balanced", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "alpha_mid_c3", "lr_mode": "wide_mid", "seq_mode": "mid"},
    {"name": "alpha_hi_c2", "anchor": "valid", "scale": 1.15, "max_len": 16, "layers": 3, "profile": "alpha_hi_c2", "lr_mode": "wide_high", "seq_mode": "mid"},
    {"name": "alpha_peak_c1", "anchor": "valid", "scale": 1.25, "max_len": 20, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "ultrawide", "seq_mode": "long"},
    {"name": "alpha_flat_c8", "anchor": "test", "scale": 1.0, "max_len": 20, "layers": 1, "profile": "alpha_flat_c8", "lr_mode": "reset_low", "seq_mode": "long"},
    {"name": "alpha_hi_tiny", "anchor": "balanced", "scale": 0.55, "max_len": 10, "layers": 1, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "short"},
    {"name": "alpha_hi_c2_len50", "anchor": "valid", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "doc_long", "dropout_delta": 0.02, "wd_mult": 1.4},
    {"name": "alpha_peak_c1_len50", "anchor": "valid", "scale": 1.15, "max_len": 50, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.01, "wd_mult": 1.2},
]

FAME_SPECIFIC = [
    {"name": "experts2_short", "anchor": "test", "scale": 0.75, "max_len": 10, "layers": 1, "profile": "experts2", "lr_mode": "reset_low", "seq_mode": "short"},
    {"name": "experts2_long", "anchor": "balanced", "scale": 0.95, "max_len": 20, "layers": 1, "profile": "experts2", "lr_mode": "wide_mid", "seq_mode": "long"},
    {"name": "experts3_mid", "anchor": "balanced", "scale": 1.0, "max_len": 14, "layers": 2, "profile": "experts3", "lr_mode": "wide_mid", "seq_mode": "mid"},
    {"name": "experts4_mid", "anchor": "balanced", "scale": 1.1, "max_len": 16, "layers": 2, "profile": "experts4", "lr_mode": "wide_high", "seq_mode": "mid"},
    {"name": "experts6_mid", "anchor": "valid", "scale": 1.2, "max_len": 16, "layers": 2, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "mid"},
    {"name": "experts6_long", "anchor": "valid", "scale": 1.3, "max_len": 22, "layers": 3, "profile": "experts6", "lr_mode": "ultrawide", "seq_mode": "long"},
    {"name": "experts4_shallow", "anchor": "test", "scale": 1.0, "max_len": 12, "layers": 1, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "short"},
    {"name": "experts4_len50", "anchor": "valid", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "doc_long", "dropout_delta": 0.02, "wd_mult": 1.3},
    {"name": "experts6_len50", "anchor": "valid", "scale": 1.1, "max_len": 50, "layers": 2, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.01, "wd_mult": 1.15},
]

DIFSR_SPECIFIC = [
    {"name": "gate_midattr_short", "anchor": "balanced", "scale": 0.8, "max_len": 14, "layers": 1, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.06, "wd_mult": 2.0, "head_mode": "narrow"},
    {"name": "gate_midattr_mid", "anchor": "balanced", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.04, "wd_mult": 1.8},
    {"name": "gate_highattr_short", "anchor": "valid", "scale": 0.85, "max_len": 16, "layers": 1, "profile": "gate_highattr", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": 0.07, "wd_mult": 2.6},
    {"name": "gate_highattr_mid", "anchor": "valid", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gate_highattr", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": 0.05, "wd_mult": 2.2},
    {"name": "gate_highattr_deep", "anchor": "valid", "scale": 1.1, "max_len": 18, "layers": 3, "profile": "gate_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.03, "wd_mult": 1.7},
    {"name": "concat_midattr_short", "anchor": "balanced", "scale": 0.85, "max_len": 14, "layers": 1, "profile": "concat_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.04, "wd_mult": 1.8},
    {"name": "concat_highattr_mid", "anchor": "valid", "scale": 1.05, "max_len": 16, "layers": 2, "profile": "concat_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.04, "wd_mult": 1.6},
    {"name": "concat_highattr_wide", "anchor": "valid", "scale": 1.2, "max_len": 18, "layers": 3, "profile": "concat_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.02, "wd_mult": 1.4},
    {"name": "gate_zeroattr_probe", "anchor": "test", "scale": 0.75, "max_len": 12, "layers": 1, "profile": "gate_zeroattr", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.06, "wd_mult": 2.4},
    {"name": "gate_highattr_len50", "anchor": "valid", "scale": 0.95, "max_len": 50, "layers": 2, "profile": "gate_highattr", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.04, "wd_mult": 1.9},
    {"name": "gate_midattr_len50", "anchor": "balanced", "scale": 0.85, "max_len": 50, "layers": 1, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.06, "wd_mult": 2.2, "head_mode": "narrow"},
    {"name": "concat_highattr_len50", "anchor": "valid", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "concat_highattr", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.03, "wd_mult": 1.6},
]

GRU4REC_SPECIFIC = [
    {"name": "gru_tiny_1l", "anchor": "test", "scale": 0.45, "max_len": 8, "layers": 1, "profile": "gru_reset", "lr_mode": "reset_low", "seq_mode": "ultrashort"},
    {"name": "gru_small_2l", "anchor": "balanced", "scale": 0.7, "max_len": 12, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "short"},
    {"name": "gru_base_3l", "anchor": "balanced", "scale": 1.0, "max_len": 16, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid"},
    {"name": "gru_wide_3l", "anchor": "valid", "scale": 1.25, "max_len": 16, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid"},
    {"name": "gru_huge_4l", "anchor": "valid", "scale": 1.55, "max_len": 20, "layers": 4, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "long"},
    {"name": "gru_long_1l", "anchor": "test", "scale": 0.95, "max_len": 24, "layers": 1, "profile": "gru_reset", "lr_mode": "reset_low", "seq_mode": "long"},
    {"name": "gru_long_2l", "anchor": "balanced", "scale": 1.0, "max_len": 28, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "long"},
    {"name": "gru_len50_1l", "anchor": "balanced", "scale": 0.95, "max_len": 50, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "doc_long"},
    {"name": "gru_len50_2l", "anchor": "valid", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_high", "seq_mode": "doc_long"},
]

BSAREC_APPEND = [
    {"name": "stable_alpha_hi_short_refresh", "anchor": "best_test", "scale": 0.95, "max_len": 10, "layers": 1, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "stable_alpha_hi_mid_refresh", "anchor": "balanced_best", "scale": 1.0, "max_len": 14, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_alpha_mid_mid_refresh", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "alpha_mid_c3", "lr_mode": "wide_mid", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.05},
    {"name": "stable_alpha_lo_mid_refresh", "anchor": "mid_best", "scale": 0.95, "max_len": 14, "layers": 2, "profile": "alpha_lo_c5", "lr_mode": "wide_mid", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "stable_alpha_hi_short_wide", "anchor": "test_top2", "scale": 1.15, "max_len": 10, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_high", "seq_mode": "short", "dropout_delta": -0.01, "wd_mult": 0.9},
    {"name": "stable_alpha_hi_base_short", "anchor": "short_best", "scale": 1.0, "max_len": 12, "layers": 1, "profile": "alpha_hi_c2", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.0, "wd_mult": 0.95},
    {"name": "stable_alpha_mid_long_shallow", "anchor": "long_best", "scale": 1.0, "max_len": 24, "layers": 1, "profile": "alpha_mid_c3", "lr_mode": "wide_mid", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.05},
    {"name": "stable_alpha_hi_large_mid", "anchor": "test_top3", "scale": 1.2, "max_len": 16, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.9},
    {"name": "aggr_alpha_peak_long", "anchor": "test_top2", "scale": 1.2, "max_len": 20, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_alpha_peak_len50", "anchor": "valid_best", "scale": 1.1, "max_len": 50, "layers": 3, "profile": "alpha_peak_c1", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.01, "wd_mult": 1.15},
    {"name": "aggr_alpha_hi_huge_mid", "anchor": "test_top2", "scale": 1.45, "max_len": 18, "layers": 2, "profile": "alpha_hi_c2", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.85},
    {"name": "aggr_alpha_flat_c8_long", "anchor": "long_best", "scale": 1.05, "max_len": 20, "layers": 1, "profile": "alpha_flat_c8", "lr_mode": "reset_low", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_alpha_lo_c8_short", "anchor": "short_best", "scale": 0.85, "max_len": 10, "layers": 1, "profile": "alpha_lo_c8", "lr_mode": "reset_low", "seq_mode": "short", "dropout_delta": 0.02, "wd_mult": 1.3},
    {"name": "aggr_alpha_mid_len50", "anchor": "valid_best", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "alpha_mid_c3", "lr_mode": "wide_mid", "seq_mode": "doc_long", "dropout_delta": 0.02, "wd_mult": 1.2},
    {"name": "aggr_alpha_hi_deep_short", "anchor": "best_test", "scale": 1.05, "max_len": 12, "layers": 3, "profile": "alpha_hi_c2", "lr_mode": "wide_high", "seq_mode": "short", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_alpha_peak_wide_short", "anchor": "test_top3", "scale": 1.3, "max_len": 10, "layers": 2, "profile": "alpha_peak_c1", "lr_mode": "ultrawide", "seq_mode": "short", "dropout_delta": -0.01, "wd_mult": 0.9},
]

FAME_APPEND = [
    {"name": "stable_experts4_long_tilt", "anchor": "best_test", "scale": 1.08, "max_len": 22, "layers": 2, "profile": "experts4", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.01, "wd_mult": 1.05},
    {"name": "stable_experts4_mid_refresh", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 0.9},
    {"name": "stable_experts3_short_refresh", "anchor": "test_top2", "scale": 0.85, "max_len": 14, "layers": 1, "profile": "experts3", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.03, "wd_mult": 1.2},
    {"name": "stable_experts3_base_short", "anchor": "short_best", "scale": 0.8, "max_len": 8, "layers": 1, "profile": "experts3", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.01, "wd_mult": 0.95},
    {"name": "stable_experts2_long_refresh", "anchor": "long_best", "scale": 0.9, "max_len": 20, "layers": 1, "profile": "experts2", "lr_mode": "wide_mid", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 0.95},
    {"name": "stable_experts4_shallow_short", "anchor": "balanced_best", "scale": 0.95, "max_len": 12, "layers": 1, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.0, "wd_mult": 0.9},
    {"name": "stable_experts4_midwide", "anchor": "test_top3", "scale": 1.1, "max_len": 14, "layers": 2, "profile": "experts4", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 0.95},
    {"name": "stable_experts3_deep_mid", "anchor": "mid_best", "scale": 1.0, "max_len": 18, "layers": 3, "profile": "experts3", "lr_mode": "wide_mid", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.05},
    {"name": "aggr_experts6_mid", "anchor": "valid_best", "scale": 1.15, "max_len": 16, "layers": 2, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "aggr_experts6_long", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "profile": "experts6", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "aggr_experts4_len50", "anchor": "best_test", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "experts4", "lr_mode": "wide_mid", "seq_mode": "doc_long", "dropout_delta": 0.02, "wd_mult": 1.2},
    {"name": "aggr_experts6_len50", "anchor": "balanced_best", "scale": 1.05, "max_len": 50, "layers": 2, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "aggr_experts3_huge_long", "anchor": "best_test", "scale": 1.3, "max_len": 24, "layers": 3, "profile": "experts3", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": -0.01, "wd_mult": 0.85},
    {"name": "aggr_experts4_deep_wide", "anchor": "test_top2", "scale": 1.15, "max_len": 20, "layers": 4, "profile": "experts4", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_experts2_short_compact", "anchor": "short_best", "scale": 0.7, "max_len": 10, "layers": 1, "profile": "experts2", "lr_mode": "reset_low", "seq_mode": "short", "dropout_delta": 0.02, "wd_mult": 1.15},
    {"name": "aggr_experts6_short", "anchor": "mid_best", "scale": 1.0, "max_len": 12, "layers": 2, "profile": "experts6", "lr_mode": "wide_high", "seq_mode": "short", "dropout_delta": 0.01, "wd_mult": 1.05},
]

DIFSR_APPEND = [
    {"name": "stable_gate_mid_short_refresh", "anchor": "best_test", "scale": 1.0, "max_len": 18, "layers": 1, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_gate_mid_14", "anchor": "test_top2", "scale": 0.9, "max_len": 14, "layers": 1, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_concat_high_len50", "anchor": "test_top3", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "concat_highattr", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_gate_mid_len50", "anchor": "test_top2", "scale": 1.0, "max_len": 50, "layers": 1, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_gate_high_mid", "anchor": "valid_best", "scale": 1.0, "max_len": 18, "layers": 2, "profile": "gate_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_gate_mid_long20", "anchor": "balanced_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "gate_midattr", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_concat_mid_mid", "anchor": "mid_best", "scale": 0.95, "max_len": 16, "layers": 2, "profile": "concat_midattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.05},
    {"name": "stable_gate_zero_short", "anchor": "short_best", "scale": 0.85, "max_len": 12, "layers": 1, "profile": "gate_zeroattr", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.01, "wd_mult": 1.15},
    {"name": "aggr_gate_high_len50_big", "anchor": "best_test", "scale": 1.15, "max_len": 50, "layers": 2, "profile": "gate_highattr", "lr_mode": "ultrawide", "seq_mode": "doc_long", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "aggr_concat_high_long24_big", "anchor": "long_best", "scale": 1.15, "max_len": 24, "layers": 2, "profile": "concat_highattr", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": 0.02, "wd_mult": 1.1},
    {"name": "aggr_gate_mid_long32", "anchor": "balanced_best", "scale": 1.1, "max_len": 28, "layers": 1, "profile": "gate_midattr", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_gate_high_short_big", "anchor": "best_test", "scale": 1.25, "max_len": 16, "layers": 2, "profile": "gate_highattr", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.1},
    {"name": "aggr_concat_high_deep", "anchor": "valid_best", "scale": 1.15, "max_len": 20, "layers": 3, "profile": "concat_highattr", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_gate_mid_attrwide", "anchor": "test_top2", "scale": 1.2, "max_len": 18, "layers": 2, "profile": "gate_midattr", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.05},
    {"name": "aggr_concat_mid_short_compact", "anchor": "short_best", "scale": 0.75, "max_len": 12, "layers": 1, "profile": "concat_midattr", "lr_mode": "wide_high", "seq_mode": "short", "dropout_delta": 0.02, "wd_mult": 1.15},
    {"name": "aggr_gate_zero_long_probe", "anchor": "mid_best", "scale": 1.0, "max_len": 20, "layers": 2, "profile": "gate_zeroattr", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
]

GRU4REC_APPEND = [
    {"name": "stable_mid_256_3l_refresh", "anchor": "best_test", "scale": 1.05, "max_len": 18, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.01, "wd_mult": 1.05},
    {"name": "stable_mid_256_4l", "anchor": "test_top2", "scale": 1.0, "max_len": 20, "layers": 4, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_short_160_1l", "anchor": "test_top3", "scale": 1.0, "max_len": 8, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "short", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_mid_224_2l", "anchor": "balanced_best", "scale": 1.0, "max_len": 16, "layers": 2, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_long_192_1l", "anchor": "long_best", "scale": 1.0, "max_len": 24, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_mid_192_3l", "anchor": "mid_best", "scale": 1.0, "max_len": 16, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_mid_224_3l", "anchor": "valid_best", "scale": 1.0, "max_len": 16, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "stable_short_224_2l", "anchor": "short_best", "scale": 1.0, "max_len": 10, "layers": 2, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "short", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_huge_320_3l", "anchor": "best_test", "scale": 1.25, "max_len": 20, "layers": 3, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.9},
    {"name": "aggr_huge_320_4l", "anchor": "test_top2", "scale": 1.25, "max_len": 20, "layers": 4, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "mid", "dropout_delta": -0.01, "wd_mult": 0.9},
    {"name": "aggr_len50_1l", "anchor": "long_best", "scale": 1.0, "max_len": 50, "layers": 1, "profile": "gru_reset", "lr_mode": "wide_mid", "seq_mode": "doc_long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_len50_2l", "anchor": "long_best", "scale": 1.0, "max_len": 50, "layers": 2, "profile": "gru_reset", "lr_mode": "wide_high", "seq_mode": "doc_long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_long_224_3l", "anchor": "valid_best", "scale": 1.0, "max_len": 24, "layers": 3, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "long", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_short_128_1l_probe", "anchor": "short_best", "scale": 0.8, "max_len": 10, "layers": 1, "profile": "gru_reset", "lr_mode": "reset_low", "seq_mode": "short", "dropout_delta": 0.02, "wd_mult": 1.2},
    {"name": "aggr_mid_256_4l", "anchor": "balanced_best", "scale": 1.1, "max_len": 18, "layers": 4, "profile": "gru_deep", "lr_mode": "wide_high", "seq_mode": "mid", "dropout_delta": 0.0, "wd_mult": 1.0},
    {"name": "aggr_long_256_4l_bold", "anchor": "valid_best", "scale": 1.2, "max_len": 28, "layers": 4, "profile": "gru_deep", "lr_mode": "ultrawide", "seq_mode": "long", "dropout_delta": -0.01, "wd_mult": 0.92},
]


def target_specs() -> list[stage1.TargetSpec]:
    return [stage1.TargetSpec(dataset, model, priority) for dataset, model, priority in TARGETS]


def load_summary_rows(summary_paths: list[Path]) -> dict[tuple[str, str], list[stage2.HistSeed]]:
    grouped: dict[tuple[str, str], list[stage2.HistSeed]] = {}
    for summary_path in summary_paths:
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

    for rows in grouped.values():
        rows.sort(key=lambda item: (item.selection_score, item.test_mrr20, item.best_valid_mrr20), reverse=True)
    return grouped


def choose_params_json(row: dict[str, str]) -> str:
    return stage2.choose_params_json(row)


def selection_score(dataset: str, best_valid: float, test_mrr20: float, seen_count: int) -> float:
    return stage2.selection_score(dataset, best_valid, test_mrr20, seen_count)


def load_history_rows() -> dict[tuple[str, str], list[stage2.HistSeed]]:
    return load_summary_rows(SUMMARY_SOURCES)


def load_current_axis_rows() -> dict[tuple[str, str], list[stage2.HistSeed]]:
    return load_summary_rows([SUMMARY_CSV])


def normalize_params(model: str, params: dict[str, Any]) -> dict[str, Any]:
    return stage2.normalize_params(model, params)


def summarize_target(rows: list[stage2.HistSeed], dataset_best_test: float) -> stage2.TargetStats:
    return stage2.summarize_target(rows, dataset_best_test)


def nearest_value(value: float, grid: list[int]) -> int:
    clipped = stage1.clamp_float(float(value), grid[0], grid[-1])
    return min(grid, key=lambda candidate: abs(candidate - clipped))


def nearest_dim(value: float) -> int:
    return nearest_value(value, DIM_GRID)


def nearest_seq(value: float) -> int:
    return nearest_value(value, SEQ_GRID)


def scale_dim(base: int, scale: float) -> int:
    return nearest_dim(base * scale)


def clamp_scale(value: float) -> float:
    return stage1.clamp_float(value, 0.4, 1.8)


def pick_heads(hidden: int, mode: str) -> int:
    if mode == "narrow":
        return 2
    if hidden >= 128:
        return 4
    return 2


def adjust_arch(
    model: str,
    base: dict[str, Any],
    *,
    hidden: int,
    layers: int,
    max_len: int,
    head_mode: str = "base",
    dropout_delta: float = 0.0,
    weight_decay_mult: float = 1.0,
) -> dict[str, Any]:
    params = dict(base)
    params["hidden_size"] = hidden
    params["embedding_size"] = hidden
    params["inner_size"] = hidden * 2
    params["max_len"] = nearest_seq(max_len)
    params["num_layers"] = stage1.clamp_int(layers, 1, 4)
    params["n_layers"] = stage1.clamp_int(layers, 1, 4)

    if model != "gru4rec":
        heads = pick_heads(hidden, head_mode)
        params["num_heads"] = heads
        params["n_heads"] = heads

    params["dropout"] = round(
        stage1.clamp_float(stage1.safe_float(params.get("dropout", 0.15), 0.15) + dropout_delta, 0.02, 0.45),
        4,
    )
    wd = stage1.safe_float(params.get("weight_decay", 1e-4), 1e-4)
    params["weight_decay"] = round(stage1.clamp_float(max(wd, 1e-6) * weight_decay_mult, 1e-6, 5e-2), 8)

    if model in {"difsr", "fdsa"}:
        params["attribute_hidden_size"] = stage1.clamp_int(nearest_dim(hidden * 0.75), 48, 256)
        params["use_attribute_predictor"] = True

    return normalize_params(model, params)


def apply_model_profile(model: str, params: dict[str, Any], profile: str) -> dict[str, Any]:
    out = dict(params)
    if model == "bsarec":
        mapping = {
            "alpha_lo_c8": (0.24, 8),
            "alpha_lo_c5": (0.35, 5),
            "alpha_mid_c3": (0.55, 3),
            "alpha_hi_c2": (0.78, 2),
            "alpha_peak_c1": (0.9, 1),
            "alpha_flat_c8": (0.62, 8),
        }
        alpha, c_value = mapping.get(profile, (0.55, 3))
        out["bsarec_alpha"] = alpha
        out["bsarec_c"] = c_value
    elif model == "fame":
        mapping = {
            "experts2": 2,
            "experts3": 3,
            "experts4": 4,
            "experts6": 6,
        }
        out["num_experts"] = mapping.get(profile, 3)
    elif model == "difsr":
        if profile == "gate_zeroattr":
            out["fusion_type"] = "gate"
            out["lambda_attr"] = 0.0
            out["attribute_hidden_size"] = 48
        elif profile == "gate_midattr":
            out["fusion_type"] = "gate"
            out["lambda_attr"] = 0.21
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 32, 64, 256)
        elif profile == "gate_highattr":
            out["fusion_type"] = "gate"
            out["lambda_attr"] = 0.24
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 96, 96, 256)
        elif profile == "concat_midattr":
            out["fusion_type"] = "concat"
            out["lambda_attr"] = 0.21
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 48, 64, 256)
        elif profile == "concat_highattr":
            out["fusion_type"] = "concat"
            out["lambda_attr"] = 0.26
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 64, 96, 256)
        elif profile == "concat_zeroattr":
            out["fusion_type"] = "concat"
            out["lambda_attr"] = 0.0
            out["attribute_hidden_size"] = 48
        elif profile == "gate_deep_attr":
            out["fusion_type"] = "gate"
            out["lambda_attr"] = 0.18
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 32, 48, 256)
        elif profile == "concat_wide_attr":
            out["fusion_type"] = "concat"
            out["lambda_attr"] = 0.22
            out["attribute_hidden_size"] = stage1.clamp_int(stage1.safe_int(out.get("hidden_size", 128), 128) + 64, 48, 256)
        out["use_attribute_predictor"] = True
    elif model == "gru4rec":
        if profile == "gru_reset":
            out["dropout"] = round(stage1.clamp_float(stage1.safe_float(out.get("dropout", 0.2), 0.2) + 0.02, 0.02, 0.45), 4)
        elif profile == "gru_deep":
            out["dropout"] = round(stage1.clamp_float(stage1.safe_float(out.get("dropout", 0.2), 0.2) - 0.02, 0.02, 0.45), 4)
    return normalize_params(model, out)


def history_signatures(model: str, rows: list[stage2.HistSeed]) -> set[str]:
    return {json.dumps(normalize_params(model, row.params), sort_keys=True, ensure_ascii=True) for row in rows}


def build_candidate(
    model: str,
    anchor_params: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    hidden = stage1.safe_int(anchor_params.get("hidden_size", anchor_params.get("embedding_size", 128)), 128)
    params = adjust_arch(
        model,
        anchor_params,
        hidden=scale_dim(hidden, spec.get("scale", 1.0)),
        layers=int(spec.get("layers", stage1.safe_int(anchor_params.get("num_layers", 2), 2))),
        max_len=int(spec.get("max_len", stage1.safe_int(anchor_params.get("max_len", 14), 14))),
        head_mode=str(spec.get("head_mode", "base")),
        dropout_delta=float(spec.get("dropout_delta", 0.0)),
        weight_decay_mult=float(spec.get("wd_mult", 1.0)),
    )
    profile = str(spec.get("profile", "")).strip()
    if profile:
        params = apply_model_profile(model, params, profile)
    meta = {
        "lr_mode": str(spec.get("lr_mode", "wide_mid")),
        "seq_mode": str(spec.get("seq_mode", "mid")),
        "profile": profile,
    }
    return str(spec["name"]), params, meta


def dedupe_candidates(
    model: str,
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]],
    history_sigs: set[str],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    seen = set(history_sigs)
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for combo_kind, params, meta in candidates:
        normalized = normalize_params(model, params)
        sig = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((combo_kind, normalized, meta))
    return out


def expand_family_specs(model: str) -> list[dict[str, Any]]:
    if model == "difsr":
        allowed_generic = {
            "tiny_short",
            "small_short",
            "small_mid",
            "base_short",
            "base_mid",
            "base_long",
            "wide_mid",
            "wide_long",
            "deep_narrow",
            "reg_heavy",
            "reg_light",
        }
        base_specs = [spec for spec in GENERIC_FAMILIES if spec["name"] in allowed_generic]
        base_specs.extend(model_specific_families(model))
    else:
        base_specs = [*GENERIC_FAMILIES, *model_specific_families(model)]
    expanded = [dict(spec) for spec in base_specs]
    if len(expanded) >= 48:
        return expanded

    if model == "difsr":
        scale_factors = [0.82, 0.92, 1.05, 1.15]
        seq_shifts = [-2, 0, 2, 4]
        layer_shifts = [0, 1, 0, -1]
        drop_shifts = [0.02, 0.04, 0.06, 0.0]
        wd_factors = [1.2, 1.6, 2.2, 2.8]
        lr_modes = ["wide_mid", "wide_high", "ultrawide", "wide_high"]
        anchors = ["balanced", "valid", "balanced"]
    else:
        scale_factors = [0.85, 0.92, 1.08, 1.18]
        seq_shifts = [-2, 2, 4, -4]
        layer_shifts = [0, 1, -1, 0]
        drop_shifts = [-0.02, 0.015, 0.03, -0.01]
        wd_factors = [0.7, 1.35, 1.8, 0.55]
        lr_modes = ["reset_low", "wide_mid", "wide_high", "ultrawide"]
        anchors = ["test", "balanced", "valid"]

    index = 0
    while len(expanded) < 52:
        base = base_specs[index % len(base_specs)]
        variant_id = index // len(base_specs)
        scale_factor = scale_factors[index % len(scale_factors)]
        seq_shift = seq_shifts[index % len(seq_shifts)]
        layer_shift = layer_shifts[index % len(layer_shifts)]
        drop_shift = drop_shifts[index % len(drop_shifts)]
        wd_factor = wd_factors[index % len(wd_factors)]
        derived = dict(base)
        derived["name"] = f"{base['name']}_alt{variant_id + 1}"
        derived["anchor"] = anchors[(index + variant_id) % len(anchors)]
        derived["scale"] = round(clamp_scale(float(base.get("scale", 1.0)) * scale_factor), 4)
        derived["max_len"] = nearest_seq(int(base.get("max_len", 14)) + seq_shift)
        derived["layers"] = stage1.clamp_int(int(base.get("layers", 2)) + layer_shift, 1, 4)
        derived["dropout_delta"] = round(float(base.get("dropout_delta", 0.0)) + drop_shift, 4)
        derived["wd_mult"] = round(stage1.clamp_float(float(base.get("wd_mult", 1.0)) * wd_factor, 0.35, 3.8), 4)
        derived["lr_mode"] = lr_modes[(index + 1) % len(lr_modes)]
        expanded.append(derived)
        index += 1

    return expanded


def base_lr_band(model: str, lr_mode: str) -> tuple[float, float]:
    bands = {
        "bsarec": {
            "reset_low": (8e-5, 1.6e-3),
            "wide_mid": (1.5e-4, 4.5e-3),
            "wide_high": (4e-4, 9e-3),
            "ultrawide": (5e-5, 1.2e-2),
        },
        "fame": {
            "reset_low": (7e-5, 1.2e-3),
            "wide_mid": (1.2e-4, 3.5e-3),
            "wide_high": (3e-4, 8e-3),
            "ultrawide": (5e-5, 1.1e-2),
        },
        "difsr": {
            "reset_low": (8e-4, 4.0e-3),
            "wide_mid": (2.0e-3, 1.0e-2),
            "wide_high": (6.0e-3, 1.8e-2),
            "ultrawide": (3.0e-3, 2.4e-2),
        },
        "gru4rec": {
            "reset_low": (2e-5, 8e-4),
            "wide_mid": (8e-5, 3.0e-3),
            "wide_high": (3e-4, 1.0e-2),
            "ultrawide": (2e-5, 2.0e-2),
        },
    }
    return bands[model][lr_mode]


def search_choices_around(center: float, values: list[float], lo: float, hi: float, precision: int = 4) -> list[float]:
    out = sorted({round(stage1.clamp_float(value, lo, hi), precision) for value in values})
    if round(stage1.clamp_float(center, lo, hi), precision) not in out:
        out.append(round(stage1.clamp_float(center, lo, hi), precision))
        out.sort()
    return out


def format_plain_float(value: float) -> str:
    text = format(float(value), ".12f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    if text in {"-0", "-0.0"}:
        return "0.0"
    return text


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format_plain_float(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    raise TypeError(f"Unsupported YAML scalar type: {type(value)!r}")


def emit_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                if not item:
                    empty = "[]" if isinstance(item, list) else "{}"
                    lines.append(f"{prefix}{key}: {empty}")
                else:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(emit_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                if not item:
                    empty = "[]" if isinstance(item, list) else "{}"
                    lines.append(f"{prefix}- {empty}")
                else:
                    lines.append(f"{prefix}-")
                    lines.extend(emit_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return lines
    return [f"{prefix}{yaml_scalar(value)}"]


def write_space_yaml(path: Path, fixed: dict[str, Any], search: dict[str, Any]) -> None:
    content = "\n".join(emit_yaml_lines({"fixed": fixed, "search": search})) + "\n"
    path.write_text(content, encoding="utf-8")


def build_search_block(model: str, combo_kind: str, params: dict[str, Any], meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    overrides: dict[str, str] = {"learning_rate": "loguniform"}
    search: dict[str, Any] = {}

    lr_lo, lr_hi = base_lr_band(model, meta["lr_mode"])
    search["learning_rate"] = [round(lr_lo, 8), round(lr_hi, 8)]

    max_len = stage1.safe_int(params.get("max_len", 14), 14)
    tuned_axes = 0

    if any(token in combo_kind for token in ["short", "long", "mid", "ultrashort", "len50"]):
        if meta["seq_mode"] == "ultrashort":
            seq_values = [max(5, max_len - 2), max_len, max_len + 2]
        elif meta["seq_mode"] == "short":
            seq_values = [max_len, max_len + 2, max_len + 4]
        elif meta["seq_mode"] == "doc_long":
            seq_values = [24, 32, 40, 50]
        elif meta["seq_mode"] == "long":
            seq_values = [max(10, max_len - 4), max_len, max_len + 4]
        else:
            seq_values = [max(8, max_len - 2), max_len, max_len + 2]
        search["MAX_ITEM_LIST_LENGTH"] = sorted({nearest_seq(value) for value in seq_values})
        overrides["MAX_ITEM_LIST_LENGTH"] = "choice"
        tuned_axes += 1
    elif "reg" in combo_kind:
        dropout = stage1.safe_float(params.get("dropout", 0.15), 0.15)
        search["dropout"] = search_choices_around(dropout, [dropout - 0.04, dropout, dropout + 0.05], 0.02, 0.45)
        overrides["dropout"] = "choice"
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

    if model == "bsarec":
        alpha = stage1.safe_float(params.get("bsarec_alpha", 0.5), 0.5)
        if "alpha" in combo_kind:
            search["bsarec_alpha"] = search_choices_around(alpha, [0.2, alpha, 0.65, 0.9], 0.1, 0.95)
            overrides["bsarec_alpha"] = "choice"
            tuned_axes += 1
        elif tuned_axes < 2:
            search["bsarec_c"] = sorted({1, 2, 5, 8})
            overrides["bsarec_c"] = "choice"
            tuned_axes += 1
    elif model == "fame":
        if "experts" in combo_kind:
            experts = stage1.safe_int(params.get("num_experts", 3), 3)
            search["num_experts"] = sorted({2, experts, 4, 6})
            overrides["num_experts"] = "choice"
            tuned_axes += 1
        elif tuned_axes < 2:
            search["num_layers"] = sorted({1, 2, 3})
            overrides["num_layers"] = "choice"
            tuned_axes += 1
    elif model == "difsr":
        if "gate" in combo_kind or "concat" in combo_kind:
            search["fusion_type"] = ["gate", "concat"]
            overrides["fusion_type"] = "choice"
            tuned_axes += 1
        elif "attr" in combo_kind:
            lambda_attr = stage1.safe_float(params.get("lambda_attr", 0.1), 0.1)
            if lambda_attr >= 0.18:
                lambda_values = [max(0.16, lambda_attr - 0.04), lambda_attr, min(0.28, lambda_attr + 0.02), min(0.3, lambda_attr + 0.05)]
            else:
                lambda_values = [0.0, 0.08, lambda_attr, 0.16]
            search["lambda_attr"] = search_choices_around(lambda_attr, lambda_values, 0.0, 0.3)
            overrides["lambda_attr"] = "choice"
            tuned_axes += 1
        elif tuned_axes < 2:
            attr_hidden = stage1.safe_int(params.get("attribute_hidden_size", 96), 96)
            if attr_hidden <= 112:
                attr_values = {attr_hidden, nearest_dim(attr_hidden * 1.2), 160, 224}
            else:
                attr_values = {96, attr_hidden, nearest_dim(attr_hidden * 1.2), 224}
            search["attribute_hidden_size"] = sorted(attr_values)
            overrides["attribute_hidden_size"] = "choice"
            tuned_axes += 1
    elif model == "gru4rec":
        if tuned_axes < 2:
            layers = stage1.safe_int(params.get("num_layers", 1), 1)
            search["num_layers"] = sorted({1, 2, 3, 4, layers})
            overrides["num_layers"] = "choice"

    return search, overrides


def model_specific_families(model: str) -> list[dict[str, Any]]:
    if model == "bsarec":
        return BSAREC_SPECIFIC
    if model == "fame":
        return FAME_SPECIFIC
    if model == "difsr":
        return DIFSR_SPECIFIC
    if model == "gru4rec":
        return GRU4REC_SPECIFIC
    return []


def scenario_variants(
    target: stage1.TargetSpec,
    base_test: dict[str, Any],
    base_balanced: dict[str, Any],
    base_valid: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    anchors = {
        "test": base_test,
        "balanced": base_balanced,
        "valid": base_valid,
    }
    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    for spec in expand_family_specs(target.model):
        anchor_params = anchors[spec["anchor"]]
        candidates.append(build_candidate(target.model, anchor_params, spec))

    return candidates


def model_append_families(model: str) -> list[dict[str, Any]]:
    if model == "bsarec":
        return BSAREC_APPEND
    if model == "fame":
        return FAME_APPEND
    if model == "difsr":
        return DIFSR_APPEND
    if model == "gru4rec":
        return GRU4REC_APPEND
    return []


def choose_append_anchors(model: str, rows: list[stage2.HistSeed]) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}

    by_test = sorted(rows, key=lambda item: (item.test_mrr20, item.best_valid_mrr20, item.selection_score), reverse=True)
    by_valid = sorted(rows, key=lambda item: (item.best_valid_mrr20, item.test_mrr20, item.selection_score), reverse=True)
    by_balanced = sorted(rows, key=lambda item: (item.selection_score, item.test_mrr20, item.best_valid_mrr20), reverse=True)

    def pick(sorted_rows: list[stage2.HistSeed], index: int = 0) -> stage2.HistSeed:
        return sorted_rows[min(index, len(sorted_rows) - 1)]

    def by_len(lo: int | None = None, hi: int | None = None) -> stage2.HistSeed:
        filtered = []
        for item in rows:
            max_len = stage1.safe_int(item.params.get("max_len", 14), 14)
            if lo is not None and max_len < lo:
                continue
            if hi is not None and max_len > hi:
                continue
            filtered.append(item)
        if not filtered:
            return pick(by_test, 0)
        filtered.sort(key=lambda item: (item.test_mrr20, item.best_valid_mrr20, item.selection_score), reverse=True)
        return filtered[0]

    anchors = {
        "best_test": pick(by_test, 0),
        "test_top2": pick(by_test, 1),
        "test_top3": pick(by_test, 2),
        "valid_best": pick(by_valid, 0),
        "balanced_best": pick(by_balanced, 0),
        "short_best": by_len(hi=12),
        "mid_best": by_len(14, 20),
        "long_best": by_len(24, None),
    }
    return {key: normalize_params(model, value.params) for key, value in anchors.items()}


def build_append_combo_specs(
    target: stage1.TargetSpec,
    history_rows: list[stage2.HistSeed],
    current_rows: list[stage2.HistSeed],
    est_runtime_sec: float,
    combo_seed_base: int,
) -> list[stage1.ComboSpec]:
    desired_count = MODEL_APPEND_BUDGET.get(target.model, 0)
    if desired_count <= 0 or not current_rows:
        return []

    anchors = choose_append_anchors(target.model, current_rows)
    if not anchors:
        return []

    candidates: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for spec in model_append_families(target.model):
        anchor_key = str(spec.get("anchor", "best_test"))
        anchor_params = anchors.get(anchor_key)
        if anchor_params is None:
            continue
        candidates.append(build_candidate(target.model, anchor_params, spec))

    history_sigs = history_signatures(target.model, [*history_rows, *current_rows])
    uniq = dedupe_candidates(target.model, candidates, history_sigs)

    if len(uniq) < desired_count:
        seen_sigs = set(history_sigs)
        for _combo_kind, params, _meta in uniq:
            seen_sigs.add(json.dumps(normalize_params(target.model, params), sort_keys=True, ensure_ascii=True))

        base_templates = model_append_families(target.model)
        scale_steps = [0.88, 0.94, 1.06, 1.12]
        seq_steps = [-4, -2, 2, 4]
        drop_steps = [0.0, 0.01, -0.01, 0.02]
        idx = 0
        while len(uniq) < desired_count and idx < 128:
            base = dict(base_templates[idx % len(base_templates)])
            base["name"] = f"{base['name']}_fill{idx + 1}"
            base["scale"] = round(clamp_scale(float(base.get("scale", 1.0)) * scale_steps[idx % len(scale_steps)]), 4)
            base["max_len"] = nearest_seq(int(base.get("max_len", 14)) + seq_steps[idx % len(seq_steps)])
            base["layers"] = stage1.clamp_int(int(base.get("layers", 2)) + (1 if idx % 5 == 0 else 0) - (1 if idx % 7 == 0 else 0), 1, 4)
            base["dropout_delta"] = round(float(base.get("dropout_delta", 0.0)) + drop_steps[idx % len(drop_steps)], 4)
            base["wd_mult"] = round(stage1.clamp_float(float(base.get("wd_mult", 1.0)) * (1.0 + 0.08 * ((idx % 3) - 1)), 0.35, 3.8), 4)
            anchor_key = str(base.get("anchor", "best_test"))
            anchor_params = anchors.get(anchor_key, anchors["best_test"])
            candidate = build_candidate(target.model, anchor_params, base)
            normalized = normalize_params(target.model, candidate[1])
            sig = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                uniq.append((candidate[0], normalized, candidate[2]))
            idx += 1

    specs: list[stage1.ComboSpec] = []
    start_idx = MODEL_COMBO_BUDGET.get(target.model, 40) + 1
    dataset_tag = stage1.sanitize_token(target.dataset)
    model_tag = stage1.sanitize_token(target.model)

    for combo_number, (combo_kind, params, meta) in zip(range(start_idx, start_idx + desired_count), uniq[:desired_count]):
        combo_id = f"K{combo_number}"
        search, type_overrides = build_search_block(target.model, combo_kind, params, meta)
        fixed = stage1.make_fixed_block(target.model, params)
        fixed["search_space_type_overrides"] = type_overrides
        space_yaml = SPACE_ROOT / f"{dataset_tag}_{model_tag}_{combo_id}.yaml"
        write_space_yaml(space_yaml, fixed, search)
        specs.append(
            stage1.ComboSpec(
                dataset=target.dataset,
                model=target.model,
                combo_id=combo_id,
                combo_kind=combo_kind,
                priority=target.priority,
                est_runtime_sec=est_runtime_sec,
                max_evals=max_evals_for_target(target, est_runtime_sec),
                seed=combo_seed_base + combo_number,
                base_params=params,
                search_space=search,
                fixed_space=fixed,
                space_yaml=space_yaml,
                run_phase=f"BASELINE2_ADDTUNE3_2_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
            )
        )
    return specs


def combo_count_for_target(target: stage1.TargetSpec, est_runtime_sec: float) -> int:
    _ = est_runtime_sec
    return MODEL_COMBO_BUDGET.get(target.model, 40)


def max_evals_for_target(target: stage1.TargetSpec, est_runtime_sec: float) -> int:
    base = MODEL_MAX_EVALS.get(target.model, 10)
    if est_runtime_sec > 240:
        base -= 1
    if est_runtime_sec < 90:
        base += 1
    return stage1.clamp_int(base, 8, 13)


def build_combo_specs(
    target: stage1.TargetSpec,
    rows: list[stage2.HistSeed],
    est_runtime_sec: float,
    combo_seed_base: int,
) -> list[stage1.ComboSpec]:
    desired_count = combo_count_for_target(target, est_runtime_sec)
    best_test, best_balanced = stage2.choose_seed_rows(rows)
    best_valid = max(rows, key=lambda item: (item.best_valid_mrr20, item.test_mrr20, item.selection_score))

    base_test = normalize_params(target.model, best_test.params)
    base_balanced = normalize_params(target.model, best_balanced.params)
    base_valid = normalize_params(target.model, best_valid.params)

    candidates = scenario_variants(target, base_test, base_balanced, base_valid)
    uniq = dedupe_candidates(target.model, candidates, history_signatures(target.model, rows))

    specs: list[stage1.ComboSpec] = []
    SPACE_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, (combo_kind, params, meta) in enumerate(uniq[:desired_count], start=1):
        combo_id = f"K{idx}"
        search, type_overrides = build_search_block(target.model, combo_kind, params, meta)
        fixed = stage1.make_fixed_block(target.model, params)
        fixed["search_space_type_overrides"] = type_overrides

        dataset_tag = stage1.sanitize_token(target.dataset)
        model_tag = stage1.sanitize_token(target.model)
        space_yaml = SPACE_ROOT / f"{dataset_tag}_{model_tag}_{combo_id}.yaml"
        write_space_yaml(space_yaml, fixed, search)

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
                run_phase=f"BASELINE2_ADDTUNE3_2_{stage1.sanitize_token(target.dataset).upper()}_{stage1.sanitize_token(target.model).upper()}_{combo_id}",
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
    parser = argparse.ArgumentParser(description="Stage-3.2 from-scratch wide beauty retuning for weak baseline_2 models.")
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
    print(f"[baseline2_addtuning3_2] archived previous output to {backup_root}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return False


def build_all_specs(args: argparse.Namespace) -> list[stage1.ComboSpec]:
    history = load_history_rows()
    current_axis = load_current_axis_rows()
    runtimes = stage1.median_runtime_by_target(history)
    filters = stage1.parse_target_filter(args.targets)
    targets = target_specs()
    dataset_best_test: dict[str, float] = {}
    for (dataset, _model), rows in history.items():
        best = max((row.test_mrr20 for row in rows), default=0.0)
        dataset_best_test[dataset] = max(dataset_best_test.get(dataset, 0.0), best)

    specs: list[stage1.ComboSpec] = []
    combo_seed_base = 2026042000
    for idx, target in enumerate(targets):
        if filters and (target.dataset, target.model) not in filters:
            continue
        rows = history.get((target.dataset, target.model), [])
        if not rows:
            continue
        est_runtime_sec = runtimes.get((target.dataset, target.model), 600.0)
        _stats = summarize_target(rows, dataset_best_test.get(target.dataset, 0.0))
        specs.extend(build_combo_specs(target, rows, est_runtime_sec, combo_seed_base + idx * 100))
        specs.extend(
            build_append_combo_specs(
                target,
                rows,
                current_axis.get((target.dataset, target.model), []),
                est_runtime_sec,
                combo_seed_base + idx * 100,
            )
        )

    specs.sort(
        key=lambda spec: (
            PRIORITY_RANK.get(spec.priority, 9),
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

    print(f"[baseline2_addtuning3_2] python={args.python_bin}")
    print(f"[baseline2_addtuning3_2] jobs={len(specs)} gpus={','.join(gpus)} plan={PLAN_CSV}")
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
        print(f"[baseline2_addtuning3_2] nothing to run; summary={SUMMARY_CSV}")
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
                f"[baseline2_addtuning3_2] start dataset={spec.dataset} model={spec.model} "
                f"combo={spec.combo_id} gpu={gpu_id} max_evals={spec.max_evals}"
            )
            row = run_one(spec, gpu_id, args.python_bin, args.search_algo)
            with write_lock:
                stage1.append_summary_row(SUMMARY_CSV, row, summary_fields)
            print(
                f"[baseline2_addtuning3_2] done dataset={spec.dataset} model={spec.model} combo={spec.combo_id} "
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

    print(f"[baseline2_addtuning3_2] completed summary={SUMMARY_CSV}")


if __name__ == "__main__":
    main()