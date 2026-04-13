#!/usr/bin/env python3
"""Stage H targeted recovery runner for weak baseline combos.

Focus:
- Recover the largest underperformers from Stage G
- Keep search compact but more diverse than transfer-only Stage G candidates
- Reuse Stage G execution/logging skeleton
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))

import run_stageG_cross6x9 as sg
from slack_progress import SlackProgressNotifier

base = sg.base


base.AXIS = "StageH_TargetedRecovery_anchor2_core5"
base.PHASE_ID = "P23"
base.PHASE_NAME = "STAGEH_TARGETEDRECOVERY_ANCHOR2_CORE5"
base.AXIS_DESC = "stageh_targetedrecovery_anchor2_core5"

_orig_model_runtime_resource_overrides = base._model_runtime_resource_overrides


TIER_BY_COMBO: Dict[Tuple[str, str], str] = {
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"): "hard",
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"): "hard",
    ("KuaiRecLargeStrictPosV2_0.2", "fearec"): "hard",
    ("KuaiRecLargeStrictPosV2_0.2", "sasrec"): "medium",
    ("lastfm0.03", "gru4rec"): "soft",
    ("lastfm0.03", "sigma"): "soft",
    ("amazon_beauty", "gru4rec"): "hard",
    ("amazon_beauty", "tisasrec"): "hard",
    ("amazon_beauty", "bsarec"): "hard",
    ("amazon_beauty", "fame"): "hard",
    ("foursquare", "gru4rec"): "medium",
    ("foursquare", "sigma"): "medium",
    ("foursquare", "tisasrec"): "soft",
    ("foursquare", "duorec"): "soft",
    ("foursquare", "bsarec"): "soft",
    ("foursquare", "fearec"): "soft",
    ("foursquare", "fame"): "soft",
    ("movielens1m", "fearec"): "soft",
    ("retail_rocket", "sasrec"): "soft",
    ("retail_rocket", "gru4rec"): "soft",
    ("retail_rocket", "tisasrec"): "soft",
    ("retail_rocket", "duorec"): "soft",
    ("retail_rocket", "fearec"): "soft",
}

TARGET_COMBOS: List[Tuple[str, str]] = list(TIER_BY_COMBO.keys())

OVERALL_BASELINE_AXES = [
    "StageA_LR_anchor2_core5",
    "StageB_Structure_anchor2_core5",
    "StageC_Focus_anchor2_core5",
    "StageD_MicroWide_anchor2_core5",
    "StageE_ReLRSeed_anchor2_core5",
    "StageF_TailBoost_anchor2_core5",
    "StageG_Cross6x9_anchor2_core5",
    "StageH_TargetedRecovery_anchor2_core5",
    "Final_all_datasets",
]

CURRENT_DYNAMIC_TIER_BY_COMBO: Dict[Tuple[str, str], str] = {}
CURRENT_DYNAMIC_TARGETS: Set[Tuple[str, str]] = set()
CURRENT_ULTRA_LOW_TARGETS: Set[Tuple[str, str]] = set()

PRIOR_BEST_TABLE_PATH = (
    base.REPO_ROOT / "experiments/run/baseline/docs/stageH_AtoG_plus_fmoeA6_best_table_20260409.csv"
)

FULL_SHORTLIST_COMBOS = {
    ("KuaiRecLargeStrictPosV2_0.2", "sasrec"),
    ("KuaiRecLargeStrictPosV2_0.2", "fearec"),
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"),
    ("amazon_beauty", "tisasrec"),
    ("foursquare", "fame"),
    ("retail_rocket", "sasrec"),
    ("retail_rocket", "duorec"),
    ("retail_rocket", "fearec"),
}

RESCUE_REDESIGN_COMBOS = {
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"),
    ("amazon_beauty", "gru4rec"),
    ("amazon_beauty", "fame"),
    ("amazon_beauty", "bsarec"),
    ("foursquare", "gru4rec"),
    ("foursquare", "sigma"),
    ("foursquare", "bsarec"),
    ("lastfm0.03", "sigma"),
    ("retail_rocket", "gru4rec"),
}

DEFAULT_DATASETS = list(dict.fromkeys(ds for ds, _ in TARGET_COMBOS))
DEFAULT_MODELS = list(dict.fromkeys(model for _, model in TARGET_COMBOS))

AGGRESSIVE_LOW_PERF_COMBOS = {
    combo for combo, tier in TIER_BY_COMBO.items() if tier == "hard"
}

MEDIUM_LOW_PERF_COMBOS = {
    combo for combo, tier in TIER_BY_COMBO.items() if tier == "medium"
}

TARGET_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    combo: (4 if tier == "hard" else 3 if tier == "medium" else 2)
    for combo, tier in TIER_BY_COMBO.items()
}

FAST_SCREEN_MIN_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    combo: (18 if tier == "hard" else 12 if tier == "medium" else 8)
    for combo, tier in TIER_BY_COMBO.items()
}

FOLLOWUP_RESCUE_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    combo: (24 if TIER_BY_COMBO[combo] == "hard" else 16 if TIER_BY_COMBO[combo] == "medium" else 12)
    for combo in TARGET_COMBOS
}

FOLLOWUP_FULL_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    combo: (4 if TIER_BY_COMBO[combo] == "hard" else 3)
    for combo in TARGET_COMBOS
}

MANUAL_TEMPLATE_BANK: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
    ("amazon_beauty", "gru4rec"): [
        {
            "id": "AB_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 264,
                "embedding_size": 264,
                "layers": 3,
                "num_layers": 3,
                "max_len": 10,
                "dropout": 0.05,
                "weight_decay": 4.5e-5,
            },
            "lr_lo": 1.2e-3,
            "lr_hi": 4.8e-3,
        },
        {
            "id": "AB_GRU_STAGEH_REC2",
            "config": {
                "hidden_size": 320,
                "embedding_size": 320,
                "layers": 3,
                "num_layers": 3,
                "max_len": 8,
                "dropout": 0.04,
                "weight_decay": 3.5e-5,
            },
            "lr_lo": 3.0e-3,
            "lr_hi": 1.0e-2,
        },
    ],
    ("amazon_beauty", "fame"): [
        {
            "id": "AB_FAME_STAGEH_REC1",
            "config": {
                "hidden_size": 88,
                "embedding_size": 88,
                "layers": 3,
                "num_layers": 3,
                "heads": 8,
                "inner_size": 176,
                "max_len": 10,
                "dropout": 0.14,
                "weight_decay": 1.2e-4,
                "num_experts": 4,
            },
            "lr_lo": 1.5e-3,
            "lr_hi": 6.0e-3,
        },
        {
            "id": "AB_FAME_STAGEH_REC2",
            "config": {
                "hidden_size": 112,
                "embedding_size": 112,
                "layers": 3,
                "num_layers": 3,
                "heads": 8,
                "inner_size": 224,
                "max_len": 8,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "num_experts": 3,
            },
            "lr_lo": 2.5e-3,
            "lr_hi": 1.0e-2,
        },
    ],
    ("amazon_beauty", "bsarec"): [
        {
            "id": "AB_BSAREC_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 10,
                "dropout": 0.12,
                "weight_decay": 2.0e-4,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 1.1e-3,
        },
    ],
    ("amazon_beauty", "tisasrec"): [
        {
            "id": "AB_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 10,
                "time_span": 256,
                "dropout": 0.10,
                "weight_decay": 1.5e-4,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 9.0e-4,
        },
        {
            "id": "AB_TISAS_STAGEH_REC2",
            "config": {
                "hidden_size": 96,
                "embedding_size": 96,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 192,
                "max_len": 8,
                "time_span": 96,
                "dropout": 0.06,
                "weight_decay": 6.0e-5,
            },
            "lr_lo": 1.2e-3,
            "lr_hi": 4.0e-3,
        },
    ],
    ("foursquare", "gru4rec"): [
        {
            "id": "FSQ_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 192,
                "embedding_size": 192,
                "layers": 2,
                "num_layers": 2,
                "max_len": 10,
                "dropout": 0.04,
                "weight_decay": 2.5e-5,
            },
            "lr_lo": 2.0e-3,
            "lr_hi": 8.0e-3,
        },
        {
            "id": "FSQ_GRU_STAGEH_REC2",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "max_len": 8,
                "dropout": 0.08,
                "weight_decay": 8.0e-5,
            },
            "lr_lo": 5.0e-4,
            "lr_hi": 2.2e-3,
        },
    ],
    ("foursquare", "tisasrec"): [
        {
            "id": "FSQ_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "time_span": 128,
                "dropout": 0.08,
                "weight_decay": 7.0e-5,
            },
            "lr_lo": 8.0e-4,
            "lr_hi": 2.6e-3,
        },
        {
            "id": "FSQ_TISAS_STAGEH_REC2",
            "config": {
                "hidden_size": 96,
                "embedding_size": 96,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 192,
                "max_len": 8,
                "time_span": 64,
                "dropout": 0.05,
                "weight_decay": 4.0e-5,
            },
            "lr_lo": 1.4e-3,
            "lr_hi": 4.0e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"): [
        {
            "id": "KR_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 320,
                "embedding_size": 320,
                "layers": 3,
                "num_layers": 3,
                "max_len": 10,
                "dropout": 0.05,
                "weight_decay": 3.0e-5,
            },
            "lr_lo": 2.0e-3,
            "lr_hi": 8.0e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"): [
        {
            "id": "KR_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 20,
                "time_span": 384,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
            },
            "lr_lo": 4.0e-4,
            "lr_hi": 1.0e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "fearec"): [
        {
            "id": "KR_FEA_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.08,
                "weight_decay": 7.0e-5,
                "contrast": "un",
                "tau": 0.16,
                "lmd": 0.016,
                "lmd_sem": 0.0,
                "global_ratio": 0.70,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 9.0e-4,
            "lr_hi": 3.6e-3,
        },
        {
            "id": "KR_FEA_STAGEH_REC2",
            "config": {
                "hidden_size": 96,
                "embedding_size": 96,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 192,
                "max_len": 8,
                "dropout": 0.12,
                "weight_decay": 1.5e-4,
                "contrast": "un",
                "tau": 0.22,
                "lmd": 0.010,
                "lmd_sem": 0.0,
                "global_ratio": 0.55,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 1.4e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "sasrec"): [
        {
            "id": "KR_SAS_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
            },
            "lr_lo": 8.0e-4,
            "lr_hi": 2.4e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "duorec"): [
        {
            "id": "KR_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 20,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.25,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 9.0e-4,
        },
    ],
    ("lastfm0.03", "gru4rec"): [
        {
            "id": "LFM_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 2,
                "num_layers": 2,
                "max_len": 10,
                "dropout": 0.06,
                "weight_decay": 4.0e-5,
            },
            "lr_lo": 5.0e-4,
            "lr_hi": 2.2e-3,
        },
    ],
    ("lastfm0.03", "sigma"): [
        {
            "id": "LFM_SIGMA_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
            },
            "lr_lo": 5.0e-4,
            "lr_hi": 1.6e-3,
        },
    ],
    ("movielens1m", "sasrec"): [
        {
            "id": "ML1M_SAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.12,
                "weight_decay": 1.0e-4,
            },
            "lr_lo": 4.0e-4,
            "lr_hi": 1.2e-3,
        },
    ],
    ("movielens1m", "gru4rec"): [
        {
            "id": "ML1M_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 264,
                "embedding_size": 264,
                "layers": 3,
                "num_layers": 3,
                "max_len": 20,
                "dropout": 0.05,
                "weight_decay": 5.0e-5,
            },
            "lr_lo": 1.0e-3,
            "lr_hi": 4.0e-3,
        },
    ],
    ("movielens1m", "duorec"): [
        {
            "id": "ML1M_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.25,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 8.0e-4,
        },
    ],
    ("movielens1m", "fearec"): [
        {
            "id": "ML1M_FEA_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "global_ratio": 0.75,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 8.0e-4,
        },
    ],
    ("foursquare", "sigma"): [
        {
            "id": "FSQ_SIGMA_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
            },
            "lr_lo": 8.0e-4,
            "lr_hi": 2.6e-3,
        },
    ],
    ("foursquare", "duorec"): [
        {
            "id": "FSQ_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.08,
                "weight_decay": 7.0e-5,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.016,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 7.0e-4,
            "lr_hi": 2.2e-3,
        },
    ],
    ("foursquare", "bsarec"): [
        {
            "id": "FSQ_BSA_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 9.0e-5,
            },
            "lr_lo": 7.0e-4,
            "lr_hi": 2.2e-3,
        },
    ],
    ("foursquare", "fearec"): [
        {
            "id": "FSQ_FEA_STAGEH_REC1",
            "config": {
                "hidden_size": 112,
                "embedding_size": 112,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 224,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.014,
                "lmd_sem": 0.0,
                "global_ratio": 0.70,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 7.0e-4,
            "lr_hi": 2.4e-3,
        },
    ],
    ("foursquare", "fame"): [
        {
            "id": "FSQ_FAME_STAGEH_REC1",
            "config": {
                "hidden_size": 80,
                "embedding_size": 80,
                "layers": 2,
                "num_layers": 2,
                "heads": 4,
                "inner_size": 160,
                "max_len": 8,
                "dropout": 0.12,
                "weight_decay": 1.0e-4,
                "num_experts": 3,
            },
            "lr_lo": 1.4e-3,
            "lr_hi": 5.0e-3,
        },
    ],
    ("retail_rocket", "sasrec"): [
        {
            "id": "RR_SAS_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 8,
                "dropout": 0.12,
                "weight_decay": 1.0e-4,
            },
            "lr_lo": 7.0e-4,
            "lr_hi": 2.4e-3,
        },
    ],
    ("retail_rocket", "gru4rec"): [
        {
            "id": "RR_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 192,
                "embedding_size": 192,
                "layers": 2,
                "num_layers": 2,
                "max_len": 10,
                "dropout": 0.04,
                "weight_decay": 2.5e-5,
            },
            "lr_lo": 1.6e-3,
            "lr_hi": 7.0e-3,
        },
        {
            "id": "RR_GRU_STAGEH_REC2",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "max_len": 8,
                "dropout": 0.08,
                "weight_decay": 8.0e-5,
            },
            "lr_lo": 5.0e-4,
            "lr_hi": 2.4e-3,
        },
    ],
    ("retail_rocket", "tisasrec"): [
        {
            "id": "RR_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 10,
                "time_span": 96,
                "dropout": 0.08,
                "weight_decay": 6.0e-5,
            },
            "lr_lo": 8.0e-4,
            "lr_hi": 2.4e-3,
        },
        {
            "id": "RR_TISAS_STAGEH_REC2",
            "config": {
                "hidden_size": 96,
                "embedding_size": 96,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 192,
                "max_len": 8,
                "time_span": 64,
                "dropout": 0.05,
                "weight_decay": 4.0e-5,
            },
            "lr_lo": 1.2e-3,
            "lr_hi": 4.0e-3,
        },
    ],
    ("retail_rocket", "fame"): [
        {
            "id": "RR_FAME_STAGEH_REC1",
            "config": {
                "hidden_size": 96,
                "embedding_size": 96,
                "layers": 2,
                "num_layers": 2,
                "heads": 4,
                "inner_size": 192,
                "max_len": 10,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
                "num_experts": 4,
            },
            "lr_lo": 1.2e-3,
            "lr_hi": 5.0e-3,
        },
        {
            "id": "RR_FAME_STAGEH_REC2",
            "config": {
                "hidden_size": 72,
                "embedding_size": 72,
                "layers": 2,
                "num_layers": 2,
                "heads": 4,
                "inner_size": 144,
                "max_len": 8,
                "dropout": 0.14,
                "weight_decay": 1.4e-4,
                "num_experts": 3,
            },
            "lr_lo": 2.2e-3,
            "lr_hi": 1.0e-2,
        },
    ],
    ("retail_rocket", "duorec"): [
        {
            "id": "RR_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 128,
                "embedding_size": 128,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 256,
                "max_len": 8,
                "dropout": 0.10,
                "weight_decay": 9.0e-5,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.016,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 7.0e-4,
            "lr_hi": 2.6e-3,
        },
    ],
    ("retail_rocket", "fearec"): [
        {
            "id": "RR_FEA_STAGEH_REC1",
            "config": {
                "hidden_size": 112,
                "embedding_size": 112,
                "layers": 2,
                "num_layers": 2,
                "heads": 2,
                "inner_size": 224,
                "max_len": 8,
                "dropout": 0.10,
                "weight_decay": 8.0e-5,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.014,
                "lmd_sem": 0.0,
                "global_ratio": 0.65,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 8.0e-4,
            "lr_hi": 2.8e-3,
        },
    ],
}


def _effective_tier(dataset: str, model_option: str) -> str:
    return CURRENT_DYNAMIC_TIER_BY_COMBO.get((str(dataset), str(model_option).lower()), TIER_BY_COMBO.get((str(dataset), str(model_option).lower()), "soft"))


def _overall_baseline_best_by_combo() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for axis in OVERALL_BASELINE_AXES:
        axis_dir = base.LOG_ROOT / axis
        if not axis_dir.exists():
            continue
        for dsdir in axis_dir.iterdir():
            if not dsdir.is_dir():
                continue
            summary = dsdir / "summary.csv"
            if not summary.exists():
                continue
            try:
                with summary.open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
            except Exception:
                continue
            for row in rows:
                ds = str(row.get("dataset", "")).strip()
                model_label = str(row.get("model", "")).strip()
                model_option = sg.LABEL_TO_OPTION.get(model_label)
                if not ds or model_option is None:
                    continue
                valid = sg._metric(row.get("run_best_valid_mrr20"))
                test = sg._metric(row.get("run_best_test_mrr20"))
                if valid is None:
                    continue
                key = (ds, model_option)
                prev = out.get(key)
                if prev is None or float(valid) > float(prev["valid"]):
                    out[key] = {
                        "valid": float(valid),
                        "test": None if test is None else float(test),
                        "axis": axis,
                        "run_phase": str(row.get("run_phase", "")),
                    }
    return out


def _compute_underperform_targets(*, weak_ratio: float, strong_ratio: float) -> Dict[Tuple[str, str], str]:
    best = _overall_baseline_best_by_combo()
    by_ds: Dict[str, float] = {}
    for (ds, _model), rec in best.items():
        cur = by_ds.get(ds)
        val = float(rec["valid"])
        by_ds[ds] = val if cur is None else max(float(cur), val)

    out: Dict[Tuple[str, str], str] = {}
    for combo, rec in best.items():
        ds = combo[0]
        ds_max = float(by_ds.get(ds, 0.0) or 0.0)
        if ds_max <= 0.0:
            continue
        ratio = float(rec["valid"]) / ds_max
        if ratio <= float(strong_ratio):
            out[combo] = "hard"
        elif ratio <= float(weak_ratio):
            out[combo] = "medium"
    return out


def _compute_lowperform2_targets(
    *,
    low_ratio: float,
    ultra_ratio: float,
) -> Tuple[Dict[Tuple[str, str], str], Set[Tuple[str, str]]]:
    best = _overall_baseline_best_by_combo()
    by_ds: Dict[str, float] = {}
    for (ds, _model), rec in best.items():
        cur = by_ds.get(ds)
        val = float(rec["valid"])
        by_ds[ds] = val if cur is None else max(float(cur), val)

    out: Dict[Tuple[str, str], str] = {}
    ultra: Set[Tuple[str, str]] = set()
    for combo, rec in best.items():
        ds = combo[0]
        ds_max = float(by_ds.get(ds, 0.0) or 0.0)
        if ds_max <= 0.0:
            continue
        ratio = float(rec["valid"]) / ds_max
        if ratio <= float(ultra_ratio):
            out[combo] = "hard"
            ultra.add(combo)
        elif ratio <= float(low_ratio):
            out[combo] = "medium"
    return out, ultra


def _candidate_goal(dataset: str, model_option: str, *, args: argparse.Namespace) -> int:
    combo = (str(dataset), str(model_option).lower())
    tier = _effective_tier(dataset, model_option)
    if args.followup_full:
        return int(FOLLOWUP_FULL_CANDIDATE_COUNT.get(combo, 4 if tier == "hard" else 3))
    if args.followup_rescue:
        return int(FOLLOWUP_RESCUE_CANDIDATE_COUNT.get(combo, 24 if tier == "hard" else 16 if tier == "medium" else 12))
    if args.lowperform2_screen:
        if combo in CURRENT_ULTRA_LOW_TARGETS:
            return 28
        if tier == "hard":
            return 22
        if tier == "medium":
            return 18
        return 14
    if args.underperform_screen:
        if combo in TIER_BY_COMBO:
            return 20 if tier == "hard" else 14
        return 14 if tier == "hard" else 10
    if args.fast_screen:
        return int(FAST_SCREEN_MIN_CANDIDATE_COUNT.get(combo, 18 if tier == "hard" else 12 if tier == "medium" else 8))
    return int(TARGET_CANDIDATE_COUNT.get(combo, 4 if tier == "hard" else 3 if tier == "medium" else 2))


def _manual_candidates(dataset: str, model_option: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for template in MANUAL_TEMPLATE_BANK.get((dataset, model_option), []):
        out.append(
            sg._manual_candidate(
                template,
                model_option,
                source_stage="stageh_manual_bank",
                transfer_from_dataset=dataset,
            )
        )
    return out


def _stageh_target_max_len(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    *,
    fast_screen: bool,
) -> int:
    model_key = str(model_option).lower()
    hidden_size = int(cfg.get("hidden_size", 128))
    layers = int(cfg.get("layers", cfg.get("num_layers", 2)))
    time_span = int(cfg.get("time_span", 384))

    # Keep StageH aggressively short by default. Only allow 20 when the config
    # is genuinely heavier or time-aware enough that a tiny horizon is likely
    # to distort the ranking too much.
    if model_key == "tisasrec":
        return 20 if (not fast_screen and time_span >= 320) else 10
    if model_key in {"duorec", "fearec"}:
        return 20 if (not fast_screen and (hidden_size >= 160 or layers >= 3)) else 10
    if dataset == "KuaiRecLargeStrictPosV2_0.2" and model_key in {"sasrec", "bsarec", "difsr", "sigma"}:
        return 20 if (not fast_screen and hidden_size >= 160 and layers >= 3) else 10
    return 10


def _speedify_candidate_config(
    *,
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    fast_screen: bool,
    lowperform2_screen: bool = False,
) -> Dict[str, Any]:
    out = sg._normalize_cfg(model_option, dict(cfg))
    tier = _effective_tier(str(dataset), str(model_option).lower())
    target_max_len = _stageh_target_max_len(dataset, model_option, out, fast_screen=fast_screen)
    out["max_len"] = min(int(out.get("max_len", target_max_len)), int(target_max_len))

    if model_option in {"sasrec", "tisasrec", "bsarec", "duorec", "fearec", "difsr", "fame"}:
        if fast_screen and tier == "hard":
            layer_cap = 3
        elif fast_screen and tier == "medium":
            layer_cap = 2
        else:
            layer_cap = 2 if fast_screen else 3
        layers = min(int(out.get("layers", 2)), int(layer_cap))
        out["layers"] = layers
        out["num_layers"] = layers

    if model_option in {"duorec", "fearec"}:
        if fast_screen and tier == "hard":
            head_cap = 4
        elif fast_screen and tier == "medium":
            head_cap = 3
        else:
            head_cap = 2 if fast_screen else 4
        heads = min(int(out.get("heads", 2)), int(head_cap))
        out["heads"] = heads
        if fast_screen and not lowperform2_screen:
            hs_cap = 160 if tier == "hard" else 144 if tier == "medium" else 128
            inner_cap = 320 if tier == "hard" else 288 if tier == "medium" else 256
            out["hidden_size"] = min(int(out.get("hidden_size", 128)), hs_cap)
            out["embedding_size"] = min(int(out.get("embedding_size", out["hidden_size"])), int(out["hidden_size"]))
            out["inner_size"] = min(int(out.get("inner_size", max(256, out["hidden_size"] * 2))), inner_cap)

    if model_option in {"sasrec", "bsarec", "difsr", "tisasrec"} and fast_screen and not lowperform2_screen:
        hs_cap = 160 if tier == "hard" else 144
        inner_cap = 320 if tier == "hard" else 288
        out["hidden_size"] = min(int(out.get("hidden_size", 128)), hs_cap)
        out["embedding_size"] = min(int(out.get("embedding_size", out["hidden_size"])), int(out["hidden_size"]))
        out["inner_size"] = min(int(out.get("inner_size", max(256, out["hidden_size"] * 2))), inner_cap)

    return sg._normalize_cfg(model_option, out)


def _lr_floor(*, lowperform2_screen: bool) -> float:
    return 1e-5 if lowperform2_screen else 8e-5


def _speedify_lr_band(
    *,
    model_option: str,
    lr_lo: float,
    lr_hi: float,
    fast_screen: bool,
    lowperform2_screen: bool = False,
) -> Tuple[float, float]:
    model_key = str(model_option).lower()
    lr_floor = _lr_floor(lowperform2_screen=lowperform2_screen)
    if lowperform2_screen:
        # For lowperform2 we want genuinely wide exploration, not just
        # local refinement around the transferred/manual band.
        if model_key in {"gru4rec", "fame"}:
            return 1e-5, 1e-2
        if model_key in {"tisasrec", "sasrec", "bsarec", "difsr", "sigma"}:
            return 2e-5, 1e-2
        return 2e-5, 8e-3
    else:
        lo_mult = 1.10 if fast_screen else 1.05
        hi_mult = 1.18 if fast_screen else 1.12
    if model_key in {"gru4rec", "fame"}:
        lo_mult += 0.05
        hi_mult += 0.07
    new_lo = max(lr_floor, min(1e-2, float(lr_lo) * lo_mult))
    new_hi = max(new_lo * 1.05, min(1e-2, float(lr_hi) * hi_mult))
    return new_lo, new_hi


def _screen_variant_candidates(
    *,
    dataset: str,
    model_option: str,
    source_candidates: List[Dict[str, Any]],
    desired: int,
    followup_rescue: bool = False,
    pass3_expand: bool = False,
    lowperform2_screen: bool = False,
) -> List[Dict[str, Any]]:
    if desired <= 0:
        return []

    variants: List[Dict[str, Any]] = []
    lr_floor = _lr_floor(lowperform2_screen=lowperform2_screen)
    aggressive = (dataset, model_option) in AGGRESSIVE_LOW_PERF_COMBOS
    medium = (dataset, model_option) in MEDIUM_LOW_PERF_COMBOS
    for cand in source_candidates:
        if len(variants) >= desired:
            break
        cfg = sg._normalize_cfg(model_option, dict(cand["config"]))
        base_lo = float(cand["lr_lo"])
        base_hi = float(cand["lr_hi"])
        center = (base_lo * base_hi) ** 0.5

        recipes: List[Tuple[str, Dict[str, Any], float, float]] = []
        if model_option in {"gru4rec", "fame"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.03, float(rec_cfg.get("dropout", 0.08)) - 0.02)
            recipes.append(("LRHI", rec_cfg, center * 0.95, min(1e-2, base_hi * 1.18)))

            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = min(0.20, float(rec_cfg.get("dropout", 0.08)) + 0.03)
            recipes.append(("DROP", rec_cfg, max(8e-5, base_lo * 0.92), center * 1.05))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 5e-5)) * 0.7)
            recipes.append(("WDLO", rec_cfg, max(8e-5, center * 0.88), min(1e-2, base_hi * 1.08)))
            if aggressive:
                rec_cfg = dict(cfg)
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.08)) - 0.04)
                rec_cfg["max_len"] = min(int(rec_cfg.get("max_len", 10)), 8)
                recipes.append(("SHIFTUP", rec_cfg, min(1e-2, center * 1.3), min(1e-2, base_hi * 1.8)))

                rec_cfg = dict(cfg)
                rec_cfg["dropout"] = min(0.22, float(rec_cfg.get("dropout", 0.08)) + 0.05)
                rec_cfg["weight_decay"] = min(6e-4, float(rec_cfg.get("weight_decay", 5e-5)) * 2.2)
                recipes.append(("SHIFTDN", rec_cfg, max(lr_floor, base_lo * 0.45), max(lr_floor, center * 0.75)))
            if aggressive or medium:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(96, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.7 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 1 if aggressive else max(1, int(rec_cfg.get("layers", 2)) - 1)
                rec_cfg["num_layers"] = rec_cfg["layers"]
                recipes.append(("STRUCTS", rec_cfg, max(lr_floor, base_lo * 0.75), min(1e-2, center * 1.20)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(320, max(128, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.25 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = min(3, int(rec_cfg.get("layers", 2)) + (1 if aggressive else 0))
                rec_cfg["num_layers"] = rec_cfg["layers"]
                if model_option == "fame":
                    rec_cfg["num_experts"] = min(8, int(rec_cfg.get("num_experts", 3)) + 1)
                    rec_cfg["heads"] = min(8, max(4, int(rec_cfg.get("heads", 4))))
                    rec_cfg["inner_size"] = min(384, max(int(rec_cfg["hidden_size"] * 2), int(rec_cfg.get("inner_size", 256))))
                recipes.append(("STRUCTL", rec_cfg, max(lr_floor, center * 0.92), min(1e-2, base_hi * 1.35)))
        elif model_option in {"duorec", "fearec"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.05, float(rec_cfg.get("dropout", 0.10)) - 0.02)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 1e-4)) * 0.6)
            recipes.append(("LIGHT", rec_cfg, max(8e-5, base_lo * 0.95), center * 1.08))

            rec_cfg = dict(cfg)
            rec_cfg["tau"] = max(0.12, float(rec_cfg.get("tau", 0.2)) - 0.04)
            rec_cfg["lmd"] = max(0.01, float(rec_cfg.get("lmd", 0.02)) * 0.8)
            recipes.append(("TAU", rec_cfg, center * 0.92, min(1e-2, base_hi * 1.10)))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = min(3e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 1.5)
            recipes.append(("WDHI", rec_cfg, max(8e-5, base_lo * 0.90), center * 1.03))
            if aggressive or medium:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(96, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.75 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = max(192, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["heads"] = 2
                rec_cfg["layers"] = max(1, int(rec_cfg.get("layers", 2)) - 1)
                rec_cfg["num_layers"] = rec_cfg["layers"]
                recipes.append(("STRUCTS", rec_cfg, max(lr_floor, base_lo * 0.80), min(1e-2, center * 1.10)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(192, max(128, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.20 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = min(384, max(256, int(rec_cfg["hidden_size"] * 2.5)))
                rec_cfg["heads"] = min(4, max(2, int(rec_cfg.get("heads", 2)) + 1))
                rec_cfg["layers"] = min(3, int(rec_cfg.get("layers", 2)) + (1 if aggressive else 0))
                rec_cfg["num_layers"] = rec_cfg["layers"]
                if model_option == "fearec":
                    rec_cfg["global_ratio"] = min(0.9, float(rec_cfg.get("global_ratio", 0.7)) + 0.1)
                recipes.append(("STRUCTL", rec_cfg, max(lr_floor, center * 0.90), min(1e-2, base_hi * 1.25)))
        elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.06, float(rec_cfg.get("dropout", 0.10)) - 0.02)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 1e-4)) * 0.7)
            recipes.append(("REGLO", rec_cfg, max(8e-5, base_lo * 0.90), center * 1.06))

            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = min(0.20, float(rec_cfg.get("dropout", 0.10)) + 0.03)
            recipes.append(("REGHI", rec_cfg, center * 0.94, min(1e-2, base_hi * 1.08)))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = min(4e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 1.8)
            recipes.append(("WDHI", rec_cfg, max(8e-5, base_lo * 0.94), center * 1.04))
            if aggressive:
                rec_cfg = dict(cfg)
                rec_cfg["max_len"] = min(int(rec_cfg.get("max_len", 10)), 8)
                rec_cfg["dropout"] = max(0.03, float(rec_cfg.get("dropout", 0.10)) - 0.04)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = min(int(rec_cfg.get("time_span", 128)), 96)
                    rec_cfg["heads"] = min(int(rec_cfg.get("heads", 2)), 2)
                recipes.append(("SHORT", rec_cfg, min(1e-2, center * 1.15), min(1e-2, base_hi * 1.55)))

                rec_cfg = dict(cfg)
                rec_cfg["dropout"] = min(0.24, float(rec_cfg.get("dropout", 0.10)) + 0.06)
                rec_cfg["weight_decay"] = min(8e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 2.5)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = max(48, int(rec_cfg.get("time_span", 128)) // 2)
                recipes.append(("WIDE", rec_cfg, max(lr_floor, base_lo * 0.40), max(lr_floor, center * 0.72)))
            if aggressive or medium:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(96, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.75 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = max(192, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["layers"] = max(1, int(rec_cfg.get("layers", 2)) - 1)
                rec_cfg["num_layers"] = rec_cfg["layers"]
                rec_cfg["heads"] = 2 if model_option != "difsr" else max(1, min(2, int(rec_cfg.get("heads", 2))))
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = min(int(rec_cfg.get("time_span", 128)), 96)
                recipes.append(("STRUCTS", rec_cfg, max(lr_floor, base_lo * 0.78), min(1e-2, center * 1.12)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(192, max(128, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.20 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = min(384, max(256, int(rec_cfg["hidden_size"] * 2.5)))
                rec_cfg["layers"] = min(3, int(rec_cfg.get("layers", 2)) + (1 if aggressive else 0))
                rec_cfg["num_layers"] = rec_cfg["layers"]
                rec_cfg["heads"] = min(4, max(2, int(rec_cfg.get("heads", 2))))
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = min(384, max(128, int(rec_cfg.get("time_span", 128)) * 2))
                if model_option == "difsr":
                    rec_cfg["lambda_attr"] = min(0.2, float(rec_cfg.get("lambda_attr", 0.1)) + 0.04)
                recipes.append(("STRUCTL", rec_cfg, max(lr_floor, center * 0.90), min(1e-2, base_hi * 1.22)))
        elif model_option == "sigma":
            rec_cfg = dict(cfg)
            rec_cfg["hidden_size"] = max(96, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.75 / 8) * 8))
            rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
            rec_cfg["state_size"] = max(8, int(float(rec_cfg.get("sigma_state", rec_cfg.get("state_size", 16))) * 0.75))
            rec_cfg["conv_kernel"] = max(3, int(rec_cfg.get("sigma_kernel", rec_cfg.get("conv_kernel", 6))) - 2)
            rec_cfg["remaining_ratio"] = max(0.35, float(rec_cfg.get("sigma_remaining_ratio", rec_cfg.get("remaining_ratio", 0.6))) - 0.10)
            recipes.append(("STRUCTS", rec_cfg, max(lr_floor, base_lo * 0.82), min(1e-2, center * 1.10)))

            rec_cfg = dict(cfg)
            rec_cfg["hidden_size"] = min(192, max(128, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.20 / 8) * 8)))
            rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
            rec_cfg["state_size"] = min(48, max(16, int(float(rec_cfg.get("sigma_state", rec_cfg.get("state_size", 16))) * 1.5)))
            rec_cfg["conv_kernel"] = min(12, int(rec_cfg.get("sigma_kernel", rec_cfg.get("conv_kernel", 6))) + 2)
            rec_cfg["remaining_ratio"] = min(0.9, float(rec_cfg.get("sigma_remaining_ratio", rec_cfg.get("remaining_ratio", 0.6))) + 0.10)
            recipes.append(("STRUCTL", rec_cfg, max(lr_floor, center * 0.90), min(1e-2, base_hi * 1.22)))
        else:
            rec_cfg = dict(cfg)
            recipes.append(("LRMID", rec_cfg, max(8e-5, base_lo * 0.93), min(1e-2, base_hi * 1.08)))

        if followup_rescue or lowperform2_screen:
            if model_option in {"gru4rec", "fame"}:
                rec_cfg = dict(cfg)
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.08)) - 0.05)
                rec_cfg["hidden_size"] = min(384, max(128, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.45 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = min(3, int(rec_cfg.get("layers", 2)) + 1)
                rec_cfg["num_layers"] = rec_cfg["layers"]
                recipes.append(("PASS2_XLRHI", rec_cfg, min(1e-2, center * 1.35), min(1e-2, base_hi * 2.05)))

                rec_cfg = dict(cfg)
                rec_cfg["dropout"] = min(0.28, float(rec_cfg.get("dropout", 0.08)) + 0.08)
                rec_cfg["weight_decay"] = min(1e-3, float(rec_cfg.get("weight_decay", 5e-5)) * 3.2)
                recipes.append(("PASS2_XLRLO", rec_cfg, max(lr_floor, base_lo * 0.20), max(lr_floor, center * 0.55)))
            elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr", "sigma"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(80, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.62 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = max(160, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 48
                    rec_cfg["heads"] = 2
                recipes.append(("PASS2_TINY", rec_cfg, max(lr_floor, base_lo * 0.45), min(1e-2, center * 1.05)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(224, max(144, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.35 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = min(448, max(288, int(rec_cfg["hidden_size"] * 2.5)))
                rec_cfg["layers"] = min(3, int(rec_cfg.get("layers", 2)) + 1)
                rec_cfg["num_layers"] = rec_cfg["layers"]
                rec_cfg["dropout"] = max(0.04, float(rec_cfg.get("dropout", 0.10)) - 0.03)
                recipes.append(("PASS2_DEEPL", rec_cfg, max(lr_floor, center * 0.82), min(1e-2, base_hi * 1.45)))
            elif model_option in {"duorec", "fearec"}:
                rec_cfg = dict(cfg)
                rec_cfg["tau"] = max(0.08, float(rec_cfg.get("tau", 0.20)) - 0.07)
                rec_cfg["lmd"] = max(0.006, float(rec_cfg.get("lmd", 0.02)) * 0.55)
                rec_cfg["dropout"] = max(0.04, float(rec_cfg.get("dropout", 0.10)) - 0.03)
                recipes.append(("PASS2_CONTRASTLITE", rec_cfg, max(lr_floor, base_lo * 0.60), min(1e-2, base_hi * 1.30)))

                rec_cfg = dict(cfg)
                rec_cfg["tau"] = min(0.32, float(rec_cfg.get("tau", 0.20)) + 0.08)
                rec_cfg["lmd"] = min(0.05, float(rec_cfg.get("lmd", 0.02)) * 1.5)
                rec_cfg["weight_decay"] = min(5e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 2.2)
                recipes.append(("PASS2_CONTRASTREG", rec_cfg, max(lr_floor, base_lo * 0.50), min(1e-2, center * 1.15)))

        if pass3_expand or lowperform2_screen:
            if model_option in {"gru4rec", "fame"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(80, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.55 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["dropout"] = min(0.30, float(rec_cfg.get("dropout", 0.08)) + 0.10)
                recipes.append(("PASS3_ULTRATINY", rec_cfg, max(lr_floor, base_lo * 0.22), max(lr_floor, center * 0.55)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(448, max(160, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.65 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.08)) - 0.05)
                recipes.append(("PASS3_ULTRAWIDE", rec_cfg, min(1e-2, center * 1.55), min(1e-2, base_hi * 2.20)))
            elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr", "sigma"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(64, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.50 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["inner_size"] = max(128, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["dropout"] = max(0.03, float(rec_cfg.get("dropout", 0.10)) - 0.04)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 32
                    rec_cfg["heads"] = 2
                recipes.append(("PASS3_MINI", rec_cfg, max(lr_floor, base_lo * 0.30), min(1e-2, center * 0.95)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(256, max(160, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.55 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["inner_size"] = min(512, max(320, int(rec_cfg["hidden_size"] * 2.5)))
                rec_cfg["dropout"] = min(0.28, float(rec_cfg.get("dropout", 0.10)) + 0.02)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = min(512, max(192, int(rec_cfg.get("time_span", 128)) * 3))
                    rec_cfg["heads"] = 4
                recipes.append(("PASS3_MAXI", rec_cfg, max(8e-5, center * 0.78), min(1e-2, base_hi * 1.65)))

        if lowperform2_screen:
            if model_option in {"gru4rec", "fame"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(64, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.45 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["dropout"] = min(0.32, float(rec_cfg.get("dropout", 0.08)) + 0.12)
                rec_cfg["max_len"] = min(int(rec_cfg.get("max_len", 10)), 6)
                recipes.append(("PASS4_EDGELO", rec_cfg, max(lr_floor, base_lo * 0.10), max(lr_floor, center * 0.45)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(512, max(192, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.85 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.08)) - 0.06)
                recipes.append(("PASS4_EDGEHI", rec_cfg, min(1e-2, center * 1.75), min(1e-2, base_hi * 2.45)))
            elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr", "sigma"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(64, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.45 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["inner_size"] = max(128, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["max_len"] = min(int(rec_cfg.get("max_len", 10)), 6)
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.10)) - 0.05)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 24
                    rec_cfg["heads"] = 2
                recipes.append(("PASS4_EDGELO", rec_cfg, max(8e-5, base_lo * 0.24), max(8e-5, center * 0.60)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(288, max(192, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.75 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["inner_size"] = min(640, max(384, int(rec_cfg["hidden_size"] * 2.75)))
                rec_cfg["dropout"] = min(0.30, float(rec_cfg.get("dropout", 0.10)) + 0.03)
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = min(640, max(256, int(rec_cfg.get("time_span", 128)) * 4))
                    rec_cfg["heads"] = 4
                if model_option == "difsr":
                    rec_cfg["lambda_attr"] = min(0.25, float(rec_cfg.get("lambda_attr", 0.1)) + 0.06)
                recipes.append(("PASS4_EDGEHI", rec_cfg, max(lr_floor, center * 0.70), min(1e-2, base_hi * 1.90)))
            elif model_option in {"duorec", "fearec"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = max(80, int(round(float(rec_cfg.get("hidden_size", 128)) * 0.55 / 8) * 8))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = max(160, int(rec_cfg["hidden_size"] * 2))
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["tau"] = max(0.07, float(rec_cfg.get("tau", 0.20)) - 0.08)
                rec_cfg["lmd"] = max(0.004, float(rec_cfg.get("lmd", 0.02)) * 0.45)
                recipes.append(("PASS4_EDGELO", rec_cfg, max(lr_floor, base_lo * 0.12), max(lr_floor, center * 0.50)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = min(224, max(160, int(round(float(rec_cfg.get("hidden_size", 128)) * 1.45 / 8) * 8)))
                rec_cfg["embedding_size"] = rec_cfg["hidden_size"]
                rec_cfg["inner_size"] = min(448, max(320, int(rec_cfg["hidden_size"] * 2.8)))
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["tau"] = min(0.36, float(rec_cfg.get("tau", 0.20)) + 0.10)
                rec_cfg["lmd"] = min(0.06, float(rec_cfg.get("lmd", 0.02)) * 1.8)
                recipes.append(("PASS4_EDGEHI", rec_cfg, max(lr_floor, center * 0.74), min(1e-2, base_hi * 1.75)))

        if lowperform2_screen and (dataset, model_option) in CURRENT_ULTRA_LOW_TARGETS:
            if model_option in {"gru4rec", "fame"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 64
                rec_cfg["embedding_size"] = 64
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["dropout"] = min(0.35, float(rec_cfg.get("dropout", 0.08)) + 0.12)
                recipes.append(("PASS5_DIM64_XLO", rec_cfg, 1e-5, min(1e-2, center * 0.35)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 512
                rec_cfg["embedding_size"] = 512
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["dropout"] = max(0.02, float(rec_cfg.get("dropout", 0.08)) - 0.05)
                recipes.append(("PASS5_DIM512_XHI", rec_cfg, max(1e-4, center * 0.90), 1e-2))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 192
                rec_cfg["embedding_size"] = 192
                rec_cfg["layers"] = 2
                rec_cfg["num_layers"] = 2
                recipes.append(("PASS5_LRWIDE_MID", rec_cfg, 1e-5, 1e-2))
            elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr", "sigma"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 64
                rec_cfg["embedding_size"] = 64
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                rec_cfg["inner_size"] = 128
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 16
                    rec_cfg["heads"] = 2
                recipes.append(("PASS5_DIM64_XLO", rec_cfg, 1e-5, min(1e-2, center * 0.35)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 320
                rec_cfg["embedding_size"] = 320
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                rec_cfg["inner_size"] = 640
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 640
                    rec_cfg["heads"] = 4
                if model_option == "difsr":
                    rec_cfg["lambda_attr"] = min(0.30, float(rec_cfg.get("lambda_attr", 0.1)) + 0.08)
                recipes.append(("PASS5_DIM320_XHI", rec_cfg, max(1e-4, center * 0.90), 1e-2))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 160
                rec_cfg["embedding_size"] = 160
                rec_cfg["layers"] = 2
                rec_cfg["num_layers"] = 2
                rec_cfg["inner_size"] = 320
                if model_option == "tisasrec":
                    rec_cfg["time_span"] = 128
                    rec_cfg["heads"] = 2
                recipes.append(("PASS5_LRWIDE_MID", rec_cfg, 1e-5, 1e-2))
            elif model_option in {"duorec", "fearec"}:
                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 80
                rec_cfg["embedding_size"] = 80
                rec_cfg["inner_size"] = 160
                rec_cfg["layers"] = 1
                rec_cfg["num_layers"] = 1
                recipes.append(("PASS5_DIM80_XLO", rec_cfg, 1e-5, min(1e-2, center * 0.40)))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 256
                rec_cfg["embedding_size"] = 256
                rec_cfg["inner_size"] = 640
                rec_cfg["layers"] = 3
                rec_cfg["num_layers"] = 3
                recipes.append(("PASS5_DIM256_XHI", rec_cfg, max(1e-4, center * 0.92), 1e-2))

                rec_cfg = dict(cfg)
                rec_cfg["hidden_size"] = 128
                rec_cfg["embedding_size"] = 128
                rec_cfg["inner_size"] = 256
                rec_cfg["layers"] = 2
                rec_cfg["num_layers"] = 2
                recipes.append(("PASS5_LRWIDE_MID", rec_cfg, 1e-5, 8e-3))

        for suffix, rec_cfg, lr_lo, lr_hi in recipes:
            if len(variants) >= desired:
                break
            variants.append(
                {
                    "candidate_id": f"{cand['candidate_id']}_{suffix}",
                    "source": "stageh_screen_variant",
                    "source_stage": f"{cand.get('source_stage', 'stageh_screen')}_variant",
                    "transfer_from_dataset": cand.get("transfer_from_dataset", dataset),
                    "config": sg._normalize_cfg(model_option, rec_cfg),
                    "lr_lo": max(lr_floor, min(1e-2, float(lr_lo))),
                    "lr_hi": max(max(lr_floor, min(1e-2, float(lr_lo))) * 1.05, min(1e-2, float(lr_hi))),
                }
            )
    return variants[:desired]


def _model_runtime_resource_overrides_stageh(row: Dict[str, Any]) -> List[str]:
    if str(row.get("stage", "")).lower() != "stageh":
        return _orig_model_runtime_resource_overrides(row)

    model_option = str(row.get("model_option", "")).lower()
    h = base._effective_model_hparams(row)
    hidden_size = int(h.get("hidden_size", 128))
    fast_screen = bool(row.get("fast_screen", False))

    def _width_scale(bs: int) -> int:
        factor = 1.0
        if hidden_size >= 224:
            factor = 0.78
        elif hidden_size >= 192:
            factor = 0.88
        elif hidden_size >= 160:
            factor = 0.94
        return max(512, int(bs * factor))

    if model_option == "gru4rec":
        train_bs = 12288 if fast_screen else 10240
        eval_bs = 16384 if fast_screen else 12288
    elif model_option in {"sasrec", "bsarec", "difsr"}:
        train_bs = 6144 if fast_screen else 5120
        eval_bs = 8192 if fast_screen else 6144
    elif model_option == "tisasrec":
        train_bs = 2048 if fast_screen else 1536
        eval_bs = 3072 if fast_screen else 2304
    elif model_option == "sigma":
        train_bs = 2560 if fast_screen else 2048
        eval_bs = 3584 if fast_screen else 3072
    elif model_option == "duorec":
        train_bs = 3072 if fast_screen else 3584
        eval_bs = 8192 if fast_screen else 9216
    elif model_option == "fearec":
        train_bs = 4096 if fast_screen else 4608
        eval_bs = 8192 if fast_screen else 10240
    elif model_option == "fame":
        train_bs = 1792 if fast_screen else 1536
        eval_bs = 2816 if fast_screen else 2304
    else:
        return _orig_model_runtime_resource_overrides(row)

    train_bs = _width_scale(int(train_bs))
    eval_bs = _width_scale(int(eval_bs))
    return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]


base._model_runtime_resource_overrides = _model_runtime_resource_overrides_stageh


def _target_budget(
    model_option: str,
    *,
    fast_screen: bool,
    followup_rescue: bool = False,
    followup_full: bool = False,
    lowperform2_screen: bool = False,
    dataset: str | None = None,
) -> Tuple[int, int, int]:
    m = str(model_option).lower()
    if followup_full:
        if m in {"gru4rec", "fame"}:
            return 7, 40, 5
        if m in {"duorec", "fearec", "tisasrec"}:
            return 6, 38, 5
        return 6, 36, 5
    if lowperform2_screen:
        combo = (str(dataset), m)
        ultra = combo in CURRENT_ULTRA_LOW_TARGETS
        if ultra:
            if m in {"gru4rec", "fame"}:
                return 18, 18, 3
            if m in {"duorec", "fearec", "tisasrec"}:
                return 16, 18, 3
            return 16, 18, 3
        if m in {"gru4rec", "fame"}:
            return 14, 16, 3
        if m in {"duorec", "fearec", "tisasrec"}:
            return 12, 16, 3
        return 12, 16, 3
    if followup_rescue:
        tier = _effective_tier(str(dataset), m)
        if tier == "hard":
            if m in {"gru4rec", "fame"}:
                return 5, 26, 4
            return 5, 24, 4
        if tier == "medium":
            if m in {"gru4rec", "fame"}:
                return 5, 24, 4
            return 4, 22, 3
        return 4, 20, 3
    if fast_screen:
        tier = _effective_tier(str(dataset), m)
        if tier == "hard":
            if m in {"gru4rec", "fame"}:
                return 5, 24, 4
            if m in {"duorec", "fearec", "tisasrec"}:
                return 4, 22, 3
            return 4, 22, 3
        if tier == "medium":
            if m in {"gru4rec", "fame"}:
                return 4, 22, 3
            return 4, 18, 3
        if m in {"duorec", "fearec"}:
            return 3, 18, 2
        return 3, 16, 2
    if m in {"gru4rec", "fame"}:
        return 6, 40, 5
    if m in {"duorec", "fearec"}:
        return 5, 36, 5
    return 5, 34, 5


def _is_lowperform2_wide_lr_candidate(candidate_id: str, lr_lo: float, lr_hi: float) -> bool:
    cid = str(candidate_id).upper()
    if "LRWIDE" in cid or "XLO" in cid or "XHI" in cid:
        return True
    if float(lr_lo) <= 2e-5 and float(lr_hi) >= 8e-3:
        return True
    return (float(lr_hi) / max(float(lr_lo), 1e-12)) >= 200.0


def _selected_combos(args: argparse.Namespace) -> List[Tuple[str, str]]:
    datasets = set(base._parse_csv_strings(args.datasets))
    models = set(m.lower() for m in base._parse_csv_strings(args.models))
    combo_source = set(TARGET_COMBOS)
    if (args.underperform_screen or args.lowperform2_screen) and CURRENT_DYNAMIC_TARGETS:
        combo_source = CURRENT_DYNAMIC_TARGETS
    selected = [(ds, m) for ds, m in combo_source if ds in datasets and m in models]
    promoted = set()
    if args.followup_full and args.promote_from_latest_rescue:
        promoted = _auto_promoted_rescue_combos(
            promote_topk=int(args.promote_topk),
            min_ratio=float(args.promote_min_ratio),
        )
    if args.followup_full and args.followup_rescue:
        selected = [c for c in selected if c in FULL_SHORTLIST_COMBOS or c in RESCUE_REDESIGN_COMBOS or c in promoted]
    elif args.followup_full:
        selected = [c for c in selected if c in FULL_SHORTLIST_COMBOS or c in promoted]
    elif args.followup_rescue:
        selected = [c for c in selected if c in RESCUE_REDESIGN_COMBOS]
    return selected


def _select_stageh_candidates(
    *,
    dataset: str,
    model_option: str,
    model_label: str,
    final_cache_by_dataset: Dict[str, Dict[str, List[Dict[str, Any]]]],
    global_cache_by_model: Dict[str, List[Dict[str, Any]]],
    stage_best_cache: Dict[Tuple[str, str], Dict[str, Any]],
    fast_screen: bool,
    followup_rescue: bool,
    followup_full: bool,
    underperform_screen: bool,
    lowperform2_screen: bool,
) -> List[Dict[str, Any]]:
    tier = _effective_tier(dataset, model_option)
    dummy_args = argparse.Namespace(
        fast_screen=fast_screen,
        followup_full=followup_full,
        followup_rescue=followup_rescue,
        underperform_screen=underperform_screen,
        lowperform2_screen=lowperform2_screen,
    )
    goal = _candidate_goal(dataset, model_option, args=dummy_args)
    base_cands = sg._select_candidates_for_combo(
        dataset=dataset,
        model_option=model_option,
        model_label=model_label,
        final_cache_by_dataset=final_cache_by_dataset,
        global_cache_by_model=global_cache_by_model,
        stage_best_cache=stage_best_cache,
    )
    manual_cands = _manual_candidates(dataset, model_option)

    chosen: List[Dict[str, Any]] = []
    if followup_full:
        chosen.extend(dict(c) for c in manual_cands[:2])
        chosen.extend(dict(c) for c in base_cands[:2])
    elif fast_screen or followup_rescue or lowperform2_screen:
        base_take = 8 if lowperform2_screen and (dataset, model_option) in CURRENT_ULTRA_LOW_TARGETS else 6 if tier == "hard" else 4 if tier == "medium" else 3
        manual_take = 8 if lowperform2_screen and (dataset, model_option) in CURRENT_ULTRA_LOW_TARGETS else 6 if tier == "hard" else 4 if tier == "medium" else 3
        if followup_rescue or lowperform2_screen:
            base_take += 1
            manual_take += 1
        chosen.extend(dict(c) for c in base_cands[: min(base_take, len(base_cands))])
        chosen.extend(manual_cands[: min(manual_take, len(manual_cands))])
    elif goal >= 3 and manual_cands:
        chosen.extend(manual_cands[:2])
        if base_cands:
            chosen.append(dict(base_cands[0]))
    else:
        if base_cands:
            chosen.append(dict(base_cands[0]))
        if manual_cands:
            chosen.append(manual_cands[0])
        elif len(base_cands) >= 2:
            chosen.append(dict(base_cands[1]))

    for cand in manual_cands[2:]:
        if len(chosen) >= goal:
            break
        chosen.append(cand)
    for cand in base_cands[1:]:
        if len(chosen) >= goal:
            break
        chosen.append(dict(cand))

    chosen = sg._dedup_candidates(chosen)
    if (fast_screen or followup_rescue or lowperform2_screen) and len(chosen) < goal:
        variant_source = list(chosen)
        source_take = 6 if lowperform2_screen else 4
        if len(variant_source) < source_take:
            variant_source.extend(dict(c) for c in base_cands[:source_take])
            variant_source.extend(dict(c) for c in manual_cands[:source_take])
        chosen.extend(
            _screen_variant_candidates(
                dataset=dataset,
                model_option=model_option,
                source_candidates=sg._dedup_candidates(variant_source),
                desired=max(0, goal - len(chosen)),
                followup_rescue=bool(followup_rescue or lowperform2_screen),
                pass3_expand=bool(lowperform2_screen or (underperform_screen and (dataset, model_option) in TIER_BY_COMBO)),
                lowperform2_screen=bool(lowperform2_screen),
            )
        )
        chosen = sg._dedup_candidates(chosen)
    while len(chosen) < goal:
        chosen.append(sg._fallback_candidate(dataset, model_option, len(chosen) + 1))
    return chosen[:goal]


def _build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    seeds = base._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    final_cache_by_dataset, global_cache_by_model = sg._load_final_hparam_cache()
    stage_best_cache = sg._load_stage_best_cache()
    combos = _selected_combos(args)
    if not combos:
        raise RuntimeError("No target combos selected")

    rows: List[Dict[str, Any]] = []
    run_cursor = 0
    candidate_tag = str(getattr(args, "candidate_tag", "") or "").strip()

    for combo_idx, (dataset, model_option) in enumerate(combos, start=1):
        model_label = sg.OPTION_TO_LABEL[model_option]
        candidates = _select_stageh_candidates(
            dataset=dataset,
            model_option=model_option,
            model_label=model_label,
            final_cache_by_dataset=final_cache_by_dataset,
            global_cache_by_model=global_cache_by_model,
            stage_best_cache=stage_best_cache,
            fast_screen=bool(args.fast_screen or args.underperform_screen or args.lowperform2_screen),
            followup_rescue=bool(args.followup_rescue),
            followup_full=bool(args.followup_full),
            underperform_screen=bool(args.underperform_screen),
            lowperform2_screen=bool(args.lowperform2_screen),
        )
        for cand_idx, cand in enumerate(candidates, start=1):
            candidate_id = str(cand["candidate_id"])
            if candidate_tag:
                candidate_id = f"{candidate_id}_{candidate_tag}"
            max_evals, tune_epochs, tune_patience = _target_budget(
                model_option,
                fast_screen=bool(args.fast_screen),
                followup_rescue=bool(args.followup_rescue),
                followup_full=bool(args.followup_full),
                lowperform2_screen=bool(args.lowperform2_screen),
                dataset=dataset,
            )
            if args.smoke_test:
                max_evals, tune_epochs, tune_patience = 1, 1, 1
            candidate_config = _speedify_candidate_config(
                dataset=dataset,
                model_option=model_option,
                cfg=dict(cand["config"]),
                fast_screen=bool(args.fast_screen or args.lowperform2_screen),
                lowperform2_screen=bool(args.lowperform2_screen),
            )
            lr_lo, lr_hi = _speedify_lr_band(
                model_option=model_option,
                lr_lo=float(cand["lr_lo"]),
                lr_hi=float(cand["lr_hi"]),
                fast_screen=bool(args.fast_screen),
                lowperform2_screen=bool(args.lowperform2_screen),
            )
            if args.lowperform2_screen:
                if _is_lowperform2_wide_lr_candidate(str(cand["candidate_id"]), float(lr_lo), float(lr_hi)):
                    max_evals = int(args.lowperform2_wide_max_evals)
                else:
                    max_evals = int(args.lowperform2_regular_max_evals)

            for seed_id in seeds:
                run_cursor += 1
                row = {
                    "dataset": dataset,
                    "phase_id": base.PHASE_ID,
                    "axis_id": "SH",
                    "axis_desc": base.AXIS_DESC,
                    "setting_id": f"STAGEH_{base._sanitize_token(model_label, upper=True)}_{base._sanitize_token(candidate_id, upper=True)}_S{seed_id}",
                    "setting_key": "STAGEH_TARGETED_RECOVERY",
                    "setting_desc": f"STAGEH_TARGETED_RECOVERY_{base._sanitize_token(model_label, upper=True)}_{base._sanitize_token(candidate_id, upper=True)}_S{seed_id}",
                    "hparam_id": base._sanitize_token(candidate_id, upper=True)[:40],
                    "seed_id": int(seed_id),
                    "run_phase": (
                        f"{base.PHASE_ID}_SH_C{combo_idx:02d}_M{base._sanitize_token(model_label, upper=True)}_"
                        f"{base._sanitize_token(candidate_id, upper=True)}_S{int(seed_id)}"
                    ),
                    "run_id": (
                        f"SH_{base._sanitize_token(dataset, upper=True)}_{base._sanitize_token(model_label, upper=True)}_"
                        f"{base._sanitize_token(candidate_id, upper=True)}_S{int(seed_id)}"
                    ),
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "stage": "stageh",
                    "model_option": model_option,
                    "model_label": model_label,
                    "candidate_id": candidate_id,
                    "candidate_source": str(cand["source"]),
                    "source_stage": str(cand["source_stage"]),
                    "transfer_from_dataset": str(cand["transfer_from_dataset"]),
                    "candidate_config": candidate_config,
                    "lr_lo": float(lr_lo),
                    "lr_hi": float(lr_hi),
                    "max_evals": int(max_evals),
                    "tune_epochs": int(tune_epochs),
                    "tune_patience": int(tune_patience),
                    "fast_screen": bool(args.fast_screen or args.lowperform2_screen),
                }
                row["estimated_cost"] = sg._stageg_estimated_cost(row)
                rows.append(row)
    return rows


def _load_prior_best_by_combo() -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    path = PRIOR_BEST_TABLE_PATH
    if not path.exists():
        return out
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                ds = str(row.get("dataset", "")).strip()
                model = str(row.get("model", "")).strip().lower()
                if not ds or not model:
                    continue
                val = sg._metric(row.get("best_valid_mrr20"))
                if val is not None:
                    out[(ds, model)] = float(val)
    except Exception:
        return {}
    return out


def _load_stageh_best_by_combo() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    root = base.LOG_ROOT / base.AXIS
    if not root.exists():
        return out
    for dsdir in root.iterdir():
        if not dsdir.is_dir():
            continue
        summary = dsdir / "summary.csv"
        if not summary.exists():
            continue
        try:
            with summary.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception:
            continue
        for row in rows:
            ds = str(row.get("dataset", "")).strip()
            model_label = str(row.get("model", "")).strip()
            model_option = sg.LABEL_TO_OPTION.get(model_label)
            if not ds or model_option is None:
                continue
            valid = sg._metric(row.get("run_best_valid_mrr20"))
            test = sg._metric(row.get("run_best_test_mrr20"))
            if valid is None:
                continue
            key = (ds, model_option)
            prev = out.get(key)
            if prev is None or float(valid) > float(prev["valid"]):
                out[key] = {
                    "valid": float(valid),
                    "test": None if test is None else float(test),
                    "candidate_id": str(row.get("candidate_id", "")),
                    "run_phase": str(row.get("run_phase", "")),
                }
    return out


def _auto_promoted_rescue_combos(*, promote_topk: int, min_ratio: float) -> set[Tuple[str, str]]:
    prior = _load_prior_best_by_combo()
    now = _load_stageh_best_by_combo()
    scored: List[Tuple[float, Tuple[str, str]]] = []
    for combo in RESCUE_REDESIGN_COMBOS:
        rec = now.get(combo)
        if not isinstance(rec, dict):
            continue
        valid = sg._metric(rec.get("valid"))
        test = sg._metric(rec.get("test"))
        if valid is None:
            continue
        prev = prior.get(combo)
        ratio = None if prev in (None, 0.0) else float(valid) / float(prev)
        if prev is not None and ratio is not None and ratio < float(min_ratio) and float(valid) < float(prev):
            continue
        score = float(valid) + 0.20 * float(test or 0.0)
        if prev is not None:
            score += 0.50 * (float(valid) - float(prev))
        scored.append((score, combo))
    scored.sort(reverse=True)
    return {combo for _, combo in scored[: max(0, int(promote_topk))]}




def _combo_plan_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    combos = _selected_combos(args)
    for dataset, model_option in combos:
        tier = _effective_tier(dataset, model_option)
        cand_count = _candidate_goal(dataset, model_option, args=args)
        seeds = len(base._parse_csv_ints(args.seeds))
        rows.append({
            "dataset": dataset,
            "model_option": model_option,
            "model_label": sg.OPTION_TO_LABEL[model_option],
            "tier": tier,
            "candidate_count": cand_count,
            "seed_count": seeds,
            "run_count": cand_count * seeds,
        })
    return rows


def _print_stageh_plan(args: argparse.Namespace) -> None:
    plan = _combo_plan_rows(args)
    by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for row in plan:
        by_dataset.setdefault(str(row["dataset"]), []).append(row)
    print(f"[{base.PHASE_ID}] combo_count={len(plan)} seed_count={len(base._parse_csv_ints(args.seeds))} total_runs={sum(int(r['run_count']) for r in plan)}")
    if args.followup_full and args.promote_from_latest_rescue:
        promoted = sorted(_auto_promoted_rescue_combos(promote_topk=int(args.promote_topk), min_ratio=float(args.promote_min_ratio)))
        if promoted:
            print("[plan] promoted_from_rescue: " + ", ".join(f"{ds}/{sg.OPTION_TO_LABEL[m]}" for ds, m in promoted))
    for dataset, items in by_dataset.items():
        items.sort(key=lambda r: (str(r["tier"]), str(r["model_label"])))
        summary = ", ".join(
            f"{r['model_label']}[{r['tier']}:c{r['candidate_count']}/r{r['run_count']}]" for r in items
        )
        print(f"[plan] {dataset}: {summary}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P23 StageH targeted recovery launcher")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--seed-base", type=int, default=230000)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=8)
    parser.add_argument("--fast-screen", action="store_true")
    parser.add_argument("--followup-rescue", action="store_true")
    parser.add_argument("--followup-full", action="store_true")
    parser.add_argument("--underperform-screen", action="store_true")
    parser.add_argument("--lowperform2-screen", action="store_true")
    parser.add_argument("--weak-ratio", type=float, default=0.90)
    parser.add_argument("--strong-ratio", type=float, default=0.75)
    parser.add_argument("--lowperform2-ratio", type=float, default=0.80)
    parser.add_argument("--lowperform2-ultra-ratio", type=float, default=0.50)
    parser.add_argument("--lowperform2-regular-max-evals", type=int, default=15)
    parser.add_argument("--lowperform2-wide-max-evals", type=int, default=50)
    parser.add_argument("--candidate-tag", default="")
    parser.add_argument("--promote-from-latest-rescue", action="store_true")
    parser.add_argument("--promote-topk", type=int, default=4)
    parser.add_argument("--promote-min-ratio", type=float, default=0.95)
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"
    gpus = base._parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"


def _apply_fast_screen_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"


def _apply_followup_modes(args: argparse.Namespace) -> None:
    args.seeds = "1"


def _apply_underperform_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"


def _apply_lowperform2_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"


def _run(args: argparse.Namespace) -> int:
    gpus = list(dict.fromkeys(base._parse_csv_strings(args.gpus)))
    rows = _build_rows(args)
    if args.smoke_test:
        rows = rows[: max(1, int(args.smoke_max_runs))]

    dataset_labels = ",".join(sorted(dict.fromkeys(r["dataset"] for r in rows)))
    summary_paths = {dataset: base._summary_path(dataset) for dataset in sorted(dict.fromkeys(r["dataset"] for r in rows))}
    for path in summary_paths.values():
        base._ensure_summary_csv(path)
    summary_state = {
        dataset: list(base._load_summary_bests(path))
        for dataset, path in summary_paths.items()
    }
    completed_keys_by_dataset: Dict[str, Set[Tuple[str, str, str]]] = {}
    for dataset, path in summary_paths.items():
        done_keys: Set[Tuple[str, str, str]] = set()
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                for rec in csv.DictReader(fh):
                    if str(rec.get("status", "")).strip().lower() != "run_complete":
                        continue
                    model = str(rec.get("model", "")).strip()
                    candidate_id = str(rec.get("candidate_id", "")).strip()
                    seed_id = str(rec.get("seed_id", "")).strip()
                    if model and candidate_id and seed_id:
                        done_keys.add((model, candidate_id, seed_id))
        except Exception:
            pass
        completed_keys_by_dataset[dataset] = done_keys

    skipped_rows: List[Dict[str, Any]] = []
    runnable: List[Dict[str, Any]] = []
    result_indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        resume_key = (
            str(row.get("model_label", "")).strip(),
            str(row.get("candidate_id", "")).strip(),
            str(row.get("seed_id", "")).strip(),
        )
        if resume_key in completed_keys_by_dataset.get(dataset, set()):
            skipped_rows.append(row)
            continue
        completed = base._is_completed_any_log(row, use_resume=bool(args.resume_from_logs))
        if not completed:
            runnable.append(row)
            continue
        if not args.verify_logging:
            skipped_rows.append(row)
            continue
        if dataset not in result_indexes:
            result_indexes[dataset] = base._scan_result_index(dataset)
        rec = result_indexes[dataset].get(str(row["run_phase"]))
        if not rec:
            runnable.append(row)
            continue
        ok, detail = base._verify_special_from_result(str(rec.get("path", "")))
        if ok:
            skipped_rows.append(row)
            continue
        print(f"[resume-check] run={row['run_phase']} special_check_failed -> rerun ({detail})")
        runnable.append(row)

    for row in rows:
        row["log_path"] = str(base._build_log_path(row))

    manifest_path = base._manifest_path(dataset_labels.replace(",", "_"), args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base._summary_fieldnames = sg._summary_fieldnames
    sg._write_manifest(dataset_labels.replace(",", "_"), args, rows)

    runnable.sort(key=lambda r: float(r.get("estimated_cost", 1.0)), reverse=True)
    print(
        f"[{base.PHASE_ID}] combos={len(_selected_combos(args))} planned_runs={len(rows)} "
        f"runnable_runs={len(runnable)} skipped_runs={len(skipped_rows)} gpus={','.join(gpus)}"
    )

    if args.dry_run:
        gpu_bins = base._plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = sg._build_command(row, gpu_id, args)
                print(
                    f"[dry-run] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} candidate={row['candidate_id']} seed={row['seed_id']}"
                )
                print("          " + " ".join(cmd))
        return 0

    notifier = SlackProgressNotifier(phase_label="baseline StageH", rows=rows)
    notifier.notify_plan(precompleted_rows=skipped_rows)

    active: Dict[str, Dict[str, Any]] = {}
    shared_queue: deque = deque(runnable)
    try:
        while True:
            for gpu_id in gpus:
                if gpu_id in active or not shared_queue:
                    continue
                row = shared_queue.popleft()
                cmd = sg._build_command(row, gpu_id, args)
                log_path = Path(str(row["log_path"]))
                sg._write_log_preamble(log_path, row, gpu_id, args, cmd)
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("HYPEROPT_RESULTS_DIR", str(base.ARTIFACT_ROOT / "results"))
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                with log_path.open("a", encoding="utf-8") as fh:
                    proc = subprocess.Popen(cmd, cwd=base.EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
                active[gpu_id] = {"proc": proc, "row": row, "log_path": str(log_path)}
                print(f"[launch] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} model={row['model_label']}")

            done_gpus: List[str] = []
            for gpu_id, slot in active.items():
                rc = slot["proc"].poll()
                if rc is None:
                    continue
                done_gpus.append(gpu_id)
                row = slot["row"]
                print(f"[done] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} rc={rc}")
                rec = base._get_result_row_from_log_or_scan(
                    dataset=row["dataset"],
                    run_phase=str(row["run_phase"]),
                    log_path=Path(str(slot["log_path"])),
                    retries=4,
                    sleep_sec=0.75,
                )
                run_best = None
                test_mrr = None
                n_completed = None
                interrupted = None
                result_path = ""
                special_ok = False
                if rec:
                    run_best = sg._metric(rec.get("best_mrr"))
                    test_mrr = sg._metric(rec.get("test_mrr"))
                    n_completed = int(rec.get("n_completed", 0) or 0)
                    interrupted = bool(rec.get("interrupted", False))
                    result_path = str(rec.get("path", "") or "")
                    if result_path:
                        special_ok, detail = base._verify_special_from_result(result_path)
                        print(f"[logging-check] run={row['run_phase']} {detail} result={result_path}")
                        base._mirror_logging_bundle(row, result_path)
                global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model = summary_state[row["dataset"]]
                if run_best is not None:
                    global_best_valid = run_best if global_best_valid is None else max(float(global_best_valid), float(run_best))
                    prev_model_best = model_best_valid_by_model.get(str(row["model_label"]))
                    model_best_valid_by_model[str(row["model_label"])] = run_best if prev_model_best is None else max(float(prev_model_best), float(run_best))
                if test_mrr is not None:
                    global_best_test = test_mrr if global_best_test is None else max(float(global_best_test), float(test_mrr))
                    prev_model_test = model_best_test_by_model.get(str(row["model_label"]))
                    model_best_test_by_model[str(row["model_label"])] = test_mrr if prev_model_test is None else max(float(prev_model_test), float(test_mrr))
                current_model_best_valid = model_best_valid_by_model.get(str(row["model_label"]))
                current_model_best_test = model_best_test_by_model.get(str(row["model_label"]))
                summary_state[row["dataset"]] = [global_best_valid, global_best_test, model_best_valid_by_model, model_best_test_by_model]
                summary_row = {
                    "model": row["model_label"],
                    "global_best_valid_mrr20": "" if global_best_valid is None else f"{float(global_best_valid):.6f}",
                    "global_best_test_mrr20": "" if global_best_test is None else f"{float(global_best_test):.6f}",
                    "model_best_valid_mrr20": "" if current_model_best_valid is None else f"{float(current_model_best_valid):.6f}",
                    "model_best_test_mrr20": "" if current_model_best_test is None else f"{float(current_model_best_test):.6f}",
                    "run_best_valid_mrr20": "" if run_best is None else f"{float(run_best):.6f}",
                    "run_best_test_mrr20": "" if test_mrr is None else f"{float(test_mrr):.6f}",
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "hparam_id": row["hparam_id"],
                    "seed_id": row["seed_id"],
                    "gpu_id": gpu_id,
                    "status": "run_complete" if int(rc) == 0 else "run_fail",
                    "n_completed": "" if n_completed is None else int(n_completed),
                    "interrupted": "" if interrupted is None else bool(interrupted),
                    "special_ok": bool(special_ok),
                    "candidate_id": row["candidate_id"],
                    "candidate_source": row["candidate_source"],
                    "source_stage": row["source_stage"],
                    "transfer_from_dataset": row["transfer_from_dataset"],
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                base._append_summary_row(summary_paths[row["dataset"]], summary_row)
                notifier.mark_complete(row)
                if int(rc) != 0:
                    raise RuntimeError(f"run failed: dataset={row['dataset']} run_phase={row['run_phase']} rc={rc}")

            for gpu_id in done_gpus:
                active.pop(gpu_id, None)
            if not shared_queue and not active:
                break
            time.sleep(1)
    except Exception:
        base._terminate_active(active)
        raise
    return 0


def main() -> int:
    global CURRENT_DYNAMIC_TIER_BY_COMBO, CURRENT_DYNAMIC_TARGETS, CURRENT_ULTRA_LOW_TARGETS
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)
    elif args.lowperform2_screen:
        _apply_lowperform2_mode(args)
    elif args.underperform_screen:
        _apply_underperform_mode(args)
    elif args.followup_rescue or args.followup_full:
        _apply_followup_modes(args)
    elif args.fast_screen:
        _apply_fast_screen_mode(args)
    if args.lowperform2_screen:
        CURRENT_DYNAMIC_TIER_BY_COMBO, CURRENT_ULTRA_LOW_TARGETS = _compute_lowperform2_targets(
            low_ratio=float(args.lowperform2_ratio),
            ultra_ratio=float(args.lowperform2_ultra_ratio),
        )
        CURRENT_DYNAMIC_TARGETS = set(CURRENT_DYNAMIC_TIER_BY_COMBO.keys())
    elif args.underperform_screen:
        CURRENT_DYNAMIC_TIER_BY_COMBO = _compute_underperform_targets(
            weak_ratio=float(args.weak_ratio),
            strong_ratio=float(args.strong_ratio),
        )
        CURRENT_DYNAMIC_TARGETS = set(CURRENT_DYNAMIC_TIER_BY_COMBO.keys())
        CURRENT_ULTRA_LOW_TARGETS = set()
    else:
        CURRENT_DYNAMIC_TIER_BY_COMBO = {}
        CURRENT_DYNAMIC_TARGETS = set()
        CURRENT_ULTRA_LOW_TARGETS = set()
    runtime_models = sorted(dict.fromkeys(m.lower() for _, m in (_selected_combos(args) or TARGET_COMBOS)))
    sg._check_runtime_models(runtime_models)
    _print_stageh_plan(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
