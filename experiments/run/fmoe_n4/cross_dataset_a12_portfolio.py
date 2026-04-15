#!/usr/bin/env python3
"""FMoE_N4: cross-dataset A12 portfolio tuning outside KuaiRec.

This launcher is meant for fast transfer-style screening on non-Kuai datasets.
It mixes:
- Kuai Stage1/2 winners that looked broadly stable
- older v3/FMoE anchors that were strong on each target dataset
- a few dataset-specific exploratory probes inspired by baseline recovery logs

Defaults are intentionally uneven by dataset:
- amazon_beauty: 12 templates
- foursquare: 12 templates
- movielens1m: 6 templates
- retail_rocket: 6 templates
- lastfm0.03: 4 templates

All runs use 100 epochs, but patience is kept short to control wall-clock.
Main reporting remains seen-target-first via the shared Stage1 command builder.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import stage1_a12_broad_templates as stage1

TRACK = stage1.TRACK
AXIS = "CrossDataset_A12_Portfolio"
AXIS_ID = "N4XDA12"
AXIS_DESC = "cross_dataset_a12_portfolio"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "P4XD"
PHASE_NAME = "FMOE_N4_CROSS_DATASET_A12_PORTFOLIO"
DEFAULT_DATASETS = ["amazon_beauty", "foursquare", "movielens1m", "retail_rocket", "lastfm0.03"]

REPO_ROOT_REAL = stage1.REPO_ROOT_REAL
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

stage1.AXIS = AXIS
stage1.AXIS_ID = AXIS_ID
stage1.AXIS_DESC = AXIS_DESC
stage1.PHASE_ID = PHASE_ID
stage1.PHASE_NAME = PHASE_NAME
stage1.LOG_ROOT = LOG_ROOT

DEFAULT_TEMPLATE_COUNTS = {
    "amazon_beauty": 12,
    "foursquare": 12,
    "movielens1m": 6,
    "retail_rocket": 6,
    "lastfm0.03": 4,
}

DATASET_BUDGETS = {
    "amazon_beauty": {"max_evals": 12, "patience": 6},
    "foursquare": {"max_evals": 12, "patience": 6},
    "movielens1m": {"max_evals": 9, "patience": 5},
    "retail_rocket": {"max_evals": 8, "patience": 5},
    "lastfm0.03": {"max_evals": 6, "patience": 4},
}


def _stage_choice(values: list[float]) -> Dict[str, Any]:
    return stage1._choice_spec(stage1._all_stage_map(float(value)) for value in values)


def _dataset_counts(raw: str) -> Dict[str, int]:
    counts = dict(DEFAULT_TEMPLATE_COUNTS)
    text = str(raw or "").strip()
    if not text:
        return counts
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise RuntimeError(f"invalid dataset template count override: {token}")
        dataset, value = token.split(":", 1)
        dataset = dataset.strip()
        if dataset not in counts:
            raise RuntimeError(f"unknown dataset in template count override: {dataset}")
        counts[dataset] = int(value.strip())
    return counts


def _dataset_budget(dataset: str) -> Dict[str, int]:
    if dataset not in DATASET_BUDGETS:
        raise RuntimeError(f"missing dataset budget config for {dataset}")
    return dict(DATASET_BUDGETS[dataset])


def _choose_batch(dataset: str, template: Dict[str, Any], fixed_values: Dict[str, Any], args: argparse.Namespace) -> tuple[int, int, int]:
    train_batch_size, eval_batch_size, max_evals = stage1._template_batches(template, fixed_values, args)
    budget = _dataset_budget(dataset)
    capped_evals = min(int(max_evals), int(budget["max_evals"]))

    if dataset == "lastfm0.03":
        train_batch_size = min(int(train_batch_size), 2048)
        eval_batch_size = min(int(eval_batch_size), 4096)
    elif dataset == "retail_rocket":
        train_batch_size = min(int(train_batch_size), 3072)
        eval_batch_size = min(int(eval_batch_size), 4096)
    elif dataset == "movielens1m":
        train_batch_size = min(int(train_batch_size), 4096)
        eval_batch_size = min(int(eval_batch_size), 6144)
    else:
        train_batch_size = min(int(train_batch_size), 4096)
        eval_batch_size = min(int(eval_batch_size), 6144)
    return int(train_batch_size), int(eval_batch_size), int(capped_evals)


def _bank_amazon_beauty() -> list[Dict[str, Any]]:
    return [
        {
            "id": "A01_h8_compact_anchor",
            "band": "exploit",
            "source": "v3_amazon_default_outlier+compact_transfer",
            "selection_score": "amazon fast compact anchor",
            "anchor": "H8",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.8e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12, 16]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        },
        {
            "id": "A02_h4_tiny_reg",
            "band": "exploit",
            "source": "compact_amazon_probe",
            "selection_score": "tiny regularized amazon probe",
            "anchor": "H4",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.6e-4, 5.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04, 0.06]),
            },
        },
        {
            "id": "A03_h9_micro_reg",
            "band": "exploit",
            "source": "compact_amazon_probe",
            "selection_score": "ultracompact amazon probe",
            "anchor": "H9",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.4e-4, 4.8e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.14],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10, 0.20]),
            },
        },
        {
            "id": "A04_h12_microtiny",
            "band": "exploit",
            "source": "v3_amazon_small_bank",
            "selection_score": "very small amazon control",
            "anchor": "H12",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.4e-4, 4.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.14],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "stage_feature_dropout_prob": _stage_choice([0.03, 0.05]),
            },
        },
        {
            "id": "A05_h14_capacity_transfer",
            "band": "transfer",
            "source": "kuai_h14_capacity",
            "selection_score": "capacity transfer from Kuai",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.8e-4, 5.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16, 20]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
            },
        },
        {
            "id": "A06_h16_width_tune",
            "band": "explore",
            "source": "v3_amazon_h16",
            "selection_score": "amazon width/depth mix",
            "anchor": "H16",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.2e-4, 8.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16, 24]),
                "expert_scale": stage1._choice_spec([2, 3]),
            },
        },
        {
            "id": "A07_h10_len25",
            "band": "transfer",
            "source": "kuai_h10_context",
            "selection_score": "context transfer for amazon",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.8e-4, 6.5e-4),
            "len": 25,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25, 30]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        },
        {
            "id": "A08_h6_e2_compact",
            "band": "transfer",
            "source": "kuai_h6_e2_core",
            "selection_score": "compact stable transfer",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 7.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([1, 2]),
            },
        },
        {
            "id": "A09_h2_regularized",
            "band": "transfer",
            "source": "kuai_h2_regularized+amazon_transfer",
            "selection_score": "regularized low-cap transfer",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.8e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
                "z_loss_lambda": stage1._choice_spec([1e-4, 2e-4]),
            },
        },
        {
            "id": "A10_h3_regularized",
            "band": "transfer",
            "source": "v3_compact_reg",
            "selection_score": "compact regularized backup",
            "anchor": "H3",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03, 0.05]),
            },
        },
        {
            "id": "A11_dropout_sched_probe",
            "band": "new_axis",
            "source": "baseline_reg_relief_style",
            "selection_score": "dropout+scheduler exploration",
            "anchor": "H8",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.8e-4, 7.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0, 0.02, 0.04]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03, 0.05]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10, 0.20]),
            },
        },
        {
            "id": "A12_auxwide_probe",
            "band": "new_axis",
            "source": "baseline_special_probe",
            "selection_score": "leave aux terms somewhat wide",
            "anchor": "H16",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 8.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4, 5e-4, 1e-3]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4, 2e-4, 4e-4]),
            },
        },
    ]


def _bank_foursquare() -> list[Dict[str, Any]]:
    return [
        {
            "id": "F01_h15_v3_anchor",
            "band": "exploit",
            "source": "v3_best_foursquare_h15",
            "selection_score": "best old foursquare FMoE anchor",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 9.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([2, 3]),
            },
        },
        {
            "id": "F02_h2_regularized_anchor",
            "band": "exploit",
            "source": "v3_foursquare_default_outlier",
            "selection_score": "old foursquare default outlier",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0, 0.02, 0.04]),
            },
        },
        {
            "id": "F03_h5_len25_transfer",
            "band": "transfer",
            "source": "kuai_h5_len35_family",
            "selection_score": "compact context transfer",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.8e-4, 6.5e-4),
            "len": 25,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25, 30]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03]),
            },
        },
        {
            "id": "F04_h7_feat24_transfer",
            "band": "transfer",
            "source": "kuai_h7_feat24",
            "selection_score": "feature transfer to foursquare",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 8.0e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16, 24, 32]),
            },
        },
        {
            "id": "F05_h10_len30_context",
            "band": "transfer",
            "source": "kuai_h10_context",
            "selection_score": "context transfer to foursquare",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 30,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
            },
        },
        {
            "id": "F06_h11_fast_transfer",
            "band": "transfer",
            "source": "kuai_h11_fast",
            "selection_score": "wider fast family transfer",
            "anchor": "H11",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (3.0e-4, 9.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "expert_scale": stage1._choice_spec([2, 3]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4, 1e-3]),
            },
        },
        {
            "id": "F07_h1_reg_backup",
            "band": "transfer",
            "source": "late_foursquare_stage1_h1",
            "selection_score": "recent regularized backup",
            "anchor": "H1",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.8e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
        },
        {
            "id": "F08_len30_zero_drop",
            "band": "exploit",
            "source": "old_n4_foursquare_best_like",
            "selection_score": "replicate zero-drop len30 behavior",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 8.0e-4),
            "len": 30,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "family_drop": 0.0,
            "feature_drop": 0.0,
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0]),
                "stage_feature_dropout_prob": _stage_choice([0.0]),
                "hidden_dropout_prob": stage1._choice_spec([0.08, 0.10]),
            },
        },
        {
            "id": "F09_h15_lrhi",
            "band": "exploit",
            "source": "v3_best_foursquare_h15",
            "selection_score": "higher-lr h15 follow-up",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (6.0e-4, 1.6e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "F10_dropout_zero_probe",
            "band": "new_axis",
            "source": "foursquare_zero_drop_probe",
            "selection_score": "probe no-drop generalization",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "family_drop": 0.0,
            "feature_drop": 0.0,
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0, 0.02]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03]),
            },
        },
        {
            "id": "F11_auxlight_probe",
            "band": "new_axis",
            "source": "baseline_special_probe",
            "selection_score": "light aux sweep",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 8.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4, 5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4, 2e-4]),
            },
        },
        {
            "id": "F12_feat32_explore",
            "band": "new_axis",
            "source": "feature_width_probe",
            "selection_score": "test wider feature embedding on fast dataset",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 7.5e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([24, 32]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
        },
    ]


def _bank_movielens1m() -> list[Dict[str, Any]]:
    return [
        {
            "id": "M01_h1_v3_anchor",
            "band": "exploit",
            "source": "v3_best_movielens_h1",
            "selection_score": "best old movielens FMoE anchor",
            "anchor": "H1",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (8.0e-4, 3.5e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "M02_h1_lrlo_stable",
            "band": "exploit",
            "source": "v3_best_movielens_h1",
            "selection_score": "stability-focused h1 follow-up",
            "anchor": "H1",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 9.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "M03_h5_len25_transfer",
            "band": "transfer",
            "source": "kuai_h5_len35_family",
            "selection_score": "compact context transfer",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 8.0e-4),
            "len": 25,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
            },
        },
        {
            "id": "M04_h10_len25_context",
            "band": "transfer",
            "source": "kuai_h10_context",
            "selection_score": "context transfer to movielens",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 25,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "M05_h6_e2_compact",
            "band": "transfer",
            "source": "kuai_h6_e2_core",
            "selection_score": "compact transfer for movielens",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 1.0e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "expert_scale": stage1._choice_spec([1, 2]),
            },
        },
        {
            "id": "M06_h14_capacity_probe",
            "band": "new_axis",
            "source": "capacity_probe",
            "selection_score": "test if movielens benefits from wider capacity",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 8.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        },
    ]


def _bank_retail_rocket() -> list[Dict[str, Any]]:
    return [
        {
            "id": "R01_h2_v3_anchor",
            "band": "exploit",
            "source": "v3_best_retail_h2",
            "selection_score": "best old retail FMoE anchor",
            "anchor": "H2",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 2.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "R02_h6_regularized",
            "band": "transfer",
            "source": "retail_default_outlier+kuai_h6",
            "selection_score": "regularized compact retail follow-up",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.5e-4, 9.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "R03_h13_width_up",
            "band": "new_axis",
            "source": "baseline_width_up_style",
            "selection_score": "retail width-up inspired probe",
            "anchor": "H13",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.2e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16, 24]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
        },
        {
            "id": "R04_h15_short_hi",
            "band": "new_axis",
            "source": "baseline_short_hi_style",
            "selection_score": "short-high intensity retail probe",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.4e-3),
            "len": 15,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([15, 20]),
            },
        },
        {
            "id": "R05_h2_reg_relief",
            "band": "exploit",
            "source": "baseline_reg_relief_style+v3_h2",
            "selection_score": "same h2 family with lighter regularization",
            "anchor": "H2",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.4e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "low",
            "attn": [0.08, 0.10],
            "extra_search": {
                "weight_decay": stage1._choice_spec(stage1._weight_decay_choices("H2", [0.5, 1.0, 1.5])),
            },
        },
        {
            "id": "R06_h3_xfer_probe",
            "band": "new_axis",
            "source": "transfer_probe_amazon_or_foursquare",
            "selection_score": "one broader transfer-inspired retail probe",
            "anchor": "H3",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 8.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4, 1e-3]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        },
    ]


def _bank_lastfm() -> list[Dict[str, Any]]:
    return [
        {
            "id": "L01_h5_anchor",
            "band": "exploit",
            "source": "v3_lastfm_default_outlier",
            "selection_score": "minimal lastfm carry-over",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "L02_h15_light",
            "band": "exploit",
            "source": "v3_sparse_light_anchor",
            "selection_score": "light sparse backup",
            "anchor": "H15",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 6.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "L03_h11_fast",
            "band": "transfer",
            "source": "kuai_h11_fast",
            "selection_score": "one wide-cap sparse probe",
            "anchor": "H11",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.5e-4, 8.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "L04_h2_regularized",
            "band": "transfer",
            "source": "kuai_h2_regularized",
            "selection_score": "regularized sparse transfer",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.8e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
        },
    ]


DATASET_TEMPLATE_BANKS = {
    "amazon_beauty": _bank_amazon_beauty,
    "foursquare": _bank_foursquare,
    "movielens1m": _bank_movielens1m,
    "retail_rocket": _bank_retail_rocket,
    "lastfm0.03": _bank_lastfm,
}


def _template_bank(dataset: str) -> list[Dict[str, Any]]:
    if dataset not in DATASET_TEMPLATE_BANKS:
        raise RuntimeError(f"no template bank for dataset={dataset}")
    return list(DATASET_TEMPLATE_BANKS[dataset]())


def _row(dataset: str, template: Dict[str, Any], template_limit: int, seed_id: int, runtime_seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    anchor = str(template["anchor"])
    cfg = stage1._anchor_cfg(anchor)
    template_id = str(template["id"])
    cons_lambda, z_lambda = template["lambda"]
    hidden_mode = str(template.get("hidden_mode", "balanced"))

    family_drop_default = float(template.get("family_drop", 0.02 if hidden_mode != "high" else 0.04))
    feature_drop_default = float(template.get("feature_drop", 0.0))
    overrides = stage1._build_overrides(cons_lambda, z_lambda, family_drop_default, feature_drop_default)

    run_id = f"XD_{stage1.sanitize_token(dataset, upper=True)}_{stage1.sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
    run_phase = f"{PHASE_ID}_{run_id}"

    max_item_list_length = int(template.get("len", cfg.get("MAX_ITEM_LIST_LENGTH", 20)))
    d_feat_emb = int(template.get("d_feat", cfg.get("d_feat_emb", 16)))
    expert_scale = int(template.get("expert", cfg.get("expert_scale", 3)))

    fixed_values: Dict[str, Any] = {
        "embedding_size": int(cfg["embedding_size"]),
        "d_ff": int(cfg["d_ff"]),
        "d_expert_hidden": int(cfg["d_expert_hidden"]),
        "d_router_hidden": int(cfg["d_router_hidden"]),
        "MAX_ITEM_LIST_LENGTH": max_item_list_length,
        "d_feat_emb": d_feat_emb,
        "expert_scale": expert_scale,
        "lr_scheduler_type": "warmup_cosine",
        "num_heads": 4,
    }

    train_batch_size, eval_batch_size, max_evals = _choose_batch(dataset, template, fixed_values, args)
    budget = _dataset_budget(dataset)

    lr_low, lr_high = template["lr_bounds"]
    search_space: Dict[str, Any] = {
        "learning_rate": stage1._loguniform_spec(float(lr_low), float(lr_high)),
        "hidden_dropout_prob": stage1._choice_spec(stage1._hidden_choices(anchor, hidden_mode)),
        "attn_dropout_prob": stage1._choice_spec(float(v) for v in template["attn"]),
        "weight_decay": stage1._choice_spec(stage1._weight_decay_choices(anchor, list(template.get("wd_scales", [0.5, 1.0, 2.0])))),
    }
    search_space.update(dict(template.get("extra_search", {}) or {}))

    band = str(template.get("band", "portfolio"))
    source_family_id = str(template.get("source", template_id))
    return {
        "dataset": dataset,
        "phase_id": PHASE_ID,
        "axis_id": AXIS_ID,
        "axis_desc": AXIS_DESC,
        "architecture_id": ARCH_ID,
        "architecture_key": ARCH_KEY,
        "architecture_name": ARCH_NAME,
        "exp_brief": ARCH_NAME,
        "run_phase": run_phase,
        "run_id": run_id,
        "setting_id": template_id,
        "setting_key": template_id,
        "setting_desc": template_id,
        "stage": "cross_dataset",
        "tuning_stage": "cross_dataset",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": band,
        "capacity_anchor": anchor,
        "selected_from_stage": str(template.get("source", band)),
        "selection_score": str(template.get("selection_score", "")),
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "cross_dataset",
        "source_family_id": source_family_id,
        "template_count": int(template_limit),
        "aux_route_consistency_lambda": float(cons_lambda),
        "aux_z_loss_lambda": float(z_lambda),
        "fixed_values": fixed_values,
        "search_space": search_space,
        "overrides": overrides,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "max_evals": int(max_evals),
        "tune_epochs": int(args.tune_epochs),
        "tune_patience": int(budget["patience"]),
    }


def build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    datasets = stage1._parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets selected")
    seeds = stage1._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")
    counts = _dataset_counts(args.dataset_template_counts)

    rows: list[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        stage1._validate_session_fixed_files(dataset)
        template_limit = int(counts.get(dataset, 0))
        if template_limit <= 0:
            continue
        bank = _template_bank(dataset)
        templates = bank[:template_limit]
        if len(templates) != template_limit:
            raise RuntimeError(f"dataset={dataset} requested {template_limit} templates but bank has {len(bank)}")
        for template in templates:
            for seed_id in seeds:
                cursor += 1
                rows.append(
                    _row(
                        dataset=dataset,
                        template=template,
                        template_limit=template_limit,
                        seed_id=int(seed_id),
                        runtime_seed=int(args.seed_base) + cursor - 1,
                        args=args,
                    )
                )
    return rows


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        raw = Path(str(args.manifest_out))
        if raw.suffix:
            return raw
        return raw / "cross_dataset_manifest.json"
    return LOG_ROOT / "cross_dataset_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "cross_dataset",
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_count": len(rows),
        "datasets": sorted({str(row.get("dataset", "")) for row in rows}),
        "rows": [stage1._serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summary_path(dataset: str) -> Path:
    path = LOG_ROOT / str(dataset) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N4 cross-dataset A12 portfolio tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="CSV datasets")
    parser.add_argument(
        "--dataset-template-counts",
        default="amazon_beauty:12,foursquare:12,movielens1m:6,retail_rocket:6,lastfm0.03:4",
        help="CSV dataset:count overrides",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=264000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=6144)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--max-evals", type=int, default=12)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    args = parser.parse_args()
    if int(args.max_evals) < 1:
        raise RuntimeError("--max-evals must be >= 1")
    return args


def maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 2) or 2))])


def main() -> int:
    args = parse_args()
    rows = maybe_limit_smoke(build_rows(args), args)
    manifest = write_manifest(args, rows)
    print(f"[cross-dataset] manifest -> {manifest}")

    fieldnames = stage1.build_summary_fieldnames(
        [
            "architecture_id",
            "architecture_name",
            "tuning_stage",
            "family_id",
            "family_group",
            "variant_id",
            "capacity_anchor",
            "selected_from_stage",
            "selection_score",
            "search_algo",
            "source_family_id",
            "stage_group",
            "template_count",
            "aux_route_consistency_lambda",
            "aux_z_loss_lambda",
        ]
    )

    gpus = stage1._parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    return int(
        stage1.launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=LOG_ROOT,
            summary_path=LOG_ROOT / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[
                col
                for col in fieldnames
                if col
                not in {
                    "global_best_valid_mrr20",
                    "run_best_valid_mrr20",
                    "run_phase",
                    "exp_brief",
                    "stage",
                    "trigger",
                    "dataset",
                    "seed_id",
                    "gpu_id",
                    "status",
                    "test_mrr20",
                    "n_completed",
                    "interrupted",
                    "special_ok",
                    "diag_ok",
                    "result_path",
                    "timestamp_utc",
                }
            ],
            build_command=stage1.build_command,
            build_log_path=stage1.build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: summary_path(str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())