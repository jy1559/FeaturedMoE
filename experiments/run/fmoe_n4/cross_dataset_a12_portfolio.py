#!/usr/bin/env python3
"""FMoE_N4: cross-dataset A12 portfolio tuning outside KuaiRec.

This launcher is meant for fast transfer-style screening on non-Kuai datasets.
It mixes:
- Kuai Stage1/2 winners that looked broadly stable
- older v3/FMoE anchors that were strong on each target dataset
- a few dataset-specific exploratory probes inspired by baseline recovery logs

Defaults are intentionally uneven by dataset:
- beauty: 24 templates
- foursquare: 12 templates
- movielens1m: 6 templates
- retail_rocket: 6 templates
- lastfm0.03: 4 templates

All runs use 100 epochs. Dataset budgets cap max-evals, and patience can be
overridden from CLI when a longer sweep is needed.
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
DEFAULT_DATASETS = ["beauty", "foursquare", "movielens1m", "retail_rocket", "lastfm0.03"]

REPO_ROOT_REAL = stage1.REPO_ROOT_REAL
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

stage1.AXIS = AXIS
stage1.AXIS_ID = AXIS_ID
stage1.AXIS_DESC = AXIS_DESC
stage1.PHASE_ID = PHASE_ID
stage1.PHASE_NAME = PHASE_NAME
stage1.LOG_ROOT = LOG_ROOT

DEFAULT_TEMPLATE_COUNTS = {
    "beauty": 24,
    "foursquare": 12,
    "movielens1m": 6,
    "retail_rocket": 6,
    "lastfm0.03": 4,
}

DATASET_BUDGETS = {
    "beauty": {"max_evals": 24, "patience": 10},
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

    if dataset == "beauty":
        anchor = str(template.get("anchor", ""))
        expert = int(template.get("expert", 3))
        d_feat = int(template.get("d_feat", 16))
        if expert <= 3 and d_feat <= 12 and anchor not in {"H14", "H15", "H16"}:
            train_batch_size = min(int(train_batch_size), 3584)
            eval_batch_size = min(int(eval_batch_size), 4608)
        else:
            train_batch_size = min(int(train_batch_size), 3072)
            eval_batch_size = min(int(eval_batch_size), 4096)
        if expert >= 4 or d_feat >= 20:
            train_batch_size = min(int(train_batch_size), 2048)
            eval_batch_size = min(int(eval_batch_size), 3072)
        if expert >= 4 and d_feat >= 20 and anchor in {"H14", "H15", "H16"}:
            train_batch_size = min(int(train_batch_size), 1536)
            eval_batch_size = min(int(eval_batch_size), 2048)
    elif dataset == "lastfm0.03":
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


TOPUP_STAGE1_TEMPLATE_IDS = {
    "beauty": [
        "T01_balanced_h7",
        "T02_capacity_h14",
        "T03_regularized_h2",
        "T04_longctx_h10",
        "T05_lowaux_h1",
        "T06_midaux_h3",
        "T07_dimwide_h6",
        "T08_highaux_h8",
        "T09_bigdim_h14",
        "T10_smallsparse_h12",
        "T11_steady_h5",
        "T12_fastlr_h2",
        "T13_lowlr_h10",
        "T14_noisy_h9",
        "T15_depthcap_h11",
        "T16_widerexpert_h7",
    ],
    "foursquare": ["T11_steady_h5", "T07_dimwide_h6", "T04_longctx_h10", "T08_highaux_h8"],
    "movielens1m": ["T02_capacity_h14", "T09_bigdim_h14", "T13_lowlr_h10", "T15_depthcap_h11", "T11_steady_h5", "T05_lowaux_h1"],
    "retail_rocket": ["T03_regularized_h2", "T06_midaux_h3", "T07_dimwide_h6", "T10_smallsparse_h12", "T12_fastlr_h2", "T14_noisy_h9"],
    "lastfm0.03": ["T03_regularized_h2", "T05_lowaux_h1", "T06_midaux_h3", "T08_highaux_h8", "T10_smallsparse_h12", "T12_fastlr_h2", "T13_lowlr_h10", "T14_noisy_h9"],
}

TOPUP_DATASET_PREFIX = {
    "beauty": "B",
    "foursquare": "F",
    "movielens1m": "M",
    "retail_rocket": "R",
    "lastfm0.03": "L",
}

TOPUP_START_INDEX = {
    "beauty": 9,
    "foursquare": 29,
    "movielens1m": 11,
    "retail_rocket": 19,
    "lastfm0.03": 5,
}


def _topup_hidden_mode(family_drop_values: list[float]) -> str:
    if max(float(v) for v in family_drop_values) >= 0.06:
        return "high"
    if max(float(v) for v in family_drop_values) <= 0.04:
        return "low"
    return "balanced"


def _generated_topup_templates(dataset: str) -> list[Dict[str, Any]]:
    template_ids = list(TOPUP_STAGE1_TEMPLATE_IDS.get(dataset, []))
    if not template_ids:
        return []

    lookup = {str(template["id"]): template for template in stage1._template_bank_16()}
    prefix = TOPUP_DATASET_PREFIX[dataset]
    start_idx = int(TOPUP_START_INDEX[dataset])
    out: list[Dict[str, Any]] = []

    for offset, template_id in enumerate(template_ids):
        if template_id not in lookup:
            raise RuntimeError(f"missing stage1 template for top-up generation: {template_id}")

        source_template = dict(lookup[template_id])
        lr_values = [float(value) for value in source_template["lr"]]
        len_values = [int(value) for value in source_template["len"]]
        d_feat_values = [int(value) for value in source_template["d_feat"]]
        expert_values = [int(value) for value in source_template["expert"]]
        family_drop_values = [float(value) for value in source_template["fdrop"]]
        feature_drop_values = [float(value) for value in source_template["xdrop"]]
        attn_values = [float(value) for value in source_template["attn"]]
        head_values = [int(value) for value in source_template["heads"]]

        extra_search: Dict[str, Any] = {}
        if len(len_values) > 1:
            extra_search["MAX_ITEM_LIST_LENGTH"] = stage1._choice_spec(len_values)
        if len(d_feat_values) > 1:
            extra_search["d_feat_emb"] = stage1._choice_spec(d_feat_values)
        if len(expert_values) > 1:
            extra_search["expert_scale"] = stage1._choice_spec(expert_values)
        if len(family_drop_values) > 1:
            extra_search["stage_family_dropout_prob"] = _stage_choice(family_drop_values)
        if len(feature_drop_values) > 1:
            extra_search["stage_feature_dropout_prob"] = _stage_choice(feature_drop_values)
        if len(head_values) > 1 or head_values[0] != 4:
            extra_search["num_heads"] = stage1._choice_spec(head_values)

        out.append(
            {
                "id": f"{prefix}{start_idx + offset:02d}_{str(template_id).lower()}",
                "band": "topup",
                "source": f"stage1_{str(template_id).lower()}",
                "selection_score": f"top-up bridge from {template_id}",
                "anchor": str(source_template["anchor"]),
                "lambda": tuple(source_template["lambda"]),
                "lr_bounds": (min(lr_values), max(lr_values)),
                "len": int(len_values[0]),
                "d_feat": int(d_feat_values[0]),
                "expert": int(expert_values[0]),
                "wd_scales": [0.5, 1.0, 2.0],
                "hidden_mode": _topup_hidden_mode(family_drop_values),
                "family_drop": float(family_drop_values[0]),
                "feature_drop": float(feature_drop_values[0]),
                "attn": attn_values,
                "extra_search": extra_search,
            }
        )

    return out


def _beauty_template(
    template_id: str,
    band: str,
    source: str,
    selection_score: str,
    anchor: str,
    lambdas: tuple[float, float],
    lr_bounds: tuple[float, float],
    length: int,
    d_feat: int,
    expert: int,
    wd_scales: list[float],
    hidden_mode: str,
    attn: list[float],
    extra_search: Dict[str, Any] | None = None,
    family_drop: float | None = None,
    feature_drop: float | None = None,
) -> Dict[str, Any]:
    def _trim_choice_spec(spec: Any, keep: int = 1) -> Any:
        if not isinstance(spec, dict) or str(spec.get("type", "choice")) != "choice":
            return spec
        values = list(spec.get("values", []))
        if len(values) <= keep:
            return spec
        if keep <= 1:
            trimmed = [values[0]]
        else:
            trimmed = [values[0], values[-1]]
        return stage1._choice_spec(trimmed)

    wide_keep_keys = {
        "MAX_ITEM_LIST_LENGTH",
        "hidden_dropout_prob",
        "expert_scale",
        "d_feat_emb",
        "stage_family_dropout_prob",
        "route_consistency_lambda",
        "lr_scheduler_min_lr_ratio",
        "num_heads",
    }
    normalized_extra_search: Dict[str, Any] = {}
    for key, value in dict(extra_search or {}).items():
        keep = 1
        if key == "MAX_ITEM_LIST_LENGTH":
            keep = 2
        if band == "lr_topup" and key in wide_keep_keys:
            keep = 2
        normalized_extra_search[key] = _trim_choice_spec(value, keep=keep)

    lr_low, lr_high = lr_bounds
    widened_low = max(2.0e-4, float(lr_low))
    widened_high = max(float(lr_high), 7.0e-3 if band != "transfer" else 4.0e-3)
    template: Dict[str, Any] = {
        "id": template_id,
        "band": band,
        "source": source,
        "selection_score": selection_score,
        "anchor": anchor,
        "lambda": lambdas,
        "lr_bounds": (widened_low, widened_high),
        "len": int(length),
        "d_feat": int(d_feat),
        "expert": int(expert),
        "wd_scales": [float(wd_scales[0])],
        "hidden_mode": hidden_mode,
        "attn": [float(attn[0])],
        "extra_search": normalized_extra_search,
    }
    if family_drop is not None:
        template["family_drop"] = float(family_drop)
    if feature_drop is not None:
        template["feature_drop"] = float(feature_drop)
    return template


def _bank_beauty() -> list[Dict[str, Any]]:
    return [
        _beauty_template(
            template_id="B01_xfer_midaux_amazon",
            band="transfer",
            source="amazon_beauty_a15_t06_midaux_h3",
            selection_score="transfer the amazon winner pattern: H3, len40, d_feat12, expert3, low aux and moderate dropout",
            anchor="H3",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.4e-4, 5.8e-4),
            length=40,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.05,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.05]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.02]),
                "d_feat_emb": stage1._choice_spec([12]),
                "expert_scale": stage1._choice_spec([3]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        ),
        _beauty_template(
            template_id="B02_xfer_testcarry_h12",
            band="transfer",
            source="beauty_abcd_a12_l04",
            selection_score="beauty prior with strongest test carry-over, but narrowed toward compact dims",
            anchor="H12",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 4.8e-4),
            length=20,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.12, 0.14, 0.16]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.03, 0.04]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
            },
        ),
        _beauty_template(
            template_id="B03_xfer_testcarry_h11",
            band="transfer",
            source="beauty_abcd_a11_l04",
            selection_score="second beauty carry-over prior, shifted down from wide H11 into safer search bounds",
            anchor="H11",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.6e-4, 4.5e-4),
            length=20,
            d_feat=16,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.03, 0.04]),
            },
        ),
        _beauty_template(
            template_id="B04_xfer_regcarry_h13",
            band="transfer",
            source="beauty_abcd_a13_l04",
            selection_score="beauty prior with strong test and lighter regularization, plus small length variation",
            anchor="H13",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 5.2e-4),
            length=20,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "hidden_dropout_prob": stage1._choice_spec([0.12, 0.14, 0.16]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        ),
        _beauty_template(
            template_id="B05_xfer_compact_h7",
            band="transfer",
            source="beauty_abcd_a07_l04",
            selection_score="compact prior from top beauty run, retaining tiny-capacity bias but broadening dropout slightly",
            anchor="H7",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.6e-4, 4.8e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="high",
            attn=[0.08, 0.10],
            family_drop=0.04,
            feature_drop=0.02,
            extra_search={
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.22, 0.26, 0.30]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04, 0.05]),
            },
        ),
        _beauty_template(
            template_id="B06_xfer_compact_h8",
            band="transfer",
            source="beauty_abcd_a08_l04",
            selection_score="second compact prior from beauty, using H8 as a slightly richer version of the H7 family",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.8e-4, 5.2e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.04,
            feature_drop=0.0,
            extra_search={
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04, 0.05]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10, 0.15]),
            },
        ),
        _beauty_template(
            template_id="B07_xfer_regcompact_h4",
            band="transfer",
            source="beauty_abcd_a04_l04",
            selection_score="small highly regularized prior from beauty, reopened with shorter length and milder dropout",
            anchor="H4",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.4e-4, 4.2e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.10, 0.12],
            family_drop=0.04,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.22, 0.25]),
                "stage_feature_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        ),
        _beauty_template(
            template_id="B08_xfer_steady_h5",
            band="transfer",
            source="stage1_t11_steady_h5",
            selection_score="borrow the broad stable H5 transfer family but keep beauty regularization and lr narrow",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.2e-4, 5.2e-4),
            length=30,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30, 40]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12, 0.14]),
            },
        ),
        _beauty_template(
            template_id="B09_aggr_dimwide_h6",
            band="aggressive",
            source="beauty_dimwidth_probe",
            selection_score="aggressive width probe on the H6 family with bounded expert growth",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 6.0e-4),
            length=20,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([12, 16, 20]),
                "expert_scale": stage1._choice_spec([2, 3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18]),
            },
        ),
        _beauty_template(
            template_id="B10_aggr_depthcap_h11",
            band="aggressive",
            source="beauty_depthcap_probe",
            selection_score="deeper-capacity probe around H11 with beauty-sized lr and stronger batch cap",
            anchor="H11",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.4e-4, 4.2e-4),
            length=20,
            d_feat=16,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20]),
            },
        ),
        _beauty_template(
            template_id="B11_aggr_bigdim_h14",
            band="aggressive",
            source="beauty_bigdim_probe",
            selection_score="test whether beauty gains from one controlled H14 big-dim family rather than pure compactness",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.2e-4, 3.6e-4),
            length=20,
            d_feat=16,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.12, 0.14, 0.16]),
            },
        ),
        _beauty_template(
            template_id="B12_aggr_longctx_h10",
            band="aggressive",
            source="beauty_long_context_probe",
            selection_score="push sequence length as the main axis while keeping the rest compact and stable",
            anchor="H10",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.0e-4, 4.0e-4),
            length=40,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40, 50]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16, 0.18]),
            },
        ),
        _beauty_template(
            template_id="B13_aggr_widerexpert_h7",
            band="aggressive",
            source="beauty_widerexpert_probe",
            selection_score="try tiny-width H7 with wider expert branching instead of wider hidden states",
            anchor="H7",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 5.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="high",
            attn=[0.08, 0.12],
            family_drop=0.04,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12, 16]),
                "expert_scale": stage1._choice_spec([2, 3, 4]),
                "num_heads": stage1._choice_spec([2, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.22, 0.26, 0.30]),
            },
        ),
        _beauty_template(
            template_id="B14_aggr_highaux_h8",
            band="aggressive",
            source="beauty_high_aux_probe",
            selection_score="explicitly test whether beauty benefits from stronger route regularization when capacity stays compact",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.6e-4, 4.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="high",
            attn=[0.10, 0.12],
            family_drop=0.05,
            feature_drop=0.02,
            extra_search={
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20, 0.22]),
                "route_consistency_lambda": stage1._choice_spec([5e-4, 8e-4, 1.2e-3]),
                "z_loss_lambda": stage1._choice_spec([1e-4, 2e-4, 3e-4]),
                "stage_family_dropout_prob": _stage_choice([0.04, 0.05, 0.06]),
            },
        ),
        _beauty_template(
            template_id="B15_aggr_fastadapt_h2",
            band="aggressive",
            source="beauty_fast_adapt_probe",
            selection_score="one faster-adaptation H2 family with extra head and lr variation, but not full fastlr extremes",
            anchor="H2",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(3.0e-4, 9.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.10, 0.12],
            family_drop=0.04,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "num_heads": stage1._choice_spec([2, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16, 0.18]),
            },
        ),
        _beauty_template(
            template_id="B16_aggr_lenmix_h5",
            band="aggressive",
            source="beauty_lenmix_probe",
            selection_score="test if H5 benefits from mixing moderate context with slightly richer width on beauty",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 6.0e-4),
            length=30,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30, 40]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12, 0.14]),
            },
        ),
        _beauty_template(
            template_id="B17_hyp_midaux_lowaux_h3",
            band="hypothesis",
            source="beauty_fmoe_small_dataset_hypothesis",
            selection_score="hypothesis: beauty likes H3 mid-capacity but lower aux than amazon because the target is smaller and noisier",
            anchor="H3",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.2e-4, 4.8e-4),
            length=40,
            d_feat=12,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.04,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
            },
        ),
        _beauty_template(
            template_id="B18_hyp_smallfeat_e2_h6",
            band="hypothesis",
            source="beauty_fmoe_compact_expert_hypothesis",
            selection_score="hypothesis: beauty only needs H6 representation width if expert count is kept very small",
            anchor="H6",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.0e-4, 5.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.02]),
            },
        ),
        _beauty_template(
            template_id="B19_hyp_lowaux_sparse_h1",
            band="hypothesis",
            source="beauty_fmoe_sparse_session_hypothesis",
            selection_score="hypothesis: beauty behaves like a sparse short-session problem where low-aux H1 can stay surprisingly competitive",
            anchor="H1",
            lambdas=(2e-4, 5e-5),
            lr_bounds=(2.5e-4, 6.0e-4),
            length=30,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.10, 0.12],
            family_drop=0.03,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30, 40]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "num_heads": stage1._choice_spec([2, 4]),
                "hidden_dropout_prob": stage1._choice_spec([0.12, 0.14, 0.16]),
            },
        ),
        _beauty_template(
            template_id="B20_hyp_lowlr_longctx_h10",
            band="hypothesis",
            source="beauty_fmoe_long_context_hypothesis",
            selection_score="hypothesis: longer context helps beauty only when lr and aux are both kept deliberately low",
            anchor="H10",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(8.0e-5, 3.5e-4),
            length=40,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40, 50]),
                "d_feat_emb": stage1._choice_spec([12]),
                "expert_scale": stage1._choice_spec([3]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16, 0.18]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        ),
        _beauty_template(
            template_id="B21_hyp_compactdense_h12",
            band="hypothesis",
            source="beauty_fmoe_compact_dense_hypothesis",
            selection_score="hypothesis: H12 can work on beauty if kept compact in feature width and with low decay",
            anchor="H12",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(1.8e-4, 4.5e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.06, 0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.12, 0.14, 0.16]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.03, 0.04]),
            },
        ),
        _beauty_template(
            template_id="B22_hyp_steady_mid_h5",
            band="hypothesis",
            source="beauty_fmoe_mid_capacity_hypothesis",
            selection_score="hypothesis: a very steady H5 with restrained width may generalize better than both tiny and wide families",
            anchor="H5",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.2e-4, 5.0e-4),
            length=30,
            d_feat=12,
            expert=3,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="low",
            attn=[0.08, 0.10],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([12]),
                "expert_scale": stage1._choice_spec([3]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12, 0.14]),
            },
        ),
        _beauty_template(
            template_id="B23_hyp_tiny_highdrop_h7",
            band="hypothesis",
            source="beauty_fmoe_tiny_highdrop_hypothesis",
            selection_score="hypothesis: on beauty, a tiny H7 with very high hidden dropout may outperform larger models by avoiding memorization",
            anchor="H7",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(1.5e-4, 4.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="high",
            attn=[0.08, 0.10],
            family_drop=0.05,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "hidden_dropout_prob": stage1._choice_spec([0.24, 0.27, 0.30]),
                "stage_family_dropout_prob": _stage_choice([0.04, 0.05, 0.06]),
            },
        ),
        _beauty_template(
            template_id="B24_hyp_regrelief_h2",
            band="hypothesis",
            source="beauty_fmoe_reg_relief_hypothesis",
            selection_score="hypothesis: H2 improves on beauty when classic regularization is eased but compact sequence modeling stays intact",
            anchor="H2",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.5e-4, 8.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125, 0.25, 0.5],
            hidden_mode="balanced",
            attn=[0.10, 0.12],
            family_drop=0.03,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        ),
        _beauty_template(
            template_id="B25_lr_h8_seen_anchor",
            band="lr_topup",
            source="topup_from_b06_seen_best",
            selection_score="follow up the best seen-target H8 compact line with wider lr and slight regularization movement",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=12,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="balanced",
            attn=[0.10],
            family_drop=0.04,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        ),
        _beauty_template(
            template_id="B26_lr_h6_dimwide_seen",
            band="lr_topup",
            source="topup_from_b09_seen_best",
            selection_score="follow up the H6 dimwide winner with lr emphasis and only mild width/expert movement",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=12,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="balanced",
            attn=[0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([2, 3]),
            },
        ),
        _beauty_template(
            template_id="B27_lr_h4_regcompact_seen",
            band="lr_topup",
            source="topup_from_b07_seen_best",
            selection_score="follow up the H4 regularized compact line with lr emphasis and small dropout movement",
            anchor="H4",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 7.0e-3),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="balanced",
            attn=[0.10],
            family_drop=0.04,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.22, 0.25]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04]),
            },
        ),
        _beauty_template(
            template_id="B28_lr_h8_highaux_seen",
            band="lr_topup",
            source="topup_from_b14_seen_best",
            selection_score="follow up the strong running H8 high-aux family with lr emphasis and a small route/dropout fork",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="high",
            attn=[0.10],
            family_drop=0.05,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.22]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "route_consistency_lambda": stage1._choice_spec([8e-4, 1.2e-3]),
                "stage_family_dropout_prob": _stage_choice([0.04, 0.05]),
            },
        ),
        _beauty_template(
            template_id="B29_lr_h7_compact_seen",
            band="lr_topup",
            source="topup_from_b05_seen_best",
            selection_score="follow up the strong H7 compact line with lr emphasis and a small dropout/family-drop fork",
            anchor="H7",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="high",
            attn=[0.10],
            family_drop=0.04,
            feature_drop=0.02,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.24, 0.30]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.05]),
            },
        ),
        _beauty_template(
            template_id="B30_lr_h3_midaux_seen",
            band="lr_topup",
            source="topup_from_b01_seen_strong",
            selection_score="follow up the H3 midaux amazon-transfer line with wider lr and small context/dropout movement",
            anchor="H3",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 7.0e-3),
            length=40,
            d_feat=12,
            expert=3,
            wd_scales=[0.125],
            hidden_mode="balanced",
            attn=[0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40]),
                "hidden_dropout_prob": stage1._choice_spec([0.16, 0.18]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.05]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        ),
        _beauty_template(
            template_id="B31_lr_h5_steady_seen",
            band="lr_topup",
            source="topup_from_b08_seen_stable",
            selection_score="follow up the H5 steady transfer line with lr emphasis and slight length/expert flexibility",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=30,
            d_feat=12,
            expert=3,
            wd_scales=[0.125],
            hidden_mode="low",
            attn=[0.08],
            family_drop=0.03,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 40]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
        ),
        _beauty_template(
            template_id="B32_lr_h8_route_seen",
            band="lr_topup",
            source="topup_from_b06_b14_mix",
            selection_score="follow up the best H8 family with a small route-regularization fork and wider lr ceiling",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=8,
            expert=2,
            wd_scales=[0.125],
            hidden_mode="high",
            attn=[0.10],
            family_drop=0.04,
            feature_drop=0.0,
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 30]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20]),
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "route_consistency_lambda": stage1._choice_spec([5e-4, 8e-4]),
                "stage_family_dropout_prob": _stage_choice([0.04, 0.05]),
            },
        ),
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
        {
            "id": "F13_h2_zero_drop_refine",
            "band": "followup_conservative",
            "source": "F10_dropout_zero_probe",
            "selection_score": "refine current best zero-drop foursquare family",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.6e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "high",
            "attn": [0.10],
            "family_drop": 0.0,
            "feature_drop": 0.0,
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0]),
                "stage_feature_dropout_prob": _stage_choice([0.0]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12]),
            },
        },
        {
            "id": "F14_h15_auxlight_refine",
            "band": "followup_conservative",
            "source": "F11_auxlight_probe",
            "selection_score": "refine aux-light h15 foursquare winner",
            "anchor": "H15",
            "lambda": (2.5e-4, 5e-5),
            "lr_bounds": (2.4e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4]),
            },
        },
        {
            "id": "F15_h7_feat32_refine",
            "band": "followup_conservative",
            "source": "F12_feat32_explore",
            "selection_score": "refine feat32 foursquare winner with tighter lr",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.4e-4, 6.5e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([24, 32]),
            },
        },
        {
            "id": "F16_h5_len25_refine",
            "band": "followup_conservative",
            "source": "F03_h5_len25_transfer",
            "selection_score": "refine compact h5 transfer winner",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 5.5e-4),
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
            "id": "F17_h8_highaux_probe",
            "band": "followup_aggressive",
            "source": "beauty_b25_h8_seen_anchor",
            "selection_score": "aggressive high-aux h8 import from beauty winner",
            "anchor": "H8",
            "lambda": (1.6e-3, 4e-4),
            "lr_bounds": (1.8e-4, 5.0e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.16],
            "family_drop": 0.04,
            "feature_drop": 0.02,
        },
        {
            "id": "F18_h6_dimwide_probe",
            "band": "followup_aggressive",
            "source": "beauty_b09_dimwide_h6",
            "selection_score": "aggressive dim-wide h6 carry-over from beauty",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.4e-4, 7.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
        },
        {
            "id": "F19_h7_widerexpert_probe",
            "band": "followup_aggressive",
            "source": "beauty_b13_widerexpert_h7",
            "selection_score": "aggressive wider-expert h7 carry-over from beauty",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 4,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "low",
            "attn": [0.08, 0.10],
        },
        {
            "id": "F20_h3_midaux_probe",
            "band": "followup_aggressive",
            "source": "beauty_b17_midaux_lowaux_h3",
            "selection_score": "aggressive mid-aux h3 carry-over from beauty",
            "anchor": "H3",
            "lambda": (4e-4, 8e-5),
            "lr_bounds": (2.2e-4, 6.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4]),
            },
        },
        {
            "id": "F21_h5_lr_validate",
            "band": "followup_validation",
            "source": "F16_h5_len25_refine+F03_h5_len25_transfer",
            "selection_score": "validate the strongest compact H5 line with mostly lr-only search and a wider ceiling",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 9.0e-4),
            "len": 25,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "low",
            "attn": [0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25]),
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.10]),
            },
        },
        {
            "id": "F22_h2_zero_drop_validate",
            "band": "followup_validation",
            "source": "F10_dropout_zero_probe+F13_h2_zero_drop_refine",
            "selection_score": "validate the zero-drop H2 family with lr-only search around the winning band",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.2e-4, 7.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "high",
            "attn": [0.10],
            "family_drop": 0.0,
            "feature_drop": 0.0,
            "extra_search": {
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
                "stage_family_dropout_prob": _stage_choice([0.0]),
                "stage_feature_dropout_prob": _stage_choice([0.0]),
            },
        },
        {
            "id": "F23_h6_dimwide_validate",
            "band": "followup_validation",
            "source": "F18_h6_dimwide_probe",
            "selection_score": "validate the H6 dim-wide family with lr-only search and fixed width",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.6e-4, 8.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16]),
                "expert_scale": stage1._choice_spec([3]),
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
            },
        },
        {
            "id": "F24_h7_feat32_validate",
            "band": "followup_validation",
            "source": "F15_h7_feat32_refine",
            "selection_score": "validate the strongest feat32 H7 line with lr-only search",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.2e-4, 9.5e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([32]),
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
            },
        },
        {
            "id": "F25_h3_midaux_attack",
            "band": "followup_aggressive_v2",
            "source": "F20_h3_midaux_probe",
            "selection_score": "aggressive H3 mid-aux recheck with a higher lr ceiling and slightly longer context",
            "anchor": "H3",
            "lambda": (4e-4, 8e-5),
            "lr_bounds": (2.2e-4, 8.5e-4),
            "len": 25,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25]),
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
            },
        },
        {
            "id": "F26_h11_fast_attack",
            "band": "followup_aggressive_v2",
            "source": "F06_h11_fast_transfer",
            "selection_score": "aggressive H11 replay from the v4 transfer line with a deliberately higher lr ceiling",
            "anchor": "H11",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (3.5e-4, 1.3e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
                "expert_scale": stage1._choice_spec([2]),
            },
        },
        {
            "id": "F27_h15_auxlight_attack",
            "band": "followup_aggressive_v2",
            "source": "F14_h15_auxlight_refine+F09_h15_lrhi",
            "selection_score": "aggressive H15 aux-light retry with a broader upper lr than the current refine sweep",
            "anchor": "H15",
            "lambda": (2.5e-4, 5e-5),
            "lr_bounds": (3.0e-4, 1.1e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "hidden_dropout_prob": stage1._choice_spec([0.18]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5]),
            },
        },
        {
            "id": "F28_h5_len30_zero_attack",
            "band": "followup_aggressive_v2",
            "source": "F08_len30_zero_drop+F16_h5_len25_refine",
            "selection_score": "aggressive merge of the v4 len30 zero-drop line and the current H5 compact winner",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.8e-4, 9.0e-4),
            "len": 30,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "low",
            "attn": [0.08],
            "family_drop": 0.0,
            "feature_drop": 0.0,
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30]),
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
                "stage_family_dropout_prob": _stage_choice([0.0]),
                "stage_feature_dropout_prob": _stage_choice([0.0]),
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
        {
            "id": "M07_h1_highlr_validate",
            "band": "followup_validation",
            "source": "M01_h1_v3_anchor",
            "selection_score": "validate the strongest movielens H1 line with lr-only search around the winning high-lr region",
            "anchor": "H1",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (9.0e-4, 3.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "hidden_dropout_prob": stage1._choice_spec([0.08]),
            },
        },
        {
            "id": "M08_h6_compact_validate",
            "band": "followup_validation",
            "source": "M05_h6_e2_compact",
            "selection_score": "validate the compact H6 movielens line with lr-only search",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.5e-4, 1.1e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "expert_scale": stage1._choice_spec([2]),
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
            },
        },
        {
            "id": "M09_h5_transfer_attack",
            "band": "followup_aggressive_v2",
            "source": "M03_h5_len25_transfer",
            "selection_score": "aggressive H5 transfer retry with a higher lr ceiling",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.4e-3),
            "len": 25,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "low",
            "attn": [0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25]),
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
            },
        },
        {
            "id": "M10_h14_capacity_attack",
            "band": "followup_aggressive_v2",
            "source": "M06_h14_capacity_probe",
            "selection_score": "aggressive H14 capacity retry with a broader upper lr",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.3e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "low",
            "attn": [0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([20]),
                "hidden_dropout_prob": stage1._choice_spec([0.10]),
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
        {
            "id": "R07_h3_xfer_refine",
            "band": "followup_conservative",
            "source": "R06_h3_xfer_probe",
            "selection_score": "refine current best retail h3 transfer family",
            "anchor": "H3",
            "lambda": (7e-4, 1.5e-4),
            "lr_bounds": (2.2e-4, 6.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        },
        {
            "id": "R08_h6_regularized_refine",
            "band": "followup_conservative",
            "source": "R02_h6_regularized",
            "selection_score": "refine compact regularized retail winner",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.5e-4, 6.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "R09_h15_short_refine",
            "band": "followup_conservative",
            "source": "R04_h15_short_hi",
            "selection_score": "refine short high-intensity retail winner",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.0e-3),
            "len": 15,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([15, 20]),
            },
        },
        {
            "id": "R10_h13_width_refine",
            "band": "followup_conservative",
            "source": "R03_h13_width_up",
            "selection_score": "refine width-up retail winner with tighter wd",
            "anchor": "H13",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 8.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16, 24]),
            },
        },
        {
            "id": "R11_h8_highaux_probe",
            "band": "followup_aggressive",
            "source": "beauty_b25_h8_seen_anchor",
            "selection_score": "aggressive high-aux h8 import for retail",
            "anchor": "H8",
            "lambda": (1.6e-3, 4e-4),
            "lr_bounds": (1.8e-4, 4.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.16],
            "family_drop": 0.04,
            "feature_drop": 0.02,
        },
        {
            "id": "R12_h6_dimwide_probe",
            "band": "followup_aggressive",
            "source": "beauty_b09_dimwide_h6",
            "selection_score": "aggressive dim-wide h6 import for retail",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.2e-4, 7.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
        },
        {
            "id": "R13_h2_fastadapt_probe",
            "band": "followup_aggressive",
            "source": "beauty_b15_fastadapt_h2",
            "selection_score": "aggressive fast-lr h2 import for retail",
            "anchor": "H2",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (5.0e-4, 1.4e-3),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.10, 0.14],
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.03, 0.05]),
            },
        },
        {
            "id": "R14_h7_widerexpert_probe",
            "band": "followup_aggressive",
            "source": "beauty_b13_widerexpert_h7",
            "selection_score": "aggressive wider-expert h7 import for retail",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.4e-4, 7.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 4,
            "wd_scales": [0.5, 1.0],
            "hidden_mode": "low",
            "attn": [0.08, 0.10],
        },
        {
            "id": "R15_h13_width_lr_validate",
            "band": "followup_validation",
            "source": "R03_h13_width_up+R10_h13_width_refine",
            "selection_score": "validate the strongest retail width family with lr-only search and a broad upper ceiling",
            "anchor": "H13",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.5e-4, 1.4e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([24]),
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
            },
        },
        {
            "id": "R16_h3_xfer_lr_validate",
            "band": "followup_validation",
            "source": "R06_h3_xfer_probe+R07_h3_xfer_refine",
            "selection_score": "validate the strong retail H3 transfer family with lr-only search",
            "anchor": "H3",
            "lambda": (7e-4, 1.5e-4),
            "lr_bounds": (3.0e-4, 9.0e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "high",
            "attn": [0.10],
            "extra_search": {
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4]),
            },
        },
        {
            "id": "R17_h15_short_attack",
            "band": "followup_aggressive_v2",
            "source": "R04_h15_short_hi+R09_h15_short_refine",
            "selection_score": "aggressive retail short-context retry with a higher lr ceiling than the refine sweep",
            "anchor": "H15",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (5.0e-4, 1.6e-3),
            "len": 15,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([15]),
                "hidden_dropout_prob": stage1._choice_spec([0.16]),
            },
        },
        {
            "id": "R18_h6_dimlift_attack",
            "band": "followup_aggressive_v2",
            "source": "R02_h6_regularized+R12_h6_dimwide_probe",
            "selection_score": "aggressive retail H6 retry that keeps the regularized backbone but lifts width slightly",
            "anchor": "H6",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (3.2e-4, 1.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [1.0],
            "hidden_mode": "balanced",
            "attn": [0.08],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16]),
                "hidden_dropout_prob": stage1._choice_spec([0.14]),
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
    "beauty": _bank_beauty,
    "foursquare": _bank_foursquare,
    "movielens1m": _bank_movielens1m,
    "retail_rocket": _bank_retail_rocket,
    "lastfm0.03": _bank_lastfm,
}


def _template_bank(dataset: str) -> list[Dict[str, Any]]:
    if dataset not in DATASET_TEMPLATE_BANKS:
        raise RuntimeError(f"no template bank for dataset={dataset}")
    base_bank = list(DATASET_TEMPLATE_BANKS[dataset]())
    return base_bank + _generated_topup_templates(dataset)


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
        "tune_patience": int(args.tune_patience if int(args.tune_patience) > 0 else budget["patience"]),
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
        start_index = max(0, int(getattr(args, "template_start_index", 0) or 0))
        templates = bank[start_index : start_index + template_limit]
        if len(templates) != template_limit:
            raise RuntimeError(
                f"dataset={dataset} requested {template_limit} templates from start_index={start_index} but bank has {len(bank)}"
            )
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
    parser.add_argument("--tune-patience", type=int, default=0, help="Override per-dataset patience when > 0")
    parser.add_argument("--template-start-index", type=int, default=0, help="Start index into the template bank for follow-up slices")
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
    if int(args.tune_patience) < 0:
        raise RuntimeError("--tune-patience must be >= 0")
    if int(args.template_start_index) < 0:
        raise RuntimeError("--template-start-index must be >= 0")
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