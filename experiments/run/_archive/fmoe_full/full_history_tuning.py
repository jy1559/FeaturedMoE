#!/usr/bin/env python3
"""FMoE_N3 full-history tuning across the feature_added_v4 main six datasets.

This launcher keeps the existing session_fixed target protocol, but switches the
input construction to full_history_session_targets with strict_train_prefix.
Main metrics remain seen-target metrics by default; unseen-target metrics are
logged separately for inspection.

Each dataset gets 8 curated combos:
- 4 prior strong lines carried over as-is
- 2 conservative full-history retunes
- 2 aggressive full-history probes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
FMOE_N3_DIR = THIS_DIR.parent / "fmoe_n3"
FMOE_N4_DIR = THIS_DIR.parent / "fmoe_n4"
for extra_path in (FMOE_N3_DIR, FMOE_N4_DIR):
    if str(extra_path) not in sys.path:
        sys.path.append(str(extra_path))

import stage1_a12_broad_templates as stage1  # noqa: E402
from run_phase9_auxloss import _parse_csv_ints, _parse_csv_strings, hydra_literal  # noqa: E402
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows, sanitize_token  # noqa: E402

TRACK = "fmoe_full"
AXIS = "FullHistory_Portfolio8"
AXIS_ID = "FH8"
AXIS_DESC = "full_history_portfolio8"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "FH1"
PHASE_NAME = "FMOE_FULL_HISTORY_PORTFOLIO"
RUN_STAGE = "full_history"
FEATURE_MODE = "full_v4"
FEATURE_DATASET_DIR = "feature_added_v4"
HISTORY_INPUT_MODE = "full_history_session_targets"
HISTORY_EVAL_POLICY = "strict_train_prefix"
MAIN_EVAL_TARGET_MODE = "seen_target"
DEFAULT_DATASETS = [
    "beauty",
    "KuaiRecLargeStrictPosV2_0.2",
    "foursquare",
    "retail_rocket",
    "movielens1m",
    "lastfm0.03",
]

REPO_ROOT_REAL = THIS_DIR.parents[2]
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

DATASET_BUDGETS: Dict[str, Dict[str, int]] = {
    "beauty": {"max_evals": 8, "train_batch_size": 4608, "eval_batch_size": 6144},
    "KuaiRecLargeStrictPosV2_0.2": {"max_evals": 6, "train_batch_size": 2048, "eval_batch_size": 4096},
    "foursquare": {"max_evals": 5, "train_batch_size": 2560, "eval_batch_size": 4096},
    "retail_rocket": {"max_evals": 4, "train_batch_size": 2560, "eval_batch_size": 4096},
    "movielens1m": {"max_evals": 4, "train_batch_size": 2048, "eval_batch_size": 4096},
    "lastfm0.03": {"max_evals": 3, "train_batch_size": 1536, "eval_batch_size": 3072},
}


def _stage_choice(values: list[float]) -> Dict[str, Any]:
    return stage1._choice_spec(stage1._all_stage_map(float(value)) for value in values)


def _combo(
    combo_id: str,
    *,
    band: str,
    source: str,
    selection_score: str,
    anchor: str,
    lambdas: tuple[float, float],
    lr_bounds: tuple[float, float],
    length: int,
    d_feat: int,
    expert: int,
    hidden_mode: str,
    wd_scales: list[float],
    attn: list[float],
    extra_search: Dict[str, Any] | None = None,
    family_drop: float | None = None,
    feature_drop: float | None = None,
) -> Dict[str, Any]:
    return {
        "id": combo_id,
        "band": band,
        "source": source,
        "selection_score": selection_score,
        "anchor": anchor,
        "lambda": tuple(lambdas),
        "lr_bounds": tuple(lr_bounds),
        "len": int(length),
        "d_feat": int(d_feat),
        "expert": int(expert),
        "hidden_mode": hidden_mode,
        "wd_scales": list(wd_scales),
        "attn": list(attn),
        "extra_search": dict(extra_search or {}),
        "family_drop": family_drop,
        "feature_drop": feature_drop,
    }


def _bank_beauty() -> list[Dict[str, Any]]:
    return [
        _combo(
            "B01_h8_seen_anchor",
            band="prior_strong",
            source="B25_lr_h8_seen_anchor",
            selection_score="best current beauty line by both valid and test",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 8.0e-3),
            length=20,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.125],
            attn=[0.10],
            extra_search={
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20]),
                "stage_family_dropout_prob": _stage_choice([0.03, 0.04]),
            },
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B02_h8_compact",
            band="prior_strong",
            source="B06_xfer_compact_h8",
            selection_score="compact H8 transfer line with stable valid and strong test",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.8e-4, 5.2e-4),
            length=20,
            d_feat=8,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.125, 0.25, 0.5],
            attn=[0.08, 0.10],
            extra_search={"d_feat_emb": stage1._choice_spec([8, 12])},
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B03_h3_midaux",
            band="prior_strong",
            source="B17_hyp_midaux_lowaux_h3",
            selection_score="H3 mid-cap line that held valid while lifting test close to the best band",
            anchor="H3",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.2e-4, 4.8e-4),
            length=40,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.125, 0.25, 0.5],
            attn=[0.08, 0.10],
            extra_search={"hidden_dropout_prob": stage1._choice_spec([0.16, 0.18, 0.20])},
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B04_h7_widerexpert",
            band="prior_strong",
            source="B13_aggr_widerexpert_h7",
            selection_score="wider-expert H7 line with acceptable valid and strong test carry-over",
            anchor="H7",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 5.0e-4),
            length=20,
            d_feat=8,
            expert=2,
            hidden_mode="high",
            wd_scales=[0.125, 0.25, 0.5],
            attn=[0.08, 0.12],
            extra_search={"expert_scale": stage1._choice_spec([2, 3, 4])},
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B05_fh_h8_ctx_lowlr",
            band="full_history_conservative",
            source="B25_lr_h8_seen_anchor",
            selection_score="best H8 line adjusted for longer context with milder lr ceiling and slightly longer crop",
            anchor="H8",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.5e-4, 3.5e-3),
            length=30,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.25, 0.5, 1.0],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40]),
                "hidden_dropout_prob": stage1._choice_spec([0.20, 0.22]),
            },
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B06_fh_h3_ctx_regularized",
            band="full_history_conservative",
            source="B17_hyp_midaux_lowaux_h3",
            selection_score="H3 full-history retune with more context and slightly stronger regularization",
            anchor="H3",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(1.6e-4, 4.0e-4),
            length=50,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.25, 0.5, 1.0],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([40, 50]),
                "hidden_dropout_prob": stage1._choice_spec([0.18, 0.20, 0.22]),
            },
            family_drop=0.04,
            feature_drop=0.0,
        ),
        _combo(
            "B07_fh_h10_longctx_attack",
            band="full_history_aggressive",
            source="B12_aggr_longctx_h10",
            selection_score="explicit long-context attack once full-history is enabled",
            anchor="H10",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.0e-4, 6.0e-4),
            length=64,
            d_feat=12,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.125, 0.25, 0.5],
            attn=[0.06, 0.08],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([56, 64, 72]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16, 0.18]),
            },
            family_drop=0.03,
            feature_drop=0.0,
        ),
        _combo(
            "B08_fh_h6_dimwide_attack",
            band="full_history_aggressive",
            source="B09_aggr_dimwide_h6",
            selection_score="dimension-lifted H6 probe under longer context",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 7.0e-4),
            length=36,
            d_feat=16,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[0.125, 0.25, 0.5],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([32, 36, 40]),
                "d_feat_emb": stage1._choice_spec([16, 20]),
                "expert_scale": stage1._choice_spec([3, 4]),
            },
            family_drop=0.03,
            feature_drop=0.0,
        ),
    ]


def _bank_kuai() -> list[Dict[str, Any]]:
    return [
        _combo(
            "K01_h14_seen_hi",
            band="prior_strong",
            source="S02_h14_seen_hi",
            selection_score="best Kuai high-capacity seen-valid line with strong seen-test",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.5e-4, 5.5e-4),
            length=20,
            d_feat=16,
            expert=4,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={"expert_scale": stage1._choice_spec([4, 5])},
        ),
        _combo(
            "K02_h7_feat32_core",
            band="prior_strong",
            source="S04_h7_feat32_core",
            selection_score="best feature-rich Kuai family by valid with top-tier seen-test",
            anchor="H7",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 7.0e-4),
            length=25,
            d_feat=32,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={"d_feat_emb": stage1._choice_spec([24, 32])},
        ),
        _combo(
            "K03_h10_ctx_feat24",
            band="prior_strong",
            source="S07_h10_len25_f24",
            selection_score="best context-plus-feature Kuai family with strong test carry-over",
            anchor="H10",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.8e-4, 6.0e-4),
            length=25,
            d_feat=24,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.06, 0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30])},
        ),
        _combo(
            "K04_h6_e2_core",
            band="prior_strong",
            source="S06_h6_e2_core",
            selection_score="best compact non-H14 Kuai line by valid",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.2e-4, 8.0e-4),
            length=20,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0, 4.0],
            attn=[0.08, 0.10],
            extra_search={"expert_scale": stage1._choice_spec([1, 2])},
        ),
        _combo(
            "K05_fh_h10_ctx_lowlr",
            band="full_history_conservative",
            source="S07_h10_len25_f24",
            selection_score="full-history retune of the Kuai context family with longer crop and lower lr ceiling",
            anchor="H10",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.2e-4, 4.5e-4),
            length=40,
            d_feat=24,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.06, 0.08],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 50]),
                "hidden_dropout_prob": stage1._choice_spec([0.14, 0.16]),
            },
        ),
        _combo(
            "K06_fh_h14_lowlr",
            band="full_history_conservative",
            source="S02_h14_seen_hi",
            selection_score="high-capacity Kuai line adjusted downward for longer prefixes",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.0e-4, 3.8e-4),
            length=30,
            d_feat=16,
            expert=4,
            hidden_mode="low",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 40]),
                "hidden_dropout_prob": stage1._choice_spec([0.10, 0.12]),
            },
        ),
        _combo(
            "K07_fh_h11_auxwide_attack",
            band="full_history_aggressive",
            source="N14_aux_wide_probe",
            selection_score="aux-wide probe kept alive because longer history may stabilize routing",
            anchor="H11",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.8e-4, 7.5e-4),
            length=40,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([36, 40, 45]),
                "route_consistency_lambda": stage1._choice_spec([5e-4, 8e-4, 1.2e-3]),
                "z_loss_lambda": stage1._choice_spec([1e-4, 2e-4, 3e-4]),
            },
        ),
        _combo(
            "K08_fh_h14_e5_longctx_attack",
            band="full_history_aggressive",
            source="S03_h14_expert5+S11_h10_longctx_transfer",
            selection_score="capacity-plus-context attack reserved for full-history only",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.5e-4, 6.5e-4),
            length=50,
            d_feat=20,
            expert=5,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([45, 50, 55]),
                "expert_scale": stage1._choice_spec([4, 5]),
            },
        ),
    ]


def _bank_foursquare() -> list[Dict[str, Any]]:
    return [
        _combo(
            "F01_h11_fast",
            band="prior_strong",
            source="F26_h11_fast_attack",
            selection_score="best observed foursquare test line with acceptable valid",
            anchor="H11",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(3.0e-4, 1.1e-3),
            length=20,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0],
            attn=[0.08],
        ),
        _combo(
            "F02_h2_zero_drop",
            band="prior_strong",
            source="F22_h2_zero_drop_validate",
            selection_score="zero-drop H2 line with solid valid and strong test",
            anchor="H2",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.6e-4, 6.0e-4),
            length=20,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[1.0, 2.0],
            attn=[0.10],
            extra_search={"stage_family_dropout_prob": _stage_choice([0.0])},
            family_drop=0.0,
            feature_drop=0.0,
        ),
        _combo(
            "F03_h7_feat32",
            band="prior_strong",
            source="F15_h7_feat32_refine",
            selection_score="feat32 foursquare winner with good balance between valid and test",
            anchor="H7",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.4e-4, 6.5e-4),
            length=20,
            d_feat=24,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0],
            attn=[0.08, 0.10],
            extra_search={"d_feat_emb": stage1._choice_spec([24, 32])},
        ),
        _combo(
            "F04_h5_len25",
            band="prior_strong",
            source="F16_h5_len25_refine",
            selection_score="compact H5 line with stable valid and near-top test",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 5.5e-4),
            length=25,
            d_feat=12,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.06, 0.08],
        ),
        _combo(
            "F05_fh_h11_ctx_lowlr",
            band="full_history_conservative",
            source="F26_h11_fast_attack",
            selection_score="H11 full-history retune with longer crop and lower lr ceiling",
            anchor="H11",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 7.0e-4),
            length=30,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30, 35])},
        ),
        _combo(
            "F06_fh_h2_ctx_regularized",
            band="full_history_conservative",
            source="F22_h2_zero_drop_validate",
            selection_score="zero-drop H2 line re-opened with modestly longer context and stronger wd",
            anchor="H2",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 5.0e-4),
            length=25,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.10],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30])},
            family_drop=0.0,
            feature_drop=0.0,
        ),
        _combo(
            "F07_fh_h3_midaux_attack",
            band="full_history_aggressive",
            source="F25_h3_midaux_attack",
            selection_score="mid-aux H3 attack kept for full-history sensitivity",
            anchor="H3",
            lambdas=(4e-4, 8e-5),
            lr_bounds=(2.2e-4, 8.0e-4),
            length=40,
            d_feat=12,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 45]),
            },
        ),
        _combo(
            "F08_fh_h15_auxlight_attack",
            band="full_history_aggressive",
            source="F27_h15_auxlight_attack",
            selection_score="aux-light H15 retry with broader context and wider upper lr",
            anchor="H15",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.4e-4, 9.0e-4),
            length=36,
            d_feat=12,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4]),
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 36, 40]),
            },
        ),
    ]


def _bank_retail_rocket() -> list[Dict[str, Any]]:
    return [
        _combo(
            "R01_h13_width_validate",
            band="prior_strong",
            source="R15_h13_width_lr_validate",
            selection_score="best retail line by both valid and test among the current portfolio",
            anchor="H13",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(4.5e-4, 1.4e-3),
            length=20,
            d_feat=16,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={"d_feat_emb": stage1._choice_spec([24])},
        ),
        _combo(
            "R02_h13_width_refine",
            band="prior_strong",
            source="R10_h13_width_refine",
            selection_score="width-up retail winner with top valid and strong test",
            anchor="H13",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(3.0e-4, 9.0e-4),
            length=20,
            d_feat=16,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
        ),
        _combo(
            "R03_h3_transfer",
            band="prior_strong",
            source="R07_h3_xfer_refine",
            selection_score="strong H3 transfer line with decent valid and durable test",
            anchor="H3",
            lambdas=(7e-4, 1.5e-4),
            lr_bounds=(2.8e-4, 8.0e-4),
            length=20,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.10, 0.12],
        ),
        _combo(
            "R04_h6_regularized",
            band="prior_strong",
            source="R02_h6_regularized",
            selection_score="regularized compact retail line that stayed competitive on both valid and test",
            anchor="H6",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.5e-4, 9.0e-4),
            length=20,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.08, 0.10],
        ),
        _combo(
            "R05_fh_h13_len25",
            band="full_history_conservative",
            source="R15_h13_width_lr_validate",
            selection_score="retail width family with slightly longer crop and reduced lr for full-history",
            anchor="H13",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.5e-4, 9.0e-4),
            length=25,
            d_feat=16,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25, 30])},
        ),
        _combo(
            "R06_fh_h3_len25",
            band="full_history_conservative",
            source="R07_h3_xfer_refine",
            selection_score="H3 retail transfer family re-opened with a modestly longer prefix window",
            anchor="H3",
            lambdas=(7e-4, 1.5e-4),
            lr_bounds=(2.2e-4, 7.0e-4),
            length=25,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.10],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25])},
        ),
        _combo(
            "R07_fh_h15_short_attack",
            band="full_history_aggressive",
            source="R17_h15_short_attack",
            selection_score="keep the short-context retail attack as a counter-hypothesis under longer history",
            anchor="H15",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(5.0e-4, 1.6e-3),
            length=20,
            d_feat=12,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 24, 28])},
        ),
        _combo(
            "R08_fh_h6_dimlift_attack",
            band="full_history_aggressive",
            source="R18_h6_dimlift_attack",
            selection_score="regularized H6 backbone with width lift under full-history",
            anchor="H6",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(3.0e-4, 1.1e-3),
            length=30,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={
                "d_feat_emb": stage1._choice_spec([16]),
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30, 35]),
            },
        ),
    ]


def _bank_movielens1m() -> list[Dict[str, Any]]:
    return [
        _combo(
            "M01_h6_compact_validate",
            band="prior_strong",
            source="M08_h6_compact_validate",
            selection_score="best movielens valid line in the current portfolio",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(3.5e-4, 1.1e-3),
            length=20,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={"expert_scale": stage1._choice_spec([2])},
        ),
        _combo(
            "M02_h6_e2_compact",
            band="prior_strong",
            source="M05_h6_e2_compact",
            selection_score="compact H6 transfer line with the strongest early movielens carry-over",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.5e-4, 1.0e-3),
            length=20,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0, 4.0],
            attn=[0.08, 0.10],
            extra_search={"expert_scale": stage1._choice_spec([1, 2])},
        ),
        _combo(
            "M03_h5_len25_transfer",
            band="prior_strong",
            source="M03_h5_len25_transfer",
            selection_score="compact H5 transfer family with not-bad valid and decent test",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 7.0e-4),
            length=25,
            d_feat=12,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.06, 0.08],
        ),
        _combo(
            "M04_h14_capacity_attack",
            band="prior_strong",
            source="M10_h14_capacity_attack",
            selection_score="capacity attack retained as the strongest wider-family movielens candidate",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(4.0e-4, 1.3e-3),
            length=20,
            d_feat=16,
            expert=3,
            hidden_mode="low",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={"d_feat_emb": stage1._choice_spec([20])},
        ),
        _combo(
            "M05_fh_h6_len35",
            band="full_history_conservative",
            source="M08_h6_compact_validate",
            selection_score="compact H6 line adjusted for longer full-history windows",
            anchor="H6",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.5e-4, 8.0e-4),
            length=35,
            d_feat=12,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 35, 40])},
        ),
        _combo(
            "M06_fh_h5_ctx",
            band="full_history_conservative",
            source="M03_h5_len25_transfer",
            selection_score="H5 transfer family with longer crop and slightly stronger regularization",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.6e-4, 5.5e-4),
            length=40,
            d_feat=12,
            expert=3,
            hidden_mode="low",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.06, 0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 50])},
        ),
        _combo(
            "M07_fh_h14_ctx_attack",
            band="full_history_aggressive",
            source="M10_h14_capacity_attack",
            selection_score="larger-capacity movielens family under longer prefixes",
            anchor="H14",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.5e-4, 1.1e-3),
            length=40,
            d_feat=20,
            expert=3,
            hidden_mode="low",
            wd_scales=[1.0, 2.0],
            attn=[0.08],
            extra_search={
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 45]),
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
            },
        ),
        _combo(
            "M08_fh_h1_highlr_attack",
            band="full_history_aggressive",
            source="M07_h1_highlr_validate",
            selection_score="keep one high-lr compact baseline as a full-history stress test",
            anchor="H1",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(6.0e-4, 2.5e-3),
            length=35,
            d_feat=16,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0],
            attn=[0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 35, 40])},
        ),
    ]


def _bank_lastfm() -> list[Dict[str, Any]]:
    return [
        _combo(
            "L01_h2_regularized",
            band="prior_strong",
            source="L04_h2_regularized",
            selection_score="best lastfm valid line with strong test retention",
            anchor="H2",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.8e-4, 6.0e-4),
            length=20,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.10, 0.12],
        ),
        _combo(
            "L02_h5_anchor",
            band="prior_strong",
            source="L01_h5_anchor",
            selection_score="minimal lastfm carry-over with the best observed test among the base four",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(2.0e-4, 7.0e-4),
            length=20,
            d_feat=16,
            expert=3,
            hidden_mode="low",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.06, 0.08],
        ),
        _combo(
            "L03_h11_fast",
            band="prior_strong",
            source="L03_h11_fast",
            selection_score="wide-cap sparse probe that stayed near the top on valid and test",
            anchor="H11",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.5e-4, 8.5e-4),
            length=20,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
        ),
        _combo(
            "L04_h15_light",
            band="prior_strong",
            source="L02_h15_light",
            selection_score="lighter sparse backup retained as the fourth prior for coverage",
            anchor="H15",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 6.5e-4),
            length=20,
            d_feat=12,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08, 0.10],
        ),
        _combo(
            "L05_fh_h2_ctx",
            band="full_history_conservative",
            source="L04_h2_regularized",
            selection_score="regularized sparse line adjusted to longer prefixes",
            anchor="H2",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(1.4e-4, 4.5e-4),
            length=35,
            d_feat=12,
            expert=3,
            hidden_mode="high",
            wd_scales=[2.0, 4.0, 8.0],
            attn=[0.10, 0.12],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 35, 40])},
        ),
        _combo(
            "L06_fh_h5_ctx",
            band="full_history_conservative",
            source="L01_h5_anchor",
            selection_score="steady H5 line with longer crop and slightly lower lr for full-history",
            anchor="H5",
            lambdas=(5e-4, 1e-4),
            lr_bounds=(1.6e-4, 5.0e-4),
            length=40,
            d_feat=16,
            expert=3,
            hidden_mode="low",
            wd_scales=[1.0, 2.0, 4.0],
            attn=[0.06, 0.08],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 50])},
        ),
        _combo(
            "L07_fh_h11_longctx_attack",
            band="full_history_aggressive",
            source="L03_h11_fast",
            selection_score="wide-cap sparse attack once full-history context is available",
            anchor="H11",
            lambdas=(8e-4, 2e-4),
            lr_bounds=(2.0e-4, 1.0e-3),
            length=50,
            d_feat=16,
            expert=2,
            hidden_mode="balanced",
            wd_scales=[0.5, 1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={"MAX_ITEM_LIST_LENGTH": stage1._choice_spec([45, 50, 55])},
        ),
        _combo(
            "L08_fh_h15_lowaux_attack",
            band="full_history_aggressive",
            source="L02_h15_light",
            selection_score="lighter sparse family re-opened with lower aux and broader lr",
            anchor="H15",
            lambdas=(2.5e-4, 5e-5),
            lr_bounds=(2.0e-4, 8.0e-4),
            length=40,
            d_feat=12,
            expert=3,
            hidden_mode="balanced",
            wd_scales=[1.0, 2.0],
            attn=[0.08, 0.10],
            extra_search={
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4]),
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([35, 40, 45]),
            },
        ),
    ]


DATASET_BANKS = {
    "beauty": _bank_beauty,
    "KuaiRecLargeStrictPosV2_0.2": _bank_kuai,
    "foursquare": _bank_foursquare,
    "retail_rocket": _bank_retail_rocket,
    "movielens1m": _bank_movielens1m,
    "lastfm0.03": _bank_lastfm,
}


def _dataset_budget(dataset: str) -> Dict[str, int]:
    if dataset not in DATASET_BUDGETS:
        raise RuntimeError(f"missing dataset budget for {dataset}")
    return dict(DATASET_BUDGETS[dataset])


def _choose_batches(dataset: str, combo: Dict[str, Any]) -> tuple[int, int]:
    budget = _dataset_budget(dataset)
    train_batch_size = int(budget["train_batch_size"])
    eval_batch_size = int(budget["eval_batch_size"])
    length = int(combo["len"])
    anchor = str(combo["anchor"])
    d_feat = int(combo["d_feat"])
    expert = int(combo["expert"])
    if length >= 40:
        train_batch_size = max(512, train_batch_size // 2)
        eval_batch_size = max(1024, eval_batch_size // 2)
    if length >= 50:
        train_batch_size = max(512, train_batch_size // 2)
        eval_batch_size = max(1024, eval_batch_size // 2)
    if anchor in {"H11", "H14"} or d_feat >= 24 or expert >= 4:
        train_batch_size = max(512, int(train_batch_size * 0.75))
        eval_batch_size = max(1024, int(eval_batch_size * 0.75))
    return int(train_batch_size), int(eval_batch_size)


def _build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    datasets = _parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets selected")
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")

    rows: list[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        if dataset not in DATASET_BANKS:
            raise RuntimeError(f"Unsupported dataset for full-history tuning: {dataset}")
        stage1._validate_session_fixed_files(dataset)
        combos = list(DATASET_BANKS[dataset]())
        if len(combos) != 8:
            raise RuntimeError(f"Expected 8 combos for dataset={dataset}, got {len(combos)}")
        budget = _dataset_budget(dataset)
        for combo in combos:
            anchor = str(combo["anchor"])
            cfg = dict(stage1._anchor_cfg(anchor))
            family_drop = combo.get("family_drop")
            if family_drop is None:
                family_drop = 0.04 if str(combo["hidden_mode"]) == "high" else 0.03
            feature_drop = combo.get("feature_drop")
            if feature_drop is None:
                feature_drop = 0.0
            overrides = stage1._build_overrides(
                float(combo["lambda"][0]),
                float(combo["lambda"][1]),
                float(family_drop),
                float(feature_drop),
            )
            fixed_values: Dict[str, Any] = {
                "embedding_size": int(cfg["embedding_size"]),
                "d_ff": int(cfg["d_ff"]),
                "d_expert_hidden": int(cfg["d_expert_hidden"]),
                "d_router_hidden": int(cfg["d_router_hidden"]),
                "MAX_ITEM_LIST_LENGTH": int(combo["len"]),
                "d_feat_emb": int(combo["d_feat"]),
                "expert_scale": int(combo["expert"]),
                "num_heads": 4,
                "lr_scheduler_type": "warmup_cosine",
                "attn_dropout_prob": float(combo["attn"][0]),
            }
            search_space: Dict[str, Any] = {
                "learning_rate": stage1._loguniform_spec(float(combo["lr_bounds"][0]), float(combo["lr_bounds"][1])),
                "weight_decay": stage1._choice_spec(stage1._weight_decay_choices(anchor, list(combo["wd_scales"]))),
                "hidden_dropout_prob": stage1._choice_spec(stage1._hidden_choices(anchor, str(combo["hidden_mode"]))),
            }
            search_space.update(dict(combo.get("extra_search", {}) or {}))

            train_batch_size, eval_batch_size = _choose_batches(dataset, combo)
            for seed_id in seeds:
                cursor += 1
                run_id = f"FH_{sanitize_token(dataset, upper=True)}_{sanitize_token(str(combo['id']), upper=True)}_S{int(seed_id)}"
                rows.append(
                    {
                        "dataset": dataset,
                        "phase_id": PHASE_ID,
                        "axis_id": AXIS_ID,
                        "axis_desc": AXIS_DESC,
                        "architecture_id": ARCH_ID,
                        "architecture_key": ARCH_KEY,
                        "architecture_name": ARCH_NAME,
                        "exp_brief": ARCH_NAME,
                        "run_phase": f"{PHASE_ID}_{run_id}",
                        "run_id": run_id,
                        "setting_id": str(combo["id"]),
                        "setting_key": str(combo["id"]),
                        "setting_desc": str(combo["selection_score"]),
                        "stage": RUN_STAGE,
                        "tuning_stage": RUN_STAGE,
                        "family_id": str(combo["id"]),
                        "family_group": "full_history_combo",
                        "variant_id": str(combo["band"]),
                        "capacity_anchor": anchor,
                        "selected_from_stage": str(combo["source"]),
                        "selection_score": str(combo["selection_score"]),
                        "search_algo": str(args.search_algo),
                        "seed_id": int(seed_id),
                        "runtime_seed": int(args.seed_base) + cursor - 1,
                        "stage_group": RUN_STAGE,
                        "source_family_id": str(combo["source"]),
                        "template_count": 8,
                        "combo_band": str(combo["band"]),
                        "main_eval_target_mode": MAIN_EVAL_TARGET_MODE,
                        "history_input_mode": HISTORY_INPUT_MODE,
                        "history_eval_policy": HISTORY_EVAL_POLICY,
                        "aux_route_consistency_lambda": float(combo["lambda"][0]),
                        "aux_z_loss_lambda": float(combo["lambda"][1]),
                        "fixed_values": fixed_values,
                        "search_space": search_space,
                        "overrides": overrides,
                        "train_batch_size": train_batch_size,
                        "eval_batch_size": eval_batch_size,
                        "max_evals": int(budget["max_evals"]),
                        "tune_epochs": 100,
                        "tune_patience": 10,
                    }
                )
    return rows


def _serialize_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "dataset",
        "phase_id",
        "axis_id",
        "axis_desc",
        "architecture_id",
        "architecture_key",
        "architecture_name",
        "exp_brief",
        "run_phase",
        "run_id",
        "setting_id",
        "setting_key",
        "setting_desc",
        "stage",
        "tuning_stage",
        "family_id",
        "family_group",
        "variant_id",
        "capacity_anchor",
        "selected_from_stage",
        "selection_score",
        "search_algo",
        "seed_id",
        "runtime_seed",
        "stage_group",
        "source_family_id",
        "template_count",
        "combo_band",
        "main_eval_target_mode",
        "history_input_mode",
        "history_eval_policy",
        "aux_route_consistency_lambda",
        "aux_z_loss_lambda",
        "fixed_values",
        "search_space",
        "overrides",
        "train_batch_size",
        "eval_batch_size",
        "max_evals",
        "tune_epochs",
        "tune_patience",
    }
    return {key: row.get(key) for key in keep if key in row}


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        raw = Path(str(args.manifest_out))
        if raw.suffix:
            return raw
        return raw / "full_history_manifest.json"
    return LOG_ROOT / "full_history_manifest.json"


def _write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": RUN_STAGE,
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_count": len(rows),
        "datasets": [str(row["dataset"]) for row in rows if str(row.get("seed_id")) == "1"],
        "rows": [_serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _summary_path(dataset: str) -> Path:
    path = LOG_ROOT / str(dataset) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_log_path(*, log_dir: Path, row: Dict[str, Any], phase_id: str) -> Path:
    dataset_dir = log_dir / str(row["dataset"])
    combo_dir = dataset_dir / str(row["family_id"])
    filename = f"{phase_id}_{sanitize_token(str(row['family_id']), upper=True)}_S{int(row['seed_id'])}.log"
    return combo_dir / filename


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = str(Path("/venv/FMoE/bin/python"))
    if not Path(python_bin).exists():
        python_bin = sys.executable

    fixed_values = dict(row.get("fixed_values", {}) or {})
    search_space = dict(row.get("search_space", {}) or {})

    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(row["search_algo"]),
        "--max-evals",
        str(int(row["max_evals"])),
        "--tune-epochs",
        "100",
        "--tune-patience",
        "10",
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        f"feature_mode={FEATURE_MODE}",
        f"++history_input_mode={HISTORY_INPUT_MODE}",
        f"++history_eval_policy={HISTORY_EVAL_POLICY}",
        "++history_group_field=user_id",
        "++target_group_field=session_id",
        f"++load_col.inter={hydra_literal(['session_id', 'item_id', 'timestamp', 'user_id'])}",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++special_logging=true",
        "++exclude_unseen_target_from_main_eval=true",
        "++log_unseen_target_metrics=true",
        f"++main_eval_target_mode={hydra_literal(MAIN_EVAL_TARGET_MODE)}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_logging_layout={hydra_literal('axis_dataset_arch_hparam')}",
        f"++fmoe_architecture_id={hydra_literal(ARCH_ID)}",
        f"++fmoe_architecture_key={hydra_literal(ARCH_KEY)}",
        f"++fmoe_hparam_id={hydra_literal(row['capacity_anchor'])}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"train_batch_size={int(row['train_batch_size'])}",
        f"eval_batch_size={int(row['eval_batch_size'])}",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(AXIS_ID)}",
        f"++phase_axis_desc={hydra_literal(AXIS_DESC)}",
        f"++phase_setting_id={hydra_literal(str(row['family_id']))}",
        f"++phase_setting_key={hydra_literal(str(row['family_id']))}",
        f"++phase_hparam_id={hydra_literal(str(row['capacity_anchor']))}",
        f"++phase_seed_id={hydra_literal(int(row['seed_id']))}",
        f"++phase_run_id={hydra_literal(str(row['run_id']))}",
    ]

    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, value in fixed_values.items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, spec in search_space.items():
        search_type, values = stage1._normalize_search_spec(spec)
        cmd.append(f"++search.{key}={hydra_literal(values)}")
        cmd.append(f"++search_space_type_overrides.{key}={search_type}")
    return cmd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 full-history tuning across six feature_added_v4 datasets")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=260500)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=4)
    return parser.parse_args()


def _maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 4) or 4))])


def main() -> int:
    args = _parse_args()
    rows = _maybe_limit_smoke(_build_rows(args), args)
    manifest = _write_manifest(args, rows)
    print(f"[full-history] manifest -> {manifest}")

    fieldnames = build_summary_fieldnames(
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
            "combo_band",
            "main_eval_target_mode",
            "history_input_mode",
            "history_eval_policy",
            "aux_route_consistency_lambda",
            "aux_z_loss_lambda",
        ]
    )

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    return int(
        launch_wide_rows(
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
            build_command=_build_command,
            build_log_path=_build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: _summary_path(str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())