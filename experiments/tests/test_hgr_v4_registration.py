#!/usr/bin/env python3
"""Registration/config tests for FeaturedMoE_HGRv4."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_hgr_v4_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_hgr_v4",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "hgrv4" in model_name or "hgr_v4" in model_name
    assert cfg.get("group_router_mode") in {"stage_wide", "per_group", "hybrid"}
    assert cfg.get("outer_router_design") in {"legacy_concat", "group_factorized_interaction"}
    assert cfg.get("inner_router_design") in {"legacy_concat", "group_factorized_interaction"}
    assert cfg.get("outer_router_use_hidden") is True
    assert cfg.get("outer_router_use_feature") is True
    assert cfg.get("inner_rule_teacher_kind") == "group_stat_soft"


def test_recbole_patch_contains_hgr_v4_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_HGRv4" in text
    assert "'featured_moe_hgr_v4'" in text
    assert "'featuredmoe_hgr_v4'" in text
