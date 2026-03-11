#!/usr/bin/env python3
"""Registration/config tests for FeaturedMoE_HGRv3."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_hgr_v3_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_hgr_v3",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "hgrv3" in model_name or "hgr_v3" in model_name
    assert cfg.get("group_router_mode") == "stage_wide"
    assert cfg.get("router_design") == "group_factorized_interaction"
    assert cfg.get("outer_router_use_hidden") is True
    assert cfg.get("outer_router_use_feature") is False
    assert cfg.get("inner_rule_mode") in {"off", "distill", "fused_bias", "distill_and_fused_bias"}


def test_recbole_patch_contains_hgr_v3_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_HGRv3" in text
    assert "'featured_moe_hgr_v3'" in text
    assert "'featuredmoe_hgr_v3'" in text
