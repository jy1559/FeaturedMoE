#!/usr/bin/env python3
"""Registration/config tests for FeaturedMoE_HGR."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_hgr_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_hgr",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "hgr" in model_name
    assert cfg.get("group_router_mode") in {"per_group", "stage_wide", "hybrid"}
    assert "group_top_k" in cfg
    assert cfg.get("stage_merge_mode") in {"serial", "parallel"}


def test_recbole_patch_contains_hgr_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_HGR" in text
    assert "'featured_moe_hgr'" in text
    assert "'featuredmoe_hgr'" in text
