#!/usr/bin/env python3
"""Registration/config tests for FeaturedMoE_Individual."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_individual_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_individual",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "individual" in model_name
    assert cfg.get("feature_top_k") == 4
    assert cfg.get("inner_expert_top_k") == 0
    assert cfg.get("outer_router_use_hidden") is True
    assert cfg.get("outer_router_use_feature") is True
    assert cfg.get("use_aux_loss") is False


def test_recbole_patch_contains_individual_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_Individual" in text
    assert "'featured_moe_individual'" in text
    assert "'featuredmoe_individual'" in text
