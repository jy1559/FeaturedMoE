#!/usr/bin/env python3
"""Registration/config smoke tests for FeaturedMoE_N."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

pytest.importorskip("omegaconf")
from hydra_utils import load_hydra_config  # noqa: E402


def test_fmoe_n_hydra_config_load():
    cfg = load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_n",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )

    assert str(cfg.get("model", "")).lower() in {"featuredmoe_n", "featured_moe_n"}
    assert cfg["router_design"] == "simple_flat"
    assert cfg["feature_encoder_mode"] == "linear"
    assert cfg["fmoe_special_logging"] is True


def test_recbole_patch_contains_n_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_N" in text
    assert "'featured_moe_n'" in text
    assert "'featuredmoe_n'" in text
