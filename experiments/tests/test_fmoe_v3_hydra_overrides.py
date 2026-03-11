#!/usr/bin/env python3
"""Hydra compose smoke tests for FeaturedMoE_v3 router/distill overrides."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from hydra_utils import load_hydra_config  # noqa: E402


def test_featured_moe_v3_accepts_flat_router_and_distill_overrides():
    cfg = load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_v3_tune",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
            "router_design=flat_group_clone_combo",
            "router_group_bias_scale=0.5",
            "router_clone_residual_scale=0.5",
            "router_distill_enable=true",
            "router_distill_mode=clone_only",
            "router_distill_lambda_group=0.0",
            "router_distill_lambda_clone=0.003",
            "router_distill_temperature=1.5",
            "router_distill_until=0.2",
            "router_impl=learned",
            "++router_impl_by_stage={}",
            "moe_top_k=2",
        ],
    )

    assert cfg["router_design"] == "flat_group_clone_combo"
    assert cfg["router_group_bias_scale"] == 0.5
    assert cfg["router_clone_residual_scale"] == 0.5
    assert cfg["router_distill_enable"] is True
    assert cfg["router_distill_mode"] == "clone_only"
    assert cfg["router_distill_lambda_group"] == 0.0
    assert cfg["router_distill_lambda_clone"] == 0.003
    assert cfg["router_impl"] == "learned"
    assert cfg["moe_top_k"] == 2
