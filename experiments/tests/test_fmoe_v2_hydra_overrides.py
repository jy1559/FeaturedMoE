#!/usr/bin/env python3
"""Hydra compose smoke tests for bare FeaturedMoE_v2 router overrides."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from hydra_utils import load_hydra_config  # noqa: E402


def test_featured_moe_v2_accepts_bare_router_overrides():
    cfg = load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_v2",
            "dataset=retail_rocket",
            "eval_mode=session",
            "feature_mode=full_v2",
            "router_design=group_factorized_interaction",
            "group_top_k=0",
            "expert_top_k=1",
            "router_distill_enable=false",
            "router_distill_lambda=0.0",
            "router_distill_temperature=1.5",
            "router_distill_until=0.2",
            "router_impl=learned",
            "router_use_hidden=true",
            "router_use_feature=true",
            "expert_use_hidden=true",
            "expert_use_feature=true",
            "macro_routing_scope=session",
            "macro_session_pooling=query",
            "mid_router_temperature=1.3",
            "micro_router_temperature=1.3",
            "mid_router_feature_dropout=0.1",
            "micro_router_feature_dropout=0.1",
            "use_valid_ratio_gating=true",
        ],
    )

    assert cfg["router_design"] == "group_factorized_interaction"
    assert cfg["group_top_k"] == 0
    assert cfg["expert_top_k"] == 1
    assert cfg["router_distill_enable"] is False
    assert cfg["router_impl"] == "learned"
    assert cfg["macro_routing_scope"] == "session"
