#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_v2 config resolver additions."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "models" / "FeaturedMoE_v2"))

from config_schema import ConfigResolver  # noqa: E402


def test_config_resolver_reads_factorized_router_keys_from_groups():
    resolver = ConfigResolver(
        {
            "model_core": {"embedding_size": 128},
            "routing_common": {
                "router_design": "group_factorized_interaction",
                "group_top_k": 0,
                "expert_top_k": 1,
                "router_distill_enable": False,
                "router_distill_lambda": 0.0,
                "router_distill_temperature": 1.5,
                "router_distill_until": 0.2,
            },
            "loss_regularization": {
                "balance_loss_lambda": 0.003,
                "fmoe_v2_feature_spec_aux_enable": True,
                "fmoe_v2_feature_spec_aux_lambda": 3e-4,
                "fmoe_v2_feature_spec_stages": ["mid"],
                "fmoe_v2_feature_spec_min_tokens": 8,
            },
        }
    )

    assert resolver.get("router_design") == "group_factorized_interaction"
    assert resolver.get("group_top_k") == 0
    assert resolver.get("expert_top_k") == 1
    assert resolver.get("router_distill_enable") is False
    assert resolver.get("router_distill_temperature") == 1.5
    assert resolver.get("balance_loss_lambda") == 0.003
    assert resolver.get("fmoe_v2_feature_spec_aux_enable") is True
    assert resolver.get("fmoe_v2_feature_spec_stages") == ["mid"]


def test_config_resolver_keeps_legacy_defaults_usable():
    resolver = ConfigResolver({"embedding_size": 64})

    assert resolver.get("router_design", "flat_legacy") == "flat_legacy"
    assert resolver.get("group_top_k", 0) == 0
    assert resolver.get("expert_top_k", 1) == 1
    assert resolver.get("router_distill_enable", False) is False
