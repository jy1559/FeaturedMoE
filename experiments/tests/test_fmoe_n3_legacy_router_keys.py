#!/usr/bin/env python3
"""Legacy router config guard tests for FeaturedMoE_N3."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.FeaturedMoE_N3.featured_moe_n3 import FeaturedMoE_N3  # noqa: E402


class _DummyConfig:
    def __init__(self, payload):
        self.final_config_dict = dict(payload)


def test_legacy_router_key_is_rejected():
    cfg = _DummyConfig({"stage_router_type": {"macro": "standard"}})
    with pytest.raises(ValueError, match="Legacy router config keys"):
        FeaturedMoE_N3._assert_no_legacy_router_keys(cfg)


def test_empty_legacy_router_key_is_ignored():
    cfg = _DummyConfig({"stage_router_type": {}})
    FeaturedMoE_N3._assert_no_legacy_router_keys(cfg)
