#!/usr/bin/env python3
"""Hydra compose smoke tests for FeaturedMoE_v4_Distillation."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from hydra_utils import load_hydra_config  # noqa: E402


def test_featured_moe_v4_distillation_accepts_rule_teacher_and_variant_overrides():
    cfg = load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_v4_distillation_tune",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
            "router_design=flat_legacy",
            "router_clone_residual_scale=0.20",
            "teacher_design=rule_soft12",
            "teacher_delivery=distill_and_fused_bias",
            "teacher_stage_mask=mid_micro_only",
            "teacher_kl_lambda=0.002",
            "teacher_bias_scale=0.20",
            "teacher_temperature=1.5",
            "teacher_until=0.25",
            "teacher_stat_sharpness=16.0",
            "router_impl=learned",
            "++router_impl_by_stage={mid:rule_soft,micro:rule_soft}",
            "rule_router.variant=teacher_gls",
            "rule_router.n_bins=10",
            "rule_router.feature_per_expert=4",
            "++search={rule_router.variant:[teacher_gls],rule_router.n_bins:[10],rule_router.feature_per_expert:[4]}",
            "moe_top_k=0",
        ],
    )

    assert cfg["router_design"] == "flat_legacy"
    assert cfg["router_clone_residual_scale"] == 0.20
    assert cfg["teacher_design"] == "rule_soft12"
    assert cfg["teacher_delivery"] == "distill_and_fused_bias"
    assert cfg["teacher_stage_mask"] == "mid_micro_only"
    assert cfg["teacher_kl_lambda"] == 0.002
    assert cfg["teacher_bias_scale"] == 0.20
    assert cfg["teacher_temperature"] == 1.5
    assert cfg["teacher_until"] == 0.25
    assert cfg["teacher_stat_sharpness"] == 16.0
    assert cfg["router_impl"] == "learned"
    assert cfg["router_impl_by_stage"]["mid"] == "rule_soft"
    assert cfg["router_impl_by_stage"]["micro"] == "rule_soft"
    assert cfg["rule_router"]["variant"] == "teacher_gls"
    assert cfg["rule_router"]["n_bins"] == 10
    assert cfg["rule_router"]["feature_per_expert"] == 4
    assert cfg["search"]["rule_router.variant"] == ["teacher_gls"]
    assert cfg["search"]["rule_router.n_bins"] == [10]
    assert cfg["search"]["rule_router.feature_per_expert"] == [4]
    assert cfg["moe_top_k"] == 0
