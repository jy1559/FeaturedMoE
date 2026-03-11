#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_HGRv4 core modules."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGES, STAGE_ALL_FEATURES, build_column_to_index  # noqa: E402
from models.FeaturedMoE_HGRv4.hgr_v4_moe_stages import (  # noqa: E402
    GroupStatSoftRuleTeacher,
    HierarchicalGroupStageMoEv4,
)


def _build_stage(
    *,
    stage_name: str = "macro",
    expert_scale: int = 4,
    inner_rule_mode: str = "distill",
    outer_router_design: str = "legacy_concat",
    inner_router_design: str = "legacy_concat",
):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)[stage_name]
    return HierarchicalGroupStageMoEv4(
        stage_name=stage_name,
        group_names=list(stage_map.keys()),
        group_feature_lists=list(stage_map.values()),
        stage_all_features=STAGE_ALL_FEATURES[stage_name],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=24,
        d_router_hidden=12,
        expert_scale=expert_scale,
        expert_top_k=1,
        group_top_k=0,
        dropout=0.1,
        expert_use_hidden=True,
        expert_use_feature=False,
        router_design="group_factorized_interaction",
        router_mode="token",
        session_pooling="query",
        router_temperature=1.0,
        router_feature_dropout=0.0,
        reliability_feature_name=None,
        outer_router_use_hidden=True,
        outer_router_use_feature=True,
        inner_router_use_hidden=True,
        inner_router_use_feature=True,
        outer_router_design=outer_router_design,
        inner_router_design=inner_router_design,
        inner_rule_enable=True,
        inner_rule_mode=inner_rule_mode,
        inner_rule_bias_scale=1.0,
        inner_rule_bin_sharpness=16.0,
        inner_rule_group_feature_pool="mean_ratio",
        inner_rule_apply=True,
        group_router_mode="hybrid",
        inner_rule_teacher_kind="group_stat_soft",
    )


def test_hgr_v4_group_stat_teacher_shapes():
    torch.manual_seed(21)
    teacher = GroupStatSoftRuleTeacher(expert_scale=4, sharpness=16.0)
    group_feat = torch.rand(2, 5, 4)
    valid_mask = torch.ones(2, 5, dtype=torch.bool)
    logits, stats, signature = teacher(
        group_feat=group_feat,
        valid_mask=valid_mask,
        item_seq_len=None,
        router_mode="token",
        session_pooling="query",
    )

    assert logits.shape == (2, 5, 4)
    assert stats.shape == (2, 5, 6)
    assert signature.shape == (4, 6)


def test_hgr_v4_feature_aware_outer_and_teacher_aux_shapes():
    torch.manual_seed(22)
    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(stage_name="mid", inner_rule_mode="distill")

    next_hidden, gate_w, gate_l, group_w, group_l, stage_delta = stage(hidden, feat)
    aux = stage.last_router_aux

    assert next_hidden.shape == hidden.shape
    assert stage_delta.shape == hidden.shape
    assert group_w.shape == (2, 5, 4)
    assert group_l.shape == (2, 5, 4)
    assert gate_w.shape == (2, 5, 16)
    assert gate_l.shape == (2, 5, 16)
    assert aux["teacher_intra_group_logits"].shape == (2, 5, 4, 4)
    assert aux["inner_rule_score"].shape == (2, 5, 4, 6)
    assert aux["inner_rule_bin_centers"].shape == (4, 6)
    assert torch.allclose(group_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_hgr_v4_mixed_router_designs_work():
    torch.manual_seed(24)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(
        stage_name="mid",
        inner_rule_mode="distill",
        outer_router_design="legacy_concat",
        inner_router_design="group_factorized_interaction",
    )
    _, gate_w, _, group_w, _, _ = stage(hidden, feat)
    assert gate_w.shape == (2, 4, 16)
    assert group_w.shape == (2, 4, 4)


def test_hgr_v4_fused_bias_changes_inner_logits_only():
    torch.manual_seed(23)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(stage_name="micro", inner_rule_mode="distill_and_fused_bias")
    _, _, _, _, _, _ = stage(hidden, feat)
    aux = stage.last_router_aux

    assert aux["intra_group_logits"].shape == aux["intra_group_logits_raw"].shape
    assert not torch.allclose(aux["intra_group_logits"], aux["intra_group_logits_raw"])
