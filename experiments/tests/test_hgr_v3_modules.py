#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_HGRv3 core modules."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGES, STAGE_ALL_FEATURES, build_column_to_index  # noqa: E402
from models.FeaturedMoE_HGRv3.hgr_v3_moe_stages import (  # noqa: E402
    InnerBinRuleTeacher,
    HierarchicalGroupStageMoEv3,
)
from models.FeaturedMoE_HGRv3.losses import compute_inner_rule_distill_aux_loss  # noqa: E402


def _build_stage(
    *,
    stage_name: str = "macro",
    expert_scale: int = 3,
    inner_rule_mode: str = "distill",
    router_mode: str = "token",
    session_pooling: str = "query",
):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)[stage_name]
    return HierarchicalGroupStageMoEv3(
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
        router_mode=router_mode,
        session_pooling=session_pooling,
        router_temperature=1.0,
        router_feature_dropout=0.0,
        reliability_feature_name=None,
        outer_router_use_hidden=True,
        outer_router_use_feature=False,
        inner_router_use_hidden=True,
        inner_router_use_feature=True,
        inner_rule_enable=True,
        inner_rule_mode=inner_rule_mode,
        inner_rule_bias_scale=1.0,
        inner_rule_bin_sharpness=16.0,
        inner_rule_group_feature_pool="mean_ratio",
        inner_rule_apply=True,
    )


def test_hgr_v3_outer_hidden_only_and_inner_teacher_shapes():
    torch.manual_seed(11)
    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(stage_name="mid", inner_rule_mode="distill")

    next_hidden, gate_w, gate_l, group_w, group_l, stage_delta = stage(hidden, feat)
    aux = stage.last_router_aux

    assert next_hidden.shape == hidden.shape
    assert stage_delta.shape == hidden.shape
    assert group_w.shape == (2, 5, 4)
    assert group_l.shape == (2, 5, 4)
    assert gate_w.shape == (2, 5, 12)
    assert gate_l.shape == (2, 5, 12)
    assert aux["teacher_intra_group_logits"].shape == (2, 5, 4, 3)
    assert aux["inner_rule_score"].shape == (2, 5, 4, 1)
    assert aux["inner_rule_bin_centers"].shape == (3,)
    assert torch.allclose(group_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_hgr_v3_expert_top1_keeps_one_clone_per_group():
    torch.manual_seed(12)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(stage_name="micro", expert_scale=3, inner_rule_mode="distill")
    _, gate_w, _, _, _, _ = stage(hidden, feat)

    per_group = gate_w.view(2, 4, 4, 3)
    nonzero_per_group = (per_group > 0).sum(dim=-1)
    assert torch.all(nonzero_per_group <= 1)


def test_hgr_v3_distill_and_fused_bias_changes_only_inner_logits():
    torch.manual_seed(13)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))

    stage_off = _build_stage(stage_name="macro", inner_rule_mode="off")
    stage_fused = _build_stage(stage_name="macro", inner_rule_mode="distill_and_fused_bias")

    _, _, _, group_w_off, group_l_off, _ = stage_off(hidden, feat)
    aux_off = stage_off.last_router_aux
    _, _, _, group_w_fused, group_l_fused, _ = stage_fused(hidden, feat)
    aux_fused = stage_fused.last_router_aux

    assert group_w_off.shape == group_w_fused.shape
    assert group_l_off.shape == group_l_fused.shape
    assert aux_off["intra_group_logits_raw"].shape == aux_fused["intra_group_logits_raw"].shape
    assert not torch.allclose(aux_fused["intra_group_logits"], aux_fused["intra_group_logits_raw"])


def test_hgr_v3_teacher_handles_expert_scale_one():
    torch.manual_seed(14)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage(stage_name="mid", expert_scale=1, inner_rule_mode="distill")
    _, gate_w, _, group_w, _, _ = stage(hidden, feat)
    aux = stage.last_router_aux

    assert gate_w.shape == (2, 4, 4)
    assert torch.allclose(gate_w, group_w, atol=1e-6)
    assert aux["teacher_intra_group_logits"].shape == (2, 4, 4, 1)


def test_hgr_v3_inner_bin_teacher_and_distill_loss_are_finite():
    torch.manual_seed(15)
    teacher = InnerBinRuleTeacher(expert_scale=3, bin_sharpness=12.0)
    group_feat = torch.randn(2, 5, 4)
    valid_mask = torch.ones(2, 5, dtype=torch.bool)
    logits, score, centers = teacher(
        group_feat=group_feat,
        valid_mask=valid_mask,
        item_seq_len=None,
        router_mode="token",
        session_pooling="query",
    )

    student = torch.randn(2, 5, 1, 3).expand(-1, -1, 4, -1).contiguous()
    teacher4 = logits.unsqueeze(-2).expand(-1, -1, 4, -1).contiguous()
    loss = compute_inner_rule_distill_aux_loss(
        teacher_intra_group_logits={"mid": teacher4},
        student_intra_group_logits_raw={"mid": student},
        item_seq_len=None,
        aux_lambda=5e-3,
        distill_temperature=1.5,
        enabled=True,
        progress=0.0,
        until=0.2,
        device=student.device,
    )

    assert logits.shape == (2, 5, 3)
    assert score.shape == (2, 5, 1)
    assert centers.shape == (3,)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
