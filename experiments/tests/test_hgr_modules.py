#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_HGR core modules."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    MID_ALL_FEATURES,
    MID_EXPERTS,
    STAGE_ALL_FEATURES,
    STAGES,
    build_column_to_index,
)
from models.FeaturedMoE_HGR.hgr_moe_stages import (  # noqa: E402
    HiddenExpertMLP,
    HierarchicalGroupStageMoE,
    HierarchicalMoEHGR,
)
from models.FeaturedMoE_HiR.hir_moe_stages import HierarchicalStageMoE  # noqa: E402


def _build_stage(
    stage_name: str = "macro",
    mode: str = "per_group",
    expert_scale: int = 2,
    *,
    expert_use_feature: bool = False,
    router_mode: str = "token",
    session_pooling: str = "query",
):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)[stage_name]
    return HierarchicalGroupStageMoE(
        stage_name=stage_name,
        group_names=list(stage_map.keys()),
        group_feature_lists=list(stage_map.values()),
        stage_all_features=STAGE_ALL_FEATURES[stage_name],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=expert_scale,
        expert_top_k=2,
        group_top_k=2,
        group_router_mode=mode,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=expert_use_feature,
        router_mode=router_mode,
        session_pooling=session_pooling,
        router_temperature=1.0,
        router_feature_dropout=0.1,
        reliability_feature_name=None,
    )


def test_hgr_group_router_modes_keep_simplex():
    torch.manual_seed(123)
    bsz, tlen, d_model = 3, 5, 16
    n_feat = len(ALL_FEATURE_COLUMNS)
    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, n_feat)

    for mode in ("per_group", "stage_wide", "hybrid"):
        stage = _build_stage(stage_name="macro", mode=mode, expert_scale=2)
        next_hidden, gate_w, gate_l, group_w, group_l, stage_delta = stage(hidden, feat)

        assert next_hidden.shape == hidden.shape
        assert stage_delta.shape == hidden.shape
        assert group_w.shape == (bsz, tlen, 4)
        assert group_l.shape == group_w.shape
        assert gate_w.shape == (bsz, tlen, 8)
        assert gate_l.shape == gate_w.shape
        assert torch.allclose(group_w.sum(dim=-1), torch.ones(bsz, tlen), atol=1e-6)
        assert torch.allclose(gate_w.sum(dim=-1), torch.ones(bsz, tlen), atol=1e-6)
        assert torch.isfinite(next_hidden).all()
        assert torch.isfinite(gate_w).all()


def test_hgr_stage_wide_matches_hir_shape_contract():
    torch.manual_seed(124)
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))

    hgr_stage = HierarchicalGroupStageMoE(
        stage_name="mid",
        group_names=list(MID_EXPERTS.keys()),
        group_feature_lists=list(MID_EXPERTS.values()),
        stage_all_features=MID_ALL_FEATURES,
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=2,
        expert_top_k=2,
        group_top_k=0,
        group_router_mode="stage_wide",
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=False,
        router_mode="token",
        session_pooling="query",
        router_temperature=1.0,
        router_feature_dropout=0.1,
        reliability_feature_name=None,
    )
    hir_stage = HierarchicalStageMoE(
        stage_name="mid",
        bundle_names=list(MID_EXPERTS.keys()),
        bundle_feature_lists=list(MID_EXPERTS.values()),
        stage_all_features=MID_ALL_FEATURES,
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=2,
        expert_top_k=2,
        bundle_top_k=0,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        router_temperature=1.0,
        router_feature_dropout=0.1,
        reliability_feature_name=None,
    )

    hgr_out = hgr_stage(hidden, feat)
    hir_out = hir_stage(hidden, feat)

    assert hgr_out[0].shape == hir_out[0].shape == hidden.shape
    assert hgr_out[1].shape == hir_out[1].shape == (2, 4, 8)
    assert hgr_out[3].shape == hir_out[3].shape == (2, 4, 4)
    assert all(isinstance(expert, HiddenExpertMLP) for expert in hgr_stage.experts)


def test_hgr_macro_session_router_broadcasts_weights():
    torch.manual_seed(126)
    bsz, tlen, d_model = 2, 5, 16
    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 3], dtype=torch.long)

    stage = _build_stage(
        stage_name="macro",
        mode="per_group",
        expert_scale=2,
        expert_use_feature=False,
        router_mode="session",
        session_pooling="query",
    )
    _, gate_w, _, group_w, _, _ = stage(hidden, feat, item_seq_len=item_seq_len)

    assert gate_w.shape == (bsz, tlen, 8)
    assert group_w.shape == (bsz, tlen, 4)
    assert torch.allclose(group_w, group_w[:, :1, :].expand_as(group_w), atol=1e-6)
    assert torch.allclose(gate_w, gate_w[:, :1, :].expand_as(gate_w), atol=1e-6)


def test_hgr_feature_aware_experts_keep_simplex():
    torch.manual_seed(127)
    bsz, tlen, d_model = 2, 4, 16
    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, len(ALL_FEATURE_COLUMNS))

    stage = _build_stage(
        stage_name="mid",
        mode="hybrid",
        expert_scale=2,
        expert_use_feature=True,
    )
    next_hidden, gate_w, gate_l, group_w, group_l, stage_delta = stage(hidden, feat)

    assert next_hidden.shape == hidden.shape
    assert stage_delta.shape == hidden.shape
    assert gate_w.shape == (bsz, tlen, 8)
    assert gate_l.shape == gate_w.shape
    assert group_w.shape == (bsz, tlen, 4)
    assert group_l.shape == group_w.shape
    assert torch.allclose(group_w.sum(dim=-1), torch.ones(bsz, tlen), atol=1e-6)
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(bsz, tlen), atol=1e-6)
    assert all(expert.use_feature for expert in stage.experts)


def test_hgr_serial_and_parallel_stage_merge_shapes():
    torch.manual_seed(125)
    bsz, tlen, d_model = 2, 4, 16
    n_feat = len(ALL_FEATURE_COLUMNS)
    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, n_feat)
    item_seq_len = torch.tensor([4, 3], dtype=torch.long)

    serial_moe = HierarchicalMoEHGR(
        d_model=d_model,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=2,
        top_k=2,
        group_top_k=2,
        group_router_mode="per_group",
        parallel_stage_gate_top_k=0,
        dropout=0.1,
        stage_merge_mode="serial",
    )
    serial_out = hidden
    for stage_name in ("macro", "mid", "micro"):
        serial_out, gate_w, gate_l, group_w, group_l, _ = serial_moe.forward_stage(
            stage_name,
            serial_out,
            feat,
            item_seq_len=item_seq_len,
        )
        assert gate_w.shape == (bsz, tlen, 8)
        assert group_w.shape == (bsz, tlen, 4)
        assert gate_l.shape == gate_w.shape
        assert group_l.shape == group_w.shape
    assert serial_out.shape == hidden.shape

    parallel_moe = HierarchicalMoEHGR(
        d_model=d_model,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=2,
        top_k=2,
        group_top_k=2,
        group_router_mode="hybrid",
        parallel_stage_gate_top_k=2,
        dropout=0.1,
        stage_merge_mode="parallel",
    )
    stage_deltas = {}
    for stage_name in ("macro", "mid", "micro"):
        _, _, _, _, _, delta = parallel_moe.forward_stage(
            stage_name,
            hidden,
            feat,
            item_seq_len=item_seq_len,
        )
        stage_deltas[stage_name] = delta

    merged_hidden, stage_w, stage_l = parallel_moe.parallel_merge(hidden, feat, stage_deltas)
    assert merged_hidden.shape == hidden.shape
    assert stage_w.shape == (bsz, tlen, 3)
    assert stage_l.shape == stage_w.shape
    assert torch.allclose(stage_w.sum(dim=-1), torch.ones(bsz, tlen), atol=1e-6)
