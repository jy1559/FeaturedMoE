#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_ProtoX core modules."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    STAGES,
    build_column_to_index,
)
from models.FeaturedMoE_ProtoX.protox_modules import (  # noqa: E402
    PrototypeConditionedStageBlock,
    PrototypeConditionedStageGate,
    SessionPrototypeAllocator,
)


def _build_stage_blocks(d_model=16, proto_dim=12):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    n_feat = len(ALL_FEATURE_COLUMNS)
    blocks = {}
    for stage_name, expert_dict in STAGES:
        blocks[stage_name] = PrototypeConditionedStageBlock(
            stage_name=stage_name,
            expert_feature_lists=list(expert_dict.values()),
            expert_names=list(expert_dict.keys()),
            col2idx=col2idx,
            d_model=d_model,
            n_features=n_feat,
            proto_dim=proto_dim,
            d_feat_emb=8,
            d_expert_hidden=32,
            d_router_hidden=16,
            expert_scale=1,
            top_k=2,
            dropout=0.1,
            router_use_hidden=True,
            router_use_feature=True,
            expert_use_hidden=True,
            expert_use_feature=True,
            router_temperature=1.0,
            router_feature_dropout=0.1,
            reliability_feature_name=None,
        )
    return blocks


def test_prototype_allocator_simplex_topk_and_context_shape():
    torch.manual_seed(31)
    bsz, tlen, d_model = 4, 6, 16
    n_feat = len(ALL_FEATURE_COLUMNS)

    allocator = SessionPrototypeAllocator(
        d_model=d_model,
        n_features=n_feat,
        d_feat_emb=8,
        d_hidden=16,
        proto_num=6,
        proto_dim=12,
        dropout=0.1,
        top_k=2,
        temperature=1.0,
        use_hidden=True,
        use_feature=True,
        pooling="query",
    )

    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, n_feat)
    item_seq_len = torch.tensor([6, 5, 4, 3], dtype=torch.long)

    pi, logits, ctx = allocator(hidden=hidden, feat=feat, item_seq_len=item_seq_len)

    assert pi.shape == (bsz, 6)
    assert logits.shape == (bsz, 6)
    assert ctx.shape == (bsz, 12)
    assert torch.allclose(pi.sum(dim=-1), torch.ones(bsz), atol=1e-6)
    assert int((pi > 0).sum(dim=-1).max().item()) <= 2


def test_proto_conditioned_stage_gate_and_block_shapes():
    torch.manual_seed(37)
    bsz, tlen, d_model = 3, 5, 16
    proto_dim = 12
    n_feat = len(ALL_FEATURE_COLUMNS)

    stage_gate = PrototypeConditionedStageGate(
        d_model=d_model,
        n_features=n_feat,
        d_feat_emb=8,
        d_hidden=16,
        proto_dim=proto_dim,
        dropout=0.1,
        use_hidden=True,
        use_feature=True,
        pooling="query",
    )
    blocks = _build_stage_blocks(d_model=d_model, proto_dim=proto_dim)

    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, n_feat)
    proto_ctx = torch.randn(bsz, proto_dim)
    item_seq_len = torch.tensor([5, 4, 3], dtype=torch.long)

    stage_w, stage_l = stage_gate(
        hidden=hidden,
        feat=feat,
        proto_context=proto_ctx,
        item_seq_len=item_seq_len,
    )
    assert stage_w.shape == (bsz, 3)
    assert stage_l.shape == (bsz, 3)
    assert torch.allclose(stage_w.sum(dim=-1), torch.ones(bsz), atol=1e-6)

    for stage_name in ("macro", "mid", "micro"):
        delta, gate_w, gate_l = blocks[stage_name](
            hidden,
            feat,
            proto_ctx,
            item_seq_len=item_seq_len,
        )
        assert delta.shape == hidden.shape
        assert gate_w.shape[:2] == (bsz, tlen)
        assert gate_l.shape == gate_w.shape
        assert torch.isfinite(delta).all()
        assert torch.isfinite(gate_w).all()


def test_stage_floor_and_delta_scale_keep_gradients_finite():
    torch.manual_seed(41)
    bsz, tlen, d_model = 2, 4, 16
    proto_dim = 10
    n_feat = len(ALL_FEATURE_COLUMNS)

    blocks = _build_stage_blocks(d_model=d_model, proto_dim=proto_dim)
    block = blocks["mid"]

    hidden = torch.randn(bsz, tlen, d_model, requires_grad=True)
    feat = torch.randn(bsz, tlen, n_feat, requires_grad=True)
    proto_ctx = torch.randn(bsz, proto_dim, requires_grad=True)
    item_seq_len = torch.tensor([4, 3], dtype=torch.long)

    stage_logits = torch.randn(bsz, 3, requires_grad=True)
    stage_w = torch.softmax(stage_logits, dim=-1)
    stage_floor = 0.1
    stage_w = stage_w * (1.0 - stage_floor) + stage_floor / 3.0

    delta, gate_w, _ = block(hidden, feat, proto_ctx, item_seq_len=item_seq_len)
    out = hidden + 2.0 * stage_w[:, 1].view(-1, 1, 1) * delta
    loss = out.pow(2).mean() + 0.01 * gate_w.pow(2).mean()
    loss.backward()

    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    assert proto_ctx.grad is not None and torch.isfinite(proto_ctx.grad).all()
    assert stage_logits.grad is not None and torch.isfinite(stage_logits.grad).all()

    param_grads = [p.grad for p in block.parameters() if p.grad is not None]
    assert len(param_grads) > 0
    assert all(torch.isfinite(g).all() for g in param_grads)
