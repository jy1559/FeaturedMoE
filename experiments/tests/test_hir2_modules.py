#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_HiR2 core modules."""

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
from models.FeaturedMoE_HiR2.hir2_modules import StageAllocator, StageExpertBlock  # noqa: E402


def _build_stage_blocks(d_model=16):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    blocks = {}
    for stage_name, expert_dict in STAGES:
        blocks[stage_name] = StageExpertBlock(
            stage_name=stage_name,
            expert_feature_lists=list(expert_dict.values()),
            expert_names=list(expert_dict.keys()),
            col2idx=col2idx,
            d_model=d_model,
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


def test_stage_allocator_simplex_with_topk():
    torch.manual_seed(13)
    bsz, tlen, d_model = 4, 6, 16
    n_feat = len(ALL_FEATURE_COLUMNS)

    allocator = StageAllocator(
        d_model=d_model,
        n_features=n_feat,
        d_feat_emb=8,
        d_hidden=16,
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

    weights, logits = allocator(hidden=hidden, feat=feat, item_seq_len=item_seq_len)

    assert weights.shape == (bsz, 3)
    assert logits.shape == (bsz, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(bsz), atol=1e-6)
    assert int((weights > 0).sum(dim=-1).max().item()) <= 2


def test_hir2_serial_and_parallel_output_shapes():
    torch.manual_seed(21)
    bsz, tlen, d_model = 3, 5, 16
    n_feat = len(ALL_FEATURE_COLUMNS)

    allocator = StageAllocator(
        d_model=d_model,
        n_features=n_feat,
        d_feat_emb=8,
        d_hidden=16,
        dropout=0.1,
        top_k=0,
        temperature=1.0,
        use_hidden=True,
        use_feature=True,
        pooling="query",
    )
    blocks = _build_stage_blocks(d_model=d_model)

    hidden = torch.randn(bsz, tlen, d_model)
    feat = torch.randn(bsz, tlen, n_feat)
    item_seq_len = torch.tensor([5, 4, 5], dtype=torch.long)
    stage_w, _ = allocator(hidden=hidden, feat=feat, item_seq_len=item_seq_len)

    # serial_weighted
    serial_out = hidden
    for sid, stage_name in enumerate(("macro", "mid", "micro")):
        delta, gate_w, gate_l = blocks[stage_name](serial_out, feat, item_seq_len=item_seq_len)
        assert gate_w.shape[:2] == (bsz, tlen)
        assert gate_l.shape == gate_w.shape
        serial_out = serial_out + stage_w[:, sid].view(-1, 1, 1) * delta
    assert serial_out.shape == hidden.shape

    # parallel_weighted
    base = hidden
    total_delta = torch.zeros_like(base)
    for sid, stage_name in enumerate(("macro", "mid", "micro")):
        delta, gate_w, gate_l = blocks[stage_name](base, feat, item_seq_len=item_seq_len)
        assert gate_w.shape[:2] == (bsz, tlen)
        assert gate_l.shape == gate_w.shape
        total_delta = total_delta + stage_w[:, sid].view(-1, 1, 1) * delta
    parallel_out = base + total_delta
    assert parallel_out.shape == hidden.shape
