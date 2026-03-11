#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_v3 flat-router variants."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE_v3.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    STAGE_ALL_FEATURES,
    STAGES,
    build_column_to_index,
)
from models.FeaturedMoE_v3.legacy_moe_stages import MoEStage  # noqa: E402
from models.FeaturedMoE_v3.losses import compute_router_distill_aux_loss  # noqa: E402


def _build_inputs(batch_size=2, seq_len=5, d_model=16):
    hidden = torch.randn(batch_size, seq_len, d_model)
    feat = torch.randn(batch_size, seq_len, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([seq_len, max(seq_len - 1, 1)], dtype=torch.long)
    return hidden, feat, item_seq_len


def _build_stage(router_design: str, *, top_k=0, expert_scale=3):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)["mid"]
    use_feature = router_design != "flat_hidden_only"
    return MoEStage(
        stage_name="mid",
        expert_feature_lists=list(stage_map.values()),
        stage_all_features=STAGE_ALL_FEATURES["mid"],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=expert_scale,
        top_k=top_k,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=use_feature,
        expert_use_hidden=True,
        expert_use_feature=True,
        expert_names=list(stage_map.keys()),
        router_impl="learned",
        router_design=router_design,
        router_group_bias_scale=0.5,
        router_clone_residual_scale=0.5,
        router_mode="token",
        session_pooling="query",
        router_temperature=1.0,
        router_feature_dropout=0.1,
    )


@pytest.mark.parametrize(
    "router_design",
    [
        "flat_hidden_only",
        "flat_global_interaction",
        "flat_hidden_group_clone12",
        "flat_group_bias12",
        "flat_clone_residual12",
        "flat_group_clone_combo",
    ],
)
def test_v3_flat_router_designs_produce_dense_12way_logits(router_design):
    torch.manual_seed(2026)
    hidden, feat, item_seq_len = _build_inputs()
    stage = _build_stage(router_design, top_k=0)

    next_hidden, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)

    assert next_hidden.shape == hidden.shape
    assert gate_weights.shape == (2, 5, 12)
    assert gate_logits.shape == (2, 5, 12)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    active = (gate_weights > 0).sum(dim=-1)
    assert int(active.min().item()) == 12
    assert int(active.max().item()) == 12
    assert torch.isfinite(next_hidden).all()
    assert torch.isfinite(gate_weights).all()
    assert torch.isfinite(gate_logits).all()


def test_v3_flat_router_top2_is_sparse():
    torch.manual_seed(2027)
    hidden, feat, item_seq_len = _build_inputs()
    stage = _build_stage("flat_clone_residual12", top_k=2)

    _, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)

    assert gate_weights.shape == gate_logits.shape == (2, 5, 12)
    active = (gate_weights > 0).sum(dim=-1)
    assert int(active.min().item()) == 2
    assert int(active.max().item()) == 2
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_v3_flat_router_accepts_expert_scale_eight():
    torch.manual_seed(2029)
    hidden, feat, item_seq_len = _build_inputs()
    stage = _build_stage("flat_clone_residual12", top_k=0, expert_scale=8)

    next_hidden, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)

    assert next_hidden.shape == hidden.shape
    assert gate_weights.shape == (2, 5, 32)
    assert gate_logits.shape == (2, 5, 32)
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    active = (gate_weights > 0).sum(dim=-1)
    assert int(active.min().item()) == 32
    assert int(active.max().item()) == 32


@pytest.mark.parametrize(
    "mode,lambda_group,lambda_clone",
    [
        ("group_only", 0.002, 0.0),
        ("clone_only", 0.0, 0.003),
        ("group_plus_clone", 0.0015, 0.003),
    ],
)
def test_v3_router_distill_loss_is_finite(mode, lambda_group, lambda_clone):
    torch.manual_seed(2028)
    hidden, feat, item_seq_len = _build_inputs()
    stage = _build_stage("flat_group_clone_combo", top_k=0)

    _, _, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)
    stage_groups = {stage.stage_name: [list(x) for x in stage.base_group_feat_idx]}

    loss = compute_router_distill_aux_loss(
        gate_logits={"mid@1": gate_logits},
        feat=feat,
        stage_group_feature_indices=stage_groups,
        item_seq_len=item_seq_len,
        expert_scale=3,
        mode=mode,
        enabled=True,
        lambda_group=lambda_group,
        lambda_clone=lambda_clone,
        distill_temperature=1.5,
        progress=0.1,
        until=0.2,
        device=feat.device,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
