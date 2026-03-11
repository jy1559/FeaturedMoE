#!/usr/bin/env python3
"""Unit tests for FMoEv2 factorized interaction router."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    STAGE_ALL_FEATURES,
    STAGES,
    build_column_to_index,
)
from models.FeaturedMoE.moe_stages import MoEStage  # noqa: E402
from models.FeaturedMoE_v2.losses import compute_router_distill_aux_loss  # noqa: E402


def _build_inputs(batch_size=3, seq_len=6, d_model=16):
    hidden = torch.randn(batch_size, seq_len, d_model)
    feat = torch.randn(batch_size, seq_len, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([seq_len, max(seq_len - 1, 1), max(seq_len - 2, 1)], dtype=torch.long)
    return hidden, feat, item_seq_len


def _build_stage(stage_name="mid", *, expert_scale=3, router_mode="token", session_pooling="query", distill=False):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)[stage_name]
    return MoEStage(
        stage_name=stage_name,
        expert_feature_lists=list(stage_map.values()),
        stage_all_features=STAGE_ALL_FEATURES[stage_name],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=expert_scale,
        top_k=None,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=True,
        expert_names=list(stage_map.keys()),
        router_impl="learned",
        router_design="group_factorized_interaction",
        group_top_k=0,
        expert_top_k=1,
        router_distill_enable=distill,
        router_mode=router_mode,
        session_pooling=session_pooling,
        router_temperature=1.0,
        router_feature_dropout=0.1,
    )


def test_factorized_router_dense_groups_sparse_clones():
    torch.manual_seed(2026)
    hidden, feat, item_seq_len = _build_inputs()
    stage = _build_stage(stage_name="mid", expert_scale=3)

    next_hidden, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)
    aux = stage.last_router_aux

    assert next_hidden.shape == hidden.shape
    assert gate_weights.shape == (3, 6, 12)
    assert gate_logits.shape == (3, 6, 12)
    assert aux["group_weights"].shape == (3, 6, 4)
    assert aux["group_logits"].shape == (3, 6, 4)
    assert aux["intra_group_weights"].shape == (3, 6, 4, 3)
    assert aux["intra_group_logits"].shape == (3, 6, 4, 3)

    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(3, 6), atol=1e-6)
    assert torch.allclose(aux["group_weights"].sum(dim=-1), torch.ones(3, 6), atol=1e-6)
    assert torch.allclose(aux["intra_group_weights"].sum(dim=-1), torch.ones(3, 6, 4), atol=1e-6)
    assert torch.isfinite(next_hidden).all()
    assert torch.isfinite(gate_weights).all()

    active_clones = (aux["intra_group_weights"] > 0).sum(dim=-1)
    assert torch.equal(active_clones, torch.ones_like(active_clones))
    active_experts = (gate_weights > 0).sum(dim=-1)
    assert int(active_experts.min().item()) == 4
    assert int(active_experts.max().item()) == 4


def test_factorized_router_scale_one_session_broadcast():
    torch.manual_seed(2027)
    hidden, feat, item_seq_len = _build_inputs(batch_size=2, seq_len=5, d_model=16)
    stage = _build_stage(
        stage_name="macro",
        expert_scale=1,
        router_mode="session",
        session_pooling="mean",
    )

    next_hidden, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len[:2])
    aux = stage.last_router_aux

    assert next_hidden.shape == hidden.shape[:2] + (16,)
    assert gate_weights.shape == (2, 5, 4)
    assert gate_logits.shape == (2, 5, 4)
    assert aux["intra_group_weights"].shape == (2, 5, 4, 1)
    assert torch.allclose(aux["intra_group_weights"], torch.ones_like(aux["intra_group_weights"]))
    assert torch.allclose(gate_weights, gate_weights[:, :1, :].expand_as(gate_weights), atol=1e-6)
    assert torch.allclose(aux["group_weights"], aux["group_weights"][:, :1, :].expand_as(aux["group_weights"]), atol=1e-6)


def test_factorized_router_dense_clones_when_expert_top_k_zero():
    torch.manual_seed(2029)
    hidden, feat, item_seq_len = _build_inputs()
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)["mid"]
    stage = MoEStage(
        stage_name="mid",
        expert_feature_lists=list(stage_map.values()),
        stage_all_features=STAGE_ALL_FEATURES["mid"],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=3,
        top_k=None,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=True,
        expert_names=list(stage_map.keys()),
        router_impl="learned",
        router_design="group_factorized_interaction",
        group_top_k=0,
        expert_top_k=0,
        router_distill_enable=False,
        router_mode="token",
        session_pooling="query",
        router_temperature=1.0,
        router_feature_dropout=0.1,
    )

    _, gate_weights, _ = stage(hidden, feat, item_seq_len=item_seq_len)
    aux = stage.last_router_aux

    active_clones = (aux["intra_group_weights"] > 0).sum(dim=-1)
    assert torch.equal(active_clones, torch.full_like(active_clones, 3))
    active_experts = (gate_weights > 0).sum(dim=-1)
    assert int(active_experts.min().item()) == 12
    assert int(active_experts.max().item()) == 12
    assert torch.allclose(aux["intra_group_weights"].sum(dim=-1), torch.ones(3, 6, 4), atol=1e-6)


def test_router_distill_aux_is_finite_for_teacher_student_group_logits():
    torch.manual_seed(2028)
    hidden, feat, item_seq_len = _build_inputs(batch_size=2, seq_len=4, d_model=16)
    stage = _build_stage(stage_name="mid", expert_scale=3, distill=True)

    stage(hidden, feat, item_seq_len=item_seq_len[:2])
    aux = stage.last_router_aux

    loss = compute_router_distill_aux_loss(
        teacher_group_logits={"mid@1": aux["teacher_group_logits"]},
        student_group_logits={"mid@1": aux["group_logits_raw"]},
        item_seq_len=item_seq_len[:2],
        aux_lambda=5e-3,
        distill_temperature=1.5,
        enabled=True,
        progress=0.1,
        until=0.2,
        device=feat.device,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
