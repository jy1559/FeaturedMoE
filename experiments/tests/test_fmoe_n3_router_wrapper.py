#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_N3 primitive/wrapper router architecture."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE_N3.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    build_column_to_index,
    build_stage_feature_spec,
)
from models.FeaturedMoE_N3.router_wrapper import (  # noqa: E402
    GroupConditionalIntraRouter,
    GroupConditionalResidualJointWrapper,
    GroupSharedIntraProductWrapper,
    GroupScalarRouter,
    PrimitiveRoutingSpec,
    ScalarGroupConditionalProductWrapper,
    StageGroupRouter,
    StageJointExpertRouter,
    StageSharedIntraRouter,
    required_primitives_for_wrapper,
)
from models.FeaturedMoE_N3.stage_modules import N3StageBlock, StageRuntimeConfigN3  # noqa: E402


def _default_stage_primitives(d_top_k=0):
    return {
        "a_joint": {"source": "both", "temperature": 1.0, "top_k": 0},
        "b_group": {"source": "both", "temperature": 1.0, "top_k": 0},
        "c_shared": {"source": "both", "temperature": 1.0, "top_k": 0},
        "d_cond": {"source": "feature", "temperature": 1.0, "top_k": d_top_k},
        "e_scalar": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "wrapper": {"alpha_d": 1.0},
    }


def _build_mid_stage_cfg(*, wrapper: str, d_top_k: int = 0, expert_scale: int = 3) -> StageRuntimeConfigN3:
    spec = build_stage_feature_spec(macro_history_window=5, stage_feature_family_mask={})
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    mid_features = [name for name in spec["stage_all_features"]["mid"] if name in col2idx]

    return StageRuntimeConfigN3(
        stage_name="mid",
        d_model=16,
        d_ff=32,
        d_feat_emb=8,
        d_expert_hidden=24,
        d_router_hidden=12,
        expert_depth=1,
        expert_scale=expert_scale,
        top_k=0,
        dropout=0.0,
        attn_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        stage_feature_indices=tuple(int(col2idx[name]) for name in mid_features),
        stage_feature_names=tuple(mid_features),
        stage_family_features=spec["stage_family_features"]["mid"],
        stage_feature_encoder_mode="linear",
        stage_compute_mode="moe",
        stage_router_mode="learned",
        stage_router_source="both",
        stage_feature_injection="none",
        routing_granularity="token",
        session_pooling="mean",
        rule_router_cfg={"variant": "ratio_bins", "n_bins": 5, "feature_per_expert": 4},
        rule_bias_scale=0.0,
        feature_group_bias_lambda=0.0,
        feature_group_prior_temperature=1.0,
        stage_router_wrapper=wrapper,
        stage_router_primitives=_default_stage_primitives(d_top_k=d_top_k),
        router_temperature=1.0,
        dense_hidden_scale=1.0,
        stage_residual_mode="base",
        residual_alpha_fixed=0.5,
        residual_alpha_init=0.0,
        shared_ffn_scale=1.0,
        stage_family_dropout_prob=0.0,
        stage_feature_dropout_prob=0.0,
        stage_feature_dropout_scope="token",
    )


def test_primitives_shape_and_simplex():
    torch.manual_seed(7)
    bsz, seq_len = 2, 5
    n_groups, n_per_group = 4, 3
    n_experts = n_groups * n_per_group

    stage_input = torch.randn(bsz, seq_len, 24)
    group_input = torch.randn(bsz, seq_len, n_groups, 24)
    spec = PrimitiveRoutingSpec(source="both", temperature=1.0, top_k=0)

    a = StageJointExpertRouter(d_in=24, n_experts=n_experts, d_hidden=12, dropout=0.0)
    b = StageGroupRouter(d_in=24, n_groups=n_groups, d_hidden=12, dropout=0.0)
    c = StageSharedIntraRouter(d_in=24, n_experts_per_group=n_per_group, d_hidden=12, dropout=0.0)
    d = GroupConditionalIntraRouter(d_in=24, n_groups=n_groups, n_experts_per_group=n_per_group, d_hidden=12, dropout=0.0)
    e = GroupScalarRouter(d_in=24, n_groups=n_groups)

    out_a = a(stage_input, spec)
    out_b = b(stage_input, spec)
    out_c = c(stage_input, spec)
    out_d = d(group_input, spec)
    out_e = e(group_input, spec)

    assert out_a["logits"].shape == (bsz, seq_len, n_experts)
    assert out_b["logits"].shape == (bsz, seq_len, n_groups)
    assert out_c["logits"].shape == (bsz, seq_len, n_per_group)
    assert out_d["logits"].shape == (bsz, seq_len, n_groups, n_per_group)
    assert out_e["logits"].shape == (bsz, seq_len, n_groups)

    assert torch.allclose(out_a["probs"].sum(dim=-1), torch.ones(bsz, seq_len), atol=1e-6)
    assert torch.allclose(out_b["probs"].sum(dim=-1), torch.ones(bsz, seq_len), atol=1e-6)
    assert torch.allclose(out_c["probs"].sum(dim=-1), torch.ones(bsz, seq_len), atol=1e-6)
    assert torch.allclose(out_d["probs"].sum(dim=-1), torch.ones(bsz, seq_len, n_groups), atol=1e-6)
    assert torch.allclose(out_e["probs"].sum(dim=-1), torch.ones(bsz, seq_len), atol=1e-6)


def test_primitive_topk_sparsity_is_applied():
    torch.manual_seed(11)
    bsz, seq_len = 2, 4
    n_groups, n_per_group = 4, 3

    stage_input = torch.randn(bsz, seq_len, 24)
    group_input = torch.randn(bsz, seq_len, n_groups, 24)

    b_router = StageGroupRouter(d_in=24, n_groups=n_groups, d_hidden=12, dropout=0.0)
    d_router = GroupConditionalIntraRouter(
        d_in=24,
        n_groups=n_groups,
        n_experts_per_group=n_per_group,
        d_hidden=12,
        dropout=0.0,
    )

    out_b = b_router(stage_input, PrimitiveRoutingSpec(source="both", temperature=1.0, top_k=2))
    out_d = d_router(group_input, PrimitiveRoutingSpec(source="both", temperature=1.0, top_k=1))

    b_active = (out_b["probs"] > 0).sum(dim=-1)
    d_active = (out_d["probs"] > 0).sum(dim=-1)

    assert int(b_active.min().item()) == 2
    assert int(b_active.max().item()) == 2
    assert int(d_active.min().item()) == 1
    assert int(d_active.max().item()) == 1


def test_wrapper_probability_products_match_expected_values():
    bsz, seq_len = 1, 1
    n_groups, n_per_group = 4, 3

    b_probs = torch.tensor([[[0.7, 0.2, 0.1, 0.0]]], dtype=torch.float)
    c_probs = torch.tensor([[[0.6, 0.3, 0.1]]], dtype=torch.float)
    d_probs = torch.tensor([[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]]], dtype=torch.float)
    e_probs = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float)

    primitives = {
        "a_joint": {"scaled_logits": torch.zeros(bsz, seq_len, n_groups * n_per_group), "probs": torch.full((bsz, seq_len, n_groups * n_per_group), 1.0 / (n_groups * n_per_group))},
        "b_group": {"scaled_logits": torch.log(b_probs.clamp(min=1e-8)), "probs": b_probs},
        "c_shared": {"scaled_logits": torch.log(c_probs.clamp(min=1e-8)), "probs": c_probs},
        "d_cond": {"scaled_logits": torch.log(d_probs.clamp(min=1e-8)), "probs": d_probs},
        "e_scalar": {"scaled_logits": torch.log(e_probs.clamp(min=1e-8)), "probs": e_probs},
    }

    w3 = GroupSharedIntraProductWrapper()
    w5 = ScalarGroupConditionalProductWrapper()
    w6 = GroupConditionalResidualJointWrapper()

    out_w3 = w3(primitives=primitives, stage_temperature=1.0, n_groups=n_groups, n_experts_per_group=n_per_group, params={})
    out_w5 = w5(primitives=primitives, stage_temperature=1.0, n_groups=n_groups, n_experts_per_group=n_per_group, params={})
    out_w6 = w6(
        primitives=primitives,
        stage_temperature=1.0,
        n_groups=n_groups,
        n_experts_per_group=n_per_group,
        params={"alpha_struct": 1.0, "alpha_a": 1.0},
    )

    expected_w3 = (b_probs.unsqueeze(-1) * c_probs.unsqueeze(-2)).reshape(1, 1, -1)
    expected_w5 = (e_probs.unsqueeze(-1) * d_probs).reshape(1, 1, -1)
    expected_w6 = torch.log((b_probs.unsqueeze(-1) * d_probs).reshape(1, 1, -1).clamp(min=1e-8)) + primitives["a_joint"]["scaled_logits"]

    got_w3 = torch.exp(out_w3["scaled_logits"])
    got_w5 = torch.exp(out_w5["scaled_logits"])

    assert torch.allclose(got_w3, expected_w3, atol=1e-6)
    assert torch.allclose(got_w5, expected_w5, atol=1e-6)
    assert torch.allclose(out_w6["scaled_logits"], expected_w6, atol=1e-6)


@pytest.mark.parametrize("wrapper", ["w1_flat", "w2_a_plus_d", "w3_bxc", "w4_bxd", "w5_exd", "w6_bxd_plus_a"])
def test_stage_block_wrapper_forward_is_finite(wrapper):
    torch.manual_seed(2026)
    cfg = _build_mid_stage_cfg(wrapper=wrapper, d_top_k=0, expert_scale=3)
    stage = N3StageBlock(cfg)

    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 4], dtype=torch.long)

    next_hidden, gate_w, gate_l, router_aux, _ = stage(hidden, feat, item_seq_len=item_seq_len)

    assert next_hidden.shape == hidden.shape
    assert gate_w.shape == (2, 5, 12)
    assert gate_l.shape == (2, 5, 12)
    assert torch.isfinite(next_hidden).all()
    assert torch.isfinite(gate_w).all()
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert "final_expert_logits" in router_aux
    assert "final_expert_probs" in router_aux
    assert "wrapper_alias" in router_aux
    assert "primitive_outputs" in router_aux
    assert set(router_aux["primitive_outputs"].keys()) == set(required_primitives_for_wrapper(wrapper))


def test_stage_block_supports_mixed_primitive_sources():
    torch.manual_seed(2028)
    cfg = _build_mid_stage_cfg(wrapper="w4_bxd", d_top_k=0, expert_scale=3)
    cfg.stage_router_primitives = {
        "a_joint": {"source": "hidden", "temperature": 1.0, "top_k": 0},
        "b_group": {"source": "feature", "temperature": 1.0, "top_k": 0},
        "c_shared": {"source": "both", "temperature": 1.0, "top_k": 0},
        "d_cond": {"source": "hidden", "temperature": 1.0, "top_k": 0},
        "e_scalar": {"source": "both", "temperature": 1.0, "top_k": 0},
        "wrapper": {"alpha_d": 0.7},
    }
    stage = N3StageBlock(cfg)

    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 4], dtype=torch.long)

    next_hidden, gate_w, gate_l, router_aux, _ = stage(hidden, feat, item_seq_len=item_seq_len)
    assert next_hidden.shape == hidden.shape
    assert gate_w.shape == (2, 5, 12)
    assert gate_l.shape == (2, 5, 12)
    assert torch.isfinite(next_hidden).all()
    assert torch.isfinite(gate_w).all()
    assert torch.isfinite(gate_l).all()
    assert "primitive_outputs" in router_aux


def test_w5_with_d_top1_enforces_one_expert_per_group():
    torch.manual_seed(2027)
    cfg = _build_mid_stage_cfg(wrapper="w5_exd", d_top_k=1, expert_scale=3)
    stage = N3StageBlock(cfg)

    hidden = torch.randn(2, 6, 16)
    feat = torch.randn(2, 6, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([6, 5], dtype=torch.long)

    _, gate_w, _, _, _ = stage(hidden, feat, item_seq_len=item_seq_len)

    active = (gate_w > 0).sum(dim=-1)
    assert int(active.min().item()) == 4
    assert int(active.max().item()) == 4


def test_stage_block_intra_group_bias_gls_adds_bias_logits():
    torch.manual_seed(2031)
    cfg = _build_mid_stage_cfg(wrapper="w4_bxd", d_top_k=0, expert_scale=3)
    cfg.rule_bias_scale = 0.0
    cfg.feature_group_bias_lambda = 0.0
    cfg.intra_group_bias_mode = "gls_stats12"
    cfg.intra_group_bias_scale = 0.12
    stage = N3StageBlock(cfg)

    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 4], dtype=torch.long)

    _, gate_w, gate_l, router_aux, _ = stage(hidden, feat, item_seq_len=item_seq_len)
    bias_logits = router_aux.get("intra_group_bias_logits")

    assert torch.is_tensor(bias_logits)
    assert bias_logits.shape == gate_l.shape
    assert torch.isfinite(bias_logits).all()
    assert torch.isfinite(gate_w).all()
