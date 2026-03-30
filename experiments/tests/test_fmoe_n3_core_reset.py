#!/usr/bin/env python3
"""Core reset tests for FeaturedMoE_N3."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")
pytest.importorskip("recbole")

from models.FeaturedMoE_N3.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    build_column_to_index,
    build_stage_feature_spec,
)
from models.FeaturedMoE_N3.featured_moe_n3 import FeaturedMoE_N3  # noqa: E402
from models.FeaturedMoE_N3.stage_executor import StageExecutorN3  # noqa: E402
from models.FeaturedMoE_N3.stage_modules import (  # noqa: E402
    SASRecStyleAttentionBlock,
    SASRecStyleFFNBlock,
    SASRecStyleLayerBlock,
)


def _build_executor(
    *,
    layer_layout,
    stage_compute_mode=None,
    stage_router_mode=None,
    stage_router_source=None,
    stage_feature_injection=None,
    stage_router_granularity=None,
    stage_feature_encoder_mode=None,
    stage_spec=None,
):
    spec = stage_spec if isinstance(stage_spec, dict) else build_stage_feature_spec(
        macro_history_window=5,
        stage_feature_family_mask={},
    )
    return StageExecutorN3(
        layer_layout=layer_layout,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
        attn_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        d_feat_emb=8,
        d_expert_hidden=24,
        d_router_hidden=12,
        expert_depth_by_stage={"macro": 1, "mid": 1, "micro": 1},
        expert_hidden_by_stage={"macro": 24, "mid": 24, "micro": 24},
        expert_scale=1,
        stage_top_k=0,
        macro_session_pooling="mean",
        stage_router_granularity=stage_router_granularity or {"macro": "session", "mid": "session", "micro": "token"},
        stage_all_features=spec["stage_all_features"],
        stage_family_features=spec["stage_family_features"],
        stage_feature_encoder_mode=stage_feature_encoder_mode or {"macro": "linear", "mid": "linear", "micro": "linear"},
        stage_compute_mode=stage_compute_mode or {"macro": "moe", "mid": "moe", "micro": "moe"},
        stage_router_mode=stage_router_mode or {"macro": "learned", "mid": "learned", "micro": "learned"},
        stage_router_source=stage_router_source or {"macro": "both", "mid": "both", "micro": "both"},
        stage_feature_injection=stage_feature_injection or {"macro": "none", "mid": "none", "micro": "none"},
        rule_router_cfg={"variant": "ratio_bins", "n_bins": 5, "feature_per_expert": 4},
        rule_bias_scale=0.0,
        feature_group_bias_lambda=0.0,
        feature_group_prior_temperature=1.0,
        stage_router_wrapper={"macro": "w1_flat", "mid": "w1_flat", "micro": "w1_flat"},
        stage_router_primitives={
            "macro": {
                "a_joint": {"source": "both"},
                "b_group": {"source": "both"},
                "c_shared": {"source": "both"},
                "d_cond": {"source": "feature"},
                "e_scalar": {"source": "feature"},
            },
            "mid": {
                "a_joint": {"source": "both"},
                "b_group": {"source": "both"},
                "c_shared": {"source": "both"},
                "d_cond": {"source": "feature"},
                "e_scalar": {"source": "feature"},
            },
            "micro": {
                "a_joint": {"source": "both"},
                "b_group": {"source": "both"},
                "c_shared": {"source": "both"},
                "d_cond": {"source": "feature"},
                "e_scalar": {"source": "feature"},
            },
        },
        mid_router_temperature=1.2,
        micro_router_temperature=1.2,
        dense_hidden_scale=1.0,
        stage_residual_mode={"macro": "base", "mid": "base", "micro": "base"},
        residual_alpha_fixed={"macro": 0.5, "mid": 0.5, "micro": 0.5},
        residual_alpha_init={"macro": 0.0, "mid": 0.0, "micro": 0.0},
        residual_shared_ffn_scale=1.0,
        stage_family_dropout_prob={"macro": 0.0, "mid": 0.0, "micro": 0.0},
        stage_feature_dropout_prob={"macro": 0.0, "mid": 0.0, "micro": 0.0},
        stage_feature_dropout_scope={"macro": "token", "mid": "token", "micro": "token"},
        col2idx=build_column_to_index(ALL_FEATURE_COLUMNS),
    )


def _zero_linear_stack(module):
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            torch.nn.init.zeros_(child.weight)
            if child.bias is not None:
                torch.nn.init.zeros_(child.bias)


def test_sasrec_style_attention_zero_weights_matches_post_ln_residual():
    block = SASRecStyleAttentionBlock(
        d_model=8,
        n_heads=2,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    _zero_linear_stack(block)
    hidden = torch.randn(2, 4, 8)
    attn_mask = torch.zeros(2, 1, 4, 4)
    out = block(hidden, attn_mask)
    expected = block.layer_norm(hidden)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sasrec_style_ffn_zero_weights_matches_post_ln_residual():
    block = SASRecStyleFFNBlock(
        d_model=8,
        d_ff=16,
        hidden_dropout_prob=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )
    _zero_linear_stack(block)
    hidden = torch.randn(2, 4, 8)
    out = block(hidden)
    expected = block.layer_norm(hidden)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sasrec_style_layer_is_attn_then_ffn():
    block = SASRecStyleLayerBlock(
        d_model=8,
        n_heads=2,
        d_ff=16,
        hidden_dropout_prob=0.0,
        attn_dropout_prob=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    )
    _zero_linear_stack(block)
    hidden = torch.randn(2, 4, 8)
    attn_mask = torch.zeros(2, 1, 4, 4)
    out = block(hidden, attn_mask)
    expected = block.ffn.layer_norm(block.attn.layer_norm(hidden))
    assert torch.allclose(out, expected, atol=1e-6)


def test_layer_layout_reuses_same_stage_module_for_macro_and_macro_ffn():
    executor = _build_executor(layer_layout=["macro", "macro_ffn", "mid"])
    stage_ops = [op for op in executor.compiled_ops if op["kind"] == "stage"]
    macro_ops = [op for op in stage_ops if op["stage"] == "macro"]
    assert len(macro_ops) == 2
    assert "macro" in executor.stage_blocks
    assert len(executor.stage_blocks) == 3


def test_micro_granularity_must_be_token():
    with pytest.raises(ValueError, match="micro routing_granularity must be 'token'"):
        _build_executor(
            layer_layout=["micro"],
            stage_router_granularity={"macro": "session", "mid": "session", "micro": "session"},
        )


def test_dense_only_layout_has_no_diag_and_moe_layout_has_diag():
    dense_exec = _build_executor(
        layer_layout=["macro", "mid"],
        stage_compute_mode={"macro": "dense_plain", "mid": "dense_plain", "micro": "none"},
        stage_router_mode={"macro": "none", "mid": "none", "micro": "none"},
    )
    moe_exec = _build_executor(layer_layout=["macro", "mid", "micro"])
    assert dense_exec.supports_diagnostics is False
    assert moe_exec.supports_diagnostics is True


def test_route_consistency_min_sim_filters_low_similarity_pairs():
    model = FeaturedMoE_N3.__new__(FeaturedMoE_N3)
    model.route_consistency_pairs = 1
    model.route_consistency_min_sim = 0.995
    model.item_embedding = torch.nn.Embedding(8, 4)

    gate = torch.tensor(
        [
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.1, 0.9], [0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    item_seq_len = torch.tensor([2, 2], dtype=torch.long)

    feat_high_sim = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.001], [1.0, 0.001]],
        ],
        dtype=torch.float32,
    )
    feat_low_sim = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    high_loss = model._compute_route_consistency_loss({"mid": gate}, item_seq_len, feat_high_sim)
    low_loss = model._compute_route_consistency_loss({"mid": gate}, item_seq_len, feat_low_sim)

    assert float(high_loss.item()) > 0.0
    assert float(low_loss.item()) == pytest.approx(0.0, abs=1e-8)


def test_executor_handles_empty_stage_features_after_structural_drop():
    drop_all_keywords = [
        "mac",
        "mid",
        "mic",
        "ctx",
        "gap",
        "pace",
        "theme",
        "cat",
        "repeat",
        "adj",
        "item",
        "novel",
        "pop",
        "valid",
        "int",
        "sess",
        "suffix",
        "delta",
    ]
    spec = build_stage_feature_spec(
        macro_history_window=5,
        stage_feature_family_mask={},
        stage_feature_drop_keywords=drop_all_keywords,
    )
    assert all(len(spec["stage_all_features"][stage]) == 0 for stage in ("macro", "mid", "micro"))

    executor = _build_executor(
        layer_layout=["macro", "mid", "micro"],
        stage_spec=spec,
    )
    hidden = torch.randn(2, 5, 16)
    attn_mask = torch.zeros(2, 1, 5, 5)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 4], dtype=torch.long)

    out, _, _, _, _, _, _ = executor(
        hidden=hidden,
        attention_mask=attn_mask,
        feat=feat,
        item_seq_len=item_seq_len,
    )
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()
