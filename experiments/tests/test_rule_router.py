#!/usr/bin/env python3
"""Unit tests for rule-soft router ablation support."""

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
from models.FeaturedMoE_v2.layout_schema import LayoutSpec, StageLayoutSpec  # noqa: E402
from models.FeaturedMoE_v2.stage_executor import StageExecutorV2  # noqa: E402


def _build_dummy_inputs(batch_size=2, seq_len=5, d_model=16):
    hidden = torch.randn(batch_size, seq_len, d_model)
    feat = torch.randn(batch_size, seq_len, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([seq_len, max(seq_len - 1, 1)], dtype=torch.long)
    return hidden, feat, item_seq_len


def test_rule_soft_router_topk_shape_and_simplex():
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    mid_expert_lists = list(dict(STAGES)["mid"].values())

    stage = MoEStage(
        stage_name="mid",
        expert_feature_lists=mid_expert_lists,
        stage_all_features=STAGE_ALL_FEATURES["mid"],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=1,
        top_k=2,
        router_impl="rule_soft",
        rule_router_cfg={"n_bins": 5, "feature_per_expert": 4},
        router_mode="token",
    )

    hidden, feat, item_seq_len = _build_dummy_inputs(batch_size=3, seq_len=6, d_model=16)
    next_hidden, gate_weights, gate_logits = stage(hidden, feat, item_seq_len=item_seq_len)

    assert next_hidden.shape == hidden.shape
    assert gate_weights.shape == (3, 6, 4)
    assert gate_logits.shape == (3, 6, 4)

    sums = gate_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    # top_k=2 -> each token should have at most 2 non-zero experts
    nnz = (gate_weights > 0).sum(dim=-1)
    assert int(nnz.max().item()) <= 2


def test_stage_executor_router_impl_by_stage_override():
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_expert_lists = {stage_name: list(expert_dict.values()) for stage_name, expert_dict in STAGES}
    stage_expert_names = {stage_name: list(expert_dict.keys()) for stage_name, expert_dict in STAGES}

    layout = LayoutSpec(
        layout_id="Ltest",
        execution="serial",
        global_pre_layers=0,
        global_post_layers=0,
        stages={
            "macro": StageLayoutSpec(pass_layers=0, moe_blocks=1),
            "mid": StageLayoutSpec(pass_layers=0, moe_blocks=1),
            "micro": StageLayoutSpec(pass_layers=0, moe_blocks=1),
        },
    )

    executor = StageExecutorV2(
        layout=layout,
        d_model=16,
        n_features=len(ALL_FEATURE_COLUMNS),
        d_feat_emb=8,
        d_expert_hidden=32,
        d_router_hidden=16,
        expert_scale=1,
        stage_top_k=2,
        dropout=0.1,
        n_heads=2,
        d_ff=32,
        col2idx=col2idx,
        stage_expert_lists=stage_expert_lists,
        stage_expert_names=stage_expert_names,
        router_impl="learned",
        router_impl_by_stage={"mid": "rule_soft", "micro": "rule_soft"},
        rule_router_cfg={"n_bins": 5, "feature_per_expert": 4},
        router_design="flat_legacy",
        group_top_k=0,
        expert_top_k=1,
        router_distill_enable=False,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=True,
        macro_routing_scope="session",
        macro_session_pooling="mean",
        mid_router_temperature=1.3,
        micro_router_temperature=1.3,
        mid_router_feature_dropout=0.1,
        micro_router_feature_dropout=0.1,
        use_valid_ratio_gating=True,
        parallel_stage_gate_top_k=None,
        parallel_stage_gate_temperature=1.0,
    )

    macro_impl = executor.branches["macro"].stage_module.stage.router.impl
    mid_impl = executor.branches["mid"].stage_module.stage.router.impl
    micro_impl = executor.branches["micro"].stage_module.stage.router.impl

    assert macro_impl == "learned"
    assert mid_impl == "rule_soft"
    assert micro_impl == "rule_soft"


def test_default_router_impl_is_learned():
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    macro_expert_lists = list(dict(STAGES)["macro"].values())

    stage = MoEStage(
        stage_name="macro",
        expert_feature_lists=macro_expert_lists,
        stage_all_features=STAGE_ALL_FEATURES["macro"],
        col2idx=col2idx,
        d_model=8,
        d_feat_emb=4,
        d_expert_hidden=16,
        d_router_hidden=8,
        expert_scale=1,
        top_k=None,
        router_mode="token",
    )

    assert stage.router.impl == "learned"
