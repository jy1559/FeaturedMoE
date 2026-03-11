#!/usr/bin/env python3
"""Unit tests for FeaturedMoE_Individual modules."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGE_ALL_FEATURES, build_column_to_index  # noqa: E402
from models.FeaturedMoE_Individual.individual_moe_stages import FeatureIndividualStageMoE  # noqa: E402


def _build_stage(stage_name: str = "macro", expert_scale: int = 4) -> FeatureIndividualStageMoE:
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    return FeatureIndividualStageMoE(
        stage_name=stage_name,
        feature_names=STAGE_ALL_FEATURES[stage_name],
        stage_all_features=STAGE_ALL_FEATURES[stage_name],
        col2idx=col2idx,
        d_model=16,
        d_feat_emb=8,
        d_expert_hidden=24,
        d_router_hidden=12,
        expert_scale=expert_scale,
        feature_top_k=4,
        inner_expert_top_k=0,
        dropout=0.1,
        expert_use_hidden=True,
        expert_use_feature=False,
        outer_router_use_hidden=True,
        outer_router_use_feature=True,
        inner_router_use_hidden=True,
        inner_router_use_feature=True,
        router_temperature=1.0,
    )


def test_individual_macro_stage_shapes_and_topk():
    torch.manual_seed(21)
    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    stage = _build_stage("macro")

    next_hidden, gate_w, gate_l, feature_w, feature_l, stage_delta = stage(hidden, feat)
    aux = stage.last_router_aux

    assert next_hidden.shape == hidden.shape
    assert stage_delta.shape == hidden.shape
    assert feature_w.shape == (2, 5, 14)
    assert feature_l.shape == (2, 5, 14)
    assert gate_w.shape == (2, 5, 56)
    assert gate_l.shape == (2, 5, 56)
    assert aux["intra_group_weights"].shape == (2, 5, 14, 4)
    assert torch.allclose(feature_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.all((feature_w > 0).sum(dim=-1) == 4)
    assert torch.all(aux["intra_group_weights"] > 0)
    assert torch.allclose(aux["intra_group_weights"].sum(dim=-1), torch.ones(2, 5, 14), atol=1e-6)
    assert torch.allclose(gate_w.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_individual_mid_and_micro_stage_feature_counts():
    torch.manual_seed(22)
    hidden = torch.randn(2, 4, 16)
    feat = torch.randn(2, 4, len(ALL_FEATURE_COLUMNS))

    mid_stage = _build_stage("mid")
    _, _, _, mid_w, _, _ = mid_stage(hidden, feat)
    micro_stage = _build_stage("micro")
    _, _, _, micro_w, _, _ = micro_stage(hidden, feat)

    assert mid_w.shape == (2, 4, 16)
    assert micro_w.shape == (2, 4, 16)
    assert torch.all((mid_w > 0).sum(dim=-1) == 4)
    assert torch.all((micro_w > 0).sum(dim=-1) == 4)


class _DummyDataset:
    def __init__(self, item_num: int = 64):
        self.item_num = int(item_num)

    def num(self, field: str) -> int:
        if field == "item_id":
            return self.item_num
        return 8


def test_individual_model_calculate_loss_is_finite():
    pytest.importorskip("recbole")
    from models.FeaturedMoE_Individual.featured_moe_individual import FeaturedMoE_Individual  # noqa: E402

    config = {
        "USER_ID_FIELD": "session_id",
        "ITEM_ID_FIELD": "item_id",
        "LIST_SUFFIX": "_list",
        "ITEM_LIST_LENGTH_FIELD": "item_length",
        "NEG_PREFIX": "neg_",
        "MAX_ITEM_LIST_LENGTH": 5,
        "embedding_size": 16,
        "hidden_size": 16,
        "d_feat_emb": 8,
        "d_expert_hidden": 24,
        "d_router_hidden": 12,
        "expert_scale": 4,
        "num_heads": 4,
        "hidden_dropout_prob": 0.1,
        "num_layers": -1,
        "arch_layout_catalog": [[0, 0, 1, 0, 1, 0, 0, 0]],
        "arch_layout_id": 0,
        "feature_top_k": 4,
        "inner_expert_top_k": 0,
        "stage_merge_mode": "serial",
        "outer_router_use_hidden": True,
        "outer_router_use_feature": True,
        "inner_router_use_hidden": True,
        "inner_router_use_feature": True,
        "expert_use_hidden": True,
        "expert_use_feature": False,
        "use_aux_loss": False,
        "balance_loss_lambda": 0.0,
        "group_balance_lambda": 0.0,
        "intra_balance_lambda": 0.0,
        "ffn_moe": False,
        "n_ffn_experts": 4,
        "ffn_top_k": 0,
    }
    model = FeaturedMoE_Individual(config, _DummyDataset())
    model.train()

    batch_size = 3
    seq_len = 5
    interaction = {
        model.ITEM_SEQ: torch.randint(1, 20, (batch_size, seq_len)),
        model.ITEM_SEQ_LEN: torch.tensor([5, 4, 3]),
        model.POS_ITEM_ID: torch.randint(1, 20, (batch_size,)),
    }
    for field in model.feature_fields:
        interaction[field] = torch.randn(batch_size, seq_len)

    loss = model.calculate_loss(interaction)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
