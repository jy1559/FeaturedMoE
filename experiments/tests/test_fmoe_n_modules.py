#!/usr/bin/env python3
"""Module/model smoke tests for FeaturedMoE_N."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")
pytest.importorskip("recbole")

from models.FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGES, build_column_to_index  # noqa: E402
from models.FeaturedMoE_N.feature_bank import SharedFeatureBank  # noqa: E402
from models.FeaturedMoE_N.stage_modules import NMoEStage, StageRuntimeConfig  # noqa: E402
from models.FeaturedMoE_N.featured_moe_n import FeaturedMoE_N  # noqa: E402


def _build_stage_cfg(
    stage_name="mid",
    *,
    router_impl="learned",
    rule_bias_scale=0.0,
    feature_bank_dim=1,
):
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    stage_map = dict(STAGES)[stage_name]
    return StageRuntimeConfig(
        stage_name=stage_name,
        pass_layers=0,
        moe_blocks=1,
        d_model=16,
        d_ff=32,
        n_heads=2,
        dropout=0.1,
        d_expert_hidden=24,
        d_router_hidden=12,
        expert_depth=1,
        expert_scale=1,
        feature_bank_dim=feature_bank_dim,
        top_k=1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=True,
        macro_routing_scope="session",
        macro_session_pooling="mean",
        mid_router_temperature=1.2,
        micro_router_temperature=1.2,
        mid_router_feature_dropout=0.0,
        micro_router_feature_dropout=0.0,
        use_valid_ratio_gating=True,
        col2idx=col2idx,
        expert_feature_lists=list(stage_map.values()),
        expert_names=list(stage_map.keys()),
        router_impl=router_impl,
        rule_router_cfg={"variant": "ratio_bins", "n_bins": 5, "feature_per_expert": 4},
        rule_bias_scale=rule_bias_scale,
    )


def test_shared_feature_bank_modes_have_expected_shapes():
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    linear_bank = SharedFeatureBank(feature_names=ALL_FEATURE_COLUMNS, mode="linear")
    sin_bank = SharedFeatureBank(
        feature_names=ALL_FEATURE_COLUMNS,
        mode="sinusoidal_selected",
        sinusoidal_patterns=["*time*", "*gap*"],
        sinusoidal_n_freqs=3,
    )

    linear_out = linear_bank(feat)
    sin_out = sin_bank(feat)

    assert linear_out.shape == (2, 5, len(ALL_FEATURE_COLUMNS), 1)
    assert sin_out.shape == (2, 5, len(ALL_FEATURE_COLUMNS), 7)
    assert torch.isfinite(linear_out).all()
    assert torch.isfinite(sin_out).all()


@pytest.mark.parametrize(
    "router_impl,rule_bias_scale",
    [
        ("learned", 0.0),
        ("learned", 0.2),
        ("rule_soft", 0.0),
    ],
)
def test_common_stage_forward_is_finite(router_impl, rule_bias_scale):
    torch.manual_seed(1234)
    hidden = torch.randn(2, 5, 16)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([5, 4], dtype=torch.long)
    feature_bank = SharedFeatureBank(
        feature_names=ALL_FEATURE_COLUMNS,
        mode="sinusoidal_selected",
        sinusoidal_patterns=["*time*", "*gap*", "*int*", "*pop*", "*valid_r*"],
        sinusoidal_n_freqs=2,
    )
    feat_bank = feature_bank(feat)
    stage = NMoEStage(
        _build_stage_cfg(
            stage_name="mid",
            router_impl=router_impl,
            rule_bias_scale=rule_bias_scale,
            feature_bank_dim=feature_bank.bank_dim,
        )
    )

    next_hidden, stage_out, gate_weights, gate_logits, _ = stage(
        hidden,
        feat,
        feat_bank,
        item_seq_len=item_seq_len,
    )

    assert next_hidden.shape == hidden.shape
    assert stage_out.shape == hidden.shape
    assert gate_weights.shape[:2] == hidden.shape[:2]
    assert gate_logits.shape[:2] == hidden.shape[:2]
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones_like(gate_weights.sum(dim=-1)), atol=1e-6)
    assert torch.isfinite(next_hidden).all()


def test_mid_stage_reliability_gate_broadcasts_over_flat_bank_dim():
    torch.manual_seed(7)
    feat = torch.randn(2, 5, len(ALL_FEATURE_COLUMNS))
    feature_bank = SharedFeatureBank(
        feature_names=ALL_FEATURE_COLUMNS,
        mode="sinusoidal_selected",
        sinusoidal_patterns=["*time*", "*gap*", "*int*", "*pop*", "*valid_r*"],
        sinusoidal_n_freqs=2,
    )
    feat_bank = feature_bank(feat)
    stage = NMoEStage(
        _build_stage_cfg(
            stage_name="mid",
            router_impl="learned",
            rule_bias_scale=0.15,
            feature_bank_dim=feature_bank.bank_dim,
        )
    )

    stage_raw_feat = feat.index_select(-1, stage.stage_feat_idx)
    stage_bank_flat = stage._stage_bank_flat(feat_bank)
    gated_raw, gated_bank = stage._apply_reliability(stage_raw_feat, stage_bank_flat, feat)

    reliability = feat[..., stage.reliability_feat_idx].clamp(0.0, 1.0)
    assert gated_raw.shape == stage_raw_feat.shape
    assert gated_bank.shape == stage_bank_flat.shape
    assert torch.allclose(gated_raw, stage_raw_feat * reliability.unsqueeze(-1), atol=1e-6)
    assert torch.allclose(gated_bank, stage_bank_flat * reliability.unsqueeze(-1), atol=1e-6)


class _DummyDataset:
    def __init__(self, item_num: int = 64):
        self.item_num = int(item_num)

    def num(self, field: str) -> int:
        if field == "item_id":
            return self.item_num
        return 8


def _base_model_config():
    return {
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
        "expert_scale": 1,
        "num_heads": 4,
        "hidden_dropout_prob": 0.1,
        "d_ff": 32,
        "fmoe_v2_layout_catalog": [
            {
                "id": "Ltest",
                "execution": "serial",
                "global_pre_layers": 0,
                "global_post_layers": 0,
                "stages": {
                    "macro": {"pass_layers": 1, "moe_blocks": 0},
                    "mid": {"pass_layers": 0, "moe_blocks": 1},
                    "micro": {"pass_layers": 0, "moe_blocks": 1},
                },
            }
        ],
        "fmoe_v2_layout_id": 0,
        "router_design": "simple_flat",
        "router_impl": "learned",
        "router_impl_by_stage": {},
        "feature_encoder_mode": "linear",
        "feature_encoder_sinusoidal_features": ["*time*", "*gap*", "*int*", "*pop*", "*valid_r*"],
        "rule_bias_scale": 0.15,
        "fmoe_special_logging": True,
        "use_aux_loss": True,
        "balance_loss_lambda": 0.002,
        "fmoe_schedule_enable": True,
        "moe_top_k": 1,
        "moe_top_k_policy": "fixed",
        "macro_session_pooling": "mean",
        "macro_routing_scope": "session",
        "feature_mode": "full_v2",
    }


def test_common_model_allows_inert_teacher_defaults():
    config = _base_model_config()
    config.update(
        {
            "teacher_design": "none",
            "teacher_delivery": "none",
            "teacher_stage_mask": "all",
            "teacher_kl_lambda": 0.0,
            "teacher_bias_scale": 0.0,
            "teacher_until": 0.25,
            "teacher_stat_sharpness": 16.0,
            "teacher_temperature": 1.5,
            "router_distill_enable": False,
            "fmoe_v2_feature_spec_aux_enable": False,
        }
    )

    model = FeaturedMoE_N(config, _DummyDataset())
    assert isinstance(model, FeaturedMoE_N)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("teacher_design", "group_local_stat12", "teacher_design"),
        ("teacher_delivery", "distill_and_fused_bias", "teacher_delivery"),
        ("router_distill_enable", True, "router_distill_enable"),
        ("fmoe_v2_feature_spec_aux_enable", True, "feature specialization auxiliary"),
    ],
)
def test_common_model_rejects_enabled_unsupported_paths(key, value, match):
    config = _base_model_config()
    config[key] = value

    with pytest.raises(ValueError, match=match):
        FeaturedMoE_N(config, _DummyDataset())


def test_common_model_calculate_loss_is_finite():
    config = _base_model_config()
    model = FeaturedMoE_N(config, _DummyDataset())
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
