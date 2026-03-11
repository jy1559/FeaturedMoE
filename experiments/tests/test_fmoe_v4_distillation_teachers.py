#!/usr/bin/env python3
"""Teacher/fusion unit tests for FeaturedMoE_v4_Distillation."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE_v4_Distillation.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    build_column_to_index,
)
from models.FeaturedMoE_v4_Distillation.layout_schema import (  # noqa: E402
    LayoutSpec,
    StageLayoutSpec,
)
from models.FeaturedMoE_v4_Distillation.legacy_moe_stages import MoEStage  # noqa: E402
from models.FeaturedMoE_v4_Distillation.stage_executor import StageExecutorV2  # noqa: E402
from models.FeaturedMoE_v4_Distillation.losses import (  # noqa: E402
    build_teacher_logits_12way,
    compute_teacher_distill_aux_loss,
)
from models.FeaturedMoE.routers import build_group_local_stat_logits_12way  # noqa: E402


def _build_inputs(batch_size=2, seq_len=5, d_model=16):
    hidden = torch.randn(batch_size, seq_len, d_model)
    feat = torch.randn(batch_size, seq_len, len(ALL_FEATURE_COLUMNS))
    item_seq_len = torch.tensor([seq_len, max(seq_len - 1, 1)], dtype=torch.long)
    valid_mask = torch.arange(seq_len).unsqueeze(0) < item_seq_len.unsqueeze(1)
    return hidden, feat, item_seq_len, valid_mask


def _build_stage(
    stage_name: str,
    *,
    router_impl: str = "learned",
    rule_router_cfg=None,
    router_design: str = "flat_legacy",
    teacher_design: str = "none",
    teacher_delivery: str = "none",
    teacher_stage_mask: str = "all",
    router_clone_residual_scale: float = 0.20,
):
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
        expert_scale=3,
        top_k=0,
        dropout=0.1,
        router_use_hidden=True,
        router_use_feature=True,
        expert_use_hidden=True,
        expert_use_feature=True,
        expert_names=list(stage_map.keys()),
        router_impl=router_impl,
        rule_router_cfg=rule_router_cfg,
        router_design=router_design,
        router_group_bias_scale=0.5,
        router_clone_residual_scale=router_clone_residual_scale,
        router_mode="token",
        session_pooling="query",
        router_temperature=1.0,
        router_feature_dropout=0.1,
        teacher_design=teacher_design,
        teacher_delivery=teacher_delivery,
        teacher_stage_mask=teacher_stage_mask,
        teacher_bias_scale=0.20,
        teacher_stat_sharpness=16.0,
    )


@pytest.mark.parametrize(
    "teacher_design",
    [
        "group_local_stat12",
        "group_comp_stat12",
        "group_comp_shape12",
    ],
)
def test_v4_teacher_designs_produce_direct_12way_logits(teacher_design):
    torch.manual_seed(2026)
    _, feat, item_seq_len, valid_mask = _build_inputs()
    stage = _build_stage("mid")
    stage_feat = feat.index_select(-1, stage.stage_feat_idx)

    logits = build_teacher_logits_12way(
        teacher_design=teacher_design,
        stage_feat=stage_feat,
        feature_groups=stage.base_group_feat_idx,
        expert_scale=3,
        valid_mask=valid_mask,
        item_seq_len=item_seq_len,
        router_mode="token",
        session_pooling="query",
        stat_sharpness=16.0,
    )

    assert logits is not None
    assert logits.shape == (2, 5, 12)
    assert torch.isfinite(logits).all()


def test_v4_teacher_stage_mask_skips_macro_for_mid_micro_only():
    torch.manual_seed(2027)
    hidden, feat, item_seq_len, _ = _build_inputs()
    stage = _build_stage(
        "macro",
        teacher_design="group_comp_stat12",
        teacher_delivery="distill_kl",
        teacher_stage_mask="mid_micro_only",
    )

    stage(hidden, feat, item_seq_len=item_seq_len)
    aux = stage.last_router_aux

    assert aux["teacher_logits"] is None
    assert float(aux["teacher_applied"].item()) == 0.0


def test_v4_fused_bias_changes_final_logits_but_not_raw_logits():
    hidden, feat, item_seq_len, _ = _build_inputs()
    torch.manual_seed(2028)
    plain_stage = _build_stage(
        "mid",
        teacher_design="group_comp_stat12",
        teacher_delivery="none",
    )
    torch.manual_seed(2028)
    fused_stage = _build_stage(
        "mid",
        teacher_design="group_comp_stat12",
        teacher_delivery="fused_bias",
    )
    plain_stage.eval()
    fused_stage.eval()

    plain_stage(hidden, feat, item_seq_len=item_seq_len)
    fused_stage(hidden, feat, item_seq_len=item_seq_len)

    plain_aux = plain_stage.last_router_aux
    fused_aux = fused_stage.last_router_aux

    assert fused_aux["teacher_logits"] is not None
    assert torch.allclose(
        fused_aux["student_logits_raw"],
        plain_aux["student_logits_raw"],
        atol=1e-6,
    )
    assert torch.allclose(
        fused_aux["student_logits_raw"],
        plain_aux["student_logits_final"],
        atol=1e-6,
    )
    assert not torch.allclose(
        fused_aux["student_logits_final"],
        fused_aux["student_logits_raw"],
    )
    assert torch.allclose(
        plain_aux["student_logits_final"],
        plain_aux["student_logits_raw"],
        atol=1e-6,
    )


def test_v4_teacher_distill_aux_loss_is_finite_and_turns_off_after_until():
    torch.manual_seed(2029)
    hidden, feat, item_seq_len, _ = _build_inputs()
    stage = _build_stage(
        "mid",
        teacher_design="group_comp_shape12",
        teacher_delivery="distill_and_fused_bias",
    )

    stage(hidden, feat, item_seq_len=item_seq_len)
    aux = {"mid@1": stage.last_router_aux}

    live = compute_teacher_distill_aux_loss(
        router_stage_aux=aux,
        item_seq_len=item_seq_len,
        teacher_delivery="distill_and_fused_bias",
        teacher_kl_lambda=0.0015,
        teacher_temperature=1.5,
        progress=0.10,
        until=0.25,
        device=hidden.device,
    )
    dead = compute_teacher_distill_aux_loss(
        router_stage_aux=aux,
        item_seq_len=item_seq_len,
        teacher_delivery="distill_and_fused_bias",
        teacher_kl_lambda=0.0015,
        teacher_temperature=1.5,
        progress=0.30,
        until=0.25,
        device=hidden.device,
    )

    assert live.ndim == 0
    assert torch.isfinite(live)
    assert float(live.item()) >= 0.0
    assert torch.isfinite(dead)
    assert float(dead.item()) == 0.0


def test_v4_rule_soft_teacher_produces_finite_teacher_logits():
    torch.manual_seed(2030)
    hidden, feat, item_seq_len, _ = _build_inputs()
    stage = _build_stage(
        "mid",
        teacher_design="rule_soft12",
        teacher_delivery="distill_and_fused_bias",
        rule_router_cfg={"variant": "ratio_bins", "n_bins": 5, "feature_per_expert": 4},
    )

    stage(hidden, feat, item_seq_len=item_seq_len)
    aux = stage.last_router_aux

    assert aux["teacher_logits"] is not None
    assert aux["teacher_logits"].shape == (2, 5, 12)
    assert torch.isfinite(aux["teacher_logits"]).all()


def test_v4_teacher_gls_rule_router_matches_group_local_stat_helper():
    torch.manual_seed(2031)
    _, feat, item_seq_len, valid_mask = _build_inputs()
    stage = _build_stage(
        "mid",
        router_impl="rule_soft",
        rule_router_cfg={"variant": "teacher_gls", "n_bins": 5, "feature_per_expert": 4},
    )
    stage_feat = feat.index_select(-1, stage.stage_feat_idx)

    router_logits = stage.router.backend.compute_logits(stage_feat)
    helper_logits = build_group_local_stat_logits_12way(
        stage_feat,
        feature_groups=stage.base_group_feat_idx,
        expert_scale=3,
        stat_sharpness=16.0,
    )
    teacher_logits = build_teacher_logits_12way(
        teacher_design="group_local_stat12",
        stage_feat=stage_feat,
        feature_groups=stage.base_group_feat_idx,
        expert_scale=3,
        valid_mask=valid_mask,
        item_seq_len=item_seq_len,
        router_mode="token",
        session_pooling="query",
        stat_sharpness=16.0,
    )

    assert torch.allclose(router_logits, helper_logits, atol=1e-6)
    assert torch.allclose(router_logits, teacher_logits, atol=1e-6)


def test_v4_stage_executor_teacher_gls_hybrid_routes_mid_and_micro_with_rule_soft():
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
        expert_scale=3,
        stage_top_k=0,
        dropout=0.1,
        n_heads=2,
        d_ff=32,
        col2idx=col2idx,
        stage_expert_lists=stage_expert_lists,
        stage_expert_names=stage_expert_names,
        router_impl="learned",
        router_impl_by_stage={"mid": "rule_soft", "micro": "rule_soft"},
        rule_router_cfg={"variant": "teacher_gls", "n_bins": 5, "feature_per_expert": 4},
        router_design="flat_legacy",
        router_group_bias_scale=0.5,
        router_clone_residual_scale=0.20,
        teacher_design="none",
        teacher_delivery="none",
        teacher_stage_mask="all",
        teacher_bias_scale=0.0,
        teacher_stat_sharpness=16.0,
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

    macro_stage = executor.branches["macro"].stage_module.stage
    mid_stage = executor.branches["mid"].stage_module.stage
    micro_stage = executor.branches["micro"].stage_module.stage

    assert macro_stage.router.impl == "learned"
    assert mid_stage.router.impl == "rule_soft"
    assert micro_stage.router.impl == "rule_soft"
    assert mid_stage.router.backend.variant == "teacher_gls"
    assert micro_stage.router.backend.variant == "teacher_gls"
