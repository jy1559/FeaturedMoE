"""FeaturedMoE_HGRv4: feature-aware outer router + stat-soft inner teacher."""

from __future__ import annotations

import logging

from ..FeaturedMoE.analysis_logger import ExpertAnalysisLogger
from ..FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, build_column_to_index
from ..FeaturedMoE.logging_utils import MoELogger
from .hgr_v3_compat import FeaturedMoE_HGRv3
from .hgr_v4_moe_stages import HierarchicalMoEHGRv4

logger = logging.getLogger(__name__)


class FeaturedMoE_HGRv4(FeaturedMoE_HGRv3):
    """HGRv4 keeps HGRv3 training flow but swaps in richer outer routing."""

    def __init__(self, config, dataset):
        desired_router_design = "legacy_concat"
        try:
            desired_router_design = str(config.get("router_design", "legacy_concat"))
        except Exception:
            desired_router_design = "legacy_concat"
        try:
            config["router_design"] = "group_factorized_interaction"
        except Exception:
            pass
        super().__init__(config, dataset)

        def _cfg(name, default=None):
            try:
                return config.get(name, default)
            except Exception:
                return default

        self.group_router_mode = str(_cfg("group_router_mode", "hybrid")).lower().strip()
        if self.group_router_mode not in {"stage_wide", "per_group", "hybrid"}:
            raise ValueError(
                "group_router_mode must be one of ['stage_wide','per_group','hybrid'], "
                f"got {self.group_router_mode}"
            )
        self.outer_router_use_feature = bool(_cfg("outer_router_use_feature", True))
        self.outer_router_design = str(_cfg("outer_router_design", _cfg("router_design", "legacy_concat"))).lower().strip()
        self.inner_router_design = str(_cfg("inner_router_design", _cfg("router_design", "legacy_concat"))).lower().strip()
        self.inner_rule_teacher_kind = str(_cfg("inner_rule_teacher_kind", "group_stat_soft")).lower().strip()
        if self.inner_rule_teacher_kind != "group_stat_soft":
            raise ValueError(
                f"FeaturedMoE_HGRv4 supports only inner_rule_teacher_kind=group_stat_soft, got {self.inner_rule_teacher_kind}"
            )

        if self.any_moe:
            self.hierarchical_moe = HierarchicalMoEHGRv4(
                d_model=self.d_model,
                d_feat_emb=self.d_feat_emb,
                d_expert_hidden=self.d_expert_hidden,
                d_router_hidden=self.d_router_hidden,
                expert_scale=self.expert_scale,
                top_k=self.expert_top_k,
                group_top_k=self.group_top_k,
                parallel_stage_gate_top_k=self.parallel_stage_gate_top_k,
                parallel_stage_gate_temperature=self.parallel_stage_gate_temperature,
                dropout=self.dropout,
                use_macro=self.stage_has_moe["macro"],
                use_mid=self.stage_has_moe["mid"],
                use_micro=self.stage_has_moe["micro"],
                expert_use_hidden=self.expert_use_hidden,
                expert_use_feature=self.expert_use_feature,
                stage_merge_mode=self.stage_merge_mode,
                macro_routing_scope=self.macro_routing_scope,
                macro_session_pooling=self.macro_session_pooling,
                mid_router_temperature=self.mid_router_temperature,
                micro_router_temperature=self.micro_router_temperature,
                mid_router_feature_dropout=self.mid_router_feature_dropout,
                micro_router_feature_dropout=self.micro_router_feature_dropout,
                use_valid_ratio_gating=self.use_valid_ratio_gating,
                outer_router_use_hidden=self.outer_router_use_hidden,
                outer_router_use_feature=self.outer_router_use_feature,
                inner_router_use_hidden=self.inner_router_use_hidden,
                inner_router_use_feature=self.inner_router_use_feature,
                outer_router_design=self.outer_router_design,
                inner_router_design=self.inner_router_design,
                inner_rule_enable=self.inner_rule_enable,
                inner_rule_mode=self.inner_rule_mode,
                inner_rule_bias_scale=self.inner_rule_bias_scale,
                inner_rule_bin_sharpness=self.inner_rule_bin_sharpness,
                inner_rule_group_feature_pool=self.inner_rule_group_feature_pool,
                inner_rule_apply_stages=self.inner_rule_apply_stages,
                group_router_mode=self.group_router_mode,
                inner_rule_teacher_kind=self.inner_rule_teacher_kind,
            )
            self._stage_n_experts = int(getattr(self.hierarchical_moe, "stage_n_experts", 0))

            active_expert_names = self.hierarchical_moe.expert_names
            active_group_names = self.hierarchical_moe.group_names
            self.moe_logger = MoELogger(active_expert_names)
            self.group_logger = MoELogger(active_group_names)
            self._stage_group_names = active_group_names
            self._reset_router_epoch_stats()

            if self.log_expert_analysis:
                col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
                self.analysis_logger = ExpertAnalysisLogger(
                    expert_names=active_expert_names,
                    col2idx=col2idx,
                    n_bins=self.analysis_n_bins,
                    sample_rate=self.analysis_sample_rate,
                )

            self.set_schedule_epoch(epoch_idx=self._schedule_epoch, max_epochs=self._schedule_total_epochs, log_now=True)

        try:
            config["group_router_mode"] = self.group_router_mode
            config["outer_router_use_feature"] = bool(self.outer_router_use_feature)
            config["router_design"] = str(desired_router_design)
            config["outer_router_design"] = str(self.outer_router_design)
            config["inner_router_design"] = str(self.inner_router_design)
            config["inner_rule_teacher_kind"] = str(self.inner_rule_teacher_kind)
        except Exception:
            pass

        logger.info(
            "FeaturedMoE_HGRv4 override: group_router_mode=%s outer_feature=%s "
            "outer_design=%s inner_design=%s inner_teacher=%s expert_scale=%s layout_id=%s merge_mode=%s",
            self.group_router_mode,
            self.outer_router_use_feature,
            self.outer_router_design,
            self.inner_router_design,
            self.inner_rule_teacher_kind,
            self.expert_scale,
            self.arch_layout_id,
            self.stage_merge_mode,
        )
