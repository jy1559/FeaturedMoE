"""FeaturedMoE_N2 model."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict

import torch
import torch.nn.functional as F

from ..FeaturedMoE.logging_utils import MoELogger
from ..FeaturedMoE.routers import load_balance_loss
from ..FeaturedMoE_N.featured_moe_n import FeaturedMoE_N
from ..FeaturedMoE_v2.config_schema import ConfigResolver
from ..FeaturedMoE_v2.feature_config import ALL_FEATURE_COLUMNS, STAGES, build_column_to_index
from ..FeaturedMoE_v2.losses import compute_expert_aux_loss, compute_stage_merge_aux_loss
from .stage_executor import StageExecutorN2

logger = logging.getLogger(__name__)


def _set_config_key(config_obj, key: str, value) -> None:
    try:
        config_obj[key] = value
    except Exception:
        pass
    final_cfg = getattr(config_obj, "final_config_dict", None)
    if isinstance(final_cfg, dict):
        final_cfg[key] = value


class FeaturedMoE_N2(FeaturedMoE_N):
    """Forked N2 track for feature-heavy routing and ARCH3 controls."""

    @staticmethod
    def _ensure_layout_catalog(config) -> None:
        resolver = ConfigResolver(config)
        raw_catalog = resolver.get("fmoe_v2_layout_catalog", None)
        if raw_catalog is None:
            return
        catalog = copy.deepcopy(list(raw_catalog))
        ids = {str(entry.get("id", "")).strip() for entry in catalog if isinstance(entry, dict)}
        if "L34" not in ids:
            catalog.append(
                {
                    "id": "L34",
                    "execution": "serial",
                    "global_pre_layers": 1,
                    "global_post_layers": 0,
                    "stages": {
                        "macro": {"pass_layers": 0, "moe_blocks": 1},
                        "mid": {"pass_layers": 0, "moe_blocks": 0},
                        "micro": {"pass_layers": 0, "moe_blocks": 0},
                    },
                }
            )
        if "L35" not in ids:
            catalog.append(
                {
                    "id": "L35",
                    "execution": "serial",
                    "global_pre_layers": 1,
                    "global_post_layers": 0,
                    "stages": {
                        "macro": {"pass_layers": 0, "moe_blocks": 1},
                        "mid": {"pass_layers": 0, "moe_blocks": 1},
                        "micro": {"pass_layers": 0, "moe_blocks": 0},
                    },
                }
            )
        _set_config_key(config, "fmoe_v2_layout_catalog", catalog)

    def __init__(self, config, dataset):
        self._ensure_layout_catalog(config)
        super().__init__(config, dataset)

        resolver = ConfigResolver(config)
        self.router_feature_proj_dim = max(int(resolver.get("router_feature_proj_dim", 0)), 0)
        self.router_feature_proj_layers = max(int(resolver.get("router_feature_proj_layers", 1)), 1)
        self.router_feature_scale = float(resolver.get("router_feature_scale", 1.0))
        self.router_hidden_scale = float(resolver.get("router_hidden_scale", 1.0))
        self.router_group_feature_scale = float(resolver.get("router_group_feature_scale", 1.0))
        self.rule_agreement_lambda = float(resolver.get("rule_agreement_lambda", 0.0))
        self.group_coverage_lambda = float(resolver.get("group_coverage_lambda", 0.0))

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        stage_expert_lists = {stage_name: list(expert_dict.values()) for stage_name, expert_dict in STAGES}
        stage_expert_names = {stage_name: list(expert_dict.keys()) for stage_name, expert_dict in STAGES}
        expert_hidden_by_stage = self._parse_stage_int_map(
            resolver.get("expert_hidden_by_stage", {}),
            default_value=self.d_expert_hidden,
        )
        expert_depth_by_stage = self._parse_stage_int_map(
            resolver.get("expert_depth_by_stage", {}),
            default_value=1,
        )
        self.stage_executor = StageExecutorN2(
            layout=self.layout,
            d_model=self.d_model,
            n_features=self.n_features,
            d_feat_emb=self.d_feat_emb,
            d_expert_hidden=self.d_expert_hidden,
            d_router_hidden=self.d_router_hidden,
            expert_depth_by_stage=expert_depth_by_stage,
            expert_hidden_by_stage=expert_hidden_by_stage,
            expert_scale=self.expert_scale,
            feature_bank_dim=self.feature_encoder.bank_dim,
            stage_top_k=self.moe_top_k,
            dropout=self.dropout,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            col2idx=col2idx,
            stage_expert_lists=stage_expert_lists,
            stage_expert_names=stage_expert_names,
            router_impl=self.router_impl,
            router_impl_by_stage=self.router_impl_by_stage,
            rule_router_cfg=self.rule_router_cfg,
            rule_bias_scale=self.rule_bias_scale,
            router_use_hidden=self.router_use_hidden,
            router_use_feature=self.router_use_feature,
            expert_use_hidden=self.expert_use_hidden,
            expert_use_feature=self.expert_use_feature,
            macro_routing_scope=str(resolver.get("macro_routing_scope", "session")).lower(),
            macro_session_pooling=str(resolver.get("macro_session_pooling", "mean")).lower(),
            mid_router_temperature=self.mid_router_temperature,
            micro_router_temperature=self.micro_router_temperature,
            mid_router_feature_dropout=self.mid_router_feature_dropout,
            micro_router_feature_dropout=self.micro_router_feature_dropout,
            use_valid_ratio_gating=self.use_valid_ratio_gating,
            parallel_stage_gate_top_k=self.parallel_stage_gate_top_k,
            parallel_stage_gate_temperature=self.parallel_stage_gate_temperature,
            inter_layer_style=self.stage_inter_layer_style,
            router_group_feature_mode=self.router_group_feature_mode,
            moe_block_variant=self.moe_block_variant,
            router_feature_proj_dim=self.router_feature_proj_dim,
            router_feature_proj_layers=self.router_feature_proj_layers,
            router_feature_scale=self.router_feature_scale,
            router_hidden_scale=self.router_hidden_scale,
            router_group_feature_scale=self.router_group_feature_scale,
        )
        self.stage_executor.apply(self._init_weights)
        self._stage_n_experts = self.stage_executor.stage_n_experts()
        self._stage_expert_names = self.stage_executor.stage_expert_names()
        self._stage_group_names = self.stage_executor.stage_group_names()
        self.moe_logger = MoELogger(self._stage_expert_names)
        self.group_logger = MoELogger(self._stage_group_names)
        self._reset_router_epoch_stats()
        self.set_schedule_epoch(epoch_idx=self._schedule_epoch, max_epochs=self._schedule_total_epochs, log_now=True)

        for key, value in {
            "router_feature_proj_dim": self.router_feature_proj_dim,
            "router_feature_proj_layers": self.router_feature_proj_layers,
            "router_feature_scale": self.router_feature_scale,
            "router_hidden_scale": self.router_hidden_scale,
            "router_group_feature_scale": self.router_group_feature_scale,
            "rule_agreement_lambda": self.rule_agreement_lambda,
            "group_coverage_lambda": self.group_coverage_lambda,
        }.items():
            _set_config_key(config, key, value)

        logger.info(
            "FeaturedMoE_N2 extras: feature_proj_dim=%s feature_proj_layers=%s "
            "feature_scale=%.4f hidden_scale=%.4f group_feature_scale=%.4f "
            "rule_agreement_lambda=%.6f group_coverage_lambda=%.6f",
            self.router_feature_proj_dim,
            self.router_feature_proj_layers,
            self.router_feature_scale,
            self.router_hidden_scale,
            self.router_group_feature_scale,
            self.rule_agreement_lambda,
            self.group_coverage_lambda,
        )

    def _compute_rule_agreement_loss(
        self,
        gate_logits: Dict[str, torch.Tensor],
        router_aux: Dict[str, Dict[str, torch.Tensor]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        target_map = dict((router_aux or {}).get("rule_target_logits", {}) or {})
        if not target_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key, target_logits in target_map.items():
            student_logits = gate_logits.get(stage_key)
            if student_logits is None:
                continue
            mask = self._sequence_valid_mask(item_seq_len, student_logits.size(1), student_logits.device).float()
            teacher = F.softmax(target_logits.detach(), dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            kl = F.kl_div(student_log_probs, teacher, reduction="none").sum(dim=-1)
            denom = mask.sum().clamp(min=1.0)
            losses.append((kl * mask).sum() / denom)
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_group_coverage_reward(
        self,
        router_aux: Dict[str, Dict[str, torch.Tensor]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        group_map = dict((router_aux or {}).get("group_weights", {}) or {})
        if not group_map:
            return self.item_embedding.weight.new_tensor(0.0)
        rewards = []
        for group_weights in group_map.values():
            if group_weights.size(-1) <= 1:
                continue
            mask = self._sequence_valid_mask(item_seq_len, group_weights.size(1), group_weights.device).float().unsqueeze(-1)
            denom = mask.sum().clamp(min=1.0)
            usage = (group_weights * mask).sum(dim=(0, 1)) / denom
            usage = usage / usage.sum().clamp(min=1e-8)
            entropy = -(usage.clamp(min=1e-8) * usage.clamp(min=1e-8).log()).sum()
            rewards.append(entropy / math.log(float(group_weights.size(-1))))
        return torch.stack(rewards).mean() if rewards else self.item_embedding.weight.new_tensor(0.0)

    def calculate_loss(self, interaction):
        item_seq, item_seq_len, routing_item_seq_len = self._resolve_forward_lengths(interaction)
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, aux_data = self.forward(
            item_seq,
            item_seq_len,
            feat,
            routing_item_seq_len=routing_item_seq_len,
        )
        logits = seq_output @ self.item_embedding.weight.T
        ce_loss = F.cross_entropy(logits, pos_items)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.use_aux_loss and self.balance_loss_lambda > 0:
            aux_loss = aux_loss + compute_expert_aux_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len=item_seq_len,
                balance_lambda=self.balance_loss_lambda,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_stage_merge_aux_loss(
                aux_data.get("stage_merge_weights"),
                item_seq_len=item_seq_len,
                balance_lambda=self.balance_loss_lambda,
                enabled=self.stage_merge_aux_enable,
                scale=self.stage_merge_aux_lambda_scale,
                device=ce_loss.device,
            )
            if self.ffn_moe and aux_data.get("ffn_moe_weights"):
                for stage_weights in aux_data["ffn_moe_weights"].values():
                    aux_loss = aux_loss + self.balance_loss_lambda * load_balance_loss(
                        stage_weights,
                        self.n_ffn_experts,
                    )

        router_aux = aux_data.get("router_aux", {}) or {}
        z_logit_map = dict(router_aux.get("learned_gate_logits", {}) or {})
        if not z_logit_map:
            z_logit_map = aux_data.get("gate_logits", {})
        if self.z_loss_lambda > 0:
            aux_loss = aux_loss + self.z_loss_lambda * self._compute_router_z_loss(z_logit_map, item_seq_len)

        entropy_scale = self._aux_until_active(self.gate_entropy_until)
        if self.gate_entropy_lambda > 0 and entropy_scale > 0:
            aux_loss = aux_loss - (self.gate_entropy_lambda * entropy_scale) * self._compute_gate_entropy_reg(
                aux_data.get("gate_weights", {}),
                item_seq_len,
            )

        if self.rule_agreement_lambda > 0:
            aux_loss = aux_loss + self.rule_agreement_lambda * self._compute_rule_agreement_loss(
                aux_data.get("gate_logits", {}),
                router_aux,
                item_seq_len,
            )
        if self.group_coverage_lambda > 0:
            aux_loss = aux_loss - self.group_coverage_lambda * self._compute_group_coverage_reward(
                router_aux,
                item_seq_len,
            )

        if self.fmoe_debug_logging and self.training and aux_data.get("gate_weights"):
            self.moe_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                item_seq_len=item_seq_len,
            )

        return ce_loss + aux_loss
