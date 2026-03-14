"""FeaturedMoE_N3 model with layer_layout backbone."""

from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from ..FeaturedMoE.logging_utils import MoELogger
from ..FeaturedMoE.routers import load_balance_loss
from ..FeaturedMoE_N.featured_moe_n import FeaturedMoE_N
from ..FeaturedMoE_v2.config_schema import ConfigResolver
from ..FeaturedMoE_v2.losses import compute_expert_aux_loss, compute_stage_merge_aux_loss
from .diagnostics import N3DiagnosticCollector
from .feature_config import (
    ALL_FEATURE_COLUMNS,
    GROUP_ORDER,
    STAGE_NAMES,
    build_column_to_index,
    build_stage_feature_spec,
    feature_list_field,
    load_feature_meta_v3,
    validate_feature_meta_v3,
)
from .stage_executor import StageExecutorN3

logger = logging.getLogger(__name__)


_PLAIN_TOKENS = {"layer", "attn", "ffn"}


def _set_config_key(config_obj, key: str, value) -> None:
    try:
        config_obj[key] = value
    except Exception:
        pass
    final_cfg = getattr(config_obj, "final_config_dict", None)
    if isinstance(final_cfg, dict):
        final_cfg[key] = value


def _parse_layer_layout(raw_value) -> List[str]:
    if raw_value is None:
        return ["layer", "layer", "layer"]
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip().strip("'\"") for part in text.split(",")]
        out = [part for part in parts if part]
        return out or ["layer", "layer", "layer"]
    if isinstance(raw_value, (list, tuple)):
        out = [str(v).strip().lower() for v in raw_value if str(v).strip()]
        return out or ["layer", "layer", "layer"]
    return ["layer", "layer", "layer"]


def _parse_stage_str_map(raw_value, *, default_value: str) -> Dict[str, str]:
    default = {stage: str(default_value).lower().strip() for stage in STAGE_NAMES}
    if raw_value is None:
        return default
    if isinstance(raw_value, str):
        value = str(raw_value).lower().strip()
        return {stage: value for stage in STAGE_NAMES}
    if not isinstance(raw_value, dict):
        return default
    out = dict(default)
    for stage in STAGE_NAMES:
        value = raw_value.get(stage, raw_value.get(stage.capitalize(), default_value))
        out[stage] = str(value).lower().strip()
    return out


class FeaturedMoE_N3(FeaturedMoE_N):
    """N3 branch with explicit layer_layout and stage-wise controls."""

    @staticmethod
    def _ensure_layout_catalog(config) -> None:
        catalog = [
            {
                "id": "DSL0",
                "execution": "serial",
                "global_pre_layers": 0,
                "global_post_layers": 0,
                "stages": {
                    "macro": {"pass_layers": 0, "moe_blocks": 0},
                    "mid": {"pass_layers": 0, "moe_blocks": 0},
                    "micro": {"pass_layers": 0, "moe_blocks": 0},
                },
            }
        ]
        _set_config_key(config, "fmoe_v2_layout_catalog", copy.deepcopy(catalog))
        _set_config_key(config, "fmoe_v2_layout_id", 0)
        _set_config_key(config, "fmoe_stage_execution_mode", "serial")

    def __init__(self, config, dataset):
        self._ensure_layout_catalog(config)
        super().__init__(config, dataset)

        resolver = ConfigResolver(config)
        self.hidden_act = str(resolver.get("hidden_act", "gelu")).lower().strip()
        self.layer_norm_eps = float(resolver.get("layer_norm_eps", 1e-12))
        self.layer_layout = _parse_layer_layout(resolver.get("layer_layout", ["layer", "layer", "layer"]))
        self.macro_history_window = int(resolver.get("macro_history_window", 5))
        self.stage_router_granularity = _parse_stage_str_map(
            resolver.get("stage_router_granularity", None),
            default_value="session",
        )
        self.stage_router_granularity["micro"] = "token"
        self.stage_feature_encoder_mode = _parse_stage_str_map(
            resolver.get("stage_feature_encoder_mode", None),
            default_value="linear",
        )
        self.stage_compute_mode = _parse_stage_str_map(
            resolver.get("stage_compute_mode", None),
            default_value="moe",
        )
        self.stage_router_mode = _parse_stage_str_map(
            resolver.get("stage_router_mode", None),
            default_value="learned",
        )
        self.stage_router_source = _parse_stage_str_map(
            resolver.get("stage_router_source", None),
            default_value="both",
        )
        self.stage_feature_injection = _parse_stage_str_map(
            resolver.get("stage_feature_injection", None),
            default_value="none",
        )
        self.stage_router_type = _parse_stage_str_map(
            resolver.get("stage_router_type", None),
            default_value="standard",
        )
        self.stage_feature_family_mask = resolver.get("stage_feature_family_mask", {}) or {}
        self.dense_hidden_scale = float(resolver.get("dense_hidden_scale", 1.0))
        self.rule_agreement_lambda = float(resolver.get("rule_agreement_lambda", 0.0))
        self.group_coverage_lambda = float(resolver.get("group_coverage_lambda", 0.0))
        self.group_prior_align_lambda = float(resolver.get("group_prior_align_lambda", 0.0))
        self.feature_group_bias_lambda = float(resolver.get("feature_group_bias_lambda", 0.0))
        self.feature_group_prior_temperature = float(resolver.get("feature_group_prior_temperature", 1.0))
        self.factored_group_balance_lambda = float(resolver.get("factored_group_balance_lambda", 0.0))
        self.fmoe_diag_logging = bool(resolver.get("fmoe_diag_logging", True))
        self.fmoe_diag_sample_sessions = max(int(resolver.get("fmoe_diag_sample_sessions", 256)), 0)

        self._feature_ablation_mode = "none"
        self._feature_ablation_family: Optional[str] = None
        self._diag_collector: Optional[N3DiagnosticCollector] = None

        meta = load_feature_meta_v3(
            data_path=str(resolver.get("data_path", "")),
            dataset=str(resolver.get("dataset", "")),
        )
        self.feature_meta_v3 = validate_feature_meta_v3(meta) if meta else {}
        self.feature_spec = build_stage_feature_spec(
            macro_history_window=self.macro_history_window,
            stage_feature_family_mask=self.stage_feature_family_mask,
        )

        self.feature_base_fields = list(ALL_FEATURE_COLUMNS)
        self.feature_fields = [feature_list_field(col) for col in ALL_FEATURE_COLUMNS]
        self.n_features = len(self.feature_fields)
        self._missing_feature_field_once: set[str] = set()
        self._scalar_fallback_field_once: set[str] = set()

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self._feature_family_ablation_indices = {}
        for family_name in GROUP_ORDER:
            family_cols = []
            for stage_name in STAGE_NAMES:
                family_cols.extend(
                    list((self.feature_spec["stage_family_features"].get(stage_name, {}) or {}).get(family_name, []) or [])
                )
            self._feature_family_ablation_indices[family_name.lower()] = sorted(
                {int(col2idx[col]) for col in family_cols if col in col2idx}
            )
        expert_hidden_by_stage = self._parse_stage_int_map(
            resolver.get("expert_hidden_by_stage", {}),
            default_value=self.d_expert_hidden,
        )
        expert_depth_by_stage = self._parse_stage_int_map(
            resolver.get("expert_depth_by_stage", {}),
            default_value=1,
        )

        self.pre_transformer = None
        self.post_transformer = None
        self.stage_executor = StageExecutorN3(
            layer_layout=self.layer_layout,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            d_feat_emb=self.d_feat_emb,
            d_expert_hidden=self.d_expert_hidden,
            d_router_hidden=self.d_router_hidden,
            expert_depth_by_stage=expert_depth_by_stage,
            expert_hidden_by_stage=expert_hidden_by_stage,
            expert_scale=self.expert_scale,
            stage_top_k=self.moe_top_k,
            macro_session_pooling=str(resolver.get("macro_session_pooling", "mean")).lower(),
            stage_router_granularity=self.stage_router_granularity,
            stage_all_features=self.feature_spec["stage_all_features"],
            stage_family_features=self.feature_spec["stage_family_features"],
            stage_feature_encoder_mode=self.stage_feature_encoder_mode,
            stage_compute_mode=self.stage_compute_mode,
            stage_router_mode=self.stage_router_mode,
            stage_router_source=self.stage_router_source,
            stage_feature_injection=self.stage_feature_injection,
            rule_router_cfg=self.rule_router_cfg,
            rule_bias_scale=self.rule_bias_scale,
            feature_group_bias_lambda=self.feature_group_bias_lambda,
            feature_group_prior_temperature=self.feature_group_prior_temperature,
            stage_router_type=self.stage_router_type,
            mid_router_temperature=self.mid_router_temperature,
            micro_router_temperature=self.micro_router_temperature,
            dense_hidden_scale=self.dense_hidden_scale,
            col2idx=col2idx,
        )
        self.stage_executor.apply(self._init_weights)
        self._stage_n_experts = self.stage_executor.stage_n_experts()
        self._stage_expert_names = self.stage_executor.stage_expert_names()
        self._stage_group_names = self.stage_executor.stage_group_names()
        self.moe_logger = MoELogger(self._stage_expert_names)
        self.group_logger = MoELogger(self._stage_group_names)
        self._reset_router_epoch_stats()
        self.set_schedule_epoch(epoch_idx=self._schedule_epoch, max_epochs=self._schedule_total_epochs, log_now=True)

        self.n_pre_layer = 0
        self.n_pre_macro = 0
        self.n_pre_mid = 0
        self.n_pre_micro = 0
        self.n_post_layer = 0
        self.arch_layout_id = -1
        self.num_layers = int(sum(token in {"layer", "macro", "mid", "micro", "attn"} for token in self.layer_layout))
        self.n_total_attn_layers = int(sum(token in {"layer", "macro", "mid", "micro", "attn"} for token in self.layer_layout))
        self._uses_stage_tokens = any(token not in _PLAIN_TOKENS for token in self.layer_layout)
        self._requires_features = bool(self.stage_executor.requires_features)
        self._supports_diag = bool(self.stage_executor.supports_diagnostics)

        for key, value in {
            "layer_layout": list(self.layer_layout),
            "stage_router_granularity": dict(self.stage_router_granularity),
            "stage_feature_encoder_mode": dict(self.stage_feature_encoder_mode),
            "stage_compute_mode": dict(self.stage_compute_mode),
            "stage_router_mode": dict(self.stage_router_mode),
            "stage_router_source": dict(self.stage_router_source),
            "stage_feature_injection": dict(self.stage_feature_injection),
            "stage_router_type": dict(self.stage_router_type),
            "macro_history_window": self.macro_history_window,
            "stage_feature_family_mask": self.stage_feature_family_mask,
            "dense_hidden_scale": self.dense_hidden_scale,
            "group_prior_align_lambda": self.group_prior_align_lambda,
            "feature_group_bias_lambda": self.feature_group_bias_lambda,
            "feature_group_prior_temperature": self.feature_group_prior_temperature,
            "factored_group_balance_lambda": self.factored_group_balance_lambda,
            "hidden_act": self.hidden_act,
            "layer_norm_eps": self.layer_norm_eps,
            "fmoe_special_logging": bool(self.fmoe_special_logging),
            "fmoe_diag_logging": bool(self.fmoe_diag_logging),
        }.items():
            _set_config_key(config, key, value)

        logger.info(
            "FeaturedMoE_N3 init: layer_layout=%s compute=%s router_mode=%s router_source=%s "
            "feature_injection=%s routing=%s feature_groups=%s uses_stage=%s diag=%s meta_loaded=%s",
            self.layer_layout,
            self.stage_compute_mode,
            self.stage_router_mode,
            self.stage_router_source,
            self.stage_feature_injection,
            self.stage_router_granularity,
            self.feature_spec["stage_family_mask"],
            self._uses_stage_tokens,
            self._supports_diag,
            bool(self.feature_meta_v3),
        )

    def set_feature_ablation_mode(self, mode: str = "none", family: Optional[str] = None) -> None:
        normalized = str(mode or "none").lower().strip()
        if normalized not in {"none", "zero", "shuffle"}:
            normalized = "none"
        self._feature_ablation_mode = normalized
        family_name = str(family or "").strip().lower()
        self._feature_ablation_family = family_name if family_name in self._feature_family_ablation_indices else None

    def _feature_ablation_tag(self) -> str:
        if self._feature_ablation_mode == "none":
            return "none"
        if self._feature_ablation_family:
            return f"{self._feature_ablation_mode}:{self._feature_ablation_family}"
        return self._feature_ablation_mode

    @property
    def feature_family_names(self) -> List[str]:
        return list(GROUP_ORDER)

    def begin_diagnostic_eval(self, *, split_name: str) -> None:
        if not self.fmoe_diag_logging or not self._supports_diag:
            self._diag_collector = None
            return
        self._diag_collector = N3DiagnosticCollector(
            split_name=split_name,
            stage_family_features=self.feature_spec["stage_family_features"],
            stage_expert_names=self._stage_expert_names,
            all_feature_columns=ALL_FEATURE_COLUMNS,
            max_positions=int(self.max_seq_length),
            feature_mode=self._feature_ablation_tag(),
        )

    def end_diagnostic_eval(self) -> Optional[dict]:
        if self._diag_collector is None:
            return None
        payload = self._diag_collector.finalize()
        self._diag_collector = None
        return payload

    def _apply_feature_ablation(self, feat: torch.Tensor) -> torch.Tensor:
        family_indices = []
        if self._feature_ablation_family:
            family_indices = list(self._feature_family_ablation_indices.get(self._feature_ablation_family, []) or [])
            if not family_indices:
                return feat
        if self._feature_ablation_mode == "zero":
            if family_indices:
                out = feat.clone()
                out[..., family_indices] = 0.0
                return out
            return torch.zeros_like(feat)
        if self._feature_ablation_mode == "shuffle" and feat.size(0) > 1:
            perm = torch.randperm(feat.size(0), device=feat.device)
            if family_indices:
                out = feat.clone()
                shuffled = feat.index_select(0, perm)
                out[..., family_indices] = shuffled[..., family_indices]
                return out
            return feat.index_select(0, perm)
        return feat

    def _gather_features(self, interaction) -> torch.Tensor:
        feat_list = []
        item_seq = interaction[self.ITEM_SEQ]
        batch_size, seq_len = item_seq.shape
        device = item_seq.device

        for base_field, seq_field in zip(self.feature_base_fields, self.feature_fields):
            if seq_field in interaction:
                values = interaction[seq_field].float()
            elif base_field in interaction:
                values = interaction[base_field].float()
                if values.dim() == 1:
                    values = values.unsqueeze(1)
                if values.dim() == 2 and values.size(1) == 1:
                    values = values.expand(-1, seq_len)
                elif values.dim() == 2 and values.size(1) != seq_len:
                    values = values[:, :1].expand(-1, seq_len)
                if base_field not in self._scalar_fallback_field_once:
                    logger.info("Feature field '%s' using scalar fallback broadcast.", base_field)
                    self._scalar_fallback_field_once.add(base_field)
            else:
                values = torch.zeros(batch_size, seq_len, device=device)
                if seq_field not in self._missing_feature_field_once:
                    logger.warning("Feature field '%s' not found - using zeros.", seq_field)
                    self._missing_feature_field_once.add(seq_field)

            if values.dim() == 1:
                values = values.unsqueeze(1).expand(-1, seq_len)
            elif values.dim() == 3 and values.size(-1) == 1:
                values = values.squeeze(-1)
            feat_list.append(values.to(device=device))

        feat = torch.stack(feat_list, dim=-1)
        return self._apply_feature_ablation(feat)

    def _update_eval_diagnostics(self, *, interaction, feat, item_seq_len, aux_data) -> None:
        if self.training or self._diag_collector is None:
            return
        try:
            self._diag_collector.update(
                interaction=interaction,
                feat=feat,
                item_seq_len=item_seq_len,
                aux_data=aux_data,
            )
        except Exception:
            pass

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

    def _compute_group_prior_alignment_loss(
        self,
        router_aux: Dict[str, Dict[str, torch.Tensor]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        group_map = dict((router_aux or {}).get("group_weights", {}) or {})
        prior_map = dict((router_aux or {}).get("group_prior", {}) or {})
        if not group_map or not prior_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key in sorted(set(group_map) & set(prior_map)):
            student = group_map.get(stage_key)
            teacher = prior_map.get(stage_key)
            if student is None or teacher is None:
                continue
            if student.size(-1) <= 1 or teacher.size(-1) != student.size(-1):
                continue
            mask = self._sequence_valid_mask(item_seq_len, student.size(1), student.device).float()
            teacher_prob = teacher.detach()
            teacher_prob = teacher_prob / teacher_prob.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            student_prob = student / student.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kl = (teacher_prob * (teacher_prob.clamp(min=1e-8).log() - student_prob.clamp(min=1e-8).log())).sum(dim=-1)
            denom = mask.sum().clamp(min=1.0)
            losses.append((kl * mask).sum() / denom)
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def _compute_factored_group_balance_loss(
        self,
        router_aux: Dict[str, Dict[str, torch.Tensor]],
        item_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Load balance loss applied to factored group router logits.

        Penalizes collapse of group-level routing (when the factored router always
        sends everything to one group). Applied only when factored_group_logits are present.
        Uses the same mean-prob * mean-prob-softmax formulation as standard balance loss.
        """
        fgl_map = dict((router_aux or {}).get("factored_group_logits", {}) or {})
        if not fgl_map:
            return self.item_embedding.weight.new_tensor(0.0)
        losses = []
        for stage_key, fgl in fgl_map.items():
            if not torch.is_tensor(fgl) or fgl.size(-1) <= 1:
                continue
            group_probs = F.softmax(fgl, dim=-1)  # (..., n_groups)
            seq_len = fgl.size(1) if fgl.ndim == 3 else 1
            mask = self._sequence_valid_mask(item_seq_len, seq_len, fgl.device).float()
            if fgl.ndim == 3:
                valid_unsq = mask.unsqueeze(-1)
                denom = mask.sum().clamp(min=1.0)
                mean_prob = (group_probs * valid_unsq).sum(dim=(0, 1)) / denom
            else:
                mean_prob = group_probs.mean(dim=0)
            n_grps = float(mean_prob.size(-1))
            # balance_loss = n_groups * sum(mean_prob_i * mean_prob_i)  [encourages uniform distribution]
            losses.append(n_grps * (mean_prob * mean_prob).sum())
        return torch.stack(losses).mean() if losses else self.item_embedding.weight.new_tensor(0.0)

    def forward(self, item_seq, item_seq_len, feat=None):
        batch_size, seq_len = item_seq.shape
        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)
        attention_mask = self.get_attention_mask(item_seq)

        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        dense_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        stage_merge_weights = None
        stage_merge_logits = None

        tokens, gate_weights, gate_logits, router_aux, dense_aux, stage_merge_weights, stage_merge_logits = self.stage_executor(
            hidden=tokens,
            attention_mask=attention_mask,
            feat=feat,
            item_seq_len=item_seq_len,
        )

        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = tokens.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "stage_merge_weights": stage_merge_weights,
            "stage_merge_logits": stage_merge_logits,
            "ffn_moe_weights": {},
            "router_aux": router_aux,
            "dense_aux": dense_aux,
        }
        return seq_output, aux_data

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction) if self._requires_features else None
        seq_output, aux_data = self.forward(item_seq, item_seq_len, feat)
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
        if self.group_prior_align_lambda > 0:
            aux_loss = aux_loss + self.group_prior_align_lambda * self._compute_group_prior_alignment_loss(
                router_aux,
                item_seq_len,
            )
        if self.factored_group_balance_lambda > 0:
            aux_loss = aux_loss + self.factored_group_balance_lambda * self._compute_factored_group_balance_loss(
                router_aux,
                item_seq_len,
            )

        if self.fmoe_debug_logging and self.training and aux_data.get("gate_weights"):
            self.moe_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                item_seq_len=item_seq_len,
            )

        return ce_loss + aux_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        feat = self._gather_features(interaction) if self._requires_features else None
        seq_output, aux_data = self.forward(item_seq, item_seq_len, feat)
        self._update_eval_diagnostics(
            interaction=interaction,
            feat=feat,
            item_seq_len=item_seq_len,
            aux_data=aux_data,
        )
        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        feat = self._gather_features(interaction) if self._requires_features else None
        seq_output, aux_data = self.forward(item_seq, item_seq_len, feat)
        self._update_eval_diagnostics(
            interaction=interaction,
            feat=feat,
            item_seq_len=item_seq_len,
            aux_data=aux_data,
        )
        return seq_output @ self.item_embedding.weight.T
