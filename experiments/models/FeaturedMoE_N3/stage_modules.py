"""Stage/layout modules for FeaturedMoE_N3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FeaturedMoE.moe_stages import _scaled_expert_names
from ..FeaturedMoE.routers import Router, RuleSoftRouter
from ..FeaturedMoE_N.stage_modules import _normalize_top_k, _softmax_with_top_k


_STAGE_NAMES = ("macro", "mid", "micro")
_VALID_ROUTER_GRANULARITY = {"session", "token"}
_VALID_COMPUTE_MODE = {"none", "dense_plain", "moe"}
_VALID_ROUTER_MODE = {"none", "learned", "rule_soft"}
_VALID_ROUTER_SOURCE = {"hidden", "feature", "both"}
_VALID_FACTORED_GROUP_ROUTER_SOURCE = {"hidden", "feature", "both"}
_VALID_FACTORED_COMBINE_MODE = {"add", "hir", "fac_group"}
_VALID_FEATURE_ENCODER = {"linear", "complex"}
_VALID_FEATURE_INJECTION = {"none", "film", "gated_bias", "group_gated_bias"}
_VALID_ROUTER_TYPE = {"standard", "factored"}
_VALID_RESIDUAL_MODE = {
    "base",
    "shared_only",
    "shared_moe_fixed",
    "shared_moe_learned",
    "shared_moe_global",
    "shared_moe_learned_warmup",
}


def _get_activation(name: str):
    key = str(name or "gelu").lower().strip()
    if key == "gelu":
        return F.gelu
    if key == "relu":
        return F.relu
    if key == "tanh":
        return torch.tanh
    if key == "sigmoid":
        return torch.sigmoid
    raise ValueError(f"Unsupported hidden_act: {name}")


@dataclass
class StageRuntimeConfigN3:
    stage_name: str
    d_model: int
    d_ff: int
    d_feat_emb: int
    d_expert_hidden: int
    d_router_hidden: int
    expert_depth: int
    expert_scale: int
    top_k: Optional[int]
    dropout: float
    attn_dropout: float
    hidden_act: str
    layer_norm_eps: float
    stage_feature_indices: Tuple[int, ...]
    stage_feature_names: Tuple[str, ...]
    stage_family_features: Dict[str, list[str]]
    stage_feature_encoder_mode: str
    stage_compute_mode: str
    stage_router_mode: str
    stage_router_source: str
    stage_feature_injection: str
    routing_granularity: str
    session_pooling: str
    rule_router_cfg: Dict[str, object]
    rule_bias_scale: float
    feature_group_bias_lambda: float
    feature_group_prior_temperature: float
    stage_router_type: str
    stage_factored_group_router_source: str
    factored_group_logit_scale: float
    factored_intra_logit_scale: float
    stage_factored_combine_mode: str
    router_temperature: float
    dense_hidden_scale: float
    stage_residual_mode: str
    residual_alpha_fixed: float
    residual_alpha_init: float
    shared_ffn_scale: float


class SASRecStyleAttentionBlock(nn.Module):
    """Post-LN self-attention block matching RecBole SASRec semantics."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if int(d_model) % int(n_heads) != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.num_attention_heads = int(n_heads)
        self.attention_head_size = int(d_model) // int(n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(int(d_model), self.all_head_size)
        self.key = nn.Linear(int(d_model), self.all_head_size)
        self.value = nn.Linear(int(d_model), self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(float(attn_dropout_prob))
        self.dense = nn.Linear(int(d_model), int(d_model))
        self.out_dropout = nn.Dropout(float(hidden_dropout_prob))
        self.layer_norm = nn.LayerNorm(int(d_model), eps=float(layer_norm_eps))

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_x_shape)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_tensor = hidden_states
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self._transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self._transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self._transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class SASRecStyleFFNBlock(nn.Module):
    """Post-LN FFN block matching RecBole SASRec semantics."""

    def __init__(
        self,
        *,
        d_model: int,
        d_ff: int,
        hidden_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.dense_1 = nn.Linear(int(d_model), int(d_ff))
        self.dense_2 = nn.Linear(int(d_ff), int(d_model))
        self.dropout = nn.Dropout(float(hidden_dropout_prob))
        self.layer_norm = nn.LayerNorm(int(d_model), eps=float(layer_norm_eps))
        self.activation = _get_activation(hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_tensor = hidden_states
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class SASRecStyleLayerBlock(nn.Module):
    """One SASRec-style layer = attn then FFN."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_ff: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = SASRecStyleAttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.ffn = SASRecStyleFFNBlock(
            d_model=d_model,
            d_ff=d_ff,
            hidden_dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attn(hidden_states, attention_mask)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class _StageFeatureEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, *, mode: str):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.mode = str(mode).lower().strip()
        if self.mode not in _VALID_FEATURE_ENCODER:
            raise ValueError(f"Unsupported stage_feature_encoder_mode: {mode}")
        if self.d_in <= 0:
            self.net = None
        elif self.mode == "linear":
            self.net = nn.Linear(self.d_in, self.d_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(self.d_in, self.d_out),
                nn.GELU(),
                nn.Linear(self.d_out, self.d_out),
            )

    def forward(self, stage_raw_feat: torch.Tensor) -> torch.Tensor:
        if self.net is None or stage_raw_feat.size(-1) <= 0:
            return stage_raw_feat.new_zeros(stage_raw_feat.size(0), stage_raw_feat.size(1), self.d_out)
        return self.net(stage_raw_feat)


class _ExpertMLP(nn.Module):
    def __init__(self, *, d_model: int, d_hidden: int, depth: int, dropout: float):
        super().__init__()
        depth = max(int(depth), 1)
        layers = []
        in_dim = int(d_model)
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, int(d_hidden)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_dim = int(d_hidden)
        layers.extend([nn.Linear(in_dim, int(d_hidden)), nn.GELU(), nn.Dropout(float(dropout))])
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(int(d_hidden), int(d_model))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.backbone(hidden_states))


class N3StageBlock(nn.Module):
    """Shared per-stage FFN/MoE block used by layer_layout stage tokens."""

    def __init__(self, cfg: StageRuntimeConfigN3):
        super().__init__()
        self.stage_name = str(cfg.stage_name).lower().strip()
        self.stage_compute_mode = str(cfg.stage_compute_mode).lower().strip()
        self.stage_router_mode = str(cfg.stage_router_mode).lower().strip()
        self.stage_router_source = str(cfg.stage_router_source).lower().strip()
        self.stage_feature_injection = str(cfg.stage_feature_injection).lower().strip()
        self.routing_granularity = str(cfg.routing_granularity).lower().strip()
        self.session_pooling = str(cfg.session_pooling).lower().strip()
        self.rule_router_cfg = dict(cfg.rule_router_cfg or {})
        self.rule_bias_scale = float(cfg.rule_bias_scale)
        self.feature_group_bias_lambda = float(cfg.feature_group_bias_lambda)
        self.feature_group_prior_temperature = max(float(cfg.feature_group_prior_temperature), 1e-6)
        self.stage_router_type = str(cfg.stage_router_type).lower().strip()
        self.stage_factored_group_router_source = str(cfg.stage_factored_group_router_source).lower().strip()
        self.factored_group_logit_scale = float(cfg.factored_group_logit_scale)
        self.factored_intra_logit_scale = float(cfg.factored_intra_logit_scale)
        self.stage_factored_combine_mode = str(cfg.stage_factored_combine_mode).lower().strip()
        if self.stage_router_type not in _VALID_ROUTER_TYPE:
            raise ValueError(f"Unsupported stage_router_type: {cfg.stage_router_type}")
        if self.stage_factored_group_router_source not in _VALID_FACTORED_GROUP_ROUTER_SOURCE:
            raise ValueError(
                f"Unsupported stage_factored_group_router_source: {cfg.stage_factored_group_router_source}"
            )
        if self.stage_factored_combine_mode not in _VALID_FACTORED_COMBINE_MODE:
            raise ValueError(f"Unsupported stage_factored_combine_mode: {cfg.stage_factored_combine_mode}")
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.layer_norm = nn.LayerNorm(int(cfg.d_model), eps=float(cfg.layer_norm_eps))
        self.activation = _get_activation(cfg.hidden_act)
        self.current_top_k = _normalize_top_k(cfg.top_k, n_experts=1_000_000)
        self.current_router_temperature = max(float(cfg.router_temperature), 1e-6)
        self.current_alpha_scale = 1.0
        self.d_model = int(cfg.d_model)
        self.d_feat_emb = int(cfg.d_feat_emb)
        self.stage_residual_mode = str(cfg.stage_residual_mode).lower().strip()
        if self.stage_residual_mode not in _VALID_RESIDUAL_MODE:
            raise ValueError(f"Unsupported stage_residual_mode: {cfg.stage_residual_mode}")
        self.residual_alpha_fixed = float(cfg.residual_alpha_fixed)
        self.shared_ffn_scale = float(cfg.shared_ffn_scale)
        self.pre_residual_ln = nn.LayerNorm(int(cfg.d_model), eps=float(cfg.layer_norm_eps))
        self.shared_fc1 = nn.Linear(int(cfg.d_model), int(max(round(cfg.d_ff * self.shared_ffn_scale), cfg.d_model)))
        self.shared_fc2 = nn.Linear(int(max(round(cfg.d_ff * self.shared_ffn_scale), cfg.d_model)), int(cfg.d_model))
        self.shared_drop = nn.Dropout(float(cfg.dropout))
        self.residual_alpha = None
        if self.stage_residual_mode in {"shared_moe_learned", "shared_moe_global", "shared_moe_learned_warmup"}:
            self.residual_alpha = nn.Parameter(torch.tensor(float(cfg.residual_alpha_init)))
        self.stage_feature_names = list(cfg.stage_feature_names)
        self.stage_family_features = {
            str(name): list(cols) for name, cols in dict(cfg.stage_family_features or {}).items()
        }
        self.base_names = [name for name, cols in self.stage_family_features.items() if cols]
        if not self.base_names:
            self.base_names = [name for name in self.stage_family_features.keys()]
        self.group_names = list(self.base_names)
        self.expert_names = _scaled_expert_names(self.base_names, int(cfg.expert_scale))
        self.n_experts = len(self.expert_names) if self.stage_compute_mode == "moe" else 0
        self.n_base_groups = max(len(self.base_names), 1)
        expert_group_idx = [idx // max(int(cfg.expert_scale), 1) for idx in range(max(self.n_experts, 1))]
        self.register_buffer(
            "expert_group_idx",
            torch.tensor(expert_group_idx, dtype=torch.long),
            persistent=False,
        )

        if self.stage_compute_mode not in _VALID_COMPUTE_MODE:
            raise ValueError(f"Unsupported stage_compute_mode: {cfg.stage_compute_mode}")
        if self.stage_router_mode not in _VALID_ROUTER_MODE:
            raise ValueError(f"Unsupported stage_router_mode: {cfg.stage_router_mode}")
        if self.stage_router_source not in _VALID_ROUTER_SOURCE:
            raise ValueError(f"Unsupported stage_router_source: {cfg.stage_router_source}")
        if self.stage_feature_injection not in _VALID_FEATURE_INJECTION:
            raise ValueError(f"Unsupported stage_feature_injection: {cfg.stage_feature_injection}")
        if self.routing_granularity not in _VALID_ROUTER_GRANULARITY:
            raise ValueError(f"Unsupported routing_granularity: {cfg.routing_granularity}")
        if self.stage_name == "micro" and self.routing_granularity != "token":
            raise ValueError("micro routing_granularity must be 'token'")
        if self.stage_compute_mode == "dense_plain" and self.stage_router_mode != "none":
            raise ValueError("dense_plain stage requires router_mode='none'")
        if self.stage_compute_mode == "none" and self.stage_router_mode != "none":
            raise ValueError("compute_mode='none' requires router_mode='none'")
        if self.stage_compute_mode == "none" and self.stage_feature_injection != "none":
            raise ValueError("compute_mode='none' requires feature_injection='none'")
        if self.session_pooling not in {"mean", "last"}:
            self.session_pooling = "mean"

        self.register_buffer(
            "stage_feat_idx",
            torch.tensor(list(cfg.stage_feature_indices), dtype=torch.long),
            persistent=False,
        )
        self.stage_col2local = {name: idx for idx, name in enumerate(self.stage_feature_names)}
        self.group_feature_local_indices = []
        for family_name in self.base_names:
            local_idx = [
                self.stage_col2local[name]
                for name in self.stage_family_features.get(family_name, [])
                if name in self.stage_col2local
            ]
            self.group_feature_local_indices.append(local_idx)
        self.feature_encoder = _StageFeatureEncoder(
            d_in=len(self.stage_feature_names),
            d_out=int(cfg.d_feat_emb),
            mode=cfg.stage_feature_encoder_mode,
        )

        self.injection = None
        self.group_injections = None
        if self.stage_feature_injection == "group_gated_bias" and self.stage_compute_mode == "moe":
            self.group_injections = nn.ModuleList([
                nn.Linear(max(len(local_idx), 1), int(cfg.d_model) * 2)
                for local_idx in self.group_feature_local_indices
            ])
        elif self.stage_feature_injection not in {"none", "group_gated_bias"}:
            self.injection = nn.Linear(int(cfg.d_feat_emb), int(cfg.d_model) * 2)

        dense_hidden = max(
            int(round(float(cfg.d_ff) * float(cfg.dense_hidden_scale))),
            int(cfg.d_model),
        )
        self.dense_fc1 = None
        self.dense_fc2 = None
        self.experts = None
        if self.stage_compute_mode == "dense_plain":
            self.dense_fc1 = nn.Linear(int(cfg.d_model), dense_hidden)
            self.dense_fc2 = nn.Linear(dense_hidden, int(cfg.d_model))
        elif self.stage_compute_mode == "moe":
            self.experts = nn.ModuleList(
                [
                    _ExpertMLP(
                        d_model=int(cfg.d_model),
                        d_hidden=int(cfg.d_expert_hidden),
                        depth=int(cfg.expert_depth),
                        dropout=float(cfg.dropout),
                    )
                    for _ in range(self.n_experts)
                ]
            )

        self.learned_router = None
        self.rule_router = None
        self.group_router_net = None
        self.intra_router_net = None
        self.feature_projection = None  # Projection for feature-only factored router
        if self.stage_compute_mode == "moe":
            if len(self.stage_feature_names) > 0:
                selected_indices = []
                for local_idx in self.group_feature_local_indices:
                    for _ in range(max(int(cfg.expert_scale), 1)):
                        selected_indices.append(list(local_idx or [0]))
                if not selected_indices:
                    selected_indices = [[0] for _ in range(self.n_experts)]
                self.rule_router = RuleSoftRouter(
                    n_experts=self.n_experts,
                    n_stage_features=max(len(self.stage_feature_names), 1),
                    selected_feature_indices=selected_indices,
                    feature_names=list(self.stage_feature_names or ["_pad"]),
                    n_bins=int(self.rule_router_cfg.get("n_bins", 5)),
                    expert_bias=[0.0] * self.n_experts,
                    top_k=None,
                    variant="ratio_bins",
                )
            if self.stage_router_mode == "learned":
                if self.stage_router_type == "factored":
                    group_in_dim = 0
                    if self.stage_factored_group_router_source in {"hidden", "both"}:
                        group_in_dim += int(cfg.d_model)
                    if self.stage_factored_group_router_source in {"feature", "both"}:
                        group_in_dim += int(cfg.d_feat_emb)
                    self.group_router_net = nn.Linear(max(group_in_dim, 1), self.n_base_groups)
                    intra_in_dim = 0
                    if self.stage_router_source in {"hidden", "both"}:
                        intra_in_dim += int(cfg.d_model)
                    if self.stage_router_source in {"feature", "both"}:
                        intra_in_dim += int(cfg.d_feat_emb)
                    
                    # Feature-only projection: add projection layer to match router input dimension
                    intra_in_for_router = intra_in_dim
                    if self.stage_router_source == "feature" and intra_in_dim < int(cfg.d_router_hidden):
                        # Feature embedding is smaller than router hidden, use projection to expand
                        self.feature_projection = nn.Linear(intra_in_dim, int(cfg.d_router_hidden))
                        intra_in_for_router = int(cfg.d_router_hidden)
                    else:
                        # Use actual input dimension or d_model as fallback
                        intra_in_for_router = max(intra_in_dim, int(cfg.d_model))
                    
                    self.intra_router_net = Router(
                        d_in=intra_in_for_router,
                        n_experts=self.n_experts,
                        d_hidden=int(cfg.d_router_hidden),
                        top_k=None,
                        dropout=float(cfg.dropout),
                    )
                else:
                    router_in_dim = 0
                    if self.stage_router_source in {"hidden", "both"}:
                        router_in_dim += int(cfg.d_model)
                    if self.stage_router_source in {"feature", "both"}:
                        router_in_dim += int(cfg.d_feat_emb)
                    self.learned_router = Router(
                        d_in=max(router_in_dim, 1),
                        n_experts=self.n_experts,
                        d_hidden=int(cfg.d_router_hidden),
                        top_k=None,
                        dropout=float(cfg.dropout),
                    )

    @property
    def requires_features(self) -> bool:
        if self.stage_compute_mode == "none":
            return False
        if self.stage_compute_mode == "dense_plain" and self.stage_feature_injection == "none":
            return False
        if self.stage_compute_mode == "moe":
            if self.stage_router_mode == "rule_soft":
                return True
            if self.stage_router_source in {"feature", "both"}:
                return True
            if self.stage_feature_injection != "none":
                return True
            if self.stage_router_type == "factored":
                return True
        return False

    @property
    def supports_diagnostics(self) -> bool:
        return self.stage_compute_mode == "moe"

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if alpha_scale is not None:
            self.current_alpha_scale = max(float(alpha_scale), 0.0)
        if router_temperature is not None and self.stage_compute_mode == "moe":
            self.current_router_temperature = max(float(router_temperature), 1e-6)
        if top_k is not None and self.stage_compute_mode == "moe":
            self.current_top_k = _normalize_top_k(top_k, n_experts=self.n_experts)

    def _shared_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        shared_hidden = self.pre_residual_ln(hidden)
        delta = self.shared_fc1(shared_hidden)
        delta = self.activation(delta)
        delta = self.shared_fc2(delta)
        return self.shared_drop(delta)

    def _resolve_alpha(
        self,
        *,
        alpha_override: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.stage_residual_mode == "shared_moe_fixed":
            alpha = torch.tensor(self.residual_alpha_fixed, device=device, dtype=dtype)
        elif self.stage_residual_mode == "shared_moe_global":
            if alpha_override is None:
                alpha = torch.sigmoid(self.residual_alpha) if self.residual_alpha is not None else torch.tensor(0.5, device=device, dtype=dtype)
            else:
                alpha = torch.sigmoid(alpha_override.to(device=device, dtype=dtype))
        elif self.stage_residual_mode in {"shared_moe_learned", "shared_moe_learned_warmup"}:
            alpha = torch.sigmoid(self.residual_alpha) if self.residual_alpha is not None else torch.tensor(0.5, device=device, dtype=dtype)
        else:
            alpha = torch.tensor(1.0, device=device, dtype=dtype)

        if self.stage_residual_mode == "shared_moe_learned_warmup":
            alpha = alpha * float(self.current_alpha_scale)
        return alpha

    def _merge_residual(
        self,
        *,
        hidden: torch.Tensor,
        moe_delta: torch.Tensor,
        alpha_override: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        shared_delta = self._shared_delta(hidden)
        if self.stage_residual_mode == "base":
            merged_delta = moe_delta
            alpha_value = torch.tensor(1.0, device=hidden.device, dtype=hidden.dtype)
        elif self.stage_residual_mode == "shared_only":
            merged_delta = shared_delta
            alpha_value = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        else:
            alpha_value = self._resolve_alpha(alpha_override=alpha_override, device=hidden.device, dtype=hidden.dtype)
            merged_delta = shared_delta + alpha_value * moe_delta

        aux = {
            "shared_delta_norm": shared_delta.norm(dim=-1).detach(),
            "moe_delta_norm": moe_delta.norm(dim=-1).detach(),
            "residual_delta_norm": merged_delta.norm(dim=-1).detach(),
            "alpha_value": hidden.new_full((hidden.size(0), hidden.size(1)), float(alpha_value.detach().cpu().item())),
            "alpha_effective": hidden.new_full((hidden.size(0), hidden.size(1)), float(alpha_value.detach().cpu().item())),
        }
        return merged_delta, aux

    @staticmethod
    def _build_valid_mask(
        batch_size: int,
        seq_len: int,
        item_seq_len: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if item_seq_len is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        lens = item_seq_len.to(device=device).long().clamp(min=1, max=seq_len)
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        return arange < lens.unsqueeze(1)

    def _pool_sequence(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.session_pooling == "mean":
            weight = valid_mask.float().unsqueeze(-1)
            denom = weight.sum(dim=1).clamp(min=1.0)
            return (seq * weight).sum(dim=1) / denom
        if item_seq_len is None:
            idx = torch.full(
                (seq.size(0),),
                fill_value=max(seq.size(1) - 1, 0),
                dtype=torch.long,
                device=seq.device,
            )
        else:
            idx = item_seq_len.to(device=seq.device).long().clamp(min=1, max=seq.size(1)) - 1
        return seq[torch.arange(seq.size(0), device=seq.device), idx]

    def _stage_raw_features(
        self,
        feat: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if feat is None or self.stage_feat_idx.numel() <= 0:
            return torch.zeros(batch_size, seq_len, 0, device=device)
        return feat.index_select(-1, self.stage_feat_idx.to(device=feat.device))

    def _feature_context(
        self,
        *,
        stage_raw_feat: torch.Tensor,
        encoded_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.routing_granularity == "session":
            pooled_encoded = self._pool_sequence(encoded_feat, valid_mask, item_seq_len)
            pooled_raw = self._pool_sequence(stage_raw_feat, valid_mask, item_seq_len)
            encoded_ctx = pooled_encoded.unsqueeze(1).expand(-1, seq_len, -1)
            raw_ctx = pooled_raw.unsqueeze(1).expand(-1, seq_len, -1)
            return encoded_ctx, raw_ctx
        return encoded_feat, stage_raw_feat

    def _apply_injection(self, hidden: torch.Tensor, feature_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond_norm = hidden.new_zeros(hidden.size(0), hidden.size(1))
        if self.injection is None or feature_context.size(-1) <= 0:
            return hidden, cond_norm
        params = self.injection(feature_context)
        cond_norm = params.norm(dim=-1)
        if self.stage_feature_injection == "film":
            gamma, beta = params.chunk(2, dim=-1)
            return hidden * (1.0 + torch.tanh(gamma)) + beta, cond_norm
        gate, bias = params.chunk(2, dim=-1)
        return (hidden + bias) * torch.sigmoid(gate), cond_norm

    def _apply_group_injection(
        self,
        hidden: torch.Tensor,
        raw_feature_context: torch.Tensor,
    ) -> Tuple[list, torch.Tensor]:
        """Per-group gated bias: each expert group uses its own raw features for injection.

        Returns (group_hiddens, cond_norm) where group_hiddens is a list of n_base_groups
        tensors each shaped (B, S, D) — expert e in group g will forward through group_hiddens[g].
        """
        if self.group_injections is None or raw_feature_context.size(-1) <= 0:
            return [hidden] * self.n_base_groups, hidden.new_zeros(hidden.size(0), hidden.size(1))
        cond_norms = []
        group_hiddens = []
        for local_idx, inj in zip(self.group_feature_local_indices, self.group_injections):
            if local_idx:
                idx = torch.tensor(local_idx, dtype=torch.long, device=hidden.device)
                g_feat = raw_feature_context.index_select(-1, idx)
            else:
                g_feat = hidden.new_zeros(*hidden.shape[:-1], 1)
            params = inj(g_feat)
            cond_norms.append(params.norm(dim=-1))
            gate, bias = params.chunk(2, dim=-1)
            group_hiddens.append((hidden + bias) * torch.sigmoid(gate))
        cond_norm = torch.stack(cond_norms, dim=-1).mean(dim=-1)
        return group_hiddens, cond_norm

    def _build_router_input(
        self,
        *,
        hidden: torch.Tensor,
        encoded_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.routing_granularity == "session":
            parts = []
            if self.stage_router_source in {"hidden", "both"}:
                parts.append(self._pool_sequence(hidden, valid_mask, item_seq_len))
            if self.stage_router_source in {"feature", "both"}:
                parts.append(self._pool_sequence(encoded_feat, valid_mask, item_seq_len))
            if not parts:
                parts.append(self._pool_sequence(hidden, valid_mask, item_seq_len))
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        parts = []
        if self.stage_router_source in {"hidden", "both"}:
            parts.append(hidden)
        if self.stage_router_source in {"feature", "both"}:
            parts.append(encoded_feat)
        if not parts:
            parts.append(hidden)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _build_group_router_input(
        self,
        *,
        hidden: torch.Tensor,
        encoded_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.routing_granularity == "session":
            parts = []
            if self.stage_factored_group_router_source in {"hidden", "both"}:
                parts.append(self._pool_sequence(hidden, valid_mask, item_seq_len))
            if self.stage_factored_group_router_source in {"feature", "both"}:
                parts.append(self._pool_sequence(encoded_feat, valid_mask, item_seq_len))
            if not parts:
                parts.append(self._pool_sequence(encoded_feat, valid_mask, item_seq_len))
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        parts = []
        if self.stage_factored_group_router_source in {"hidden", "both"}:
            parts.append(hidden)
        if self.stage_factored_group_router_source in {"feature", "both"}:
            parts.append(encoded_feat)
        if not parts:
            parts.append(encoded_feat)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _compute_router_outputs(
        self,
        *,
        hidden: torch.Tensor,
        stage_raw_feat: torch.Tensor,
        encoded_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}
        if self.stage_router_mode == "rule_soft":
            if self.rule_router is None:
                raise RuntimeError("rule_soft router is not initialized.")
            if self.routing_granularity == "session":
                rule_input = self._pool_sequence(stage_raw_feat, valid_mask, item_seq_len)
            else:
                rule_input = stage_raw_feat
            raw_logits = self.rule_router.compute_logits(rule_input)
        else:
            if self.stage_router_type == "factored":
                # Factored router: group-level and intra-group logits are combined with tunable scales.
                if self.group_router_net is None or self.intra_router_net is None:
                    raise RuntimeError(f"factored router not initialized for stage {self.stage_name}.")
                if self.routing_granularity == "session":
                    feat_pooled = self._pool_sequence(encoded_feat, valid_mask, item_seq_len)
                    if self.stage_router_source in {"hidden", "both"}:
                        hidden_pooled = self._pool_sequence(hidden, valid_mask, item_seq_len)
                    else:
                        hidden_pooled = feat_pooled
                    if self.stage_router_source == "feature":
                        intra_input = feat_pooled
                        if self.feature_projection is not None:
                            intra_input = self.feature_projection(intra_input)  # d_feat_emb -> d_router_hidden
                    elif self.stage_router_source == "both":
                        intra_input = torch.cat([hidden_pooled, feat_pooled], dim=-1)
                    else:
                        intra_input = hidden_pooled
                else:
                    if self.stage_router_source == "feature":
                        intra_input = encoded_feat
                        if self.feature_projection is not None:
                            intra_input = self.feature_projection(intra_input)  # d_feat_emb -> d_router_hidden
                    elif self.stage_router_source == "both":
                        intra_input = torch.cat([hidden, encoded_feat], dim=-1)
                    else:
                        intra_input = hidden
                group_input = self._build_group_router_input(
                    hidden=hidden,
                    encoded_feat=encoded_feat,
                    valid_mask=valid_mask,
                    item_seq_len=item_seq_len,
                )
                group_logits = self.group_router_net(group_input)          # (..., n_groups)
                intra_logits = self.intra_router_net.net(intra_input)      # (..., n_experts)
                # Broadcast group logits to expert level: expert e in group g gets group_logits[g]
                e_grp = self.expert_group_idx[: self.n_experts].to(group_logits.device)
                expert_group_logits = group_logits.index_select(-1, e_grp)  # (..., n_experts)

                if self.stage_factored_combine_mode == "hir":
                    group_probs = F.softmax(self.factored_group_logit_scale * group_logits, dim=-1)
                    base_weights = torch.zeros_like(intra_logits)
                    for group_idx in range(self.n_base_groups):
                        mask = self.expert_group_idx[: self.n_experts] == group_idx
                        if not mask.any():
                            continue
                        idx = mask.nonzero(as_tuple=False).view(-1).to(device=intra_logits.device)
                        intra_group_logits = intra_logits.index_select(-1, idx)
                        intra_group_probs = F.softmax(self.factored_intra_logit_scale * intra_group_logits, dim=-1)
                        group_weight = group_probs[..., group_idx : group_idx + 1]
                        base_weights[..., idx] = group_weight * intra_group_probs
                    base_weights = base_weights / base_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    raw_logits = torch.log(base_weights.clamp(min=1e-8)) * self.current_router_temperature
                else:
                    # add / fac_group: dense expert logits + per-group importance bias.
                    raw_logits = (
                        self.factored_group_logit_scale * expert_group_logits
                        + self.factored_intra_logit_scale * intra_logits
                    )
                aux["factored_group_logits"] = group_logits
                aux["factored_group_logit_scale"] = group_logits.new_full(
                    group_logits.shape[:-1],
                    float(self.factored_group_logit_scale),
                )
                aux["factored_intra_logit_scale"] = group_logits.new_full(
                    group_logits.shape[:-1],
                    float(self.factored_intra_logit_scale),
                )
                aux["factored_combine_mode"] = self.stage_factored_combine_mode
            else:
                if self.learned_router is None:
                    raise RuntimeError("learned router is not initialized.")
                router_input = self._build_router_input(
                    hidden=hidden,
                    encoded_feat=encoded_feat,
                    valid_mask=valid_mask,
                    item_seq_len=item_seq_len,
                )
                raw_logits = self.learned_router.net(router_input)
            if stage_raw_feat.size(-1) > 0:
                rule_input = (
                    self._pool_sequence(stage_raw_feat, valid_mask, item_seq_len)
                    if self.routing_granularity == "session"
                    else stage_raw_feat
                )
                group_prior = self._compute_group_feature_prior(rule_input)
                aux["group_prior"] = group_prior
                if self.feature_group_bias_lambda > 0 and group_prior.size(-1) > 0:
                    expert_prior = self._expand_group_tensor_to_experts(group_prior)
                    raw_logits = raw_logits + self.feature_group_bias_lambda * torch.log(expert_prior.clamp(min=1e-8))
                rule_router = self.rule_router
                if rule_router is None:
                    raise RuntimeError("rule target router is not initialized.")
                rule_logits = rule_router.compute_logits(rule_input)
                aux["rule_target_logits"] = rule_logits / self.current_router_temperature
                if self.rule_bias_scale > 0:
                    raw_logits = raw_logits + self.rule_bias_scale * rule_logits

        scaled_logits = raw_logits / self.current_router_temperature
        weights = _softmax_with_top_k(scaled_logits, self.current_top_k)
        if self.routing_granularity == "session":
            weights = weights.unsqueeze(1).expand(-1, seq_len, -1)
            scaled_logits = scaled_logits.unsqueeze(1).expand(-1, seq_len, -1)
            if "rule_target_logits" in aux:
                aux["rule_target_logits"] = aux["rule_target_logits"].unsqueeze(1).expand(-1, seq_len, -1)
            if "group_prior" in aux:
                aux["group_prior"] = aux["group_prior"].unsqueeze(1).expand(-1, seq_len, -1)
            if "factored_group_logits" in aux:
                aux["factored_group_logits"] = aux["factored_group_logits"].unsqueeze(1).expand(-1, seq_len, -1)
        if self.stage_router_mode == "learned":
            aux["learned_gate_logits"] = scaled_logits
        return weights, scaled_logits, aux

    def _aggregate_group_weights(self, gate_weights: torch.Tensor) -> torch.Tensor:
        pieces = []
        for group_idx in range(self.n_base_groups):
            mask = self.expert_group_idx[: self.n_experts] == group_idx
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).view(-1).to(device=gate_weights.device)
            pieces.append(gate_weights.index_select(-1, idx).sum(dim=-1, keepdim=True))
        return torch.cat(pieces, dim=-1) if pieces else gate_weights.new_zeros(*gate_weights.shape[:-1], 0)

    def _compute_group_feature_prior(self, stage_raw_feat: torch.Tensor) -> torch.Tensor:
        if stage_raw_feat.size(-1) <= 0 or self.n_base_groups <= 0:
            return stage_raw_feat.new_zeros(*stage_raw_feat.shape[:-1], max(self.n_base_groups, 1))
        scores = []
        for local_idx in self.group_feature_local_indices:
            if local_idx:
                idx = torch.tensor(local_idx, dtype=torch.long, device=stage_raw_feat.device)
                score = stage_raw_feat.index_select(-1, idx).abs().mean(dim=-1)
            else:
                score = stage_raw_feat.new_zeros(stage_raw_feat.shape[:-1])
            scores.append(score.unsqueeze(-1))
        group_scores = torch.cat(scores, dim=-1)
        if group_scores.size(-1) <= 1:
            return torch.ones_like(group_scores)
        return F.softmax(group_scores / self.feature_group_prior_temperature, dim=-1)

    def _expand_group_tensor_to_experts(self, group_tensor: torch.Tensor) -> torch.Tensor:
        if group_tensor.size(-1) <= 0 or self.n_experts <= 0:
            return group_tensor.new_zeros(*group_tensor.shape[:-1], 0)
        pieces = []
        for expert_idx in range(self.n_experts):
            group_idx = int(self.expert_group_idx[expert_idx].item())
            pieces.append(group_tensor[..., group_idx : group_idx + 1])
        return torch.cat(pieces, dim=-1)

    def _dense_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.dense_fc1 is None or self.dense_fc2 is None:
            return torch.zeros_like(hidden)
        delta = self.dense_fc1(hidden)
        delta = self.activation(delta)
        delta = self.dense_fc2(delta)
        return self.dropout(delta)

    def _expert_delta(self, hidden: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.experts is None:
            zeros = hidden.new_zeros(hidden.size(0), hidden.size(1), self.n_experts, self.d_model)
            return hidden.new_zeros(hidden.shape), zeros
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        delta = self.dropout(delta)
        return delta, expert_outputs

    def _expert_delta_group_hidden(
        self,
        group_hiddens: list,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run experts where each expert group uses a group-specific hidden state.

        group_hiddens: list of n_base_groups tensors, each (B, S, D).
        Returns (delta, expert_outputs) matching _expert_delta signatures.
        """
        if self.experts is None:
            b, s = weights.shape[:2]
            zeros = weights.new_zeros(b, s, self.n_experts, self.d_model)
            return weights.new_zeros(b, s, self.d_model), zeros
        expert_outputs_list = []
        for e_idx in range(self.n_experts):
            g_idx = int(self.expert_group_idx[e_idx].item())
            expert_outputs_list.append(self.experts[e_idx](group_hiddens[g_idx]))
        expert_outputs = torch.stack(expert_outputs_list, dim=-2)  # (B, S, E, D)
        delta = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return self.dropout(delta), expert_outputs

    def forward(
        self,
        hidden: torch.Tensor,
        feat: Optional[torch.Tensor],
        item_seq_len: Optional[torch.Tensor] = None,
        alpha_override: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        batch_size, seq_len, _ = hidden.shape
        valid_mask = self._build_valid_mask(batch_size, seq_len, item_seq_len, hidden.device)

        if self.stage_compute_mode == "none":
            return hidden, None, None, {}, {}

        stage_raw_feat = self._stage_raw_features(
            feat,
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden.device,
        )
        encoded_feat = self.feature_encoder(stage_raw_feat)
        feature_context, raw_feature_context = self._feature_context(
            stage_raw_feat=stage_raw_feat,
            encoded_feat=encoded_feat,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
            seq_len=seq_len,
        )

        group_hiddens = None
        if self.stage_feature_injection == "group_gated_bias" and self.stage_compute_mode == "moe":
            group_hiddens, cond_norm = self._apply_group_injection(hidden, raw_feature_context)
            hidden_in = hidden  # fallback for dense path; moe uses group_hiddens
        else:
            hidden_in, cond_norm = self._apply_injection(hidden, feature_context)

        if self.stage_compute_mode == "dense_plain":
            delta = self._dense_delta(hidden_in)
            next_hidden = self.layer_norm(hidden + delta)
            dense_aux = {
                "condition_norm": cond_norm.detach(),
                "delta_norm": delta.norm(dim=-1).detach(),
                "shared_delta_norm": hidden.new_zeros(hidden.size(0), hidden.size(1)),
                "moe_delta_norm": delta.norm(dim=-1).detach(),
                "residual_delta_norm": delta.norm(dim=-1).detach(),
                "alpha_value": hidden.new_ones(hidden.size(0), hidden.size(1)),
                "alpha_effective": hidden.new_ones(hidden.size(0), hidden.size(1)),
            }
            return next_hidden, None, None, {}, dense_aux

        gate_weights, gate_logits, router_aux = self._compute_router_outputs(
            hidden=hidden,
            stage_raw_feat=raw_feature_context,
            encoded_feat=feature_context,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
            seq_len=seq_len,
        )
        if group_hiddens is not None:
            delta, expert_outputs = self._expert_delta_group_hidden(group_hiddens, gate_weights)
        else:
            delta, expert_outputs = self._expert_delta(hidden_in, gate_weights)
        residual_delta, residual_aux = self._merge_residual(
            hidden=hidden,
            moe_delta=delta,
            alpha_override=alpha_override,
        )
        next_hidden = self.layer_norm(hidden + residual_delta)

        router_aux = dict(router_aux)
        router_aux["group_weights"] = self._aggregate_group_weights(gate_weights)
        router_aux["expert_group_idx"] = self.expert_group_idx[: self.n_experts].to(device=hidden.device)
        router_aux["expert_output_mean"] = expert_outputs.mean(dim=(0, 1)).detach()
        dense_aux = {
            "condition_norm": cond_norm.detach(),
            "delta_norm": delta.norm(dim=-1).detach(),
            "shared_delta_norm": residual_aux["shared_delta_norm"],
            "moe_delta_norm": residual_aux["moe_delta_norm"],
            "residual_delta_norm": residual_aux["residual_delta_norm"],
            "alpha_value": residual_aux["alpha_value"],
            "alpha_effective": residual_aux["alpha_effective"],
        }
        return next_hidden, gate_weights, gate_logits, router_aux, dense_aux
