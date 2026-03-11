"""Feature-individual MoE blocks for FeaturedMoE_Individual."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FeaturedMoE.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    build_column_to_index,
    build_stage_indices,
)
from ..FeaturedMoE.routers import Router


def _topk_softmax(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or int(top_k) <= 0:
        return F.softmax(logits, dim=-1)
    k = min(int(top_k), int(logits.shape[-1]))
    if k >= int(logits.shape[-1]):
        return F.softmax(logits, dim=-1)
    top_vals, top_idx = logits.topk(k, dim=-1)
    top_w = F.softmax(top_vals, dim=-1)
    out = torch.zeros_like(logits)
    out.scatter_(-1, top_idx, top_w)
    return out


def _scaled_feature_expert_names(feature_names: Sequence[str], expert_scale: int) -> List[str]:
    if int(expert_scale) <= 1:
        return [str(name) for name in feature_names]

    out: List[str] = []
    for feature_name in feature_names:
        for idx in range(int(expert_scale)):
            suffix = chr(ord("a") + idx)
            out.append(f"{feature_name}_{suffix}")
    return out


class HiddenExpertMLP(nn.Module):
    """Expert FFN that can optionally consume feature embeddings."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        dropout: float,
        d_feat_emb: int = 0,
        use_hidden: bool = True,
        use_feature: bool = False,
    ):
        super().__init__()
        if not (use_hidden or use_feature):
            raise ValueError("HiddenExpertMLP requires at least one input source.")

        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)

        in_dim = 0
        if self.use_hidden:
            in_dim += int(d_model)
        if self.use_feature:
            in_dim += int(d_feat_emb)

        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, hidden: torch.Tensor, feat_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        inputs = []
        if self.use_hidden:
            inputs.append(hidden)
        if self.use_feature:
            if feat_emb is None:
                raise ValueError("feat_emb must be provided when use_feature=True")
            inputs.append(feat_emb)
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        return self.net(x)


class FeatureIndividualStageMoE(nn.Module):
    """One stage that routes over individual features, then over feature-local experts."""

    def __init__(
        self,
        *,
        stage_name: str,
        feature_names: Sequence[str],
        stage_all_features: Sequence[str],
        col2idx: Dict[str, int],
        d_model: int = 128,
        d_feat_emb: int = 16,
        d_expert_hidden: int = 160,
        d_router_hidden: int = 64,
        expert_scale: int = 4,
        feature_top_k: Optional[int] = 4,
        inner_expert_top_k: Optional[int] = None,
        dropout: float = 0.1,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = False,
        outer_router_use_hidden: bool = True,
        outer_router_use_feature: bool = True,
        inner_router_use_hidden: bool = True,
        inner_router_use_feature: bool = True,
        router_temperature: float = 1.0,
    ):
        super().__init__()
        if int(expert_scale) < 1:
            raise ValueError(f"expert_scale must be >= 1, got {expert_scale}")
        if float(router_temperature) <= 0:
            raise ValueError(f"router_temperature must be > 0, got {router_temperature}")
        if not (expert_use_hidden or expert_use_feature):
            raise ValueError("Experts must use at least one input source.")
        if not (outer_router_use_hidden or outer_router_use_feature):
            raise ValueError("Outer router must use at least one input source.")
        if not (inner_router_use_hidden or inner_router_use_feature):
            raise ValueError("Inner router must use at least one input source.")

        self.stage_name = str(stage_name)
        self.group_names = [str(name) for name in feature_names]
        self.stage_all_features = [str(name) for name in stage_all_features]
        self.expert_scale = int(expert_scale)
        self.n_groups = len(self.group_names)
        self.n_experts = self.n_groups * self.expert_scale
        self.expert_names = _scaled_feature_expert_names(self.group_names, self.expert_scale)

        self.outer_router_use_hidden = bool(outer_router_use_hidden)
        self.outer_router_use_feature = bool(outer_router_use_feature)
        self.inner_router_use_hidden = bool(inner_router_use_hidden)
        self.inner_router_use_feature = bool(inner_router_use_feature)
        self.expert_use_hidden = bool(expert_use_hidden)
        self.expert_use_feature = bool(expert_use_feature)

        self.router_temperature = float(router_temperature)
        self.current_router_temperature = float(router_temperature)
        self.feature_top_k = None if feature_top_k is None or int(feature_top_k) <= 0 else int(feature_top_k)
        self.inner_expert_top_k = (
            None if inner_expert_top_k is None or int(inner_expert_top_k) <= 0 else int(inner_expert_top_k)
        )
        self.current_feature_top_k = self.feature_top_k
        self.current_inner_expert_top_k = self.inner_expert_top_k
        self.alpha_scale = 1.0
        self.last_router_aux: Dict[str, torch.Tensor] = {}

        stage_idx = build_stage_indices(list(stage_all_features), col2idx)
        self.register_buffer("stage_feat_idx", torch.tensor(stage_idx, dtype=torch.long), persistent=False)

        self.pre_ln = nn.LayerNorm(d_model)
        self.hidden_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_router_hidden),
            nn.GELU(),
            nn.Linear(d_router_hidden, d_router_hidden),
        )
        self.feature_scalar_encoder = nn.Sequential(
            nn.Linear(1, d_feat_emb),
            nn.GELU(),
            nn.Linear(d_feat_emb, d_feat_emb),
        )
        self.feature_id_embedding = nn.Embedding(self.n_groups, d_feat_emb)
        self.feature_router_encoder = nn.Sequential(
            nn.LayerNorm(d_feat_emb),
            nn.Linear(d_feat_emb, d_router_hidden),
            nn.GELU(),
            nn.Linear(d_router_hidden, d_router_hidden),
        )
        self.outer_router_head = nn.Sequential(
            nn.Linear(4 * d_router_hidden, d_router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_router_hidden, 1),
        )
        self.inner_routers = nn.ModuleList(
            [
                Router(
                    d_in=4 * d_router_hidden,
                    n_experts=self.expert_scale,
                    d_hidden=d_router_hidden,
                    top_k=self.inner_expert_top_k,
                    dropout=dropout,
                )
                for _ in range(self.n_groups)
            ]
        )
        self.feature_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        HiddenExpertMLP(
                            d_model=d_model,
                            d_hidden=d_expert_hidden,
                            dropout=dropout,
                            d_feat_emb=d_feat_emb,
                            use_hidden=self.expert_use_hidden,
                            use_feature=self.expert_use_feature,
                        )
                        for _ in range(self.expert_scale)
                    ]
                )
                for _ in range(self.n_groups)
            ]
        )
        self.resid_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def set_schedule_state(
        self,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        inner_expert_top_k: Optional[int] = None,
    ) -> None:
        if alpha_scale is not None:
            self.alpha_scale = float(alpha_scale)
        if router_temperature is not None:
            self.current_router_temperature = max(float(router_temperature), 1e-6)
        if inner_expert_top_k is not None:
            self.current_inner_expert_top_k = (
                None if int(inner_expert_top_k) <= 0 else int(inner_expert_top_k)
            )

    @staticmethod
    def _compose_interaction(hidden_ctx: torch.Tensor, feat_ctx: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                feat_ctx,
                hidden_ctx,
                feat_ctx * hidden_ctx,
                (feat_ctx - hidden_ctx).abs(),
            ],
            dim=-1,
        )

    def _build_stage_feature_embeddings(
        self,
        stage_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_emb = self.feature_scalar_encoder(stage_feat.unsqueeze(-1))
        feature_ids = torch.arange(self.n_groups, device=stage_feat.device).view(1, 1, self.n_groups)
        feat_emb = feat_emb + self.feature_id_embedding(feature_ids)
        router_feat = self.feature_router_encoder(feat_emb)
        return feat_emb, router_feat

    def _build_outer_logits(
        self,
        hidden_ctx: torch.Tensor,
        feature_ctx: torch.Tensor,
    ) -> torch.Tensor:
        hidden_outer = hidden_ctx.unsqueeze(-2).expand(-1, -1, self.n_groups, -1)
        if not self.outer_router_use_hidden:
            hidden_outer = torch.zeros_like(hidden_outer)
        if not self.outer_router_use_feature:
            feature_ctx = torch.zeros_like(feature_ctx)
        outer_in = self._compose_interaction(hidden_outer, feature_ctx)
        logits = self.outer_router_head(outer_in).squeeze(-1)
        return logits / self.current_router_temperature

    def _build_inner_router_input(
        self,
        hidden_ctx: torch.Tensor,
        feature_ctx: torch.Tensor,
    ) -> torch.Tensor:
        if not self.inner_router_use_hidden:
            hidden_ctx = torch.zeros_like(hidden_ctx)
        if not self.inner_router_use_feature:
            feature_ctx = torch.zeros_like(feature_ctx)
        return self._compose_interaction(hidden_ctx, feature_ctx)

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del item_seq_len

        bsz, tlen, _ = hidden.shape
        h_norm = self.pre_ln(hidden)
        hidden_ctx = self.hidden_encoder(h_norm)

        stage_feat = feat.index_select(-1, self.stage_feat_idx)
        feature_emb, feature_ctx = self._build_stage_feature_embeddings(stage_feat)

        feature_logits = self._build_outer_logits(hidden_ctx, feature_ctx)
        feature_weights = _topk_softmax(feature_logits, self.current_feature_top_k)

        intra_weights_list: List[torch.Tensor] = []
        intra_logits_list: List[torch.Tensor] = []
        feature_outputs: List[torch.Tensor] = []

        for feature_idx in range(self.n_groups):
            local_feature_ctx = feature_ctx[:, :, feature_idx, :]
            inner_in = self._build_inner_router_input(hidden_ctx, local_feature_ctx)

            if self.expert_scale <= 1:
                intra_logits = torch.zeros(
                    (bsz, tlen, 1),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                intra_weights = torch.ones_like(intra_logits)
            else:
                intra_weights, intra_logits = self.inner_routers[feature_idx](
                    inner_in,
                    temperature=self.current_router_temperature,
                    top_k=self.current_inner_expert_top_k,
                )

            expert_outputs = torch.stack(
                [
                    expert(
                        h_norm,
                        feat_emb=feature_emb[:, :, feature_idx, :] if self.expert_use_feature else None,
                    )
                    for expert in self.feature_experts[feature_idx]
                ],
                dim=-2,
            )
            feature_output = (intra_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)

            intra_weights_list.append(intra_weights)
            intra_logits_list.append(intra_logits)
            feature_outputs.append(feature_output)

        intra_weights = torch.stack(intra_weights_list, dim=-2)
        intra_logits = torch.stack(intra_logits_list, dim=-2)
        stacked_feature_outputs = torch.stack(feature_outputs, dim=-2)

        gate_weights = (feature_weights.unsqueeze(-1) * intra_weights).reshape(bsz, tlen, self.n_experts)
        gate_logits = (feature_logits.unsqueeze(-1) + intra_logits).reshape(bsz, tlen, self.n_experts)
        stage_delta = (feature_weights.unsqueeze(-1) * stacked_feature_outputs).sum(dim=-2)
        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_delta)

        self.last_router_aux = {
            "feature_weights": feature_weights,
            "feature_logits": feature_logits,
            "group_weights": feature_weights,
            "group_logits": feature_logits,
            "group_logits_raw": feature_logits,
            "intra_feature_weights": intra_weights,
            "intra_feature_logits": intra_logits,
            "intra_group_weights": intra_weights,
            "intra_group_logits": intra_logits,
            "intra_group_logits_raw": intra_logits,
        }
        return next_hidden, gate_weights, gate_logits, feature_weights, feature_logits, stage_delta


class HierarchicalMoEIndividual(nn.Module):
    """Stage wrapper for feature-individual MoE blocks."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        d_feat_emb: int = 16,
        d_expert_hidden: int = 160,
        d_router_hidden: int = 64,
        expert_scale: int = 4,
        feature_top_k: Optional[int] = 4,
        inner_expert_top_k: Optional[int] = None,
        dropout: float = 0.1,
        use_macro: bool = True,
        use_mid: bool = True,
        use_micro: bool = True,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = False,
        outer_router_use_hidden: bool = True,
        outer_router_use_feature: bool = True,
        inner_router_use_hidden: bool = True,
        inner_router_use_feature: bool = True,
        macro_router_temperature: float = 1.0,
        mid_router_temperature: float = 1.0,
        micro_router_temperature: float = 1.0,
    ):
        super().__init__()

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.active_stages: List[str] = []
        self.expert_names: Dict[str, List[str]] = {}
        self.group_names: Dict[str, List[str]] = {}
        self.stage_default_temperatures = {
            "macro": float(macro_router_temperature),
            "mid": float(mid_router_temperature),
            "micro": float(micro_router_temperature),
        }

        for stage_name, _ in STAGES:
            if stage_name == "macro" and not use_macro:
                continue
            if stage_name == "mid" and not use_mid:
                continue
            if stage_name == "micro" and not use_micro:
                continue

            stage_module = FeatureIndividualStageMoE(
                stage_name=stage_name,
                feature_names=STAGE_ALL_FEATURES[stage_name],
                stage_all_features=STAGE_ALL_FEATURES[stage_name],
                col2idx=col2idx,
                d_model=d_model,
                d_feat_emb=d_feat_emb,
                d_expert_hidden=d_expert_hidden,
                d_router_hidden=d_router_hidden,
                expert_scale=expert_scale,
                feature_top_k=feature_top_k,
                inner_expert_top_k=inner_expert_top_k,
                dropout=dropout,
                expert_use_hidden=expert_use_hidden,
                expert_use_feature=expert_use_feature,
                outer_router_use_hidden=outer_router_use_hidden,
                outer_router_use_feature=outer_router_use_feature,
                inner_router_use_hidden=inner_router_use_hidden,
                inner_router_use_feature=inner_router_use_feature,
                router_temperature=self.stage_default_temperatures[stage_name],
            )
            setattr(self, f"{stage_name}_stage", stage_module)
            self.active_stages.append(stage_name)
            self.expert_names[stage_name] = list(stage_module.expert_names)
            self.group_names[stage_name] = list(stage_module.group_names)

        if self.active_stages:
            first_stage = getattr(self, f"{self.active_stages[0]}_stage")
            self.stage_n_experts = int(first_stage.n_experts)
        else:
            self.stage_n_experts = 0

    def has_stage(self, stage_name: str) -> bool:
        return stage_name in self.active_stages and hasattr(self, f"{stage_name}_stage")

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        inner_expert_top_k: Optional[int] = None,
    ) -> None:
        stage_temperatures = stage_temperatures or {}
        for stage_name in self.active_stages:
            stage_module = getattr(self, f"{stage_name}_stage")
            stage_module.set_schedule_state(
                alpha_scale=alpha_scale,
                router_temperature=stage_temperatures.get(stage_name, self.stage_default_temperatures[stage_name]),
                inner_expert_top_k=inner_expert_top_k,
            )

    def forward_stage(
        self,
        stage_name: str,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_stage(stage_name):
            raise ValueError(f"stage '{stage_name}' is not active in this HierarchicalMoEIndividual instance")
        stage_module = getattr(self, f"{stage_name}_stage")
        return stage_module(hidden, feat, item_seq_len=item_seq_len)

    def get_stage_router_aux(self, stage_name: str) -> Dict[str, torch.Tensor]:
        if not self.has_stage(stage_name):
            return {}
        stage_module = getattr(self, f"{stage_name}_stage")
        return dict(getattr(stage_module, "last_router_aux", {}) or {})

