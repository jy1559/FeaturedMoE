"""Hierarchical Group Routing v4 blocks for FeaturedMoE_HGRv4.

Outer routing restores old HGR-style feature-aware group routing.
Inner routing keeps learned hidden+group-feature interaction and uses a
group-local rule-soft teacher built from group feature statistics.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGES, STAGE_ALL_FEATURES, build_column_to_index
from ..FeaturedMoE.routers import Router
from ..FeaturedMoE_HGR.hgr_moe_stages import ScalarRouter
from ..FeaturedMoE_HGRv3.hgr_v3_moe_stages import (
    HiddenExpertMLP,
    HierarchicalGroupStageMoEv3,
    HierarchicalMoEHGRv3,
    _topk_softmax,
    _to_ratio,
)


class GroupStatSoftRuleTeacher(nn.Module):
    """Rule-soft teacher from group-local feature statistics.

    The teacher is intentionally non-learned:
    - convert group features into ratio space
    - compute a compact stat vector
    - derive expert logits from fixed rule formulas
    """

    def __init__(self, expert_scale: int, *, sharpness: float = 16.0):
        super().__init__()
        self.expert_scale = max(int(expert_scale), 1)
        self.sharpness = float(sharpness)

        if self.expert_scale == 4:
            signature = torch.tensor(
                [
                    [0.15, 0.05, 0.25, 0.05, 0.20, 0.10],  # low-flat
                    [0.45, 0.08, 0.55, 0.35, 0.20, 0.10],  # mid-flat
                    [0.75, 0.08, 0.85, 0.65, 0.20, 0.10],  # high-flat
                    [0.55, 0.28, 0.95, 0.15, 0.80, 0.40],  # peaky / contrast
                ],
                dtype=torch.float32,
            )
        else:
            centers = torch.linspace(0.0, 1.0, steps=self.expert_scale, dtype=torch.float32)
            signature = torch.stack(
                [
                    centers,
                    torch.full_like(centers, 0.10),
                    (centers + 0.15).clamp(max=1.0),
                    (centers - 0.15).clamp(min=0.0),
                    torch.full_like(centers, 0.20),
                    torch.full_like(centers, 0.10),
                ],
                dim=-1,
            )
        self.register_buffer("rule_signature", signature, persistent=False)

    @staticmethod
    def _pool_session_stats(
        stats: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        session_pooling: str,
    ) -> torch.Tensor:
        session_pooling = str(session_pooling).lower().strip()
        if session_pooling == "last":
            if item_seq_len is None:
                idx = torch.full(
                    (stats.size(0),),
                    fill_value=max(stats.size(1) - 1, 0),
                    dtype=torch.long,
                    device=stats.device,
                )
            else:
                idx = item_seq_len.to(device=stats.device).long().clamp(min=1, max=stats.size(1)) - 1
            return stats[torch.arange(stats.size(0), device=stats.device), idx]

        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (stats * weights).sum(dim=1) / denom

    def forward(
        self,
        *,
        group_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        router_mode: str,
        session_pooling: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = _to_ratio(group_feat)
        mean = ratio.mean(dim=-1, keepdim=True)
        std = ratio.std(dim=-1, unbiased=False, keepdim=True)
        vmax = ratio.max(dim=-1, keepdim=True).values
        vmin = ratio.min(dim=-1, keepdim=True).values
        vrange = vmax - vmin
        peak = vmax - mean
        stats = torch.cat([mean, std, vmax, vmin, vrange, peak], dim=-1)

        if str(router_mode).lower().strip() == "session":
            pooled = self._pool_session_stats(stats, valid_mask, item_seq_len, session_pooling)
            stats = pooled.unsqueeze(1).expand(-1, group_feat.size(1), -1)

        mean = stats[..., 0:1]
        std = stats[..., 1:2]
        vmax = stats[..., 2:3]
        vrange = stats[..., 4:5]
        peak = stats[..., 5:6]

        if self.expert_scale == 1:
            logits = torch.zeros_like(mean)
            return logits, stats, self.rule_signature

        if self.expert_scale == 4:
            sharp = self.sharpness
            low = -sharp * (mean - 0.15).pow(2) - 1.6 * std - 0.8 * vrange
            mid = -sharp * (mean - 0.45).pow(2) - 1.0 * std - 0.3 * vrange
            high = -sharp * (mean - 0.75).pow(2) - 1.0 * std - 0.25 * (1.0 - vmax)
            peaky = 2.0 * vrange + 1.5 * peak + 0.5 * std - 4.0 * (mean - 0.55).pow(2)
            logits = torch.cat([low, mid, high, peaky], dim=-1)
            return logits, stats, self.rule_signature

        centers = torch.linspace(0.0, 1.0, steps=self.expert_scale, device=mean.device, dtype=mean.dtype)
        logits = -self.sharpness * (mean - centers.view(1, 1, self.expert_scale)).pow(2)
        return logits, stats, self.rule_signature


class HierarchicalGroupStageMoEv4(HierarchicalGroupStageMoEv3):
    """One stage with restored feature-aware outer router and stat teacher."""

    def __init__(
        self,
        *args,
        group_router_mode: str = "hybrid",
        outer_router_design: str = "legacy_concat",
        inner_router_design: str = "legacy_concat",
        inner_rule_teacher_kind: str = "group_stat_soft",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        mode = str(group_router_mode).lower().strip()
        if mode not in {"stage_wide", "per_group", "hybrid"}:
            raise ValueError(
                "group_router_mode must be one of ['stage_wide','per_group','hybrid'], "
                f"got {group_router_mode}"
            )
        self.group_router_mode = mode
        self.outer_router_design = str(outer_router_design).lower().strip()
        self.inner_router_design = str(inner_router_design).lower().strip()
        valid_designs = {"legacy_concat", "group_factorized_interaction"}
        if self.outer_router_design not in valid_designs:
            raise ValueError(
                "outer_router_design must be one of ['legacy_concat','group_factorized_interaction'], "
                f"got {outer_router_design}"
            )
        if self.inner_router_design not in valid_designs:
            raise ValueError(
                "inner_router_design must be one of ['legacy_concat','group_factorized_interaction'], "
                f"got {inner_router_design}"
            )
        self.inner_rule_teacher_kind = str(inner_rule_teacher_kind).lower().strip()
        if self.inner_rule_teacher_kind != "group_stat_soft":
            raise ValueError(
                "inner_rule_teacher_kind must be 'group_stat_soft' for HGRv4, "
                f"got {inner_rule_teacher_kind}"
            )

        d_model = int(self.pre_ln.normalized_shape[0])
        d_feat_emb = int(self.group_feat_proj[0].out_features)
        d_router_hidden = int(self.hidden_encoder[-1].out_features)
        router_dropout = float(getattr(self.resid_drop, "p", 0.1))
        self.d_model = d_model
        self.d_feat_emb = d_feat_emb
        self.d_router_hidden = d_router_hidden

        if self.outer_router_design == "group_factorized_interaction":
            self.stage_feature_encoder = nn.Sequential(
                nn.LayerNorm(self._n_stage_features),
                nn.Linear(self._n_stage_features, d_router_hidden),
                nn.GELU(),
                nn.Linear(d_router_hidden, d_router_hidden),
            )
            stage_router_in_dim = 4 * d_router_hidden
            if self.group_router_mode in {"stage_wide", "hybrid"}:
                self.stage_wide_router = Router(
                    d_in=stage_router_in_dim,
                    n_experts=self.n_groups,
                    d_hidden=d_router_hidden,
                    top_k=self.group_top_k,
                    dropout=router_dropout,
                )
            else:
                self.stage_wide_router = None
            if self.group_router_mode in {"per_group", "hybrid"}:
                self.per_group_scorer = ScalarRouter(
                    d_in=stage_router_in_dim,
                    d_hidden=d_router_hidden,
                    dropout=router_dropout,
                )
            else:
                self.per_group_scorer = None
            self.stage_feat_proj = None
            self.per_group_scorers = None
            self.stage_wide_router_legacy = None
        else:
            self.stage_feature_encoder = None
            self.stage_feat_proj = nn.Linear(self._n_stage_features, d_feat_emb)
            outer_in_dim = 0
            if self.outer_router_use_hidden:
                outer_in_dim += d_model
            if self.outer_router_use_feature:
                outer_in_dim += d_feat_emb
            if self.group_router_mode in {"stage_wide", "hybrid"}:
                self.stage_wide_router_legacy = Router(
                    d_in=outer_in_dim,
                    n_experts=self.n_groups,
                    d_hidden=d_router_hidden,
                    top_k=self.group_top_k,
                    dropout=router_dropout,
                )
            else:
                self.stage_wide_router_legacy = None
            if self.group_router_mode in {"per_group", "hybrid"}:
                self.per_group_scorers = nn.ModuleList(
                    [
                        ScalarRouter(
                            d_in=outer_in_dim,
                            d_hidden=d_router_hidden,
                            dropout=router_dropout,
                        )
                        for _ in range(self.n_groups)
                    ]
                )
            else:
                self.per_group_scorers = None
            self.stage_wide_router = None
            self.per_group_scorer = None

        if self.inner_router_design == "legacy_concat":
            inner_in_dim = 0
            if self.inner_router_use_hidden:
                inner_in_dim += d_model
            if self.inner_router_use_feature:
                inner_in_dim += d_feat_emb
            self.intra_routers_legacy = nn.ModuleList(
                [
                    Router(
                        d_in=inner_in_dim,
                        n_experts=self.expert_scale,
                        d_hidden=d_router_hidden,
                        top_k=self.expert_top_k,
                        dropout=router_dropout,
                    )
                    for _ in range(self.n_groups)
                ]
            )
        else:
            self.intra_routers_legacy = None

        if self.router_mode == "session" and self.session_pooling == "query" and self.outer_router_design == "legacy_concat":
            if self.outer_router_use_hidden:
                self.session_query_hidden_legacy = nn.Parameter(
                    torch.randn(d_model) * (1.0 / math.sqrt(float(d_model)))
                )
            else:
                self.register_parameter("session_query_hidden_legacy", None)
            if self.outer_router_use_feature or self.inner_router_design == "legacy_concat":
                self.session_query_feature_legacy = nn.Parameter(
                    torch.randn(d_feat_emb) * (1.0 / math.sqrt(float(d_feat_emb)))
                )
            else:
                self.register_parameter("session_query_feature_legacy", None)
        else:
            self.register_parameter("session_query_hidden_legacy", None)
            self.register_parameter("session_query_feature_legacy", None)

        self.inner_rule_teacher = GroupStatSoftRuleTeacher(
            expert_scale=self.expert_scale,
            sharpness=float(kwargs.get("inner_rule_bin_sharpness", 16.0)),
        )

    def _maybe_pool_hidden_legacy(
        self,
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return hidden
        return self._pool_sequence(hidden, valid_mask, item_seq_len, query=self.session_query_hidden_legacy)

    def _maybe_pool_feature_legacy(
        self,
        feat_enc: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return feat_enc
        return self._pool_sequence(feat_enc, valid_mask, item_seq_len, query=self.session_query_feature_legacy)

    @staticmethod
    def _build_router_input(hidden: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden, feat_emb], dim=-1)

    def _encode_stage_router_feature(
        self,
        stage_feat: torch.Tensor,
        feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.outer_router_design == "legacy_concat":
            feat_emb = self.stage_feat_proj(stage_feat)
            feat_emb = self._apply_reliability(feat_emb, feat)
            feat_emb = self.router_feat_drop(feat_emb)
            return self._maybe_pool_feature_legacy(feat_emb, valid_mask, item_seq_len)

        stage_feat = self._apply_reliability_to_raw(stage_feat, feat)
        feat_enc = self.stage_feature_encoder(stage_feat)
        feat_enc = self.router_feat_drop(feat_enc)
        return self._maybe_pool_feature(feat_enc, valid_mask, item_seq_len)

    def _encode_group_router_feature_v4(
        self,
        *,
        group_idx: int,
        group_feat: torch.Tensor,
        feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        design: str,
    ) -> torch.Tensor:
        if design == "legacy_concat":
            feat_emb = self.group_feat_proj[group_idx](group_feat)
            feat_emb = self._apply_reliability(feat_emb, feat)
            feat_emb = self.router_feat_drop(feat_emb)
            return self._maybe_pool_feature_legacy(feat_emb, valid_mask, item_seq_len)
        return self._encode_group_feature(
            group_idx=group_idx,
            group_feat=group_feat,
            feat=feat,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
        )

    def _compute_group_logits(
        self,
        hidden_ctx: torch.Tensor,
        stage_feat_ctx: torch.Tensor,
        group_feat_ctxs: List[torch.Tensor],
    ) -> torch.Tensor:
        logits_parts: List[torch.Tensor] = []

        if self.group_router_mode in {"stage_wide", "hybrid"}:
            stage_hidden = hidden_ctx
            stage_feat = stage_feat_ctx
            if self.outer_router_design == "group_factorized_interaction":
                stage_hidden = stage_hidden if self.outer_router_use_hidden else torch.zeros_like(stage_feat)
                stage_feat = stage_feat if self.outer_router_use_feature else torch.zeros_like(stage_hidden)
                stage_router_in = self._compose_interaction(stage_hidden, stage_feat)
                logits_parts.append(self.stage_wide_router.net(stage_router_in) / self.current_router_temperature)
            else:
                inputs: List[torch.Tensor] = []
                if self.outer_router_use_hidden:
                    inputs.append(stage_hidden)
                if self.outer_router_use_feature:
                    inputs.append(stage_feat)
                stage_router_in = inputs[0] if len(inputs) == 1 else self._build_router_input(inputs[0], inputs[1])
                logits_parts.append(self.stage_wide_router_legacy.net(stage_router_in) / self.current_router_temperature)

        if self.group_router_mode in {"per_group", "hybrid"}:
            per_group_logits = []
            for group_idx, group_feat_ctx in enumerate(group_feat_ctxs):
                if self.outer_router_design == "group_factorized_interaction":
                    scorer_hidden = hidden_ctx if self.outer_router_use_hidden else torch.zeros_like(group_feat_ctx)
                    scorer_feat = group_feat_ctx if self.outer_router_use_feature else torch.zeros_like(scorer_hidden)
                    scorer_in = self._compose_interaction(scorer_hidden, scorer_feat)
                    per_group_logits.append(
                        self.per_group_scorer(
                            scorer_in,
                            temperature=self.current_router_temperature,
                        )
                    )
                else:
                    inputs = []
                    if self.outer_router_use_hidden:
                        inputs.append(hidden_ctx)
                    if self.outer_router_use_feature:
                        inputs.append(group_feat_ctx)
                    scorer_in = inputs[0] if len(inputs) == 1 else self._build_router_input(inputs[0], inputs[1])
                    per_group_logits.append(
                        self.per_group_scorers[group_idx](
                            scorer_in,
                            temperature=self.current_router_temperature,
                        )
                    )
            logits_parts.append(torch.stack(per_group_logits, dim=-1))

        if len(logits_parts) == 1:
            return logits_parts[0]
        return logits_parts[0] + logits_parts[1]

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, tlen, _ = hidden.shape
        h_norm = self.pre_ln(hidden)
        valid_mask = self._build_valid_mask(bsz, tlen, item_seq_len, hidden.device)

        hidden_enc = self.hidden_encoder(h_norm)
        hidden_ctx_factorized = self._maybe_pool_hidden(hidden_enc, valid_mask, item_seq_len)
        hidden_ctx_legacy = self._maybe_pool_hidden_legacy(h_norm, valid_mask, item_seq_len)

        stage_feat = feat.index_select(-1, self.stage_feat_idx)
        stage_feat_ctx = self._encode_stage_router_feature(stage_feat, feat, valid_mask, item_seq_len)
        group_feat_ctxs_outer = [
            self._encode_group_router_feature_v4(
                group_idx=group_idx,
                group_feat=self._group_feature_tensor(feat, group_idx),
                feat=feat,
                valid_mask=valid_mask,
                item_seq_len=item_seq_len,
                design=self.outer_router_design,
            )
            for group_idx in range(self.n_groups)
        ]

        outer_hidden_ctx = hidden_ctx_factorized if self.outer_router_design == "group_factorized_interaction" else hidden_ctx_legacy
        group_logits_ctx = self._compute_group_logits(outer_hidden_ctx, stage_feat_ctx, group_feat_ctxs_outer)
        group_weights_ctx = _topk_softmax(group_logits_ctx, self.current_group_top_k)
        if self.router_mode == "session":
            group_weights = group_weights_ctx.unsqueeze(1).expand(-1, tlen, -1)
            group_logits = group_logits_ctx.unsqueeze(1).expand(-1, tlen, -1)
        else:
            group_weights = group_weights_ctx
            group_logits = group_logits_ctx

        intra_weights_list: List[torch.Tensor] = []
        intra_logits_list: List[torch.Tensor] = []
        intra_logits_raw_list: List[torch.Tensor] = []
        teacher_logits_list: List[torch.Tensor] = []
        teacher_score_list: List[torch.Tensor] = []
        group_outputs: List[torch.Tensor] = []

        for group_idx in range(self.n_groups):
            group_feat = self._group_feature_tensor(feat, group_idx)
            group_feat_ctx = self._encode_group_router_feature_v4(
                group_idx=group_idx,
                group_feat=group_feat,
                feat=feat,
                valid_mask=valid_mask,
                item_seq_len=item_seq_len,
                design=self.inner_router_design,
            )
            if self.inner_router_design == "group_factorized_interaction":
                intra_in = self._build_inner_router_input(hidden_ctx_factorized, group_feat_ctx)
                intra_router = self.intra_routers[group_idx]
            else:
                inputs = []
                if self.inner_router_use_hidden:
                    inputs.append(hidden_ctx_legacy)
                if self.inner_router_use_feature:
                    inputs.append(group_feat_ctx)
                intra_in = inputs[0] if len(inputs) == 1 else self._build_router_input(inputs[0], inputs[1])
                intra_router = self.intra_routers_legacy[group_idx]

            if self.expert_scale <= 1:
                intra_raw = torch.zeros(tuple(intra_in.shape[:-1]) + (1,), device=hidden.device, dtype=hidden.dtype)
                intra_logits = intra_raw
                intra_w = torch.ones_like(intra_raw)
                teacher_logits = torch.zeros_like(intra_raw)
                teacher_score = torch.zeros(*intra_raw.shape[:-1], 6, device=hidden.device, dtype=hidden.dtype)
            else:
                _, intra_raw = intra_router(
                    intra_in,
                    temperature=self.current_router_temperature,
                    top_k=None,
                )
                if self.router_mode == "session":
                    intra_raw = intra_raw.unsqueeze(1).expand(-1, tlen, -1)

                if self.inner_rule_enable:
                    teacher_logits, teacher_score, _ = self.inner_rule_teacher(
                        group_feat=self._apply_reliability_to_raw(group_feat, feat),
                        valid_mask=valid_mask,
                        item_seq_len=item_seq_len,
                        router_mode=self.router_mode,
                        session_pooling=self.session_pooling,
                    )
                else:
                    teacher_logits = torch.zeros_like(intra_raw)
                    teacher_score = torch.zeros(
                        (bsz, tlen, 6),
                        device=intra_raw.device,
                        dtype=intra_raw.dtype,
                    )

                intra_logits = intra_raw
                if self.inner_rule_mode in {"fused_bias", "distill_and_fused_bias"}:
                    intra_logits = intra_logits + float(self.inner_rule_bias_scale) * teacher_logits
                intra_w = _topk_softmax(intra_logits, self.current_expert_top_k)

            intra_weights_list.append(intra_w)
            intra_logits_list.append(intra_logits)
            intra_logits_raw_list.append(intra_raw)
            teacher_logits_list.append(teacher_logits)
            teacher_score_list.append(teacher_score)

            expert_feat_emb = self.group_feat_proj[group_idx](group_feat)
            expert_feat_emb = self._apply_reliability(expert_feat_emb, feat)
            start = group_idx * self.expert_scale
            end = start + self.expert_scale
            expert_outputs = torch.stack(
                [
                    expert(
                        h_norm,
                        feat_emb=expert_feat_emb if self.expert_use_feature else None,
                    )
                    for expert in self.experts[start:end]
                ],
                dim=-2,
            )
            group_output = (intra_w.unsqueeze(-1) * expert_outputs).sum(dim=-2)
            group_outputs.append(group_output)

        intra_weights = torch.stack(intra_weights_list, dim=-2)
        intra_logits = torch.stack(intra_logits_list, dim=-2)
        intra_logits_raw = torch.stack(intra_logits_raw_list, dim=-2)
        teacher_intra_logits = torch.stack(teacher_logits_list, dim=-2)
        inner_rule_score = torch.stack(teacher_score_list, dim=-2)

        final_weights = group_weights.unsqueeze(-1) * intra_weights
        gate_weights = final_weights.reshape(bsz, tlen, self.n_experts)
        final_logits = group_logits.unsqueeze(-1) + intra_logits
        gate_logits = final_logits.reshape(bsz, tlen, self.n_experts)

        stacked_group_outputs = torch.stack(group_outputs, dim=-2)
        stage_delta = (group_weights.unsqueeze(-1) * stacked_group_outputs).sum(dim=-2)
        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_delta)

        self.last_router_aux = {
            "group_weights": group_weights,
            "group_logits": group_logits,
            "group_logits_raw": group_logits,
            "intra_group_weights": intra_weights,
            "intra_group_logits": intra_logits,
            "intra_group_logits_raw": intra_logits_raw,
            "teacher_intra_group_logits": teacher_intra_logits,
            "inner_rule_score": inner_rule_score,
            "inner_rule_bin_centers": self.inner_rule_teacher.rule_signature,
        }
        return next_hidden, gate_weights, gate_logits, group_weights, group_logits, stage_delta


class HierarchicalMoEHGRv4(HierarchicalMoEHGRv3):
    """HGRv4 wrapper: old HGR outer + stat-soft inner teacher."""

    def __init__(
        self,
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        top_k: Optional[int] = None,
        group_top_k: Optional[int] = None,
        dropout: float = 0.1,
        use_macro: bool = True,
        use_mid: bool = True,
        use_micro: bool = True,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = False,
        stage_merge_mode: str = "serial",
        macro_routing_scope: str = "session",
        macro_session_pooling: str = "query",
        mid_router_temperature: float = 1.3,
        micro_router_temperature: float = 1.3,
        mid_router_feature_dropout: float = 0.1,
        micro_router_feature_dropout: float = 0.1,
        use_valid_ratio_gating: bool = True,
        parallel_stage_gate_top_k: Optional[int] = None,
        parallel_stage_gate_temperature: float = 1.0,
        outer_router_use_hidden: bool = True,
        outer_router_use_feature: bool = True,
        inner_router_use_hidden: bool = True,
        inner_router_use_feature: bool = True,
        outer_router_design: str = "legacy_concat",
        inner_router_design: str = "legacy_concat",
        inner_rule_enable: bool = True,
        inner_rule_mode: str = "distill",
        inner_rule_bias_scale: float = 0.0,
        inner_rule_bin_sharpness: float = 16.0,
        inner_rule_group_feature_pool: str = "mean_ratio",
        inner_rule_apply_stages: Optional[Sequence[str]] = None,
        group_router_mode: str = "hybrid",
        inner_rule_teacher_kind: str = "group_stat_soft",
    ):
        super().__init__(
            d_model=d_model,
            d_feat_emb=d_feat_emb,
            d_expert_hidden=d_expert_hidden,
            d_router_hidden=d_router_hidden,
            expert_scale=expert_scale,
            top_k=top_k,
            group_top_k=group_top_k,
            dropout=dropout,
            use_macro=use_macro,
            use_mid=use_mid,
            use_micro=use_micro,
            expert_use_hidden=expert_use_hidden,
            expert_use_feature=expert_use_feature,
            stage_merge_mode=stage_merge_mode,
            macro_routing_scope=macro_routing_scope,
            macro_session_pooling=macro_session_pooling,
            mid_router_temperature=mid_router_temperature,
            micro_router_temperature=micro_router_temperature,
            mid_router_feature_dropout=mid_router_feature_dropout,
            micro_router_feature_dropout=micro_router_feature_dropout,
            use_valid_ratio_gating=use_valid_ratio_gating,
            parallel_stage_gate_top_k=parallel_stage_gate_top_k,
            parallel_stage_gate_temperature=parallel_stage_gate_temperature,
            outer_router_use_hidden=outer_router_use_hidden,
            outer_router_use_feature=outer_router_use_feature,
            inner_router_use_hidden=inner_router_use_hidden,
            inner_router_use_feature=inner_router_use_feature,
            inner_rule_enable=inner_rule_enable,
            inner_rule_mode=inner_rule_mode,
            inner_rule_bias_scale=inner_rule_bias_scale,
            inner_rule_bin_sharpness=inner_rule_bin_sharpness,
            inner_rule_group_feature_pool=inner_rule_group_feature_pool,
            inner_rule_apply_stages=inner_rule_apply_stages,
        )
        self.group_router_mode = str(group_router_mode).lower().strip()
        self.outer_router_design = str(outer_router_design).lower().strip()
        self.inner_router_design = str(inner_router_design).lower().strip()
        self.inner_rule_teacher_kind = str(inner_rule_teacher_kind).lower().strip()

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.active_stages = []
        self.expert_names = {}
        self.group_names = {}
        apply_stage_set = {
            str(stage).strip().lower() for stage in (inner_rule_apply_stages or ["macro", "mid", "micro"])
        }

        for stage_name, expert_dict in STAGES:
            if stage_name == "macro" and not use_macro:
                continue
            if stage_name == "mid" and not use_mid:
                continue
            if stage_name == "micro" and not use_micro:
                continue

            router_feature_dropout = 0.0
            reliability_feature_name = None
            if stage_name == "mid":
                router_feature_dropout = float(mid_router_feature_dropout)
                if use_valid_ratio_gating:
                    reliability_feature_name = "mid_valid_r"
            elif stage_name == "micro":
                router_feature_dropout = float(micro_router_feature_dropout)
                if use_valid_ratio_gating:
                    reliability_feature_name = "mic_valid_r"

            router_mode = "token"
            session_pooling = "query"
            if stage_name == "macro":
                router_mode = str(macro_routing_scope).lower().strip()
                session_pooling = str(macro_session_pooling).lower().strip()

            stage_module = HierarchicalGroupStageMoEv4(
                stage_name=stage_name,
                group_names=list(expert_dict.keys()),
                group_feature_lists=list(expert_dict.values()),
                stage_all_features=STAGE_ALL_FEATURES[stage_name],
                col2idx=col2idx,
                d_model=d_model,
                d_feat_emb=d_feat_emb,
                d_expert_hidden=d_expert_hidden,
                d_router_hidden=d_router_hidden,
                expert_scale=expert_scale,
                expert_top_k=top_k,
                group_top_k=group_top_k,
                dropout=dropout,
                expert_use_hidden=expert_use_hidden,
                expert_use_feature=expert_use_feature,
                router_design="group_factorized_interaction",
                router_mode=router_mode,
                session_pooling=session_pooling,
                router_temperature=self.stage_default_temperatures[stage_name],
                router_feature_dropout=router_feature_dropout,
                reliability_feature_name=reliability_feature_name,
                outer_router_use_hidden=outer_router_use_hidden,
                outer_router_use_feature=outer_router_use_feature,
                inner_router_use_hidden=inner_router_use_hidden,
                inner_router_use_feature=inner_router_use_feature,
                outer_router_design=self.outer_router_design,
                inner_router_design=self.inner_router_design,
                inner_rule_enable=inner_rule_enable,
                inner_rule_mode=inner_rule_mode,
                inner_rule_bias_scale=inner_rule_bias_scale,
                inner_rule_bin_sharpness=inner_rule_bin_sharpness,
                inner_rule_group_feature_pool=inner_rule_group_feature_pool,
                inner_rule_apply=(stage_name in apply_stage_set),
                group_router_mode=self.group_router_mode,
                inner_rule_teacher_kind=self.inner_rule_teacher_kind,
            )
            setattr(self, f"{stage_name}_stage", stage_module)
            self.active_stages.append(stage_name)
            self.expert_names[stage_name] = list(stage_module.expert_names)
            self.group_names[stage_name] = list(stage_module.group_names)

        self.n_active = len(self.active_stages)
        if self.n_active > 0:
            first_stage = getattr(self, f"{self.active_stages[0]}_stage")
            self.stage_n_experts = int(first_stage.n_experts)
        else:
            self.stage_n_experts = 0
