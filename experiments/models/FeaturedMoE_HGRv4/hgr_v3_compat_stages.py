"""Hierarchical Group Routing v3 blocks for FeaturedMoE_HGRv3."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FeaturedMoE.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    build_column_to_index,
    build_expert_indices,
    build_stage_indices,
)
from ..FeaturedMoE.routers import Router, load_balance_loss


def _scaled_group_expert_names(group_names: List[str], expert_scale: int) -> List[str]:
    if expert_scale <= 1:
        return list(group_names)

    out: List[str] = []
    for gname in group_names:
        for idx in range(expert_scale):
            suffix = chr(ord("a") + idx)
            out.append(f"{gname}_{suffix}")
    return out


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


def _to_ratio(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values

    x = values.float()
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.zeros_like(x)

    xf = x[finite]
    lo = xf.min()
    hi = xf.max()
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        ratio = x.clamp(0.0, 1.0)
        ratio[~finite] = 0.5
        return ratio

    ratio = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    ratio = ratio.clamp(0.0, 1.0)
    ratio[~finite] = 0.5
    return ratio


class HiddenExpertMLP(nn.Module):
    """Expert FFN that can optionally consume group feature embeddings."""

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


class InnerBinRuleTeacher(nn.Module):
    """Group-local bin teacher that maps normalized feature ratios to clone logits."""

    def __init__(
        self,
        expert_scale: int,
        *,
        bin_sharpness: float = 16.0,
        group_feature_pool: str = "mean_ratio",
    ):
        super().__init__()
        self.expert_scale = max(int(expert_scale), 1)
        self.bin_sharpness = float(bin_sharpness)
        self.group_feature_pool = str(group_feature_pool).lower().strip()
        if self.group_feature_pool != "mean_ratio":
            raise ValueError(
                f"inner_rule_group_feature_pool must be 'mean_ratio', got {group_feature_pool}"
            )
        if self.expert_scale == 1:
            centers = torch.tensor([0.5], dtype=torch.float32)
        else:
            centers = torch.linspace(0.0, 1.0, steps=self.expert_scale, dtype=torch.float32)
        self.register_buffer("bin_centers", centers, persistent=False)

    @staticmethod
    def _pool_session_score(
        score: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        session_pooling: str,
    ) -> torch.Tensor:
        session_pooling = str(session_pooling).lower().strip()
        if session_pooling == "last":
            if item_seq_len is None:
                idx = torch.full(
                    (score.size(0),),
                    fill_value=max(score.size(1) - 1, 0),
                    dtype=torch.long,
                    device=score.device,
                )
            else:
                idx = item_seq_len.to(device=score.device).long().clamp(min=1, max=score.size(1)) - 1
            return score[torch.arange(score.size(0), device=score.device), idx]

        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (score * weights).sum(dim=1) / denom

    def forward(
        self,
        *,
        group_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        router_mode: str,
        session_pooling: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio_feat = _to_ratio(group_feat)
        score = ratio_feat.mean(dim=-1, keepdim=True)
        if str(router_mode).lower().strip() == "session":
            pooled = self._pool_session_score(score, valid_mask, item_seq_len, session_pooling)
            score = pooled.unsqueeze(1).expand(-1, group_feat.size(1), -1)
        centers = self.bin_centers.view(1, 1, self.expert_scale)
        logits = -float(self.bin_sharpness) * (score - centers).pow(2)
        return logits, score, self.bin_centers


class HierarchicalGroupStageMoEv3(nn.Module):
    """One stage with hidden-only outer router and inner rule teacher."""

    def __init__(
        self,
        stage_name: str,
        group_names: List[str],
        group_feature_lists: List[List[str]],
        stage_all_features: List[str],
        col2idx: Dict[str, int],
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        expert_top_k: Optional[int] = None,
        group_top_k: Optional[int] = None,
        dropout: float = 0.1,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = False,
        router_design: str = "group_factorized_interaction",
        router_mode: str = "token",
        session_pooling: str = "query",
        router_temperature: float = 1.0,
        router_feature_dropout: float = 0.0,
        reliability_feature_name: Optional[str] = None,
        outer_router_use_hidden: bool = True,
        outer_router_use_feature: bool = False,
        inner_router_use_hidden: bool = True,
        inner_router_use_feature: bool = True,
        inner_rule_enable: bool = True,
        inner_rule_mode: str = "distill",
        inner_rule_bias_scale: float = 1.0,
        inner_rule_bin_sharpness: float = 16.0,
        inner_rule_group_feature_pool: str = "mean_ratio",
        inner_rule_apply: bool = True,
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

        router_design_key = str(router_design).lower().strip()
        if router_design_key != "group_factorized_interaction":
            raise ValueError(
                "FeaturedMoE_HGRv3 currently supports only router_design=group_factorized_interaction"
            )
        router_mode_key = str(router_mode).lower().strip()
        if router_mode_key not in {"token", "session"}:
            raise ValueError(f"router_mode must be one of ['token','session'], got {router_mode}")
        session_pooling_key = str(session_pooling).lower().strip()
        if session_pooling_key not in {"query", "mean", "last"}:
            raise ValueError(
                f"session_pooling must be one of ['query','mean','last'], got {session_pooling}"
            )
        inner_rule_mode_key = str(inner_rule_mode).lower().strip()
        if inner_rule_mode_key not in {"off", "distill", "fused_bias", "distill_and_fused_bias"}:
            raise ValueError(
                "inner_rule_mode must be one of ['off','distill','fused_bias','distill_and_fused_bias'], "
                f"got {inner_rule_mode}"
            )

        self.stage_name = str(stage_name)
        self.group_names = list(group_names)
        self.stage_all_features = list(stage_all_features)
        self.expert_scale = int(expert_scale)
        self.n_groups = len(group_feature_lists)
        self.n_experts = self.n_groups * self.expert_scale
        self.expert_names = _scaled_group_expert_names(self.group_names, self.expert_scale)
        self.router_design = router_design_key
        self.router_mode = router_mode_key
        self.session_pooling = session_pooling_key
        self.outer_router_use_hidden = bool(outer_router_use_hidden)
        self.outer_router_use_feature = bool(outer_router_use_feature)
        self.inner_router_use_hidden = bool(inner_router_use_hidden)
        self.inner_router_use_feature = bool(inner_router_use_feature)
        self.expert_use_hidden = bool(expert_use_hidden)
        self.expert_use_feature = bool(expert_use_feature)
        self.router_temperature = float(router_temperature)
        self.current_router_temperature = float(router_temperature)
        self.group_top_k = None if group_top_k is None or int(group_top_k) <= 0 else int(group_top_k)
        self.expert_top_k = None if expert_top_k is None or int(expert_top_k) <= 0 else int(expert_top_k)
        self.current_group_top_k = self.group_top_k
        self.current_expert_top_k = self.expert_top_k
        self.alpha_scale = 1.0
        self.last_router_aux: Dict[str, torch.Tensor] = {}

        self.inner_rule_enable = (
            bool(inner_rule_enable)
            and inner_rule_mode_key != "off"
            and bool(inner_rule_apply)
            and self.expert_scale > 1
        )
        self.inner_rule_mode = inner_rule_mode_key if self.inner_rule_enable else "off"
        self.inner_rule_bias_scale = float(inner_rule_bias_scale)

        stage_idx = build_stage_indices(stage_all_features, col2idx)
        self.register_buffer("stage_feat_idx", torch.tensor(stage_idx, dtype=torch.long), persistent=False)
        self._n_stage_features = len(stage_idx)

        group_idx_lists = build_expert_indices(
            OrderedDict(zip(group_names, group_feature_lists)),
            col2idx,
        )
        self.group_feature_dims: List[int] = []
        for idx, feat_idx in enumerate(group_idx_lists):
            self.register_buffer(
                f"group_feat_idx_{idx}",
                torch.tensor(feat_idx, dtype=torch.long),
                persistent=False,
            )
            self.group_feature_dims.append(len(feat_idx))

        if reliability_feature_name is not None:
            if reliability_feature_name not in col2idx:
                raise ValueError(
                    f"reliability_feature_name '{reliability_feature_name}' not found in feature columns"
                )
            self.reliability_feat_idx: Optional[int] = col2idx[reliability_feature_name]
        else:
            self.reliability_feat_idx = None

        self.pre_ln = nn.LayerNorm(d_model)
        self.group_feat_proj = nn.ModuleList([nn.Linear(fd, d_feat_emb) for fd in self.group_feature_dims])
        self.hidden_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_router_hidden),
            nn.GELU(),
            nn.Linear(d_router_hidden, d_router_hidden),
        )
        self.group_feature_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(group_dim),
                    nn.Linear(group_dim, d_router_hidden),
                    nn.GELU(),
                    nn.Linear(d_router_hidden, d_router_hidden),
                )
                for group_dim in self.group_feature_dims
            ]
        )
        self.router_feat_drop = nn.Dropout(router_feature_dropout)

        outer_in_dim = 0
        if self.outer_router_use_hidden:
            outer_in_dim += d_router_hidden
        if self.outer_router_use_feature:
            outer_in_dim += d_router_hidden
        self.outer_group_router = Router(
            d_in=outer_in_dim,
            n_experts=self.n_groups,
            d_hidden=d_router_hidden,
            top_k=self.group_top_k,
            dropout=dropout,
        )
        self.intra_routers = nn.ModuleList(
            [
                Router(
                    d_in=4 * d_router_hidden,
                    n_experts=self.expert_scale,
                    d_hidden=d_router_hidden,
                    top_k=self.expert_top_k,
                    dropout=dropout,
                )
                for _ in range(self.n_groups)
            ]
        )
        if self.router_mode == "session" and self.session_pooling == "query":
            self.session_query_hidden = nn.Parameter(
                torch.randn(d_router_hidden) * (1.0 / math.sqrt(float(d_router_hidden)))
            )
            self.session_query_feature = nn.Parameter(
                torch.randn(d_router_hidden) * (1.0 / math.sqrt(float(d_router_hidden)))
            )
        else:
            self.register_parameter("session_query_hidden", None)
            self.register_parameter("session_query_feature", None)

        self.inner_rule_teacher = InnerBinRuleTeacher(
            expert_scale=self.expert_scale,
            bin_sharpness=inner_rule_bin_sharpness,
            group_feature_pool=inner_rule_group_feature_pool,
        )

        self.experts = nn.ModuleList(
            [
                HiddenExpertMLP(
                    d_model=d_model,
                    d_hidden=d_expert_hidden,
                    dropout=dropout,
                    d_feat_emb=d_feat_emb,
                    use_hidden=self.expert_use_hidden,
                    use_feature=self.expert_use_feature,
                )
                for _ in range(self.n_experts)
            ]
        )
        self.resid_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def set_schedule_state(
        self,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        expert_top_k: Optional[int] = None,
    ) -> None:
        if alpha_scale is not None:
            self.alpha_scale = float(alpha_scale)
        if router_temperature is not None:
            self.current_router_temperature = max(float(router_temperature), 1e-6)
        if expert_top_k is not None:
            self.current_expert_top_k = None if int(expert_top_k) <= 0 else int(expert_top_k)

    def _group_feature_tensor(self, feat: torch.Tensor, group_idx: int) -> torch.Tensor:
        idx = getattr(self, f"group_feat_idx_{group_idx}")
        return feat.index_select(-1, idx)

    def _apply_reliability(self, feat_emb: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.reliability_feat_idx is None:
            return feat_emb
        rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
        return feat_emb * rel

    def _apply_reliability_to_raw(self, raw_feat: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.reliability_feat_idx is None:
            return raw_feat
        rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
        return raw_feat * rel

    @staticmethod
    def _build_valid_mask(
        batch_size: int,
        seq_len: int,
        item_seq_len: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if item_seq_len is None:
            return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        lens = item_seq_len.to(device=device).long()
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        return arange < lens.unsqueeze(1)

    @staticmethod
    def _pool_sequence_query(
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        scores = (seq * query.view(1, 1, -1)).sum(dim=-1) / math.sqrt(float(seq.size(-1)))
        scores = scores.masked_fill(~valid_mask, torch.finfo(seq.dtype).min)
        attn = torch.softmax(scores, dim=1)
        attn = attn * valid_mask.float()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom
        return (attn.unsqueeze(-1) * seq).sum(dim=1)

    def _pool_sequence(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.session_pooling == "mean":
            weights = valid_mask.float().unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            return (seq * weights).sum(dim=1) / denom

        if self.session_pooling == "last":
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

        if query is None:
            raise RuntimeError("session_pooling='query' requires a query parameter.")
        return self._pool_sequence_query(seq, valid_mask, query)

    def _maybe_pool_hidden(
        self,
        hidden_enc: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return hidden_enc
        return self._pool_sequence(hidden_enc, valid_mask, item_seq_len, query=self.session_query_hidden)

    def _maybe_pool_feature(
        self,
        feat_enc: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return feat_enc
        return self._pool_sequence(feat_enc, valid_mask, item_seq_len, query=self.session_query_feature)

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

    def _build_outer_router_input(
        self,
        hidden_ctx: torch.Tensor,
        stage_feat_ctx: torch.Tensor,
    ) -> torch.Tensor:
        inputs = []
        if self.outer_router_use_hidden:
            inputs.append(hidden_ctx)
        if self.outer_router_use_feature:
            inputs.append(stage_feat_ctx)
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

    def _build_inner_router_input(
        self,
        hidden_ctx: torch.Tensor,
        group_feat_ctx: torch.Tensor,
    ) -> torch.Tensor:
        if not self.inner_router_use_hidden:
            hidden_ctx = torch.zeros_like(group_feat_ctx)
        if not self.inner_router_use_feature:
            group_feat_ctx = torch.zeros_like(hidden_ctx)
        return self._compose_interaction(hidden_ctx, group_feat_ctx)

    def _encode_group_feature(
        self,
        *,
        group_idx: int,
        group_feat: torch.Tensor,
        feat: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        group_feat = self._apply_reliability_to_raw(group_feat, feat)
        feat_enc = self.group_feature_encoders[group_idx](group_feat)
        feat_enc = self.router_feat_drop(feat_enc)
        return self._maybe_pool_feature(feat_enc, valid_mask, item_seq_len)

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
        hidden_ctx = self._maybe_pool_hidden(hidden_enc, valid_mask, item_seq_len)

        stage_feat = feat.index_select(-1, self.stage_feat_idx)
        stage_feat = self._apply_reliability_to_raw(stage_feat, feat)
        stage_feat_ratio = _to_ratio(stage_feat)
        stage_feat_score = stage_feat_ratio.mean(dim=-1, keepdim=True)
        if self.router_mode == "session":
            stage_feat_ctx = self.inner_rule_teacher._pool_session_score(
                stage_feat_score,
                valid_mask,
                item_seq_len,
                self.session_pooling,
            )
            stage_feat_ctx = stage_feat_ctx.expand(-1, hidden_ctx.shape[-1])
        else:
            stage_feat_ctx = stage_feat_score.expand(-1, -1, hidden_ctx.shape[-1])
        outer_in = self._build_outer_router_input(hidden_ctx, stage_feat_ctx)
        group_weights_ctx, group_logits_ctx = self.outer_group_router(
            outer_in,
            temperature=self.current_router_temperature,
            top_k=self.current_group_top_k,
        )
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
            group_feat_ctx = self._encode_group_feature(
                group_idx=group_idx,
                group_feat=group_feat,
                feat=feat,
                valid_mask=valid_mask,
                item_seq_len=item_seq_len,
            )
            intra_in = self._build_inner_router_input(hidden_ctx, group_feat_ctx)

            if self.expert_scale <= 1:
                intra_raw = torch.zeros(tuple(intra_in.shape[:-1]) + (1,), device=hidden.device, dtype=hidden.dtype)
                intra_logits = intra_raw
                intra_w = torch.ones_like(intra_raw)
                teacher_logits = torch.zeros_like(intra_raw)
                teacher_score = torch.full_like(intra_raw, 0.5)
            else:
                _, intra_raw = self.intra_routers[group_idx](
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
                    teacher_score = torch.full(
                        (bsz, tlen, 1),
                        0.5,
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
            "inner_rule_bin_centers": self.inner_rule_teacher.bin_centers,
        }
        return next_hidden, gate_weights, gate_logits, group_weights, group_logits, stage_delta


class HierarchicalMoEHGRv3(nn.Module):
    """3-stage HGRv3 wrapper with optional parallel stage merge."""

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
        outer_router_use_feature: bool = False,
        inner_router_use_hidden: bool = True,
        inner_router_use_feature: bool = True,
        inner_rule_enable: bool = True,
        inner_rule_mode: str = "distill",
        inner_rule_bias_scale: float = 1.0,
        inner_rule_bin_sharpness: float = 16.0,
        inner_rule_group_feature_pool: str = "mean_ratio",
        inner_rule_apply_stages: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        mode = str(stage_merge_mode).lower().strip()
        if mode not in {"serial", "parallel"}:
            raise ValueError(f"stage_merge_mode must be one of ['serial','parallel'], got {stage_merge_mode}")
        self.stage_merge_mode = mode
        self.outer_router_use_hidden = bool(outer_router_use_hidden)
        self.outer_router_use_feature = bool(outer_router_use_feature)
        self.parallel_stage_gate_temperature = float(parallel_stage_gate_temperature)
        self.parallel_stage_gate_top_k = (
            None if parallel_stage_gate_top_k is None or int(parallel_stage_gate_top_k) <= 0
            else int(parallel_stage_gate_top_k)
        )

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.n_total_features = len(ALL_FEATURE_COLUMNS)
        self.active_stages: List[str] = []
        self.expert_names: Dict[str, List[str]] = {}
        self.group_names: Dict[str, List[str]] = {}
        self.stage_default_temperatures: Dict[str, float] = {
            "macro": 1.0,
            "mid": float(mid_router_temperature),
            "micro": float(micro_router_temperature),
        }
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

            stage_module = HierarchicalGroupStageMoEv3(
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
                inner_rule_enable=inner_rule_enable,
                inner_rule_mode=inner_rule_mode,
                inner_rule_bias_scale=inner_rule_bias_scale,
                inner_rule_bin_sharpness=inner_rule_bin_sharpness,
                inner_rule_group_feature_pool=inner_rule_group_feature_pool,
                inner_rule_apply=(stage_name in apply_stage_set),
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

        if self.n_active >= 2 and self.stage_merge_mode == "parallel":
            self.stage_merge_feat_proj = nn.Linear(self.n_total_features, d_feat_emb)
            merge_in_dim = 0
            if self.outer_router_use_hidden:
                merge_in_dim += d_model
            if self.outer_router_use_feature:
                merge_in_dim += d_feat_emb
            self.stage_merge_router = Router(
                d_in=merge_in_dim,
                n_experts=self.n_active,
                d_hidden=d_router_hidden,
                top_k=self.parallel_stage_gate_top_k,
                dropout=dropout,
            )
        else:
            self.stage_merge_feat_proj = None
            self.stage_merge_router = None

    def has_stage(self, stage_name: str) -> bool:
        return stage_name in self.active_stages and hasattr(self, f"{stage_name}_stage")

    def set_schedule_state(
        self,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        stage_temperatures = stage_temperatures or {}
        for stage_name in self.active_stages:
            stage_module = getattr(self, f"{stage_name}_stage")
            stage_temp = float(stage_temperatures.get(stage_name, self.stage_default_temperatures[stage_name]))
            stage_module.set_schedule_state(
                alpha_scale=alpha_scale,
                router_temperature=stage_temp,
                expert_top_k=top_k,
            )

    def forward_stage(
        self,
        stage_name: str,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_stage(stage_name):
            raise ValueError(f"stage '{stage_name}' is not active in this HierarchicalMoEHGRv3 instance")
        stage_module = getattr(self, f"{stage_name}_stage")
        return stage_module(hidden, feat, item_seq_len=item_seq_len)

    def get_stage_router_aux(self, stage_name: str) -> Dict[str, torch.Tensor]:
        if not self.has_stage(stage_name):
            return {}
        stage_module = getattr(self, f"{stage_name}_stage")
        return dict(getattr(stage_module, "last_router_aux", {}) or {})

    def _build_stage_merge_input(self, hidden: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        inputs = []
        if self.outer_router_use_hidden:
            inputs.append(hidden)
        if self.outer_router_use_feature:
            assert self.stage_merge_feat_proj is not None
            inputs.append(self.stage_merge_feat_proj(feat))
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

    def parallel_merge(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        stage_deltas: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.n_active == 0:
            z = torch.zeros(hidden.shape[0], hidden.shape[1], 0, device=hidden.device, dtype=hidden.dtype)
            return hidden, z, z

        ordered_deltas = [stage_deltas[s] for s in self.active_stages]
        if self.n_active == 1:
            stage_weights = torch.ones(hidden.shape[0], hidden.shape[1], 1, device=hidden.device, dtype=hidden.dtype)
            stage_logits = torch.zeros_like(stage_weights)
            return hidden + ordered_deltas[0], stage_weights, stage_logits

        if self.stage_merge_router is None:
            stacked = torch.stack(ordered_deltas, dim=-2)
            stage_weights = torch.full(
                (hidden.shape[0], hidden.shape[1], self.n_active),
                fill_value=1.0 / float(self.n_active),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            stage_logits = torch.log(stage_weights.clamp(min=1e-8))
            merged_delta = (stage_weights.unsqueeze(-1) * stacked).sum(dim=-2)
            return hidden + merged_delta, stage_weights, stage_logits

        stage_router_in = self._build_stage_merge_input(hidden, feat)
        stage_weights, stage_logits = self.stage_merge_router(
            stage_router_in,
            temperature=self.parallel_stage_gate_temperature,
            top_k=self.parallel_stage_gate_top_k,
        )
        stacked = torch.stack(ordered_deltas, dim=-2)
        merged_delta = (stage_weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return hidden + merged_delta, stage_weights, stage_logits

    @staticmethod
    def _masked_load_balance(
        gate_weights: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if item_seq_len is None:
            return load_balance_loss(gate_weights, n_experts=gate_weights.shape[-1])

        _, tlen, n_experts = gate_weights.shape
        lens = item_seq_len.to(device=gate_weights.device).long()
        valid = torch.arange(tlen, device=gate_weights.device).unsqueeze(0) < lens.unsqueeze(1)
        if not valid.any():
            return torch.tensor(0.0, device=gate_weights.device)
        flat = gate_weights[valid]
        return load_balance_loss(flat, n_experts=n_experts)

    def compute_aux_loss(
        self,
        weights: Dict[str, torch.Tensor],
        item_seq_len: Optional[torch.Tensor] = None,
        balance_lambda: float = 0.01,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        expert_total = torch.tensor(0.0, device=device)
        for w in weights.values():
            expert_total = expert_total + self._masked_load_balance(w, item_seq_len=item_seq_len)
        return balance_lambda * expert_total
