"""Core modules for FeaturedMoE_ProtoX."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FeaturedMoE.feature_config import STAGE_ALL_FEATURES
from ..FeaturedMoE.moe_stages import MoEStage


def _masked_mean(x: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
    if item_seq_len is None:
        return x.mean(dim=1)
    bsz, tlen, _ = x.shape
    lens = item_seq_len.to(device=x.device).long().clamp(min=1, max=tlen)
    valid = torch.arange(tlen, device=x.device).unsqueeze(0) < lens.unsqueeze(1)
    denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=x.dtype)
    return (x * valid.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1) / denom


def _last_valid(x: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
    bsz, tlen, dim = x.shape
    if item_seq_len is None:
        return x[:, -1, :]
    lens = item_seq_len.to(device=x.device).long().clamp(min=1, max=tlen)
    idx = (lens - 1).view(-1)
    batch_idx = torch.arange(bsz, device=x.device)
    return x[batch_idx, idx, :].view(bsz, dim)


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


class SessionPrototypeAllocator(nn.Module):
    """Session-level prototype allocator (pi over K prototypes)."""

    def __init__(
        self,
        *,
        d_model: int,
        n_features: int,
        d_feat_emb: int,
        d_hidden: int,
        proto_num: int,
        proto_dim: int,
        dropout: float,
        top_k: Optional[int],
        temperature: float,
        use_hidden: bool,
        use_feature: bool,
        pooling: str,
    ):
        super().__init__()
        if not (use_hidden or use_feature):
            raise ValueError("SessionPrototypeAllocator must use hidden and/or feature inputs.")
        if int(proto_num) <= 0:
            raise ValueError(f"proto_num must be > 0, got {proto_num}")
        if int(proto_dim) <= 0:
            raise ValueError(f"proto_dim must be > 0, got {proto_dim}")
        if float(temperature) <= 0:
            raise ValueError(f"prototype temperature must be > 0, got {temperature}")

        pooling_key = str(pooling).lower().strip()
        if pooling_key not in {"query", "last", "mean"}:
            raise ValueError(f"prototype pooling must be one of [query,last,mean], got {pooling}")

        self.top_k = None if top_k is None or int(top_k) <= 0 else int(top_k)
        self.temperature = float(temperature)
        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)
        self.pooling = pooling_key
        self.proto_num = int(proto_num)

        if self.use_feature:
            self.feature_proj = nn.Linear(n_features, d_feat_emb)
            feat_dim = d_feat_emb
        else:
            self.feature_proj = None
            feat_dim = 0

        in_dim = (d_model if self.use_hidden else 0) + feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, self.proto_num),
        )
        self.prototype_bank = nn.Parameter(torch.empty(self.proto_num, int(proto_dim)))
        nn.init.normal_(self.prototype_bank, mean=0.0, std=0.02)

    def _pool_hidden(self, hidden: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling in {"query", "last"}:
            return _last_valid(hidden, item_seq_len=item_seq_len)
        return _masked_mean(hidden, item_seq_len=item_seq_len)

    def _pool_feature(self, feat: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = _masked_mean(feat, item_seq_len=item_seq_len)
        if self.feature_proj is None:
            return pooled
        return self.feature_proj(pooled)

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parts: List[torch.Tensor] = []
        if self.use_hidden:
            parts.append(self._pool_hidden(hidden, item_seq_len=item_seq_len))
        if self.use_feature:
            parts.append(self._pool_feature(feat, item_seq_len=item_seq_len))
        pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

        temp = self.temperature if temperature is None else max(float(temperature), 1e-6)
        logits = self.net(pooled) / temp
        use_top_k = self.top_k if top_k is None else (None if int(top_k) <= 0 else int(top_k))
        weights = _topk_softmax(logits, top_k=use_top_k)
        context = weights @ self.prototype_bank
        return weights, logits, context


class PrototypeConditionedStageGate(nn.Module):
    """Prototype-conditioned stage allocator over [macro, mid, micro]."""

    def __init__(
        self,
        *,
        d_model: int,
        n_features: int,
        d_feat_emb: int,
        d_hidden: int,
        proto_dim: int,
        dropout: float,
        use_hidden: bool,
        use_feature: bool,
        pooling: str,
        token_correction: bool,
        token_correction_scale: float,
    ):
        super().__init__()
        if not (use_hidden or use_feature):
            raise ValueError("PrototypeConditionedStageGate must use hidden and/or feature inputs.")

        pooling_key = str(pooling).lower().strip()
        if pooling_key not in {"query", "last", "mean"}:
            raise ValueError(f"stage gate pooling must be one of [query,last,mean], got {pooling}")

        self.use_hidden = bool(use_hidden)
        self.use_feature = bool(use_feature)
        self.pooling = pooling_key
        self.token_correction = bool(token_correction)
        self.token_correction_scale = max(float(token_correction_scale), 0.0)

        if self.use_feature:
            self.feature_proj = nn.Linear(n_features, d_feat_emb)
            feat_dim = d_feat_emb
        else:
            self.feature_proj = None
            feat_dim = 0

        in_dim = int(proto_dim) + (d_model if self.use_hidden else 0) + feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 3),
        )
        if self.token_correction:
            self.token_net = nn.Sequential(
                nn.Linear(in_dim, d_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, 3),
            )
        else:
            self.token_net = None

    def _pool_hidden(self, hidden: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling in {"query", "last"}:
            return _last_valid(hidden, item_seq_len=item_seq_len)
        return _masked_mean(hidden, item_seq_len=item_seq_len)

    def _pool_feature(self, feat: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        pooled = _masked_mean(feat, item_seq_len=item_seq_len)
        if self.feature_proj is None:
            return pooled
        return self.feature_proj(pooled)

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        proto_context: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        parts: List[torch.Tensor] = [proto_context]
        token_parts: List[torch.Tensor] = [proto_context.unsqueeze(1).expand(-1, hidden.shape[1], -1)]
        if self.use_hidden:
            parts.append(self._pool_hidden(hidden, item_seq_len=item_seq_len))
            token_parts.append(hidden)
        if self.use_feature:
            parts.append(self._pool_feature(feat, item_seq_len=item_seq_len))
            pooled_feat = self.feature_proj(feat) if self.feature_proj is not None else feat
            token_parts.append(pooled_feat)
        pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        logits = self.net(pooled)
        if self.token_net is not None and self.token_correction_scale > 0:
            token_input = token_parts[0] if len(token_parts) == 1 else torch.cat(token_parts, dim=-1)
            token_logits = self.token_net(token_input)
            if item_seq_len is None:
                token_bias = token_logits.mean(dim=1)
            else:
                bsz, tlen, _ = token_logits.shape
                lens = item_seq_len.to(device=token_logits.device).long().clamp(min=1, max=tlen)
                valid = torch.arange(tlen, device=token_logits.device).unsqueeze(0) < lens.unsqueeze(1)
                denom = valid.sum(dim=1, keepdim=True).clamp(min=1).to(dtype=token_logits.dtype)
                token_bias = (token_logits * valid.unsqueeze(-1).to(dtype=token_logits.dtype)).sum(dim=1) / denom
            logits = logits + self.token_correction_scale * token_bias
        weights = F.softmax(logits, dim=-1)
        return weights, logits


class PrototypeConditionedStageBlock(nn.Module):
    """Token-level stage block conditioned by prototype context."""

    def __init__(
        self,
        *,
        stage_name: str,
        expert_feature_lists: List[List[str]],
        expert_names: List[str],
        col2idx: Dict[str, int],
        d_model: int,
        n_features: int,
        proto_dim: int,
        d_feat_emb: int,
        d_expert_hidden: int,
        d_router_hidden: int,
        expert_scale: int,
        top_k: Optional[int],
        dropout: float,
        router_use_hidden: bool,
        router_use_feature: bool,
        expert_use_hidden: bool,
        expert_use_feature: bool,
        router_temperature: float,
        router_feature_dropout: float,
        reliability_feature_name: Optional[str],
    ):
        super().__init__()
        self.stage = MoEStage(
            stage_name=stage_name,
            expert_feature_lists=expert_feature_lists,
            stage_all_features=STAGE_ALL_FEATURES[stage_name],
            col2idx=col2idx,
            d_model=d_model,
            d_feat_emb=d_feat_emb,
            d_expert_hidden=d_expert_hidden,
            d_router_hidden=d_router_hidden,
            expert_scale=expert_scale,
            top_k=top_k,
            dropout=dropout,
            router_use_hidden=router_use_hidden,
            router_use_feature=router_use_feature,
            expert_use_hidden=expert_use_hidden,
            expert_use_feature=expert_use_feature,
            expert_names=expert_names,
            router_impl="learned",
            rule_router_cfg={},
            router_mode="token",
            session_pooling="query",
            router_temperature=router_temperature,
            router_feature_dropout=router_feature_dropout,
            reliability_feature_name=reliability_feature_name,
        )
        self.hidden_ctx_proj = nn.Linear(proto_dim, d_model, bias=False)
        self.feature_ctx_proj = nn.Linear(proto_dim, n_features, bias=False)

    @property
    def n_experts(self) -> int:
        return int(self.stage.n_experts)

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.stage.set_schedule_state(
            alpha_scale=alpha_scale,
            router_temperature=router_temperature,
            top_k=top_k,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        proto_context: torch.Tensor,
        *,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_cond = hidden + self.hidden_ctx_proj(proto_context).unsqueeze(1)
        feat_cond = feat + self.feature_ctx_proj(proto_context).unsqueeze(1)
        next_hidden, gate_weights, gate_logits = self.stage(hidden_cond, feat_cond, item_seq_len=item_seq_len)
        stage_delta = next_hidden - hidden_cond
        return stage_delta, gate_weights, gate_logits
