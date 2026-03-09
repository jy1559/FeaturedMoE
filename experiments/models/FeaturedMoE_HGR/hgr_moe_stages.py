"""Hierarchical Group Routing blocks for FeaturedMoE_HGR."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

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


class ScalarRouter(nn.Module):
    """Group-specific scalar scorer used by per-group routing."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, router_input: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        scale = max(float(temperature), 1e-6)
        return self.net(router_input).squeeze(-1) / scale


class HierarchicalGroupStageMoE(nn.Module):
    """One stage with group router + intra-group expert router."""

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
        group_router_mode: str = "per_group",
        dropout: float = 0.1,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = False,
        router_mode: str = "token",
        session_pooling: str = "query",
        router_temperature: float = 1.0,
        router_feature_dropout: float = 0.0,
        reliability_feature_name: Optional[str] = None,
    ):
        super().__init__()
        if int(expert_scale) < 1:
            raise ValueError(f"expert_scale must be >= 1, got {expert_scale}")
        if not (router_use_hidden or router_use_feature):
            raise ValueError("Router must use at least one input source.")
        if not (expert_use_hidden or expert_use_feature):
            raise ValueError("Experts must use at least one input source.")
        if float(router_temperature) <= 0:
            raise ValueError(f"router_temperature must be > 0, got {router_temperature}")
        if not (0.0 <= float(router_feature_dropout) < 1.0):
            raise ValueError(f"router_feature_dropout must be in [0,1), got {router_feature_dropout}")

        mode = str(group_router_mode).lower().strip()
        if mode not in {"stage_wide", "per_group", "hybrid"}:
            raise ValueError(
                "group_router_mode must be one of ['stage_wide','per_group','hybrid'], "
                f"got {group_router_mode}"
            )
        router_mode_key = str(router_mode).lower().strip()
        if router_mode_key not in {"token", "session"}:
            raise ValueError(f"router_mode must be one of ['token','session'], got {router_mode}")
        session_pooling_key = str(session_pooling).lower().strip()
        if session_pooling_key not in {"query", "mean", "last"}:
            raise ValueError(
                f"session_pooling must be one of ['query','mean','last'], got {session_pooling}"
            )

        self.stage_name = stage_name
        self.group_router_mode = mode
        self.router_use_hidden = bool(router_use_hidden)
        self.router_use_feature = bool(router_use_feature)
        self.expert_use_hidden = bool(expert_use_hidden)
        self.expert_use_feature = bool(expert_use_feature)
        self.router_mode = router_mode_key
        self.session_pooling = session_pooling_key
        self.router_temperature = float(router_temperature)
        self.expert_top_k = None if expert_top_k is None or int(expert_top_k) <= 0 else int(expert_top_k)
        self.group_top_k = None if group_top_k is None or int(group_top_k) <= 0 else int(group_top_k)
        self.reliability_feature_name = reliability_feature_name

        self.n_groups = len(group_feature_lists)
        self.expert_scale = int(expert_scale)
        self.n_experts = self.n_groups * self.expert_scale
        self.expert_names = _scaled_group_expert_names(group_names, self.expert_scale)

        stage_idx = build_stage_indices(stage_all_features, col2idx)
        self.register_buffer("stage_feat_idx", torch.tensor(stage_idx, dtype=torch.long), persistent=False)

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
        self.stage_feat_proj = nn.Linear(len(stage_idx), d_feat_emb)
        self.group_feat_proj = nn.ModuleList([nn.Linear(fd, d_feat_emb) for fd in self.group_feature_dims])
        self.router_feat_drop = nn.Dropout(router_feature_dropout)

        if self.router_mode == "session" and self.session_pooling == "query" and self.router_use_hidden:
            self.session_query_hidden = nn.Parameter(
                torch.randn(d_model) * (1.0 / math.sqrt(float(d_model)))
            )
        else:
            self.register_parameter("session_query_hidden", None)
        if self.router_mode == "session" and self.session_pooling == "query" and self.router_use_feature:
            self.session_query_feature = nn.Parameter(
                torch.randn(d_feat_emb) * (1.0 / math.sqrt(float(d_feat_emb)))
            )
        else:
            self.register_parameter("session_query_feature", None)

        router_in_dim = 0
        if self.router_use_hidden:
            router_in_dim += d_model
        if self.router_use_feature:
            router_in_dim += d_feat_emb

        self.stage_wide_router = Router(
            d_in=router_in_dim,
            n_experts=self.n_groups,
            d_hidden=d_router_hidden,
            top_k=self.group_top_k,
            dropout=dropout,
        )
        self.per_group_scorers = nn.ModuleList(
            [ScalarRouter(d_in=router_in_dim, d_hidden=d_router_hidden, dropout=dropout) for _ in range(self.n_groups)]
        )
        self.intra_routers = nn.ModuleList(
            [
                Router(
                    d_in=router_in_dim,
                    n_experts=self.expert_scale,
                    d_hidden=d_router_hidden,
                    top_k=self.expert_top_k,
                    dropout=dropout,
                )
                for _ in range(self.n_groups)
            ]
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

        self.alpha_scale = 1.0
        self.current_router_temperature = self.router_temperature
        self.current_expert_top_k = self.expert_top_k

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

    def _build_router_input(self, hidden: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        inputs = []
        if self.router_use_hidden:
            inputs.append(hidden)
        if self.router_use_feature:
            inputs.append(feat_emb)
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

    def _group_feature_tensor(self, feat: torch.Tensor, group_idx: int) -> torch.Tensor:
        idx = getattr(self, f"group_feat_idx_{group_idx}")
        return feat.index_select(-1, idx)

    def _apply_reliability(self, feat_emb: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if self.reliability_feat_idx is None:
            return feat_emb
        rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
        return feat_emb * rel

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
        hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return hidden
        query = self.session_query_hidden if self.router_use_hidden else None
        return self._pool_sequence(hidden, valid_mask, item_seq_len, query=query)

    def _maybe_pool_feature(
        self,
        feat_emb: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.router_mode != "session":
            return feat_emb
        query = self.session_query_feature if self.router_use_feature else None
        return self._pool_sequence(feat_emb, valid_mask, item_seq_len, query=query)

    def _compute_group_logits(
        self,
        hidden_ctx: torch.Tensor,
        stage_feat_ctx: torch.Tensor,
        group_feat_ctxs: List[torch.Tensor],
    ) -> torch.Tensor:
        logits_parts: List[torch.Tensor] = []

        if self.group_router_mode in {"stage_wide", "hybrid"}:
            stage_router_in = self._build_router_input(hidden_ctx, stage_feat_ctx)
            stage_wide_logits = self.stage_wide_router.net(stage_router_in) / self.current_router_temperature
            logits_parts.append(stage_wide_logits)

        if self.group_router_mode in {"per_group", "hybrid"}:
            per_group_logits = []
            for group_idx, group_feat_ctx in enumerate(group_feat_ctxs):
                scorer_in = self._build_router_input(hidden_ctx, group_feat_ctx)
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

        stage_feat = feat.index_select(-1, self.stage_feat_idx)
        stage_feat_emb = self.stage_feat_proj(stage_feat)
        stage_feat_emb = self._apply_reliability(stage_feat_emb, feat)
        stage_feat_emb = self.router_feat_drop(stage_feat_emb)

        stage_hidden_ctx = self._maybe_pool_hidden(h_norm, valid_mask, item_seq_len)
        stage_feat_ctx = self._maybe_pool_feature(stage_feat_emb, valid_mask, item_seq_len)

        group_feat_embs: List[torch.Tensor] = []
        group_feat_ctxs: List[torch.Tensor] = []
        intra_weights_list: List[torch.Tensor] = []
        intra_logits_list: List[torch.Tensor] = []
        group_outputs: List[torch.Tensor] = []

        for group_idx in range(self.n_groups):
            group_feat = self._group_feature_tensor(feat, group_idx)
            group_feat_emb = self.group_feat_proj[group_idx](group_feat)
            group_feat_emb = self._apply_reliability(group_feat_emb, feat)
            group_feat_emb = self.router_feat_drop(group_feat_emb)
            group_feat_embs.append(group_feat_emb)

            group_feat_ctx = self._maybe_pool_feature(group_feat_emb, valid_mask, item_seq_len)
            group_feat_ctxs.append(group_feat_ctx)

            intra_in = self._build_router_input(stage_hidden_ctx, group_feat_ctx)
            intra_w, intra_l = self.intra_routers[group_idx](
                intra_in,
                temperature=self.current_router_temperature,
                top_k=self.current_expert_top_k,
            )
            if self.router_mode == "session":
                intra_w = intra_w.unsqueeze(1).expand(-1, tlen, -1)
                intra_l = intra_l.unsqueeze(1).expand(-1, tlen, -1)
            intra_weights_list.append(intra_w)
            intra_logits_list.append(intra_l)

            start = group_idx * self.expert_scale
            end = start + self.expert_scale
            expert_outputs = torch.stack(
                [
                    expert(
                        h_norm,
                        feat_emb=group_feat_emb if self.expert_use_feature else None,
                    )
                    for expert in self.experts[start:end]
                ],
                dim=-2,
            )
            group_output = (intra_w.unsqueeze(-1) * expert_outputs).sum(dim=-2)
            group_outputs.append(group_output)

        group_logits = self._compute_group_logits(stage_hidden_ctx, stage_feat_ctx, group_feat_ctxs)
        if self.router_mode == "session":
            group_logits = group_logits.unsqueeze(1).expand(-1, tlen, -1)
        group_weights = _topk_softmax(group_logits, top_k=self.group_top_k)

        intra_weights = torch.stack(intra_weights_list, dim=-2)
        intra_logits = torch.stack(intra_logits_list, dim=-2)
        final_weights = group_weights.unsqueeze(-1) * intra_weights
        gate_weights = final_weights.reshape(bsz, tlen, self.n_experts)

        final_logits = group_logits.unsqueeze(-1) + intra_logits
        gate_logits = final_logits.reshape(bsz, tlen, self.n_experts)

        stacked_group_outputs = torch.stack(group_outputs, dim=-2)
        stage_delta = (group_weights.unsqueeze(-1) * stacked_group_outputs).sum(dim=-2)
        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_delta)
        return next_hidden, gate_weights, gate_logits, group_weights, group_logits, stage_delta


class HierarchicalMoEHGR(nn.Module):
    """3-stage HGR wrapper with serial/parallel stage merge."""

    def __init__(
        self,
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        top_k: Optional[int] = None,
        group_top_k: Optional[int] = None,
        group_router_mode: str = "per_group",
        parallel_stage_gate_top_k: Optional[int] = None,
        parallel_stage_gate_temperature: float = 1.0,
        dropout: float = 0.1,
        use_macro: bool = True,
        use_mid: bool = True,
        use_micro: bool = True,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
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
    ):
        super().__init__()

        mode = str(stage_merge_mode).lower().strip()
        if mode not in {"serial", "parallel"}:
            raise ValueError(f"stage_merge_mode must be one of ['serial','parallel'], got {stage_merge_mode}")

        self.stage_merge_mode = mode
        self.router_use_hidden = bool(router_use_hidden)
        self.router_use_feature = bool(router_use_feature)
        self.parallel_stage_gate_temperature = float(parallel_stage_gate_temperature)
        self.parallel_stage_gate_top_k = (
            None if parallel_stage_gate_top_k is None or int(parallel_stage_gate_top_k) <= 0
            else int(parallel_stage_gate_top_k)
        )

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.n_total_features = len(ALL_FEATURE_COLUMNS)

        self.active_stages: List[str] = []
        self.expert_names: Dict[str, List[str]] = {}
        self.stage_default_temperatures: Dict[str, float] = {
            "macro": 1.0,
            "mid": float(mid_router_temperature),
            "micro": float(micro_router_temperature),
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

            stage_module = HierarchicalGroupStageMoE(
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
                group_router_mode=group_router_mode,
                dropout=dropout,
                router_use_hidden=router_use_hidden,
                router_use_feature=router_use_feature,
                expert_use_hidden=expert_use_hidden,
                expert_use_feature=expert_use_feature,
                router_mode=router_mode,
                session_pooling=session_pooling,
                router_temperature=self.stage_default_temperatures[stage_name],
                router_feature_dropout=router_feature_dropout,
                reliability_feature_name=reliability_feature_name,
            )
            setattr(self, f"{stage_name}_stage", stage_module)
            self.active_stages.append(stage_name)
            self.expert_names[stage_name] = list(stage_module.expert_names)

        self.n_active = len(self.active_stages)
        if self.n_active > 0:
            first_stage = getattr(self, f"{self.active_stages[0]}_stage")
            self.stage_n_experts = int(first_stage.n_experts)
        else:
            self.stage_n_experts = 0

        if self.n_active >= 2 and self.stage_merge_mode == "parallel":
            self.stage_merge_feat_proj = nn.Linear(self.n_total_features, d_feat_emb)
            merge_in_dim = 0
            if self.router_use_hidden:
                merge_in_dim += d_model
            if self.router_use_feature:
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
            raise ValueError(f"stage '{stage_name}' is not active in this HierarchicalMoEHGR instance")
        stage_module = getattr(self, f"{stage_name}_stage")
        return stage_module(hidden, feat, item_seq_len=item_seq_len)

    def _build_stage_merge_input(self, hidden: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        inputs = []
        if self.router_use_hidden:
            inputs.append(hidden)
        if self.router_use_feature:
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
