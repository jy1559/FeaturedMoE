"""3-stage hidden-aware residual MoE (Macro -> Mid -> Micro).

Each stage performs:
1) Pre-LN on hidden states.
2) Router weights from hidden and/or stage feature embedding.
3) Expert outputs from hidden and/or expert feature embeddings.
4) Weighted sum + residual update to hidden states.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .experts import ExpertGroup
from .routers import Router, load_balance_loss
from .feature_config import (
    STAGES,
    STAGE_ALL_FEATURES,
    ALL_FEATURE_COLUMNS,
    build_column_to_index,
    build_expert_indices,
    build_stage_indices,
)


def _scaled_expert_lists(expert_feature_lists: List[List[str]], expert_scale: int) -> List[List[str]]:
    """Replicate each base expert feature list by expert_scale.

    Example (scale=2):
      [E1, E2, E3, E4] -> [E1, E1, E2, E2, E3, E3, E4, E4]
    """
    scaled: List[List[str]] = []
    for feats in expert_feature_lists:
        for _ in range(expert_scale):
            scaled.append(list(feats))
    return scaled


def _scaled_expert_names(base_names: List[str], expert_scale: int) -> List[str]:
    if expert_scale == 1:
        return list(base_names)
    out = []
    for name in base_names:
        for i in range(expert_scale):
            suffix = chr(ord("a") + i)
            out.append(f"{name}_{suffix}")
    return out


class MoEStage(nn.Module):
    """One hidden-aware residual MoE stage.

    Parameters
    ----------
    stage_name : str
    expert_feature_lists : list[list[str]]
        Base expert feature lists (length=4 before scaling).
    stage_all_features : list[str]
        Union of stage-level feature columns.
    col2idx : dict[str, int]
    d_model : int
        Hidden state dimension.
    d_feat_emb : int
        Feature embedding dimension for router/expert inputs.
    d_expert_hidden : int
    d_router_hidden : int
    expert_scale : int
        Number of clones per base expert (1/2/3).
    top_k : int | None
    dropout : float
    router_use_hidden : bool
    router_use_feature : bool
    expert_use_hidden : bool
    expert_use_feature : bool
    """

    def __init__(
        self,
        stage_name: str,
        expert_feature_lists: List[List[str]],
        stage_all_features: List[str],
        col2idx: Dict[str, int],
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        top_k: Optional[int] = None,
        dropout: float = 0.1,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = True,
        router_mode: str = "token",
        session_pooling: str = "query",
        router_temperature: float = 1.0,
        router_feature_dropout: float = 0.0,
        reliability_feature_name: Optional[str] = None,
    ):
        super().__init__()
        if expert_scale not in (1, 2, 3):
            raise ValueError(f"expert_scale must be one of [1, 2, 3], got {expert_scale}")
        if not (router_use_hidden or router_use_feature):
            raise ValueError("Router must use at least one input source.")
        if not (expert_use_hidden or expert_use_feature):
            raise ValueError("Expert must use at least one input source.")
        if router_mode not in ("token", "session"):
            raise ValueError(f"router_mode must be one of ['token', 'session'], got {router_mode}")
        if session_pooling not in ("query", "mean", "last"):
            raise ValueError(f"session_pooling must be one of ['query', 'mean', 'last'], got {session_pooling}")
        if router_temperature <= 0:
            raise ValueError(f"router_temperature must be > 0, got {router_temperature}")
        if not (0.0 <= router_feature_dropout < 1.0):
            raise ValueError(f"router_feature_dropout must be in [0,1), got {router_feature_dropout}")

        self.stage_name = stage_name
        self.d_model = d_model
        self.d_feat_emb = d_feat_emb
        self.expert_scale = expert_scale
        self.router_use_hidden = bool(router_use_hidden)
        self.router_use_feature = bool(router_use_feature)
        self.expert_use_hidden = bool(expert_use_hidden)
        self.expert_use_feature = bool(expert_use_feature)
        self.router_mode = router_mode
        self.session_pooling = session_pooling
        self.router_temperature = float(router_temperature)
        self.router_top_k = top_k
        self.reliability_feature_name = reliability_feature_name
        if reliability_feature_name is not None:
            if reliability_feature_name not in col2idx:
                raise ValueError(
                    f"reliability_feature_name '{reliability_feature_name}' not found in feature columns."
                )
            self.reliability_feat_idx: Optional[int] = col2idx[reliability_feature_name]
        else:
            self.reliability_feat_idx = None

        # Expand base-4 experts by scale (same features, independent params).
        scaled_lists = _scaled_expert_lists(expert_feature_lists, expert_scale)
        self.n_experts = len(scaled_lists)

        # Keep readable expert names for logging.
        base_names = [f"expert_{i}" for i in range(len(expert_feature_lists))]
        self.expert_names = _scaled_expert_names(base_names, expert_scale)

        # ---- Index tensors (buffers) ----
        expert_idx = build_expert_indices(
            OrderedDict(zip([f"expert_{i}" for i in range(len(scaled_lists))], scaled_lists)),
            col2idx,
        )
        for i, idx in enumerate(expert_idx):
            self.register_buffer(
                f"expert_idx_{i}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
        self._n_expert_features = [len(idx) for idx in expert_idx]

        stage_idx = build_stage_indices(stage_all_features, col2idx)
        self.register_buffer(
            "stage_feat_idx",
            torch.tensor(stage_idx, dtype=torch.long),
            persistent=False,
        )
        self._n_stage_features = len(stage_idx)

        # ---- Submodules ----
        self.pre_ln = nn.LayerNorm(d_model)
        self.stage_feat_proj = nn.Linear(self._n_stage_features, d_feat_emb)
        self.router_feat_drop = nn.Dropout(router_feature_dropout)

        router_input_dim = 0
        if self.router_use_hidden:
            router_input_dim += d_model
        if self.router_use_feature:
            router_input_dim += d_feat_emb
        self.router = Router(
            d_in=router_input_dim,
            n_experts=self.n_experts,
            d_hidden=d_router_hidden,
            top_k=top_k,
            dropout=dropout,
        )

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

        self.expert_group = ExpertGroup(
            expert_feature_dims=self._n_expert_features,
            d_model=d_model,
            d_feat_emb=d_feat_emb,
            d_hidden=d_expert_hidden,
            d_out=d_model,
            use_hidden=self.expert_use_hidden,
            use_feature=self.expert_use_feature,
            dropout=dropout,
        )

        self.resid_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # Runtime scheduling states (defaults keep legacy behavior).
        self.alpha_scale = 1.0
        self.current_router_temperature = self.router_temperature
        self.current_top_k = self.router_top_k

    def set_schedule_state(
        self,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Update runtime MoE schedule state for this stage.

        Args:
            alpha_scale: Residual scale multiplier for stage output.
            router_temperature: Runtime temperature override.
            top_k: Runtime top-k override. ``<=0`` means dense gating.
        """
        if alpha_scale is not None:
            self.alpha_scale = float(alpha_scale)
        if router_temperature is not None:
            self.current_router_temperature = max(float(router_temperature), 1e-6)
        if top_k is not None:
            self.current_top_k = None if int(top_k) <= 0 else int(top_k)

    def _gather_expert_inputs(self, feat: torch.Tensor) -> List[torch.Tensor]:
        inputs = []
        for i in range(self.n_experts):
            idx = getattr(self, f"expert_idx_{i}")
            inputs.append(feat.index_select(-1, idx))
        return inputs

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
        arange = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        return arange < lens.unsqueeze(1)  # [B, T]

    def _pool_sequence_query(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        # Query-attention pooling as a lightweight [CLS]-like session summarizer.
        scores = (seq * query.view(1, 1, -1)).sum(dim=-1) / math.sqrt(float(seq.size(-1)))  # [B, T]
        scores = scores.masked_fill(~valid_mask, torch.finfo(seq.dtype).min)
        attn = torch.softmax(scores, dim=1)
        attn = attn * valid_mask.float()
        denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn = attn / denom
        return (attn.unsqueeze(-1) * seq).sum(dim=1)  # [B, D]

    def _pool_sequence(
        self,
        seq: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.session_pooling == "mean":
            w = valid_mask.float().unsqueeze(-1)
            denom = w.sum(dim=1).clamp(min=1.0)
            return (seq * w).sum(dim=1) / denom

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

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: [B, T, d_model]
            feat: [B, T, D_total]
            item_seq_len: [B] or None
        Returns:
            next_hidden: [B, T, d_model]
            gate_weights: [B, T, K]
            gate_logits: [B, T, K]
        """
        B, T, _ = hidden.shape
        h_norm = self.pre_ln(hidden)
        valid_mask = self._build_valid_mask(B, T, item_seq_len, hidden.device)

        # Stage feature embedding for router input.
        stage_feat = feat.index_select(-1, self.stage_feat_idx)            # [B, T, F_stage]
        stage_feat_emb = self.stage_feat_proj(stage_feat)                  # [B, T, d_feat_emb]
        if self.reliability_feat_idx is not None:
            rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
            stage_feat_emb = stage_feat_emb * rel
        stage_feat_emb = self.router_feat_drop(stage_feat_emb)

        if self.router_mode == "session":
            router_inputs = []
            if self.router_use_hidden:
                h_sess = self._pool_sequence(
                    h_norm, valid_mask, item_seq_len, query=self.session_query_hidden
                )
                router_inputs.append(h_sess)
            if self.router_use_feature:
                f_sess = self._pool_sequence(
                    stage_feat_emb, valid_mask, item_seq_len, query=self.session_query_feature
                )
                router_inputs.append(f_sess)
            router_in = router_inputs[0] if len(router_inputs) == 1 else torch.cat(router_inputs, dim=-1)
            gate_w_sess, gate_l_sess = self.router(
                router_in,
                temperature=self.current_router_temperature,
                top_k=self.current_top_k,
            )  # [B, K]
            gate_weights = gate_w_sess.unsqueeze(1).expand(-1, T, -1)
            gate_logits = gate_l_sess.unsqueeze(1).expand(-1, T, -1)
        else:
            router_inputs = []
            if self.router_use_hidden:
                router_inputs.append(h_norm)
            if self.router_use_feature:
                router_inputs.append(stage_feat_emb)
            router_in = router_inputs[0] if len(router_inputs) == 1 else torch.cat(router_inputs, dim=-1)
            gate_weights, gate_logits = self.router(
                router_in,
                temperature=self.current_router_temperature,
                top_k=self.current_top_k,
            )  # [B, T, K]

        expert_inputs = self._gather_expert_inputs(feat)
        expert_out = self.expert_group(h_norm, expert_inputs)              # [B, T, K, d_model]

        stage_out = (gate_weights.unsqueeze(-1) * expert_out).sum(dim=-2)  # [B, T, d_model]
        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_out)
        return next_hidden, gate_weights, gate_logits


class HierarchicalMoE(nn.Module):
    """Hidden-aware 3-stage residual MoE with optional stage toggles."""

    def __init__(
        self,
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        top_k: Optional[int] = None,
        dropout: float = 0.1,
        use_macro: bool = True,
        use_mid: bool = True,
        use_micro: bool = True,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
        expert_use_hidden: bool = True,
        expert_use_feature: bool = True,
        macro_routing_scope: str = "session",
        macro_session_pooling: str = "query",
        mid_router_temperature: float = 1.3,
        micro_router_temperature: float = 1.3,
        mid_router_feature_dropout: float = 0.1,
        micro_router_feature_dropout: float = 0.1,
        use_valid_ratio_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_feat_emb = d_feat_emb
        self.expert_scale = expert_scale
        self.use_macro = use_macro
        self.use_mid = use_mid
        self.use_micro = use_micro
        self.macro_routing_scope = macro_routing_scope
        self.macro_session_pooling = macro_session_pooling
        self.mid_router_temperature = float(mid_router_temperature)
        self.micro_router_temperature = float(micro_router_temperature)
        self.mid_router_feature_dropout = float(mid_router_feature_dropout)
        self.micro_router_feature_dropout = float(micro_router_feature_dropout)
        self.use_valid_ratio_gating = bool(use_valid_ratio_gating)
        if self.macro_routing_scope not in ("session", "token"):
            raise ValueError(
                f"macro_routing_scope must be one of ['session', 'token'], got {self.macro_routing_scope}"
            )
        if self.macro_session_pooling not in ("query", "mean", "last"):
            raise ValueError(
                "macro_session_pooling must be one of ['query', 'mean', 'last'], "
                f"got {self.macro_session_pooling}"
            )
        if self.mid_router_temperature <= 0 or self.micro_router_temperature <= 0:
            raise ValueError("mid/micro router temperature must be > 0.")
        if not (0.0 <= self.mid_router_feature_dropout < 1.0):
            raise ValueError("mid_router_feature_dropout must be in [0,1).")
        if not (0.0 <= self.micro_router_feature_dropout < 1.0):
            raise ValueError("micro_router_feature_dropout must be in [0,1).")

        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.n_total_features = len(ALL_FEATURE_COLUMNS)

        stage_expert_lists: Dict[str, List[List[str]]] = {}
        stage_base_names: Dict[str, List[str]] = {}
        for stage_name, expert_dict in STAGES:
            stage_expert_lists[stage_name] = list(expert_dict.values())
            stage_base_names[stage_name] = list(expert_dict.keys())

        self.active_stages: List[str] = []
        self.expert_names: Dict[str, List[str]] = {}

        if use_macro:
            self.macro_stage = MoEStage(
                stage_name="macro",
                expert_feature_lists=stage_expert_lists["macro"],
                stage_all_features=STAGE_ALL_FEATURES["macro"],
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
                router_mode=self.macro_routing_scope,
                session_pooling=self.macro_session_pooling,
                router_temperature=1.0,
                router_feature_dropout=0.0,
                reliability_feature_name=None,
            )
            self.active_stages.append("macro")
            self.expert_names["macro"] = _scaled_expert_names(stage_base_names["macro"], expert_scale)

        if use_mid:
            self.mid_stage = MoEStage(
                stage_name="mid",
                expert_feature_lists=stage_expert_lists["mid"],
                stage_all_features=STAGE_ALL_FEATURES["mid"],
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
                router_mode="token",
                session_pooling="query",
                router_temperature=self.mid_router_temperature,
                router_feature_dropout=self.mid_router_feature_dropout,
                reliability_feature_name=("mid_valid_r" if self.use_valid_ratio_gating else None),
            )
            self.active_stages.append("mid")
            self.expert_names["mid"] = _scaled_expert_names(stage_base_names["mid"], expert_scale)

        if use_micro:
            self.micro_stage = MoEStage(
                stage_name="micro",
                expert_feature_lists=stage_expert_lists["micro"],
                stage_all_features=STAGE_ALL_FEATURES["micro"],
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
                router_mode="token",
                session_pooling="query",
                router_temperature=self.micro_router_temperature,
                router_feature_dropout=self.micro_router_feature_dropout,
                reliability_feature_name=("mic_valid_r" if self.use_valid_ratio_gating else None),
            )
            self.active_stages.append("micro")
            self.expert_names["micro"] = _scaled_expert_names(stage_base_names["micro"], expert_scale)

        self.n_active = len(self.active_stages)

    def has_stage(self, stage_name: str) -> bool:
        """Return True if stage module exists and is active."""
        return stage_name in self.active_stages and hasattr(self, f"{stage_name}_stage")

    def set_schedule_state(
        self,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Propagate runtime scheduling knobs to active stages."""
        stage_temperatures = stage_temperatures or {}
        for stage_name in self.active_stages:
            stage_module = getattr(self, f"{stage_name}_stage")
            stage_module.set_schedule_state(
                alpha_scale=alpha_scale,
                router_temperature=stage_temperatures.get(stage_name, None),
                top_k=top_k,
            )

    def forward_stage(
        self,
        stage_name: str,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one stage by name.

        This is used by FeaturedMoE interleaving logic to insert attention
        blocks between stage MoE modules.
        """
        if not self.has_stage(stage_name):
            raise ValueError(f"stage '{stage_name}' is not active in this HierarchicalMoE instance")
        stage_module = getattr(self, f"{stage_name}_stage")
        return stage_module(hidden, feat, item_seq_len=item_seq_len)

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            hidden: [B, T, d_model]
            feat: [B, T, D_total]
            item_seq_len: [B] or None
        Returns:
            hidden: [B, T, d_model]
            weights: {stage: [B, T, K]}
            logits: {stage: [B, T, K]}
        """
        if self.n_active == 0:
            return hidden, {}, {}

        all_weights: Dict[str, torch.Tensor] = {}
        all_logits: Dict[str, torch.Tensor] = {}
        out = hidden

        for stage_name in self.active_stages:
            out, w, l = self.forward_stage(stage_name, out, feat, item_seq_len=item_seq_len)
            all_weights[stage_name] = w
            all_logits[stage_name] = l

        return out, all_weights, all_logits

    def compute_aux_loss(
        self,
        weights: Dict[str, torch.Tensor],
        item_seq_len: Optional[torch.Tensor] = None,
        balance_lambda: float = 0.01,
    ) -> torch.Tensor:
        """Compute combined load-balance loss across active stages."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for w in weights.values():
            if item_seq_len is not None:
                _, T, K = w.shape
                lens = item_seq_len.to(device=w.device).long()
                valid_mask = torch.arange(T, device=w.device).unsqueeze(0) < lens.unsqueeze(1)
                if valid_mask.any():
                    w_valid = w[valid_mask]  # [N_valid, K]
                    total = total + load_balance_loss(w_valid, n_experts=K)
                continue
            total = total + load_balance_loss(w, n_experts=w.shape[-1])
        return balance_lambda * total
