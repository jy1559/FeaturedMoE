"""Hierarchical-in-Stage MoE blocks for FeaturedMoE_HiR.

Core idea per stage:
1) Bundle router over 4 feature bundles.
2) Intra-bundle router over E experts per bundle (E=expert_scale).
3) Expert FFN is hidden-only.
4) Final expert weight = bundle_weight * intra_bundle_weight.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..FeaturedMoE.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    STAGE_ALL_FEATURES,
    build_column_to_index,
    build_expert_indices,
    build_stage_indices,
)
from ..FeaturedMoE.routers import Router, load_balance_loss


def _scaled_bundle_expert_names(bundle_names: List[str], expert_scale: int) -> List[str]:
    if expert_scale <= 1:
        return list(bundle_names)

    out: List[str] = []
    for bname in bundle_names:
        for i in range(expert_scale):
            suffix = chr(ord("a") + i)
            out.append(f"{bname}_{suffix}")
    return out


class HiddenExpertMLP(nn.Module):
    """Hidden-only expert FFN."""

    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class HierarchicalStageMoE(nn.Module):
    """One stage with 2-level routing (bundle -> intra-bundle experts)."""

    def __init__(
        self,
        stage_name: str,
        bundle_names: List[str],
        bundle_feature_lists: List[List[str]],
        stage_all_features: List[str],
        col2idx: Dict[str, int],
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        expert_top_k: Optional[int] = None,
        bundle_top_k: Optional[int] = None,
        dropout: float = 0.1,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
        router_temperature: float = 1.0,
        router_feature_dropout: float = 0.0,
        reliability_feature_name: Optional[str] = None,
    ):
        super().__init__()
        if int(expert_scale) < 1:
            raise ValueError(f"expert_scale must be >= 1, got {expert_scale}")
        if not (router_use_hidden or router_use_feature):
            raise ValueError("Router must use at least one input source.")
        if router_temperature <= 0:
            raise ValueError(f"router_temperature must be > 0, got {router_temperature}")
        if not (0.0 <= router_feature_dropout < 1.0):
            raise ValueError(f"router_feature_dropout must be in [0,1), got {router_feature_dropout}")

        self.stage_name = stage_name
        self.router_use_hidden = bool(router_use_hidden)
        self.router_use_feature = bool(router_use_feature)
        self.router_temperature = float(router_temperature)
        self.expert_top_k = None if expert_top_k is None or int(expert_top_k) <= 0 else int(expert_top_k)
        self.bundle_top_k = None if bundle_top_k is None or int(bundle_top_k) <= 0 else int(bundle_top_k)
        self.reliability_feature_name = reliability_feature_name

        self.n_bundles = len(bundle_feature_lists)
        self.expert_scale = int(expert_scale)
        self.n_experts = self.n_bundles * self.expert_scale

        self.expert_names = _scaled_bundle_expert_names(bundle_names, self.expert_scale)

        stage_idx = build_stage_indices(stage_all_features, col2idx)
        self.register_buffer("stage_feat_idx", torch.tensor(stage_idx, dtype=torch.long), persistent=False)

        bundle_idx_lists = build_expert_indices(
            OrderedDict(zip(bundle_names, bundle_feature_lists)),
            col2idx,
        )
        self.bundle_feature_dims: List[int] = []
        for i, idx in enumerate(bundle_idx_lists):
            self.register_buffer(
                f"bundle_feat_idx_{i}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )
            self.bundle_feature_dims.append(len(idx))

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
        self.bundle_feat_proj = nn.ModuleList(
            [nn.Linear(fd, d_feat_emb) for fd in self.bundle_feature_dims]
        )
        self.router_feat_drop = nn.Dropout(router_feature_dropout)

        router_in_dim = 0
        if self.router_use_hidden:
            router_in_dim += d_model
        if self.router_use_feature:
            router_in_dim += d_feat_emb

        self.bundle_router = Router(
            d_in=router_in_dim,
            n_experts=self.n_bundles,
            d_hidden=d_router_hidden,
            top_k=self.bundle_top_k,
            dropout=dropout,
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
                for _ in range(self.n_bundles)
            ]
        )

        self.experts = nn.ModuleList(
            [HiddenExpertMLP(d_model=d_model, d_hidden=d_expert_hidden, dropout=dropout)
             for _ in range(self.n_experts)]
        )

        self.resid_drop = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # Runtime schedule states
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

    def _build_router_input(
        self,
        h_norm: torch.Tensor,
        feat_emb: torch.Tensor,
    ) -> torch.Tensor:
        inputs = []
        if self.router_use_hidden:
            inputs.append(h_norm)
        if self.router_use_feature:
            inputs.append(feat_emb)
        return inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

    def _bundle_feature_tensor(self, feat: torch.Tensor, bundle_idx: int) -> torch.Tensor:
        idx = getattr(self, f"bundle_feat_idx_{bundle_idx}")
        return feat.index_select(-1, idx)

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run stage and return updated hidden + routing tensors.

        Returns:
            next_hidden    : [B, T, D]
            gate_weights   : [B, T, K] (K=4*expert_scale)
            gate_logits    : [B, T, K]
            bundle_weights : [B, T, 4]
            bundle_logits  : [B, T, 4]
            stage_delta    : [B, T, D]
        """
        bsz, tlen, _ = hidden.shape
        h_norm = self.pre_ln(hidden)

        stage_feat = feat.index_select(-1, self.stage_feat_idx)            # [B, T, F_stage]
        stage_feat_emb = self.stage_feat_proj(stage_feat)                  # [B, T, d_feat_emb]

        if self.reliability_feat_idx is not None:
            rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
            stage_feat_emb = stage_feat_emb * rel
        stage_feat_emb = self.router_feat_drop(stage_feat_emb)

        bundle_router_in = self._build_router_input(h_norm, stage_feat_emb)
        bundle_weights, bundle_logits = self.bundle_router(
            bundle_router_in,
            temperature=self.current_router_temperature,
            top_k=self.bundle_top_k,
        )  # [B, T, 4]

        intra_weights_list: List[torch.Tensor] = []
        intra_logits_list: List[torch.Tensor] = []
        for bidx in range(self.n_bundles):
            bundle_feat = self._bundle_feature_tensor(feat, bidx)
            bundle_feat_emb = self.bundle_feat_proj[bidx](bundle_feat)
            if self.reliability_feat_idx is not None:
                rel = feat[..., self.reliability_feat_idx].clamp(0.0, 1.0).unsqueeze(-1)
                bundle_feat_emb = bundle_feat_emb * rel
            bundle_feat_emb = self.router_feat_drop(bundle_feat_emb)

            intra_in = self._build_router_input(h_norm, bundle_feat_emb)
            iw, il = self.intra_routers[bidx](
                intra_in,
                temperature=self.current_router_temperature,
                top_k=self.current_expert_top_k,
            )  # [B, T, E]
            intra_weights_list.append(iw)
            intra_logits_list.append(il)

        # [B, T, 4, E]
        intra_weights = torch.stack(intra_weights_list, dim=-2)
        intra_logits = torch.stack(intra_logits_list, dim=-2)

        # w_{b,e} = pi_b * rho_{b,e}
        final_weights = bundle_weights.unsqueeze(-1) * intra_weights
        gate_weights = final_weights.reshape(bsz, tlen, self.n_experts)

        # logit proxy (used for analysis/logging only)
        final_logits = bundle_logits.unsqueeze(-1) + intra_logits
        gate_logits = final_logits.reshape(bsz, tlen, self.n_experts)

        expert_outputs = torch.stack([expert(h_norm) for expert in self.experts], dim=-2)
        stage_delta = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)

        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_delta)
        return next_hidden, gate_weights, gate_logits, bundle_weights, bundle_logits, stage_delta


class HierarchicalMoEHiR(nn.Module):
    """3-stage HiR MoE wrapper with serial/parallel stage merge."""

    def __init__(
        self,
        d_model: int = 128,
        d_feat_emb: int = 64,
        d_expert_hidden: int = 64,
        d_router_hidden: int = 64,
        expert_scale: int = 1,
        top_k: Optional[int] = None,
        bundle_top_k: Optional[int] = None,
        parallel_stage_gate_top_k: Optional[int] = None,
        dropout: float = 0.1,
        use_macro: bool = True,
        use_mid: bool = True,
        use_micro: bool = True,
        router_use_hidden: bool = True,
        router_use_feature: bool = True,
        stage_merge_mode: str = "serial",
        mid_router_temperature: float = 1.3,
        micro_router_temperature: float = 1.3,
        mid_router_feature_dropout: float = 0.1,
        micro_router_feature_dropout: float = 0.1,
        use_valid_ratio_gating: bool = True,
        hir_use_bundle_aux_loss: bool = True,
        hir_bundle_aux_lambda_scale: float = 1.0,
    ):
        super().__init__()

        mode = str(stage_merge_mode).lower()
        if mode not in ("serial", "parallel"):
            raise ValueError(f"stage_merge_mode must be one of ['serial','parallel'], got {stage_merge_mode}")

        self.stage_merge_mode = mode
        self.router_use_hidden = bool(router_use_hidden)
        self.router_use_feature = bool(router_use_feature)
        self.parallel_stage_gate_top_k = (
            None if parallel_stage_gate_top_k is None or int(parallel_stage_gate_top_k) <= 0
            else int(parallel_stage_gate_top_k)
        )
        self.hir_use_bundle_aux_loss = bool(hir_use_bundle_aux_loss)
        self.hir_bundle_aux_lambda_scale = float(hir_bundle_aux_lambda_scale)

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

            bundle_names = list(expert_dict.keys())
            bundle_feature_lists = list(expert_dict.values())

            stage_module = HierarchicalStageMoE(
                stage_name=stage_name,
                bundle_names=bundle_names,
                bundle_feature_lists=bundle_feature_lists,
                stage_all_features=STAGE_ALL_FEATURES[stage_name],
                col2idx=col2idx,
                d_model=d_model,
                d_feat_emb=d_feat_emb,
                d_expert_hidden=d_expert_hidden,
                d_router_hidden=d_router_hidden,
                expert_scale=expert_scale,
                expert_top_k=top_k,
                bundle_top_k=bundle_top_k,
                dropout=dropout,
                router_use_hidden=router_use_hidden,
                router_use_feature=router_use_feature,
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
            raise ValueError(f"stage '{stage_name}' is not active in this HierarchicalMoEHiR instance")
        stage_module = getattr(self, f"{stage_name}_stage")
        return stage_module(hidden, feat, item_seq_len=item_seq_len)

    def _build_stage_merge_input(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
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
        """Merge stage deltas with a learned stage-gate in parallel mode.

        Returns:
            merged_hidden: [B, T, D]
            stage_weights: [B, T, S]
            stage_logits:  [B, T, S]
        """
        if self.n_active == 0:
            z = torch.zeros(hidden.shape[0], hidden.shape[1], 0, device=hidden.device, dtype=hidden.dtype)
            return hidden, z, z

        ordered_deltas = [stage_deltas[s] for s in self.active_stages]
        if self.n_active == 1:
            stage_weights = torch.ones(
                hidden.shape[0], hidden.shape[1], 1,
                device=hidden.device,
                dtype=hidden.dtype,
            )
            stage_logits = torch.zeros_like(stage_weights)
            merged_hidden = hidden + ordered_deltas[0]
            return merged_hidden, stage_weights, stage_logits

        if self.stage_merge_router is None:
            # Fallback should not happen for n_active>=2 in parallel mode,
            # but keep deterministic behavior if module construction changed.
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
            temperature=1.0,
            top_k=self.parallel_stage_gate_top_k,
        )

        stacked = torch.stack(ordered_deltas, dim=-2)  # [B, T, S, D]
        merged_delta = (stage_weights.unsqueeze(-1) * stacked).sum(dim=-2)
        merged_hidden = hidden + merged_delta
        return merged_hidden, stage_weights, stage_logits

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

        flat = gate_weights[valid]  # [N_valid, K]
        return load_balance_loss(flat, n_experts=n_experts)

    def compute_aux_loss(
        self,
        weights: Dict[str, torch.Tensor],
        bundle_weights: Optional[Dict[str, torch.Tensor]] = None,
        item_seq_len: Optional[torch.Tensor] = None,
        balance_lambda: float = 0.01,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        expert_total = torch.tensor(0.0, device=device)
        for w in weights.values():
            expert_total = expert_total + self._masked_load_balance(w, item_seq_len=item_seq_len)

        total = balance_lambda * expert_total

        if self.hir_use_bundle_aux_loss and bundle_weights:
            bundle_total = torch.tensor(0.0, device=device)
            for bw in bundle_weights.values():
                bundle_total = bundle_total + self._masked_load_balance(bw, item_seq_len=item_seq_len)
            total = total + (self.hir_bundle_aux_lambda_scale * balance_lambda * bundle_total)

        return total
