"""Stage modules for FeaturedMoE_N2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..FeaturedMoE.routers import Router
from ..FeaturedMoE.transformer import TransformerEncoder
from ..FeaturedMoE_N.stage_modules import (
    NExpertGroup,
    NMoEStage as BaseNMoEStage,
    StageRuntimeConfig,
    _build_inter_block,
    _build_moe_replacement_block,
    _normalize_top_k,
    _softmax_with_top_k,
)


@dataclass
class StageRuntimeConfigN2(StageRuntimeConfig):
    router_feature_proj_dim: int = 0
    router_feature_proj_layers: int = 1
    router_feature_scale: float = 1.0
    router_hidden_scale: float = 1.0
    router_group_feature_scale: float = 1.0


def _build_feature_projector(
    d_in: int,
    d_out: int,
    *,
    layers: int,
    dropout: float,
) -> nn.Module:
    d_in = int(d_in)
    d_out = int(d_out)
    if d_in <= 0 or d_out <= 0:
        raise ValueError(f"Invalid router feature projection dims: {d_in} -> {d_out}")
    layers = max(int(layers), 1)
    if layers == 1:
        return nn.Linear(d_in, d_out)
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(d_out, d_out),
    )


class NMoEStageN2(BaseNMoEStage):
    """N2 stage with feature-heavy learned router paths and extra aux signals."""

    def __init__(self, cfg: StageRuntimeConfigN2):
        self.router_feature_proj_dim = max(int(cfg.router_feature_proj_dim), 0)
        self.router_feature_proj_layers = max(int(cfg.router_feature_proj_layers), 1)
        self.router_feature_scale = float(cfg.router_feature_scale)
        self.router_hidden_scale = float(cfg.router_hidden_scale)
        self.router_group_feature_scale = float(cfg.router_group_feature_scale)
        super().__init__(cfg)

        self.n_base_groups = max(len(self.base_names), 1)
        expert_group_idx = [idx // max(self.expert_scale, 1) for idx in range(self.n_experts)]
        self.register_buffer(
            "expert_group_idx",
            torch.tensor(expert_group_idx, dtype=torch.long),
            persistent=False,
        )

        raw_feature_dim = len(self.stage_all_features) * self.feature_bank_dim if self.router_use_feature else 0
        self.router_feature_proj = None
        projected_feature_dim = raw_feature_dim
        if self.router_use_feature and self.router_feature_proj_dim > 0:
            self.router_feature_proj = _build_feature_projector(
                raw_feature_dim,
                self.router_feature_proj_dim,
                layers=self.router_feature_proj_layers,
                dropout=float(cfg.dropout),
            )
            projected_feature_dim = self.router_feature_proj_dim

        router_group_dim = 0
        if self.router_group_feature_mode == "mean":
            router_group_dim = len(self._router_group_local_idx)
        elif self.router_group_feature_mode == "mean_std":
            router_group_dim = 2 * len(self._router_group_local_idx)
        router_in_dim = 0
        if self.router_use_hidden:
            router_in_dim += int(cfg.d_model)
        if self.router_use_feature:
            router_in_dim += int(projected_feature_dim)
        if router_group_dim > 0:
            router_in_dim += int(router_group_dim)

        if self.router_impl == "learned":
            self.learned_router = Router(
                d_in=router_in_dim,
                n_experts=self.n_experts,
                d_hidden=int(cfg.d_router_hidden),
                top_k=None,
                dropout=float(cfg.dropout),
            )

    def _project_router_feature_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.router_feature_proj is None:
            return tensor
        return self.router_feature_proj(tensor)

    def _aggregate_group_weights(self, gate_weights: torch.Tensor) -> torch.Tensor:
        pieces = []
        for group_idx in range(self.n_base_groups):
            mask = self.expert_group_idx == group_idx
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=False).view(-1).to(device=gate_weights.device)
            pieces.append(gate_weights.index_select(-1, idx).sum(dim=-1, keepdim=True))
        return torch.cat(pieces, dim=-1) if pieces else gate_weights.new_zeros(*gate_weights.shape[:-1], 0)

    def _build_router_inputs(
        self,
        *,
        h_norm: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_raw_feat = feat.index_select(-1, self.stage_feat_idx)
        stage_bank_flat = self._stage_bank_flat(feat_bank)
        stage_raw_feat, stage_bank_flat = self._apply_reliability(stage_raw_feat, stage_bank_flat, feat)
        stage_bank_flat = self.router_feat_drop(stage_bank_flat)
        if self.router_use_feature:
            stage_bank_flat = self._project_router_feature_tensor(stage_bank_flat) * self.router_feature_scale
        group_router_feat = self._group_router_features(stage_raw_feat)
        if group_router_feat is not None:
            group_router_feat = group_router_feat * self.router_group_feature_scale

        if self.router_mode == "session":
            pooled_parts = []
            if self.router_use_hidden:
                pooled_parts.append(self._pool_sequence(h_norm, valid_mask, item_seq_len) * self.router_hidden_scale)
            if self.router_use_feature:
                pooled_parts.append(self._pool_sequence(stage_bank_flat, valid_mask, item_seq_len))
            if group_router_feat is not None:
                pooled_parts.append(self._pool_sequence(group_router_feat, valid_mask, item_seq_len))
            router_input = pooled_parts[0] if len(pooled_parts) == 1 else torch.cat(pooled_parts, dim=-1)
            rule_features = self._pool_sequence(stage_raw_feat, valid_mask, item_seq_len)
            return router_input, rule_features

        router_parts = []
        if self.router_use_hidden:
            router_parts.append(h_norm * self.router_hidden_scale)
        if self.router_use_feature:
            router_parts.append(stage_bank_flat)
        if group_router_feat is not None:
            router_parts.append(group_router_feat)
        router_input = router_parts[0] if len(router_parts) == 1 else torch.cat(router_parts, dim=-1)
        return router_input, stage_raw_feat

    def _compute_router_outputs(
        self,
        *,
        h_norm: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        valid_mask: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        router_input, rule_features = self._build_router_inputs(
            h_norm=h_norm,
            feat=feat,
            feat_bank=feat_bank,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
        )

        rule_logits = self.rule_router.compute_logits(rule_features)
        aux: Dict[str, torch.Tensor] = {}
        if self.router_impl == "rule_soft":
            raw_logits = rule_logits
        else:
            if self.learned_router is None:
                raise RuntimeError("learned router is not initialized.")
            raw_logits = self.learned_router.net(router_input)
            if self.rule_prior_router is not None:
                raw_logits = raw_logits + self.rule_bias_scale * self.rule_prior_router.compute_logits(rule_features)
            aux["rule_target_logits"] = rule_logits / max(float(self.current_router_temperature), 1e-6)

        scaled_logits = raw_logits / max(float(self.current_router_temperature), 1e-6)
        weights = _softmax_with_top_k(scaled_logits, self.current_top_k)
        if self.router_impl == "learned":
            aux["learned_gate_logits"] = scaled_logits
        return weights, scaled_logits, aux

    def forward(
        self,
        hidden: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = hidden.shape
        h_norm = self.pre_ln(hidden)
        valid_mask = self._build_valid_mask(batch_size, seq_len, item_seq_len, hidden.device)

        gate_weights, gate_logits, aux = self._compute_router_outputs(
            h_norm=h_norm,
            feat=feat,
            feat_bank=feat_bank,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
        )
        if self.router_mode == "session":
            gate_weights = gate_weights.unsqueeze(1).expand(-1, seq_len, -1)
            gate_logits = gate_logits.unsqueeze(1).expand(-1, seq_len, -1)
            if "rule_target_logits" in aux:
                aux["rule_target_logits"] = aux["rule_target_logits"].unsqueeze(1).expand(-1, seq_len, -1)
            if "learned_gate_logits" in aux:
                aux["learned_gate_logits"] = aux["learned_gate_logits"].unsqueeze(1).expand(-1, seq_len, -1)

        expert_inputs = self._gather_expert_inputs(feat_bank)
        expert_out = self.expert_group(h_norm, expert_inputs)
        stage_out = (gate_weights.unsqueeze(-1) * expert_out).sum(dim=-2)

        next_hidden = hidden + (self.alpha * self.alpha_scale) * self.resid_drop(stage_out)
        aux["group_weights"] = self._aggregate_group_weights(gate_weights)
        self.last_router_aux = {}
        return next_hidden, stage_out, gate_weights, gate_logits, aux


class StageBranchRunnerN2(nn.Module):
    """Run one explicit pass/MoE branch for FeaturedMoE_N2."""

    def __init__(self, cfg: StageRuntimeConfigN2):
        super().__init__()
        self.stage_name = cfg.stage_name
        self.pass_layers = int(cfg.pass_layers)
        self.moe_blocks = int(cfg.moe_blocks)
        self.inter_layer_style = str(cfg.inter_layer_style).lower().strip()
        self.moe_block_variant = str(cfg.moe_block_variant or "moe").lower().strip()
        if self.moe_block_variant not in {"moe", "dense_ffn", "nonlinear", "identity"}:
            raise ValueError(
                "moe_block_variant must be one of ['moe','dense_ffn','nonlinear','identity'], "
                f"got {cfg.moe_block_variant}"
            )
        self.pass_transformer = None
        self.pass_blocks = nn.ModuleList()

        if self.pass_layers > 0:
            if self.inter_layer_style == "attn":
                self.pass_transformer = TransformerEncoder(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    n_layers=self.pass_layers,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    ffn_moe=False,
                )
            else:
                self.pass_blocks = nn.ModuleList(
                    [_build_inter_block(self.inter_layer_style, cfg) for _ in range(self.pass_layers)]
                )

        if self.moe_blocks > 0:
            self.moe_pre_blocks = nn.ModuleList(
                [_build_inter_block(self.inter_layer_style, cfg) for _ in range(self.moe_blocks)]
            )
            if self.moe_block_variant == "moe":
                self.stage_module = NMoEStageN2(cfg)
                self.moe_replacement_blocks = nn.ModuleList()
            else:
                self.stage_module = None
                self.moe_replacement_blocks = nn.ModuleList(
                    [_build_moe_replacement_block(self.moe_block_variant, cfg) for _ in range(self.moe_blocks)]
                )
        else:
            self.moe_pre_blocks = nn.ModuleList()
            self.stage_module = None
            self.moe_replacement_blocks = nn.ModuleList()

    @property
    def n_experts(self) -> int:
        return 0 if self.stage_module is None else int(self.stage_module.n_experts)

    @property
    def expert_names(self) -> list[str]:
        return [] if self.stage_module is None else list(self.stage_module.expert_names)

    @property
    def group_names(self) -> list[str]:
        return [] if self.stage_module is None else list(self.stage_module.group_names)

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        router_temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if self.stage_module is None:
            return
        self.stage_module.set_schedule_state(
            alpha_scale=alpha_scale,
            router_temperature=router_temperature,
            top_k=top_k,
        )

    def _run_pass_layers(self, hidden: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        if self.pass_transformer is None:
            out = hidden
            for block in self.pass_blocks:
                if isinstance(block, TransformerEncoder):
                    out, _ = block(out, item_seq)
                else:
                    out = block(out, item_seq)
            return out
        out, _ = self.pass_transformer(hidden, item_seq)
        return out

    def forward_serial(
        self,
        *,
        hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        out = self._run_pass_layers(hidden, item_seq)

        weights: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.stage_module is None and len(self.moe_replacement_blocks) == 0:
            return out, weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            if isinstance(pre_block, TransformerEncoder):
                out, _ = pre_block(out, item_seq)
            else:
                out = pre_block(out, item_seq)
            if self.stage_module is not None:
                out, _delta, w, l, aux = self.stage_module(out, feat, feat_bank, item_seq_len=item_seq_len)
                key = f"{self.stage_name}@{idx}"
                weights[key] = w
                logits[key] = l
                for aux_key, aux_tensor in aux.items():
                    router_aux.setdefault(aux_key, {})[key] = aux_tensor
            else:
                out = self.moe_replacement_blocks[idx - 1](out, item_seq)
        return out, weights, logits, router_aux

    def forward_parallel(
        self,
        *,
        base_hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        feat_bank: torch.Tensor,
        item_seq_len: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        out = self._run_pass_layers(base_hidden, item_seq)

        weights: Dict[str, torch.Tensor] = {}
        logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        if self.stage_module is None and len(self.moe_replacement_blocks) == 0:
            return out, (out - base_hidden), weights, logits, router_aux

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            if isinstance(pre_block, TransformerEncoder):
                out, _ = pre_block(out, item_seq)
            else:
                out = pre_block(out, item_seq)
            if self.stage_module is not None:
                out, _delta, w, l, aux = self.stage_module(out, feat, feat_bank, item_seq_len=item_seq_len)
                key = f"{self.stage_name}@{idx}"
                weights[key] = w
                logits[key] = l
                for aux_key, aux_tensor in aux.items():
                    router_aux.setdefault(aux_key, {})[key] = aux_tensor
            else:
                out = self.moe_replacement_blocks[idx - 1](out, item_seq)

        return out, (out - base_hidden), weights, logits, router_aux
