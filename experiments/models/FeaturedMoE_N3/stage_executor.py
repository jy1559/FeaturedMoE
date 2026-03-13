"""layer_layout-based executor for FeaturedMoE_N3."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .stage_modules import (
    N3StageBlock,
    SASRecStyleAttentionBlock,
    SASRecStyleFFNBlock,
    SASRecStyleLayerBlock,
    StageRuntimeConfigN3,
)


_PLAIN_TOKENS = {"layer", "attn", "ffn"}
_STAGE_NAMES = ("macro", "mid", "micro")
_STAGE_TOKENS = {
    "macro",
    "macro_ffn",
    "mid",
    "mid_ffn",
    "micro",
    "micro_ffn",
}
_VALID_TOKENS = _PLAIN_TOKENS | _STAGE_TOKENS


class StageExecutorN3(nn.Module):
    """Execute explicit N3 layer_layout tokens left-to-right."""

    def __init__(
        self,
        *,
        layer_layout: List[str],
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attn_dropout: float,
        hidden_act: str,
        layer_norm_eps: float,
        d_feat_emb: int,
        d_expert_hidden: int,
        d_router_hidden: int,
        expert_depth_by_stage: Dict[str, int],
        expert_hidden_by_stage: Dict[str, int],
        expert_scale: int,
        stage_top_k: Optional[int],
        macro_session_pooling: str,
        stage_router_granularity: Dict[str, str],
        stage_all_features: Dict[str, List[str]],
        stage_family_features: Dict[str, Dict[str, List[str]]],
        stage_feature_encoder_mode: Dict[str, str],
        stage_compute_mode: Dict[str, str],
        stage_router_mode: Dict[str, str],
        stage_router_source: Dict[str, str],
        stage_feature_injection: Dict[str, str],
        rule_router_cfg: Dict[str, object],
        rule_bias_scale: float,
        mid_router_temperature: float,
        micro_router_temperature: float,
        dense_hidden_scale: float,
        col2idx: Dict[str, int],
    ):
        super().__init__()
        self.layer_layout = [str(token).lower().strip() for token in list(layer_layout or [])]
        if not self.layer_layout:
            raise ValueError("layer_layout cannot be empty.")
        unknown = [token for token in self.layer_layout if token not in _VALID_TOKENS]
        if unknown:
            raise ValueError(f"Unsupported layer_layout tokens: {unknown}")

        self.layer_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        self.ffn_blocks = nn.ModuleList()
        self.stage_attn_blocks = nn.ModuleList()
        self.stage_blocks = nn.ModuleDict()
        self.compiled_ops: List[dict] = []

        self._stage_counts = defaultdict(int)
        self._stage_all_features = {stage: list(stage_all_features.get(stage, [])) for stage in _STAGE_NAMES}
        self._stage_family_features = {
            stage: {name: list(cols) for name, cols in dict(stage_family_features.get(stage, {}) or {}).items()}
            for stage in _STAGE_NAMES
        }

        for stage_name in _STAGE_NAMES:
            if stage_name not in self.stage_blocks:
                router_temperature = 1.0
                if stage_name == "mid":
                    router_temperature = float(mid_router_temperature)
                elif stage_name == "micro":
                    router_temperature = float(micro_router_temperature)
                feature_names = self._stage_all_features.get(stage_name, [])
                cfg = StageRuntimeConfigN3(
                    stage_name=stage_name,
                    d_model=d_model,
                    d_ff=d_ff,
                    d_feat_emb=d_feat_emb,
                    d_expert_hidden=int(expert_hidden_by_stage.get(stage_name, d_expert_hidden)),
                    d_router_hidden=d_router_hidden,
                    expert_depth=int(expert_depth_by_stage.get(stage_name, 1)),
                    expert_scale=expert_scale,
                    top_k=stage_top_k,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    stage_feature_indices=tuple(
                        int(col2idx[name]) for name in feature_names if name in col2idx
                    ),
                    stage_feature_names=tuple(name for name in feature_names if name in col2idx),
                    stage_family_features=self._stage_family_features.get(stage_name, {}),
                    stage_feature_encoder_mode=str(stage_feature_encoder_mode.get(stage_name, "linear")),
                    stage_compute_mode=str(stage_compute_mode.get(stage_name, "moe")),
                    stage_router_mode=str(stage_router_mode.get(stage_name, "learned")),
                    stage_router_source=str(stage_router_source.get(stage_name, "both")),
                    stage_feature_injection=str(stage_feature_injection.get(stage_name, "none")),
                    routing_granularity=str(stage_router_granularity.get(stage_name, "token")),
                    session_pooling=str(macro_session_pooling),
                    rule_router_cfg=dict(rule_router_cfg or {}),
                    rule_bias_scale=float(rule_bias_scale),
                    router_temperature=router_temperature,
                    dense_hidden_scale=float(dense_hidden_scale),
                )
                self.stage_blocks[stage_name] = N3StageBlock(cfg)

        op_idx = 0
        for token in self.layer_layout:
            if token == "layer":
                op_idx += 1
                self.layer_blocks.append(
                    SASRecStyleLayerBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        d_ff=d_ff,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        hidden_act=hidden_act,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "layer", "index": len(self.layer_blocks) - 1, "name": f"{op_idx:02d}_layer"})
                continue
            if token == "attn":
                op_idx += 1
                self.attn_blocks.append(
                    SASRecStyleAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "attn", "index": len(self.attn_blocks) - 1, "name": f"{op_idx:02d}_attn"})
                continue
            if token == "ffn":
                op_idx += 1
                self.ffn_blocks.append(
                    SASRecStyleFFNBlock(
                        d_model=d_model,
                        d_ff=d_ff,
                        hidden_dropout_prob=dropout,
                        hidden_act=hidden_act,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append({"kind": "ffn", "index": len(self.ffn_blocks) - 1, "name": f"{op_idx:02d}_ffn"})
                continue

            stage_name = token.replace("_ffn", "")
            if token == stage_name:
                op_idx += 1
                self.stage_attn_blocks.append(
                    SASRecStyleAttentionBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        hidden_dropout_prob=dropout,
                        attn_dropout_prob=attn_dropout,
                        layer_norm_eps=layer_norm_eps,
                    )
                )
                self.compiled_ops.append(
                    {
                        "kind": "stage_attn",
                        "index": len(self.stage_attn_blocks) - 1,
                        "stage": stage_name,
                        "name": f"{op_idx:02d}_{token}.attn",
                    }
                )
            self._stage_counts[stage_name] += 1
            self.compiled_ops.append(
                {
                    "kind": "stage",
                    "stage": stage_name,
                    "stage_key": f"{stage_name}@{self._stage_counts[stage_name]}",
                    "name": f"{op_idx:02d}_{token}.ffn",
                }
            )

        self.requires_features = any(
            op["kind"] == "stage" and self.stage_blocks[op["stage"]].requires_features
            for op in self.compiled_ops
        )
        self.supports_diagnostics = any(
            op["kind"] == "stage" and self.stage_blocks[op["stage"]].supports_diagnostics
            for op in self.compiled_ops
        )

    def stage_n_experts(self) -> int:
        return max((int(self.stage_blocks[name].n_experts) for name in _STAGE_NAMES), default=0)

    def stage_expert_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.stage_blocks[stage_name].expert_names)
            for stage_name in _STAGE_NAMES
            if self.stage_blocks[stage_name].supports_diagnostics
        }

    def stage_group_names(self) -> Dict[str, list[str]]:
        return {
            stage_name: list(self.stage_blocks[stage_name].group_names)
            for stage_name in _STAGE_NAMES
            if self.stage_blocks[stage_name].supports_diagnostics
        }

    def set_schedule_state(
        self,
        *,
        alpha_scale: Optional[float] = None,
        stage_temperatures: Optional[Dict[str, float]] = None,
        top_k: Optional[int] = None,
    ) -> None:
        _ = alpha_scale
        stage_temperatures = dict(stage_temperatures or {})
        for stage_name in _STAGE_NAMES:
            self.stage_blocks[stage_name].set_schedule_state(
                router_temperature=stage_temperatures.get(stage_name),
                top_k=top_k,
            )

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        feat: Optional[torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, Dict[str, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        out = hidden
        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        router_aux: Dict[str, Dict[str, torch.Tensor]] = {}
        dense_aux: Dict[str, Dict[str, torch.Tensor]] = {}

        for op in self.compiled_ops:
            kind = op["kind"]
            if kind == "layer":
                out = self.layer_blocks[op["index"]](out, attention_mask)
                continue
            if kind == "attn":
                out = self.attn_blocks[op["index"]](out, attention_mask)
                continue
            if kind == "ffn":
                out = self.ffn_blocks[op["index"]](out)
                continue
            if kind == "stage_attn":
                out = self.stage_attn_blocks[op["index"]](out, attention_mask)
                continue

            stage_key = op["stage_key"]
            next_hidden, weights, logits, stage_router_aux, stage_dense_aux = self.stage_blocks[op["stage"]](
                out,
                feat,
                item_seq_len=item_seq_len,
            )
            out = next_hidden
            if weights is not None:
                gate_weights[stage_key] = weights
            if logits is not None:
                gate_logits[stage_key] = logits
            for aux_name, aux_tensor in dict(stage_router_aux or {}).items():
                router_aux.setdefault(aux_name, {})[stage_key] = aux_tensor
            if stage_dense_aux:
                dense_aux[stage_key] = dict(stage_dense_aux)

        return out, gate_weights, gate_logits, router_aux, dense_aux, None, None
