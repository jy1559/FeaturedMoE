"""FeaturedMoE_HiR2 model.

Stage-first two-step gating:
1) Session-level stage allocator (macro/mid/micro ratios).
2) Token-level expert routing inside each stage.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from ..FeaturedMoE.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGES,
    build_column_to_index,
    feature_list_field,
)
from ..FeaturedMoE.routers import load_balance_loss
from ..FeaturedMoE.transformer import TransformerEncoder
from .hir2_modules import StageAllocator, StageExpertBlock

logger = logging.getLogger(__name__)

_STAGE_NAMES = ("macro", "mid", "micro")


class FeaturedMoE_HiR2(SequentialRecommender):
    """Stage-first hierarchical routing model."""

    input_type = "point"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(key, default):
            return config[key] if key in config else default

        # ---- Core dimensions ----
        self.n_items = dataset.item_num
        self.d_model = int(_cfg("embedding_size", 128))
        self.d_feat_emb = int(_cfg("d_feat_emb", 16))
        self.d_expert_hidden = int(_cfg("d_expert_hidden", 128))
        self.d_router_hidden = int(_cfg("d_router_hidden", 64))
        self.expert_scale = int(_cfg("expert_scale", 2))
        self.n_heads = int(_cfg("num_heads", 8))
        self.d_ff = int(_cfg("d_ff", 0) or (4 * self.d_model))
        self.dropout = float(_cfg("hidden_dropout_prob", 0.1))
        self.max_seq_length = int(config["MAX_ITEM_LIST_LENGTH"])

        # ---- Merge / allocator ----
        merge_mode = str(_cfg("hir2_stage_merge_mode", "serial_weighted")).lower().strip()
        if merge_mode == "serial":
            merge_mode = "serial_weighted"
        elif merge_mode == "parallel":
            merge_mode = "parallel_weighted"
        if merge_mode not in {"serial_weighted", "parallel_weighted"}:
            raise ValueError(
                f"hir2_stage_merge_mode must be one of ['serial_weighted','parallel_weighted'], got {merge_mode}"
            )
        self.stage_merge_mode = merge_mode

        alloc_top_k_raw = int(_cfg("hir2_stage_allocator_top_k", 0))
        self.stage_allocator_top_k = None if alloc_top_k_raw <= 0 else alloc_top_k_raw
        self.stage_allocator_temperature = float(_cfg("hir2_stage_allocator_temperature", 1.0))
        self.stage_allocator_pooling = str(_cfg("hir2_stage_allocator_pooling", "query")).lower().strip()
        self.stage_allocator_use_hidden = bool(_cfg("hir2_stage_allocator_use_hidden", True))
        self.stage_allocator_use_feature = bool(_cfg("hir2_stage_allocator_use_feature", True))

        # ---- Stage router / expert toggles ----
        self.router_use_hidden = bool(_cfg("router_use_hidden", True))
        self.router_use_feature = bool(_cfg("router_use_feature", True))
        self.expert_use_hidden = bool(_cfg("expert_use_hidden", True))
        self.expert_use_feature = bool(_cfg("expert_use_feature", True))
        if not (self.router_use_hidden or self.router_use_feature):
            raise ValueError("router_use_hidden and router_use_feature cannot both be false.")
        if not (self.expert_use_hidden or self.expert_use_feature):
            raise ValueError("expert_use_hidden and expert_use_feature cannot both be false.")

        raw_top_k = int(_cfg("moe_top_k", 0))
        self.moe_top_k = None if raw_top_k <= 0 else raw_top_k
        self.macro_router_temperature = float(_cfg("macro_router_temperature", 1.0))
        self.mid_router_temperature = float(_cfg("mid_router_temperature", 1.3))
        self.micro_router_temperature = float(_cfg("micro_router_temperature", 1.3))
        self.mid_router_feature_dropout = float(_cfg("mid_router_feature_dropout", 0.1))
        self.micro_router_feature_dropout = float(_cfg("micro_router_feature_dropout", 0.1))
        self.use_valid_ratio_gating = bool(_cfg("use_valid_ratio_gating", True))

        # ---- Pre/post blocks ----
        self.global_pre_layers = max(int(_cfg("hir2_global_pre_layers", 0)), 0)
        self.global_post_layers = max(int(_cfg("hir2_global_post_layers", 0)), 0)
        self.stage_pre_depths = {
            "macro": max(int(_cfg("hir2_macro_pre_layers", 0)), 0),
            "mid": max(int(_cfg("hir2_mid_pre_layers", 0)), 0),
            "micro": max(int(_cfg("hir2_micro_pre_layers", 0)), 0),
        }

        # ---- Loss ----
        self.use_aux_loss = bool(_cfg("use_aux_loss", True))
        self.balance_loss_lambda = float(_cfg("balance_loss_lambda", 0.01))
        self.hir2_stage_alloc_aux_scale = float(_cfg("hir2_stage_alloc_aux_scale", 1.0))
        self.hir2_stage_entropy_aux_lambda = max(float(_cfg("hir2_stage_entropy_aux_lambda", 0.0)), 0.0)
        self.hir2_stage_weight_floor = float(_cfg("hir2_stage_weight_floor", 0.0))
        self.hir2_stage_weight_floor = min(max(self.hir2_stage_weight_floor, 0.0), 0.95)
        self.hir2_stage_delta_scale = max(float(_cfg("hir2_stage_delta_scale", 1.0)), 0.0)

        # ---- Optional FFN-MoE ----
        self.ffn_moe = bool(_cfg("ffn_moe", False))
        self.n_ffn_experts = int(_cfg("n_ffn_experts", 4))
        raw_ffn_top_k = int(_cfg("ffn_top_k", 0))
        self.ffn_top_k = None if raw_ffn_top_k <= 0 else raw_ffn_top_k

        # ---- Features ----
        self.feature_fields = [feature_list_field(col) for col in ALL_FEATURE_COLUMNS]
        self.n_features = len(self.feature_fields)
        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)

        # ---- Embeddings ----
        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        # ---- Global pre/post ----
        if self.global_pre_layers > 0:
            self.pre_transformer = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.global_pre_layers,
                d_ff=self.d_ff,
                dropout=self.dropout,
                ffn_moe=False,
            )
        else:
            self.pre_transformer = None

        self.stage_pre_transformers = nn.ModuleDict()
        for stage_name in _STAGE_NAMES:
            depth = self.stage_pre_depths[stage_name]
            if depth <= 0:
                continue
            self.stage_pre_transformers[stage_name] = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=depth,
                d_ff=self.d_ff,
                dropout=self.dropout,
                ffn_moe=False,
            )

        self.stage_allocator = StageAllocator(
            d_model=self.d_model,
            n_features=self.n_features,
            d_feat_emb=self.d_feat_emb,
            d_hidden=self.d_router_hidden,
            dropout=self.dropout,
            top_k=self.stage_allocator_top_k,
            temperature=self.stage_allocator_temperature,
            use_hidden=self.stage_allocator_use_hidden,
            use_feature=self.stage_allocator_use_feature,
            pooling=self.stage_allocator_pooling,
        )

        self.stage_modules = nn.ModuleDict()
        for stage_name, expert_dict in STAGES:
            stage_temp = self.macro_router_temperature
            stage_feat_drop = 0.0
            reliability_name = None
            if stage_name == "mid":
                stage_temp = self.mid_router_temperature
                stage_feat_drop = self.mid_router_feature_dropout
                if self.use_valid_ratio_gating:
                    reliability_name = "mid_valid_r"
            elif stage_name == "micro":
                stage_temp = self.micro_router_temperature
                stage_feat_drop = self.micro_router_feature_dropout
                if self.use_valid_ratio_gating:
                    reliability_name = "mic_valid_r"

            self.stage_modules[stage_name] = StageExpertBlock(
                stage_name=stage_name,
                expert_feature_lists=list(expert_dict.values()),
                expert_names=list(expert_dict.keys()),
                col2idx=col2idx,
                d_model=self.d_model,
                d_feat_emb=self.d_feat_emb,
                d_expert_hidden=self.d_expert_hidden,
                d_router_hidden=self.d_router_hidden,
                expert_scale=self.expert_scale,
                top_k=self.moe_top_k,
                dropout=self.dropout,
                router_use_hidden=self.router_use_hidden,
                router_use_feature=self.router_use_feature,
                expert_use_hidden=self.expert_use_hidden,
                expert_use_feature=self.expert_use_feature,
                router_temperature=stage_temp,
                router_feature_dropout=stage_feat_drop,
                reliability_feature_name=reliability_name,
            )

        self.post_transformer = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.global_post_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            ffn_moe=self.ffn_moe,
            n_ffn_experts=self.n_ffn_experts,
            ffn_top_k=self.ffn_top_k,
        )

        self.apply(self._init_weights)

        try:
            config["hir2_stage_merge_mode"] = self.stage_merge_mode
            config["hir2_stage_allocator_top_k"] = -1 if self.stage_allocator_top_k is None else int(self.stage_allocator_top_k)
            config["hir2_stage_allocator_temperature"] = float(self.stage_allocator_temperature)
            config["hir2_stage_alloc_aux_scale"] = float(self.hir2_stage_alloc_aux_scale)
            config["hir2_stage_entropy_aux_lambda"] = float(self.hir2_stage_entropy_aux_lambda)
            config["hir2_stage_weight_floor"] = float(self.hir2_stage_weight_floor)
            config["hir2_stage_delta_scale"] = float(self.hir2_stage_delta_scale)
        except Exception:
            pass

        logger.info(
            "FeaturedMoE_HiR2 init: merge_mode=%s alloc_top_k=%s alloc_temp=%.3f alloc_pool=%s "
            "d_model=%s d_feat=%s d_exp=%s d_router=%s expert_scale=%s moe_top_k=%s "
            "global_pre=%s global_post=%s stage_pre=%s aux=%s stage_alloc_aux_scale=%.3f "
            "stage_entropy_aux=%.5f stage_weight_floor=%.3f stage_delta_scale=%.3f",
            self.stage_merge_mode,
            self.stage_allocator_top_k,
            self.stage_allocator_temperature,
            self.stage_allocator_pooling,
            self.d_model,
            self.d_feat_emb,
            self.d_expert_hidden,
            self.d_router_hidden,
            self.expert_scale,
            self.moe_top_k,
            self.global_pre_layers,
            self.global_post_layers,
            self.stage_pre_depths,
            self.use_aux_loss,
            self.hir2_stage_alloc_aux_scale,
            self.hir2_stage_entropy_aux_lambda,
            self.hir2_stage_weight_floor,
            self.hir2_stage_delta_scale,
        )

    @staticmethod
    def _masked_load_balance(gate_weights: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
        if item_seq_len is None:
            return load_balance_loss(gate_weights, n_experts=gate_weights.shape[-1])
        _, tlen, n_experts = gate_weights.shape
        lens = item_seq_len.to(device=gate_weights.device).long().clamp(min=1, max=tlen)
        valid = torch.arange(tlen, device=gate_weights.device).unsqueeze(0) < lens.unsqueeze(1)
        if not valid.any():
            return torch.tensor(0.0, device=gate_weights.device)
        flat = gate_weights[valid]
        return load_balance_loss(flat, n_experts=n_experts)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _gather_features(self, interaction) -> torch.Tensor:
        item_seq = interaction[self.ITEM_SEQ]
        bsz, tlen = item_seq.shape
        feat_list = []
        for field in self.feature_fields:
            if field in interaction:
                feat_list.append(interaction[field].float())
            else:
                feat_list.append(torch.zeros(bsz, tlen, device=item_seq.device))
                logger.warning("Feature field '%s' not found - using zeros.", field)
        return torch.stack(feat_list, dim=-1)

    def _apply_stage_pre(self, stage_name: str, hidden: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        if stage_name not in self.stage_pre_transformers:
            return hidden
        out, _ = self.stage_pre_transformers[stage_name](hidden, item_seq)
        return out

    def _stabilize_stage_alloc(self, stage_alloc_w: torch.Tensor) -> torch.Tensor:
        if self.hir2_stage_weight_floor <= 0:
            return stage_alloc_w
        n_stage = int(stage_alloc_w.shape[-1])
        floor_each = self.hir2_stage_weight_floor / float(max(n_stage, 1))
        return stage_alloc_w * (1.0 - self.hir2_stage_weight_floor) + floor_each

    def forward(self, item_seq, item_seq_len, feat):
        bsz, tlen = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(tlen, device=item_seq.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)
        if self.pre_transformer is not None:
            tokens, _ = self.pre_transformer(tokens, item_seq)

        stage_alloc_w, stage_alloc_l = self.stage_allocator(
            hidden=tokens,
            feat=feat,
            item_seq_len=item_seq_len,
        )
        stage_alloc_w = self._stabilize_stage_alloc(stage_alloc_w)

        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}

        if self.stage_merge_mode == "serial_weighted":
            out = tokens
            for sid, stage_name in enumerate(_STAGE_NAMES):
                stage_hidden = self._apply_stage_pre(stage_name, out, item_seq)
                stage_delta, w, l = self.stage_modules[stage_name](
                    stage_hidden,
                    feat,
                    item_seq_len=item_seq_len,
                )
                stage_weight = stage_alloc_w[:, sid].view(-1, 1, 1).to(dtype=out.dtype)
                out = out + self.hir2_stage_delta_scale * stage_weight * stage_delta
                gate_weights[stage_name] = w
                gate_logits[stage_name] = l
            tokens = out
        else:
            base = tokens
            total_delta = torch.zeros_like(base)
            for sid, stage_name in enumerate(_STAGE_NAMES):
                stage_hidden = self._apply_stage_pre(stage_name, base, item_seq)
                stage_delta, w, l = self.stage_modules[stage_name](
                    stage_hidden,
                    feat,
                    item_seq_len=item_seq_len,
                )
                stage_weight = stage_alloc_w[:, sid].view(-1, 1, 1).to(dtype=base.dtype)
                total_delta = total_delta + self.hir2_stage_delta_scale * stage_weight * stage_delta
                gate_weights[stage_name] = w
                gate_logits[stage_name] = l
            tokens = base + total_delta

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)
        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "stage_allocation_weights": stage_alloc_w,
            "stage_allocation_logits": stage_alloc_l,
            "ffn_moe_weights": ffn_moe_weights,
        }
        return seq_output, aux_data

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, aux_data = self.forward(item_seq, item_seq_len, feat)

        logits = seq_output @ self.item_embedding.weight.T
        ce_loss = F.cross_entropy(logits, pos_items)

        aux_loss = torch.tensor(0.0, device=ce_loss.device)
        if self.use_aux_loss and self.balance_loss_lambda > 0:
            for w in aux_data["gate_weights"].values():
                aux_loss = aux_loss + self.balance_loss_lambda * self._masked_load_balance(
                    w,
                    item_seq_len=item_seq_len,
                )

            stage_alloc = aux_data["stage_allocation_weights"]
            aux_loss = aux_loss + (
                self.balance_loss_lambda
                * self.hir2_stage_alloc_aux_scale
                * load_balance_loss(stage_alloc, n_experts=stage_alloc.shape[-1])
            )
            if self.hir2_stage_entropy_aux_lambda > 0:
                stage_probs = stage_alloc.clamp(min=1e-8)
                uniform_logp = -math.log(float(stage_alloc.shape[-1]))
                stage_kl = (stage_probs * (stage_probs.log() - uniform_logp)).sum(dim=-1).mean()
                aux_loss = aux_loss + self.hir2_stage_entropy_aux_lambda * stage_kl

            if self.ffn_moe and aux_data["ffn_moe_weights"]:
                for lw in aux_data["ffn_moe_weights"].values():
                    aux_loss = aux_loss + self.balance_loss_lambda * load_balance_loss(
                        lw,
                        self.n_ffn_experts,
                    )

        return ce_loss + aux_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(item_seq, item_seq_len, feat)
        test_item_emb = self.item_embedding(test_item)
        return (seq_output * test_item_emb).sum(dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        feat = self._gather_features(interaction)
        seq_output, _ = self.forward(item_seq, item_seq_len, feat)
        return seq_output @ self.item_embedding.weight.T
