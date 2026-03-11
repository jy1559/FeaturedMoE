"""FeaturedMoE_Individual: feature-individual hierarchical MoE."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from ..FeaturedMoE.feature_config import ALL_FEATURE_COLUMNS, STAGE_ALL_FEATURES, build_column_to_index, feature_list_field
from ..FeaturedMoE.transformer import TransformerEncoder
from ..FeaturedMoE_HGRv3.losses import (
    compute_expert_aux_loss,
    compute_group_balance_aux_loss,
    compute_intra_balance_aux_loss,
)
from .individual_moe_stages import HierarchicalMoEIndividual

logger = logging.getLogger(__name__)

_STAGE_NAMES = ("macro", "mid", "micro")
_DEFAULT_LAYOUT = (0, 0, 1, 0, 1, 0, 0, 0)


@dataclass(frozen=True)
class IndividualStageLayout:
    pass_layers: int
    moe_blocks: int


@dataclass(frozen=True)
class IndividualLayoutSpec:
    raw: Tuple[int, ...]
    global_pre_layers: int
    global_post_layers: int
    stages: Dict[str, IndividualStageLayout]


def _layout_attn_sum(layout: IndividualLayoutSpec) -> int:
    return int(layout.global_pre_layers) + int(layout.global_post_layers) + sum(
        int(spec.pass_layers) + int(spec.moe_blocks) for spec in layout.stages.values()
    )


def _parse_layout_catalog(raw_catalog) -> list[IndividualLayoutSpec]:
    if raw_catalog is None:
        raw_catalog = [list(_DEFAULT_LAYOUT)]
    if not isinstance(raw_catalog, (list, tuple)) or len(raw_catalog) == 0:
        raise ValueError("arch_layout_catalog must be a non-empty list of 5-int or 8-int layouts")

    parsed: list[IndividualLayoutSpec] = []
    for idx, layout in enumerate(raw_catalog):
        if not isinstance(layout, (list, tuple)):
            raise ValueError(f"arch_layout_catalog[{idx}] must be a list/tuple, got: {layout}")
        vals = tuple(int(v) for v in layout)
        if len(vals) == 5:
            if vals[0] < 0 or vals[4] < 0:
                raise ValueError(
                    f"arch_layout_catalog[{idx}] invalid global depth: pre/post must be >=0, got {vals}"
                )
            stages: Dict[str, IndividualStageLayout] = {}
            for sid, stage_name in enumerate(_STAGE_NAMES, start=1):
                depth = int(vals[sid])
                if depth < -1:
                    raise ValueError(
                        f"arch_layout_catalog[{idx}] invalid stage depth for {stage_name}: got {depth}"
                    )
                if depth < 0:
                    stages[stage_name] = IndividualStageLayout(pass_layers=0, moe_blocks=0)
                else:
                    stages[stage_name] = IndividualStageLayout(pass_layers=depth, moe_blocks=1)
            parsed.append(
                IndividualLayoutSpec(
                    raw=vals,
                    global_pre_layers=int(vals[0]),
                    global_post_layers=int(vals[4]),
                    stages=stages,
                )
            )
            continue

        if len(vals) == 8:
            if vals[0] < 0 or vals[7] < 0:
                raise ValueError(
                    f"arch_layout_catalog[{idx}] invalid global depth: pre/post must be >=0, got {vals}"
                )
            stage_specs: Dict[str, IndividualStageLayout] = {}
            triplets = {
                "macro": (vals[1], vals[2]),
                "mid": (vals[3], vals[4]),
                "micro": (vals[5], vals[6]),
            }
            for stage_name, (pass_layers, moe_blocks) in triplets.items():
                if int(pass_layers) < 0 or int(moe_blocks) < 0:
                    raise ValueError(
                        f"arch_layout_catalog[{idx}] invalid {stage_name} pass/moe values: {(pass_layers, moe_blocks)}"
                    )
                stage_specs[stage_name] = IndividualStageLayout(
                    pass_layers=int(pass_layers),
                    moe_blocks=int(moe_blocks),
                )
            parsed.append(
                IndividualLayoutSpec(
                    raw=vals,
                    global_pre_layers=int(vals[0]),
                    global_post_layers=int(vals[7]),
                    stages=stage_specs,
                )
            )
            continue

        raise ValueError(
            f"arch_layout_catalog[{idx}] must be length-5 legacy or length-8 extended layout, got: {layout}"
        )
    return parsed


class IndividualStageBranchRunner(nn.Module):
    """Run one stage with optional pass layers and repeated MoE blocks."""

    def __init__(
        self,
        *,
        stage_name: str,
        pass_layers: int,
        moe_blocks: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.stage_name = stage_name
        self.pass_layers = int(pass_layers)
        self.moe_blocks = int(moe_blocks)

        if self.pass_layers > 0:
            self.pass_transformer = TransformerEncoder(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=self.pass_layers,
                d_ff=d_ff,
                dropout=dropout,
                ffn_moe=False,
            )
        else:
            self.pass_transformer = None

        self.moe_pre_blocks = nn.ModuleList(
            [
                TransformerEncoder(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=1,
                    d_ff=d_ff,
                    dropout=dropout,
                    ffn_moe=False,
                )
                for _ in range(max(self.moe_blocks, 0))
            ]
        )

    def _run_pass(self, hidden: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        if self.pass_transformer is None:
            return hidden
        out, _ = self.pass_transformer(hidden, item_seq)
        return out

    def run_serial(
        self,
        *,
        hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        hierarchical_moe: HierarchicalMoEIndividual,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        out = self._run_pass(hidden, item_seq)
        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        group_weights: Dict[str, torch.Tensor] = {}
        group_logits: Dict[str, torch.Tensor] = {}
        intra_group_weights: Dict[str, torch.Tensor] = {}
        intra_group_logits: Dict[str, torch.Tensor] = {}

        if self.moe_blocks <= 0:
            return (
                out,
                gate_weights,
                gate_logits,
                group_weights,
                group_logits,
                intra_group_weights,
                intra_group_logits,
            )

        for idx, pre_block in enumerate(self.moe_pre_blocks, start=1):
            out, _ = pre_block(out, item_seq)
            out, w, l, gw, gl, _ = hierarchical_moe.forward_stage(
                self.stage_name,
                out,
                feat,
                item_seq_len=item_seq_len,
            )
            key = self.stage_name if self.moe_blocks == 1 else f"{self.stage_name}@{idx}"
            gate_weights[key] = w
            gate_logits[key] = l
            group_weights[key] = gw
            group_logits[key] = gl

            router_aux = hierarchical_moe.get_stage_router_aux(self.stage_name)
            if "intra_group_weights" in router_aux:
                intra_group_weights[key] = router_aux["intra_group_weights"]
            if "intra_group_logits" in router_aux:
                intra_group_logits[key] = router_aux["intra_group_logits"]

        return (
            out,
            gate_weights,
            gate_logits,
            group_weights,
            group_logits,
            intra_group_weights,
            intra_group_logits,
        )


class FeaturedMoE_Individual(SequentialRecommender):
    """Layout-based Transformer + feature-individual 2-level MoE."""

    input_type = "point"

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(key, default):
            return config[key] if key in config else default

        self.n_items = dataset.item_num
        self.d_model = int(config["embedding_size"])
        self.d_feat_emb = int(_cfg("d_feat_emb", 16))
        self.d_expert_hidden = int(_cfg("d_expert_hidden", 160))
        self.d_router_hidden = int(_cfg("d_router_hidden", 64))
        self.expert_scale = int(_cfg("expert_scale", 4))
        if self.expert_scale < 1:
            raise ValueError(f"expert_scale must be >= 1, got {self.expert_scale}")

        self.n_heads = int(_cfg("num_heads", 8))
        self.d_ff = int(_cfg("d_ff", 0) or (4 * self.d_model))
        self.dropout = float(_cfg("hidden_dropout_prob", 0.1))
        self.max_seq_length = int(config["MAX_ITEM_LIST_LENGTH"])

        self.arch_layout_catalog = _parse_layout_catalog(_cfg("arch_layout_catalog", [list(_DEFAULT_LAYOUT)]))
        self.arch_layout_id = int(_cfg("arch_layout_id", 0))
        if not (0 <= self.arch_layout_id < len(self.arch_layout_catalog)):
            raise ValueError(
                f"arch_layout_id out of range: id={self.arch_layout_id}, "
                f"catalog_size={len(self.arch_layout_catalog)}"
            )

        selected_layout = self.arch_layout_catalog[self.arch_layout_id]
        self.layout_spec = selected_layout
        self.n_pre_layer = int(selected_layout.global_pre_layers)
        self.n_post_layer = int(selected_layout.global_post_layers)
        self.stage_pass_layers = {
            stage_name: int(selected_layout.stages[stage_name].pass_layers) for stage_name in _STAGE_NAMES
        }
        self.stage_moe_blocks = {
            stage_name: int(selected_layout.stages[stage_name].moe_blocks) for stage_name in _STAGE_NAMES
        }
        self.stage_has_moe = {stage: self.stage_moe_blocks[stage] > 0 for stage in _STAGE_NAMES}
        self.stage_active = {
            stage: (self.stage_pass_layers[stage] > 0 or self.stage_moe_blocks[stage] > 0)
            for stage in _STAGE_NAMES
        }
        self.any_moe = any(self.stage_has_moe.values())

        requested_num_layers = int(_cfg("num_layers", -1))
        if requested_num_layers >= 0:
            for lid, layout in enumerate(self.arch_layout_catalog):
                layout_sum = _layout_attn_sum(layout)
                if layout_sum != requested_num_layers:
                    raise ValueError(
                        "num_layers budget mismatch: "
                        f"num_layers={requested_num_layers}, "
                        f"arch_layout_catalog[{lid}]={list(layout.raw)} has total_attn={layout_sum}. "
                        "All catalog layouts must have identical total attn when num_layers>=0."
                    )

        self.n_total_attn_layers = _layout_attn_sum(selected_layout)
        self.num_layers = self.n_total_attn_layers if requested_num_layers < 0 else requested_num_layers
        logger.warning(
            "[FeaturedMoE_Individual Layout] arch_layout_id=%d selected_layout=%s "
            "global_pre=%d stage_pass=%s stage_moe=%s global_post=%d",
            self.arch_layout_id,
            list(selected_layout.raw),
            self.n_pre_layer,
            self.stage_pass_layers,
            self.stage_moe_blocks,
            self.n_post_layer,
        )

        self.outer_router_use_hidden = bool(_cfg("outer_router_use_hidden", True))
        self.outer_router_use_feature = bool(_cfg("outer_router_use_feature", True))
        self.inner_router_use_hidden = bool(_cfg("inner_router_use_hidden", True))
        self.inner_router_use_feature = bool(_cfg("inner_router_use_feature", True))
        self.expert_use_hidden = bool(_cfg("expert_use_hidden", True))
        self.expert_use_feature = bool(_cfg("expert_use_feature", False))
        if not (self.expert_use_hidden or self.expert_use_feature):
            raise ValueError("expert_use_hidden and expert_use_feature cannot both be false.")

        raw_feature_top_k = int(_cfg("feature_top_k", _cfg("group_top_k", 4)))
        self.feature_top_k = None if raw_feature_top_k <= 0 else raw_feature_top_k
        raw_inner_top_k = int(_cfg("inner_expert_top_k", _cfg("expert_top_k", 0)))
        self.inner_expert_top_k = None if raw_inner_top_k <= 0 else raw_inner_top_k

        self.stage_merge_mode = str(_cfg("stage_merge_mode", "serial")).lower().strip()
        if self.stage_merge_mode != "serial":
            raise ValueError("FeaturedMoE_Individual currently supports only stage_merge_mode=serial")

        self.use_aux_loss = bool(_cfg("use_aux_loss", False))
        self.balance_loss_lambda = float(_cfg("balance_loss_lambda", 0.0))
        self.group_balance_lambda = float(_cfg("group_balance_lambda", 0.0))
        self.intra_balance_lambda = float(_cfg("intra_balance_lambda", 0.0))

        self.ffn_moe = bool(_cfg("ffn_moe", False))
        self.n_ffn_experts = int(_cfg("n_ffn_experts", 4))
        raw_ffn_top_k = _cfg("ffn_top_k", 0)
        self.ffn_top_k = None if int(raw_ffn_top_k) <= 0 else int(raw_ffn_top_k)

        self.feature_fields = [feature_list_field(col) for col in ALL_FEATURE_COLUMNS]
        self.n_features = len(ALL_FEATURE_COLUMNS)
        col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
        self.stage_feature_indices = {
            stage_name: [int(col2idx[col]) for col in STAGE_ALL_FEATURES.get(stage_name, []) if col in col2idx]
            for stage_name in _STAGE_NAMES
        }

        self.item_embedding = nn.Embedding(self.n_items, self.d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.d_model)
        self.input_ln = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(self.dropout)

        if self.n_pre_layer > 0:
            self.pre_transformer = TransformerEncoder(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_pre_layer,
                d_ff=self.d_ff,
                dropout=self.dropout,
                ffn_moe=False,
            )
        else:
            self.pre_transformer = None

        self.stage_branches = nn.ModuleDict()
        for stage_name in _STAGE_NAMES:
            branch = IndividualStageBranchRunner(
                stage_name=stage_name,
                pass_layers=self.stage_pass_layers[stage_name],
                moe_blocks=self.stage_moe_blocks[stage_name],
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
            )
            if branch.pass_layers <= 0 and branch.moe_blocks <= 0:
                continue
            self.stage_branches[stage_name] = branch

        if self.any_moe:
            self.hierarchical_moe = HierarchicalMoEIndividual(
                d_model=self.d_model,
                d_feat_emb=self.d_feat_emb,
                d_expert_hidden=self.d_expert_hidden,
                d_router_hidden=self.d_router_hidden,
                expert_scale=self.expert_scale,
                feature_top_k=self.feature_top_k,
                inner_expert_top_k=self.inner_expert_top_k,
                dropout=self.dropout,
                use_macro=self.stage_has_moe["macro"],
                use_mid=self.stage_has_moe["mid"],
                use_micro=self.stage_has_moe["micro"],
                expert_use_hidden=self.expert_use_hidden,
                expert_use_feature=self.expert_use_feature,
                outer_router_use_hidden=self.outer_router_use_hidden,
                outer_router_use_feature=self.outer_router_use_feature,
                inner_router_use_hidden=self.inner_router_use_hidden,
                inner_router_use_feature=self.inner_router_use_feature,
                macro_router_temperature=1.0,
                mid_router_temperature=1.0,
                micro_router_temperature=1.0,
            )
        else:
            self.hierarchical_moe = None

        self.post_transformer = TransformerEncoder(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_post_layer,
            d_ff=self.d_ff,
            dropout=self.dropout,
            ffn_moe=self.ffn_moe,
            n_ffn_experts=self.n_ffn_experts,
            ffn_top_k=self.ffn_top_k,
        )

        try:
            config["num_layers"] = int(self.num_layers)
            config["feature_top_k"] = 0 if self.feature_top_k is None else int(self.feature_top_k)
            config["inner_expert_top_k"] = 0 if self.inner_expert_top_k is None else int(self.inner_expert_top_k)
            config["outer_router_use_hidden"] = bool(self.outer_router_use_hidden)
            config["outer_router_use_feature"] = bool(self.outer_router_use_feature)
            config["inner_router_use_hidden"] = bool(self.inner_router_use_hidden)
            config["inner_router_use_feature"] = bool(self.inner_router_use_feature)
            config["expert_use_hidden"] = bool(self.expert_use_hidden)
            config["expert_use_feature"] = bool(self.expert_use_feature)
            config["group_balance_lambda"] = float(self.group_balance_lambda)
            config["intra_balance_lambda"] = float(self.intra_balance_lambda)
        except Exception:
            pass

        self.apply(self._init_weights)

        logger.info(
            "FeaturedMoE_Individual: d_model=%s d_feat_emb=%s d_expert_hidden=%s d_router_hidden=%s "
            "expert_scale=%s feature_top_k=%s inner_expert_top_k=%s layout_id=%s layout=%s "
            "stage_pass=%s stage_moe=%s use_aux_loss=%s outer_router(hidden=%s,feature=%s) "
            "inner_router(hidden=%s,feature=%s) expert_use_feature=%s",
            self.d_model,
            self.d_feat_emb,
            self.d_expert_hidden,
            self.d_router_hidden,
            self.expert_scale,
            self.feature_top_k,
            self.inner_expert_top_k,
            self.arch_layout_id,
            list(selected_layout.raw),
            self.stage_pass_layers,
            self.stage_moe_blocks,
            self.use_aux_loss,
            self.outer_router_use_hidden,
            self.outer_router_use_feature,
            self.inner_router_use_hidden,
            self.inner_router_use_feature,
            self.expert_use_feature,
        )

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

    def set_schedule_epoch(self, epoch_idx: int, max_epochs: Optional[int] = None, log_now: bool = False) -> None:
        del epoch_idx, max_epochs, log_now

    def _gather_features(self, interaction) -> Optional[torch.Tensor]:
        if not self.any_moe:
            return None

        feat_list = []
        for field in self.feature_fields:
            if field in interaction:
                feat_list.append(interaction[field].float())
            else:
                bsz = interaction[self.ITEM_SEQ].shape[0]
                tlen = interaction[self.ITEM_SEQ].shape[1]
                feat_list.append(torch.zeros(bsz, tlen, device=interaction[self.ITEM_SEQ].device))
                logger.warning("Feature field '%s' not found - using zeros.", field)
        return torch.stack(feat_list, dim=-1)

    def forward(self, item_seq, item_seq_len, feat=None):
        bsz, tlen = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(tlen, device=item_seq.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)
        if self.pre_transformer is not None:
            tokens, _ = self.pre_transformer(tokens, item_seq)

        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        group_weights: Dict[str, torch.Tensor] = {}
        group_logits: Dict[str, torch.Tensor] = {}
        intra_group_weights: Dict[str, torch.Tensor] = {}
        intra_group_logits: Dict[str, torch.Tensor] = {}

        if self.any_moe and feat is not None and self.hierarchical_moe is not None:
            for stage_name in _STAGE_NAMES:
                if stage_name not in self.stage_branches or not self.stage_active[stage_name]:
                    continue

                tokens, w, l, gw, gl, igw, igl = self.stage_branches[stage_name].run_serial(
                    hidden=tokens,
                    item_seq=item_seq,
                    feat=feat,
                    item_seq_len=item_seq_len,
                    hierarchical_moe=self.hierarchical_moe,
                )
                gate_weights.update(w)
                gate_logits.update(l)
                group_weights.update(gw)
                group_logits.update(gl)
                intra_group_weights.update(igw)
                intra_group_logits.update(igl)

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)
        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "group_weights": group_weights,
            "group_logits": group_logits,
            "intra_group_weights": intra_group_weights,
            "intra_group_logits": intra_group_logits,
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
        if self.use_aux_loss:
            aux_loss = aux_loss + compute_expert_aux_loss(
                aux_data.get("gate_weights", {}),
                item_seq_len=item_seq_len,
                balance_lambda=self.balance_loss_lambda,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_group_balance_aux_loss(
                aux_data.get("group_weights", {}),
                item_seq_len=item_seq_len,
                aux_lambda=self.group_balance_lambda,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_intra_balance_aux_loss(
                aux_data.get("intra_group_weights", {}),
                item_seq_len=item_seq_len,
                aux_lambda=self.intra_balance_lambda,
                device=ce_loss.device,
            )
            if self.ffn_moe and aux_data.get("ffn_moe_weights"):
                aux_loss = aux_loss + compute_expert_aux_loss(
                    aux_data.get("ffn_moe_weights", {}),
                    item_seq_len=item_seq_len,
                    balance_lambda=self.balance_loss_lambda,
                    device=ce_loss.device,
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

