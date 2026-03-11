"""FeaturedMoE_HGR: Hierarchical Group Routing recommender."""

from __future__ import annotations

import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender

from ..FeaturedMoE.feature_config import (
    ALL_FEATURE_COLUMNS,
    STAGE_ALL_FEATURES,
    feature_list_field,
    build_column_to_index,
)
from ..FeaturedMoE.transformer import TransformerEncoder
from ..FeaturedMoE.logging_utils import MoELogger
from ..FeaturedMoE.analysis_logger import ExpertAnalysisLogger
from .hgr_moe_stages import HierarchicalMoEHGR
from .losses import (
    compute_expert_aux_loss,
    compute_group_balance_aux_loss,
    compute_group_feature_specialization_aux_loss,
    compute_intra_balance_aux_loss,
    compute_router_distill_aux_loss,
)

logger = logging.getLogger(__name__)

_STAGE_NAMES = ("macro", "mid", "micro")
_DEFAULT_LAYOUT = (1, 1, 1, 1, 0)


@dataclass(frozen=True)
class HGRStageLayout:
    pass_layers: int
    moe_blocks: int


@dataclass(frozen=True)
class HGRLayoutSpec:
    raw: Tuple[int, ...]
    global_pre_layers: int
    global_post_layers: int
    stages: Dict[str, HGRStageLayout]


def _layout_attn_sum(layout: HGRLayoutSpec) -> int:
    return int(layout.global_pre_layers) + int(layout.global_post_layers) + sum(
        int(spec.pass_layers) + int(spec.moe_blocks) for spec in layout.stages.values()
    )


def _parse_layout_catalog(raw_catalog) -> list[HGRLayoutSpec]:
    if raw_catalog is None:
        raw_catalog = [list(_DEFAULT_LAYOUT)]
    if not isinstance(raw_catalog, (list, tuple)) or len(raw_catalog) == 0:
        raise ValueError("arch_layout_catalog must be a non-empty list of 5-int or 8-int layouts")

    parsed: list[HGRLayoutSpec] = []
    for idx, layout in enumerate(raw_catalog):
        if not isinstance(layout, (list, tuple)):
            raise ValueError(f"arch_layout_catalog[{idx}] must be a list/tuple, got: {layout}")
        vals = tuple(int(v) for v in layout)
        if len(vals) == 5:
            if vals[0] < 0 or vals[4] < 0:
                raise ValueError(
                    f"arch_layout_catalog[{idx}] invalid global depth: pre/post must be >=0, got {vals}"
                )
            stages: Dict[str, HGRStageLayout] = {}
            for sid, stage_name in enumerate(_STAGE_NAMES, start=1):
                depth = int(vals[sid])
                if depth < -1:
                    raise ValueError(
                        f"arch_layout_catalog[{idx}] invalid stage depth for {stage_name}: got {depth}"
                    )
                if depth < 0:
                    stages[stage_name] = HGRStageLayout(pass_layers=0, moe_blocks=0)
                else:
                    stages[stage_name] = HGRStageLayout(pass_layers=depth, moe_blocks=1)
            parsed.append(
                HGRLayoutSpec(
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
            stage_specs: Dict[str, HGRStageLayout] = {}
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
                stage_specs[stage_name] = HGRStageLayout(
                    pass_layers=int(pass_layers),
                    moe_blocks=int(moe_blocks),
                )
            parsed.append(
                HGRLayoutSpec(
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


class HGRStageBranchRunner(nn.Module):
    """Run one HGR stage with optional pass layers and repeated MoE blocks."""

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
        hierarchical_moe: HierarchicalMoEHGR,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
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
        teacher_group_logits: Dict[str, torch.Tensor] = {}

        if self.moe_blocks <= 0:
            return (
                out,
                gate_weights,
                gate_logits,
                group_weights,
                group_logits,
                intra_group_weights,
                intra_group_logits,
                teacher_group_logits,
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
            if "teacher_group_logits" in router_aux:
                teacher_group_logits[key] = router_aux["teacher_group_logits"]
        return (
            out,
            gate_weights,
            gate_logits,
            group_weights,
            group_logits,
            intra_group_weights,
            intra_group_logits,
            teacher_group_logits,
        )

    def run_parallel(
        self,
        *,
        base_hidden: torch.Tensor,
        item_seq: torch.Tensor,
        feat: torch.Tensor,
        item_seq_len: Optional[torch.Tensor],
        hierarchical_moe: HierarchicalMoEHGR,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        out = self._run_pass(base_hidden, item_seq)
        gate_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        group_weights: Dict[str, torch.Tensor] = {}
        group_logits: Dict[str, torch.Tensor] = {}
        intra_group_weights: Dict[str, torch.Tensor] = {}
        intra_group_logits: Dict[str, torch.Tensor] = {}
        teacher_group_logits: Dict[str, torch.Tensor] = {}

        if self.moe_blocks <= 0:
            return (
                out,
                (out - base_hidden),
                gate_weights,
                gate_logits,
                group_weights,
                group_logits,
                intra_group_weights,
                intra_group_logits,
                teacher_group_logits,
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
            if "teacher_group_logits" in router_aux:
                teacher_group_logits[key] = router_aux["teacher_group_logits"]
        return (
            out,
            (out - base_hidden),
            gate_weights,
            gate_logits,
            group_weights,
            group_logits,
            intra_group_weights,
            intra_group_logits,
            teacher_group_logits,
        )


class FeaturedMoE_HGR(SequentialRecommender):
    """Layout-based Transformer + Hierarchical Group Routing."""

    input_type = "point"

    @staticmethod
    def _parse_stage_list(raw_value, default: list[str]) -> list[str]:
        if raw_value is None:
            return list(default)
        if isinstance(raw_value, (list, tuple)):
            vals = [str(v).strip().lower() for v in raw_value if str(v).strip()]
            return vals if vals else list(default)
        if isinstance(raw_value, str):
            txt = raw_value.strip()
            if txt.startswith("[") and txt.endswith("]"):
                txt = txt[1:-1]
            vals = [tok.strip().lower() for tok in txt.split(",") if tok.strip()]
            return vals if vals else list(default)
        return list(default)

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        def _cfg(key, default):
            return config[key] if key in config else default

        self.n_items = dataset.item_num
        self.d_model = config["embedding_size"]
        self.d_feat_emb = _cfg("d_feat_emb", 64)
        self.d_expert_hidden = _cfg("d_expert_hidden", 64)
        self.d_router_hidden = _cfg("d_router_hidden", 64)
        self.expert_scale = int(_cfg("expert_scale", 1))
        if self.expert_scale < 1:
            raise ValueError(f"expert_scale must be >= 1, got {self.expert_scale}")

        self.n_heads = _cfg("num_heads", 4)
        self.d_ff = _cfg("d_ff", 0) or (4 * self.d_model)
        self.dropout = _cfg("hidden_dropout_prob", 0.1)
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]

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
        self.n_pre_macro = self.stage_pass_layers["macro"]
        self.n_pre_mid = self.stage_pass_layers["mid"]
        self.n_pre_micro = self.stage_pass_layers["micro"]
        self.stage_has_moe = {stage: self.stage_moe_blocks[stage] > 0 for stage in _STAGE_NAMES}
        self.stage_active = {
            stage: (self.stage_pass_layers[stage] > 0 or self.stage_moe_blocks[stage] > 0)
            for stage in _STAGE_NAMES
        }
        self.stage_enabled = dict(self.stage_has_moe)
        self.any_moe = any(self.stage_has_moe.values())
        logger.warning(
            "[FeaturedMoE_HGR Layout] arch_layout_id=%d selected_layout=%s "
            "global_pre=%d stage_pass=%s stage_moe=%s global_post=%d",
            self.arch_layout_id,
            list(selected_layout.raw),
            self.n_pre_layer,
            self.stage_pass_layers,
            self.stage_moe_blocks,
            self.n_post_layer,
        )

        requested_num_layers = int(_cfg("num_layers", 2))
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

        try:
            config["num_layers"] = int(self.num_layers)
            config["n_pre_layer"] = int(self.n_pre_layer)
            config["n_pre_macro"] = int(self.n_pre_macro)
            config["n_pre_mid"] = int(self.n_pre_mid)
            config["n_pre_micro"] = int(self.n_pre_micro)
            config["n_post_layer"] = int(self.n_post_layer)
            config["global_pre_layers"] = int(self.n_pre_layer)
            config["global_post_layers"] = int(self.n_post_layer)
            config["macro_pass_layers"] = int(self.stage_pass_layers["macro"])
            config["macro_moe_blocks"] = int(self.stage_moe_blocks["macro"])
            config["mid_pass_layers"] = int(self.stage_pass_layers["mid"])
            config["mid_moe_blocks"] = int(self.stage_moe_blocks["mid"])
            config["micro_pass_layers"] = int(self.stage_pass_layers["micro"])
            config["micro_moe_blocks"] = int(self.stage_moe_blocks["micro"])
            config["n_total_attn_layers"] = int(self.n_total_attn_layers)
        except Exception:
            pass

        self.router_use_hidden = bool(_cfg("router_use_hidden", True))
        self.router_use_feature = bool(_cfg("router_use_feature", True))
        self.expert_use_hidden = bool(_cfg("expert_use_hidden", True))
        self.expert_use_feature = bool(_cfg("expert_use_feature", False))
        if not (self.expert_use_hidden or self.expert_use_feature):
            raise ValueError("expert_use_hidden and expert_use_feature cannot both be false.")

        self.router_design = str(_cfg("router_design", "group_factorized_interaction")).lower().strip()
        if self.router_design not in {"legacy_concat", "group_factorized_interaction"}:
            raise ValueError(
                "router_design must be one of ['legacy_concat','group_factorized_interaction'], "
                f"got {self.router_design}"
            )
        self.rule_router_cfg = _cfg("rule_router", {}) or {}
        if not isinstance(self.rule_router_cfg, dict):
            raise ValueError("rule_router must be a dict when provided.")

        raw_legacy_top_k = _cfg("moe_top_k", 0)
        self.moe_top_k = None if int(raw_legacy_top_k) <= 0 else int(raw_legacy_top_k)
        raw_expert_top_k = _cfg("expert_top_k", raw_legacy_top_k if int(raw_legacy_top_k) > 0 else 1)
        self.expert_top_k = None if int(raw_expert_top_k) <= 0 else int(raw_expert_top_k)
        raw_group_top_k = int(_cfg("group_top_k", 0))
        self.group_top_k = None if raw_group_top_k <= 0 else raw_group_top_k
        self.group_router_mode = str(_cfg("group_router_mode", "per_group")).lower().strip()
        if self.group_router_mode not in {"stage_wide", "per_group", "hybrid"}:
            raise ValueError(
                "group_router_mode must be one of ['stage_wide','per_group','hybrid'], "
                f"got {self.group_router_mode}"
            )

        self.stage_merge_mode = str(_cfg("stage_merge_mode", "serial")).lower()
        if self.stage_merge_mode not in {"serial", "parallel"}:
            raise ValueError(
                f"stage_merge_mode must be one of ['serial','parallel'], got {self.stage_merge_mode}"
            )
        self.macro_routing_scope = str(_cfg("macro_routing_scope", "session")).lower().strip()
        if self.macro_routing_scope not in {"token", "session"}:
            raise ValueError("macro_routing_scope must be one of ['token','session']")
        self.macro_session_pooling = str(_cfg("macro_session_pooling", "query")).lower().strip()
        if self.macro_session_pooling not in {"query", "mean", "last"}:
            raise ValueError("macro_session_pooling must be one of ['query','mean','last']")
        raw_parallel_gate_top_k = int(_cfg("parallel_stage_gate_top_k", 0))
        self.parallel_stage_gate_top_k = None if raw_parallel_gate_top_k <= 0 else raw_parallel_gate_top_k
        self.parallel_stage_gate_temperature = float(_cfg("parallel_stage_gate_temperature", 1.0))

        self.mid_router_temperature = float(_cfg("mid_router_temperature", 1.3))
        self.micro_router_temperature = float(_cfg("micro_router_temperature", 1.3))
        self.mid_router_feature_dropout = float(_cfg("mid_router_feature_dropout", 0.1))
        self.micro_router_feature_dropout = float(_cfg("micro_router_feature_dropout", 0.1))
        self.use_valid_ratio_gating = bool(_cfg("use_valid_ratio_gating", True))

        self.fmoe_schedule_enable = bool(_cfg("fmoe_schedule_enable", False))
        self.alpha_warmup_until = _cfg("alpha_warmup_until", _cfg("alpha_warmup_steps", 0))
        self.alpha_warmup_start = float(_cfg("alpha_warmup_start", 0.0))
        self.alpha_warmup_end = float(_cfg("alpha_warmup_end", 1.0))
        self.mid_router_temperature_start = float(_cfg("mid_router_temperature_start", self.mid_router_temperature))
        self.micro_router_temperature_start = float(
            _cfg("micro_router_temperature_start", self.micro_router_temperature)
        )
        self.temperature_warmup_until = _cfg("temperature_warmup_until", _cfg("temperature_warmup_steps", 0))

        self.moe_top_k_policy = str(_cfg("moe_top_k_policy", "auto")).lower()
        if self.moe_top_k_policy not in {"auto", "fixed", "half", "ratio", "dense"}:
            raise ValueError(
                "moe_top_k_policy must be one of ['auto','fixed','half','ratio','dense'], "
                f"got {self.moe_top_k_policy}"
            )
        self.moe_top_k_ratio = float(_cfg("moe_top_k_ratio", 0.5))
        self.moe_top_k_min = max(int(_cfg("moe_top_k_min", 1)), 1)
        self.moe_top_k_start = int(_cfg("moe_top_k_start", 0))
        self.moe_top_k_warmup_until = _cfg("moe_top_k_warmup_until", _cfg("moe_top_k_warmup_steps", 0))

        self.fmoe_schedule_log_every_epoch = max(
            int(_cfg("fmoe_schedule_log_every_epoch", _cfg("fmoe_schedule_log_every", 1))),
            1,
        )
        self._schedule_epoch = 0
        self._schedule_total_epochs = max(int(_cfg("epochs", 1)), 1)
        self._last_logged_top_k: Optional[int] = None

        self.use_aux_loss = _cfg("use_aux_loss", True)
        self.balance_loss_lambda = float(_cfg("balance_loss_lambda", 0.003))
        self.group_balance_lambda = float(_cfg("group_balance_lambda", 0.001))
        self.intra_balance_lambda = float(_cfg("intra_balance_lambda", 0.001))
        self.group_feature_spec_aux_enable = bool(_cfg("group_feature_spec_aux_enable", True))
        self.group_feature_spec_aux_lambda = float(_cfg("group_feature_spec_aux_lambda", 3e-4))
        self.group_feature_spec_stages = self._parse_stage_list(
            _cfg("group_feature_spec_stages", ["mid"]),
            default=["mid"],
        )
        self.group_feature_spec_min_tokens = float(_cfg("group_feature_spec_min_tokens", 8))
        self.router_distill_enable = bool(_cfg("router_distill_enable", False))
        self.router_distill_lambda = float(_cfg("router_distill_lambda", 5e-3))
        self.router_distill_temperature = float(_cfg("router_distill_temperature", 1.5))
        self.router_distill_until = float(_cfg("router_distill_until", 0.2))

        self.ffn_moe = _cfg("ffn_moe", False)
        self.n_ffn_experts = _cfg("n_ffn_experts", 4)
        raw_ffn_top_k = _cfg("ffn_top_k", 0)
        self.ffn_top_k = None if raw_ffn_top_k == 0 else int(raw_ffn_top_k)

        raw_debug_logging = _cfg("fmoe_debug_logging", None)
        if raw_debug_logging is None:
            legacy_weights = _cfg("log_expert_weights", False)
            legacy_analysis = _cfg("log_expert_analysis", False)
            self.fmoe_debug_logging = bool(legacy_weights or legacy_analysis)
        else:
            self.fmoe_debug_logging = bool(raw_debug_logging)

        self.log_expert_weights = self.fmoe_debug_logging
        self.log_expert_analysis = self.fmoe_debug_logging
        self.analysis_sample_rate = _cfg("analysis_sample_rate", 0.2)
        self.analysis_n_bins = _cfg("analysis_n_bins", 5)

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
            branch = HGRStageBranchRunner(
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
            self.hierarchical_moe = HierarchicalMoEHGR(
                d_model=self.d_model,
                d_feat_emb=self.d_feat_emb,
                d_expert_hidden=self.d_expert_hidden,
                d_router_hidden=self.d_router_hidden,
                expert_scale=self.expert_scale,
                top_k=self.expert_top_k,
                group_top_k=self.group_top_k,
                group_router_mode=self.group_router_mode,
                router_design=self.router_design,
                rule_router_cfg=self.rule_router_cfg,
                router_distill_enable=self.router_distill_enable,
                parallel_stage_gate_top_k=self.parallel_stage_gate_top_k,
                parallel_stage_gate_temperature=self.parallel_stage_gate_temperature,
                dropout=self.dropout,
                use_macro=self.stage_has_moe["macro"],
                use_mid=self.stage_has_moe["mid"],
                use_micro=self.stage_has_moe["micro"],
                router_use_hidden=self.router_use_hidden,
                router_use_feature=self.router_use_feature,
                expert_use_hidden=self.expert_use_hidden,
                expert_use_feature=self.expert_use_feature,
                stage_merge_mode=self.stage_merge_mode,
                macro_routing_scope=self.macro_routing_scope,
                macro_session_pooling=self.macro_session_pooling,
                mid_router_temperature=self.mid_router_temperature,
                micro_router_temperature=self.micro_router_temperature,
                mid_router_feature_dropout=self.mid_router_feature_dropout,
                micro_router_feature_dropout=self.micro_router_feature_dropout,
                use_valid_ratio_gating=self.use_valid_ratio_gating,
            )
        else:
            self.hierarchical_moe = None

        self._stage_n_experts = 0
        if self.hierarchical_moe is not None:
            self._stage_n_experts = int(getattr(self.hierarchical_moe, "stage_n_experts", 0))

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

        active_expert_names = self.hierarchical_moe.expert_names if self.hierarchical_moe is not None else {}
        active_group_names = self.hierarchical_moe.group_names if self.hierarchical_moe is not None else {}
        self.moe_logger = MoELogger(active_expert_names)
        self.group_logger = MoELogger(active_group_names)
        self._stage_group_names = active_group_names
        self._reset_router_epoch_stats()

        if self.log_expert_analysis and self.any_moe:
            col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
            self.analysis_logger = ExpertAnalysisLogger(
                expert_names=active_expert_names,
                col2idx=col2idx,
                n_bins=self.analysis_n_bins,
                sample_rate=self.analysis_sample_rate,
            )
        else:
            self.analysis_logger = None

        try:
            config["router_design"] = self.router_design
            config["group_top_k"] = 0 if self.group_top_k is None else int(self.group_top_k)
            config["expert_top_k"] = 0 if self.expert_top_k is None else int(self.expert_top_k)
            config["group_balance_lambda"] = float(self.group_balance_lambda)
            config["intra_balance_lambda"] = float(self.intra_balance_lambda)
            config["group_feature_spec_aux_enable"] = bool(self.group_feature_spec_aux_enable)
            config["group_feature_spec_aux_lambda"] = float(self.group_feature_spec_aux_lambda)
            config["group_feature_spec_stages"] = list(self.group_feature_spec_stages)
            config["group_feature_spec_min_tokens"] = float(self.group_feature_spec_min_tokens)
            config["router_distill_enable"] = bool(self.router_distill_enable)
            config["router_distill_lambda"] = float(self.router_distill_lambda)
            config["router_distill_temperature"] = float(self.router_distill_temperature)
            config["router_distill_until"] = float(self.router_distill_until)
        except Exception:
            pass

        self.set_schedule_epoch(epoch_idx=0, max_epochs=self._schedule_total_epochs, log_now=True)
        self.apply(self._init_weights)

        active_stages = [stage for stage in _STAGE_NAMES if self.stage_enabled[stage]]
        logger.info(
            "FeaturedMoE_HGR: d_model=%s d_feat_emb=%s expert_scale=%s layout_id=%s layout=%s "
            "effective_num_layers=%s n_total_attn_layers=%s active_stages=%s merge_mode=%s "
            "router_design=%s group_router_mode=%s group_top_k=%s expert_top_k=%s moe_top_k=%s "
            "group_balance_lambda=%s intra_balance_lambda=%s group_feature_spec_aux=%s "
            "group_feature_spec_lambda=%s group_feature_spec_stages=%s router_distill_enable=%s "
            "router_distill_lambda=%s router_distill_temperature=%s router_distill_until=%s "
            "parallel_stage_gate_top_k=%s "
            "parallel_stage_gate_temperature=%s macro_scope=%s macro_pool=%s "
            "expert_use_hidden=%s expert_use_feature=%s stage_pass=%s stage_moe=%s "
            "mid_temp=%s micro_temp=%s mid_feat_drop=%s micro_feat_drop=%s use_aux_loss=%s "
            "ffn_moe=%s n_features=%s debug_logging=%s schedule_enable=%s top_k_policy=%s "
            "top_k_ratio=%s top_k_warmup_until=%s",
            self.d_model,
            self.d_feat_emb,
            self.expert_scale,
            self.arch_layout_id,
            list(selected_layout.raw),
            self.num_layers,
            self.n_total_attn_layers,
            active_stages,
            self.stage_merge_mode,
            self.router_design,
            self.group_router_mode,
            self.group_top_k,
            self.expert_top_k,
            self.moe_top_k,
            self.group_balance_lambda,
            self.intra_balance_lambda,
            self.group_feature_spec_aux_enable,
            self.group_feature_spec_aux_lambda,
            self.group_feature_spec_stages,
            self.router_distill_enable,
            self.router_distill_lambda,
            self.router_distill_temperature,
            self.router_distill_until,
            self.parallel_stage_gate_top_k,
            self.parallel_stage_gate_temperature,
            self.macro_routing_scope,
            self.macro_session_pooling,
            self.expert_use_hidden,
            self.expert_use_feature,
            self.stage_pass_layers,
            self.stage_moe_blocks,
            self.mid_router_temperature,
            self.micro_router_temperature,
            self.mid_router_feature_dropout,
            self.micro_router_feature_dropout,
            self.use_aux_loss,
            self.ffn_moe,
            self.n_features,
            self.fmoe_debug_logging,
            self.fmoe_schedule_enable,
            self.moe_top_k_policy,
            self.moe_top_k_ratio,
            self.moe_top_k_warmup_until,
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

    @staticmethod
    def _resolve_warmup_end_epoch(warmup_until, total_epochs: int) -> int:
        if warmup_until is None:
            return 0
        try:
            value = float(warmup_until)
        except (TypeError, ValueError):
            return 0
        if value <= 0:
            return 0
        if 0 < value <= 1:
            return max(1, int(math.ceil(float(total_epochs) * value)))
        return max(1, int(round(value)))

    @staticmethod
    def _epoch_progress(epoch_idx: int, end_epoch: int) -> float:
        if end_epoch <= 1:
            return 1.0
        e1 = max(int(epoch_idx) + 1, 1)
        return min(max((float(e1) - 1.0) / float(end_epoch - 1), 0.0), 1.0)

    @classmethod
    def _linear_warmup(cls, epoch_idx: int, end_epoch: int, start: float, end: float) -> float:
        if end_epoch <= 0:
            return float(end)
        progress = cls._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        return float(start + (end - start) * progress)

    @staticmethod
    def _normalize_top_k(top_k: Optional[int], n_experts: int) -> Optional[int]:
        if n_experts <= 0 or top_k is None:
            return None
        value = int(top_k)
        if value <= 0:
            return None
        value = min(value, int(n_experts))
        return None if value >= n_experts else value

    def _resolve_top_k_target(self, n_experts: int) -> Optional[int]:
        if n_experts <= 0:
            return None
        if self.router_design == "group_factorized_interaction":
            return self._normalize_top_k(self.expert_top_k, n_experts=n_experts)
        if self.moe_top_k_policy == "dense":
            return None
        if self.moe_top_k_policy == "half":
            target = int(math.ceil(0.5 * float(n_experts)))
        elif self.moe_top_k_policy == "ratio":
            ratio = min(max(self.moe_top_k_ratio, 0.0), 1.0)
            if ratio <= 0.0:
                return None
            target = int(math.ceil(ratio * float(n_experts)))
        elif self.moe_top_k_policy == "fixed":
            if self.moe_top_k is None:
                return None
            target = int(self.moe_top_k)
        else:
            if self.moe_top_k is None:
                return None
            ratio_target = int(math.ceil(min(max(self.moe_top_k_ratio, 0.0), 1.0) * float(n_experts)))
            target = max(int(self.moe_top_k), ratio_target)

        target = max(self.moe_top_k_min, int(target))
        return self._normalize_top_k(target, n_experts=n_experts)

    def _scheduled_top_k(self, epoch_idx: int, total_epochs: int, n_experts: int) -> Optional[int]:
        target_top_k = self._resolve_top_k_target(n_experts=n_experts)
        if target_top_k is None:
            return None

        if not self.fmoe_schedule_enable:
            return target_top_k

        end_epoch = self._resolve_warmup_end_epoch(
            warmup_until=self.moe_top_k_warmup_until,
            total_epochs=total_epochs,
        )
        if end_epoch <= 0:
            return target_top_k

        start_top_k = self._normalize_top_k(self.moe_top_k_start, n_experts=n_experts)
        start_k = n_experts if start_top_k is None else int(start_top_k)
        target_k = int(target_top_k)
        if start_k == target_k:
            return target_top_k

        progress = self._epoch_progress(epoch_idx=epoch_idx, end_epoch=end_epoch)
        interpolated = int(round(start_k + (target_k - start_k) * progress))
        lower = min(start_k, target_k)
        upper = max(start_k, target_k)
        interpolated = min(max(interpolated, lower), upper)
        return self._normalize_top_k(interpolated, n_experts=n_experts)

    def _apply_training_schedule(self, log_now: bool = False) -> None:
        if not self.any_moe or self.hierarchical_moe is None:
            return

        if not self.fmoe_schedule_enable:
            static_top_k = self._resolve_top_k_target(n_experts=self._stage_n_experts)
            runtime_top_k = -1 if static_top_k is None else int(static_top_k)
            self.hierarchical_moe.set_schedule_state(
                alpha_scale=1.0,
                stage_temperatures={"mid": self.mid_router_temperature, "micro": self.micro_router_temperature},
                top_k=runtime_top_k,
            )
            return

        epoch_idx = int(self._schedule_epoch)
        total_epochs = int(self._schedule_total_epochs)
        alpha_end_epoch = self._resolve_warmup_end_epoch(self.alpha_warmup_until, total_epochs=total_epochs)
        temp_end_epoch = self._resolve_warmup_end_epoch(self.temperature_warmup_until, total_epochs=total_epochs)
        alpha_scale = self._linear_warmup(epoch_idx, alpha_end_epoch, self.alpha_warmup_start, self.alpha_warmup_end)
        mid_temp = self._linear_warmup(
            epoch_idx,
            temp_end_epoch,
            self.mid_router_temperature_start,
            self.mid_router_temperature,
        )
        micro_temp = self._linear_warmup(
            epoch_idx,
            temp_end_epoch,
            self.micro_router_temperature_start,
            self.micro_router_temperature,
        )
        top_k = self._scheduled_top_k(epoch_idx, total_epochs, n_experts=self._stage_n_experts)
        runtime_top_k = -1 if top_k is None else int(top_k)

        self.hierarchical_moe.set_schedule_state(
            alpha_scale=alpha_scale,
            stage_temperatures={"mid": mid_temp, "micro": micro_temp},
            top_k=runtime_top_k,
        )

        should_log = log_now or (epoch_idx % self.fmoe_schedule_log_every_epoch == 0)
        if self._last_logged_top_k != top_k:
            should_log = True
        if should_log:
            logger.info(
                "FMoE_HGR schedule epoch=%s/%s alpha_scale=%.4f mid_temp=%.4f micro_temp=%.4f top_k=%s",
                epoch_idx + 1,
                total_epochs,
                alpha_scale,
                mid_temp,
                micro_temp,
                ("dense" if top_k is None else top_k),
            )
            self._last_logged_top_k = top_k

    def set_schedule_epoch(self, epoch_idx: int, max_epochs: Optional[int] = None, log_now: bool = False) -> None:
        self._schedule_epoch = max(int(epoch_idx), 0)
        if max_epochs is not None:
            self._schedule_total_epochs = max(int(max_epochs), 1)
        self._apply_training_schedule(log_now=log_now)

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

    def _reset_router_epoch_stats(self) -> None:
        self._router_group_entropy_sum = defaultdict(float)
        self._router_group_entropy_count = defaultdict(int)
        self._router_active_clone_sum = {}
        self._router_active_clone_count = defaultdict(int)
        self._router_clone_weight_sum = {}
        self._router_clone_weight_sq_sum = {}
        self._router_clone_weight_count = defaultdict(int)

    def _accumulate_router_stats(
        self,
        *,
        group_weights: Dict[str, torch.Tensor],
        intra_group_weights: Dict[str, torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
    ) -> None:
        for stage_key, group_w in group_weights.items():
            _, tlen, _ = group_w.shape
            if item_seq_len is not None:
                lens = item_seq_len.to(device=group_w.device).long()
                valid = torch.arange(tlen, device=group_w.device).unsqueeze(0) < lens.unsqueeze(1)
                flat_group = group_w[valid]
            else:
                flat_group = group_w.reshape(-1, group_w.shape[-1])
            if flat_group.numel() == 0:
                continue
            entropy = -(flat_group.clamp(min=1e-8) * flat_group.clamp(min=1e-8).log()).sum(dim=-1)
            self._router_group_entropy_sum[stage_key] += float(entropy.sum().item())
            self._router_group_entropy_count[stage_key] += int(entropy.numel())

        for stage_key, intra_w in intra_group_weights.items():
            _, tlen, _, _ = intra_w.shape
            if item_seq_len is not None:
                lens = item_seq_len.to(device=intra_w.device).long()
                valid = torch.arange(tlen, device=intra_w.device).unsqueeze(0) < lens.unsqueeze(1)
                flat_intra = intra_w[valid]
            else:
                flat_intra = intra_w.reshape(-1, intra_w.shape[-2], intra_w.shape[-1])
            if flat_intra.numel() == 0:
                continue
            active = (flat_intra > 0).sum(dim=-1).float()
            active_sum = active.sum(dim=0).detach().cpu()
            if stage_key not in self._router_active_clone_sum:
                self._router_active_clone_sum[stage_key] = torch.zeros_like(active_sum)
            self._router_active_clone_sum[stage_key] += active_sum
            self._router_active_clone_count[stage_key] += int(active.shape[0])

            clone_sum = flat_intra.sum(dim=0).detach().cpu()
            clone_sq_sum = flat_intra.pow(2).sum(dim=0).detach().cpu()
            if stage_key not in self._router_clone_weight_sum:
                self._router_clone_weight_sum[stage_key] = torch.zeros_like(clone_sum)
                self._router_clone_weight_sq_sum[stage_key] = torch.zeros_like(clone_sq_sum)
            self._router_clone_weight_sum[stage_key] += clone_sum
            self._router_clone_weight_sq_sum[stage_key] += clone_sq_sum
            self._router_clone_weight_count[stage_key] += int(flat_intra.shape[0])

    def forward(self, item_seq, item_seq_len, feat=None):
        bsz, tlen = item_seq.shape

        item_emb = self.item_embedding(item_seq)
        position_ids = torch.arange(tlen, device=item_seq.device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.position_embedding(position_ids)

        tokens = self.input_ln(item_emb + pos_emb)
        tokens = self.input_drop(tokens)
        if self.pre_transformer is not None:
            tokens, _ = self.pre_transformer(tokens, item_seq)

        gate_weights, gate_logits, group_weights, group_logits = {}, {}, {}, {}
        intra_group_weights, intra_group_logits, teacher_group_logits = {}, {}, {}
        stage_weights = None
        stage_logits = None

        if self.any_moe and feat is not None and self.hierarchical_moe is not None:
            if self.stage_merge_mode == "parallel":
                stage_deltas: Dict[str, torch.Tensor] = {}
                for stage_name in _STAGE_NAMES:
                    if stage_name not in self.stage_branches or not self.stage_active[stage_name]:
                        continue

                    _, stage_delta, w, l, gw, gl, igw, igl, tgl = self.stage_branches[stage_name].run_parallel(
                        base_hidden=tokens,
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
                    teacher_group_logits.update(tgl)
                    stage_deltas[stage_name] = stage_delta

                tokens, stage_weights, stage_logits = self.hierarchical_moe.parallel_merge(
                    hidden=tokens,
                    feat=feat,
                    stage_deltas=stage_deltas,
                )
            else:
                for stage_name in _STAGE_NAMES:
                    if stage_name not in self.stage_branches or not self.stage_active[stage_name]:
                        continue

                    tokens, w, l, gw, gl, igw, igl, tgl = self.stage_branches[stage_name].run_serial(
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
                    teacher_group_logits.update(tgl)

        hidden, ffn_moe_weights = self.post_transformer(tokens, item_seq)
        gather_idx = (item_seq_len - 1).long().view(-1, 1, 1).expand(-1, 1, self.d_model)
        seq_output = hidden.gather(1, gather_idx).squeeze(1)

        aux_data = {
            "gate_weights": gate_weights,
            "gate_logits": gate_logits,
            "group_weights": group_weights,
            "group_logits": group_logits,
            "group_logits_raw": group_logits,
            "intra_group_weights": intra_group_weights,
            "intra_group_logits": intra_group_logits,
            "intra_group_logits_raw": intra_group_logits,
            "teacher_group_logits": teacher_group_logits,
            "stage_weights": stage_weights,
            "stage_logits": stage_logits,
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
            aux_loss = aux_loss + compute_group_feature_specialization_aux_loss(
                weights=aux_data.get("group_weights", {}),
                feat=feat,
                stage_feature_indices=self.stage_feature_indices,
                selected_stages=self.group_feature_spec_stages,
                item_seq_len=item_seq_len,
                min_tokens_per_group=self.group_feature_spec_min_tokens,
                aux_lambda=self.group_feature_spec_aux_lambda,
                enabled=self.group_feature_spec_aux_enable,
                device=ce_loss.device,
            )
            aux_loss = aux_loss + compute_router_distill_aux_loss(
                teacher_group_logits=aux_data.get("teacher_group_logits", {}),
                student_group_logits=aux_data.get("group_logits_raw", {}),
                item_seq_len=item_seq_len,
                aux_lambda=self.router_distill_lambda,
                distill_temperature=self.router_distill_temperature,
                enabled=self.router_distill_enable,
                progress=float(self._schedule_epoch) / float(max(self._schedule_total_epochs - 1, 1)),
                until=self.router_distill_until,
                device=ce_loss.device,
            )
            if self.ffn_moe and aux_data.get("ffn_moe_weights"):
                aux_loss = aux_loss + compute_expert_aux_loss(
                    aux_data.get("ffn_moe_weights", {}),
                    item_seq_len=item_seq_len,
                    balance_lambda=self.balance_loss_lambda,
                    device=ce_loss.device,
                )

        total_loss = ce_loss + aux_loss

        if self.log_expert_weights and self.training and aux_data["gate_weights"]:
            self.moe_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                item_seq_len=item_seq_len,
            )
        if self.fmoe_debug_logging and self.training and aux_data.get("group_weights"):
            self.group_logger.accumulate(
                gate_weights=aux_data["group_weights"],
                item_seq_len=item_seq_len,
            )
            self._accumulate_router_stats(
                group_weights=aux_data.get("group_weights", {}),
                intra_group_weights=aux_data.get("intra_group_weights", {}),
                item_seq_len=item_seq_len,
            )

        if self.analysis_logger is not None and self.training and aux_data["gate_weights"]:
            self.analysis_logger.accumulate(
                gate_weights=aux_data["gate_weights"],
                feat=feat,
                logits=logits,
                pos_items=pos_items,
                item_seq_len=item_seq_len,
            )

        return total_loss

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

    def get_epoch_log_summary(self) -> Dict:
        summary = self.moe_logger.get_and_reset()
        group_summary = self.group_logger.get_and_reset()
        summary["group_stages"] = group_summary.get("stages", {})
        summary["router"] = {}

        all_stage_keys = (
            set(self._router_group_entropy_sum)
            | set(self._router_active_clone_sum)
            | set(self._router_clone_weight_sum)
        )
        for stage_key in sorted(all_stage_keys):
            base_stage = stage_key.split("@", 1)[0]
            entropy_count = max(self._router_group_entropy_count.get(stage_key, 0), 1)
            active_count = max(self._router_active_clone_count.get(stage_key, 0), 1)
            active_sum = self._router_active_clone_sum.get(stage_key)
            active_list = [] if active_sum is None else (active_sum / float(active_count)).tolist()
            clone_count = max(self._router_clone_weight_count.get(stage_key, 0), 1)
            clone_sum = self._router_clone_weight_sum.get(stage_key)
            clone_sq_sum = self._router_clone_weight_sq_sum.get(stage_key)
            if clone_sum is None or clone_sq_sum is None:
                clone_mean = []
                clone_std = []
            else:
                clone_mean_tensor = clone_sum / float(clone_count)
                clone_var = (clone_sq_sum / float(clone_count) - clone_mean_tensor.pow(2)).clamp(min=0.0)
                clone_mean = clone_mean_tensor.tolist()
                clone_std = clone_var.sqrt().tolist()
            summary["router"][stage_key] = {
                "group_names": list(self._stage_group_names.get(base_stage, [])),
                "group_entropy": float(self._router_group_entropy_sum.get(stage_key, 0.0)) / float(entropy_count),
                "active_clones_per_group": active_list,
                "clone_load": clone_mean,
                "clone_load_std": clone_std,
            }

        self._reset_router_epoch_stats()
        if self.analysis_logger is not None:
            summary["analysis"] = self.analysis_logger.get_and_reset()
        return summary
