"""Evaluation diagnostics for FeaturedMoE_N3."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def _base_stage_name(stage_key: str) -> str:
    text = str(stage_key or "")
    if "@" in text:
        text = text.split("@", 1)[0]
    if "." in text:
        text = text.split(".", 1)[0]
    return text


def _as_tensor(value, *, device):
    if torch.is_tensor(value):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _valid_mask(item_seq_len: Optional[torch.Tensor], seq_len: int, device: torch.device) -> torch.Tensor:
    if item_seq_len is None:
        return torch.ones(1, seq_len, dtype=torch.bool, device=device)
    lens = item_seq_len.to(device=device).long().clamp(min=1, max=seq_len)
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    return arange < lens.unsqueeze(1)


def _float_list(tensor: torch.Tensor) -> List[float]:
    return [float(v) for v in tensor.detach().cpu().tolist()]


class N3DiagnosticCollector:
    """Collect compact routing/conditioning diagnostics over one evaluation split."""

    def __init__(
        self,
        *,
        split_name: str,
        stage_family_features: Dict[str, Dict[str, List[str]]],
        stage_expert_names: Dict[str, List[str]],
        all_feature_columns: List[str],
        max_positions: int,
        feature_mode: str = "none",
    ):
        self.split_name = str(split_name)
        self.stage_family_features = {
            str(stage): {str(group): list(cols) for group, cols in groups.items()}
            for stage, groups in dict(stage_family_features or {}).items()
        }
        self.stage_expert_names = {
            str(stage): list(names) for stage, names in dict(stage_expert_names or {}).items()
        }
        self.col2idx = {name: idx for idx, name in enumerate(list(all_feature_columns or []))}
        self.max_positions = max(int(max_positions), 1)
        self.feature_mode = str(feature_mode or "none")
        self._stage_data: Dict[str, dict] = {}

    def _ensure_stage(self, stage_key: str, *, n_slots: int, mode: str) -> dict:
        if stage_key not in self._stage_data:
            base_stage = _base_stage_name(stage_key)
            family_names = list(self.stage_family_features.get(base_stage, {}).keys())
            self._stage_data[stage_key] = {
                "mode": mode,
                "n_slots": int(n_slots),
                "usage_sum": torch.zeros(n_slots),
                "top1_count": torch.zeros(n_slots),
                "entropy_sum": 0.0,
                "n_tokens": 0,
                "n_sessions": 0,
                "dead_tokens": 0,
                "jitter_adj_sum": 0.0,
                "jitter_adj_count": 0,
                "jitter_session_sum": 0.0,
                "family_expert": torch.zeros(len(family_names), n_slots),
                "family_names": family_names,
                "position_usage": torch.zeros(self.max_positions, n_slots),
                "transition": torch.zeros(n_slots, n_slots),
                "cond_norm_sum": 0.0,
                "cond_norm_count": 0,
                "delta_norm_sum": 0.0,
                "delta_norm_count": 0,
                "expert_out_sum": torch.zeros(n_slots, 1),
                "expert_out_dim": 1,
            }
        return self._stage_data[stage_key]

    @torch.no_grad()
    def update(
        self,
        *,
        interaction,
        feat: Optional[torch.Tensor],
        item_seq_len: Optional[torch.Tensor],
        aux_data: Dict[str, object],
    ) -> None:
        gate_weights = dict(aux_data.get("gate_weights", {}) or {})
        router_aux = dict(aux_data.get("router_aux", {}) or {})
        dense_aux = dict(aux_data.get("dense_aux", {}) or {})
        if not gate_weights and not dense_aux:
            return

        device = item_seq_len.device if torch.is_tensor(item_seq_len) else None
        if device is None and gate_weights:
            device = next(iter(gate_weights.values())).device
        if device is None and dense_aux:
            device = next(iter(dense_aux.values())).get("delta_norm", torch.zeros(1)).device
        if device is None:
            return

        for stage_key, weights in gate_weights.items():
            if not torch.is_tensor(weights):
                continue
            seq_len = weights.size(1)
            mask = _valid_mask(item_seq_len, seq_len, weights.device)
            if mask.size(0) == 1 and weights.size(0) > 1:
                mask = mask.expand(weights.size(0), -1)

            stage_store = self._ensure_stage(stage_key, n_slots=weights.size(-1), mode="moe")
            w = weights.detach()
            valid = mask.unsqueeze(-1).float()
            n_valid = int(mask.sum().item())
            if n_valid <= 0:
                continue
            stage_store["usage_sum"] += (w * valid).sum(dim=(0, 1)).cpu()
            entropy = -(w.clamp(min=1e-8) * w.clamp(min=1e-8).log()).sum(dim=-1)
            stage_store["entropy_sum"] += float((entropy * mask.float()).sum().item())
            stage_store["n_tokens"] += n_valid
            stage_store["n_sessions"] += int(weights.size(0))

            top1 = w.argmax(dim=-1)
            flat_top1 = top1[mask]
            stage_store["top1_count"] += torch.bincount(flat_top1.cpu(), minlength=weights.size(-1)).float()

            for b_idx in range(weights.size(0)):
                valid_pos = mask[b_idx].nonzero(as_tuple=False).view(-1)
                if valid_pos.numel() <= 0:
                    continue
                top1_seq = top1[b_idx, valid_pos]
                if top1_seq.numel() > 1:
                    changes = (top1_seq[1:] != top1_seq[:-1]).float()
                    stage_store["jitter_adj_sum"] += float(changes.sum().item())
                    stage_store["jitter_adj_count"] += int(changes.numel())
                    majority = torch.bincount(top1_seq.cpu(), minlength=weights.size(-1)).float().max().item()
                    stage_store["jitter_session_sum"] += 1.0 - (majority / float(top1_seq.numel()))
                    src = top1_seq[:-1]
                    dst = top1_seq[1:]
                    trans = torch.zeros_like(stage_store["transition"])
                    trans.index_put_(
                        (src.cpu(), dst.cpu()),
                        torch.ones(src.numel()),
                        accumulate=True,
                    )
                    stage_store["transition"] += trans

                capped = valid_pos[: self.max_positions]
                top1_cap = top1[b_idx, capped]
                pos_idx = torch.arange(top1_cap.numel(), dtype=torch.long)
                pos_updates = torch.zeros_like(stage_store["position_usage"])
                pos_updates.index_put_((pos_idx, top1_cap.cpu()), torch.ones(top1_cap.numel()), accumulate=True)
                stage_store["position_usage"] += pos_updates

            family_map = self.stage_family_features.get(_base_stage_name(stage_key), {})
            if feat is not None and family_map:
                raw_feat = feat.to(device=weights.device)
                for fam_idx, family_name in enumerate(stage_store["family_names"]):
                    cols = [self.col2idx[col] for col in family_map.get(family_name, []) if col in self.col2idx]
                    if not cols:
                        continue
                    fam_score = raw_feat.index_select(-1, torch.tensor(cols, device=raw_feat.device)).mean(dim=-1)
                    family_usage = (fam_score.unsqueeze(-1) * w * mask.unsqueeze(-1).float()).sum(dim=(0, 1)).cpu()
                    stage_store["family_expert"][fam_idx] += family_usage

            cond = dict(dense_aux.get(stage_key, {}) or {})
            if cond:
                delta_norm = cond.get("delta_norm")
                cond_norm = cond.get("condition_norm")
                if torch.is_tensor(delta_norm):
                    stage_store["delta_norm_sum"] += float(delta_norm.detach().sum().item())
                    stage_store["delta_norm_count"] += int(delta_norm.numel())
                if torch.is_tensor(cond_norm):
                    stage_store["cond_norm_sum"] += float(cond_norm.detach().sum().item())
                    stage_store["cond_norm_count"] += int(cond_norm.numel())

            expert_out_mean = dict(router_aux.get("expert_output_mean", {}) or {}).get(stage_key)
            if torch.is_tensor(expert_out_mean) and expert_out_mean.ndim == 2:
                stage_store["expert_out_sum"] = stage_store["expert_out_sum"].new_zeros(expert_out_mean.size(0), expert_out_mean.size(1))
                stage_store["expert_out_sum"] += expert_out_mean.detach().cpu()
                stage_store["expert_out_dim"] = int(expert_out_mean.size(1))

        for stage_key, cond in dense_aux.items():
            stage_store = self._ensure_stage(stage_key, n_slots=1, mode="dense")
            delta_norm = cond.get("delta_norm")
            cond_norm = cond.get("condition_norm")
            if torch.is_tensor(delta_norm):
                stage_store["delta_norm_sum"] += float(delta_norm.detach().sum().item())
                stage_store["delta_norm_count"] += int(delta_norm.numel())
            if torch.is_tensor(cond_norm):
                stage_store["cond_norm_sum"] += float(cond_norm.detach().sum().item())
                stage_store["cond_norm_count"] += int(cond_norm.numel())

    @staticmethod
    def _usage_scalars(usage_sum: torch.Tensor, top1_count: torch.Tensor) -> dict:
        if usage_sum.numel() <= 0 or float(usage_sum.sum().item()) <= 0:
            return {
                "usage_share": [],
                "n_eff": 0.0,
                "cv_usage": 0.0,
                "dead_expert_frac": 0.0,
                "top1_max_frac": 0.0,
            }
        usage_share = usage_sum / usage_sum.sum().clamp(min=1e-8)
        usage_mean = float(usage_share.mean().item())
        usage_std = float(usage_share.std(unbiased=False).item())
        top1_total = float(top1_count.sum().item())
        top1_max_frac = float((top1_count.max().item() / top1_total) if top1_total > 0 else 0.0)
        return {
            "usage_share": _float_list(usage_share),
            "n_eff": float((1.0 / usage_share.pow(2).sum().item()) if usage_share.numel() > 0 else 0.0),
            "cv_usage": float(usage_std / max(usage_mean, 1e-8)),
            "dead_expert_frac": float((usage_sum <= 1e-8).float().mean().item()),
            "top1_max_frac": top1_max_frac,
        }

    @staticmethod
    def _expert_similarity(expert_out_sum: torch.Tensor) -> dict:
        if expert_out_sum.ndim != 2 or expert_out_sum.size(0) <= 1 or expert_out_sum.size(1) <= 1:
            return {"expert_similarity_mean": 0.0, "expert_similarity_max": 0.0}
        normed = F.normalize(expert_out_sum, dim=-1)
        sim = normed @ normed.transpose(0, 1)
        off = sim[~torch.eye(sim.size(0), dtype=torch.bool)]
        if off.numel() <= 0:
            return {"expert_similarity_mean": 0.0, "expert_similarity_max": 0.0}
        return {
            "expert_similarity_mean": float(off.mean().item()),
            "expert_similarity_max": float(off.max().item()),
        }

    def finalize(self) -> dict:
        stage_payload = {}
        flat_scalars = {}
        for stage_key, raw in self._stage_data.items():
            usage_scalars = self._usage_scalars(raw["usage_sum"], raw["top1_count"])
            n_tokens = max(int(raw["n_tokens"]), 1)
            entropy_mean = float(raw["entropy_sum"] / n_tokens) if raw["mode"] == "moe" else 0.0
            jitter_adj = float(raw["jitter_adj_sum"] / max(raw["jitter_adj_count"], 1))
            jitter_session = float(raw["jitter_session_sum"] / max(raw["n_sessions"], 1))
            condition_norm = float(raw["cond_norm_sum"] / max(raw["cond_norm_count"], 1))
            delta_norm = float(raw["delta_norm_sum"] / max(raw["delta_norm_count"], 1))
            sim_payload = self._expert_similarity(raw["expert_out_sum"])
            stage_payload[stage_key] = {
                "mode": raw["mode"],
                "n_slots": raw["n_slots"],
                "expert_names": list(self.stage_expert_names.get(_base_stage_name(stage_key), [])),
                "family_names": list(raw["family_names"]),
                "usage_sum": _float_list(raw["usage_sum"]),
                "top1_count": _float_list(raw["top1_count"]),
                "entropy_mean": entropy_mean,
                "route_jitter_adjacent": jitter_adj,
                "route_jitter_session": jitter_session,
                "condition_norm": condition_norm,
                "stage_delta_norm": delta_norm,
                "usage_share": usage_scalars["usage_share"],
                "n_eff": usage_scalars["n_eff"],
                "cv_usage": usage_scalars["cv_usage"],
                "dead_expert_frac": usage_scalars["dead_expert_frac"],
                "top1_max_frac": usage_scalars["top1_max_frac"],
                "feature_family_expert_heatmap": {
                    "family_names": list(raw["family_names"]),
                    "values": raw["family_expert"].tolist(),
                },
                "position_expert_usage": {
                    "values": raw["position_usage"].tolist(),
                },
                "route_transition_matrix": {
                    "values": raw["transition"].tolist(),
                },
                **sim_payload,
            }
            prefix = stage_key.replace("@", "_")
            flat_scalars[f"{prefix}.entropy_mean"] = entropy_mean
            flat_scalars[f"{prefix}.n_eff"] = usage_scalars["n_eff"]
            flat_scalars[f"{prefix}.cv_usage"] = usage_scalars["cv_usage"]
            flat_scalars[f"{prefix}.dead_expert_frac"] = usage_scalars["dead_expert_frac"]
            flat_scalars[f"{prefix}.top1_max_frac"] = usage_scalars["top1_max_frac"]
            flat_scalars[f"{prefix}.route_jitter_adjacent"] = jitter_adj
            flat_scalars[f"{prefix}.route_jitter_session"] = jitter_session
            flat_scalars[f"{prefix}.condition_norm"] = condition_norm
            flat_scalars[f"{prefix}.stage_delta_norm"] = delta_norm
            flat_scalars[f"{prefix}.expert_similarity_mean"] = sim_payload["expert_similarity_mean"]
            flat_scalars[f"{prefix}.expert_similarity_max"] = sim_payload["expert_similarity_max"]

        return {
            "split": self.split_name,
            "feature_mode": self.feature_mode,
            "stage_metrics": stage_payload,
            "scalar_metrics": flat_scalars,
        }
