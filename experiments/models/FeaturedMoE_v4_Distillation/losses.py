"""Auxiliary losses and rule-teacher utilities for FeaturedMoE_v4_Distillation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..FeaturedMoE.routers import build_group_local_stat_logits_12way, load_balance_loss


def _masked_load_balance(gate_weights: torch.Tensor, item_seq_len: Optional[torch.Tensor]) -> torch.Tensor:
    if item_seq_len is None:
        return load_balance_loss(gate_weights, n_experts=gate_weights.shape[-1])

    _, tlen, n_experts = gate_weights.shape
    lens = item_seq_len.to(device=gate_weights.device).long()
    valid = torch.arange(tlen, device=gate_weights.device).unsqueeze(0) < lens.unsqueeze(1)
    if not valid.any():
        return torch.tensor(0.0, device=gate_weights.device)
    return load_balance_loss(gate_weights[valid], n_experts=n_experts)


def _to_ratio(values: torch.Tensor) -> torch.Tensor:
    """Map features to [0,1] ratio space for stable rule extraction."""
    if values.numel() == 0:
        return values

    x = values.float()
    finite = torch.isfinite(x)
    if not finite.any():
        return torch.zeros_like(x)

    xf = x[finite]
    lo = xf.min()
    hi = xf.max()
    if lo >= -1e-6 and hi <= 1.0 + 1e-6:
        ratio = x.clamp(0.0, 1.0)
        ratio[~finite] = 0.5
        return ratio

    ratio = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    ratio = ratio.clamp(0.0, 1.0)
    ratio[~finite] = 0.5
    return ratio


def compute_expert_aux_loss(
    weights: Dict[str, torch.Tensor],
    item_seq_len: Optional[torch.Tensor],
    balance_lambda: float,
    device: torch.device,
) -> torch.Tensor:
    if not weights:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for w in weights.values():
        total = total + _masked_load_balance(w, item_seq_len=item_seq_len)
    return float(balance_lambda) * total


def compute_stage_merge_aux_loss(
    stage_merge_weights: Optional[torch.Tensor],
    *,
    item_seq_len: Optional[torch.Tensor],
    balance_lambda: float,
    enabled: bool,
    scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not enabled or stage_merge_weights is None:
        return torch.tensor(0.0, device=device)
    stage_aux = _masked_load_balance(stage_merge_weights, item_seq_len=item_seq_len)
    return float(balance_lambda) * float(scale) * stage_aux


def compute_feature_specialization_aux_loss(
    *,
    weights: Dict[str, torch.Tensor],
    feat: Optional[torch.Tensor],
    stage_feature_indices: Dict[str, Sequence[int]],
    selected_stages: Iterable[str],
    item_seq_len: Optional[torch.Tensor],
    min_tokens_per_expert: float,
    aux_lambda: float,
    enabled: bool,
    device: torch.device,
) -> torch.Tensor:
    """Encourage feature-consistent routing by minimizing within-expert dispersion."""
    if (not enabled) or feat is None or (not weights):
        return torch.tensor(0.0, device=device)
    if float(aux_lambda) <= 0:
        return torch.tensor(0.0, device=device)

    stage_whitelist = {str(s).strip().lower() for s in selected_stages if str(s).strip()}
    if not stage_whitelist:
        return torch.tensor(0.0, device=device)

    bsz, tlen, _ = feat.shape
    if item_seq_len is not None:
        lens = item_seq_len.to(device=feat.device).long()
        valid_mask = torch.arange(tlen, device=feat.device).unsqueeze(0) < lens.unsqueeze(1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)
    else:
        valid_mask = torch.ones(bsz, tlen, dtype=torch.bool, device=feat.device)

    stage_losses = []
    min_tokens = max(float(min_tokens_per_expert), 0.0)
    for stage_key, gate_w in weights.items():
        base_stage = str(stage_key).split("@", 1)[0].strip().lower()
        if base_stage not in stage_whitelist:
            continue

        feat_idx = list(stage_feature_indices.get(base_stage, ()))
        if not feat_idx:
            continue

        idx = torch.tensor(feat_idx, device=feat.device, dtype=torch.long)
        x = feat.index_select(-1, idx)
        x = _to_ratio(x)

        w_flat = gate_w[valid_mask]
        x_flat = x[valid_mask]
        if w_flat.numel() == 0:
            continue

        sum_w = w_flat.sum(dim=0)
        valid_expert = sum_w >= min_tokens
        if not valid_expert.any():
            continue

        mu = (w_flat.transpose(0, 1) @ x_flat) / sum_w.clamp(min=1e-8).unsqueeze(-1)
        sq_dist = (x_flat.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(dim=-1)
        var = (w_flat * sq_dist).sum(dim=0) / sum_w.clamp(min=1e-8)
        stage_losses.append(var[valid_expert].mean())

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return float(aux_lambda) * torch.stack(stage_losses).mean()


def teacher_mask_applies(stage_name: str, stage_mask: str) -> bool:
    mask_key = str(stage_mask).lower().strip()
    stage_key = str(stage_name).lower().strip()
    if mask_key == "all":
        return True
    if mask_key == "mid_micro_only":
        return stage_key in {"mid", "micro"}
    raise ValueError(f"Unsupported teacher_stage_mask={stage_mask}")


def _pool_session_tensor(
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    item_seq_len: Optional[torch.Tensor],
    session_pooling: str,
) -> torch.Tensor:
    mode = str(session_pooling).lower().strip()
    if mode == "query":
        mode = "mean"
    if mode == "last":
        if item_seq_len is None:
            idx = torch.full(
                (values.size(0),),
                fill_value=max(values.size(1) - 1, 0),
                dtype=torch.long,
                device=values.device,
            )
        else:
            idx = item_seq_len.to(device=values.device).long().clamp(min=1, max=values.size(1)) - 1
        pooled = values[torch.arange(values.size(0), device=values.device), idx]
    else:
        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp(min=1.0)
        pooled = (values * weights).sum(dim=1) / denom
    return pooled.unsqueeze(1).expand(-1, values.size(1), -1)


def _prepare_teacher_features(
    values: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    item_seq_len: Optional[torch.Tensor],
    router_mode: str,
    session_pooling: str,
) -> torch.Tensor:
    if str(router_mode).lower().strip() != "session":
        return values
    return _pool_session_tensor(
        values,
        valid_mask=valid_mask,
        item_seq_len=item_seq_len,
        session_pooling=session_pooling,
    )


def _clone_centers(expert_scale: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = [(2 * i + 1) / float(2 * expert_scale) for i in range(expert_scale)]
    return torch.tensor(vals, device=device, dtype=dtype)


def _gaussian_clone_logits(
    score: torch.Tensor,
    *,
    expert_scale: int,
    sigma: float = 0.20,
) -> torch.Tensor:
    centers = _clone_centers(expert_scale, device=score.device, dtype=score.dtype)
    score_exp = score.unsqueeze(-1)
    return -((score_exp - centers) ** 2) / (2.0 * sigma * sigma)


def _compute_ratio_stats(values: torch.Tensor) -> Dict[str, torch.Tensor]:
    ratio = _to_ratio(values)
    mean = ratio.mean(dim=-1, keepdim=True)
    std = ratio.std(dim=-1, unbiased=False, keepdim=True)
    vmax = ratio.max(dim=-1, keepdim=True).values
    vmin = ratio.min(dim=-1, keepdim=True).values
    vrange = vmax - vmin
    peak = vmax - mean
    zero_frac = (ratio <= 0.10).float().mean(dim=-1, keepdim=True)
    low_frac = (ratio <= 0.25).float().mean(dim=-1, keepdim=True)
    mid_frac = ((ratio >= 0.30) & (ratio <= 0.70)).float().mean(dim=-1, keepdim=True)
    high_frac = (ratio >= 0.75).float().mean(dim=-1, keepdim=True)
    spike_frac = (ratio >= 0.90).float().mean(dim=-1, keepdim=True)
    mean_abs_dev = (ratio - mean).abs().mean(dim=-1, keepdim=True)
    return {
        "ratio": ratio,
        "mean": mean,
        "std": std,
        "vmax": vmax,
        "vmin": vmin,
        "vrange": vrange,
        "peak": peak,
        "zero_frac": zero_frac,
        "low_frac": low_frac,
        "mid_frac": mid_frac,
        "high_frac": high_frac,
        "spike_frac": spike_frac,
        "mean_abs_dev": mean_abs_dev,
    }


def _group_local_stat_clone_logits(stats: Dict[str, torch.Tensor], sharpness: float) -> torch.Tensor:
    mean = stats["mean"]
    std = stats["std"]
    vmax = stats["vmax"]
    vrange = stats["vrange"]
    peak = stats["peak"]
    zero_frac = stats["zero_frac"]
    low_frac = stats["low_frac"]
    mid_frac = stats["mid_frac"]
    high_frac = stats["high_frac"]
    spike_frac = stats["spike_frac"]
    sharp = float(sharpness)
    low = (
        1.1 * zero_frac
        + 0.7 * low_frac
        - sharp * (mean - 0.18).pow(2)
        - 1.1 * std
        - 0.4 * peak
        - 0.3 * high_frac
    )
    mid = (
        0.8 * mid_frac
        - sharp * (mean - 0.50).pow(2)
        - 0.5 * std
        - 0.2 * vrange
        - 0.2 * peak
    )
    high = (
        1.2 * high_frac
        + 0.9 * spike_frac
        + 0.7 * peak
        + 0.5 * std
        + 0.3 * vrange
        - sharp * (mean - 0.78).pow(2)
        - 0.3 * zero_frac
    )
    return torch.cat([low, mid, high], dim=-1)


def _group_local_activity_logit(stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"]
    std = stats["std"]
    vrange = stats["vrange"]
    peak = stats["peak"]
    zero_frac = stats["zero_frac"]
    high_frac = stats["high_frac"]
    spike_frac = stats["spike_frac"]
    return (
        0.9 * mean
        + 0.7 * high_frac
        + 0.4 * spike_frac
        + 0.5 * peak
        + 0.35 * std
        + 0.20 * vrange
        - 0.80 * zero_frac
    )


def _stack_group_stat(group_stats: Sequence[Dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.cat([stats[key] for stats in group_stats], dim=-1)


def _group_competition_logits_from_stats(
    group_stats: Sequence[Dict[str, torch.Tensor]],
    *,
    sharpness: float,
) -> torch.Tensor:
    mean = _stack_group_stat(group_stats, "mean")
    std = _stack_group_stat(group_stats, "std")
    vrange = _stack_group_stat(group_stats, "vrange")
    peak = _stack_group_stat(group_stats, "peak")
    zero_frac = _stack_group_stat(group_stats, "zero_frac")
    high_frac = _stack_group_stat(group_stats, "high_frac")
    spike_frac = _stack_group_stat(group_stats, "spike_frac")
    strength = (
        1.0 * mean
        + 0.9 * high_frac
        + 0.6 * spike_frac
        + 0.8 * peak
        + 0.45 * std
        + 0.25 * vrange
        - 0.95 * zero_frac
    )
    centered = strength - strength.mean(dim=-1, keepdim=True)
    normalized = centered / (strength.std(dim=-1, keepdim=True, unbiased=False) + 1e-6)
    gain = max(float(sharpness) / 8.0, 1.0)
    return gain * normalized


def _group_shape_clone_logits(stats: Dict[str, torch.Tensor], sharpness: float) -> torch.Tensor:
    if stats["mean"].shape[-1] != 1:
        raise ValueError("Expected per-group scalar stats.")
    feature_vec = torch.cat(
        [
            stats["zero_frac"],
            stats["mean"],
            stats["std"],
            stats["high_frac"],
            stats["spike_frac"],
            stats["peak"],
        ],
        dim=-1,
    )
    prototypes = torch.tensor(
        [
            [0.78, 0.16, 0.08, 0.05, 0.02, 0.08],
            [0.28, 0.48, 0.18, 0.22, 0.08, 0.18],
            [0.06, 0.78, 0.30, 0.56, 0.32, 0.42],
        ],
        device=feature_vec.device,
        dtype=feature_vec.dtype,
    )
    weights = torch.tensor(
        [1.4, 1.0, 0.8, 1.2, 1.1, 1.0],
        device=feature_vec.device,
        dtype=feature_vec.dtype,
    )
    diff = feature_vec.unsqueeze(-2) - prototypes.view(1, 1, 3, 6)
    sq_dist = (diff.pow(2) * weights.view(1, 1, 1, 6)).sum(dim=-1)
    gain = max(float(sharpness) / 2.0, 1.0)
    return -gain * sq_dist


def build_teacher_logits_12way(
    *,
    teacher_design: str,
    stage_feat: torch.Tensor,
    feature_groups: Sequence[Sequence[int]],
    expert_scale: int,
    valid_mask: torch.Tensor,
    item_seq_len: Optional[torch.Tensor],
    router_mode: str,
    session_pooling: str,
    stat_sharpness: float,
) -> Optional[torch.Tensor]:
    """Build non-learned 12-way teacher logits for one stage."""
    design = str(teacher_design).lower().strip()
    if design == "none":
        return None

    if design in {"group_local_stat12", "group_comp_stat12", "group_comp_shape12"}:
        prepared_stage_feat = _prepare_teacher_features(
            stage_feat,
            valid_mask=valid_mask,
            item_seq_len=item_seq_len,
            router_mode=router_mode,
            session_pooling=session_pooling,
        )
        if design == "group_local_stat12":
            return build_group_local_stat_logits_12way(
                prepared_stage_feat,
                feature_groups=feature_groups,
                expert_scale=expert_scale,
                stat_sharpness=stat_sharpness,
            )

        group_stats = []
        for feat_idx in feature_groups:
            idx = torch.tensor(list(feat_idx), device=prepared_stage_feat.device, dtype=torch.long)
            if idx.numel() == 0:
                idx = torch.tensor([0], device=prepared_stage_feat.device, dtype=torch.long)
            group_feat = prepared_stage_feat.index_select(-1, idx)
            group_stats.append(_compute_ratio_stats(group_feat))

        per_group_logits = []
        if design == "group_comp_stat12":
            group_bias = _group_competition_logits_from_stats(group_stats, sharpness=stat_sharpness)
        elif design == "group_comp_shape12":
            group_bias = _group_competition_logits_from_stats(group_stats, sharpness=stat_sharpness)
        else:
            group_bias = None

        for group_idx, stats in enumerate(group_stats):
            if design == "group_comp_shape12" and expert_scale == 3:
                clone_logits = _group_shape_clone_logits(stats, sharpness=stat_sharpness)
            elif expert_scale == 3:
                clone_logits = _group_local_stat_clone_logits(stats, sharpness=stat_sharpness)
            else:
                clone_logits = _gaussian_clone_logits(stats["mean"], expert_scale=expert_scale)

            local_activity = _group_local_activity_logit(stats)
            if group_bias is not None:
                logits = local_activity + group_bias[..., group_idx : group_idx + 1] + clone_logits
            else:
                logits = local_activity + clone_logits
            per_group_logits.append(logits)
        return torch.cat(per_group_logits, dim=-1)

    raise ValueError(f"Unsupported teacher_design={teacher_design}")


def compute_teacher_distill_aux_loss(
    *,
    router_stage_aux: Dict[str, Dict[str, torch.Tensor]],
    item_seq_len: Optional[torch.Tensor],
    teacher_delivery: str,
    teacher_kl_lambda: float,
    teacher_temperature: float,
    progress: float,
    until: float,
    device: torch.device,
) -> torch.Tensor:
    delivery = str(teacher_delivery).lower().strip()
    if delivery not in {"distill_kl", "distill_and_fused_bias"}:
        return torch.tensor(0.0, device=device)
    if float(teacher_kl_lambda) <= 0:
        return torch.tensor(0.0, device=device)
    until = float(until)
    if until <= 0:
        return torch.tensor(0.0, device=device)
    progress = max(float(progress), 0.0)
    if progress >= until:
        return torch.tensor(0.0, device=device)

    tau = max(float(teacher_temperature), 1e-6)
    weight = max(0.0, 1.0 - (progress / until))
    stage_losses = []

    for _, aux in sorted(router_stage_aux.items()):
        if not aux:
            continue
        student_logits = aux.get("student_logits_raw")
        teacher_logits = aux.get("teacher_logits")
        if student_logits is None or teacher_logits is None:
            continue
        if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
            continue

        if item_seq_len is not None:
            _, tlen, _ = student_logits.shape
            lens = item_seq_len.to(device=student_logits.device).long()
            valid_mask = torch.arange(tlen, device=student_logits.device).unsqueeze(0) < lens.unsqueeze(1)
            if not valid_mask.any():
                continue
            student_logits = student_logits[valid_mask]
            teacher_logits = teacher_logits[valid_mask]
        else:
            student_logits = student_logits.reshape(-1, student_logits.shape[-1])
            teacher_logits = teacher_logits.reshape(-1, teacher_logits.shape[-1])

        teacher_prob = F.softmax(teacher_logits / tau, dim=-1)
        student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
        stage_loss = (tau * tau) * F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        if torch.isfinite(stage_loss):
            stage_losses.append(stage_loss)

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return float(teacher_kl_lambda) * weight * torch.stack(stage_losses).mean()
