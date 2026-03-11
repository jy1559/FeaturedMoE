"""Auxiliary losses for FeaturedMoE_v2."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

from ..FeaturedMoE.routers import load_balance_loss


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
    """Map features to [0,1] ratio space for stable variance regularization."""
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
    """Encourage feature-consistent routing by minimizing within-expert feature dispersion."""
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

        # mu_k = sum_n w_nk * x_n / sum_n w_nk
        mu = (w_flat.transpose(0, 1) @ x_flat) / sum_w.clamp(min=1e-8).unsqueeze(-1)

        # var_k = sum_n w_nk * ||x_n - mu_k||^2 / sum_n w_nk
        sq_dist = (x_flat.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(dim=-1)
        var = (w_flat * sq_dist).sum(dim=0) / sum_w.clamp(min=1e-8)
        stage_losses.append(var[valid_expert].mean())

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return float(aux_lambda) * torch.stack(stage_losses).mean()


def _clone_centers(expert_scale: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vals = [(2 * i + 1) / float(2 * expert_scale) for i in range(expert_scale)]
    return torch.tensor(vals, device=device, dtype=dtype)


def _stage_teacher_logits(
    feat: torch.Tensor,
    feature_groups: Sequence[Sequence[int]],
    expert_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_scores = []
    for feat_idx in feature_groups:
        idx = torch.tensor(list(feat_idx), device=feat.device, dtype=torch.long)
        if idx.numel() == 0:
            idx = torch.tensor([0], device=feat.device, dtype=torch.long)
        x = feat.index_select(-1, idx)
        ratio = _to_ratio(x)
        bins = torch.floor(ratio * 5.0).clamp(0, 4)
        bin_center = (bins + 0.5) / 5.0
        score = bin_center.mean(dim=-1)
        group_scores.append(score)

    teacher_group_logits = torch.stack(group_scores, dim=-1)
    centers = _clone_centers(expert_scale, device=feat.device, dtype=feat.dtype)
    sigma = 0.20
    teacher_clone_logits = []
    for score in group_scores:
        score_exp = score.unsqueeze(-1)
        logits = -((score_exp - centers) ** 2) / (2.0 * sigma * sigma)
        teacher_clone_logits.append(logits)
    teacher_clone_logits = torch.stack(teacher_clone_logits, dim=-2)
    return teacher_group_logits, teacher_clone_logits


def _student_group_and_clone_logits(
    gate_logits: torch.Tensor,
    expert_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_logits.shape[-1] % expert_scale != 0:
        raise ValueError(
            f"gate_logits last dim {gate_logits.shape[-1]} is not divisible by expert_scale={expert_scale}"
        )
    n_groups = gate_logits.shape[-1] // expert_scale
    clone_logits = gate_logits.view(*gate_logits.shape[:-1], n_groups, expert_scale)
    group_logits = torch.logsumexp(clone_logits, dim=-1)
    return group_logits, clone_logits


def compute_router_distill_aux_loss(
    *,
    gate_logits: Dict[str, torch.Tensor],
    feat: Optional[torch.Tensor],
    stage_group_feature_indices: Dict[str, Sequence[Sequence[int]]],
    item_seq_len: Optional[torch.Tensor],
    expert_scale: int,
    mode: str,
    enabled: bool,
    lambda_group: float,
    lambda_clone: float,
    distill_temperature: float,
    progress: float,
    until: float,
    device: torch.device,
) -> torch.Tensor:
    if not enabled or feat is None or not gate_logits or mode == "none":
        return torch.tensor(0.0, device=device)
    if float(lambda_group) <= 0 and float(lambda_clone) <= 0:
        return torch.tensor(0.0, device=device)

    until = float(until)
    if until <= 0:
        return torch.tensor(0.0, device=device)

    progress = max(float(progress), 0.0)
    if progress >= until:
        return torch.tensor(0.0, device=device)

    tau = max(float(distill_temperature), 1e-6)
    weight = max(0.0, 1.0 - (progress / until))
    stage_losses = []

    for stage_key, student_gate_logits in sorted(gate_logits.items()):
        base_stage = str(stage_key).split("@", 1)[0].strip().lower()
        feature_groups = stage_group_feature_indices.get(base_stage)
        if not feature_groups or student_gate_logits.ndim != 3:
            continue

        try:
            student_group_logits, student_clone_logits = _student_group_and_clone_logits(
                student_gate_logits,
                expert_scale=expert_scale,
            )
        except ValueError:
            continue
        teacher_group_logits, teacher_clone_logits = _stage_teacher_logits(
            feat=feat,
            feature_groups=feature_groups,
            expert_scale=expert_scale,
        )
        if teacher_group_logits.shape[:2] != student_group_logits.shape[:2]:
            continue

        _, tlen, _ = teacher_group_logits.shape
        if item_seq_len is not None:
            lens = item_seq_len.to(device=student_gate_logits.device).long()
            valid_mask = torch.arange(tlen, device=student_gate_logits.device).unsqueeze(0) < lens.unsqueeze(1)
            if not valid_mask.any():
                continue
            teacher_group_logits = teacher_group_logits[valid_mask]
            student_group_logits = student_group_logits[valid_mask]
            teacher_clone_logits = teacher_clone_logits[valid_mask]
            student_clone_logits = student_clone_logits[valid_mask]
        else:
            teacher_group_logits = teacher_group_logits.reshape(-1, teacher_group_logits.shape[-1])
            student_group_logits = student_group_logits.reshape(-1, student_group_logits.shape[-1])
            teacher_clone_logits = teacher_clone_logits.reshape(
                -1, teacher_clone_logits.shape[-2], teacher_clone_logits.shape[-1]
            )
            student_clone_logits = student_clone_logits.reshape(
                -1, student_clone_logits.shape[-2], student_clone_logits.shape[-1]
            )

        if teacher_group_logits.numel() == 0:
            continue

        stage_loss = torch.tensor(0.0, device=device)
        if mode in {"group_only", "group_plus_clone"} and float(lambda_group) > 0:
            teacher_prob = F.softmax(teacher_group_logits / tau, dim=-1)
            student_log_prob = F.log_softmax(student_group_logits / tau, dim=-1)
            stage_loss = stage_loss + float(lambda_group) * (tau * tau) * F.kl_div(
                student_log_prob, teacher_prob, reduction="batchmean"
            )

        if mode in {"clone_only", "group_plus_clone"} and float(lambda_clone) > 0:
            teacher_clone_logits = teacher_clone_logits.reshape(-1, teacher_clone_logits.shape[-1])
            student_clone_logits = student_clone_logits.reshape(-1, student_clone_logits.shape[-1])
            teacher_prob = F.softmax(teacher_clone_logits / tau, dim=-1)
            student_log_prob = F.log_softmax(student_clone_logits / tau, dim=-1)
            stage_loss = stage_loss + float(lambda_clone) * (tau * tau) * F.kl_div(
                student_log_prob, teacher_prob, reduction="batchmean"
            )

        if torch.isfinite(stage_loss):
            stage_losses.append(stage_loss)

    if not stage_losses:
        return torch.tensor(0.0, device=device)

    return weight * torch.stack(stage_losses).mean()
