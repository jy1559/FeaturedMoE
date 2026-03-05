"""
Router / Gating modules for the MoE stages.

Supports:
  - Dense softmax gating (all K experts contribute)
  - Top-k sparse gating (only top-k experts contribute; rest zeroed)
  - Load-balance regularisation loss for collapse prevention
  - Entropy regularisation (optional)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """Gating network for one MoE stage.

    Parameters
    ----------
    d_in : int
        Dimension of the router input (= concat of stage features + prev stage
        output, when applicable).
    n_experts : int
        Number of experts (K).
    d_hidden : int
        Hidden size of the 2-layer MLP.
    top_k : int or None
        If None → dense softmax (all experts).
        If int  → top-k sparse gating (only top-k experts, rest weight=0).
    dropout : float
        Dropout in router MLP.
    """

    def __init__(
        self,
        d_in: int,
        n_experts: int = 4,
        d_hidden: int = 64,
        top_k: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_experts),
        )

    def forward(
        self,
        router_input: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gating weights.

        Args:
            router_input: [*, d_in]
            temperature: Softmax temperature for gating logits.
            top_k: Optional runtime override for sparse top-k routing.
                - ``None``: use module default ``self.top_k``.
                - ``<=0``: dense softmax over all experts.
                - ``>0``: sparse top-k softmax.
        Returns:
            weights : [*, K]   — soft gating weights (sum=1 per sample).
            logits  : [*, K]   — raw logits (for auxiliary losses / logging).
        """
        logits = self.net(router_input)  # [*, K]
        scale = max(float(temperature), 1e-6)
        scaled_logits = logits / scale

        active_top_k = self.top_k if top_k is None else int(top_k)
        if active_top_k is not None and active_top_k <= 0:
            active_top_k = None

        if active_top_k is not None and active_top_k < self.n_experts:
            weights = self._top_k_softmax(scaled_logits, active_top_k)
        else:
            weights = F.softmax(scaled_logits, dim=-1)

        return weights, scaled_logits

    # ------------------------------------------------------------------
    @staticmethod
    def _top_k_softmax(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k logits, re-normalise with softmax, zero the rest."""
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        # Re-normalise within top-k
        topk_weights = F.softmax(topk_vals, dim=-1)  # [*, k]
        # Scatter back into full-size tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(-1, topk_idx, topk_weights)
        return weights


# -----------------------------------------------------------------------
# Auxiliary losses
# -----------------------------------------------------------------------

def load_balance_loss(
    gate_weights: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Compute load-balance loss: encourages uniform expert utilisation.

    L_balance = Σ_k (mean(g_{:,k}) − 1/K)²

    Args:
        gate_weights: [B, (T,) K]  — gating weights from the router.
        n_experts: K
    Returns:
        Scalar loss.
    """
    # Flatten all dims except the last (expert dim)
    flat = gate_weights.reshape(-1, n_experts)  # [N, K]
    mean_load = flat.mean(dim=0)  # [K]
    target = 1.0 / n_experts
    return ((mean_load - target) ** 2).sum()


def router_entropy_loss(
    gate_weights: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Negative entropy of the mean gating distribution — encourages diversity.

    Maximising entropy ↔ minimising negative entropy.

    Args:
        gate_weights: [B, (T,) K]
        n_experts: K
    Returns:
        Scalar loss (lower = more uniform distribution).
    """
    flat = gate_weights.reshape(-1, n_experts)  # [N, K]
    mean_dist = flat.mean(dim=0)  # [K]
    # Clamp for numerical safety
    mean_dist = mean_dist.clamp(min=1e-8)
    entropy = -(mean_dist * mean_dist.log()).sum()
    # Return negative entropy so that minimising this loss maximises entropy
    return -entropy
