#!/usr/bin/env python3
"""Unit tests for FMoEv2 feature-specialization auxiliary loss."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.feature_config import (  # noqa: E402
    ALL_FEATURE_COLUMNS,
    STAGE_ALL_FEATURES,
    build_column_to_index,
)
from models.FeaturedMoE_v2.losses import compute_feature_specialization_aux_loss  # noqa: E402


def _stage_feature_indices():
    col2idx = build_column_to_index(ALL_FEATURE_COLUMNS)
    return {
        stage: [int(col2idx[f]) for f in feats if f in col2idx]
        for stage, feats in STAGE_ALL_FEATURES.items()
    }


def test_feature_spec_aux_returns_finite_scalar():
    torch.manual_seed(7)
    bsz, tlen, n_feat = 3, 6, len(ALL_FEATURE_COLUMNS)
    n_exp = 4

    feat = torch.randn(bsz, tlen, n_feat)
    w_mid = torch.softmax(torch.randn(bsz, tlen, n_exp), dim=-1)
    w_micro = torch.softmax(torch.randn(bsz, tlen, n_exp), dim=-1)
    item_seq_len = torch.tensor([6, 5, 3], dtype=torch.long)

    loss = compute_feature_specialization_aux_loss(
        weights={"mid@1": w_mid, "micro@1": w_micro},
        feat=feat,
        stage_feature_indices=_stage_feature_indices(),
        selected_stages=["mid", "micro"],
        item_seq_len=item_seq_len,
        min_tokens_per_expert=1.0,
        aux_lambda=3e-4,
        enabled=True,
        device=feat.device,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_feature_spec_aux_respects_min_tokens_mask():
    torch.manual_seed(11)
    bsz, tlen, n_feat = 2, 4, len(ALL_FEATURE_COLUMNS)
    n_exp = 4

    feat = torch.randn(bsz, tlen, n_feat)
    w = torch.softmax(torch.randn(bsz, tlen, n_exp), dim=-1)
    item_seq_len = torch.tensor([4, 4], dtype=torch.long)

    # Larger than total possible soft mass -> all experts filtered out.
    loss = compute_feature_specialization_aux_loss(
        weights={"mid@1": w},
        feat=feat,
        stage_feature_indices=_stage_feature_indices(),
        selected_stages=["mid"],
        item_seq_len=item_seq_len,
        min_tokens_per_expert=9999.0,
        aux_lambda=1e-3,
        enabled=True,
        device=feat.device,
    )

    assert torch.allclose(loss, torch.tensor(0.0, device=feat.device))


def test_feature_spec_aux_zero_when_disabled_or_empty():
    feat = torch.zeros(2, 3, len(ALL_FEATURE_COLUMNS))

    loss_disabled = compute_feature_specialization_aux_loss(
        weights={"mid@1": torch.full((2, 3, 4), 0.25)},
        feat=feat,
        stage_feature_indices=_stage_feature_indices(),
        selected_stages=["mid"],
        item_seq_len=None,
        min_tokens_per_expert=1.0,
        aux_lambda=1e-3,
        enabled=False,
        device=feat.device,
    )
    loss_empty = compute_feature_specialization_aux_loss(
        weights={},
        feat=feat,
        stage_feature_indices=_stage_feature_indices(),
        selected_stages=["mid"],
        item_seq_len=None,
        min_tokens_per_expert=1.0,
        aux_lambda=1e-3,
        enabled=True,
        device=feat.device,
    )

    assert torch.allclose(loss_disabled, torch.tensor(0.0, device=feat.device))
    assert torch.allclose(loss_empty, torch.tensor(0.0, device=feat.device))


def test_feature_spec_aux_supports_factorized_clone_weights():
    torch.manual_seed(13)
    bsz, tlen, n_feat = 2, 5, len(ALL_FEATURE_COLUMNS)
    feat = torch.randn(bsz, tlen, n_feat)
    w = torch.zeros(bsz, tlen, 12)
    for group_idx in range(4):
        w[..., group_idx * 3] = 0.25

    loss = compute_feature_specialization_aux_loss(
        weights={"mid@1": w},
        feat=feat,
        stage_feature_indices=_stage_feature_indices(),
        selected_stages=["mid"],
        item_seq_len=torch.tensor([5, 4], dtype=torch.long),
        min_tokens_per_expert=1.0,
        aux_lambda=3e-4,
        enabled=True,
        device=feat.device,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0
