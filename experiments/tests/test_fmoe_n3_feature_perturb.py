#!/usr/bin/env python3
"""Feature perturb target/behavior regression tests for FeaturedMoE_N3."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")
pytest.importorskip("recbole")

from models.FeaturedMoE_N3.featured_moe_n3 import FeaturedMoE_N3  # noqa: E402


def _bare_model() -> FeaturedMoE_N3:
    return FeaturedMoE_N3.__new__(FeaturedMoE_N3)


def test_keyword_targets_override_family_targets():
    model = _bare_model()
    model.feature_perturb_keywords = ["cat", "theme"]
    model.feature_perturb_family = ["tempo"]
    model._feature_col2idx = {
        "mid_valid_r": 0,
        "mid_cat_top1": 1,
        "mid_repeat_r": 2,
        "mac5_theme_top1_mean": 3,
    }
    model._feature_family_ablation_indices = {
        "tempo": [0],
        "focus": [1, 3],
        "memory": [2],
        "exposure": [],
    }

    selected = model._selected_feature_indices_for_perturb(feature_dim=4)
    assert selected == [1, 3]


def test_category_zero_perturb_keeps_shape_and_zeros_only_target_columns():
    model = _bare_model()
    model.feature_perturb_mode = "zero"
    model.feature_perturb_apply = "both"
    model.feature_perturb_keywords = ["cat", "theme"]
    model.feature_perturb_family = []
    model._feature_col2idx = {
        "mid_valid_r": 0,
        "mid_cat_top1": 1,
        "mac5_theme_top1_mean": 2,
        "mid_repeat_r": 3,
    }
    model._feature_family_ablation_indices = {
        "tempo": [0],
        "focus": [1, 2],
        "memory": [3],
        "exposure": [],
    }
    model.training = True

    feat = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        ]
    )
    out = model._apply_feature_perturb(feat)

    assert out.shape == feat.shape
    assert torch.allclose(out[..., 1], torch.zeros_like(out[..., 1]))
    assert torch.allclose(out[..., 2], torch.zeros_like(out[..., 2]))
    assert torch.allclose(out[..., 0], feat[..., 0])
    assert torch.allclose(out[..., 3], feat[..., 3])


def test_role_swap_default_pairs_use_canonical_family_mapping():
    model = _bare_model()
    model.feature_perturb_family = []
    model._feature_family_ablation_indices = {
        "tempo": [0],
        "focus": [1],
        "memory": [2],
        "exposure": [3],
    }

    feat = torch.tensor([[[10.0, 20.0, 30.0, 40.0]]])
    out = model._apply_role_swap(feat)

    expected = torch.tensor([[[40.0, 30.0, 20.0, 10.0]]])
    assert torch.allclose(out, expected)
