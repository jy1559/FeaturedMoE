#!/usr/bin/env python3
"""Unit tests for aggregated special slice metrics."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
torch = pytest.importorskip("torch")

from models.FeaturedMoE.run_logger import RunLogger  # noqa: E402
from models.FeaturedMoE.special_metrics import SpecialMetricCollector, default_new_user_field  # noqa: E402


def test_special_metric_collector_aggregates_expected_slices(tmp_path):
    item_counts = torch.tensor([0, 2, 7, 3, 25, 150], dtype=torch.long)
    collector = SpecialMetricCollector(
        split_name="valid",
        item_counts=item_counts,
        item_seq_len_field="item_length",
        new_user_field=default_new_user_field(),
        config_snapshot={"enabled": True},
    )

    scores = torch.tensor(
        [
            [0.0, 0.2, 0.3, 0.9, 0.4, 0.1],
            [0.0, 0.1, 0.2, 0.3, 0.8, 0.4],
            [0.0, 0.9, 0.7, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    interaction = {
        "item_length": torch.tensor([2, 5, 11]),
        default_new_user_field(): torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }
    collector.update(
        interaction=interaction,
        scores=scores,
        positive_u=torch.tensor([0, 1, 2]),
        positive_i=torch.tensor([3, 4, 2]),
    )

    summary = collector.finalize()
    assert summary["overall"]["count"] == 3
    assert "target_popularity_abs" in summary["slices"]
    assert "cold_0" in summary["slices"]["target_popularity_abs"]
    assert "rare_1_5" in summary["slices"]["target_popularity_abs"]
    assert "21_100" in summary["slices"]["target_popularity_abs"]
    assert "target_popularity_abs_legacy" in summary["slices"]
    assert "<=5" in summary["slices"]["target_popularity_abs_legacy"]
    assert "session_len" in summary["slices"]
    assert "<=7" in summary["slices"]["session_len"]
    assert "session_len_legacy" in summary["slices"]
    assert "1-2" in summary["slices"]["session_len_legacy"]
    assert "new_user" in summary["slices"]

    logger = RunLogger(run_name="special_metrics_test", config={"model": "FeaturedMoE_N"}, output_root=tmp_path)
    logger.log_special_metrics(valid_special_metrics=summary, test_special_metrics=summary)
    assert (tmp_path / "special_metrics_test" / "special_metrics.json").exists()
