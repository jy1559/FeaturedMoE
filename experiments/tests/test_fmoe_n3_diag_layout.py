#!/usr/bin/env python3
"""Diagnostics layout regression tests for P8 run_dir/diag schema."""

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from hyperopt_tune import (  # noqa: E402
    _diag_artifact_paths,
    _diag_tier_rows,
    _write_csv_gz_with_schema,
    _write_csv_with_schema,
)


def test_diag_artifact_paths_use_run_dir_diag_layout(tmp_path):
    result_file = tmp_path / "result.json"
    result_file.write_text("{}", encoding="utf-8")
    paths = _diag_artifact_paths(
        result_file=result_file,
        dataset="KuaiRecLargeStrictPosV2_0.2",
        model="FeaturedMoE_N3",
        run_group="fmoe_n3",
        run_axis="phase8",
        run_phase="P8",
    )
    for key, path in paths.items():
        assert str(path).startswith(str((tmp_path / "diag").resolve()) if path.is_absolute() else str(tmp_path / "diag"))
        assert "diag" in str(path)
        assert key


def test_diag_tier_rows_reads_payload():
    payload = {
        "diag_tiers": {
            "tier_a_final": [{"node_name": "final.expert", "stage_name": "mid"}],
            "tier_b_internal": [{"node_name": "primitive.a_joint", "stage_name": "mid"}],
        }
    }
    a_rows = _diag_tier_rows(payload, tier_key="tier_a_final")
    b_rows = _diag_tier_rows(payload, tier_key="tier_b_internal")
    assert len(a_rows) == 1
    assert len(b_rows) == 1
    assert a_rows[0]["node_name"] == "final.expert"
    assert b_rows[0]["node_name"] == "primitive.a_joint"


def test_diag_csv_helpers_write_empty_schema(tmp_path):
    plain_path = tmp_path / "diag" / "tier_a_final" / "final_metrics.csv"
    gz_path = tmp_path / "diag" / "tier_c_viz" / "viz_feature_pca.csv.gz"
    _write_csv_with_schema(
        plain_path,
        rows=[],
        fieldnames=["stage_name", "node_name", "knn_consistency_score"],
    )
    _write_csv_gz_with_schema(
        gz_path,
        rows=[],
        fieldnames=["pc1", "pc2", "stage_name"],
    )
    assert plain_path.exists()
    assert gz_path.exists()
