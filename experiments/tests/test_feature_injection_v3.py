"""Smoke tests for feature_injection_v3 dataset generation."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))


def _load_v3_module():
    module_path = REPO_ROOT / "Datasets" / "feature_injection_v3.py"
    spec = importlib.util.spec_from_file_location("feature_injection_v3", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_toy_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "processed" / "basic" / "ToyV3"
    root.mkdir(parents=True, exist_ok=True)
    inter_path = root / "ToyV3.inter"
    item_path = root / "ToyV3.item"

    inter_rows = [
        ("s1", "i1", 1, "u1"),
        ("s1", "i1", 2, "u1"),
        ("s1", "i2", 3, "u1"),
        ("s1", "i3", 4, "u1"),
        ("s1", "i4", 5, "u1"),
        ("s1", "i5", 6, "u1"),
        ("s3", "i2", 10, "u2"),
        ("s3", "i3", 11, "u2"),
        ("s3", "i4", 12, "u2"),
        ("s3", "i5", 13, "u2"),
        ("s3", "i6", 14, "u2"),
        ("s2", "i2", 100, "u1"),
        ("s2", "i6", 101, "u1"),
        ("s2", "i7", 102, "u1"),
        ("s2", "i8", 103, "u1"),
        ("s2", "i9", 104, "u1"),
    ]
    inter_df = pd.DataFrame(inter_rows, columns=["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
    inter_df.to_csv(inter_path, sep="\t", index=False)

    item_rows = [
        ("i1", "A"),
        ("i2", "A"),
        ("i3", "B"),
        ("i4", "B"),
        ("i5", "C"),
        ("i6", "C"),
        ("i7", "D"),
        ("i8", "D"),
        ("i9", "E"),
    ]
    item_df = pd.DataFrame(item_rows, columns=["item_id:token", "category:token"])
    item_df.to_csv(item_path, sep="\t", index=False)
    return tmp_path / "processed"


def test_feature_injection_v3_generates_expected_prefix_features(tmp_path):
    mod = _load_v3_module()
    processed_root = _write_toy_dataset(tmp_path)

    out_dir = mod.build_feature_dataset(
        processed_root=processed_root,
        dataset_name="ToyV3",
        source_subdir="basic",
        output_subdir="feature_added_v3",
        macro_windows=(5, 10),
        mid_scope="session_full",
        micro_window=5,
        fit_session_ratio=0.7,
    )

    meta = json.loads((out_dir / "feature_meta_v3.json").read_text())
    assert meta["macro_windows"] == [5, 10]
    assert meta["micro_window"] == 5
    assert len(meta["all_features"]) == 64

    df = pd.read_csv(out_dir / "ToyV3.inter", sep="\t")
    col_map = {c.split(":", 1)[0]: c for c in df.columns}

    feature_cols = [col_map[name] for name in meta["all_features"]]
    feature_df = df[feature_cols].rename(columns={col_map[name]: name for name in meta["all_features"]})
    assert ((feature_df >= 0.0) & (feature_df <= 1.0)).all().all()

    first_row = df.iloc[0]
    assert math.isclose(first_row[col_map["mac5_ctx_valid_r"]], 0.0, abs_tol=1e-9)
    assert math.isclose(first_row[col_map["mac5_theme_shift_r"]], 0.5, abs_tol=1e-9)
    assert math.isclose(first_row[col_map["mac5_gap_last"]], 0.5, abs_tol=1e-9)

    second_event_s1 = df[df[col_map["session_id"]] == "s1"].iloc[1]
    assert math.isclose(second_event_s1[col_map["mid_valid_r"]], 0.1, abs_tol=1e-9)

    sixth_event_s1 = df[df[col_map["session_id"]] == "s1"].iloc[5]
    assert math.isclose(sixth_event_s1[col_map["mic_valid_r"]], 1.0, abs_tol=1e-9)

    first_event_s2 = df[df[col_map["session_id"]] == "s2"].iloc[0]
    assert math.isclose(first_event_s2[col_map["mac5_ctx_valid_r"]], 0.2, abs_tol=1e-9)
    assert math.isclose(first_event_s2[col_map["mac10_ctx_valid_r"]], 0.1, abs_tol=1e-9)
