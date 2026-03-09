#!/usr/bin/env python3
"""Registration/config/script dry-run tests for FeaturedMoE_HiR2 plan."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_hir2_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_hir2",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "hir2" in model_name
    assert cfg.get("hir2_stage_merge_mode") in {"serial_weighted", "parallel_weighted"}
    assert "hir2_stage_allocator_top_k" in cfg


def test_recbole_patch_contains_hir2_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_HiR2" in text
    assert "'featured_moe_hir2'" in text
    assert "'featuredmoe_hir2'" in text


def test_new_scripts_dry_run():
    scripts = [
        [ROOT_DIR / "run/fmoe_v2/final_v2_ml1_rr.sh", "--datasets", "movielens1m,retail_rocket", "--gpus", "0,1", "--dry-run"],
        [ROOT_DIR / "run/fmoe_hir2/run_first_pass_hir2.sh", "--datasets", "movielens1m,retail_rocket", "--gpus", "2,3", "--dry-run"],
        [
            ROOT_DIR / "run/fmoe_hir2/pipeline_split_v2_hir2.sh",
            "--group-a-gpus",
            "0,1",
            "--group-b-gpus",
            "2,3",
            "--datasets",
            "movielens1m,retail_rocket",
            "--dry-run",
        ],
    ]

    for cmd in scripts:
        proc = subprocess.run(
            ["bash", str(cmd[0]), *[str(x) for x in cmd[1:]]],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, f"script failed: {cmd[0]}\nstdout={proc.stdout}\nstderr={proc.stderr}"
