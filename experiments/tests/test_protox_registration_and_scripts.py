#!/usr/bin/env python3
"""Registration/config/script dry-run tests for FeaturedMoE_ProtoX."""

from pathlib import Path
import subprocess
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def test_protox_hydra_config_load():
    hydra_utils = pytest.importorskip("hydra_utils")
    cfg = hydra_utils.load_hydra_config(
        config_dir=ROOT_DIR / "configs",
        config_name="config",
        overrides=[
            "model=featured_moe_protox",
            "dataset=movielens1m",
            "eval_mode=session",
            "feature_mode=full_v2",
        ],
    )
    model_name = str(cfg.get("model", "")).lower()
    assert "protox" in model_name
    assert int(cfg.get("proto_num", 0)) > 0
    assert "proto_temperature_start" in cfg
    assert "stage_delta_scale" in cfg


def test_recbole_patch_contains_protox_registration_aliases():
    text = (ROOT_DIR / "recbole_patch.py").read_text(encoding="utf-8")
    assert "FeaturedMoE_ProtoX" in text
    assert "'featured_moe_protox'" in text
    assert "'featuredmoe_protox'" in text


def test_protox_scripts_dry_run():
    commands = [
        [
            ROOT_DIR / "run/fmoe_protox/run_first_pass_protox.sh",
            "--datasets",
            "movielens1m,retail_rocket",
            "--gpus",
            "4,5,6,7",
            "--combos-per-gpu",
            "3",
            "--dry-run",
        ],
        [
            ROOT_DIR / "run/fmoe_protox/run_final_group.sh",
            "--gpus",
            "0,1",
            "--datasets",
            "movielens1m,retail_rocket",
            "--combos-per-gpu",
            "1",
            "--dry-run",
        ],
        [
            ROOT_DIR / "run/fmoe_protox/run_protox_group.sh",
            "--gpus",
            "2,3",
            "--datasets",
            "movielens1m,retail_rocket",
            "--combos-per-gpu",
            "1",
            "--dry-run",
        ],
        [
            ROOT_DIR / "run/fmoe_protox/pipeline_split_v2_protox.sh",
            "--group-a-gpus",
            "0,1",
            "--group-b-gpus",
            "2,3",
            "--datasets",
            "movielens1m,retail_rocket",
            "--combos-per-gpu",
            "1",
            "--dry-run",
        ],
    ]

    outputs = []
    for cmd in commands:
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
        outputs.append(proc.stdout)

    assert "ml1_total_jobs=12" in outputs[0]
    assert "[FINAL_ONLY]" in outputs[1]
    assert "[PROTOX_ONLY]" in outputs[2]
    assert "role=fmoe_protox_first" in outputs[3]
