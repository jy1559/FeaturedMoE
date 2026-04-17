#!/usr/bin/env python3
"""Tests for FMoE_N4 ablation helpers."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from run.fmoe_n4.ablation import common  # noqa: E402


KUAI_RESULT = (
    ROOT_DIR.parent
    / "experiments"
    / "run"
    / "artifacts"
    / "results"
    / "fmoe_n4"
    / "KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_p4s3_s3_kuaireclargestrictposv2_0_2_s02_h14_seen_hi_s1_20260415_081315_824627_pid124072.json"
)

BEAUTY_RESULT = (
    ROOT_DIR.parent
    / "experiments"
    / "run"
    / "artifacts"
    / "results"
    / "fmoe_n4"
    / "beauty_FeaturedMoE_N3_p4xd_xd_beauty_b25_lr_h8_seen_anchor_s1_20260416_030201_046233_pid244350.json"
)


def test_select_top_result_candidates_dedupes_by_run_phase():
    selected = common.select_top_result_candidates(
        common.RESULT_ROOT,
        ["beauty", "KuaiRecLargeStrictPosV2_0.2"],
        topk_per_dataset=2,
    )
    assert len(selected) == 4
    run_phases = [item.run_phase for item in selected]
    assert len(run_phases) == len(set(run_phases))
    assert any(item.dataset == "beauty" for item in selected)
    assert any(item.dataset == "KuaiRecLargeStrictPosV2_0.2" for item in selected)


def test_resolve_base_spec_from_manifest_for_kuairec():
    base = common.resolve_base_spec(KUAI_RESULT)
    assert base["source"] == "manifest"
    assert base["dataset"] == "KuaiRecLargeStrictPosV2_0.2"
    assert base["setting_id"] == "S02_h14_seen_hi"
    assert base["best_learning_rate"] > 0
    assert base["overrides"]["stage_router_source"] == {"macro": "both", "mid": "both", "micro": "both"}
    assert base["fixed_values"]["d_feat_emb"] == 24
    assert base["fixed_values"]["MAX_ITEM_LIST_LENGTH"] == 25
    assert base["overrides"]["route_consistency_lambda"] == 0.00025


def test_resolve_base_spec_from_command_log_for_beauty():
    base = common.resolve_base_spec(BEAUTY_RESULT)
    assert base["source"] == "command_log"
    assert base["dataset"] == "beauty"
    assert base["setting_id"] == "B25_lr_h8_seen_anchor"
    assert base["source_feature_mode"] == "full_v3"
    assert base["runtime_feature_mode"] == "full_v4"
    assert base["fixed_values"]["embedding_size"] == 128
    assert base["fixed_values"]["d_feat_emb"] == 12
    assert base["fixed_values"]["MAX_ITEM_LIST_LENGTH"] == 20
    assert base["fixed_values"]["hidden_dropout_prob"] == 0.18
    assert base["overrides"]["stage_router_wrapper"] == {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"}


def test_build_lr_choices_uses_expected_presets():
    assert common.build_lr_choices(0.001, "tight3") == [0.00085, 0.001, 0.00115]
    assert common.build_lr_choices(0.001, "screen5") == [0.0005, 0.00075, 0.001, 0.00125, 0.0015]