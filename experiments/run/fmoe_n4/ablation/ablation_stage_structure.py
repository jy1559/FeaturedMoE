#!/usr/bin/env python3
"""Stage-structure ablations for FMoE_N4 on Beauty and KuaiRec."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402

AXIS = "ablation_dualset_stage_structure_v1"
AXIS_ID = "N4ABLB"
AXIS_DESC = "stage_structure_dualset"
PHASE_ID = "P4B"
PHASE_NAME = "N4_STAGE_STRUCTURE"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS


def _stage_map(*, macro: str, mid: str, micro: str) -> dict[str, str]:
    return {"macro": macro, "mid": mid, "micro": micro}


def build_settings() -> list[dict[str, object]]:
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-01",
            "setting_key": "REMOVE_MACRO",
            "setting_desc": "remove_macro",
            "setting_group": "stage_structure",
            "setting_detail": "Remove the macro stage while keeping mid and micro MoE blocks.",
            "delta_overrides": {
                "layer_layout": ["attn", "mid_ffn", "attn", "micro_ffn"],
                "stage_compute_mode": _stage_map(macro="none", mid="moe", micro="moe"),
                "stage_router_mode": _stage_map(macro="none", mid="learned", micro="learned"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-02",
            "setting_key": "REMOVE_MID",
            "setting_desc": "remove_mid",
            "setting_group": "stage_structure",
            "setting_detail": "Remove the mid stage while keeping macro and micro MoE blocks.",
            "delta_overrides": {
                "layer_layout": ["attn", "macro_ffn", "attn", "micro_ffn"],
                "stage_compute_mode": _stage_map(macro="moe", mid="none", micro="moe"),
                "stage_router_mode": _stage_map(macro="learned", mid="none", micro="learned"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-03",
            "setting_key": "REMOVE_MICRO",
            "setting_desc": "remove_micro",
            "setting_group": "stage_structure",
            "setting_detail": "Remove the micro stage while keeping macro and mid MoE blocks.",
            "delta_overrides": {
                "layer_layout": ["attn", "macro_ffn", "mid_ffn"],
                "stage_compute_mode": _stage_map(macro="moe", mid="moe", micro="none"),
                "stage_router_mode": _stage_map(macro="learned", mid="learned", micro="none"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-04",
            "setting_key": "SINGLE_STAGE_MACRO",
            "setting_desc": "single_stage_macro",
            "setting_group": "stage_structure",
            "setting_detail": "Keep only the macro routing stage.",
            "delta_overrides": {
                "layer_layout": ["attn", "macro_ffn"],
                "stage_compute_mode": _stage_map(macro="moe", mid="none", micro="none"),
                "stage_router_mode": _stage_map(macro="learned", mid="none", micro="none"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-05",
            "setting_key": "SINGLE_STAGE_MID",
            "setting_desc": "single_stage_mid",
            "setting_group": "stage_structure",
            "setting_detail": "Keep only the mid routing stage.",
            "delta_overrides": {
                "layer_layout": ["attn", "mid_ffn"],
                "stage_compute_mode": _stage_map(macro="none", mid="moe", micro="none"),
                "stage_router_mode": _stage_map(macro="none", mid="learned", micro="none"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-06",
            "setting_key": "SINGLE_STAGE_MICRO",
            "setting_desc": "single_stage_micro",
            "setting_group": "stage_structure",
            "setting_detail": "Keep only the micro routing stage.",
            "delta_overrides": {
                "layer_layout": ["attn", "micro_ffn"],
                "stage_compute_mode": _stage_map(macro="none", mid="none", micro="moe"),
                "stage_router_mode": _stage_map(macro="none", mid="none", micro="learned"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "ST-07",
            "setting_key": "DENSE_FULL_ONLY",
            "setting_desc": "dense_full_only",
            "setting_group": "stage_structure",
            "setting_detail": "Replace all routing blocks with dense FFNs.",
            "delta_overrides": {
                "layer_layout": ["macro", "mid", "micro"],
                "stage_compute_mode": _stage_map(macro="dense_plain", mid="dense_plain", micro="dense_plain"),
                "stage_router_mode": _stage_map(macro="none", mid="none", micro="none"),
                "stage_feature_injection": _stage_map(macro="none", mid="none", micro="none"),
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-08",
            "setting_key": "ATTN_BEFORE_MID",
            "setting_desc": "attn_before_mid",
            "setting_group": "stage_structure",
            "setting_detail": "Insert one extra attention block before the mid MoE stage.",
            "delta_overrides": {"layer_layout": ["attn", "macro_ffn", "attn", "mid_ffn", "attn", "micro_ffn"]},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-09",
            "setting_key": "EXTRA_ATTN_BEFORE_MICRO",
            "setting_desc": "extra_attn_before_micro",
            "setting_group": "stage_structure",
            "setting_detail": "Insert one extra attention block just before the micro stage.",
            "delta_overrides": {"layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "attn", "micro_ffn"]},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-10",
            "setting_key": "ORDER_MACRO_MICRO_MID",
            "setting_desc": "order_macro_micro_mid",
            "setting_group": "stage_structure",
            "setting_detail": "Swap the mid and micro stage order.",
            "delta_overrides": {"layer_layout": ["attn", "macro_ffn", "micro_ffn", "attn", "mid_ffn"]},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-11",
            "setting_key": "WRAPPER_ALL_W1_FLAT",
            "setting_desc": "wrapper_w1_flat",
            "setting_group": "stage_structure",
            "setting_detail": "Switch every stage wrapper to flat joint routing.",
            "delta_overrides": {"stage_router_wrapper": _stage_map(macro="w1_flat", mid="w1_flat", micro="w1_flat")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-12",
            "setting_key": "WRAPPER_ALL_W4_BXD",
            "setting_desc": "wrapper_w4_bxd",
            "setting_group": "stage_structure",
            "setting_detail": "Switch every stage wrapper to group-conditional routing.",
            "delta_overrides": {"stage_router_wrapper": _stage_map(macro="w4_bxd", mid="w4_bxd", micro="w4_bxd")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-13",
            "setting_key": "PREPEND_DENSE_CONTEXT",
            "setting_desc": "prepend_dense_context",
            "setting_group": "stage_structure",
            "setting_detail": "Prepend one dense contextualization layer before the staged routing stack.",
            "delta_overrides": {"layer_layout": ["layer", "attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-14",
            "setting_key": "ORDER_MID_MACRO_MICRO",
            "setting_desc": "order_mid_macro_micro",
            "setting_group": "stage_structure",
            "setting_detail": "Start from the mid stage before macro to test whether coarse-to-fine ordering matters.",
            "delta_overrides": {"layer_layout": ["attn", "mid_ffn", "macro_ffn", "attn", "micro_ffn"]},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "ST-15",
            "setting_key": "BUNDLE_MACROMID_THEN_MICRO",
            "setting_desc": "bundle_macro_mid_then_micro",
            "setting_group": "stage_structure",
            "setting_detail": "Replace the staged macro-then-mid stack with a bundled macro-mid composition before the micro stage.",
            "delta_overrides": {"layer_layout": ["bundle_macro_mid_router", "micro"]},
        },
    ]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 stage-structure ablations",
        default_datasets=common.DEFAULT_DATASETS,
        default_scope="core",
    )
    args = parser.parse_args()
    args = common.finalize_common_args(args)
    args.axis = AXIS
    return args


def main() -> int:
    args = parse_args()
    base_specs = common.resolve_base_specs_from_args(args)
    settings = common.filter_settings(build_settings(), args)
    rows = common.maybe_limit_smoke(
        common.build_study_rows(
            args=args,
            base_specs=base_specs,
            settings=settings,
            phase_id=PHASE_ID,
            axis_id=AXIS_ID,
            axis_desc=AXIS_DESC,
            stage_name="stage_structure",
            diag_logging=True,
            special_logging=True,
            feature_ablation_logging=False,
        ),
        args,
    )
    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="stage_structure_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[stage-structure] manifest -> {manifest}")
    return common.launch_rows(
        rows=rows,
        args=args,
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        log_root=LOG_ROOT,
        fieldnames=common.build_fieldnames(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
