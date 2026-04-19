#!/usr/bin/env python3
"""Objective and regularization ablations for FMoE_N4."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402

AXIS = "ablation_dualset_objective_variants_v1"
AXIS_ID = "N4ABLD"
AXIS_DESC = "objective_variants_dualset"
PHASE_ID = "P4D"
PHASE_NAME = "N4_OBJECTIVE_VARIANTS"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS


def build_settings() -> list[dict[str, object]]:
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-01",
            "setting_key": "NO_CONSISTENCY",
            "setting_desc": "no_consistency",
            "setting_group": "objective",
            "setting_detail": "Disable route consistency regularization.",
            "delta_overrides": {"route_consistency_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-02",
            "setting_key": "NO_ZLOSS",
            "setting_desc": "no_zloss",
            "setting_group": "objective",
            "setting_detail": "Disable router z-loss regularization.",
            "delta_overrides": {"z_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-03",
            "setting_key": "NO_BALANCE",
            "setting_desc": "no_balance",
            "setting_group": "objective",
            "setting_detail": "Disable balance regularization.",
            "delta_overrides": {"balance_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-04",
            "setting_key": "ALL_AUX_OFF",
            "setting_desc": "all_aux_off",
            "setting_group": "objective",
            "setting_detail": "Disable consistency, z-loss, and balance regularization together.",
            "delta_overrides": {"route_consistency_lambda": 0.0, "z_loss_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-05",
            "setting_key": "CONSISTENCY_ONLY",
            "setting_desc": "consistency_only",
            "setting_group": "objective",
            "setting_detail": "Keep only consistency regularization.",
            "delta_overrides": {"z_loss_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-06",
            "setting_key": "ZLOSS_ONLY",
            "setting_desc": "zloss_only",
            "setting_group": "objective",
            "setting_detail": "Keep only z-loss regularization.",
            "delta_overrides": {"route_consistency_lambda": 0.0, "balance_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "extended",
            "setting_id": "OBJ-07",
            "setting_key": "BALANCE_ONLY",
            "setting_desc": "balance_only",
            "setting_group": "objective",
            "setting_detail": "Keep only balance regularization.",
            "delta_overrides": {"route_consistency_lambda": 0.0, "z_loss_lambda": 0.0},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "OBJ-08",
            "setting_key": "CONSISTENCY_PLUS_ZLOSS",
            "setting_desc": "consistency_plus_zloss",
            "setting_group": "objective",
            "setting_detail": "Keep consistency and z-loss but disable balance.",
            "delta_overrides": {"balance_loss_lambda": 0.0},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-09",
            "setting_key": "CONSISTENCY_X2",
            "setting_desc": "consistency_x2",
            "setting_group": "objective",
            "setting_detail": "Double the consistency regularization weight.",
            "delta_builder": lambda base: {
                "route_consistency_lambda": 2.0 * float(dict(base.get("overrides") or {}).get("route_consistency_lambda", 0.0))
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-10",
            "setting_key": "ZLOSS_X2",
            "setting_desc": "zloss_x2",
            "setting_group": "objective",
            "setting_detail": "Double the z-loss regularization weight.",
            "delta_builder": lambda base: {
                "z_loss_lambda": 2.0 * float(dict(base.get("overrides") or {}).get("z_loss_lambda", 0.0))
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-11",
            "setting_key": "CONSISTENCY_X0P5",
            "setting_desc": "consistency_x0p5",
            "setting_group": "objective",
            "setting_detail": "Halve the consistency regularization weight.",
            "delta_builder": lambda base: {
                "route_consistency_lambda": 0.5 * float(dict(base.get("overrides") or {}).get("route_consistency_lambda", 0.0))
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-12",
            "setting_key": "ZLOSS_X0P5",
            "setting_desc": "zloss_x0p5",
            "setting_group": "objective",
            "setting_detail": "Halve the z-loss regularization weight.",
            "delta_builder": lambda base: {
                "z_loss_lambda": 0.5 * float(dict(base.get("overrides") or {}).get("z_loss_lambda", 0.0))
            },
        },
        {
            "scope": "appendix",
            "tier": "optional",
            "setting_id": "OBJ-13",
            "setting_key": "CONSISTENCY_PLUS_BALANCE",
            "setting_desc": "consistency_plus_balance",
            "setting_group": "objective",
            "setting_detail": "Keep consistency and balance, but disable z-loss.",
            "delta_overrides": {"z_loss_lambda": 0.0},
        },
        {
            "scope": "appendix",
            "tier": "optional",
            "setting_id": "OBJ-14",
            "setting_key": "ZLOSS_PLUS_BALANCE",
            "setting_desc": "zloss_plus_balance",
            "setting_group": "objective",
            "setting_detail": "Keep z-loss and balance, but disable consistency.",
            "delta_overrides": {"route_consistency_lambda": 0.0},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-15",
            "setting_key": "SMALL_ROUTER_HIDDEN",
            "setting_desc": "small_router_hidden",
            "setting_group": "objective",
            "setting_detail": "Light capacity sensitivity: reduce router hidden width by 25%.",
            "delta_builder": lambda base: {
                "d_router_hidden": max(16, int(round(float(dict(base.get("fixed_values") or {}).get("d_router_hidden", 64)) * 0.75)))
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "OBJ-16",
            "setting_key": "LARGE_ROUTER_HIDDEN",
            "setting_desc": "large_router_hidden",
            "setting_group": "objective",
            "setting_detail": "Light capacity sensitivity: increase router hidden width by 25%.",
            "delta_builder": lambda base: {
                "d_router_hidden": int(round(float(dict(base.get("fixed_values") or {}).get("d_router_hidden", 64)) * 1.25))
            },
        },
    ]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 objective-variant ablations",
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
            stage_name="objective_variants",
            diag_logging=True,
            special_logging=True,
            feature_ablation_logging=False,
        ),
        args,
    )
    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="objective_variants_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[objective-variants] manifest -> {manifest}")
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
