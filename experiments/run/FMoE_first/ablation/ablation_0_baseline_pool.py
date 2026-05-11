#!/usr/bin/env python3
"""Baseline replay / retuning pool for FMoE_N4 ablations."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402

AXIS = "ablation_dualset_baseline_pool_v1"
AXIS_ID = "N4ABL0"
AXIS_DESC = "baseline_pool_dualset"
PHASE_ID = "P40"
PHASE_NAME = "N4_BASELINE_POOL"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS


def build_settings() -> list[dict[str, object]]:
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "BL-00",
            "setting_key": "BASE_REPLAY",
            "setting_desc": "base_replay",
            "setting_group": "baseline_pool",
            "setting_detail": "Replay and retune the selected base configuration under the ablation runtime.",
            "force_identity": True,
            "delta_overrides": {},
        }
    ]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 baseline replay pool",
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
            stage_name="baseline_pool",
            diag_logging=True,
            special_logging=True,
            feature_ablation_logging=False,
        ),
        args,
    )
    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="baseline_pool_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[baseline-pool] manifest -> {manifest}")
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