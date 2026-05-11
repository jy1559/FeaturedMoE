#!/usr/bin/env python3
"""Cue-family ablations for FMoE_N4 on Beauty and KuaiRec."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402

AXIS = "ablation_dualset_cue_family_v1"
AXIS_ID = "N4ABLC"
AXIS_DESC = "cue_family_dualset"
PHASE_ID = "P4C"
PHASE_NAME = "N4_CUE_ABLATION"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS


def build_settings() -> list[dict[str, object]]:
    category_drop = common._all_stage_mask(["Tempo", "Memory", "Exposure"])
    time_drop = common._all_stage_mask(["Focus", "Memory", "Exposure"])
    sequence_only = common._all_stage_mask(["Memory", "Exposure"])
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "CF-01",
            "setting_key": "DROP_CATEGORY_DERIVED",
            "setting_desc": "drop_category",
            "setting_group": "cue_family",
            "setting_detail": "Drop category-derived routing cues while keeping time and sequence cues.",
            "delta_overrides": {
                "stage_feature_family_mask": category_drop,
                "stage_feature_drop_keywords": list(common.CATEGORY_DROP_KEYWORDS),
                "feature_perturb_mode": "none",
                "feature_perturb_apply": "none",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "CF-02",
            "setting_key": "DROP_TIMESTAMP_DERIVED",
            "setting_desc": "drop_time",
            "setting_group": "cue_family",
            "setting_detail": "Drop time-derived routing cues while keeping category and sequence cues.",
            "delta_overrides": {
                "stage_feature_family_mask": time_drop,
                "stage_feature_drop_keywords": list(common.TIMESTAMP_DROP_KEYWORDS),
                "feature_perturb_mode": "none",
                "feature_perturb_apply": "none",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "CF-03",
            "setting_key": "SEQUENCE_ONLY_PORTABLE",
            "setting_desc": "sequence_only",
            "setting_group": "cue_family",
            "setting_detail": "Keep only sequence-derived Memory and Exposure cue families.",
            "delta_overrides": {
                "stage_feature_family_mask": sequence_only,
                "stage_feature_drop_keywords": list(common.CATEGORY_DROP_KEYWORDS) + list(common.TIMESTAMP_DROP_KEYWORDS),
                "feature_perturb_mode": "none",
                "feature_perturb_apply": "none",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-04",
            "setting_key": "ONLY_MEMORY",
            "setting_desc": "only_memory",
            "setting_group": "cue_family",
            "setting_detail": "Use only the Memory cue family.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Memory"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-05",
            "setting_key": "ONLY_EXPOSURE",
            "setting_desc": "only_exposure",
            "setting_group": "cue_family",
            "setting_detail": "Use only the Exposure cue family.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Exposure"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "essential",
            "setting_id": "CF-06",
            "setting_key": "MEMORY_EXPOSURE",
            "setting_desc": "memory_exposure",
            "setting_group": "cue_family",
            "setting_detail": "Use the portable Memory plus Exposure pair.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Memory", "Exposure"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-07",
            "setting_key": "FOCUS_MEMORY",
            "setting_desc": "focus_memory",
            "setting_group": "cue_family",
            "setting_detail": "Keep a category-sensitive Focus plus Memory pair.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Focus", "Memory"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-08",
            "setting_key": "NO_TEMPO",
            "setting_desc": "no_tempo",
            "setting_group": "cue_family",
            "setting_detail": "Remove the Tempo family but keep all others.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Focus", "Memory", "Exposure"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "essential",
            "setting_id": "CF-09",
            "setting_key": "NO_EXPOSURE",
            "setting_desc": "no_exposure",
            "setting_group": "cue_family",
            "setting_detail": "Remove the Exposure family but keep Tempo, Focus, and Memory.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Tempo", "Focus", "Memory"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-10",
            "setting_key": "TEMPO_FOCUS",
            "setting_desc": "tempo_focus",
            "setting_group": "cue_family",
            "setting_detail": "Use the non-sequence Tempo plus Focus pair.",
            "delta_overrides": {"stage_feature_family_mask": common._all_stage_mask(["Tempo", "Focus"]), "stage_feature_drop_keywords": []},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-11",
            "setting_key": "EVAL_ALL_ZERO",
            "setting_desc": "eval_all_zero",
            "setting_group": "cue_family",
            "setting_detail": "Zero all cue values only at evaluation time to test whether trained routing still uses the cue branch.",
            "delta_overrides": {
                "feature_perturb_mode": "zero",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-12",
            "setting_key": "EVAL_ALL_SHUFFLE",
            "setting_desc": "eval_all_shuffle",
            "setting_group": "cue_family",
            "setting_detail": "Shuffle all cues only at evaluation time while preserving marginal statistics.",
            "delta_overrides": {
                "feature_perturb_mode": "shuffle",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-13",
            "setting_key": "EVAL_SHUFFLE_MEMORY",
            "setting_desc": "eval_shuffle_memory",
            "setting_group": "cue_family",
            "setting_detail": "Shuffle only the Memory family at evaluation to localize family-specific routing dependence.",
            "delta_overrides": {
                "feature_perturb_mode": "family_permute",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["Memory"],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-14",
            "setting_key": "EVAL_SHUFFLE_EXPOSURE",
            "setting_desc": "eval_shuffle_exposure",
            "setting_group": "cue_family",
            "setting_detail": "Shuffle only the Exposure family at evaluation to test whether the most portable cues remain causally useful.",
            "delta_overrides": {
                "feature_perturb_mode": "family_permute",
                "feature_perturb_apply": "eval",
                "feature_perturb_family": ["Exposure"],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-15",
            "setting_key": "TRAIN_BATCH_PERMUTE_ALL",
            "setting_desc": "train_batch_permute_all",
            "setting_group": "cue_family",
            "setting_detail": "Corrupt cue-sequence alignment during training within local batches while preserving cue distributions.",
            "delta_overrides": {
                "feature_perturb_mode": "batch_permute",
                "feature_perturb_apply": "train",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-16",
            "setting_key": "TRAIN_GLOBAL_PERMUTE_ALL",
            "setting_desc": "train_global_permute_all",
            "setting_group": "cue_family",
            "setting_detail": "Corrupt global cue alignment during training to separate aligned routing gains from parameter-count effects.",
            "delta_overrides": {
                "feature_perturb_mode": "global_permute",
                "feature_perturb_apply": "train",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-17",
            "setting_key": "ROLE_SWAP",
            "setting_desc": "feature_role_swap",
            "setting_group": "cue_family",
            "setting_detail": "Swap cue-family semantic roles while keeping the branch present, testing whether meaning alignment matters beyond feature presence.",
            "delta_overrides": {
                "feature_perturb_mode": "role_swap",
                "feature_perturb_apply": "both",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
            },
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "CF-18",
            "setting_key": "POSITION_SHIFT",
            "setting_desc": "position_shift_feature",
            "setting_group": "cue_family",
            "setting_detail": "Shift cue positions along the sequence to break temporal alignment without removing the cue branch.",
            "delta_overrides": {
                "feature_perturb_mode": "position_shift",
                "feature_perturb_apply": "both",
                "feature_perturb_family": [],
                "feature_perturb_keywords": [],
                "feature_perturb_shift": 1,
            },
        },
    ]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 cue-family ablations",
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
            stage_name="cue_family",
            diag_logging=True,
            special_logging=True,
            feature_ablation_logging=True,
        ),
        args,
    )
    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="cue_family_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[cue-family] manifest -> {manifest}")
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
