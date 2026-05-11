#!/usr/bin/env python3
"""Routing-control ablations for FMoE_N4 on Beauty and KuaiRec."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402

AXIS = "ablation_dualset_routing_control_v1"
AXIS_ID = "N4ABLA"
AXIS_DESC = "routing_control_dualset"
PHASE_ID = "P4A"
PHASE_NAME = "N4_ROUTING_CONTROL"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS


def _all_stage(value: str) -> dict[str, str]:
    return {"macro": value, "mid": value, "micro": value}


def _primitive_topk_delta(base: dict[str, object], updates: dict[str, int]) -> dict[str, object]:
    source = common.clone_base_overrides(dict(base.get("overrides") or {})).get("stage_router_primitives") or {}
    mutated = common.clone_base_overrides(source)
    for stage in ("macro", "mid", "micro"):
        stage_raw = dict(mutated.get(stage) or {})
        for primitive, top_k in dict(updates).items():
            primitive_raw = dict(stage_raw.get(primitive) or {})
            primitive_raw["top_k"] = int(top_k)
            stage_raw[primitive] = primitive_raw
        mutated[stage] = stage_raw
    return {"stage_router_primitives": mutated}


def _perturb_delta(
    *,
    mode: str,
    apply: str,
    family: list[str] | None = None,
    keywords: list[str] | None = None,
    router_source: str | None = None,
) -> dict[str, object]:
    delta: dict[str, object] = {
        "feature_perturb_mode": str(mode),
        "feature_perturb_apply": str(apply),
        "feature_perturb_family": list(family or []),
        "feature_perturb_keywords": list(keywords or []),
    }
    if router_source is not None:
        delta["stage_router_source"] = _all_stage(str(router_source))
    return delta


def build_settings() -> list[dict[str, object]]:
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-01",
            "setting_key": "SHARED_FFN",
            "setting_desc": "shared_ffn",
            "setting_group": "routing_control",
            "setting_detail": "Dense FFN only without routing or feature injection.",
            "delta_overrides": {
                "layer_layout": ["macro", "mid", "micro"],
                "stage_compute_mode": _all_stage("dense_plain"),
                "stage_router_mode": _all_stage("none"),
                "stage_feature_injection": _all_stage("none"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-02",
            "setting_key": "ROUTER_SOURCE_HIDDEN",
            "setting_desc": "router_hidden_only",
            "setting_group": "routing_control",
            "setting_detail": "Use hidden-state-only routing at every stage.",
            "delta_overrides": {"stage_router_source": _all_stage("hidden")},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-03",
            "setting_key": "ROUTER_SOURCE_BOTH",
            "setting_desc": "router_hidden_plus_behavior",
            "setting_group": "routing_control",
            "setting_detail": "Use hidden plus behavior routing at every stage.",
            "force_identity": True,
            "delta_overrides": {"stage_router_source": _all_stage("both")},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-04",
            "setting_key": "ROUTER_SOURCE_FEATURE",
            "setting_desc": "router_behavior_only",
            "setting_group": "routing_control",
            "setting_detail": "Use behavior-feature-only routing at every stage.",
            "delta_overrides": {"stage_router_source": _all_stage("feature")},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-05",
            "setting_key": "NO_ROUTER_WITH_BIAS_INJECTION",
            "setting_desc": "no_router_gated_bias",
            "setting_group": "routing_control",
            "setting_detail": "Disable routing and switch to dense FFN while keeping gated feature bias injection.",
            "delta_overrides": {
                "stage_compute_mode": _all_stage("dense_plain"),
                "stage_router_mode": _all_stage("none"),
                "stage_feature_injection": _all_stage("gated_bias"),
            },
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RC-06",
            "setting_key": "NO_ROUTER_WITH_GROUP_GATED_BIAS",
            "setting_desc": "no_router_group_gated_bias",
            "setting_group": "routing_control",
            "setting_detail": "Disable routing and switch to dense FFN with stronger group-gated bias injection.",
            "delta_overrides": {
                "stage_compute_mode": _all_stage("dense_plain"),
                "stage_router_mode": _all_stage("none"),
                "stage_feature_injection": _all_stage("group_gated_bias"),
            },
        },
        {
            "scope": "appendix",
            "setting_id": "RC-07",
            "tier": "extended",
            "setting_key": "GLOBAL_TOPK_4_OF_12",
            "setting_desc": "global_topk_4_of_12",
            "setting_group": "routing_control",
            "setting_detail": "Use a global expert budget: activate 4 experts out of the full 12-way pool.",
            "delta_overrides": {"topk_scope_mode": "global_flat", "moe_top_k": 4},
        },
        {
            "scope": "appendix",
            "setting_id": "RC-08",
            "tier": "extended",
            "setting_key": "PER_GROUP_TOPK_1",
            "setting_desc": "per_group_topk_1",
            "setting_group": "routing_control",
            "setting_detail": "Keep one expert per active group through scalar intra-group routing.",
            "delta_builder": lambda base: _primitive_topk_delta(base, {"e_scalar": 1}),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-09",
            "setting_key": "GROUP_TOPK_2",
            "setting_desc": "group_topk_2",
            "setting_group": "routing_control",
            "setting_detail": "Activate only two groups out of four via group router sparsity.",
            "delta_builder": lambda base: _primitive_topk_delta(base, {"b_group": 2}),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-10",
            "setting_key": "GROUP_TOPK_2_INNER1",
            "setting_desc": "group_topk_2_inner1",
            "setting_group": "routing_control",
            "setting_detail": "Pick two groups, then one expert inside each active group.",
            "delta_builder": lambda base: _primitive_topk_delta(base, {"b_group": 2, "e_scalar": 1}),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-11",
            "setting_key": "ALL_SESSION",
            "setting_desc": "all_session_routing",
            "setting_group": "routing_control",
            "setting_detail": "Force session-level routing at every stage.",
            "delta_overrides": {"stage_router_granularity": _all_stage("session")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-12",
            "setting_key": "ALL_TOKEN",
            "setting_desc": "all_token_routing",
            "setting_group": "routing_control",
            "setting_detail": "Force token-level routing at every stage.",
            "delta_overrides": {"stage_router_granularity": _all_stage("token")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-13",
            "setting_key": "INJECTION_GATED_BIAS",
            "setting_desc": "routing_plus_gated_bias",
            "setting_group": "routing_control",
            "setting_detail": "Keep routing on and add gated feature bias injection everywhere.",
            "delta_overrides": {"stage_feature_injection": _all_stage("gated_bias")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-14",
            "setting_key": "INJECTION_GROUP_GATED_BIAS",
            "setting_desc": "routing_plus_group_gated_bias",
            "setting_group": "routing_control",
            "setting_detail": "Keep routing on and add group-gated feature bias injection everywhere.",
            "delta_overrides": {"stage_feature_injection": _all_stage("group_gated_bias")},
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-15",
            "setting_key": "FEATURE_ROUTER_EVAL_ZERO",
            "setting_desc": "feature_router_eval_zero",
            "setting_group": "routing_control",
            "setting_detail": "Keep behavior-only routing, but zero cue values at evaluation to test whether routing still depends on aligned cues.",
            "delta_overrides": _perturb_delta(mode="zero", apply="eval", router_source="feature"),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-16",
            "setting_key": "FEATURE_ROUTER_EVAL_SHUFFLE_ALL",
            "setting_desc": "feature_router_eval_shuffle_all",
            "setting_group": "routing_control",
            "setting_detail": "Keep behavior-only routing, but shuffle all cues at evaluation while preserving their marginal distribution.",
            "delta_overrides": _perturb_delta(mode="shuffle", apply="eval", router_source="feature"),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-17",
            "setting_key": "FEATURE_ROUTER_EVAL_SHUFFLE_MEMORY",
            "setting_desc": "feature_router_eval_shuffle_memory",
            "setting_group": "routing_control",
            "setting_detail": "Keep behavior-only routing, but shuffle the portable Memory family to probe which cue family the router actually depends on.",
            "delta_overrides": _perturb_delta(mode="family_permute", apply="eval", family=["Memory"], router_source="feature"),
        },
        {
            "scope": "appendix",
            "tier": "extended",
            "setting_id": "RC-18",
            "setting_key": "HIDDEN_ROUTER_EVAL_SHUFFLE_ALL",
            "setting_desc": "hidden_router_eval_shuffle_all",
            "setting_group": "routing_control",
            "setting_detail": "Apply the same evaluation-time cue shuffle to hidden-only routing as a control for cue-independent routing.",
            "delta_overrides": _perturb_delta(mode="shuffle", apply="eval", router_source="hidden"),
        },
    ]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 routing-control ablations",
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
            stage_name="routing_control",
            diag_logging=True,
            special_logging=True,
            feature_ablation_logging=False,
        ),
        args,
    )
    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="routing_control_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[routing-control] manifest -> {manifest}")
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
