#!/usr/bin/env python3
"""Global queue launcher for ablations 0-4 on dual datasets."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import ablation_0_baseline_pool as baseline_pool  # noqa: E402
import ablation_cue_family as cue_family  # noqa: E402
import ablation_objective_variants as objective_variants  # noqa: E402
import ablation_routing_control as routing_control  # noqa: E402
import ablation_stage_structure as stage_structure  # noqa: E402
import common  # noqa: E402
from run_phase_wide_common import (  # noqa: E402
    _count_ok_trials,
    _is_completed_log,
    _load_json_payload,
    _verify_special_diag_from_result,
    launch_wide_rows,
)

AXIS = "ablation_dualset_core_global_queue_v1"
PHASE_ID = "P4G"
PHASE_NAME = "N4_CORE_GLOBAL_QUEUE"
AXIS_ID = "N4ABLG"
AXIS_DESC = "core_global_queue"
RESUME_AXIS_PREFIX = AXIS

STUDIES = [
    {
        "name": "baseline_pool",
        "order": 0,
        "module": baseline_pool,
        "phase_id": baseline_pool.PHASE_ID,
        "axis_id": baseline_pool.AXIS_ID,
        "axis_desc": baseline_pool.AXIS_DESC,
        "diag_logging": True,
        "special_logging": True,
        "feature_ablation_logging": False,
    },
    {
        "name": "routing_control",
        "order": 1,
        "module": routing_control,
        "phase_id": routing_control.PHASE_ID,
        "axis_id": routing_control.AXIS_ID,
        "axis_desc": routing_control.AXIS_DESC,
        "diag_logging": True,
        "special_logging": True,
        "feature_ablation_logging": False,
    },
    {
        "name": "stage_structure",
        "order": 2,
        "module": stage_structure,
        "phase_id": stage_structure.PHASE_ID,
        "axis_id": stage_structure.AXIS_ID,
        "axis_desc": stage_structure.AXIS_DESC,
        "diag_logging": True,
        "special_logging": True,
        "feature_ablation_logging": False,
    },
    {
        "name": "cue_family",
        "order": 3,
        "module": cue_family,
        "phase_id": cue_family.PHASE_ID,
        "axis_id": cue_family.AXIS_ID,
        "axis_desc": cue_family.AXIS_DESC,
        "diag_logging": True,
        "special_logging": True,
        "feature_ablation_logging": True,
    },
    {
        "name": "objective_variants",
        "order": 4,
        "module": objective_variants,
        "phase_id": objective_variants.PHASE_ID,
        "axis_id": objective_variants.AXIS_ID,
        "axis_desc": objective_variants.AXIS_DESC,
        "diag_logging": True,
        "special_logging": True,
        "feature_ablation_logging": False,
    },
]


def parse_args():
    parser = common.common_arg_parser(
        "FMoE_N4 global queue for ablations 0-4",
        default_datasets=common.DEFAULT_DATASETS,
        default_scope="all",
    )
    parser.add_argument("--axis-suffix", default="")
    parser.add_argument("--include-baseline-always", action="store_true")
    parser.add_argument(
        "--only-stage",
        default="",
        help="Comma-separated study filter: baseline|0, routing|1, stage|2, cue|3, objective|4",
    )
    args = parser.parse_args()
    args = common.finalize_common_args(args)
    return args


def _resolve_axis(args: Any) -> str:
    axis_suffix = str(getattr(args, "axis_suffix", "") or "").strip()
    if not axis_suffix:
        return AXIS
    return f"{AXIS}_{common.sanitize_token(axis_suffix, upper=False)}"


def _order_rows(
    rows: list[dict[str, Any]],
    *,
    study_order: dict[str, int],
    setting_order: dict[tuple[str, str], int],
    base_order: dict[str, int],
) -> list[dict[str, Any]]:
    dataset_order = {
        "beauty": 0,
        "KuaiRecLargeStrictPosV2_0.2": 1,
    }

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
        stage_name = str(row.get("stage", ""))
        setting_id = str(row.get("setting_id", ""))
        base_key = str(row.get("base_key", ""))
        dataset = str(row.get("dataset", ""))
        base_rank = int(row.get("base_rank", 0) or 0)
        seed_id = int(row.get("seed_id", 0) or 0)
        return (
            base_rank if base_rank > 0 else int(base_order.get(base_key, 999)) + 1,
            int(study_order.get(stage_name, 999)),
            int(setting_order.get((stage_name, setting_id), 999)),
            int(dataset_order.get(dataset, 999)),
            int(base_order.get(base_key, 999)),
            seed_id,
        )

    return sorted(rows, key=sort_key)


def _selected_stages(args: Any) -> set[str]:
    aliases = {
        "0": "baseline_pool",
        "baseline": "baseline_pool",
        "baseline_pool": "baseline_pool",
        "1": "routing_control",
        "routing": "routing_control",
        "routing_control": "routing_control",
        "2": "stage_structure",
        "stage": "stage_structure",
        "stage_structure": "stage_structure",
        "3": "cue_family",
        "cue": "cue_family",
        "cue_family": "cue_family",
        "4": "objective_variants",
        "objective": "objective_variants",
        "objective_variants": "objective_variants",
    }
    selected: set[str] = set()
    for token in common._parse_csv_strings(str(getattr(args, "only_stage", "") or "")):
        normalized = aliases.get(str(token).strip().lower())
        if normalized:
            selected.add(normalized)
    return selected


def _build_log_path(log_dir: Path, row: dict[str, Any], phase_id: str) -> Path:
    _ = phase_id
    dataset_dir = log_dir / str(row["dataset"])
    family_dir = dataset_dir / common.sanitize_token(str(row["family_id"]), upper=False)
    filename = (
        f"{str(row['phase_id'])}_"
        f"{common.sanitize_token(str(row['base_setting_id']), upper=True)}_"
        f"{common.sanitize_token(str(row['setting_id']), upper=True)}_"
        f"S{int(row['seed_id'])}.log"
    )
    return family_dir / filename


def _semantic_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(row.get("dataset", "") or ""),
        str(row.get("stage", "") or ""),
        common.sanitize_token(str(row.get("base_setting_id", "") or ""), upper=False),
        common.sanitize_token(str(row.get("setting_id", "") or ""), upper=False),
        int(row.get("seed_id", 0) or 0),
    )


def _resume_axis_names(current_axis: str) -> set[str]:
    names: set[str] = set()
    for path in sorted(common.ABLATION_LOGS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        name = str(path.name)
        if name == str(current_axis) or name.startswith(f"{RESUME_AXIS_PREFIX}_") or name == RESUME_AXIS_PREFIX:
            names.add(name)
    return names


def _phase_to_stage(phase_id: str) -> str:
    return {
        "P40": "baseline_pool",
        "P4A": "routing_control",
        "P4B": "stage_structure",
        "P4C": "cue_family",
        "P4D": "objective_variants",
    }.get(str(phase_id).upper(), "")


def _parse_core_queue_run_phase(run_phase: str) -> dict[str, Any] | None:
    tokens = [token for token in str(run_phase).strip().split("_") if token]
    if len(tokens) < 6:
        return None
    phase_id = str(tokens[0]).strip()
    stage = _phase_to_stage(phase_id)
    if not stage:
        return None
    base_idx = -1
    for idx in range(1, len(tokens)):
        token = str(tokens[idx]).strip().upper()
        if len(token) == 3 and token.startswith("B") and token[1:].isdigit():
            base_idx = idx
            break
    if base_idx < 0 or base_idx + 4 >= len(tokens):
        return None
    dataset = "_".join(tokens[1:base_idx])
    seed_token = str(tokens[-1]).strip().upper()
    if not (len(seed_token) >= 2 and seed_token.startswith("S") and seed_token[1:].isdigit()):
        return None
    setting_id = f"{tokens[-3]}-{tokens[-2]}"
    base_setting_id = "_".join(tokens[base_idx + 1 : -3])
    if not dataset or not base_setting_id:
        return None
    return {
        "phase_id": phase_id,
        "stage": stage,
        "dataset": dataset,
        "base_rank": int(tokens[base_idx][1:]),
        "base_setting_id": base_setting_id,
        "setting_id": setting_id,
        "seed_id": int(seed_token[1:]),
    }


def _find_completed_log_for_result(axis_dir: Path, parsed: dict[str, Any]) -> Path | None:
    filename = (
        f"{str(parsed['phase_id'])}_"
        f"{common.sanitize_token(str(parsed['base_setting_id']), upper=True)}_"
        f"{common.sanitize_token(str(parsed['setting_id']), upper=True)}_"
        f"S{int(parsed['seed_id'])}.log"
    )
    dataset_dir = axis_dir / str(parsed["dataset"])
    if not dataset_dir.exists():
        return None
    for log_path in sorted(dataset_dir.rglob(filename)):
        if _is_completed_log(log_path):
            return log_path
    return None


def _cross_axis_completed_keys(current_axis: str, verify_logging: bool) -> dict[tuple[str, str, str, str, int], dict[str, str]]:
    completed: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    axis_names = _resume_axis_names(current_axis)
    for result_path in sorted(common.RESULT_ROOT.glob("*.json")):
        if result_path.name == "meta.json":
            continue
        payload = _load_json_payload(result_path)
        if not isinstance(payload, dict):
            continue
        run_axis = str(payload.get("run_axis", "") or "").strip()
        if run_axis not in axis_names:
            continue
        run_phase = str(payload.get("run_phase", "") or "").strip()
        parsed = _parse_core_queue_run_phase(run_phase)
        if parsed is None:
            continue
        parsed["dataset"] = str(payload.get("dataset", "") or parsed.get("dataset", ""))
        if bool(payload.get("interrupted", False)):
            continue
        if int(_count_ok_trials(payload) or 0) <= 0:
            continue
        axis_dir = common.ABLATION_LOGS_ROOT / run_axis
        log_path = _find_completed_log_for_result(axis_dir, parsed)
        if log_path is None:
            continue
        if verify_logging:
            special_ok, diag_ok, _detail = _verify_special_diag_from_result(str(result_path))
            if not special_ok or not diag_ok:
                continue
        key = (
            str(parsed["dataset"]),
            str(parsed["stage"]),
            common.sanitize_token(str(parsed["base_setting_id"]), upper=False),
            common.sanitize_token(str(parsed["setting_id"]), upper=False),
            int(parsed["seed_id"]),
        )
        completed[key] = {
            "axis": run_axis,
            "run_phase": run_phase,
            "result_path": str(result_path),
        }
    return completed


def main() -> int:
    args = parse_args()
    axis = _resolve_axis(args)
    log_root = common.ABLATION_LOGS_ROOT / axis
    args.axis = axis
    base_specs = common.resolve_base_specs_from_args(args)
    selected_stages = _selected_stages(args)
    study_order: dict[str, int] = {}
    setting_order: dict[tuple[str, str], int] = {}
    base_order = {str(base["base_key"]): idx for idx, base in enumerate(base_specs)}

    rows: list[dict[str, Any]] = []
    for study in STUDIES:
        stage_name = str(study["name"])
        if selected_stages and stage_name not in selected_stages:
            continue
        study_order[stage_name] = int(study["order"])
        module = study["module"]
        raw_settings = module.build_settings()
        settings = common.filter_settings(raw_settings, args)
        if stage_name == "baseline_pool" and bool(getattr(args, "include_baseline_always", False)) and not settings:
            fallback_args = copy.copy(args)
            fallback_args.setting_tier = "all"
            settings = common.filter_settings(raw_settings, fallback_args)
        for idx, setting in enumerate(settings):
            setting_order[(stage_name, str(setting["setting_id"]))] = idx
        study_rows = common.build_study_rows(
            args=args,
            base_specs=base_specs,
            settings=settings,
            phase_id=str(study["phase_id"]),
            axis_id=str(study["axis_id"]),
            axis_desc=str(study["axis_desc"]),
            stage_name=stage_name,
            diag_logging=bool(study["diag_logging"]),
            special_logging=bool(study["special_logging"]),
            feature_ablation_logging=bool(study["feature_ablation_logging"]),
        )
        rows.extend(study_rows)

    rows = _order_rows(
        common.maybe_limit_smoke(rows, args),
        study_order=study_order,
        setting_order=setting_order,
        base_order=base_order,
    )

    completed_keys = _cross_axis_completed_keys(axis, bool(args.verify_logging))
    cross_axis_skipped = [row for row in rows if _semantic_row_key(row) in completed_keys]
    if cross_axis_skipped:
        print(
            f"[core-global-queue] cross-axis resume: skipped {len(cross_axis_skipped)} runs "
            f"already completed in sibling ablation axes."
        )
    rows = [row for row in rows if _semantic_row_key(row) not in completed_keys]

    manifest = common.write_manifest(
        args=args,
        log_root=log_root,
        default_name="core_global_queue_manifest.json",
        axis=axis,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[core-global-queue] manifest -> {manifest}")

    fieldnames = common.build_fieldnames(["queue_stage_order"])
    gpus = common._parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPU or cpu token selected")
    return int(
        launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=axis,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=log_root,
            summary_path=log_root / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[
                col
                for col in fieldnames
                if col
                not in {
                    "global_best_valid_mrr20",
                    "run_best_valid_mrr20",
                    "run_phase",
                    "exp_brief",
                    "stage",
                    "trigger",
                    "dataset",
                    "seed_id",
                    "gpu_id",
                    "status",
                    "test_mrr20",
                    "n_completed",
                    "interrupted",
                    "special_ok",
                    "diag_ok",
                    "result_path",
                    "timestamp_utc",
                }
            ],
            build_command=common.build_command,
            build_log_path=_build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: common.summary_path(log_root, str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
