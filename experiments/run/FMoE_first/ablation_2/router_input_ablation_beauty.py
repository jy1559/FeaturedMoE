#!/usr/bin/env python3
"""Router-input ablation runner for FMoE_N4 (Ablation-2).

Compares three router-input settings:
  - RI-00: baseline
  - RI-01: hidden only
  - RI-02: hidden + feature

Base settings are selected from artifacts/results/fmoe_n4 JSONs by test performance.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

THIS_DIR = Path(__file__).resolve().parent
ABLA_DIR = THIS_DIR.parent / "ablation"
if str(ABLA_DIR) not in sys.path:
    sys.path.insert(0, str(ABLA_DIR))

import common  # noqa: E402

AXIS = "ablation_2_router_input_v3"
AXIS_ID = "N4AB2"
AXIS_DESC = "router_input_3way"
PHASE_ID = "P4R"
PHASE_NAME = "N4_ROUTER_INPUT_ABLATION2"
LOG_ROOT = common.ABLATION_LOGS_ROOT / AXIS

RESULT_ROOT = common.RESULT_ROOT


def _metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except Exception:
            return None
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    return None


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_test_mrr(payload: Dict[str, Any]) -> float | None:
    for key in (
        "test_mrr@20",
        "test_score",
    ):
        val = _metric_to_float(payload.get(key))
        if val is not None:
            return val
    test_result = payload.get("test_result")
    if isinstance(test_result, dict):
        for key in ("mrr@20", "MRR@20"):
            val = _metric_to_float(test_result.get(key))
            if val is not None:
                return val
    return None


def _extract_best_lr(payload: Dict[str, Any]) -> float | None:
    best_params = payload.get("best_params")
    if isinstance(best_params, dict):
        val = _metric_to_float(best_params.get("learning_rate"))
        if val is not None and val > 0:
            return val
    val = _metric_to_float(payload.get("best_learning_rate"))
    if val is not None and val > 0:
        return val
    return None


def _all_stage(value: str) -> dict[str, str]:
    return {"macro": value, "mid": value, "micro": value}


def _router_source_delta(base: dict[str, Any], source: str) -> dict[str, Any]:
    src = str(source).lower().strip()
    if src not in {"hidden", "feature", "both"}:
        raise RuntimeError(f"unsupported router source: {source}")
    base_overrides = common.clone_base_overrides(dict(base.get("overrides") or {}))
    raw_prims = common.clone_base_overrides(dict(base_overrides.get("stage_router_primitives") or {}))
    stage_names = ("macro", "mid", "micro")
    primitive_names = ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar")
    for stage in stage_names:
        stage_cfg = dict(raw_prims.get(stage) or {})
        for pname in primitive_names:
            node = dict(stage_cfg.get(pname) or {})
            node["source"] = src
            stage_cfg[pname] = node
        raw_prims[stage] = stage_cfg
    return {
        "stage_router_source": _all_stage(src),
        "stage_router_primitives": raw_prims,
    }


def _logspace_lr_grid(base_lr: float, *, points: int = 5, log_span: float = 0.35) -> list[float]:
    base = float(base_lr)
    if base <= 0:
        raise RuntimeError(f"invalid base lr: {base_lr}")
    points = max(int(points), 2)
    span = max(float(log_span), 1e-6)
    left = math.log(base) - span
    right = math.log(base) + span
    out: list[float] = []
    for idx in range(points):
        alpha = idx / float(points - 1)
        val = math.exp(left * (1.0 - alpha) + right * alpha)
        out.append(round(float(val), 12))
    deduped: list[float] = []
    seen: set[float] = set()
    for val in out:
        if val <= 0:
            continue
        if val in seen:
            continue
        seen.add(val)
        deduped.append(val)
    return deduped


def build_settings() -> list[dict[str, Any]]:
    return [
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RI-00",
            "setting_key": "BASELINE",
            "setting_desc": "baseline_replay",
            "setting_group": "router_input",
            "setting_detail": "Replay base setting as-is.",
            "force_identity": True,
            "delta_overrides": {},
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RI-01",
            "setting_key": "ROUTER_INPUT_HIDDEN_ONLY",
            "setting_desc": "router_hidden_only",
            "setting_group": "router_input",
            "setting_detail": "Force all stage router inputs to hidden only.",
            "delta_builder": lambda base: _router_source_delta(base, "hidden"),
        },
        {
            "scope": "core",
            "tier": "essential",
            "setting_id": "RI-02",
            "setting_key": "ROUTER_INPUT_HIDDEN_PLUS_FEATURE",
            "setting_desc": "router_hidden_plus_feature",
            "setting_group": "router_input",
            "setting_detail": "Force all stage router inputs to hidden+feature.",
            "force_identity": True,
            "delta_builder": lambda base: _router_source_delta(base, "both"),
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = common.common_arg_parser(
        "FMoE_N4 router-input ablation (dataset/seed/hparam configurable)",
        default_datasets=["beauty"],
        default_scope="core",
    )
    parser.add_argument("--n-hparam", type=int, default=2, help="number of base hparam settings per dataset")
    parser.add_argument("--n-seeds", type=int, default=2, help="number of seeds per (base, setting)")
    parser.add_argument("--lr-log-span", type=float, default=0.35, help="log-space half-width around base lr")
    parser.add_argument("--lr-grid-points", type=int, default=5)
    parser.add_argument("--diag-pair-max-points", type=int, default=4096)
    parser.add_argument("--diag-pair-bin-count", type=int, default=20)
    args = parser.parse_args()
    args = common.finalize_common_args(args)
    args.axis = AXIS
    # User-requested defaults for this ablation.
    args.topk_per_dataset = max(int(getattr(args, "n_hparam", 2) or 2), 1)
    if not str(getattr(args, "seeds", "")).strip():
        n_seeds = max(int(getattr(args, "n_seeds", 2) or 2), 1)
        args.seeds = ",".join(str(i) for i in range(1, n_seeds + 1))
    args.max_evals = int(getattr(args, "max_evals", 0) or 5)
    args.max_evals = min(max(args.max_evals, 1), 5)
    args.tune_epochs = min(int(getattr(args, "tune_epochs", 0) or 60), 60)
    args.tune_patience = min(int(getattr(args, "tune_patience", 0) or 8), 8)
    return args


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_specs: list[dict[str, Any]] = common.resolve_base_specs_from_args(args)

    settings = common.filter_settings(build_settings(), args)
    rows: list[dict[str, Any]] = []
    seeds = common._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")

    cursor = 0
    for base in base_specs:
        dataset = str(base["dataset"])
        common.validate_session_fixed_files(dataset, args.feature_dataset_dir)
        lr_choices = _logspace_lr_grid(
            float(base["best_learning_rate"]),
            points=int(args.lr_grid_points),
            log_span=float(args.lr_log_span),
        )

        for setting in settings:
            base_overrides = common.clone_base_overrides(dict(base.get("overrides") or {}))
            fixed_values = copy.deepcopy(dict(base.get("fixed_values") or {}))
            delta_builder = setting.get("delta_builder")
            if callable(delta_builder):
                delta = dict(delta_builder(base) or {})
            else:
                delta = dict(setting.get("delta_overrides") or {})
            overrides = common.apply_delta_overrides(base_overrides, delta)
            for key in list(overrides.keys()):
                fixed_values.pop(str(key), None)

            if not bool(setting.get("force_identity", False)) and json.dumps(overrides, sort_keys=True) == json.dumps(base_overrides, sort_keys=True):
                continue

            search_space = {
                "learning_rate": {"type": "choice", "values": lr_choices},
            }

            for seed_id in seeds:
                cursor += 1
                run_suffix = (
                    f"{common.sanitize_token(dataset, upper=True)}_"
                    f"B{int(base['base_rank']):02d}_"
                    f"{common.sanitize_token(str(base['setting_id']), upper=True)}_"
                    f"{common.sanitize_token(str(setting['setting_id']), upper=True)}_"
                    f"S{int(seed_id)}"
                )
                run_phase = f"{PHASE_ID}_{run_suffix}"
                family_id = f"B{int(base['base_rank']):02d}_{base['setting_id']}__{setting['setting_id']}"
                row = {
                    "track": common.TRACK,
                    "axis_id": AXIS_ID,
                    "axis_desc": AXIS_DESC,
                    "phase_id": PHASE_ID,
                    "architecture_id": str(base.get("architecture_id") or common.ARCH_ID),
                    "architecture_key": str(base.get("architecture_key") or common.ARCH_KEY),
                    "architecture_name": str(base.get("architecture_name") or common.ARCH_NAME),
                    "exp_brief": str(base.get("architecture_name") or common.ARCH_NAME),
                    "dataset": dataset,
                    "run_phase": run_phase,
                    "run_id": run_suffix,
                    "stage": "router_input",
                    "tuning_stage": "router_input",
                    "setting_id": str(setting["setting_id"]),
                    "setting_key": str(setting["setting_key"]),
                    "setting_tier": str(setting.get("tier") or "essential"),
                    "setting_desc": str(setting.get("setting_desc") or setting["setting_key"]),
                    "setting_group": str(setting.get("setting_group") or "router_input"),
                    "setting_detail": str(setting.get("setting_detail") or ""),
                    "family_id": family_id,
                    "family_group": "router_input",
                    "variant_id": str(setting["setting_id"]),
                    "capacity_anchor": str(base.get("capacity_anchor") or ""),
                    "search_algo": str(args.search_algo),
                    "seed_id": int(seed_id),
                    "runtime_seed": int(args.seed_base) + cursor - 1,
                    "fixed_values": fixed_values,
                    "search_space": search_space,
                    "overrides": overrides,
                    "train_batch_size": int(base.get("train_batch_size") or 4096),
                    "eval_batch_size": int(base.get("eval_batch_size") or 4096),
                    "max_evals": int(args.max_evals),
                    "tune_epochs": int(args.tune_epochs),
                    "tune_patience": int(args.tune_patience),
                    "feature_mode": str(args.feature_mode),
                    "eval_mode": str(args.eval_mode),
                    "diag_logging": True,
                    "special_logging": True,
                    "feature_ablation_logging": True,
                    "base_dataset": dataset,
                    "base_rank": int(base["base_rank"]),
                    "base_key": str(base["base_key"]),
                    "base_run_phase": str(base["run_phase"]),
                    "base_result_json": str(base["result_json"]),
                    "base_source": str(base["source"]),
                    "base_test_mrr20": float(base.get("test_mrr20") or 0.0),
                    "base_valid_mrr20": float(base.get("valid_mrr20") or 0.0),
                    "base_setting_id": str(base.get("setting_id") or ""),
                    "base_capacity_anchor": str(base.get("capacity_anchor") or ""),
                    "source_feature_mode": str(base.get("source_feature_mode") or ""),
                    "runtime_feature_mode": str(args.feature_mode),
                    "runtime_eval_mode": str(args.eval_mode),
                    "lr_mode": "loguniform_grid5",
                    "diag_pair_max_points": int(args.diag_pair_max_points),
                    "diag_pair_bin_count": int(args.diag_pair_bin_count),
                }
                rows.append(row)
    return base_specs, rows


def build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    cmd = common.build_command(row, gpu_id, args)
    cmd.append(f"++fmoe_diag_pair_max_points={int(row.get('diag_pair_max_points', 4096))}")
    cmd.append(f"++fmoe_diag_pair_bin_count={int(row.get('diag_pair_bin_count', 20))}")
    return cmd


def main() -> int:
    args = parse_args()
    base_specs, rows = build_rows(args)
    rows = common.maybe_limit_smoke(rows, args)

    manifest = common.write_manifest(
        args=args,
        log_root=LOG_ROOT,
        default_name="router_input_ablation_manifest.json",
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        base_specs=base_specs,
        rows=rows,
    )
    print(f"[ablation-2-router-input] manifest -> {manifest}")

    # Reuse common launcher but inject extended summary columns.
    fieldnames = common.build_fieldnames(extra_cols=["diag_pair_max_points", "diag_pair_bin_count"])
    gpus = common._parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPU or cpu token selected")
    return int(
        common.launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=Path(LOG_ROOT),
            summary_path=Path(LOG_ROOT) / "summary.csv",
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
            build_command=build_command,
            build_log_path=common.build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: common.summary_path(Path(LOG_ROOT), str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
