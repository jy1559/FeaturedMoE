#!/usr/bin/env python3
"""Launch KuaiRec-only A12 ablation 12-setting runs for FMoE_N3.

Key decisions baked into this runner:
- Baseline is fixed (no code-time baseline selection).
- Search uses learning_rate choice(5) only with max_evals=5.
- Choice-only search enables UniqueChoice de-duplication in hyperopt_tune.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from run_phase9_auxloss import (  # noqa: E402
    LOG_ROOT,
    TRACK,
    _apply_base_overrides,
    _base_fixed_overrides,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows

AXIS = "ablation_feature_add_v3_a12_kuairec_v1"
PHASE_ID = "P17A"
PHASE_NAME = "A12_ABLATION_12"
RUN_STAGE = "wide"
AXIS_ID = "A"
AXIS_DESC = "kuairec_a12_ablation12"

BASELINE_RESULT_FILE = (
    "/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n3/normal/final_tuning_a12/"
    "P16A/KuaiRecLargeStrictPosV2_0.2/FMoEN3/"
    "KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_p16a_stage1_kuaireclargestrictposv2_0_2_"
    "ku_h14_feature_strong_s1_20260413_082506_778127_pid572496.json"
)
BASELINE_SCORE = 0.1721 + 0.5 * 0.1695

# Baseline run settings mirrored from P16A stage1 (KU strong group) launch policy.
BASELINE_RUN_SETTINGS: Dict[str, Any] = {
    "search_algo": "random",
    "max_evals": 5,
    "tune_epochs": 30,
    "tune_patience": 4,
    "batch_size": 4096,
    "eval_batch_size": 4096,
}

# Fixed baseline params from selected run + H14 anchor.
BASELINE_FIXED_VALUES: Dict[str, Any] = {
    "embedding_size": 256,
    "d_ff": 512,
    "d_expert_hidden": 256,
    "d_router_hidden": 128,
    "weight_decay": 5e-7,
    "num_heads": 4,
    "attn_dropout_prob": 0.1,
    "hidden_dropout_prob": 0.05,
    "MAX_ITEM_LIST_LENGTH": 20,
    "d_feat_emb": 16,
    "expert_scale": 4,
    "lr_scheduler_type": "warmup_cosine",
}

# Narrow LR candidates (choice-only) for fast ablation.
LR_CHOICES = [0.00015, 0.0002679394909038652, 0.00035, 0.0005, 0.0008]

CATEGORY_DROP_KEYWORDS = ["cat", "theme"]
TIMESTAMP_DROP_KEYWORDS = [
    "timestamp",
    "gap",
    "pace",
    "int_",
    "_int",
    "sess_age",
    "ctx_valid_r",
    "valid_r",
    "delta_vs_mid",
]



def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}



def _phase_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / str(dataset)



def _summary_path(dataset: str) -> Path:
    return _phase_log_dir(dataset) / "summary.csv"



def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        return Path(args.manifest_out)
    return _phase_log_dir(args.dataset) / "ablation12_matrix.json"



def _stage_mask(*, keep_macro: list[str], keep_mid: list[str], keep_micro: list[str]) -> Dict[str, list[str]]:
    return {
        "macro": list(keep_macro),
        "mid": list(keep_mid),
        "micro": list(keep_micro),
    }



def _base_a12_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    base_cfg = {
        "wrapper_map": {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
        "source_profile": "src_abc_feature",
        "bias_mode": "bias_both",
    }
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
    )

    # A12 core defaults.
    overrides["layer_layout"] = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
    overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
    overrides["stage_compute_mode"] = _all_stage_map("moe")
    overrides["stage_router_mode"] = _all_stage_map("learned")
    overrides["stage_router_source"] = _all_stage_map("both")
    overrides["stage_feature_injection"] = _all_stage_map("none")

    # Baseline-fixed regularization knobs.
    overrides["macro_history_window"] = int(args.macro_history_window)
    overrides["stage_family_dropout_prob"] = _all_stage_map(0.02)
    overrides["stage_feature_dropout_prob"] = _all_stage_map(0.0)
    overrides["stage_feature_dropout_scope"] = _all_stage_map("token")

    # Keep A2-like stabilization but disable balance/bias penalties.
    overrides["route_consistency_pairs"] = 1
    overrides["route_consistency_lambda"] = float(args.a2_route_consistency_lambda)
    overrides["route_consistency_min_sim"] = float(args.a2_route_consistency_min_sim)
    overrides["z_loss_lambda"] = float(args.a2_z_loss_lambda)
    overrides["balance_loss_lambda"] = 0.0
    overrides["route_monopoly_lambda"] = 0.0

    # Bias-free A12 baseline.
    overrides["bias_mode"] = "none"
    overrides["rule_bias_scale"] = 0.0
    overrides["feature_group_bias_lambda"] = 0.0

    # Ensure feature removal is structural (not perturbation/zero-fill).
    overrides["feature_perturb_mode"] = "none"
    overrides["feature_perturb_apply"] = "none"
    overrides["feature_perturb_family"] = []
    overrides["feature_perturb_keywords"] = []
    overrides["stage_feature_family_mask"] = {}
    overrides["stage_feature_family_topk"] = {}
    overrides["stage_feature_family_custom"] = {}
    overrides["stage_feature_drop_keywords"] = []
    return overrides



def _setting(
    *,
    idx: int,
    key: str,
    group: str,
    detail: str,
    base: Dict[str, Any],
    extra_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides = copy.deepcopy(base)
    for k, v in dict(extra_overrides or {}).items():
        overrides[str(k)] = copy.deepcopy(v)
    return {
        "setting_idx": int(idx),
        "setting_id": f"{int(idx):02d}",
        "setting_key": f"ABL-{int(idx):02d}_{key}",
        "setting_desc": key,
        "setting_group": group,
        "setting_detail": detail,
        "overrides": overrides,
    }



def _build_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    base = _base_a12_overrides(args)
    settings: list[Dict[str, Any]] = []

    # 1) Baseline
    settings.append(
        _setting(
            idx=1,
            key="BASELINE_A12",
            group="baseline",
            detail="A12 fixed baseline selected by valid+0.5*test score",
            base=base,
        )
    )

    # 2) Layout ablations (4)
    settings.append(
        _setting(
            idx=2,
            key="LAYOUT_NO_MACRO",
            group="layout",
            detail="Replace macro_ffn with dense_plain FFN (no skip)",
            base=base,
            extra_overrides={
                "layer_layout": ["attn", "macro", "mid_ffn", "attn", "micro_ffn"],
                "stage_compute_mode": {"macro": "dense_plain", "mid": "moe", "micro": "moe"},
                "stage_router_mode": {"macro": "none", "mid": "learned", "micro": "learned"},
            },
        )
    )
    settings.append(
        _setting(
            idx=3,
            key="LAYOUT_NO_MID",
            group="layout",
            detail="Replace mid_ffn with dense_plain FFN (no skip)",
            base=base,
            extra_overrides={
                "layer_layout": ["attn", "macro_ffn", "mid", "attn", "micro_ffn"],
                "stage_compute_mode": {"macro": "moe", "mid": "dense_plain", "micro": "moe"},
                "stage_router_mode": {"macro": "learned", "mid": "none", "micro": "learned"},
            },
        )
    )
    settings.append(
        _setting(
            idx=4,
            key="LAYOUT_NO_MICRO",
            group="layout",
            detail="Replace micro_ffn with dense_plain FFN (no skip)",
            base=base,
            extra_overrides={
                "layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "micro"],
                "stage_compute_mode": {"macro": "moe", "mid": "moe", "micro": "dense_plain"},
                "stage_router_mode": {"macro": "learned", "mid": "learned", "micro": "none"},
            },
        )
    )
    settings.append(
        _setting(
            idx=5,
            key="LAYOUT_ATTN_BEFORE_MID",
            group="layout",
            detail="Insert attention block before mid_ffn",
            base=base,
            extra_overrides={"layer_layout": ["attn", "macro_ffn", "attn", "mid_ffn", "attn", "micro_ffn"]},
        )
    )

    # 3) Feature removals (structural, no zero padding) (3)
    settings.append(
        _setting(
            idx=6,
            key="FEATURE_DROP_CATEGORY",
            group="feature_drop",
            detail="Drop category-derived features",
            base=base,
            extra_overrides={
                "stage_feature_family_mask": _stage_mask(
                    keep_macro=["Tempo", "Memory", "Exposure"],
                    keep_mid=["Tempo", "Memory", "Exposure"],
                    keep_micro=["Tempo", "Memory", "Exposure"],
                ),
                "stage_feature_drop_keywords": list(CATEGORY_DROP_KEYWORDS),
            },
        )
    )
    settings.append(
        _setting(
            idx=7,
            key="FEATURE_DROP_TIMESTAMP_DERIVED",
            group="feature_drop",
            detail="Drop timestamp-derived features",
            base=base,
            extra_overrides={
                "stage_feature_family_mask": _stage_mask(
                    keep_macro=["Focus", "Memory", "Exposure"],
                    keep_mid=["Focus", "Memory", "Exposure"],
                    keep_micro=["Focus", "Memory", "Exposure"],
                ),
                "stage_feature_drop_keywords": list(TIMESTAMP_DROP_KEYWORDS),
            },
        )
    )
    settings.append(
        _setting(
            idx=8,
            key="FEATURE_DROP_CATEGORY_TIMESTAMP",
            group="feature_drop",
            detail="Drop category + timestamp-derived features",
            base=base,
            extra_overrides={
                "stage_feature_family_mask": _stage_mask(
                    keep_macro=["Memory", "Exposure"],
                    keep_mid=["Memory", "Exposure"],
                    keep_micro=["Memory", "Exposure"],
                ),
                "stage_feature_drop_keywords": list(CATEGORY_DROP_KEYWORDS + TIMESTAMP_DROP_KEYWORDS),
            },
        )
    )

    # 4) Dense FFN + gated_bias injection (2)
    settings.append(
        _setting(
            idx=9,
            key="DENSE_GATED_BIAS_MACRO_MID",
            group="no_moe_bias",
            detail="Dense macro+mid with gated_bias injection",
            base=base,
            extra_overrides={
                "layer_layout": ["macro", "mid"],
                "stage_compute_mode": {"macro": "dense_plain", "mid": "dense_plain", "micro": "none"},
                "stage_router_mode": {"macro": "none", "mid": "none", "micro": "none"},
                "stage_feature_injection": {"macro": "gated_bias", "mid": "gated_bias", "micro": "none"},
            },
        )
    )
    settings.append(
        _setting(
            idx=10,
            key="DENSE_GATED_BIAS_FULL",
            group="no_moe_bias",
            detail="Dense macro+mid+micro with gated_bias injection",
            base=base,
            extra_overrides={
                "layer_layout": ["macro", "mid", "micro"],
                "stage_compute_mode": {"macro": "dense_plain", "mid": "dense_plain", "micro": "dense_plain"},
                "stage_router_mode": {"macro": "none", "mid": "none", "micro": "none"},
                "stage_feature_injection": _all_stage_map("gated_bias"),
            },
        )
    )

    # 5) Rule routers (2)
    settings.append(
        _setting(
            idx=11,
            key="RULE_ROUTER_ALL",
            group="rule_router",
            detail="Rule-soft router on all stages",
            base=base,
            extra_overrides={
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_mode": _all_stage_map("rule_soft"),
                "stage_router_source": _all_stage_map("both"),
            },
        )
    )
    settings.append(
        _setting(
            idx=12,
            key="RULE_ROUTER_MACRO_ONLY",
            group="rule_router",
            detail="Macro rule-soft, mid/micro learned",
            base=base,
            extra_overrides={
                "stage_compute_mode": _all_stage_map("moe"),
                "stage_router_mode": {"macro": "rule_soft", "mid": "learned", "micro": "learned"},
                "stage_router_source": _all_stage_map("both"),
            },
        )
    )

    only = {tok.upper() for tok in _parse_csv_strings(args.only_setting)}
    if only:
        settings = [
            s
            for s in settings
            if str(s["setting_key"]).upper() in only
            or str(s["setting_desc"]).upper() in only
            or str(s["setting_id"]).upper() in only
        ]

    settings.sort(key=lambda x: int(x["setting_idx"]))
    if not settings:
        raise RuntimeError("No settings selected. Check --only-setting.")
    return settings



def _build_rows(args: argparse.Namespace, settings: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    for setting in settings:
        for seed_id in seeds:
            run_cursor += 1
            setting_id = str(setting["setting_id"])
            run_phase = f"{PHASE_ID}_{AXIS_ID}{setting_id}_S{int(seed_id)}"
            run_id = f"{AXIS_ID}{setting_id}_S{int(seed_id)}"
            rows.append(
                {
                    "dataset": args.dataset,
                    "phase_id": PHASE_ID,
                    "axis_id": AXIS_ID,
                    "axis_desc": AXIS_DESC,
                    "setting_idx": int(setting["setting_idx"]),
                    "setting_id": setting_id,
                    "setting_key": str(setting["setting_key"]),
                    "setting_desc": str(setting["setting_desc"]),
                    "setting_group": str(setting["setting_group"]),
                    "setting_detail": str(setting["setting_detail"]),
                    "seed_id": int(seed_id),
                    "run_phase": run_phase,
                    "run_id": run_id,
                    "exp_brief": f"{setting['setting_key']} | {setting['setting_detail']}",
                    "stage": RUN_STAGE,
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "fixed_values": copy.deepcopy(BASELINE_FIXED_VALUES),
                    "overrides": copy.deepcopy(dict(setting.get("overrides", {}) or {})),
                }
            )
    return rows



def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
        "--search-algo",
        str(args.search_algo),
        "--max-evals",
        str(int(args.max_evals)),
        "--tune-epochs",
        str(int(args.tune_epochs)),
        "--tune-patience",
        str(int(args.tune_patience)),
        "--seed",
        str(int(row["runtime_seed"])),
        "--run-group",
        TRACK,
        "--run-axis",
        AXIS,
        "--run-phase",
        str(row["run_phase"]),
        "model=featured_moe_n3_tune",
        f"dataset={row['dataset']}",
        "eval_mode=session_fixed",
        "feature_mode=full_v3",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "enable_tf32=true",
        "fmoe_debug_logging=false",
        "fmoe_diag_logging=true",
        "fmoe_special_logging=true",
        "fmoe_feature_ablation_logging=false",
        "fmoe_eval_logging_timing=final_only",
        f"++fmoe_logging_output_root={hydra_literal('run/artifacts/logging')}",
        f"++fmoe_phase={hydra_literal(PHASE_ID)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.eval_batch_size)}",
        f"++search.learning_rate={hydra_literal(list(LR_CHOICES))}",
        "++search_space_type_overrides.learning_rate=choice",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(row['axis_id'])}",
        f"++phase_axis_desc={hydra_literal(row['axis_desc'])}",
        f"++phase_setting_id={hydra_literal(row['setting_id'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++phase_seed_id={hydra_literal(row['seed_id'])}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
    ]

    for key, value in dict(row.get("fixed_values", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    for key, value in dict(row.get("overrides", {}) or {}).items():
        cmd.append(f"++{key}={hydra_literal(value)}")
    return cmd



def _write_manifest(args: argparse.Namespace, settings: list[Dict[str, Any]], rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "phase": PHASE_ID,
        "dataset": args.dataset,
        "execution_type": RUN_STAGE,
        "baseline": {
            "selected_result_file": BASELINE_RESULT_FILE,
            "score_formula": "best_valid_mrr20 + 0.5 * test_mrr20",
            "score": BASELINE_SCORE,
            "baseline_run_settings": BASELINE_RUN_SETTINGS,
            "baseline_fixed_values": BASELINE_FIXED_VALUES,
            "lr_choices": LR_CHOICES,
        },
        "setting_count": len(settings),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "settings": [
            {
                "setting_idx": s["setting_idx"],
                "setting_id": s["setting_id"],
                "setting_key": s["setting_key"],
                "setting_desc": s["setting_desc"],
                "setting_group": s["setting_group"],
                "setting_detail": s["setting_detail"],
            }
            for s in settings
        ],
        "rows": [
            {
                "run_phase": r["run_phase"],
                "run_id": r["run_id"],
                "setting_key": r["setting_key"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 KuaiRec A12 ablation-12 launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=171000)

    parser.add_argument("--search-algo", choices=("tpe", "random"), default=str(BASELINE_RUN_SETTINGS["search_algo"]))
    parser.add_argument("--max-evals", type=int, default=int(BASELINE_RUN_SETTINGS["max_evals"]))
    parser.add_argument("--tune-epochs", type=int, default=int(BASELINE_RUN_SETTINGS["tune_epochs"]))
    parser.add_argument("--tune-patience", type=int, default=int(BASELINE_RUN_SETTINGS["tune_patience"]))

    parser.add_argument("--batch-size", type=int, default=int(BASELINE_RUN_SETTINGS["batch_size"]))
    parser.add_argument("--eval-batch-size", type=int, default=int(BASELINE_RUN_SETTINGS["eval_batch_size"]))

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--macro-history-window", type=int, default=5)
    parser.add_argument("--a2-route-consistency-lambda", type=float, default=8e-4)
    parser.add_argument("--a2-route-consistency-min-sim", type=float, default=0.995)
    parser.add_argument("--a2-z-loss-lambda", type=float, default=2e-4)

    parser.add_argument("--only-setting", default="", help="Comma-separated subset by key/desc/id")
    parser.add_argument("--manifest-out", default="", help="Optional matrix JSON path")

    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")

    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    return parser.parse_args()



def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.max_evals = 1
    args.tune_epochs = 1
    args.tune_patience = 1
    args.seeds = "1"
    gpus = _parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"
    if not args.only_setting:
        args.only_setting = "ABL-01_BASELINE_A12,ABL-07_FEATURE_DROP_TIMESTAMP_DERIVED"



def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)

    gpus = _parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs provided")

    settings = _build_settings(args)
    rows = _build_rows(args, settings)
    if args.smoke_test:
        rows = rows[: max(int(args.smoke_max_runs), 1)]
    if not rows:
        raise RuntimeError("No rows to launch")

    manifest_path = _write_manifest(args, settings, rows)
    print(
        f"[{PHASE_ID}] dataset={args.dataset} execution={RUN_STAGE} settings={len(settings)} rows={len(rows)} "
        f"axis={AXIS} manifest={manifest_path}"
    )
    print(
        f"[{PHASE_ID}] baseline_score={BASELINE_SCORE:.6f} lr_choices={LR_CHOICES} "
        f"resume_from_logs={args.resume_from_logs} verify_logging={args.verify_logging} dry_run={args.dry_run}"
    )
    print(
        f"[{PHASE_ID}] run_settings search_algo={args.search_algo} max_evals={args.max_evals} "
        f"tune_epochs={args.tune_epochs} tune_patience={args.tune_patience} "
        f"train_batch_size={args.batch_size} eval_batch_size={args.eval_batch_size}"
    )

    log_dir = _phase_log_dir(args.dataset)
    summary_path = _summary_path(args.dataset)
    extra_cols = [
        "phase_id",
        "axis_id",
        "axis_desc",
        "setting_id",
        "setting_key",
        "setting_desc",
        "setting_group",
        "setting_detail",
        "seed_id",
        "run_id",
    ]
    fieldnames = build_summary_fieldnames(extra_cols)
    return launch_wide_rows(
        rows=rows,
        gpus=gpus,
        args=args,
        axis=AXIS,
        phase_id=PHASE_ID,
        phase_name=PHASE_NAME,
        log_dir=log_dir,
        summary_path=summary_path,
        fieldnames=fieldnames,
        extra_cols=extra_cols,
        build_command=_build_command,
        verify_logging=bool(args.verify_logging),
    )


if __name__ == "__main__":
    raise SystemExit(main())
