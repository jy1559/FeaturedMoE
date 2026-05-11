#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 Phase13 feature sanity wide runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from run_phase9_auxloss import (  # noqa: E402
    LOG_ROOT,
    TRACK,
    _apply_base_overrides,
    _base_definitions,
    _base_fixed_overrides,
    _dataset_tag,
    _parse_csv_ints,
    _parse_csv_strings,
    hydra_literal,
)
from run_phase_wide_common import build_summary_fieldnames, launch_wide_rows

AXIS = "phase13_feature_sanity_v1"
PHASE_ID = "P13"
PHASE_NAME = "PHASE13"
RUN_STAGE = "wide"
AXIS_ID = "A"
AXIS_DESC = "feature_sanity"


def _all_stage_map(value: Any) -> Dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _base_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    base_cfg = _base_definitions()["B4"]
    overrides = _base_fixed_overrides()
    _apply_base_overrides(
        overrides=overrides,
        base_cfg=base_cfg,
        feature_group_bias_lambda=float(args.feature_group_bias_lambda),
        rule_bias_scale=float(args.rule_bias_scale),
    )
    overrides["z_loss_lambda"] = float(args.z_loss_lambda)
    overrides["balance_loss_lambda"] = float(args.balance_loss_lambda)
    overrides["macro_history_window"] = int(args.macro_history_window)
    overrides["layer_layout"] = ["macro", "mid", "micro"]
    overrides["stage_router_granularity"] = {
        "macro": "session",
        "mid": "session",
        "micro": "token",
    }
    overrides["stage_feature_family_mask"] = {}
    overrides["stage_feature_family_topk"] = {}
    overrides["stage_feature_family_custom"] = {}
    overrides["stage_family_dropout_prob"] = _all_stage_map(0.0)
    overrides["stage_feature_dropout_prob"] = _all_stage_map(0.0)
    overrides["stage_feature_dropout_scope"] = _all_stage_map("token")
    overrides["feature_perturb_mode"] = "none"
    overrides["feature_perturb_apply"] = "none"
    overrides["feature_perturb_family"] = []
    overrides["feature_perturb_keywords"] = []
    overrides["feature_perturb_shift"] = 1
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
        "setting_key": f"P13-{int(idx):02d}_{key}",
        "setting_desc": key,
        "setting_group": group,
        "setting_detail": detail,
        "overrides": overrides,
    }


def _build_settings(args: argparse.Namespace) -> list[Dict[str, Any]]:
    base = _base_overrides(args)
    settings: list[Dict[str, Any]] = []

    # 4.1 Category-zero preprocessing proxy (2)
    settings.append(_setting(idx=0, key="FULL_DATA", group="data_condition", detail="Clean full feature setup", base=base))
    settings.append(
        _setting(
            idx=1,
            key="CATEGORY_ZERO_DATA",
            group="data_condition",
            detail="Category/theme columns zeroed while feature shape is preserved",
            base=base,
            extra_overrides={
                "feature_perturb_mode": "zero",
                "feature_perturb_apply": "both",
                "feature_perturb_keywords": ["cat", "theme"],
            },
        )
    )

    # 4.2 Eval-time perturbation (6)
    settings.append(_setting(idx=2, key="EVAL_ALL_ZERO", group="eval_perturb", detail="Eval-only zero all families", base=base, extra_overrides={"feature_perturb_mode": "zero", "feature_perturb_apply": "eval", "feature_perturb_family": []}))
    settings.append(_setting(idx=3, key="EVAL_ALL_SHUFFLE", group="eval_perturb", detail="Eval-only shuffle all families", base=base, extra_overrides={"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval", "feature_perturb_family": []}))
    settings.append(_setting(idx=4, key="EVAL_SHUFFLE_TEMPO", group="eval_perturb", detail="Eval-only shuffle Tempo family", base=base, extra_overrides={"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval", "feature_perturb_family": ["Tempo"]}))
    settings.append(_setting(idx=5, key="EVAL_SHUFFLE_FOCUS", group="eval_perturb", detail="Eval-only shuffle Focus family", base=base, extra_overrides={"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval", "feature_perturb_family": ["Focus"]}))
    settings.append(_setting(idx=6, key="EVAL_SHUFFLE_MEMORY", group="eval_perturb", detail="Eval-only shuffle Memory family", base=base, extra_overrides={"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval", "feature_perturb_family": ["Memory"]}))
    settings.append(_setting(idx=7, key="EVAL_SHUFFLE_EXPOSURE", group="eval_perturb", detail="Eval-only shuffle Exposure family", base=base, extra_overrides={"feature_perturb_mode": "shuffle", "feature_perturb_apply": "eval", "feature_perturb_family": ["Exposure"]}))

    # 4.3 Train-time corruption (6)
    settings.append(_setting(idx=8, key="TRAIN_GLOBAL_PERMUTE_ALL", group="train_corruption", detail="Train-only global permutation", base=base, extra_overrides={"feature_perturb_mode": "global_permute", "feature_perturb_apply": "train", "feature_perturb_family": []}))
    settings.append(_setting(idx=9, key="TRAIN_BATCH_PERMUTE_ALL", group="train_corruption", detail="Train-only batch permutation", base=base, extra_overrides={"feature_perturb_mode": "batch_permute", "feature_perturb_apply": "train", "feature_perturb_family": []}))
    settings.append(_setting(idx=10, key="TRAIN_PERMUTE_TEMPO", group="train_corruption", detail="Train-only family permutation Tempo", base=base, extra_overrides={"feature_perturb_mode": "family_permute", "feature_perturb_apply": "train", "feature_perturb_family": ["Tempo"]}))
    settings.append(_setting(idx=11, key="TRAIN_PERMUTE_FOCUS", group="train_corruption", detail="Train-only family permutation Focus", base=base, extra_overrides={"feature_perturb_mode": "family_permute", "feature_perturb_apply": "train", "feature_perturb_family": ["Focus"]}))
    settings.append(_setting(idx=12, key="TRAIN_PERMUTE_MEMORY", group="train_corruption", detail="Train-only family permutation Memory", base=base, extra_overrides={"feature_perturb_mode": "family_permute", "feature_perturb_apply": "train", "feature_perturb_family": ["Memory"]}))
    settings.append(_setting(idx=13, key="TRAIN_PERMUTE_EXPOSURE", group="train_corruption", detail="Train-only family permutation Exposure", base=base, extra_overrides={"feature_perturb_mode": "family_permute", "feature_perturb_apply": "train", "feature_perturb_family": ["Exposure"]}))

    # 4.4 Semantic mismatch / role mismatch (3)
    settings.append(_setting(idx=14, key="FEATURE_ROLE_SWAP", group="semantic_mismatch", detail="Role swap across families", base=base, extra_overrides={"feature_perturb_mode": "role_swap", "feature_perturb_apply": "both", "feature_perturb_family": []}))
    settings.append(_setting(idx=15, key="STAGE_MISMATCH_ASSIGN", group="semantic_mismatch", detail="Stage mismatch assignment", base=base, extra_overrides={"feature_perturb_mode": "stage_mismatch", "feature_perturb_apply": "both", "feature_perturb_family": []}))
    settings.append(_setting(idx=16, key="POSITION_SHIFT_FEATURE", group="semantic_mismatch", detail="Position shift on all families (shift=1)", base=base, extra_overrides={"feature_perturb_mode": "position_shift", "feature_perturb_apply": "both", "feature_perturb_shift": 1, "feature_perturb_family": []}))

    # 8-multiple completion (+7)
    settings.append(_setting(idx=17, key="EVAL_ZERO_TEMPO", group="eval_perturb_extra", detail="Eval-only zero Tempo family", base=base, extra_overrides={"feature_perturb_mode": "zero", "feature_perturb_apply": "eval", "feature_perturb_family": ["Tempo"]}))
    settings.append(_setting(idx=18, key="EVAL_ZERO_FOCUS", group="eval_perturb_extra", detail="Eval-only zero Focus family", base=base, extra_overrides={"feature_perturb_mode": "zero", "feature_perturb_apply": "eval", "feature_perturb_family": ["Focus"]}))
    settings.append(_setting(idx=19, key="EVAL_ZERO_MEMORY", group="eval_perturb_extra", detail="Eval-only zero Memory family", base=base, extra_overrides={"feature_perturb_mode": "zero", "feature_perturb_apply": "eval", "feature_perturb_family": ["Memory"]}))
    settings.append(_setting(idx=20, key="EVAL_ZERO_EXPOSURE", group="eval_perturb_extra", detail="Eval-only zero Exposure family", base=base, extra_overrides={"feature_perturb_mode": "zero", "feature_perturb_apply": "eval", "feature_perturb_family": ["Exposure"]}))
    settings.append(_setting(idx=21, key="TRAIN_POSITION_SHIFT_PLUS1", group="train_shift_extra", detail="Train-only position shift +1", base=base, extra_overrides={"feature_perturb_mode": "position_shift", "feature_perturb_apply": "train", "feature_perturb_shift": 1, "feature_perturb_family": []}))
    settings.append(_setting(idx=22, key="TRAIN_POSITION_SHIFT_PLUS2", group="train_shift_extra", detail="Train-only position shift +2", base=base, extra_overrides={"feature_perturb_mode": "position_shift", "feature_perturb_apply": "train", "feature_perturb_shift": 2, "feature_perturb_family": []}))
    settings.append(_setting(idx=23, key="TRAIN_POSITION_SHIFT_PLUS3", group="train_shift_extra", detail="Train-only position shift +3", base=base, extra_overrides={"feature_perturb_mode": "position_shift", "feature_perturb_apply": "train", "feature_perturb_shift": 3, "feature_perturb_family": []}))

    only = {tok.upper() for tok in _parse_csv_strings(args.only_setting)}
    if only:
        settings = [
            s for s in settings
            if str(s["setting_key"]).upper() in only
            or str(s["setting_desc"]).upper() in only
            or str(s["setting_id"]).upper() in only
        ]

    settings.sort(key=lambda x: int(x["setting_idx"]))
    if not settings:
        raise RuntimeError("No phase13 settings selected. Check --only-setting.")
    return settings


def _phase_log_dir(dataset: str) -> Path:
    return LOG_ROOT / AXIS / _dataset_tag(dataset)


def _summary_path(dataset: str) -> Path:
    return _phase_log_dir(dataset) / "summary.csv"


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        return Path(args.manifest_out)
    return _phase_log_dir(args.dataset) / "phase13_feature_sanity_matrix.json"


def _hparam_profile(hparam_id: int, args: argparse.Namespace) -> Dict[str, Any]:
    if int(hparam_id) != 1:
        raise ValueError(f"Unsupported hparam_id={hparam_id}. Phase13 wide defaults to H1 only.")
    return {
        "embedding_size": int(args.embedding_size),
        "d_ff": int(args.d_ff),
        "d_expert_hidden": int(args.d_expert_hidden),
        "d_router_hidden": int(args.d_router_hidden),
        "search_lr_min": float(args.search_lr_min),
        "search_lr_max": float(args.search_lr_max),
        "fixed_weight_decay": float(args.fixed_weight_decay),
        "fixed_hidden_dropout_prob": float(args.fixed_hidden_dropout_prob),
        "search_lr_scheduler": _parse_csv_strings(args.search_lr_scheduler),
    }


def _build_rows(args: argparse.Namespace, settings: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    hparams = _parse_csv_ints(args.hparams)
    seeds = _parse_csv_ints(args.seeds)
    if not hparams:
        raise RuntimeError("No hparams provided")
    if not seeds:
        raise RuntimeError("No seeds provided")

    rows: list[Dict[str, Any]] = []
    run_cursor = 0
    for setting in settings:
        for hparam_id in hparams:
            _hparam_profile(int(hparam_id), args)
            for seed_id in seeds:
                run_cursor += 1
                setting_id = str(setting["setting_id"])
                run_phase = f"{PHASE_ID}_{AXIS_ID}{setting_id}_H{int(hparam_id)}_S{int(seed_id)}"
                run_id = f"{AXIS_ID}{setting_id}_H{int(hparam_id)}_S{int(seed_id)}"
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
                        "hparam_id": int(hparam_id),
                        "seed_id": int(seed_id),
                        "run_phase": run_phase,
                        "run_id": run_id,
                        "exp_brief": f"{setting['setting_key']} | {setting['setting_group']} | {setting['setting_detail']}",
                        "stage": RUN_STAGE,
                        "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                        "overrides": copy.deepcopy(dict(setting.get("overrides", {}) or {})),
                    }
                )
    return rows


def _build_command(row: Dict[str, Any], gpu_id: str, args: argparse.Namespace) -> list[str]:
    profile = _hparam_profile(int(row["hparam_id"]), args)
    python_bin = os.environ.get("RUN_PYTHON_BIN", "/venv/FMoE/bin/python")
    cmd = [
        python_bin,
        "hyperopt_tune.py",
        "--config-name",
        "config",
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
        "eval_mode=session",
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
        f"MAX_ITEM_LIST_LENGTH={int(args.max_item_list_length)}",
        f"train_batch_size={int(args.batch_size)}",
        f"eval_batch_size={int(args.batch_size)}",
        f"embedding_size={int(profile['embedding_size'])}",
        f"num_heads={int(args.num_heads)}",
        f"attn_dropout_prob={hydra_literal(float(args.attn_dropout_prob))}",
        f"d_ff={int(profile['d_ff'])}",
        f"d_feat_emb={int(args.d_feat_emb)}",
        f"d_expert_hidden={int(profile['d_expert_hidden'])}",
        f"d_router_hidden={int(profile['d_router_hidden'])}",
        f"expert_scale={int(args.expert_scale)}",
        f"++search.learning_rate={hydra_literal([float(profile['search_lr_min']), float(profile['search_lr_max'])])}",
        f"++search.weight_decay={hydra_literal([float(profile['fixed_weight_decay'])])}",
        f"++search.hidden_dropout_prob={hydra_literal([float(profile['fixed_hidden_dropout_prob'])])}",
        f"++search.lr_scheduler_type={hydra_literal(profile['search_lr_scheduler'])}",
        "++search_space_type_overrides.learning_rate=loguniform",
        "++search_space_type_overrides.weight_decay=choice",
        "++search_space_type_overrides.hidden_dropout_prob=choice",
        "++search_space_type_overrides.lr_scheduler_type=choice",
        f"++phase_run_type={hydra_literal(RUN_STAGE)}",
        f"++phase_axis_id={hydra_literal(row['axis_id'])}",
        f"++phase_axis_desc={hydra_literal(row['axis_desc'])}",
        f"++phase_setting_id={hydra_literal(row['setting_id'])}",
        f"++phase_setting_key={hydra_literal(row['setting_key'])}",
        f"++phase_setting_desc={hydra_literal(row['setting_desc'])}",
        f"++phase_hparam_id={hydra_literal(row['hparam_id'])}",
        f"++phase_seed_id={hydra_literal(row['seed_id'])}",
        f"++phase_run_id={hydra_literal(row['run_id'])}",
    ]
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
        "setting_count": len(settings),
        "hparam_count": len(_parse_csv_ints(args.hparams)),
        "seed_count": len(_parse_csv_ints(args.seeds)),
        "run_count": len(rows),
        "run_count_formula": f"{len(settings)} x {len(_parse_csv_ints(args.hparams))} x {len(_parse_csv_ints(args.seeds))}",
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
                "hparam_id": r["hparam_id"],
                "seed_id": r["seed_id"],
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Phase13 feature sanity wide launcher")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--hparams", default="1", help="Wide default H1")
    parser.add_argument("--seeds", default="1", help="Wide default S1")
    parser.add_argument("--seed-base", type=int, default=73000)

    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--z-loss-lambda", type=float, default=1e-4)
    parser.add_argument("--balance-loss-lambda", type=float, default=0.0)
    parser.add_argument("--macro-history-window", type=int, default=5)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)

    parser.add_argument("--embedding-size", type=int, default=160)
    parser.add_argument("--d-ff", type=int, default=320)
    parser.add_argument("--d-expert-hidden", type=int, default=160)
    parser.add_argument("--d-router-hidden", type=int, default=80)
    parser.add_argument("--fixed-weight-decay", type=float, default=2e-6)
    parser.add_argument("--fixed-hidden-dropout-prob", type=float, default=0.18)

    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--only-setting", default="", help="Comma-separated subset of setting key/desc/id")
    parser.add_argument("--manifest-out", default="", help="Optional matrix JSON output path")

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
    args.hparams = "1"
    args.seeds = "1"
    gpus = _parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"
    if not args.only_setting:
        args.only_setting = "P13-00_FULL_DATA,P13-17_EVAL_ZERO_TEMPO"


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
        raise RuntimeError("No phase13 rows to launch")

    manifest_path = _write_manifest(args, settings, rows)
    print(
        f"[{PHASE_ID}] dataset={args.dataset} execution={RUN_STAGE} settings={len(settings)} rows={len(rows)} "
        f"axis={AXIS} manifest={manifest_path}"
    )
    print(
        f"[{PHASE_ID}] resume_from_logs={args.resume_from_logs} verify_logging={args.verify_logging} "
        f"dry_run={args.dry_run}"
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
        "hparam_id",
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
