#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    import stage1_a12_broad_templates as stage1
except ModuleNotFoundError:
    from run.fmoe_n4 import stage1_a12_broad_templates as stage1

TRACK = stage1.TRACK
AXIS = "MovieLens_V4_SessionFixed_Portfolio_2"
AXIS_ID = "N4MLV4_2"
AXIS_DESC = "movielens_v4_session_fixed_portfolio_2"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "P4ML2"
PHASE_NAME = "FMOE_N4_MOVIELENS_V4_SESSION_FIXED_2"
DEFAULT_DATASETS = ["movielens1m"]

REPO_ROOT_REAL = stage1.REPO_ROOT_REAL
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

# Reuse stage1 machinery but point metadata at this axis.
stage1.AXIS = AXIS
stage1.AXIS_ID = AXIS_ID
stage1.AXIS_DESC = AXIS_DESC
stage1.PHASE_ID = PHASE_ID
stage1.PHASE_NAME = PHASE_NAME
stage1.LOG_ROOT = LOG_ROOT


def _template_bank_8() -> list[Dict[str, Any]]:
    return [
        {
            "id": "ML201_h5_compact30_core",
            "band": "baseline_aligned_safe",
            "source": "BSARec_C2_compact30",
            "selection_score": "movielens_baseline_aligned_h5_len30_core",
            "anchor": "H5",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (2.0e-4, 9.0e-4),
            "len": 30,
            "d_feat": 12,
            "expert": 3,
            "router": 84,
            "hidden_dropout": 0.15,
            "attn_dropout": 0.10,
            "weight_decay": 2.0e-4,
            "family_drop": 0.0,
            "feature_drop": 0.0,
        },
        {
            "id": "ML202_h5_compact30_feat16",
            "band": "baseline_aligned_safe",
            "source": "FDSA_C1_compact30_gate",
            "selection_score": "movielens_feature_biased_h5_len30_feat16",
            "anchor": "H5",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (2.0e-4, 9.5e-4),
            "len": 30,
            "d_feat": 16,
            "expert": 3,
            "router": 84,
            "hidden_dropout": 0.15,
            "attn_dropout": 0.10,
            "weight_decay": 2.0e-4,
            "family_drop": 0.0,
            "feature_drop": 0.0,
        },
        {
            "id": "ML203_h4_light50_core",
            "band": "baseline_aligned_safe",
            "source": "BSARec_C1_light50",
            "selection_score": "movielens_light_h4_len50_core",
            "anchor": "H4",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (1.4e-4, 7.0e-4),
            "len": 50,
            "d_feat": 12,
            "expert": 2,
            "router": 56,
            "hidden_dropout": 0.25,
            "attn_dropout": 0.20,
            "weight_decay": 5.0e-4,
            "family_drop": 0.02,
            "feature_drop": 0.0,
        },
        {
            "id": "ML204_h7_compact30_core",
            "band": "baseline_aligned_safe",
            "source": "BSARec_C2_compact30_transfer",
            "selection_score": "movielens_baseline_aligned_h7_len30_core",
            "anchor": "H7",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (2.0e-4, 9.0e-4),
            "len": 30,
            "d_feat": 12,
            "expert": 3,
            "router": 80,
            "hidden_dropout": 0.15,
            "attn_dropout": 0.10,
            "weight_decay": 2.0e-4,
            "family_drop": 0.0,
            "feature_drop": 0.0,
        },
        {
            "id": "ML205_h5_compact30_feat16_router96",
            "band": "baseline_aligned_attack",
            "source": "FDSA_C1_plus_router_headroom",
            "selection_score": "movielens_feature_bias_router96_attack",
            "anchor": "H5",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (2.2e-4, 1.0e-3),
            "len": 30,
            "d_feat": 16,
            "expert": 3,
            "router": 96,
            "hidden_dropout": 0.15,
            "attn_dropout": 0.10,
            "weight_decay": 2.0e-4,
            "family_drop": 0.0,
            "feature_drop": 0.0,
        },
        {
            "id": "ML206_h7_compact40_feat16_expert4",
            "band": "baseline_aligned_attack",
            "source": "BSARec_C2_plus_feat16_expert4",
            "selection_score": "movielens_compact40_feat16_expert4_attack",
            "anchor": "H7",
            "lambda": (2.5e-4, 6e-5),
            "lr_bounds": (1.7e-4, 8.5e-4),
            "len": 40,
            "d_feat": 16,
            "expert": 4,
            "router": 80,
            "hidden_dropout": 0.17,
            "attn_dropout": 0.10,
            "weight_decay": 2.8e-4,
            "family_drop": 0.02,
            "feature_drop": 0.0,
        },
        {
            "id": "ML207_h4_light50_feat16_regularized",
            "band": "baseline_aligned_attack",
            "source": "FDSA_C2_light50_regularized",
            "selection_score": "movielens_light50_feat16_regularized_attack",
            "anchor": "H4",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (1.4e-4, 7.5e-4),
            "len": 50,
            "d_feat": 16,
            "expert": 2,
            "router": 56,
            "hidden_dropout": 0.25,
            "attn_dropout": 0.20,
            "weight_decay": 5.0e-4,
            "family_drop": 0.02,
            "feature_drop": 0.01,
        },
        {
            "id": "ML208_h6_bridge30_router84",
            "band": "bridge_attack",
            "source": "ML205_old_best_h6_bridge",
            "selection_score": "movielens_keep_one_h6_but_baseline_regularized",
            "anchor": "H6",
            "lambda": (2e-4, 5e-5),
            "lr_bounds": (2.4e-4, 1.0e-3),
            "len": 30,
            "d_feat": 16,
            "expert": 2,
            "router": 84,
            "hidden_dropout": 0.15,
            "attn_dropout": 0.10,
            "weight_decay": 2.2e-4,
            "family_drop": 0.01,
            "feature_drop": 0.0,
        },
    ]


def _select_templates(n_templates: int) -> list[Dict[str, Any]]:
    bank = _template_bank_8()
    if int(n_templates) == 8:
        return bank
    keep = {
        "ML201_h6_router84_focus",
        "ML202_h6_feat16_focus",
        "ML203_h6_expert3_focus",
        "ML204_h6_feat12_focus",
    }
    out = [template for template in bank if str(template["id"]) in keep]
    if len(out) != 4:
        raise RuntimeError("template subset construction failed")
    return out


def _row(dataset: str, template: Dict[str, Any], seed_id: int, runtime_seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    anchor = str(template["anchor"])
    cfg = stage1._anchor_cfg(anchor)
    template_id = str(template["id"])
    cons_lambda, z_lambda = template["lambda"]

    family_drop = float(template.get("family_drop", 0.02))
    feature_drop = float(template.get("feature_drop", 0.0))
    overrides = stage1._build_overrides(cons_lambda, z_lambda, family_drop, feature_drop)

    run_id = f"ML2_{stage1.sanitize_token(dataset, upper=True)}_{stage1.sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
    run_phase = f"{PHASE_ID}_{run_id}"

    fixed_values: Dict[str, Any] = {
        "embedding_size": int(cfg["embedding_size"]),
        "d_ff": int(cfg["d_ff"]),
        "d_expert_hidden": int(cfg["d_expert_hidden"]),
        "d_router_hidden": int(template["router"]),
        "MAX_ITEM_LIST_LENGTH": int(template["len"]),
        "d_feat_emb": int(template["d_feat"]),
        "expert_scale": int(template["expert"]),
        "lr_scheduler_type": "warmup_cosine",
        "num_heads": int(template.get("num_heads", 4)),
        "hidden_dropout_prob": float(template["hidden_dropout"]),
        "attn_dropout_prob": float(template["attn_dropout"]),
        "weight_decay": float(template["weight_decay"]),
    }

    train_batch_size, eval_batch_size, max_evals = stage1._template_batches(template, fixed_values, args)

    search_space: Dict[str, Any] = {
        "learning_rate": stage1._loguniform_spec(*template["lr_bounds"]),
    }

    band = str(template.get("band", "movielens2"))
    source_family_id = str(template.get("source", template_id))
    return {
        "dataset": dataset,
        "phase_id": PHASE_ID,
        "axis_id": AXIS_ID,
        "axis_desc": AXIS_DESC,
        "architecture_id": ARCH_ID,
        "architecture_key": ARCH_KEY,
        "architecture_name": ARCH_NAME,
        "exp_brief": ARCH_NAME,
        "run_phase": run_phase,
        "run_id": run_id,
        "setting_id": template_id,
        "setting_key": template_id,
        "setting_desc": template_id,
        "stage": "movielens_portfolio_v2",
        "tuning_stage": "movielens_portfolio_v2",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": band,
        "capacity_anchor": anchor,
        "selected_from_stage": str(template.get("source", band)),
        "selection_score": str(template.get("selection_score", "")),
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "movielens_portfolio_v2",
        "source_family_id": source_family_id,
        "template_count": int(args.template_count),
        "aux_route_consistency_lambda": float(cons_lambda),
        "aux_z_loss_lambda": float(z_lambda),
        "fixed_values": fixed_values,
        "search_space": search_space,
        "overrides": overrides,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "max_evals": int(max_evals),
        "tune_epochs": int(args.tune_epochs),
        "tune_patience": int(args.tune_patience),
    }


def build_rows(args: argparse.Namespace) -> list[Dict[str, Any]]:
    datasets = stage1._parse_csv_strings(args.datasets)
    if not datasets:
        raise RuntimeError("No datasets selected")
    seeds = stage1._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds selected")

    templates = _select_templates(int(args.template_count))

    rows: list[Dict[str, Any]] = []
    cursor = 0
    for dataset in datasets:
        stage1._validate_session_fixed_files(dataset)
        for template in templates:
            for seed_id in seeds:
                cursor += 1
                rows.append(
                    _row(
                        dataset=dataset,
                        template=template,
                        seed_id=int(seed_id),
                        runtime_seed=int(args.seed_base) + cursor - 1,
                        args=args,
                    )
                )
    return rows


def _manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest_out:
        raw = Path(str(args.manifest_out))
        if raw.suffix:
            return raw
        return raw / "movielens2_manifest.json"
    return LOG_ROOT / "movielens2_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "movielens_portfolio_v2",
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_count": len(rows),
        "datasets": sorted({str(row.get("dataset", "")) for row in rows}),
        "rows": [stage1._serialize_manifest_row(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def summary_path(dataset: str) -> Path:
    path = LOG_ROOT / str(dataset) / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N4 MovieLens v4 session_fixed portfolio 2 tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--template-count", type=int, choices=[4, 8], default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=267000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=6144)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--max-evals", type=int, default=12)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    args = parser.parse_args()
    if int(args.max_evals) < 1:
        raise RuntimeError("--max-evals must be >= 1")
    return args


def maybe_limit_smoke(rows: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if not bool(getattr(args, "smoke_test", False)):
        return rows
    return list(rows[: max(1, int(getattr(args, "smoke_max_runs", 2) or 2))])


def main() -> int:
    args = parse_args()
    rows = maybe_limit_smoke(build_rows(args), args)
    manifest = write_manifest(args, rows)
    print(f"[movielens-portfolio-2] manifest -> {manifest}")

    fieldnames = stage1.build_summary_fieldnames(
        [
            "architecture_id",
            "architecture_name",
            "tuning_stage",
            "family_id",
            "family_group",
            "variant_id",
            "capacity_anchor",
            "selected_from_stage",
            "selection_score",
            "search_algo",
            "source_family_id",
            "stage_group",
            "template_count",
            "aux_route_consistency_lambda",
            "aux_z_loss_lambda",
        ]
    )

    gpus = stage1._parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    return int(
        stage1.launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=AXIS,
            phase_id=PHASE_ID,
            phase_name=PHASE_NAME,
            log_dir=LOG_ROOT,
            summary_path=LOG_ROOT / "summary.csv",
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
            build_command=stage1.build_command,
            build_log_path=stage1.build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: summary_path(str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())