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
AXIS = "MovieLens_V4_SessionFixed_Portfolio"
AXIS_ID = "N4MLV4"
AXIS_DESC = "movielens_v4_session_fixed_portfolio"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "P4ML"
PHASE_NAME = "FMOE_N4_MOVIELENS_V4_SESSION_FIXED"
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


def _template_bank_16() -> list[Dict[str, Any]]:
    return [
        {
            "id": "ML01_h6_lr_core",
            "band": "validate_core",
            "source": "M08_h6_compact_validate",
            "selection_score": "current_best_movielens_h6_lr_refine",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 1.4e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "router": 72,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML02_h6_len_scout",
            "band": "validate_local",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_compact_len16_20_24",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.2e-3),
            "len_choices": [16, 20, 24],
            "d_feat": 12,
            "expert": 2,
            "router": 72,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML03_h6_expert_up",
            "band": "validate_local",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_compact_expert2_3",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.2e-4, 1.2e-3),
            "len": 20,
            "d_feat": 12,
            "expert_choices": [2, 3],
            "router": 72,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML04_h6_feat_up",
            "band": "validate_local",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_compact_feat12_16",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.2e-3),
            "len": 20,
            "d_feat_choices": [12, 16],
            "expert": 2,
            "router": 72,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML05_h6_router_up",
            "band": "validate_local",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_compact_router72_84",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.2e-3),
            "len": 20,
            "d_feat": 12,
            "expert": 2,
            "router_choices": [72, 84],
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML06_h6_len_expert_combo",
            "band": "combo_small",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_len_expert_small_combo",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.2e-4, 1.2e-3),
            "len_choices": [16, 20, 24],
            "d_feat": 12,
            "expert_choices": [2, 3],
            "router": 72,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML07_h6_feat_router_combo",
            "band": "combo_small",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_feat_router_small_combo",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 1.15e-3),
            "len": 20,
            "d_feat_choices": [12, 16],
            "expert": 2,
            "router_choices": [72, 84],
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.67, 1.0],
        },
        {
            "id": "ML08_h6_len_dropout_combo",
            "band": "combo_small",
            "source": "M08_h6_compact_validate",
            "selection_score": "h6_len_hidden_local_combo",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.8e-4, 1.1e-3),
            "len_choices": [20, 24, 28],
            "d_feat": 12,
            "expert": 2,
            "router": 72,
            "hidden": [0.08, 0.10, 0.12],
            "attn": [0.08, 0.10],
            "wd_scales": [0.67, 1.0],
            "family_drop_choices": [0.02, 0.04],
            "feature_drop_choices": [0.0, 0.02],
        },
        {
            "id": "ML09_h5_long_context",
            "band": "aggressive_transfer",
            "source": "M03_h5_len25_transfer",
            "selection_score": "h5_revisit_longer_context",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 8.5e-4),
            "len_choices": [24, 28, 32],
            "d_feat": 12,
            "expert": 3,
            "router": 84,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [1.0, 2.0],
        },
        {
            "id": "ML10_h5_compact_attack",
            "band": "aggressive_transfer",
            "source": "M03_h5_len25_transfer",
            "selection_score": "h5_compact_len_feat_mix",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 1.0e-3),
            "len_choices": [20, 24, 28],
            "d_feat_choices": [12, 16],
            "expert_choices": [3, 4],
            "router": 84,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.5, 1.0],
        },
        {
            "id": "ML11_h10_context_bridge",
            "band": "aggressive_context",
            "source": "M04_h10_len25_context",
            "selection_score": "h10_context_bridge_len24_32",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 7.5e-4),
            "len_choices": [24, 28, 32],
            "d_feat_choices": [12, 16],
            "expert": 3,
            "router_choices": [96, 112],
            "hidden": [0.08, 0.10],
            "attn": [0.06, 0.08],
            "wd_scales": [0.5, 1.0],
        },
        {
            "id": "ML12_h10_context_wide",
            "band": "very_aggressive_context",
            "source": "M04_h10_len25_context",
            "selection_score": "h10_context_feat24_expert4",
            "anchor": "H10",
            "lambda": (4e-4, 8e-5),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len_choices": [24, 28, 32],
            "d_feat_choices": [16, 24],
            "expert_choices": [3, 4],
            "router": 96,
            "hidden": [0.08, 0.10],
            "attn": [0.06, 0.08],
            "wd_scales": [0.5, 1.0],
            "num_heads_choices": [2, 4],
            "feature_drop_choices": [0.0, 0.02],
        },
        {
            "id": "ML13_h14_capacity_recheck",
            "band": "very_aggressive_capacity",
            "source": "M10_h14_capacity_attack",
            "selection_score": "h14_capacity_but_lower_lr",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.5e-4, 7.5e-4),
            "len_choices": [16, 20, 24],
            "d_feat_choices": [16, 20],
            "expert": 3,
            "router_choices": [112, 128],
            "hidden": [0.08, 0.10],
            "attn": [0.06, 0.08],
            "wd_scales": [0.5, 1.0],
            "cons_choices": [3e-4, 5e-4, 8e-4],
            "z_choices": [8e-5, 1e-4, 2e-4],
        },
        {
            "id": "ML14_h14_expert_wide",
            "band": "very_aggressive_capacity",
            "source": "M10_h14_capacity_attack",
            "selection_score": "h14_feat20_expert4_attack",
            "anchor": "H14",
            "lambda": (4e-4, 8e-5),
            "lr_bounds": (1.8e-4, 8.0e-4),
            "len_choices": [16, 20],
            "d_feat_choices": [16, 20, 24],
            "expert_choices": [3, 4],
            "router": 128,
            "hidden": [0.08, 0.10],
            "attn": [0.06, 0.08],
            "wd_scales": [0.5, 1.0],
            "family_drop_choices": [0.02, 0.04, 0.06],
            "feature_drop_choices": [0.0, 0.02, 0.05],
        },
        {
            "id": "ML15_h1_highlr_reframe",
            "band": "very_aggressive_sparse",
            "source": "M07_h1_highlr_validate",
            "selection_score": "h1_lowered_highlr_sparse_retry",
            "anchor": "H1",
            "lambda": (6e-4, 1.5e-4),
            "lr_bounds": (5.5e-4, 1.8e-3),
            "len_choices": [16, 20, 24],
            "d_feat_choices": [12, 16],
            "expert_choices": [2, 3],
            "router": 64,
            "hidden": [0.08, 0.10],
            "attn": [0.08],
            "wd_scales": [0.5, 1.0],
            "num_heads_choices": [2, 4],
            "feature_drop_choices": [0.0, 0.02],
        },
        {
            "id": "ML16_h3_midaux_probe",
            "band": "very_aggressive_new",
            "source": "beauty_midaux_h3_carryover",
            "selection_score": "h3_midaux_movielens_port",
            "anchor": "H3",
            "lambda": (4e-4, 8e-5),
            "lr_bounds": (2.2e-4, 9.0e-4),
            "len_choices": [20, 24, 28],
            "d_feat_choices": [12, 16],
            "expert_choices": [2, 3],
            "router_choices": [48, 64],
            "hidden": [0.10, 0.12],
            "attn": [0.08, 0.10],
            "wd_scales": [1.0, 2.0],
            "family_drop": 0.03,
            "family_drop_choices": [0.03, 0.05],
            "feature_drop_choices": [0.0, 0.02],
            "cons_choices": [4e-4, 8e-4],
            "z_choices": [8e-5, 1.6e-4],
            "num_heads_choices": [2, 4],
        },
    ]


def _select_templates(n_templates: int) -> list[Dict[str, Any]]:
    bank = _template_bank_16()
    if int(n_templates) == 16:
        return bank
    keep = {
        "ML01_h6_lr_core",
        "ML02_h6_len_scout",
        "ML03_h6_expert_up",
        "ML04_h6_feat_up",
        "ML06_h6_len_expert_combo",
        "ML09_h5_long_context",
        "ML11_h10_context_bridge",
        "ML13_h14_capacity_recheck",
    }
    out = [template for template in bank if str(template["id"]) in keep]
    if len(out) != 8:
        raise RuntimeError("template subset construction failed")
    return out


def _row(dataset: str, template: Dict[str, Any], seed_id: int, runtime_seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    anchor = str(template["anchor"])
    cfg = stage1._anchor_cfg(anchor)
    template_id = str(template["id"])
    cons_lambda, z_lambda = template["lambda"]

    family_drop_default = float(template.get("family_drop", 0.02))
    feature_drop_default = float(template.get("feature_drop", 0.0))
    overrides = stage1._build_overrides(cons_lambda, z_lambda, family_drop_default, feature_drop_default)

    run_id = f"ML_{stage1.sanitize_token(dataset, upper=True)}_{stage1.sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
    run_phase = f"{PHASE_ID}_{run_id}"

    max_item_list_length = int(template.get("len", cfg.get("MAX_ITEM_LIST_LENGTH", 20)))
    d_feat_emb = int(template.get("d_feat", cfg.get("d_feat_emb", 16)))
    expert_scale = int(template.get("expert", cfg.get("expert_scale", 3)))
    d_router_hidden = int(template.get("router", cfg.get("d_router_hidden", 64)))

    fixed_values: Dict[str, Any] = {
        "embedding_size": int(cfg["embedding_size"]),
        "d_ff": int(cfg["d_ff"]),
        "d_expert_hidden": int(cfg["d_expert_hidden"]),
        "d_router_hidden": d_router_hidden,
        "MAX_ITEM_LIST_LENGTH": max_item_list_length,
        "d_feat_emb": d_feat_emb,
        "expert_scale": expert_scale,
        "lr_scheduler_type": "warmup_cosine",
        "num_heads": 4,
    }

    train_batch_size, eval_batch_size, max_evals = stage1._template_batches(template, fixed_values, args)
    lr_low, lr_high = template["lr_bounds"]

    search_space: Dict[str, Any] = {
        "learning_rate": stage1._loguniform_spec(float(lr_low), float(lr_high)),
        "hidden_dropout_prob": stage1._choice_spec(float(value) for value in template.get("hidden", [0.10])),
        "attn_dropout_prob": stage1._choice_spec(float(value) for value in template.get("attn", [0.08])),
        "weight_decay": stage1._choice_spec(
            stage1._weight_decay_choices(anchor, list(template.get("wd_scales", [1.0])))
        ),
    }

    if "len_choices" in template:
        search_space["MAX_ITEM_LIST_LENGTH"] = stage1._choice_spec(int(value) for value in template["len_choices"])
    if "d_feat_choices" in template:
        search_space["d_feat_emb"] = stage1._choice_spec(int(value) for value in template["d_feat_choices"])
    if "expert_choices" in template:
        search_space["expert_scale"] = stage1._choice_spec(int(value) for value in template["expert_choices"])
    if "router_choices" in template:
        search_space["d_router_hidden"] = stage1._choice_spec(int(value) for value in template["router_choices"])
    if "family_drop_choices" in template:
        search_space["stage_family_dropout_prob"] = stage1._choice_spec(
            stage1._all_stage_map(float(value)) for value in template["family_drop_choices"]
        )
    if "feature_drop_choices" in template:
        search_space["stage_feature_dropout_prob"] = stage1._choice_spec(
            stage1._all_stage_map(float(value)) for value in template["feature_drop_choices"]
        )
    if "num_heads_choices" in template:
        search_space["num_heads"] = stage1._choice_spec(int(value) for value in template["num_heads_choices"])
    if "cons_choices" in template:
        search_space["route_consistency_lambda"] = stage1._choice_spec(float(value) for value in template["cons_choices"])
    if "z_choices" in template:
        search_space["z_loss_lambda"] = stage1._choice_spec(float(value) for value in template["z_choices"])

    band = str(template.get("band", "portfolio"))
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
        "stage": "movielens_portfolio",
        "tuning_stage": "movielens_portfolio",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": band,
        "capacity_anchor": anchor,
        "selected_from_stage": str(template.get("source", band)),
        "selection_score": str(template.get("selection_score", "")),
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "movielens_portfolio",
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
        return raw / "movielens_manifest.json"
    return LOG_ROOT / "movielens_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "movielens_portfolio",
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
    parser = argparse.ArgumentParser(description="FMoE_N4 MovieLens v4 session_fixed portfolio tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--template-count", type=int, choices=[8, 16], default=16)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=266000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=6144)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--max-evals", type=int, default=10)
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
    print(f"[movielens-portfolio] manifest -> {manifest}")

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