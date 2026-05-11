#!/usr/bin/env python3
"""FMoE_N4: Stage3 seen-first A12 tuning with extended budgets.

Stage3 assumes promotion and reporting should be driven by seen-target metrics.
The launcher inherits the seen-only main-eval behavior from Stage1's command
builder and expands the search around the most plausible Kuai A12 regions.

Compared with Stage2, Stage3 adds a few controlled new axes:
- route_consistency_lambda
- z_loss_lambda
- stage_family_dropout_prob
- stage_feature_dropout_prob
- lr_scheduler_min_lr_ratio
- local structural choices for d_feat_emb / expert_scale / MAX_ITEM_LIST_LENGTH
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import stage1_a12_broad_templates as stage1

TRACK = stage1.TRACK
AXIS = "Stage3_A12_SeenFocus"
AXIS_ID = "N4S3A12"
AXIS_DESC = "stage3_a12_seen_focus"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "P4S3"
PHASE_NAME = "FMOE_N4_STAGE3_A12_SEEN_FOCUS"
DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2"]

REPO_ROOT_REAL = stage1.REPO_ROOT_REAL
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

# Reuse the Stage1 command/log machinery, but point metadata at Stage3.
stage1.AXIS = AXIS
stage1.AXIS_ID = AXIS_ID
stage1.AXIS_DESC = AXIS_DESC
stage1.PHASE_ID = PHASE_ID
stage1.PHASE_NAME = PHASE_NAME
stage1.LOG_ROOT = LOG_ROOT


def _stage_choice(values: list[float]) -> Dict[str, Any]:
    return stage1._choice_spec(stage1._all_stage_map(float(value)) for value in values)


def _template_bank_16() -> list[Dict[str, Any]]:
    return [
        {
            "id": "S01_h14_seen_lo",
            "band": "exploit",
            "source": "E07_h14_lo_test+T05_capacity_h14_lo",
            "selection_score": "best seen-test h14 low-lr line",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 5.8e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([16, 20]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4]),
            },
        },
        {
            "id": "S02_h14_seen_hi",
            "band": "exploit",
            "source": "E01_h14hi_e4+T06_capacity_h14_hi",
            "selection_score": "best seen-valid h14 high-capacity line",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 8.8e-4),
            "len": 20,
            "d_feat": 20,
            "expert": 4,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
                "expert_scale": stage1._choice_spec([3, 4]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4]),
            },
        },
        {
            "id": "S03_h14_expert5",
            "band": "exploit",
            "source": "X14_h14_expert5",
            "selection_score": "seen-test strong expert5 variant",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.0e-4, 9.0e-4),
            "len": 20,
            "d_feat": 20,
            "expert": 5,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.06, 0.08],
            "extra_search": {
                "expert_scale": stage1._choice_spec([4, 5]),
                "d_feat_emb": stage1._choice_spec([16, 20, 24]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
            },
        },
        {
            "id": "S04_h7_feat32_core",
            "band": "exploit",
            "source": "X12_h7_feat32",
            "selection_score": "best seen-valid stage2 family",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 8.5e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([24, 32]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03]),
            },
        },
        {
            "id": "S05_h7_feat24_core",
            "band": "exploit",
            "source": "E04_h7_feat24+T09_feat24_h7",
            "selection_score": "stable seen-test feature line",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 7.8e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([16, 24, 32]),
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03]),
            },
        },
        {
            "id": "S06_h6_e2_core",
            "band": "exploit",
            "source": "E02_h6_e2_core+T10_expert2_h6",
            "selection_score": "best compact seen-valid non-h14 family",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 8.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0, 4.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "expert_scale": stage1._choice_spec([1, 2]),
                "weight_decay": stage1._choice_spec(stage1._weight_decay_choices("H6", [0.5, 1.0, 2.0, 4.0])),
            },
        },
        {
            "id": "S07_h10_len25_f24",
            "band": "exploit",
            "source": "X13_h10_len25_f24",
            "selection_score": "best context+feature stage2 family",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 25,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30]),
                "d_feat_emb": stage1._choice_spec([16, 24]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4]),
            },
        },
        {
            "id": "S08_h5_len35_f12",
            "band": "exploit",
            "source": "X16_h5_len35_f12+T11_steady_h5",
            "selection_score": "strong seen-test long compact family",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.6e-4, 6.5e-4),
            "len": 35,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([30, 35]),
                "d_feat_emb": stage1._choice_spec([12, 16]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03, 0.05]),
            },
        },
        {
            "id": "S09_h11_e2_fast",
            "band": "exploit",
            "source": "E03_h11_e2_fast+T16_fastwide_h11",
            "selection_score": "stable h11 compact-fast family",
            "anchor": "H11",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (3.0e-4, 1.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "expert_scale": stage1._choice_spec([2, 3]),
                "route_consistency_lambda": stage1._choice_spec([2.5e-4, 5e-4, 1e-3]),
                "z_loss_lambda": stage1._choice_spec([1e-4, 2e-4]),
            },
        },
        {
            "id": "S10_h2_regularized_transfer",
            "band": "transfer",
            "source": "T03_regularized_h2",
            "selection_score": "best overall/unseen transfer family",
            "anchor": "H2",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 7.0e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([5e-4, 8e-4, 1e-3]),
                "z_loss_lambda": stage1._choice_spec([1e-4, 2e-4]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04]),
            },
        },
        {
            "id": "S11_h10_longctx_transfer",
            "band": "transfer",
            "source": "T04_longctx_h10+T13_lowlr_h10",
            "selection_score": "strong overall/unseen long-context family",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.6e-4, 6.0e-4),
            "len": 30,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([25, 30]),
                "d_feat_emb": stage1._choice_spec([16, 24]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10]),
            },
        },
        {
            "id": "S12_h5_steady_transfer",
            "band": "transfer",
            "source": "T11_steady_h5",
            "selection_score": "balanced overall/unseen transfer family",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.6e-4, 5.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "MAX_ITEM_LIST_LENGTH": stage1._choice_spec([20, 25]),
                "d_feat_emb": stage1._choice_spec([12, 16, 24]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03]),
            },
        },
        {
            "id": "N13_dropout_sched_probe",
            "band": "new_axis",
            "source": "dropout+scheduler wide probe",
            "selection_score": "keep one broad generalization probe",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 9.5e-4),
            "len": 20,
            "d_feat": 20,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
            "extra_search": {
                "stage_family_dropout_prob": _stage_choice([0.0, 0.02, 0.04]),
                "stage_feature_dropout_prob": _stage_choice([0.0, 0.03, 0.05]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10, 0.20]),
            },
        },
        {
            "id": "N14_aux_wide_probe",
            "band": "new_axis",
            "source": "aux wide probe",
            "selection_score": "leave aux terms somewhat wide",
            "anchor": "H11",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 9.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
            "extra_search": {
                "route_consistency_lambda": stage1._choice_spec([0.0, 2.5e-4, 5e-4, 1e-3]),
                "z_loss_lambda": stage1._choice_spec([5e-5, 1e-4, 2e-4, 4e-4]),
                "stage_family_dropout_prob": _stage_choice([0.0, 0.02]),
            },
        },
        {
            "id": "N15_h4_tiny_wide",
            "band": "new_axis",
            "source": "X09_h4_tiny",
            "selection_score": "compact tiny family worth one more wide check",
            "anchor": "H4",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.8e-4, 6.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12]),
                "expert_scale": stage1._choice_spec([2, 3]),
                "stage_family_dropout_prob": _stage_choice([0.02, 0.04, 0.06]),
            },
        },
        {
            "id": "N16_h9_ultracompact_wide",
            "band": "new_axis",
            "source": "X10_h9_ultracompact",
            "selection_score": "compact h9 family with room to recover",
            "anchor": "H9",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.5e-4, 6.0e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
            "extra_search": {
                "d_feat_emb": stage1._choice_spec([8, 12, 16]),
                "expert_scale": stage1._choice_spec([1, 2, 3]),
                "lr_scheduler_min_lr_ratio": stage1._choice_spec([0.05, 0.10, 0.20]),
            },
        },
    ]


def _select_templates(n_templates: int) -> list[Dict[str, Any]]:
    bank = _template_bank_16()
    if int(n_templates) == 16:
        return bank
    keep = {
        "S01_h14_seen_lo",
        "S02_h14_seen_hi",
        "S04_h7_feat32_core",
        "S06_h6_e2_core",
        "S07_h10_len25_f24",
        "S10_h2_regularized_transfer",
        "N13_dropout_sched_probe",
        "N14_aux_wide_probe",
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

    family_drop_default = float(template.get("family_drop", 0.02 if str(template.get("hidden_mode", "balanced")) != "high" else 0.04))
    feature_drop_default = float(template.get("feature_drop", 0.0))
    overrides = stage1._build_overrides(cons_lambda, z_lambda, family_drop_default, feature_drop_default)

    hidden_mode = str(template.get("hidden_mode", "balanced"))
    run_id = f"S3_{stage1.sanitize_token(dataset, upper=True)}_{stage1.sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
    run_phase = f"{PHASE_ID}_{run_id}"

    max_item_list_length = int(template["len"]) if "len" in template else int(cfg.get("MAX_ITEM_LIST_LENGTH", 20))
    d_feat_emb = int(template["d_feat"]) if "d_feat" in template else int(cfg.get("d_feat_emb", 16))
    expert_scale = int(template["expert"]) if "expert" in template else int(cfg.get("expert_scale", 3))

    fixed_values: Dict[str, Any] = {
        "embedding_size": int(cfg["embedding_size"]),
        "d_ff": int(cfg["d_ff"]),
        "d_expert_hidden": int(cfg["d_expert_hidden"]),
        "d_router_hidden": int(cfg["d_router_hidden"]),
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
        "hidden_dropout_prob": stage1._choice_spec(stage1._hidden_choices(anchor, hidden_mode)),
        "attn_dropout_prob": stage1._choice_spec(float(v) for v in template["attn"]),
        "weight_decay": stage1._choice_spec(
            stage1._weight_decay_choices(anchor, list(template.get("wd_scales", [0.5, 1.0, 2.0])))
        ),
    }
    search_space.update(dict(template.get("extra_search", {}) or {}))

    band = str(template.get("band", "seen_focus"))
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
        "stage": "stage3",
        "tuning_stage": "stage3",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": band,
        "capacity_anchor": anchor,
        "selected_from_stage": str(template.get("source", band)),
        "selection_score": str(template.get("selection_score", "")),
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "stage3",
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
        return raw / "stage3_manifest.json"
    return LOG_ROOT / "stage3_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "stage3",
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
    parser = argparse.ArgumentParser(description="FMoE_N4 Stage3 seen-first A12 tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="CSV datasets (default: KuaiRec)")
    parser.add_argument("--template-count", type=int, choices=[8, 16], default=16)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=262000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
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
    print(f"[stage3] manifest -> {manifest}")

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