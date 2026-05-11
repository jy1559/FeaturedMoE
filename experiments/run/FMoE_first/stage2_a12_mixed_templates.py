#!/usr/bin/env python3
"""FMoE_N4: Stage2 mixed A12 tuning with 8 exploit + 8 explore templates.

Stage2 reuses the strongest Stage1 Kuai regions and drops the weaker repeats.
Half of the bank refines the best-performing Stage1 templates with narrower,
winner-centered ranges. The other half probes fresh capacity directions such as
much smaller anchors, larger feature embeddings, altered expert counts, and
longer contexts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import stage1_a12_broad_templates as stage1

TRACK = stage1.TRACK
AXIS = "Stage2_A12_MixedTemplates"
AXIS_ID = "N4S2A12"
AXIS_DESC = "stage2_a12_mixed_templates"
ARCH_ID = stage1.ARCH_ID
ARCH_KEY = stage1.ARCH_KEY
ARCH_NAME = stage1.ARCH_NAME
PHASE_ID = "P4S2"
PHASE_NAME = "FMOE_N4_STAGE2_A12_MIXED"
DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2"]

REPO_ROOT_REAL = stage1.REPO_ROOT_REAL
LOG_ROOT = REPO_ROOT_REAL / "experiments" / "run" / "artifacts" / "logs" / TRACK / AXIS

# Reuse the Stage1 implementation machinery, but point command/log metadata at Stage2.
stage1.AXIS = AXIS
stage1.AXIS_ID = AXIS_ID
stage1.AXIS_DESC = AXIS_DESC
stage1.PHASE_ID = PHASE_ID
stage1.PHASE_NAME = PHASE_NAME
stage1.LOG_ROOT = LOG_ROOT


def _template_bank_16() -> list[Dict[str, Any]]:
    return [
        {
            "id": "E01_h14hi_e4",
            "band": "exploit",
            "source": "T06_capacity_h14_hi",
            "selection_score": "s1_valid=0.0147,test=0.1117",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (5.0e-4, 1.4e-3),
            "len": 20,
            "d_feat": 20,
            "expert": 4,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.08, 0.10],
        },
        {
            "id": "E02_h6_e2_core",
            "band": "exploit",
            "source": "T10_expert2_h6",
            "selection_score": "s1_valid=0.0147,test=0.1114",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (5.0e-4, 1.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "E03_h11_e2_fast",
            "band": "exploit",
            "source": "T16_fastwide_h11",
            "selection_score": "s1_valid=0.0143,test=0.1114",
            "anchor": "H11",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (5.0e-4, 1.5e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 2,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "E04_h7_feat24",
            "band": "exploit",
            "source": "T09_feat24_h7",
            "selection_score": "s1_valid=0.0141,test=0.1116",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.5e-4, 9.5e-4),
            "len": 20,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "E05_h7_lrhi",
            "band": "exploit",
            "source": "T03_lrhi_h7",
            "selection_score": "s1_valid=0.0140,test=0.1122",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (5.0e-4, 1.2e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "E06_h10_len30",
            "band": "exploit",
            "source": "T12_len30_h10",
            "selection_score": "s1_valid=0.0140,test=0.1112",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (3.5e-4, 9.5e-4),
            "len": 30,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "E07_h14_lo_test",
            "band": "exploit",
            "source": "T05_capacity_h14_lo",
            "selection_score": "s1_valid=0.0137,test=0.1132",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.2e-4, 5.5e-4),
            "len": 20,
            "d_feat": 16,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "E08_h3_compact",
            "band": "exploit",
            "source": "T14_compact_h3",
            "selection_score": "s1_valid=0.0144,test=0.1072",
            "anchor": "H3",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.2e-4, 7.5e-4),
            "len": 20,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [2.0, 4.0, 8.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
        },
        {
            "id": "X09_h4_tiny",
            "band": "explore",
            "source": "new_compact_anchor",
            "selection_score": "new_anchor_H4",
            "anchor": "H4",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (2.0e-4, 7.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.10, 0.12],
        },
        {
            "id": "X10_h9_ultracompact",
            "band": "explore",
            "source": "new_compact_anchor",
            "selection_score": "new_anchor_H9",
            "anchor": "H9",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.5e-4, 5.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.14],
        },
        {
            "id": "X11_h12_microtiny",
            "band": "explore",
            "source": "new_compact_anchor",
            "selection_score": "new_anchor_H12",
            "anchor": "H12",
            "lambda": (8e-4, 2e-4),
            "lr_bounds": (1.5e-4, 4.5e-4),
            "len": 20,
            "d_feat": 8,
            "expert": 2,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "high",
            "attn": [0.12, 0.14],
        },
        {
            "id": "X12_h7_feat32",
            "band": "explore",
            "source": "featdim_up",
            "selection_score": "d_feat=32",
            "anchor": "H7",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.8e-4, 9.0e-4),
            "len": 20,
            "d_feat": 32,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "X13_h10_len25_f24",
            "band": "explore",
            "source": "context_feat_mix",
            "selection_score": "len25+d_feat24",
            "anchor": "H10",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (2.5e-4, 8.5e-4),
            "len": 25,
            "d_feat": 24,
            "expert": 3,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "X14_h14_expert5",
            "band": "explore",
            "source": "expertcount_up",
            "selection_score": "expert=5",
            "anchor": "H14",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.0e-3),
            "len": 20,
            "d_feat": 20,
            "expert": 5,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
        {
            "id": "X15_h6_expert1",
            "band": "explore",
            "source": "expertcount_down",
            "selection_score": "expert=1",
            "anchor": "H6",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (4.0e-4, 1.0e-3),
            "len": 20,
            "d_feat": 16,
            "expert": 1,
            "wd_scales": [0.5, 1.0, 2.0],
            "hidden_mode": "balanced",
            "attn": [0.08, 0.10],
        },
        {
            "id": "X16_h5_len35_f12",
            "band": "explore",
            "source": "longctx_compact",
            "selection_score": "len35+d_feat12",
            "anchor": "H5",
            "lambda": (5e-4, 1e-4),
            "lr_bounds": (1.8e-4, 6.5e-4),
            "len": 35,
            "d_feat": 12,
            "expert": 3,
            "wd_scales": [1.0, 2.0, 4.0],
            "hidden_mode": "low",
            "attn": [0.06, 0.08],
        },
    ]


def _select_templates(n_templates: int) -> list[Dict[str, Any]]:
    bank = _template_bank_16()
    if int(n_templates) == 16:
        return bank
    keep = {
        "E01_h14hi_e4",
        "E03_h11_e2_fast",
        "E04_h7_feat24",
        "E06_h10_len30",
        "X09_h4_tiny",
        "X10_h9_ultracompact",
        "X12_h7_feat32",
        "X14_h14_expert5",
    }
    out = [t for t in bank if str(t["id"]) in keep]
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
    run_id = f"S2_{stage1.sanitize_token(dataset, upper=True)}_{stage1.sanitize_token(template_id, upper=True)}_S{int(seed_id)}"
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

    band = str(template.get("band", "mixed"))
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
        "stage": "stage2",
        "tuning_stage": "stage2",
        "family_id": template_id,
        "family_group": "template",
        "variant_id": band,
        "capacity_anchor": anchor,
        "selected_from_stage": str(template.get("source", band)),
        "selection_score": str(template.get("selection_score", "")),
        "search_algo": str(args.search_algo),
        "seed_id": int(seed_id),
        "runtime_seed": int(runtime_seed),
        "stage_group": "stage2",
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
        return raw / "stage2_manifest.json"
    return LOG_ROOT / "stage2_manifest.json"


def write_manifest(args: argparse.Namespace, rows: list[Dict[str, Any]]) -> Path:
    path = _manifest_path(args)
    payload = {
        "track": TRACK,
        "axis": AXIS,
        "stage": "stage2",
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
    parser = argparse.ArgumentParser(description="FMoE_N4 Stage2 mixed A12 template tuning")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="CSV datasets (default: KuaiRec)")
    parser.add_argument("--template-count", type=int, choices=[8, 16], default=16)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=261000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--max-evals", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=60)
    parser.add_argument("--tune-patience", type=int, default=6)
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
    print(f"[stage2] manifest -> {manifest}")

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