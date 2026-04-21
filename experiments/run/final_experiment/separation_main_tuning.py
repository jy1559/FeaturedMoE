#!/usr/bin/env python3
"""Separation-centric RouteRec tuning for KuaiRec and Foursquare.

This runner fixes the routing topology to the paper's hierarchical sparse setup,
uses 4 anchor configs per dataset, disables unrelated auxiliary losses, and
searches only LR / weight decay / consistency / separation.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REAL_FINAL_DIR = CODE_DIR / "real_final_ablation"
if str(REAL_FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_FINAL_DIR))

from common import (  # noqa: E402
    BaseCandidate,
    QUESTION_AXIS,
    build_route_row,
    common_arg_parser,
    load_base_candidates,
    parse_csv_ints,
    parse_csv_list,
    run_jobs,
    validate_session_fixed_files,
    write_manifest,
    canonical_stage_maps,
)


QUESTION = "sep_main"
QUESTION_AXIS[QUESTION] = "separation_main_tuning"

DEFAULT_DATASETS = ["KuaiRecLargeStrictPosV2_0.2", "foursquare"]

LR_SCALE_LO = 0.65
LR_SCALE_HI = 1.80
WD_LO = 1e-6
WD_HI = 8e-4
CONSISTENCY_LO = 7e-5
CONSISTENCY_HI = 8e-4
SEPARATION_LO = 1.2e-3
SEPARATION_HI = 3e-2

ZERO_AUX_KEYS = (
    "balance_loss_lambda",
    "z_loss_lambda",
    "gate_entropy_lambda",
    "gate_entropy_until",
    "rule_agreement_lambda",
    "group_coverage_lambda",
    "group_prior_align_lambda",
    "feature_group_bias_lambda",
    "factored_group_balance_lambda",
    "route_smoothness_lambda",
    "route_sharpness_lambda",
    "route_monopoly_lambda",
    "route_monopoly_tau",
    "route_prior_lambda",
    "fmoe_v2_stage_merge_aux_lambda_scale",
)

ANCHOR_SOURCE_RANKS = (1, 2)

MANUAL_ANCHOR_OVERRIDES = {
    "KuaiRecLargeStrictPosV2_0.2": {
        "general": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 256,
            "embedding_size": 256,
            "inner_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 128,
            "d_expert_hidden": 256,
            "expert_scale": 4,
            "hidden_dropout_prob": 0.10,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.0,
            "mid_router_feature_dropout": 0.0,
            "micro_router_feature_dropout": 0.0,
        },
        "challenge": {
            "MAX_ITEM_LIST_LENGTH": 30,
            "hidden_size": 320,
            "embedding_size": 320,
            "inner_size": 384,
            "num_layers": 3,
            "num_heads": 4,
            "d_feat_emb": 24,
            "d_router_hidden": 160,
            "d_expert_hidden": 320,
            "expert_scale": 6,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.12,
            "stage_feature_dropout_prob": 0.05,
            "mid_router_feature_dropout": 0.05,
            "micro_router_feature_dropout": 0.05,
        },
    },
    "foursquare": {
        "general": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 256,
            "embedding_size": 256,
            "inner_size": 256,
            "num_layers": 2,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 128,
            "d_expert_hidden": 256,
            "expert_scale": 4,
            "hidden_dropout_prob": 0.10,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.0,
            "mid_router_feature_dropout": 0.0,
            "micro_router_feature_dropout": 0.0,
        },
        "challenge": {
            "MAX_ITEM_LIST_LENGTH": 20,
            "hidden_size": 288,
            "embedding_size": 288,
            "inner_size": 384,
            "num_layers": 3,
            "num_heads": 4,
            "d_feat_emb": 16,
            "d_router_hidden": 160,
            "d_expert_hidden": 320,
            "expert_scale": 5,
            "hidden_dropout_prob": 0.12,
            "attn_dropout_prob": 0.10,
            "stage_feature_dropout_prob": 0.05,
            "mid_router_feature_dropout": 0.05,
            "micro_router_feature_dropout": 0.05,
        },
    },
}


def separation_settings() -> list[dict]:
    full = canonical_stage_maps()
    return [{
        "setting_key": "separation_main",
        "setting_label": "Separation Main Tuning",
        "variant_label": "sep_main",
        "variant_group": "separation_main",
        "panel_family": "route_separation",
        "variant_order": 0,
        "overrides": {
            **deepcopy(full),
            "topk_scope_mode": "per_group",
            "group_top_k": 3,
            "expert_top_k": 2,
            "moe_top_k": 0,
            "fmoe_v2_stage_merge_aux_enable": False,
        },
    }]


def _lr_spec(base_lr: float) -> tuple[list[float], str]:
    lr = float(base_lr if base_lr and base_lr > 0 else 1e-3)
    lo = max(lr * LR_SCALE_LO, 1e-6)
    hi = max(lr * LR_SCALE_HI, lo * 1.05)
    return [lo, hi], "loguniform"


def _weight_decay_spec(base_wd: float) -> tuple[list[float], str]:
    _ = float(base_wd if base_wd and base_wd > 0 else 0.0)
    return [WD_LO, WD_HI], "loguniform_zero"


def _clone_candidate(
    source: BaseCandidate,
    *,
    rank: int,
    tag_suffix: str,
    notes_suffix: str,
    overrides: dict,
) -> BaseCandidate:
    base_config = deepcopy(source.base_config)
    base_config.update(deepcopy(overrides))
    return BaseCandidate(
        dataset=source.dataset,
        model=source.model,
        rank=rank,
        tag=f"{source.tag}_{tag_suffix}",
        notes=f"{source.notes}; {notes_suffix}",
        result_json=source.result_json,
        payload=source.payload,
        base_config=base_config,
        checkpoint_file=source.checkpoint_file,
    )


def _build_anchor_candidates(args) -> list[BaseCandidate]:
    datasets = parse_csv_list(args.datasets) or list(DEFAULT_DATASETS)
    models = parse_csv_list(args.models) or ["featured_moe_n3"]
    base_csv = Path(args.base_csv).expanduser().resolve()
    for dataset in datasets:
        validate_session_fixed_files(dataset)

    loaded = load_base_candidates(
        base_csv,
        datasets=datasets,
        models=models,
        top_k_configs=8,
    )
    grouped: dict[str, dict[int, BaseCandidate]] = {}
    for candidate in loaded:
        grouped.setdefault(candidate.dataset, {})[candidate.rank] = candidate

    selected: list[BaseCandidate] = []
    for dataset in datasets:
        dataset_candidates = grouped.get(dataset, {})
        missing = [rank for rank in ANCHOR_SOURCE_RANKS if rank not in dataset_candidates]
        if missing:
            raise RuntimeError(f"Missing anchor source ranks {missing} for dataset={dataset}")

        best_anchor = dataset_candidates[ANCHOR_SOURCE_RANKS[0]]
        selected.extend(dataset_candidates[rank] for rank in ANCHOR_SOURCE_RANKS)

        manual = MANUAL_ANCHOR_OVERRIDES.get(dataset)
        if not manual:
            raise RuntimeError(f"Missing manual anchor overrides for dataset={dataset}")
        selected.append(
            _clone_candidate(
                best_anchor,
                rank=91,
                tag_suffix="general_anchor",
                notes_suffix="manual general anchor",
                overrides=manual["general"],
            )
        )
        selected.append(
            _clone_candidate(
                best_anchor,
                rank=92,
                tag_suffix="challenge_anchor",
                notes_suffix="manual challenge anchor",
                overrides=manual["challenge"],
            )
        )
    return selected


def build_rows(args) -> list[dict]:
    candidates = _build_anchor_candidates(args)
    settings = separation_settings()
    seeds = parse_csv_ints(args.seeds) or [1]
    rows: list[dict] = []
    cursor = 0

    for candidate in candidates:
        for setting in settings:
            for seed in seeds:
                cursor += 1
                row = build_route_row(
                    question=QUESTION,
                    candidate=candidate,
                    setting=setting,
                    seed=seed,
                    runtime_seed=980000 + cursor,
                    max_evals=args.max_evals,
                    max_run_hours=args.max_run_hours,
                    tune_epochs=args.tune_epochs,
                    tune_patience=args.tune_patience,
                    lr_mode="fixed",
                )

                fixed_context = row["fixed_context"]
                for key in ZERO_AUX_KEYS:
                    fixed_context[key] = 0.0
                fixed_context["fmoe_v2_stage_merge_aux_enable"] = False

                fixed_context.pop("learning_rate", None)
                fixed_context.pop("weight_decay", None)
                fixed_context.pop("route_consistency_lambda", None)
                fixed_context.pop("route_separation_lambda", None)

                base_lr = float(candidate.base_config.get("learning_rate") or 1e-3)
                base_wd = float(candidate.base_config.get("weight_decay") or 0.0)
                lr_vals, lr_type = _lr_spec(base_lr)
                wd_vals, wd_type = _weight_decay_spec(base_wd)

                row["search_space"] = {
                    "learning_rate": lr_vals,
                    "weight_decay": wd_vals,
                    "route_consistency_lambda": [CONSISTENCY_LO, CONSISTENCY_HI],
                    "route_separation_lambda": [SEPARATION_LO, SEPARATION_HI],
                }
                row["search_space_types"] = {
                    "learning_rate": lr_type,
                    "weight_decay": wd_type,
                    "route_consistency_lambda": "loguniform",
                    "route_separation_lambda": "loguniform",
                }
                rows.append(row)

    if bool(args.smoke_test):
        rows = rows[: max(1, int(args.smoke_max_runs))]
    return rows


def parse_args():
    parser = common_arg_parser("Separation-centric tuning for KuaiRec/Foursquare", question=QUESTION)
    parser.set_defaults(
        datasets=",".join(DEFAULT_DATASETS),
        models="featured_moe_n3",
        top_k_configs=4,
        max_evals=30,
        max_run_hours=3.0,
        tune_epochs=100,
        tune_patience=10,
        lr_mode="fixed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    manifest = write_manifest(QUESTION, rows)
    print(f"[{QUESTION}] manifest -> {manifest} ({len(rows)} jobs)", flush=True)
    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")
    return run_jobs(
        rows,
        question=QUESTION,
        gpus=gpus,
        search_algo=args.search_algo,
        resume_from_logs=bool(args.resume_from_logs),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())