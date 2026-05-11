#!/usr/bin/env python3
"""Launch FeaturedMoE_N3 final A8~A12 wrapper sweep runs (session_fixed)."""

from __future__ import annotations

import argparse
from typing import Any, Dict

import run_final_all_datasets as base

_BASE_SELECTED_HPARAMS_FOR_DATASET = base._selected_hparams_for_dataset


ARCH_ORDER = ("A8", "A10", "A11", "A12")
ARCH_METADATA: Dict[str, Dict[str, str]] = {
    "A8": {
        "arch_key": "A8_ATTN_MICRO_BEFORE_NO_BIAS",
        "arch_name": "A8_ATTN_MICRO_BEFORE_NO_BIAS",
        "setting_group": "final_wrapper_sweep",
        "setting_key": "A8_ATTN_MICRO_BEFORE_NO_BIAS",
        "setting_desc_prefix": "A8_ATTN_MICRO_BEFORE_NO_BIAS",
        "setting_detail": "ATTN_MICRO_BEFORE layout + strict NN + z-loss + no_bias + mixed_2 wrapper",
    },
    "A10": {
        "arch_key": "A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4",
        "arch_name": "A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4",
        "setting_group": "final_wrapper_sweep",
        "setting_key": "A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4",
        "setting_desc_prefix": "A10_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W4",
        "setting_detail": "A8 base with unified all_w4 wrapper (macro/mid/micro all w4_bxd)",
    },
    "A11": {
        "arch_key": "A11_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W2",
        "arch_name": "A11_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W2",
        "setting_group": "final_wrapper_sweep",
        "setting_key": "A11_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W2",
        "setting_desc_prefix": "A11_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W2",
        "setting_detail": "A8 base with unified all_w2 wrapper (macro/mid/micro all w2_a_plus_d)",
    },
    "A12": {
        "arch_key": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
        "arch_name": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
        "setting_group": "final_wrapper_sweep",
        "setting_key": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
        "setting_desc_prefix": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
        "setting_detail": "A8 base with unified all_w5 wrapper (macro/mid/micro all w5_exd)",
    },
}

DATASET_HPARAM_PRESET_4: Dict[str, list[str]] = {
    "KuaiRecLargeStrictPosV2_0.2": ["H14", "H3", "H2", "H7"],
    "lastfm0.03": ["H3", "H11", "H7", "H15"],
    "amazon_beauty": ["H14", "H10", "H13", "H3"],
    "foursquare": ["H15", "H5", "H1", "H3"],
    # Favor one proven A1 hparam (H5) while keeping the strongest wrapper
    # candidates already seen on movielens (H8/H9) and the top A1 config (H3).
    "movielens1m": ["H3", "H5", "H8", "H9"],
    # Keep retail to a core-4 compare set so the remaining A sweep can finish
    # in a single overnight window while preserving like-for-like A comparison.
    "retail_rocket": ["H2", "H3", "H1", "H6"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N3 Final A8/A10~A12 wrapper sweep launcher")
    parser.add_argument("--datasets", default=",".join(base.DEFAULT_DATASETS))
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--seed-base", type=int, default=97000)

    parser.add_argument("--architectures", default="A8,A10,A11,A12", help="CSV from {A8,A10,A11,A12}")
    parser.add_argument("--architecture", default="", help="Alias of --architectures")

    parser.add_argument("--common-hparams", default="AUTO4")
    parser.add_argument("--default-outlier-hparam", default="H4")
    parser.add_argument(
        "--dataset-outlier-hparams",
        default="",
        help="CSV map: dataset:Hid,dataset2:Hid ... (used only when --common-hparams is not AUTO4)",
    )

    parser.add_argument("--max-evals", type=int, default=20)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)

    parser.add_argument("--feature-group-bias-lambda", type=float, default=0.05)
    parser.add_argument("--rule-bias-scale", type=float, default=0.1)
    parser.add_argument("--z-loss-lambda", type=float, default=1e-4)
    parser.add_argument("--balance-loss-lambda", type=float, default=0.0)
    parser.add_argument("--macro-history-window", type=int, default=5)
    parser.add_argument("--family-dropout-prob", type=float, default=0.10)
    parser.add_argument("--feature-dropout-prob", type=float, default=0.0)

    parser.add_argument("--a2-route-consistency-lambda", type=float, default=8e-4)
    parser.add_argument("--a2-route-consistency-min-sim", type=float, default=0.995)
    parser.add_argument("--a2-z-loss-lambda", type=float, default=2e-4)
    parser.add_argument("--a4-intra-group-bias-scale", type=float, default=0.12)

    parser.add_argument("--max-item-list-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--dataset-batch-sizes",
        default="movielens1m:8192,retail_rocket:8192",
        help="CSV map: dataset:int,dataset2:int overriding train_batch_size",
    )
    parser.add_argument(
        "--dataset-eval-batch-sizes",
        default="movielens1m:12288,retail_rocket:12288",
        help="CSV map: dataset:int,dataset2:int overriding eval_batch_size",
    )
    parser.add_argument(
        "--dataset-max-evals",
        default="",
        help="Optional CSV map for per-dataset max_evals overrides",
    )
    parser.add_argument(
        "--dataset-tune-epochs",
        default="",
        help="Optional CSV map for per-dataset tune_epochs overrides",
    )
    parser.add_argument(
        "--dataset-tune-patience",
        default="",
        help="Optional CSV map for per-dataset tune_patience overrides",
    )
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-dropout-prob", type=float, default=0.1)
    parser.add_argument("--d-feat-emb", type=int, default=16)
    parser.add_argument("--expert-scale", type=int, default=3)
    parser.add_argument("--search-lr-min", type=float, default=1.5e-4)
    parser.add_argument("--search-lr-max", type=float, default=8e-3)
    parser.add_argument("--search-lr-scheduler", default="warmup_cosine")

    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")

    parser.add_argument("--migrate-existing-layout", dest="migrate_existing_layout", action="store_true")
    parser.add_argument("--no-migrate-existing-layout", dest="migrate_existing_layout", action="store_false")
    parser.set_defaults(migrate_existing_layout=True)

    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    return parser.parse_args()


def _selected_hparams_for_dataset(dataset: str, args: argparse.Namespace) -> list[str]:
    common = [h.upper() for h in base._parse_csv_strings(args.common_hparams)]
    if common == ["AUTO4"]:
        preset = list(DATASET_HPARAM_PRESET_4.get(dataset, []))
        selected = [h for h in preset if h in base.HPARAM_BANK]
        if selected:
            return selected
    return _BASE_SELECTED_HPARAMS_FOR_DATASET(dataset, args)


def _attn_micro_before_core(args: argparse.Namespace, wrapper_map: Dict[str, str], *, bias_mode: str) -> Dict[str, Any]:
    overrides = base._core_overrides(
        args,
        wrapper_map=wrapper_map,
        source_profile="src_abc_feature",
        bias_mode="bias_both",
    )
    base._apply_a2_aux_profile(overrides, args)
    overrides["layer_layout"] = ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"]
    overrides["stage_router_granularity"] = {"macro": "session", "mid": "session", "micro": "token"}
    overrides["stage_family_dropout_prob"] = base._all_stage_map(float(args.family_dropout_prob))
    overrides["stage_feature_dropout_prob"] = base._all_stage_map(float(args.feature_dropout_prob))
    overrides["stage_feature_dropout_scope"] = base._all_stage_map("token")
    if bias_mode == "none":
        overrides["bias_mode"] = "none"
        overrides["rule_bias_scale"] = 0.0
        overrides["feature_group_bias_lambda"] = 0.0
    return overrides


def _arch_overrides(arch_id: str, args: argparse.Namespace) -> Dict[str, Any]:
    arch = str(arch_id).upper().strip()
    if arch == "A8":
        return _attn_micro_before_core(
            args,
            {"macro": "w4_bxd", "mid": "w6_bxd_plus_a", "micro": "w1_flat"},
            bias_mode="none",
        )
    if arch == "A10":
        return _attn_micro_before_core(
            args,
            {"macro": "w4_bxd", "mid": "w4_bxd", "micro": "w4_bxd"},
            bias_mode="none",
        )
    if arch == "A11":
        return _attn_micro_before_core(
            args,
            {"macro": "w2_a_plus_d", "mid": "w2_a_plus_d", "micro": "w2_a_plus_d"},
            bias_mode="none",
        )
    if arch == "A12":
        return _attn_micro_before_core(
            args,
            {"macro": "w5_exd", "mid": "w5_exd", "micro": "w5_exd"},
            bias_mode="none",
        )
    raise ValueError(f"Unsupported architecture: {arch_id}")


def main() -> int:
    base.ARCH_ORDER = ARCH_ORDER
    base.ARCH_METADATA = ARCH_METADATA
    base.parse_args = _parse_args
    base._selected_hparams_for_dataset = _selected_hparams_for_dataset
    base._arch_overrides = _arch_overrides
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
