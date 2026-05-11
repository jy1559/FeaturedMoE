#!/usr/bin/env python3
"""AI502 transfer learning 실험 manifest 생성 및 실행 launcher."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from copy import copy
from collections import Counter, deque
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
RUN_DIR = THIS_DIR.parent
EXP_DIR = RUN_DIR.parent
REPO_ROOT = EXP_DIR.parent
DATA_ROOT = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"
HYPEROPT = EXP_DIR / "hyperopt_tune.py"
DEFAULT_FMOE_PYTHON = Path("/venv/FMoE/bin/python")
PYTHON_BIN = Path(
    os.environ.get("RUN_PYTHON_BIN")
    or os.environ.get("PYTHON_BIN")
    or (str(DEFAULT_FMOE_PYTHON) if DEFAULT_FMOE_PYTHON.exists() else sys.executable)
)

ARTIFACT_DIR = THIS_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
LOG_DIR = ARTIFACT_DIR / "logs"
MANIFEST_DIR = ARTIFACT_DIR / "manifests"
SUMMARY_DIR = ARTIFACT_DIR / "summaries"
RESULTS_DIR = ARTIFACT_DIR / "hyperopt_results"
LOGGING_DIR = ARTIFACT_DIR / "logging"

DEFAULT_DATASETS = [
    "beauty",
    "foursquare",
    "KuaiRecLargeStrictPosV2_0.2",
    "lastfm0.03",
    "movielens1m",
    "retail_rocket",
]

DATASET_ALIAS = {
    "KuaiRec": "KuaiRecLargeStrictPosV2_0.2",
    "kuairec": "KuaiRecLargeStrictPosV2_0.2",
    "lastfm": "lastfm0.03",
    "ml1m": "movielens1m",
    "movielens": "movielens1m",
    "retail": "retail_rocket",
}

LR_CENTER = {
    "beauty": 0.0005672,
    "foursquare": 0.00095164,
    "KuaiRecLargeStrictPosV2_0.2": 0.0005476,
    "lastfm0.03": 0.0004983,
    "movielens1m": 0.00129037,
    "retail_rocket": 0.00056966,
}

PAIR_SPECS = [
    ("foursquare", "KuaiRecLargeStrictPosV2_0.2", "rich_context_cross"),
    ("KuaiRecLargeStrictPosV2_0.2", "foursquare", "rich_context_cross"),
    ("KuaiRecLargeStrictPosV2_0.2", "lastfm0.03", "rich_context_music_video"),
    ("lastfm0.03", "KuaiRecLargeStrictPosV2_0.2", "rich_context_music_video"),
    ("beauty", "retail_rocket", "low_context_commerce"),
    ("retail_rocket", "beauty", "low_context_commerce"),
    ("retail_rocket", "KuaiRecLargeStrictPosV2_0.2", "low_to_rich"),
    ("beauty", "lastfm0.03", "low_to_rich"),
    ("KuaiRecLargeStrictPosV2_0.2", "movielens1m", "challenging_ml_target"),
    ("lastfm0.03", "movielens1m", "challenging_ml_target"),
    ("KuaiRecLargeStrictPosV2_0.2", "beauty", "target_coverage"),
    ("lastfm0.03", "foursquare", "target_coverage"),
]

TRIPLET_SPECS = [
    ("foursquare", "KuaiRecLargeStrictPosV2_0.2", "lastfm0.03"),
    ("beauty", "retail_rocket", "KuaiRecLargeStrictPosV2_0.2"),
    ("lastfm0.03", "KuaiRecLargeStrictPosV2_0.2", "movielens1m"),
    ("KuaiRecLargeStrictPosV2_0.2", "foursquare", "retail_rocket"),
]

TRANSFER_MODES = [
    "feature_encoder_init",
    "group_router_init",
    "feature_encoder_group_router_init",
    "all_router_init",
    "feature_encoder_router_init",
    "feature_encoder_a12_router_init",
    "full_model_init",
    "full_except_feature_router_init",
]

FAST_HPARAMS = ["shared_3", "shared_4", "shared_5", "shared_6"]
FAST_SEEDS = ["1", "2", "3"]
FAST_PAIR_IDS = [
    "foursquare_to_KuaiRec",
    "KuaiRec_to_foursquare",
    "KuaiRec_to_lastfm",
    "lastfm_to_KuaiRec",
    "beauty_to_retail_rocket",
    "retail_rocket_to_beauty",
    "retail_rocket_to_KuaiRec",
    "beauty_to_lastfm",
    "KuaiRec_to_movielens1m",
    "lastfm_to_movielens1m",
]
FAST_TRANSFER_MODES = list(TRANSFER_MODES)
FAST_FREEZE_MODES = ["feature_encoder_group_router_init", "feature_encoder_a12_router_init", "full_model_init"]
FAST_MULTIHOP_MODES = ["feature_encoder_a12_router_init", "full_model_init"]
FAST_TRIPLET_IDS = [
    "foursquare_to_KuaiRec_to_lastfm",
    "beauty_to_retail_rocket_to_KuaiRec",
]

MODE_BACKEND = {
    "feature_encoder_init": "all_stage_feature_encoder",
    "group_router_init": "all_stage_group_router",
    "feature_encoder_group_router_init": "feature_encoder_group_router",
    "all_router_init": "all_stage_full_router",
    "feature_encoder_router_init": "feature_encoder_router",
    "feature_encoder_a12_router_init": "feature_encoder_active_router",
    "full_model_init": "full_model",
    "full_except_feature_router_init": "full_except_feature_router",
}

HPARAM_PRESETS: dict[str, dict[str, Any]] = {
    "shared_1": dict(embedding_size=128, hidden_size=128, d_ff=256, d_expert_hidden=128, d_router_hidden=32, d_feat_emb=8, MAX_ITEM_LIST_LENGTH=20, num_heads=4, hidden_dropout_prob=0.14, attn_dropout_prob=0.05, route_consistency_lambda=0.00025, z_loss_lambda=0.0001),
    "shared_2": dict(embedding_size=160, hidden_size=160, d_ff=320, d_expert_hidden=160, d_router_hidden=32, d_feat_emb=8, MAX_ITEM_LIST_LENGTH=30, num_heads=4, hidden_dropout_prob=0.14, attn_dropout_prob=0.07, route_consistency_lambda=0.0005, z_loss_lambda=0.0001),
    "shared_3": dict(embedding_size=192, hidden_size=192, d_ff=384, d_expert_hidden=192, d_router_hidden=32, d_feat_emb=16, MAX_ITEM_LIST_LENGTH=20, num_heads=4, hidden_dropout_prob=0.18, attn_dropout_prob=0.05, route_consistency_lambda=0.00025, z_loss_lambda=0.0002),
    "shared_4": dict(embedding_size=192, hidden_size=192, d_ff=384, d_expert_hidden=192, d_router_hidden=64, d_feat_emb=8, MAX_ITEM_LIST_LENGTH=20, num_heads=4, hidden_dropout_prob=0.16, attn_dropout_prob=0.05, route_consistency_lambda=0.00025, z_loss_lambda=0.0004),
    "shared_5": dict(embedding_size=224, hidden_size=224, d_ff=448, d_expert_hidden=224, d_router_hidden=64, d_feat_emb=20, MAX_ITEM_LIST_LENGTH=20, num_heads=4, hidden_dropout_prob=0.12, attn_dropout_prob=0.05, route_consistency_lambda=0.0012, z_loss_lambda=0.0001),
    "shared_6": dict(embedding_size=224, hidden_size=224, d_ff=448, d_expert_hidden=224, d_router_hidden=96, d_feat_emb=12, MAX_ITEM_LIST_LENGTH=30, num_heads=4, hidden_dropout_prob=0.12, attn_dropout_prob=0.07, route_consistency_lambda=0.00025, z_loss_lambda=0.0001),
    "shared_7": dict(embedding_size=128, hidden_size=128, d_ff=256, d_expert_hidden=128, d_router_hidden=96, d_feat_emb=16, MAX_ITEM_LIST_LENGTH=10, num_heads=4, hidden_dropout_prob=0.20, attn_dropout_prob=0.08, route_consistency_lambda=0.0012, z_loss_lambda=0.0004),
    "shared_8": dict(embedding_size=256, hidden_size=256, d_ff=512, d_expert_hidden=256, d_router_hidden=64, d_feat_emb=16, MAX_ITEM_LIST_LENGTH=20, num_heads=4, hidden_dropout_prob=0.16, attn_dropout_prob=0.05, route_consistency_lambda=0.0005, z_loss_lambda=0.0002),
}


def all_stage_map(value: Any) -> dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def a12_stage_router_primitives() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stage in ("macro", "mid", "micro"):
        stage_cfg: dict[str, Any] = {}
        for primitive in ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar"):
            stage_cfg[primitive] = {"source": "feature", "temperature": 1.0, "top_k": 0}
        stage_cfg["wrapper"] = {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0}
        out[stage] = stage_cfg
    return out


COMMON_FIXED = {
    "model": "featured_moe_n3_tune",
    "feature_mode": "full_v4",
    "eval_mode": "session_fixed",
    "fmoe_architecture_id": "A12",
    "fmoe_architecture_key": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
    "train_neg_sample_args": None,
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    "topk": [10, 20],
    "valid_metric": "MRR@20",
    "eval_args": {"split": {"LS": "valid_and_test"}, "order": "TO", "group_by": "user", "mode": "full"},
    "eval_sampling": {"mode": "full", "auto_full_threshold": 999999999},
    "exclude_unseen_target_from_main_eval": True,
    "log_unseen_target_metrics": True,
    "layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"],
    "stage_compute_mode": all_stage_map("moe"),
    "stage_router_mode": all_stage_map("learned"),
    "stage_router_wrapper": all_stage_map("w5_exd"),
    "stage_router_primitives": a12_stage_router_primitives(),
    "stage_router_source": all_stage_map("feature"),
    "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
    "stage_feature_injection": all_stage_map("none"),
    "stage_feature_dropout_scope": all_stage_map("token"),
    "topk_scope_mode": "global_flat",
    "moe_top_k": 0,
    "balance_loss_lambda": 0.0,
    "gate_entropy_lambda": 0.0,
    "route_smoothness_lambda": 0.0,
    "route_sharpness_lambda": 0.0,
    "route_monopoly_lambda": 0.0,
    "route_prior_lambda": 0.0,
    "group_prior_align_lambda": 0.0,
    "factored_group_balance_lambda": 0.0,
    "rule_agreement_lambda": 0.0,
    "group_coverage_lambda": 0.0,
    "feature_group_bias_lambda": 0.0,
    "rule_bias_scale": 0.0,
    "bias_mode": "none",
    "macro_history_window": 5,
    "route_consistency_pairs": 1,
    "route_consistency_min_sim": 0.995,
    "epochs": 100,
    "stopping_step": 10,
    "eval_step": 1,
    "search_algo": "random",
    "fmoe_special_logging": True,
    "fmoe_diag_logging": True,
    "fmoe_eval_logging_timing": "final_only",
    "fmoe_artifact_logging_policy": "combo_best",
    "show_progress": False,
    "log_wandb": False,
}


def hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{hydra_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Hydra literal로 바꿀 수 없는 타입입니다: {type(value).__name__}")


def token(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "x"


def canonical_dataset(name: str) -> str:
    raw = str(name).strip()
    return DATASET_ALIAS.get(raw, raw)


def dataset_short(name: str) -> str:
    mapping = {
        "KuaiRecLargeStrictPosV2_0.2": "KuaiRec",
        "lastfm0.03": "lastfm",
        "movielens1m": "movielens1m",
    }
    return mapping.get(name, name)


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_datasets(value: str | None) -> list[str]:
    return [canonical_dataset(x) for x in parse_csv(value, DEFAULT_DATASETS)]


def parse_hparams(value: str | None) -> list[str]:
    items = parse_csv(value, list(HPARAM_PRESETS))
    bad = [x for x in items if x not in HPARAM_PRESETS]
    if bad:
        raise SystemExit(f"알 수 없는 hparam preset: {bad}")
    return items


def parse_seeds(value: str | None) -> list[int]:
    return [int(x) for x in parse_csv(value, ["1", "2", "3", "4", "5"])]


def parse_gpus(value: str | None) -> list[str]:
    gpus = parse_csv(value, ["0"])
    return gpus or ["0"]


def pair_id(source: str, target: str) -> str:
    return f"{dataset_short(source)}_to_{dataset_short(target)}"


def triplet_id(a: str, b: str, c: str) -> str:
    return f"{dataset_short(a)}_to_{dataset_short(b)}_to_{dataset_short(c)}"


def native_checkpoint(dataset: str, hparam: str, seed: int) -> Path:
    return CHECKPOINT_DIR / "native" / dataset_short(dataset) / hparam / f"seed_{seed}" / "best.pth"


def native_result_key(dataset: str, hparam: str, seed: int) -> str:
    return f"ai502_native://{dataset_short(dataset)}/{hparam}/seed_{seed}"


def transfer_checkpoint(phase: str, source: str, target: str, hparam: str, seed: int, mode: str, freeze_policy: str = "no_freeze") -> Path:
    return CHECKPOINT_DIR / phase / pair_id(source, target) / hparam / f"seed_{seed}" / mode / freeze_policy / "best.pth"


def lr_values(dataset: str, lr_mode: str) -> list[float]:
    center = LR_CENTER[dataset]
    if lr_mode in {"fixed1", "center1"}:
        return [center]
    if lr_mode != "tight3":
        raise SystemExit(f"현재 지원하는 lr-mode는 fixed1, tight3입니다: {lr_mode}")
    return [center * 0.75, center, center * 1.25]


def common_row(dataset: str, hparam: str, seed: int) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "dataset_short": dataset_short(dataset),
        "hparam_id": hparam,
        "seed": seed,
        "preset": HPARAM_PRESETS[hparam],
    }


def build_native_rows(datasets: list[str], hparams: list[str], seeds: list[int], lr_mode: str) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        for hparam in hparams:
            for seed in seeds:
                row = common_row(dataset, hparam, seed)
                row.update(
                    phase="native",
                    source_dataset="",
                    target_dataset=dataset,
                    transfer_mode="none",
                    backend_transfer_mode="none",
                    freeze_policy="scratch",
                    pair_id="",
                    triplet_id="",
                    job_key=f"native/{dataset_short(dataset)}/{hparam}/seed_{seed}/none/scratch/lr_{lr_mode}",
                    run_phase=f"AI502_P0_NATIVE_{token(dataset_short(dataset))}_{hparam}_s{seed}",
                    checkpoint_export=str(native_checkpoint(dataset, hparam, seed)),
                    source_checkpoint="",
                    baseline_native_checkpoint=str(native_checkpoint(dataset, hparam, seed)),
                    baseline_native_result=native_result_key(dataset, hparam, seed),
                    lr_values=lr_values(dataset, lr_mode),
                )
                rows.append(row)
    return rows


def selected_pairs(pair_filter: str | None, datasets: list[str]) -> list[tuple[str, str, str]]:
    allowed = set(datasets)
    pairs = [p for p in PAIR_SPECS if p[0] in allowed and p[1] in allowed]
    if not pair_filter:
        return pairs
    wanted = {token(x) for x in parse_csv(pair_filter, [])}
    return [p for p in pairs if token(pair_id(p[0], p[1])) in wanted or token(f"{p[0]}->{p[1]}") in wanted]


def build_transfer_rows(
    *,
    phase: str,
    datasets: list[str],
    hparams: list[str],
    seeds: list[int],
    lr_mode: str,
    modes: list[str],
    pair_filter: str | None,
    freeze_loaded: bool,
) -> list[dict[str, Any]]:
    rows = []
    for source, target, pair_group in selected_pairs(pair_filter, datasets):
        for hparam in hparams:
            for seed in seeds:
                for mode in modes:
                    freeze_policy = "freeze_loaded" if freeze_loaded else "no_freeze"
                    row = common_row(target, hparam, seed)
                    row.update(
                        phase=phase,
                        source_dataset=source,
                        target_dataset=target,
                        pair_id=pair_id(source, target),
                        pair_group=pair_group,
                        triplet_id="",
                        transfer_mode=mode,
                        backend_transfer_mode=MODE_BACKEND[mode],
                        freeze_policy=freeze_policy,
                        job_key=f"{phase}/{pair_id(source,target)}/{hparam}/seed_{seed}/{mode}/{freeze_policy}/lr_{lr_mode}",
                        run_phase=f"AI502_{'P2_FREEZE' if freeze_loaded else 'P1_INIT'}_{token(pair_id(source,target))}_{hparam}_{token(mode)}_s{seed}",
                        checkpoint_export=str(transfer_checkpoint(phase, source, target, hparam, seed, mode, freeze_policy)),
                        source_checkpoint=str(native_checkpoint(source, hparam, seed)),
                        baseline_native_checkpoint=str(native_checkpoint(target, hparam, seed)),
                        baseline_native_result=native_result_key(target, hparam, seed),
                        lr_values=lr_values(target, lr_mode),
                    )
                    rows.append(row)
    return rows


def read_top2_modes(summary_path: Path) -> dict[str, list[str]]:
    if not summary_path.exists():
        return {}
    out: dict[str, list[str]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            target = row.get("target_dataset") or row.get("dataset") or ""
            mode = row.get("transfer_mode") or ""
            if target and mode:
                out.setdefault(target, [])
                if mode not in out[target] and len(out[target]) < 2:
                    out[target].append(mode)
    return out


def build_freeze_rows(args: argparse.Namespace, datasets: list[str], hparams: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    fallback_modes = parse_csv(args.freeze_modes, ["feature_encoder_router_init", "full_model_init"])
    rows = []
    for source, target, _ in selected_pairs(args.pairs, datasets):
        rows.extend(
            build_transfer_rows(
                phase="freeze",
                datasets=[source, target],
                hparams=hparams,
                seeds=seeds,
                lr_mode=args.lr_mode,
                modes=fallback_modes,
                pair_filter=pair_id(source, target),
                freeze_loaded=True,
            )
        )
    return rows


def selected_triplets(args: argparse.Namespace, datasets: list[str]) -> list[tuple[str, str, str]]:
    allowed = set(datasets)
    triplets = [t for t in TRIPLET_SPECS if all(x in allowed for x in t)]
    wanted_raw = parse_csv(args.triplets, [])
    if not wanted_raw:
        return triplets
    wanted = {token(x) for x in wanted_raw}
    return [t for t in triplets if token(triplet_id(*t)) in wanted or token("->".join(t)) in wanted]


def build_multihop_rows(args: argparse.Namespace, datasets: list[str], hparams: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    modes = parse_csv(args.multihop_modes, ["feature_encoder_router_init", "full_model_init"])
    rows: list[dict[str, Any]] = []
    seen_bridge: set[str] = set()
    for a, b, c in selected_triplets(args, datasets):
        tid = triplet_id(a, b, c)
        for hparam in hparams:
            for seed in seeds:
                for mode in modes:
                    bridge_key = f"bridge/{pair_id(a,b)}/{hparam}/seed_{seed}/{mode}/no_freeze/lr_{args.lr_mode}"
                    bridge_ckpt = transfer_checkpoint("init", a, b, hparam, seed, mode, "no_freeze")
                    if bridge_key not in seen_bridge:
                        seen_bridge.add(bridge_key)
                        row = common_row(b, hparam, seed)
                        row.update(
                            phase="multihop_bridge",
                            source_dataset=a,
                            target_dataset=b,
                            pair_id=pair_id(a, b),
                            triplet_id=tid,
                            comparison_role="a_to_b_bridge",
                            transfer_mode=mode,
                            backend_transfer_mode=MODE_BACKEND[mode],
                            freeze_policy="no_freeze",
                            job_key=bridge_key,
                            run_phase=f"AI502_P3_BRIDGE_{token(pair_id(a,b))}_{hparam}_{token(mode)}_s{seed}",
                            checkpoint_export=str(bridge_ckpt),
                            source_checkpoint=str(native_checkpoint(a, hparam, seed)),
                            baseline_native_checkpoint=str(native_checkpoint(b, hparam, seed)),
                            baseline_native_result=native_result_key(b, hparam, seed),
                            lr_values=lr_values(b, args.lr_mode),
                        )
                        rows.append(row)
                    for source, role, source_ckpt in [
                        (a, "a_to_c_direct", native_checkpoint(a, hparam, seed)),
                        (b, "b_to_c_direct", native_checkpoint(b, hparam, seed)),
                        (b, "a_to_b_to_c", bridge_ckpt),
                    ]:
                        row = common_row(c, hparam, seed)
                        row.update(
                            phase="multihop",
                            source_dataset=source,
                            target_dataset=c,
                            pair_id=pair_id(source, c),
                            triplet_id=tid,
                            comparison_role=role,
                            transfer_mode=mode,
                            backend_transfer_mode=MODE_BACKEND[mode],
                            freeze_policy="no_freeze",
                            job_key=f"multihop/{tid}/{role}/{hparam}/seed_{seed}/{mode}/no_freeze/lr_{args.lr_mode}",
                            run_phase=f"AI502_P3_{token(tid)}_{token(role)}_{hparam}_{token(mode)}_s{seed}",
                            checkpoint_export=str(transfer_checkpoint("multihop", source, c, hparam, seed, mode, role)),
                            source_checkpoint=str(source_ckpt),
                            baseline_native_checkpoint=str(native_checkpoint(c, hparam, seed)),
                            baseline_native_result=native_result_key(c, hparam, seed),
                            lr_values=lr_values(c, args.lr_mode),
                        )
                        rows.append(row)
    return rows


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = parse_datasets(args.datasets)
    hparams = parse_hparams(args.hparams)
    seeds = parse_seeds(args.seeds)
    if args.smoke_test:
        datasets = datasets[:2]
        hparams = hparams[:1]
        seeds = seeds[:1]

    phases = ["native", "init", "freeze", "multihop"] if args.phase == "all" else [args.phase]
    rows: list[dict[str, Any]] = []
    if "native" in phases:
        rows.extend(build_native_rows(datasets, hparams, seeds, args.lr_mode))
    if "init" in phases:
        modes = parse_csv(args.transfer_modes, TRANSFER_MODES)
        rows.extend(
            build_transfer_rows(
                phase="init",
                datasets=datasets,
                hparams=hparams,
                seeds=seeds,
                lr_mode=args.lr_mode,
                modes=modes,
                pair_filter=args.pairs,
                freeze_loaded=False,
            )
        )
    if "freeze" in phases:
        rows.extend(build_freeze_rows(args, datasets, hparams, seeds))
    if "multihop" in phases:
        rows.extend(build_multihop_rows(args, datasets, hparams, seeds))

    if args.smoke_test:
        limit = int(args.smoke_max_runs)
        rows = rows[:limit]
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["job_key"] for row in rows)
    dup = [key for key, count in counts.items() if count > 1]
    if dup:
        sample = "\n".join(dup[:10])
        raise SystemExit(f"중복 job_key가 있습니다:\n{sample}")


def write_manifest(rows: list[dict[str, Any]], phase: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{phase}_{int(time.time())}.csv"
    fieldnames = [
        "phase",
        "job_key",
        "run_phase",
        "dataset",
        "source_dataset",
        "target_dataset",
        "pair_id",
        "triplet_id",
        "comparison_role",
        "pair_group",
        "hparam_id",
        "seed",
        "transfer_mode",
        "backend_transfer_mode",
        "freeze_policy",
        "source_checkpoint",
        "baseline_native_checkpoint",
        "checkpoint_export",
        "lr_values",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False) if key == "lr_values" else row.get(key, "") for key in fieldnames})
    return path


def build_command(row: dict[str, Any], gpu: str, args: argparse.Namespace) -> list[str]:
    search_lr = row["lr_values"]
    gpu_value: Any = int(gpu) if str(gpu).isdigit() else str(gpu)
    max_evals = int(args.max_evals) if int(args.max_evals) > 0 else max(1, len(search_lr))
    overrides: dict[str, Any] = {
        **COMMON_FIXED,
        **row["preset"],
        "dataset": row["dataset"],
        "gpu_id": gpu_value,
        "seed": int(row["seed"]),
        "data_path": str(DATA_ROOT),
        "run_group": "ai502_transfer",
        "run_axis": "transfer",
        "run_phase": row["run_phase"],
        "fmoe_logging_output_root": str(LOGGING_DIR),
        "__artifact_combo_best_export_path": row["checkpoint_export"],
        "baseline_native_checkpoint": row["baseline_native_checkpoint"],
        "baseline_native_result": row["baseline_native_result"],
        "search_space_type_overrides": {"learning_rate": "choice"},
        "search": {"learning_rate": search_lr},
    }
    if row["transfer_mode"] != "none":
        overrides["transfer"] = {
            "enable": True,
            "mode": row["backend_transfer_mode"],
            "source_checkpoint": row["source_checkpoint"],
            "source_architecture": "A12",
            "strict_shape": False,
            "freeze_loaded": row["freeze_policy"] == "freeze_loaded",
            "allow_empty": False,
            "allow_router_fallback": False,
        }
    else:
        overrides["transfer"] = {"enable": False, "mode": "none"}

    cmd = [
        str(PYTHON_BIN),
        str(HYPEROPT),
        "--config-name",
        "config",
        "--max-evals",
        str(max_evals),
        "--tune-epochs",
        str(args.epochs),
        "--tune-patience",
        str(args.patience),
        "--search-algo",
        "random",
        "--seed",
        str(row["seed"]),
        "--run-group",
        "ai502_transfer",
        "--run-axis",
        "transfer",
        "--run-phase",
        row["run_phase"],
    ]
    plain_override_keys = {"model", "dataset", "gpu_id", "feature_mode", "eval_mode"}
    for key, value in overrides.items():
        prefix = "++" if key not in plain_override_keys else ""
        cmd.append(f"{prefix}{key}={hydra_literal(value)}")
    return cmd


def write_log_preamble(log_path: Path, row: dict[str, Any], gpu: str, cmd: list[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: row.get(k, "") for k in row if k != "preset"}
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("[AI502_ROW] " + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        fh.write(f"[GPU] {gpu}\n")
        fh.write("[COMMAND] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")


def log_has_zero_ok_trials(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "0/1 trials OK" in text or "0 / 1 trials OK" in text or "0/1 trials ok" in text.lower()


def run_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> int:
    gpus = parse_gpus(args.gpus)
    queue = deque(rows)
    active: dict[str, dict[str, Any]] = {}
    env = dict(os.environ)
    env["HYPEROPT_RESULTS_DIR"] = str(RESULTS_DIR)
    pythonpath_parts = [str(EXP_DIR), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            row = queue.popleft()
            if args.resume_from_logs and Path(str(row.get("checkpoint_export", ""))).exists():
                print(f"[skip] checkpoint exists job={row['job_key']} ckpt={row['checkpoint_export']}")
                continue
            row["assigned_gpu"] = gpu
            log_path = LOG_DIR / row["phase"] / f"{token(row['run_phase'])}.log"
            row["log_path"] = str(log_path)
            cmd = build_command(row, gpu, args)
            write_log_preamble(log_path, row, gpu, cmd)
            if args.dry_run:
                print(f"[dry-run] gpu={gpu} {row['job_key']} -> {log_path}")
                print("          " + " ".join(shlex.quote(x) for x in cmd))
                continue
            with log_path.open("a", encoding="utf-8") as fh:
                proc = subprocess.Popen(cmd, cwd=str(EXP_DIR), env=env, stdout=fh, stderr=subprocess.STDOUT)
            active[gpu] = {"proc": proc, "row": row, "log": log_path}
            print(f"[launch] gpu={gpu} phase={row['phase']} job={row['job_key']}")

        if args.dry_run:
            continue

        done = []
        for gpu, slot in active.items():
            rc = slot["proc"].poll()
            if rc is None:
                continue
            row = slot["row"]
            if rc == 0 and log_has_zero_ok_trials(Path(slot["log"])):
                rc = 2
            with Path(slot["log"]).open("a", encoding="utf-8") as fh:
                fh.write(f"\n[RUN_STATUS] rc={rc} job_key={row['job_key']}\n")
            print(f"[done] gpu={gpu} rc={rc} job={row['job_key']} log={slot['log']}")
            done.append(gpu)
            if rc != 0 and not args.keep_going:
                raise SystemExit(rc)
        for gpu in done:
            active.pop(gpu, None)
        if queue or active:
            time.sleep(5)
    return 0


def run_summarizer() -> int:
    cmd = [str(PYTHON_BIN), str(THIS_DIR / "summarize_ai502_transfer.py")]
    print("[summary] " + " ".join(shlex.quote(x) for x in cmd))
    return subprocess.call(cmd, cwd=str(THIS_DIR))


def run_one_phase(phase: str, args: argparse.Namespace) -> int:
    phase_args = copy(args)
    phase_args.phase = phase
    rows = build_rows(phase_args)
    validate_rows(rows)
    manifest = write_manifest(rows, phase)
    print(f"[manifest] phase={phase} rows={len(rows)} path={manifest}")
    if args.resume_from_logs:
        print("[note] resume-from-logs=on: checkpoint_export가 이미 있으면 해당 job을 skip합니다.")
    if not rows:
        return 0
    return run_rows(rows, phase_args)


def main() -> int:
    parser = argparse.ArgumentParser(description="AI502 transfer learning phase launcher")
    parser.add_argument("--phase", choices=["native", "init", "freeze", "multihop", "all"], default="native")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--profile", choices=["fast", "full"], default="fast")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--triplets", default="")
    parser.add_argument("--hparams", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--lr-mode", default="")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--transfer-modes", default="")
    parser.add_argument("--freeze-modes", default="")
    parser.add_argument("--multihop-modes", default="")
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=12)
    parser.add_argument("--resume-from-logs", action="store_true", default=False)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.profile == "fast":
        if not args.hparams:
            args.hparams = ",".join(FAST_HPARAMS)
        if not args.seeds:
            args.seeds = ",".join(FAST_SEEDS)
        if not args.pairs:
            args.pairs = ",".join(FAST_PAIR_IDS)
        if not args.lr_mode:
            args.lr_mode = "fixed1"
        if not args.transfer_modes:
            args.transfer_modes = ",".join(FAST_TRANSFER_MODES)
        if not args.freeze_modes:
            args.freeze_modes = ",".join(FAST_FREEZE_MODES)
        if not args.multihop_modes:
            args.multihop_modes = ",".join(FAST_MULTIHOP_MODES)
        if not args.triplets:
            args.triplets = ",".join(FAST_TRIPLET_IDS)
    else:
        if not args.hparams:
            args.hparams = ",".join(HPARAM_PRESETS)
        if not args.seeds:
            args.seeds = "1,2,3,4,5"
        if not args.lr_mode:
            args.lr_mode = "tight3"
        if not args.transfer_modes:
            args.transfer_modes = ",".join(TRANSFER_MODES)
        if not args.freeze_modes:
            args.freeze_modes = "feature_encoder_router_init,full_model_init"
        if not args.multihop_modes:
            args.multihop_modes = "feature_encoder_router_init,full_model_init"

    if args.phase == "all":
        for phase in ["native", "init", "freeze", "multihop"]:
            rc = run_one_phase(phase, args)
            if rc != 0:
                return rc
            if not args.dry_run and phase in {"init", "freeze", "multihop"}:
                rc = run_summarizer()
                if rc != 0 and not args.keep_going:
                    return rc
        return 0

    return run_one_phase(args.phase, args)


if __name__ == "__main__":
    raise SystemExit(main())
