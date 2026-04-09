#!/usr/bin/env python3
"""Stage H targeted recovery runner for weak baseline combos.

Focus:
- Recover the largest underperformers from Stage G
- Keep search compact but more diverse than transfer-only Stage G candidates
- Reuse Stage G execution/logging skeleton
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import run_stageG_cross6x9 as sg

base = sg.base


base.AXIS = "StageH_TargetedRecovery_anchor2_core5"
base.PHASE_ID = "P23"
base.PHASE_NAME = "STAGEH_TARGETEDRECOVERY_ANCHOR2_CORE5"
base.AXIS_DESC = "stageh_targetedrecovery_anchor2_core5"

_orig_model_runtime_resource_overrides = base._model_runtime_resource_overrides


TARGET_COMBOS: List[Tuple[str, str]] = [
    ("amazon_beauty", "bsarec"),
    ("amazon_beauty", "gru4rec"),
    ("amazon_beauty", "fame"),
    ("amazon_beauty", "tisasrec"),
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"),
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"),
    ("KuaiRecLargeStrictPosV2_0.2", "duorec"),
    ("movielens1m", "sasrec"),
    ("movielens1m", "gru4rec"),
    ("movielens1m", "duorec"),
    ("movielens1m", "fearec"),
]

DEFAULT_DATASETS = list(dict.fromkeys(ds for ds, _ in TARGET_COMBOS))
DEFAULT_MODELS = list(dict.fromkeys(model for _, model in TARGET_COMBOS))

TARGET_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    ("amazon_beauty", "gru4rec"): 3,
    ("amazon_beauty", "fame"): 3,
    ("amazon_beauty", "bsarec"): 2,
    ("amazon_beauty", "tisasrec"): 2,
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"): 2,
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"): 2,
    ("KuaiRecLargeStrictPosV2_0.2", "duorec"): 2,
    ("movielens1m", "sasrec"): 2,
    ("movielens1m", "gru4rec"): 2,
    ("movielens1m", "duorec"): 2,
    ("movielens1m", "fearec"): 2,
}

FAST_SCREEN_MIN_CANDIDATE_COUNT: Dict[Tuple[str, str], int] = {
    ("amazon_beauty", "gru4rec"): 10,
    ("amazon_beauty", "fame"): 10,
    ("amazon_beauty", "bsarec"): 10,
    ("amazon_beauty", "tisasrec"): 10,
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"): 10,
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"): 10,
    ("KuaiRecLargeStrictPosV2_0.2", "duorec"): 10,
    ("movielens1m", "sasrec"): 10,
    ("movielens1m", "gru4rec"): 10,
    ("movielens1m", "duorec"): 10,
    ("movielens1m", "fearec"): 10,
}

MANUAL_TEMPLATE_BANK: Dict[Tuple[str, str], List[Dict[str, Any]]] = {
    ("amazon_beauty", "gru4rec"): [
        {
            "id": "AB_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 264,
                "embedding_size": 264,
                "layers": 3,
                "num_layers": 3,
                "max_len": 10,
                "dropout": 0.05,
                "weight_decay": 4.5e-5,
            },
            "lr_lo": 1.2e-3,
            "lr_hi": 4.8e-3,
        },
        {
            "id": "AB_GRU_STAGEH_REC2",
            "config": {
                "hidden_size": 320,
                "embedding_size": 320,
                "layers": 3,
                "num_layers": 3,
                "max_len": 8,
                "dropout": 0.04,
                "weight_decay": 3.5e-5,
            },
            "lr_lo": 3.0e-3,
            "lr_hi": 1.0e-2,
        },
    ],
    ("amazon_beauty", "fame"): [
        {
            "id": "AB_FAME_STAGEH_REC1",
            "config": {
                "hidden_size": 88,
                "embedding_size": 88,
                "layers": 3,
                "num_layers": 3,
                "heads": 8,
                "inner_size": 176,
                "max_len": 10,
                "dropout": 0.14,
                "weight_decay": 1.2e-4,
                "num_experts": 4,
            },
            "lr_lo": 1.5e-3,
            "lr_hi": 6.0e-3,
        },
        {
            "id": "AB_FAME_STAGEH_REC2",
            "config": {
                "hidden_size": 112,
                "embedding_size": 112,
                "layers": 3,
                "num_layers": 3,
                "heads": 8,
                "inner_size": 224,
                "max_len": 8,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "num_experts": 3,
            },
            "lr_lo": 2.5e-3,
            "lr_hi": 1.0e-2,
        },
    ],
    ("amazon_beauty", "bsarec"): [
        {
            "id": "AB_BSAREC_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 10,
                "dropout": 0.12,
                "weight_decay": 2.0e-4,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 1.1e-3,
        },
    ],
    ("amazon_beauty", "tisasrec"): [
        {
            "id": "AB_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 10,
                "time_span": 256,
                "dropout": 0.10,
                "weight_decay": 1.5e-4,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 9.0e-4,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "gru4rec"): [
        {
            "id": "KR_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 320,
                "embedding_size": 320,
                "layers": 3,
                "num_layers": 3,
                "max_len": 10,
                "dropout": 0.05,
                "weight_decay": 3.0e-5,
            },
            "lr_lo": 2.0e-3,
            "lr_hi": 8.0e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "tisasrec"): [
        {
            "id": "KR_TISAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 20,
                "time_span": 384,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
            },
            "lr_lo": 4.0e-4,
            "lr_hi": 1.0e-3,
        },
    ],
    ("KuaiRecLargeStrictPosV2_0.2", "duorec"): [
        {
            "id": "KR_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 20,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.25,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 9.0e-4,
        },
    ],
    ("movielens1m", "sasrec"): [
        {
            "id": "ML1M_SAS_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.12,
                "weight_decay": 1.0e-4,
            },
            "lr_lo": 4.0e-4,
            "lr_hi": 1.2e-3,
        },
    ],
    ("movielens1m", "gru4rec"): [
        {
            "id": "ML1M_GRU_STAGEH_REC1",
            "config": {
                "hidden_size": 264,
                "embedding_size": 264,
                "layers": 3,
                "num_layers": 3,
                "max_len": 20,
                "dropout": 0.05,
                "weight_decay": 5.0e-5,
            },
            "lr_lo": 1.0e-3,
            "lr_hi": 4.0e-3,
        },
    ],
    ("movielens1m", "duorec"): [
        {
            "id": "ML1M_DUO_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.25,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 8.0e-4,
        },
    ],
    ("movielens1m", "fearec"): [
        {
            "id": "ML1M_FEA_STAGEH_REC1",
            "config": {
                "hidden_size": 160,
                "embedding_size": 160,
                "layers": 3,
                "num_layers": 3,
                "heads": 4,
                "inner_size": 320,
                "max_len": 30,
                "dropout": 0.10,
                "weight_decay": 1.0e-4,
                "contrast": "un",
                "tau": 0.18,
                "lmd": 0.02,
                "lmd_sem": 0.0,
                "global_ratio": 0.75,
                "semantic_sample_max_tries": 2,
            },
            "lr_lo": 3.0e-4,
            "lr_hi": 8.0e-4,
        },
    ],
}


def _manual_candidates(dataset: str, model_option: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for template in MANUAL_TEMPLATE_BANK.get((dataset, model_option), []):
        out.append(
            sg._manual_candidate(
                template,
                model_option,
                source_stage="stageh_manual_bank",
                transfer_from_dataset=dataset,
            )
        )
    return out


def _stageh_target_max_len(
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    *,
    fast_screen: bool,
) -> int:
    model_key = str(model_option).lower()
    hidden_size = int(cfg.get("hidden_size", 128))
    layers = int(cfg.get("layers", cfg.get("num_layers", 2)))
    time_span = int(cfg.get("time_span", 384))

    # Keep StageH aggressively short by default. Only allow 20 when the config
    # is genuinely heavier or time-aware enough that a tiny horizon is likely
    # to distort the ranking too much.
    if model_key == "tisasrec":
        return 20 if (not fast_screen and time_span >= 320) else 10
    if model_key in {"duorec", "fearec"}:
        return 20 if (not fast_screen and (hidden_size >= 160 or layers >= 3)) else 10
    if dataset == "KuaiRecLargeStrictPosV2_0.2" and model_key in {"sasrec", "bsarec", "difsr", "sigma"}:
        return 20 if (not fast_screen and hidden_size >= 160 and layers >= 3) else 10
    return 10


def _speedify_candidate_config(
    *,
    dataset: str,
    model_option: str,
    cfg: Dict[str, Any],
    fast_screen: bool,
) -> Dict[str, Any]:
    out = sg._normalize_cfg(model_option, dict(cfg))
    target_max_len = _stageh_target_max_len(dataset, model_option, out, fast_screen=fast_screen)
    out["max_len"] = min(int(out.get("max_len", target_max_len)), int(target_max_len))

    if model_option in {"sasrec", "tisasrec", "bsarec", "duorec", "fearec", "difsr", "fame"}:
        layer_cap = 2 if fast_screen else 3
        layers = min(int(out.get("layers", 2)), int(layer_cap))
        out["layers"] = layers
        out["num_layers"] = layers

    if model_option in {"duorec", "fearec"}:
        heads = min(int(out.get("heads", 2)), 2 if fast_screen else 4)
        out["heads"] = heads
        if fast_screen:
            out["hidden_size"] = min(int(out.get("hidden_size", 128)), 128)
            out["embedding_size"] = min(int(out.get("embedding_size", out["hidden_size"])), int(out["hidden_size"]))
            out["inner_size"] = min(int(out.get("inner_size", max(256, out["hidden_size"] * 2))), 256)

    if model_option in {"sasrec", "bsarec", "difsr", "tisasrec"} and fast_screen:
        out["hidden_size"] = min(int(out.get("hidden_size", 128)), 144)
        out["embedding_size"] = min(int(out.get("embedding_size", out["hidden_size"])), int(out["hidden_size"]))
        out["inner_size"] = min(int(out.get("inner_size", max(256, out["hidden_size"] * 2))), 288)

    return sg._normalize_cfg(model_option, out)


def _speedify_lr_band(
    *,
    model_option: str,
    lr_lo: float,
    lr_hi: float,
    fast_screen: bool,
) -> Tuple[float, float]:
    model_key = str(model_option).lower()
    lo_mult = 1.10 if fast_screen else 1.05
    hi_mult = 1.18 if fast_screen else 1.12
    if model_key in {"gru4rec", "fame"}:
        lo_mult += 0.05
        hi_mult += 0.07
    new_lo = max(8e-5, min(1e-2, float(lr_lo) * lo_mult))
    new_hi = max(new_lo * 1.05, min(1e-2, float(lr_hi) * hi_mult))
    return new_lo, new_hi


def _screen_variant_candidates(
    *,
    dataset: str,
    model_option: str,
    source_candidates: List[Dict[str, Any]],
    desired: int,
) -> List[Dict[str, Any]]:
    if desired <= 0:
        return []

    variants: List[Dict[str, Any]] = []
    for cand in source_candidates:
        if len(variants) >= desired:
            break
        cfg = sg._normalize_cfg(model_option, dict(cand["config"]))
        base_lo = float(cand["lr_lo"])
        base_hi = float(cand["lr_hi"])
        center = (base_lo * base_hi) ** 0.5

        recipes: List[Tuple[str, Dict[str, Any], float, float]] = []
        if model_option in {"gru4rec", "fame"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.03, float(rec_cfg.get("dropout", 0.08)) - 0.02)
            recipes.append(("LRHI", rec_cfg, center * 0.95, min(1e-2, base_hi * 1.18)))

            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = min(0.20, float(rec_cfg.get("dropout", 0.08)) + 0.03)
            recipes.append(("DROP", rec_cfg, max(8e-5, base_lo * 0.92), center * 1.05))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 5e-5)) * 0.7)
            recipes.append(("WDLO", rec_cfg, max(8e-5, center * 0.88), min(1e-2, base_hi * 1.08)))
        elif model_option in {"duorec", "fearec"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.05, float(rec_cfg.get("dropout", 0.10)) - 0.02)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 1e-4)) * 0.6)
            recipes.append(("LIGHT", rec_cfg, max(8e-5, base_lo * 0.95), center * 1.08))

            rec_cfg = dict(cfg)
            rec_cfg["tau"] = max(0.12, float(rec_cfg.get("tau", 0.2)) - 0.04)
            rec_cfg["lmd"] = max(0.01, float(rec_cfg.get("lmd", 0.02)) * 0.8)
            recipes.append(("TAU", rec_cfg, center * 0.92, min(1e-2, base_hi * 1.10)))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = min(3e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 1.5)
            recipes.append(("WDHI", rec_cfg, max(8e-5, base_lo * 0.90), center * 1.03))
        elif model_option in {"sasrec", "bsarec", "tisasrec", "difsr"}:
            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = max(0.06, float(rec_cfg.get("dropout", 0.10)) - 0.02)
            rec_cfg["weight_decay"] = max(1e-6, float(rec_cfg.get("weight_decay", 1e-4)) * 0.7)
            recipes.append(("REGLO", rec_cfg, max(8e-5, base_lo * 0.90), center * 1.06))

            rec_cfg = dict(cfg)
            rec_cfg["dropout"] = min(0.20, float(rec_cfg.get("dropout", 0.10)) + 0.03)
            recipes.append(("REGHI", rec_cfg, center * 0.94, min(1e-2, base_hi * 1.08)))

            rec_cfg = dict(cfg)
            rec_cfg["weight_decay"] = min(4e-4, float(rec_cfg.get("weight_decay", 1e-4)) * 1.8)
            recipes.append(("WDHI", rec_cfg, max(8e-5, base_lo * 0.94), center * 1.04))
        else:
            rec_cfg = dict(cfg)
            recipes.append(("LRMID", rec_cfg, max(8e-5, base_lo * 0.93), min(1e-2, base_hi * 1.08)))

        for suffix, rec_cfg, lr_lo, lr_hi in recipes:
            if len(variants) >= desired:
                break
            variants.append(
                {
                    "candidate_id": f"{cand['candidate_id']}_{suffix}",
                    "source": "stageh_screen_variant",
                    "source_stage": f"{cand.get('source_stage', 'stageh_screen')}_variant",
                    "transfer_from_dataset": cand.get("transfer_from_dataset", dataset),
                    "config": sg._normalize_cfg(model_option, rec_cfg),
                    "lr_lo": max(8e-5, min(1e-2, float(lr_lo))),
                    "lr_hi": max(max(8e-5, min(1e-2, float(lr_lo))) * 1.05, min(1e-2, float(lr_hi))),
                }
            )
    return variants[:desired]


def _model_runtime_resource_overrides_stageh(row: Dict[str, Any]) -> List[str]:
    if str(row.get("stage", "")).lower() != "stageh":
        return _orig_model_runtime_resource_overrides(row)

    model_option = str(row.get("model_option", "")).lower()
    h = base._effective_model_hparams(row)
    hidden_size = int(h.get("hidden_size", 128))
    fast_screen = bool(row.get("fast_screen", False))

    def _width_scale(bs: int) -> int:
        factor = 1.0
        if hidden_size >= 224:
            factor = 0.78
        elif hidden_size >= 192:
            factor = 0.88
        elif hidden_size >= 160:
            factor = 0.94
        return max(512, int(bs * factor))

    if model_option == "gru4rec":
        train_bs = 12288 if fast_screen else 10240
        eval_bs = 16384 if fast_screen else 12288
    elif model_option in {"sasrec", "bsarec", "difsr"}:
        train_bs = 6144 if fast_screen else 5120
        eval_bs = 8192 if fast_screen else 6144
    elif model_option == "tisasrec":
        train_bs = 2048 if fast_screen else 1536
        eval_bs = 3072 if fast_screen else 2304
    elif model_option == "sigma":
        train_bs = 2560 if fast_screen else 2048
        eval_bs = 3584 if fast_screen else 3072
    elif model_option == "duorec":
        train_bs = 3072 if fast_screen else 3584
        eval_bs = 8192 if fast_screen else 9216
    elif model_option == "fearec":
        train_bs = 4096 if fast_screen else 4608
        eval_bs = 8192 if fast_screen else 10240
    elif model_option == "fame":
        train_bs = 1792 if fast_screen else 1536
        eval_bs = 2816 if fast_screen else 2304
    else:
        return _orig_model_runtime_resource_overrides(row)

    train_bs = _width_scale(int(train_bs))
    eval_bs = _width_scale(int(eval_bs))
    return [f"++train_batch_size={int(train_bs)}", f"++eval_batch_size={int(eval_bs)}"]


base._model_runtime_resource_overrides = _model_runtime_resource_overrides_stageh


def _target_budget(model_option: str, *, fast_screen: bool) -> Tuple[int, int, int]:
    m = str(model_option).lower()
    if fast_screen:
        if m in {"gru4rec", "fame"}:
            return 4, 24, 3
        if m in {"duorec", "fearec"}:
            return 3, 20, 3
        return 3, 20, 3
    if m in {"gru4rec", "fame"}:
        return 6, 40, 5
    if m in {"duorec", "fearec"}:
        return 5, 36, 5
    return 5, 34, 5


def _selected_combos(args: argparse.Namespace) -> List[Tuple[str, str]]:
    datasets = set(base._parse_csv_strings(args.datasets))
    models = set(m.lower() for m in base._parse_csv_strings(args.models))
    return [(ds, m) for ds, m in TARGET_COMBOS if ds in datasets and m in models]


def _select_stageh_candidates(
    *,
    dataset: str,
    model_option: str,
    model_label: str,
    final_cache_by_dataset: Dict[str, Dict[str, List[Dict[str, Any]]]],
    global_cache_by_model: Dict[str, List[Dict[str, Any]]],
    stage_best_cache: Dict[Tuple[str, str], Dict[str, Any]],
    fast_screen: bool,
) -> List[Dict[str, Any]]:
    goal = int(TARGET_CANDIDATE_COUNT[(dataset, model_option)])
    if fast_screen:
        goal = max(goal, int(FAST_SCREEN_MIN_CANDIDATE_COUNT.get((dataset, model_option), goal)))
    base_cands = sg._select_candidates_for_combo(
        dataset=dataset,
        model_option=model_option,
        model_label=model_label,
        final_cache_by_dataset=final_cache_by_dataset,
        global_cache_by_model=global_cache_by_model,
        stage_best_cache=stage_best_cache,
    )
    manual_cands = _manual_candidates(dataset, model_option)

    chosen: List[Dict[str, Any]] = []
    if fast_screen:
        chosen.extend(dict(c) for c in base_cands[: min(4, len(base_cands))])
        chosen.extend(manual_cands[: min(4, len(manual_cands))])
    elif goal >= 3 and manual_cands:
        chosen.extend(manual_cands[:2])
        if base_cands:
            chosen.append(dict(base_cands[0]))
    else:
        if base_cands:
            chosen.append(dict(base_cands[0]))
        if manual_cands:
            chosen.append(manual_cands[0])
        elif len(base_cands) >= 2:
            chosen.append(dict(base_cands[1]))

    for cand in manual_cands[2:]:
        if len(chosen) >= goal:
            break
        chosen.append(cand)
    for cand in base_cands[1:]:
        if len(chosen) >= goal:
            break
        chosen.append(dict(cand))

    chosen = sg._dedup_candidates(chosen)
    if fast_screen and len(chosen) < goal:
        chosen.extend(
            _screen_variant_candidates(
                dataset=dataset,
                model_option=model_option,
                source_candidates=list(chosen),
                desired=max(0, goal - len(chosen)),
            )
        )
        chosen = sg._dedup_candidates(chosen)
    while len(chosen) < goal:
        chosen.append(sg._fallback_candidate(dataset, model_option, len(chosen) + 1))
    return chosen[:goal]


def _build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    seeds = base._parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No seeds provided")

    final_cache_by_dataset, global_cache_by_model = sg._load_final_hparam_cache()
    stage_best_cache = sg._load_stage_best_cache()
    combos = _selected_combos(args)
    if not combos:
        raise RuntimeError("No target combos selected")

    rows: List[Dict[str, Any]] = []
    run_cursor = 0

    for combo_idx, (dataset, model_option) in enumerate(combos, start=1):
        model_label = sg.OPTION_TO_LABEL[model_option]
        candidates = _select_stageh_candidates(
            dataset=dataset,
            model_option=model_option,
            model_label=model_label,
            final_cache_by_dataset=final_cache_by_dataset,
            global_cache_by_model=global_cache_by_model,
            stage_best_cache=stage_best_cache,
            fast_screen=bool(args.fast_screen),
        )
        for cand_idx, cand in enumerate(candidates, start=1):
            max_evals, tune_epochs, tune_patience = _target_budget(model_option, fast_screen=bool(args.fast_screen))
            if args.smoke_test:
                max_evals, tune_epochs, tune_patience = 1, 1, 1
            candidate_config = _speedify_candidate_config(
                dataset=dataset,
                model_option=model_option,
                cfg=dict(cand["config"]),
                fast_screen=bool(args.fast_screen),
            )
            lr_lo, lr_hi = _speedify_lr_band(
                model_option=model_option,
                lr_lo=float(cand["lr_lo"]),
                lr_hi=float(cand["lr_hi"]),
                fast_screen=bool(args.fast_screen),
            )

            for seed_id in seeds:
                run_cursor += 1
                row = {
                    "dataset": dataset,
                    "phase_id": base.PHASE_ID,
                    "axis_id": "SH",
                    "axis_desc": base.AXIS_DESC,
                    "setting_id": f"STAGEH_{base._sanitize_token(model_label, upper=True)}_{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{seed_id}",
                    "setting_key": "STAGEH_TARGETED_RECOVERY",
                    "setting_desc": f"STAGEH_TARGETED_RECOVERY_{base._sanitize_token(model_label, upper=True)}_{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{seed_id}",
                    "hparam_id": base._sanitize_token(str(cand["candidate_id"]), upper=True)[:40],
                    "seed_id": int(seed_id),
                    "run_phase": (
                        f"{base.PHASE_ID}_SH_C{combo_idx:02d}_M{base._sanitize_token(model_label, upper=True)}_"
                        f"{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{int(seed_id)}"
                    ),
                    "run_id": (
                        f"SH_{base._sanitize_token(dataset, upper=True)}_{base._sanitize_token(model_label, upper=True)}_"
                        f"{base._sanitize_token(str(cand['candidate_id']), upper=True)}_S{int(seed_id)}"
                    ),
                    "runtime_seed": int(args.seed_base) + int(run_cursor - 1),
                    "stage": "stageh",
                    "model_option": model_option,
                    "model_label": model_label,
                    "candidate_id": str(cand["candidate_id"]),
                    "candidate_source": str(cand["source"]),
                    "source_stage": str(cand["source_stage"]),
                    "transfer_from_dataset": str(cand["transfer_from_dataset"]),
                    "candidate_config": candidate_config,
                    "lr_lo": float(lr_lo),
                    "lr_hi": float(lr_hi),
                    "max_evals": int(max_evals),
                    "tune_epochs": int(tune_epochs),
                    "tune_patience": int(tune_patience),
                    "fast_screen": bool(args.fast_screen),
                }
                row["estimated_cost"] = sg._stageg_estimated_cost(row)
                rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline P23 StageH targeted recovery launcher")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--seed-base", type=int, default=230000)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=8)
    parser.add_argument("--fast-screen", action="store_true")
    return parser.parse_args()


def _apply_smoke_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"
    gpus = base._parse_csv_strings(args.gpus)
    args.gpus = gpus[0] if gpus else "0"


def _apply_fast_screen_mode(args: argparse.Namespace) -> None:
    args.seeds = "1"


def _run(args: argparse.Namespace) -> int:
    gpus = list(dict.fromkeys(base._parse_csv_strings(args.gpus)))
    rows = _build_rows(args)
    if args.smoke_test:
        rows = rows[: max(1, int(args.smoke_max_runs))]

    dataset_labels = ",".join(sorted(dict.fromkeys(r["dataset"] for r in rows)))
    summary_paths = {dataset: base._summary_path(dataset) for dataset in sorted(dict.fromkeys(r["dataset"] for r in rows))}
    for path in summary_paths.values():
        base._ensure_summary_csv(path)

    for row in rows:
        row["log_path"] = str(base._build_log_path(row))

    manifest_path = base._manifest_path(dataset_labels.replace(",", "_"), args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base._summary_fieldnames = sg._summary_fieldnames
    sg._write_manifest(dataset_labels.replace(",", "_"), args, rows)

    runnable = rows
    runnable.sort(key=lambda r: float(r.get("estimated_cost", 1.0)), reverse=True)
    print(f"[{base.PHASE_ID}] combos={len(TARGET_COMBOS)} selected_runs={len(runnable)} gpus={','.join(gpus)}")

    if args.dry_run:
        gpu_bins = base._plan_gpu_bins(runnable, gpus)
        for gpu_id in gpus:
            for row in gpu_bins[gpu_id]:
                cmd = sg._build_command(row, gpu_id, args)
                print(
                    f"[dry-run] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} "
                    f"model={row['model_label']} candidate={row['candidate_id']} seed={row['seed_id']}"
                )
                print("          " + " ".join(cmd))
        return 0

    active: Dict[str, Dict[str, Any]] = {}
    shared_queue: deque = deque(runnable)
    try:
        while True:
            for gpu_id in gpus:
                if gpu_id in active or not shared_queue:
                    continue
                row = shared_queue.popleft()
                cmd = sg._build_command(row, gpu_id, args)
                log_path = Path(str(row["log_path"]))
                sg._write_log_preamble(log_path, row, gpu_id, args, cmd)
                env = dict(os.environ)
                env["PYTHONUNBUFFERED"] = "1"
                env.setdefault("HYPEROPT_RESULTS_DIR", str(base.ARTIFACT_ROOT / "results"))
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
                with log_path.open("a", encoding="utf-8") as fh:
                    proc = subprocess.Popen(cmd, cwd=base.EXP_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT)
                active[gpu_id] = {"proc": proc, "row": row, "log_path": str(log_path)}
                print(f"[launch] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} model={row['model_label']}")

            done_gpus: List[str] = []
            for gpu_id, slot in active.items():
                rc = slot["proc"].poll()
                if rc is None:
                    continue
                done_gpus.append(gpu_id)
                row = slot["row"]
                print(f"[done] dataset={row['dataset']} gpu={gpu_id} run_phase={row['run_phase']} rc={rc}")
                rec = base._get_result_row_from_log_or_scan(
                    dataset=row["dataset"],
                    run_phase=str(row["run_phase"]),
                    log_path=Path(str(slot["log_path"])),
                    retries=4,
                    sleep_sec=0.75,
                )
                run_best = None
                test_mrr = None
                n_completed = None
                interrupted = None
                result_path = ""
                special_ok = False
                if rec:
                    run_best = sg._metric(rec.get("best_mrr"))
                    test_mrr = sg._metric(rec.get("test_mrr"))
                    n_completed = int(rec.get("n_completed", 0) or 0)
                    interrupted = bool(rec.get("interrupted", False))
                    result_path = str(rec.get("path", "") or "")
                    if result_path:
                        special_ok, detail = base._verify_special_from_result(result_path)
                        print(f"[logging-check] run={row['run_phase']} {detail} result={result_path}")
                        base._mirror_logging_bundle(row, result_path)
                summary_row = {
                    "model": row["model_label"],
                    "global_best_valid_mrr20": "",
                    "global_best_test_mrr20": "",
                    "model_best_valid_mrr20": "",
                    "model_best_test_mrr20": "",
                    "run_best_valid_mrr20": "" if run_best is None else f"{float(run_best):.6f}",
                    "run_best_test_mrr20": "" if test_mrr is None else f"{float(test_mrr):.6f}",
                    "run_phase": row["run_phase"],
                    "run_id": row["run_id"],
                    "dataset": row["dataset"],
                    "hparam_id": row["hparam_id"],
                    "seed_id": row["seed_id"],
                    "gpu_id": gpu_id,
                    "status": "run_complete" if int(rc) == 0 else "run_fail",
                    "n_completed": "" if n_completed is None else int(n_completed),
                    "interrupted": "" if interrupted is None else bool(interrupted),
                    "special_ok": bool(special_ok),
                    "candidate_id": row["candidate_id"],
                    "candidate_source": row["candidate_source"],
                    "source_stage": row["source_stage"],
                    "transfer_from_dataset": row["transfer_from_dataset"],
                    "result_path": result_path,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                }
                base._append_summary_row(summary_paths[row["dataset"]], summary_row)
                if int(rc) != 0:
                    raise RuntimeError(f"run failed: dataset={row['dataset']} run_phase={row['run_phase']} rc={rc}")

            for gpu_id in done_gpus:
                active.pop(gpu_id, None)
            if not shared_queue and not active:
                break
            time.sleep(1)
    except Exception:
        base._terminate_active(active)
        raise
    return 0


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        _apply_smoke_mode(args)
    elif args.fast_screen:
        _apply_fast_screen_mode(args)
    sg._check_runtime_models(list(dict.fromkeys(m.lower() for _, m in TARGET_COMBOS)))
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
