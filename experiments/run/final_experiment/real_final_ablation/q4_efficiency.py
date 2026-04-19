#!/usr/bin/env python3
"""Q4 efficiency benchmark for representative datasets."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    LOG_ROOT,
    REPO_ROOT,
    build_eval_config_from_result_payload,
    canonical_stage_maps,
    common_arg_parser,
    ensure_dir,
    now_utc,
    parse_csv_list,
    selected_candidates_from_args,
    write_json,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_benchmark_datasets(dataset_text: str, explicit_text: str) -> list[tuple[str, str]]:
    dynamic_pref = ["beauty", "KuaiRecLargeStrictPosV2_0.2", "retail_rocket"]
    stable_pref = ["movielens1m", "lastfm0.03", "foursquare"]
    explicit = parse_csv_list(explicit_text)
    if len(explicit) >= 2:
        return [("dynamic", explicit[0]), ("stable", explicit[1])]
    requested = parse_csv_list(dataset_text)
    dynamic = next((d for d in dynamic_pref if d in requested), dynamic_pref[0])
    stable = next((d for d in stable_pref if d in requested and d != dynamic), stable_pref[0])
    return [("dynamic", dynamic), ("stable", stable)]


def _route_sparse_overrides() -> dict[str, Any]:
    cfg = canonical_stage_maps()
    cfg.update({"topk_scope_mode": "per_group", "group_top_k": 3, "expert_top_k": 2, "moe_top_k": 0})
    return cfg


def _route_dense_overrides() -> dict[str, Any]:
    cfg = canonical_stage_maps()
    cfg.update(
        {
            "stage_compute_mode": {stage: "dense_plain" for stage in ("macro", "mid", "micro")},
            "stage_router_mode": {stage: "none" for stage in ("macro", "mid", "micro")},
            "stage_router_source": {stage: "hidden" for stage in ("macro", "mid", "micro")},
            "stage_feature_injection": {stage: "none" for stage in ("macro", "mid", "micro")},
            "router_use_hidden": False,
            "router_use_feature": False,
            "expert_use_feature": False,
            "group_top_k": 0,
            "expert_top_k": 0,
            "moe_top_k": 0,
        }
    )
    return cfg


def _apply_shared_protocol(cfg: dict[str, Any], *, dataset: str, gpu_id: int, epochs: int, train_batch: int, eval_batch: int) -> dict[str, Any]:
    out = deepcopy(cfg)
    out["dataset"] = dataset
    out["data_path"] = str((REPO_ROOT / "Datasets" / "processed" / "feature_added_v4").resolve())
    out["benchmark_filename"] = ["train", "valid", "test"]
    out["eval_args"] = {"group_by": "user", "order": "TO", "mode": "full"}
    out["eval_sampling"] = {"mode": "full", "auto_full_threshold": 999999999}
    out["history_input_mode"] = "session_only"
    out["history_group_field"] = "user_id"
    out["target_group_field"] = "session_id"
    out["history_eval_policy"] = "strict_train_prefix"
    out["feature_mode"] = "full_v4"
    out["eval_mode"] = "session_fixed"
    out["epochs"] = int(epochs)
    out["train_batch_size"] = int(train_batch)
    out["eval_batch_size"] = int(eval_batch)
    out["stopping_step"] = int(epochs) + 5
    out["eval_every"] = int(epochs)
    out["gpu_id"] = int(gpu_id)
    out["use_gpu"] = True
    out["log_wandb"] = False
    out["show_progress"] = False
    out["save_dataset"] = False
    out["save_dataloaders"] = False
    out["enable_tf32"] = True
    out["special_logging"] = False
    out["fmoe_debug_logging"] = False
    out["fmoe_diag_logging"] = False
    out["fmoe_special_logging"] = False
    out["fmoe_feature_family_ablation_logging"] = False
    out["fmoe_feature_ablation_logging"] = False
    out["exclude_unseen_target_from_main_eval"] = True
    out["log_unseen_target_metrics"] = False
    return out


def _named_model_config(dataset: str, model_name: str, route_cfg: dict[str, Any], *, gpu_id: int, epochs: int, train_batch: int, eval_batch: int) -> dict[str, Any]:
    from hydra_utils import configure_eval_sampling, enforce_v4_feature_mode, load_hydra_config
    from omegaconf import OmegaConf

    cfg = load_hydra_config(
        config_dir=REPO_ROOT / "experiments" / "configs",
        config_name="config",
        overrides=[f"model={model_name}", f"dataset={dataset}"],
    )
    cfg_omega = OmegaConf.create(cfg)
    cfg_omega = configure_eval_sampling(cfg_omega)
    cfg_dict = OmegaConf.to_container(cfg_omega, resolve=True)
    enforce_v4_feature_mode(cfg_dict)
    for key in ("embedding_size", "hidden_size", "num_heads", "hidden_dropout_prob"):
        if key in route_cfg:
            cfg_dict[key] = route_cfg[key]
    return _apply_shared_protocol(
        cfg_dict,
        dataset=dataset,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
    )


def _count_active_params(model: Any, cfg: dict[str, Any]) -> int:
    total = sum(p.numel() for p in model.parameters())
    expert_total = sum(p.numel() for name, p in model.named_parameters() if "expert" in name.lower())
    if expert_total <= 0:
        return total
    n_experts = int(getattr(model, "_stage_n_experts", 0) or 0)
    if n_experts <= 0 and hasattr(model, "stage_executor") and hasattr(model.stage_executor, "stage_n_experts"):
        try:
            n_experts = int(model.stage_executor.stage_n_experts())
        except Exception:
            n_experts = 0
    if n_experts <= 0:
        return total
    if str(cfg.get("model", "")).lower() not in {"featuredmoe_n3", "featured_moe_n3"}:
        return total
    group_top_k = int(cfg.get("group_top_k", 0) or 0)
    expert_top_k = int(cfg.get("expert_top_k", 0) or 0)
    moe_top_k = int(cfg.get("moe_top_k", 0) or 0)
    if group_top_k > 0 and expert_top_k > 0:
        active_experts = min(n_experts, group_top_k * expert_top_k)
    elif moe_top_k > 0:
        active_experts = min(n_experts, moe_top_k)
    elif expert_top_k > 0:
        active_experts = min(n_experts, expert_top_k)
    else:
        active_experts = n_experts
    ratio = float(active_experts) / float(max(n_experts, 1))
    non_expert_total = total - expert_total
    return int(round(non_expert_total + expert_total * ratio))


def _measure_one(config: dict[str, Any]) -> dict[str, Any]:
    import recbole_patch  # noqa: F401
    import torch
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_seed
    from recbole.utils import utils as recbole_utils

    model_name = str(config["model"])
    dataset_name = str(config["dataset"])
    seed = int(config.get("seed", 42) or 42)
    init_seed(seed, bool(config.get("reproducibility", True)))
    cfg = Config(model=model_name, dataset=dataset_name, config_dict=config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    dataset = create_dataset(cfg)
    train_data, valid_data, test_data = data_preparation(cfg, dataset)
    build_sec = time.perf_counter() - t0

    model_cls = recbole_utils.get_model(cfg["model"])
    model = model_cls(cfg, train_data.dataset).to(device)
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    t1 = time.perf_counter()
    trainer.fit(train_data, valid_data, verbose=False, saved=False, show_progress=False)
    train_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    infer_sec = time.perf_counter() - t2

    total_params = int(sum(p.numel() for p in model.parameters()))
    active_params = int(_count_active_params(model, cfg))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "build_sec": build_sec,
        "train_sec": train_sec,
        "infer_sec": infer_sec,
        "total_params": total_params,
        "active_params": active_params,
    }


def _model_specs(dataset: str, route_cfg: dict[str, Any], *, gpu_id: int, epochs: int, train_batch: int, eval_batch: int, include_fame: bool) -> list[tuple[str, dict[str, Any]]]:
    route_sparse = _apply_shared_protocol(
        {**deepcopy(route_cfg), **_route_sparse_overrides()},
        dataset=dataset,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
    )
    route_dense = _apply_shared_protocol(
        {**deepcopy(route_cfg), **_route_dense_overrides()},
        dataset=dataset,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
    )
    specs = [
        ("SASRec", _named_model_config(dataset, "sasrec", route_cfg, gpu_id=gpu_id, epochs=epochs, train_batch=train_batch, eval_batch=eval_batch)),
        ("RouteRec-dense", route_dense),
        ("RouteRec-sparse-final", route_sparse),
    ]
    if include_fame:
        specs.append(
            ("FAME", _named_model_config(dataset, "fame", route_cfg, gpu_id=gpu_id, epochs=epochs, train_batch=train_batch, eval_batch=eval_batch))
        )
    return specs


def main() -> int:
    parser = common_arg_parser("Q4 efficiency benchmark", question="q4")
    parser.add_argument("--benchmark-datasets", default="")
    parser.add_argument("--benchmark-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--include-fame", action="store_true")
    args = parser.parse_args()

    benchmark_scopes = _resolve_benchmark_datasets(args.datasets, args.benchmark_datasets)
    candidates = selected_candidates_from_args(args)
    by_dataset = {}
    for cand in candidates:
        by_dataset.setdefault(cand.dataset, cand)
    gpu_id = int((parse_csv_list(args.gpus) or ["0"])[0])

    summary_rows: list[dict[str, Any]] = []
    previous_rows = {
        (str(row.get("dataset_scope", "")), str(row.get("model_name", ""))): row
        for row in _read_csv(LOG_ROOT / "q4" / "summary.csv")
        if str(row.get("status", "")).lower() == "ok"
    }

    for dataset_scope, dataset_name in benchmark_scopes:
        candidate = by_dataset.get(dataset_name)
        if candidate is None:
            raise RuntimeError(f"No base candidate available for Q4 dataset={dataset_name}")
        route_cfg = build_eval_config_from_result_payload(candidate.payload)
        specs = _model_specs(
            dataset_name,
            route_cfg,
            gpu_id=gpu_id,
            epochs=int(args.benchmark_epochs),
            train_batch=int(args.train_batch_size),
            eval_batch=int(args.eval_batch_size),
            include_fame=bool(args.include_fame),
        )
        raw_rows: list[dict[str, Any]] = []
        for model_name, cfg in specs:
            key = (dataset_scope, model_name)
            if bool(args.resume_from_logs) and key in previous_rows:
                raw_rows.append(dict(previous_rows[key]))
                continue
            if args.dry_run:
                raw_rows.append(
                    {
                        "question": "q4",
                        "dataset_scope": dataset_scope,
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "model": str(cfg.get("model", "")),
                        "status": "planned",
                        "error": "",
                        "benchmark_epochs": int(args.benchmark_epochs),
                        "build_sec": "",
                        "train_sec": "",
                        "infer_sec": "",
                        "total_params": "",
                        "active_params": "",
                        "timestamp_utc": now_utc(),
                    }
                )
                continue
            try:
                metrics = _measure_one(cfg)
                raw_rows.append(
                    {
                        "question": "q4",
                        "dataset_scope": dataset_scope,
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "model": str(cfg.get("model", "")),
                        "status": "ok",
                        "error": "",
                        "benchmark_epochs": int(args.benchmark_epochs),
                        **metrics,
                        "timestamp_utc": now_utc(),
                    }
                )
            except Exception as exc:
                raw_rows.append(
                    {
                        "question": "q4",
                        "dataset_scope": dataset_scope,
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "model": str(cfg.get("model", "")),
                        "status": "error",
                        "error": str(exc),
                        "benchmark_epochs": int(args.benchmark_epochs),
                        "build_sec": "",
                        "train_sec": "",
                        "infer_sec": "",
                        "total_params": "",
                        "active_params": "",
                        "timestamp_utc": now_utc(),
                    }
                )
        sasrec_row = next((row for row in raw_rows if row.get("model_name") == "SASRec" and row.get("status") == "ok"), None)
        sasrec_train = float(sasrec_row.get("train_sec", 0.0) or 0.0) if sasrec_row else 0.0
        sasrec_infer = float(sasrec_row.get("infer_sec", 0.0) or 0.0) if sasrec_row else 0.0
        for row in raw_rows:
            if row.get("status") == "ok" and sasrec_train > 0 and sasrec_infer > 0:
                row["train_time_ratio"] = float(row["train_sec"]) / sasrec_train
                row["infer_time_ratio"] = float(row["infer_sec"]) / sasrec_infer
            else:
                row["train_time_ratio"] = ""
                row["infer_time_ratio"] = ""
            summary_rows.append(row)

    q4_root = ensure_dir(LOG_ROOT / "q4")
    _write_csv(q4_root / "summary.csv", summary_rows)
    write_json(
        q4_root / "manifest.json",
        {
            "generated_at": now_utc(),
            "question": "q4",
            "benchmark_scopes": benchmark_scopes,
            "rows": summary_rows,
        },
    )
    print(f"[q4] summary -> {q4_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
