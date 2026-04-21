#!/usr/bin/env python3
"""Deprecated wrapper for the legacy efficiency-first Q4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    legacy_script = Path(__file__).resolve().parent / "legacy" / "q4_efficiency.py"
    print(f"[deprecated] q4_efficiency.py moved to {legacy_script}", flush=True)
    cmd = [sys.executable, str(legacy_script), *sys.argv[1:]]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Q4 efficiency sweep with active-parameter matching.

This script stays intentionally narrow: it does not do broad tuning.
It builds a small, diverse RouteRec screening portfolio around the selected
base candidate, measures metric/parameter/time together, and exports matched
comparison rows for the paper.
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common import (  # noqa: E402
    DATA_ROOT,
    LOG_ROOT,
    REPO_ROOT,
    build_eval_config_from_result_payload,
    canonical_stage_maps,
    common_arg_parser,
    ensure_dir,
    now_utc,
    parse_csv_ints,
    parse_csv_list,
    selected_candidates_from_args,
    write_json,
)


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"recbole\.data\.dataset\.dataset",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*GradScaler.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"ray\._private\.parameter",
    message=r".*pkg_resources is deprecated as an API.*",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if path.exists():
            path.unlink()
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def _cfg_get(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def _round_width(value: float, *, step: int, minimum: int) -> int:
    rounded = int(round(float(value) / float(step)) * step)
    return max(minimum, rounded)


def _resolve_benchmark_datasets(dataset_text: str, explicit_text: str) -> list[tuple[str, str]]:
    explicit = parse_csv_list(explicit_text)
    if explicit:
        out: list[tuple[str, str]] = []
        for idx, dataset in enumerate(explicit):
            scope = "primary" if idx == 0 else f"supporting_{idx}"
            out.append((scope, dataset))
        return out
    requested = parse_csv_list(dataset_text)
    preferred = ["KuaiRecLargeStrictPosV2_0.2", "foursquare"]
    resolved: list[str] = []
    for dataset in preferred:
        if dataset in requested and dataset not in resolved:
            resolved.append(dataset)
    if not resolved:
        resolved.append("KuaiRecLargeStrictPosV2_0.2")
    if len(resolved) == 1 and "foursquare" in requested:
        resolved.append("foursquare")
    return [("primary" if idx == 0 else f"supporting_{idx}", dataset) for idx, dataset in enumerate(resolved)]


def _route_sparse_overrides(group_top_k: int = 3, expert_top_k: int = 2) -> dict[str, Any]:
    cfg = canonical_stage_maps()
    cfg.update(
        {
            "topk_scope_mode": "per_group",
            "group_top_k": int(group_top_k),
            "expert_top_k": int(expert_top_k),
            "moe_top_k": 0,
        }
    )
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


def _apply_shared_protocol(
    cfg: dict[str, Any], *, dataset: str, gpu_id: int, epochs: int, train_batch: int, eval_batch: int
) -> dict[str, Any]:
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


def _set_epoch_budget(config: dict[str, Any], epochs: int) -> dict[str, Any]:
    updated = deepcopy(config)
    updated["epochs"] = int(epochs)
    updated["stopping_step"] = int(epochs) + 5
    updated["eval_every"] = int(epochs)
    return updated


def _named_model_config(
    dataset: str,
    model_name: str,
    route_cfg: dict[str, Any],
    *,
    gpu_id: int,
    epochs: int,
    train_batch: int,
    eval_batch: int,
    hidden_size: int | None = None,
) -> dict[str, Any]:
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
    target_hidden = int(hidden_size or route_cfg.get("hidden_size") or route_cfg.get("embedding_size") or 128)
    base_inner_ratio = _safe_float(route_cfg.get("inner_size"), default=float(target_hidden)) / float(
        max(_safe_int(route_cfg.get("hidden_size"), target_hidden), 1)
    )
    target_inner = max(target_hidden, _round_width(target_hidden * base_inner_ratio, step=32, minimum=target_hidden))
    for key in ("embedding_size", "hidden_size"):
        if key in cfg_dict:
            cfg_dict[key] = target_hidden
    for key in ("inner_size", "attribute_hidden_size"):
        if key in cfg_dict:
            cfg_dict[key] = target_inner if key == "inner_size" else target_hidden
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
    model_name = str(_cfg_get(cfg, "model", "")).lower()
    if model_name not in {"featuredmoe_n3", "featured_moe_n3"}:
        return total
    group_top_k = int(_cfg_get(cfg, "group_top_k", 0) or 0)
    expert_top_k = int(_cfg_get(cfg, "expert_top_k", 0) or 0)
    moe_top_k = int(_cfg_get(cfg, "moe_top_k", 0) or 0)
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


def _maybe_sync_cuda(torch_module: Any) -> None:
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
    except Exception:
        return


def _extract_metric(metrics: dict[str, Any], key: str) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    candidates = [key, key.lower(), key.upper()]
    if "@" in key:
        metric_name, suffix = key.split("@", 1)
        candidates.extend(
            [
                f"{metric_name.lower()}@{suffix}",
                f"{metric_name.upper()}@{suffix}",
                f"{metric_name.capitalize()}@{suffix}",
            ]
        )
    for candidate in candidates:
        if candidate in metrics:
            return _safe_float(metrics.get(candidate))
    return 0.0


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
    _maybe_sync_cuda(torch)
    build_sec = time.perf_counter() - t0

    model_cls = recbole_utils.get_model(cfg["model"])
    model = model_cls(cfg, train_data.dataset).to(device)
    trainer = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, model)

    _maybe_sync_cuda(torch)
    t1 = time.perf_counter()
    fit_output = trainer.fit(train_data, valid_data, verbose=False, saved=False, show_progress=False)
    _maybe_sync_cuda(torch)
    train_sec = time.perf_counter() - t1

    _maybe_sync_cuda(torch)
    t2 = time.perf_counter()
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
    _maybe_sync_cuda(torch)
    infer_sec = time.perf_counter() - t2

    best_valid_score = 0.0
    best_valid_result: dict[str, Any] = {}
    if isinstance(fit_output, tuple) and len(fit_output) >= 2:
        best_valid_score = _safe_float(fit_output[0])
        if isinstance(fit_output[1], dict):
            best_valid_result = fit_output[1]
    elif hasattr(trainer, "best_valid_score"):
        best_valid_score = _safe_float(getattr(trainer, "best_valid_score", 0.0))
        best_valid_result = dict(getattr(trainer, "best_valid_result", {}) or {})

    total_params = int(sum(p.numel() for p in model.parameters()))
    active_params = int(_count_active_params(model, cfg))
    epochs = int(config.get("epochs", 1) or 1)
    avg_train_epoch_sec = float(train_sec) / float(max(epochs, 1))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "build_sec": build_sec,
        "train_sec": train_sec,
        "avg_train_epoch_sec": avg_train_epoch_sec,
        "infer_sec": infer_sec,
        "total_params": total_params,
        "active_params": active_params,
        "best_valid_score": best_valid_score,
        "best_valid_mrr20": _extract_metric(best_valid_result, "mrr@20"),
        "best_valid_ndcg10": _extract_metric(best_valid_result, "ndcg@10"),
        "best_valid_hit10": _extract_metric(best_valid_result, "hit@10"),
        "test_mrr20": _extract_metric(test_result, "mrr@20"),
        "test_ndcg10": _extract_metric(test_result, "ndcg@10"),
        "test_hit10": _extract_metric(test_result, "hit@10"),
    }


def _route_config_with_width(
    route_cfg: dict[str, Any],
    *,
    hidden_size: int,
    group_top_k: int,
    expert_top_k: int,
    dense: bool,
) -> dict[str, Any]:
    cfg = deepcopy(route_cfg)
    hidden = int(hidden_size)
    head_count = int(cfg.get("num_heads") or cfg.get("n_heads") or 4)
    step = max(32, head_count)
    hidden = _round_width(hidden, step=step, minimum=max(step, 64))
    inner_ratio = _safe_float(cfg.get("inner_size"), default=float(hidden)) / float(
        max(_safe_int(cfg.get("hidden_size"), hidden), 1)
    )
    cfg["hidden_size"] = hidden
    cfg["embedding_size"] = hidden
    if "attribute_hidden_size" in cfg:
        cfg["attribute_hidden_size"] = hidden
    if "inner_size" in cfg:
        cfg["inner_size"] = max(hidden, _round_width(hidden * inner_ratio, step=32, minimum=hidden))
    cfg.update(_route_dense_overrides() if dense else _route_sparse_overrides(group_top_k, expert_top_k))
    return cfg


def _portfolio_specs(
    dataset: str,
    route_cfg: dict[str, Any],
    *,
    gpu_id: int,
    epochs: int,
    train_batch: int,
    eval_batch: int,
    include_fame: bool,
    width_multipliers: list[float],
    group_topk_grid: list[int],
    expert_topk_grid: list[int],
    max_route_screen_runs: int,
) -> list[dict[str, Any]]:
    base_hidden = int(route_cfg.get("hidden_size") or route_cfg.get("embedding_size") or 128)
    hidden_values = sorted({_round_width(base_hidden * mult, step=32, minimum=64) for mult in (width_multipliers or [1.0])})
    specs: list[dict[str, Any]] = []

    def add_spec(
        *,
        setting_key: str,
        setting_label: str,
        model_name: str,
        row_type: str,
        cfg: dict[str, Any],
        hidden_size: int,
        group_top_k: int,
        expert_top_k: int,
        selection_tag: str = "",
    ) -> None:
        specs.append(
            {
                "setting_key": setting_key,
                "setting_label": setting_label,
                "model_name": model_name,
                "row_type": row_type,
                "selection_tag": selection_tag,
                "hidden_size": hidden_size,
                "embedding_size": _safe_int(cfg.get("embedding_size"), hidden_size),
                "inner_size": _safe_int(cfg.get("inner_size"), hidden_size),
                "group_top_k": int(group_top_k),
                "expert_top_k": int(expert_top_k),
                "moe_top_k": _safe_int(cfg.get("moe_top_k"), 0),
                "topk_scope_mode": str(cfg.get("topk_scope_mode", "")),
                "config": cfg,
            }
        )

    sasrec_cfg = _named_model_config(
        dataset,
        "sasrec",
        route_cfg,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
        hidden_size=base_hidden,
    )
    add_spec(
        setting_key="sasrec_width_matched",
        setting_label="SASRec (width-matched)",
        model_name="SASRec",
        row_type="reference",
        cfg=sasrec_cfg,
        hidden_size=base_hidden,
        group_top_k=0,
        expert_top_k=0,
        selection_tag="baseline",
    )

    dense_cfg = _apply_shared_protocol(
        _route_config_with_width(route_cfg, hidden_size=base_hidden, group_top_k=0, expert_top_k=0, dense=True),
        dataset=dataset,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
    )
    add_spec(
        setting_key="routerec_dense_base",
        setting_label="RouteRec dense reference",
        model_name="RouteRec-dense",
        row_type="reference",
        cfg=dense_cfg,
        hidden_size=base_hidden,
        group_top_k=0,
        expert_top_k=0,
        selection_tag="dense_reference",
    )

    sparse_base_cfg = _apply_shared_protocol(
        _route_config_with_width(route_cfg, hidden_size=base_hidden, group_top_k=3, expert_top_k=2, dense=False),
        dataset=dataset,
        gpu_id=gpu_id,
        epochs=epochs,
        train_batch=train_batch,
        eval_batch=eval_batch,
    )
    add_spec(
        setting_key="routerec_sparse_base",
        setting_label="RouteRec sparse final",
        model_name="RouteRec-sparse-final",
        row_type="reference",
        cfg=sparse_base_cfg,
        hidden_size=base_hidden,
        group_top_k=3,
        expert_top_k=2,
        selection_tag="final_sparse",
    )

    if include_fame:
        fame_cfg = _named_model_config(
            dataset,
            "fame",
            route_cfg,
            gpu_id=gpu_id,
            epochs=epochs,
            train_batch=train_batch,
            eval_batch=eval_batch,
            hidden_size=base_hidden,
        )
        add_spec(
            setting_key="fame_width_matched",
            setting_label="FAME (width-matched)",
            model_name="FAME",
            row_type="reference",
            cfg=fame_cfg,
            hidden_size=base_hidden,
            group_top_k=0,
            expert_top_k=0,
            selection_tag="baseline",
        )

    screen_candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    for hidden_size in hidden_values:
        for group_top_k in group_topk_grid:
            for expert_top_k in expert_topk_grid:
                if group_top_k <= 0 or expert_top_k <= 0:
                    continue
                cfg = _apply_shared_protocol(
                    _route_config_with_width(
                        route_cfg,
                        hidden_size=hidden_size,
                        group_top_k=group_top_k,
                        expert_top_k=expert_top_k,
                        dense=False,
                    ),
                    dataset=dataset,
                    gpu_id=gpu_id,
                    epochs=epochs,
                    train_batch=train_batch,
                    eval_batch=eval_batch,
                )
                screen_candidates.append((hidden_size, group_top_k, expert_top_k, cfg))

    screen_candidates.sort(key=lambda item: (item[0], item[1] * item[2], item[1], item[2]))
    if max_route_screen_runs > 0 and len(screen_candidates) > max_route_screen_runs:
        picked: list[tuple[int, int, int, dict[str, Any]]] = []
        for idx in range(max_route_screen_runs):
            pos = 0 if max_route_screen_runs == 1 else int(round(idx * (len(screen_candidates) - 1) / float(max_route_screen_runs - 1)))
            candidate = screen_candidates[pos]
            if candidate not in picked:
                picked.append(candidate)
        screen_candidates = picked

    for hidden_size, group_top_k, expert_top_k, cfg in screen_candidates:
        add_spec(
            setting_key=f"route_screen_w{hidden_size}_g{group_top_k}e{expert_top_k}",
            setting_label=f"RouteRec sparse w={hidden_size} g{group_top_k}e{expert_top_k}",
            model_name=f"RouteRec-scan-g{group_top_k}e{expert_top_k}",
            row_type="screen",
            cfg=cfg,
            hidden_size=hidden_size,
            group_top_k=group_top_k,
            expert_top_k=expert_top_k,
        )
    return specs


def _row_resume_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row.get("dataset", "")),
        str(row.get("run_phase", "")),
        str(row.get("setting_key", "")),
        str(row.get("model_name", "")),
        _safe_int(row.get("benchmark_epochs"), 0),
    )


def _apply_learning_rate(config: dict[str, Any], learning_rate: float) -> dict[str, Any]:
    tuned = deepcopy(config)
    tuned["learning_rate"] = float(learning_rate)
    return tuned


def _candidate_learning_rates(config: dict[str, Any], multipliers: list[float]) -> list[float]:
    base_lr = _safe_float(config.get("learning_rate"), 0.0)
    if base_lr <= 0:
        base_lr = 1e-3
    candidates = [max(1e-6, base_lr * float(multiplier)) for multiplier in multipliers]
    deduped: list[float] = []
    seen: set[float] = set()
    for lr in candidates:
        rounded = round(lr, 12)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(lr)
    return deduped or [base_lr]


def _measure_spec_with_lr_scan(spec: dict[str, Any], lr_multipliers: list[float]) -> dict[str, Any]:
    candidates = _candidate_learning_rates(spec["config"], lr_multipliers)
    best_metrics: dict[str, Any] | None = None
    best_lr = candidates[0]
    for learning_rate in candidates:
        metrics = _measure_one(_apply_learning_rate(spec["config"], learning_rate))
        if best_metrics is None:
            best_metrics = metrics
            best_lr = learning_rate
            continue
        current_key = (
            _safe_float(metrics.get("best_valid_mrr20")),
            _safe_float(metrics.get("best_valid_ndcg10")),
            _safe_float(metrics.get("test_mrr20")),
            -_safe_float(metrics.get("avg_train_epoch_sec"), 0.0),
        )
        best_key = (
            _safe_float(best_metrics.get("best_valid_mrr20")),
            _safe_float(best_metrics.get("best_valid_ndcg10")),
            _safe_float(best_metrics.get("test_mrr20")),
            -_safe_float(best_metrics.get("avg_train_epoch_sec"), 0.0),
        )
        if current_key > best_key:
            best_metrics = metrics
            best_lr = learning_rate
    return {
        **(best_metrics or {}),
        "chosen_lr": best_lr,
        "lr_scan_values": ",".join(f"{lr:.8g}" for lr in candidates),
        "lr_probe_count": len(candidates),
    }


def _measure_specs(
    specs: list[dict[str, Any]],
    *,
    dataset_scope: str,
    dataset_name: str,
    benchmark_epochs: int,
    lr_multipliers: list[float],
    run_phase: str,
    resume_from_logs: bool,
    previous_rows: dict[tuple[str, str, str, int], dict[str, str]],
    dry_run: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        base_row = {
            "question": "q4",
            "dataset_scope": dataset_scope,
            "dataset": dataset_name,
            "model_name": spec["model_name"],
            "model": str(spec["config"].get("model", "")),
            "row_type": spec["row_type"],
            "run_phase": run_phase,
            "setting_key": spec["setting_key"],
            "setting_label": spec["setting_label"],
            "selection_tag": spec.get("selection_tag", ""),
            "benchmark_epochs": int(benchmark_epochs),
            "lr_scan_values": ",".join(f"{lr:.8g}" for lr in _candidate_learning_rates(spec["config"], lr_multipliers)),
            "hidden_size": int(spec["hidden_size"]),
            "embedding_size": int(spec["embedding_size"]),
            "inner_size": int(spec["inner_size"]),
            "group_top_k": int(spec["group_top_k"]),
            "expert_top_k": int(spec["expert_top_k"]),
            "moe_top_k": int(spec["moe_top_k"]),
            "topk_scope_mode": spec["topk_scope_mode"],
        }
        key = _row_resume_key(base_row)
        if resume_from_logs and key in previous_rows:
            resumed = dict(previous_rows[key])
            resumed["timestamp_utc"] = now_utc()
            rows.append(resumed)
            continue
        if dry_run:
            rows.append(
                {
                    **base_row,
                    "status": "planned",
                    "error": "",
                    "build_sec": "",
                    "train_sec": "",
                    "avg_train_epoch_sec": "",
                    "infer_sec": "",
                    "total_params": "",
                    "active_params": "",
                    "best_valid_score": "",
                    "best_valid_mrr20": "",
                    "best_valid_ndcg10": "",
                    "best_valid_hit10": "",
                    "test_mrr20": "",
                    "test_ndcg10": "",
                    "test_hit10": "",
                    "chosen_lr": "",
                    "lr_probe_count": len(_candidate_learning_rates(spec["config"], lr_multipliers)),
                    "timestamp_utc": now_utc(),
                }
            )
            continue
        try:
            metrics = _measure_spec_with_lr_scan(spec, lr_multipliers)
            rows.append(
                {
                    **base_row,
                    **metrics,
                    "status": "ok",
                    "error": "",
                    "timestamp_utc": now_utc(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error": str(exc),
                    "build_sec": "",
                    "train_sec": "",
                    "avg_train_epoch_sec": "",
                    "infer_sec": "",
                    "total_params": "",
                    "active_params": "",
                    "best_valid_score": "",
                    "best_valid_mrr20": "",
                    "best_valid_ndcg10": "",
                    "best_valid_hit10": "",
                    "test_mrr20": "",
                    "test_ndcg10": "",
                    "test_hit10": "",
                    "chosen_lr": "",
                    "lr_probe_count": len(_candidate_learning_rates(spec["config"], lr_multipliers)),
                    "timestamp_utc": now_utc(),
                }
            )
    return rows


def _augment_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dataset.setdefault(str(row.get("dataset", "")), []).append(row)

    for dataset_rows in by_dataset.values():
        sasrec = next(
            (
                row
                for row in dataset_rows
                if str(row.get("model_name", "")) == "SASRec" and str(row.get("status", "")).lower() == "ok"
            ),
            None,
        )
        sasrec_avg_train = _safe_float((sasrec or {}).get("avg_train_epoch_sec"), 0.0)
        sasrec_infer = _safe_float((sasrec or {}).get("infer_sec"), 0.0)
        sasrec_active = _safe_float((sasrec or {}).get("active_params"), 0.0)
        for row in dataset_rows:
            if str(row.get("status", "")).lower() == "ok" and sasrec_avg_train > 0 and sasrec_infer > 0:
                row["train_time_ratio"] = _safe_float(row.get("avg_train_epoch_sec")) / sasrec_avg_train
                row["infer_time_ratio"] = _safe_float(row.get("infer_sec")) / sasrec_infer
            else:
                row["train_time_ratio"] = ""
                row["infer_time_ratio"] = ""
            if str(row.get("status", "")).lower() == "ok" and sasrec_active > 0:
                row["active_param_ratio_vs_sasrec"] = _safe_float(row.get("active_params")) / sasrec_active
                row["total_param_ratio_vs_sasrec"] = _safe_float(row.get("total_params")) / sasrec_active
            else:
                row["active_param_ratio_vs_sasrec"] = ""
                row["total_param_ratio_vs_sasrec"] = ""
    return rows


def _pick_route_match(
    candidates: list[dict[str, Any]],
    *,
    target_active: float,
    active_match_tolerance: float,
    exclude_keys: set[str],
) -> dict[str, Any] | None:
    usable = [row for row in candidates if str(row.get("setting_key", "")) not in exclude_keys]
    if not usable:
        return None
    within_tol = [
        row
        for row in usable
        if target_active > 0 and abs(_safe_float(row.get("active_params")) - target_active) / target_active <= active_match_tolerance
    ]
    pool = within_tol or usable
    pool.sort(
        key=lambda row: (
            abs(_safe_float(row.get("active_params")) - target_active),
            -_safe_float(row.get("test_mrr20")),
            _safe_float(row.get("avg_train_epoch_sec"), default=1e18),
        )
    )
    return pool[0] if pool else None


def _build_matched_rows(rows: list[dict[str, Any]], *, active_match_tolerance: float) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("status", "")).lower() == "ok":
            by_dataset.setdefault(str(row.get("dataset", "")), []).append(row)

    for dataset_rows in by_dataset.values():
        sasrec = next((row for row in dataset_rows if str(row.get("model_name")) == "SASRec"), None)
        fame = next((row for row in dataset_rows if str(row.get("model_name")) == "FAME"), None)
        dense_ref = next((row for row in dataset_rows if str(row.get("setting_key")) == "routerec_dense_base"), None)
        sparse_base = next((row for row in dataset_rows if str(row.get("setting_key")) == "routerec_sparse_base"), None)
        route_screen = [row for row in dataset_rows if str(row.get("row_type", "")) == "screen"]
        route_pool = route_screen + ([sparse_base] if sparse_base is not None else [])
        route_pool = [row for row in route_pool if row is not None]
        route_pool.sort(
            key=lambda row: (
                -_safe_float(row.get("test_mrr20")),
                _safe_float(row.get("active_params"), default=1e18),
                _safe_float(row.get("avg_train_epoch_sec"), default=1e18),
            )
        )
        if sasrec is not None:
            matched.append({**dict(sasrec), "match_rule": "reference_sasrec", "selected_for_main": True})
        if fame is not None:
            matched.append({**dict(fame), "match_rule": "reference_fame", "selected_for_main": True})
        if dense_ref is not None:
            matched.append({**dict(dense_ref), "match_rule": "reference_dense", "selected_for_main": True})
        if not route_pool:
            continue

        exclude_keys: set[str] = set()
        best_route = route_pool[0]
        matched.append({**dict(best_route), "match_rule": "best_route_metric", "selected_for_main": True})
        exclude_keys.add(str(best_route.get("setting_key", "")))

        target_active = _safe_float((sasrec or {}).get("active_params"), 0.0)
        active_match = _pick_route_match(
            route_pool,
            target_active=target_active,
            active_match_tolerance=active_match_tolerance,
            exclude_keys=exclude_keys,
        )
        if active_match is not None:
            matched.append({**dict(active_match), "match_rule": "active_match_to_sasrec", "selected_for_main": True})
            exclude_keys.add(str(active_match.get("setting_key", "")))

        if fame is not None:
            fame_active_match = _pick_route_match(
                route_pool,
                target_active=_safe_float(fame.get("active_params"), 0.0),
                active_match_tolerance=active_match_tolerance,
                exclude_keys=exclude_keys,
            )
            if fame_active_match is not None:
                matched.append({**dict(fame_active_match), "match_rule": "active_match_to_fame", "selected_for_main": True})
                exclude_keys.add(str(fame_active_match.get("setting_key", "")))

        if fame is not None:
            fame_metric_target = _safe_float(fame.get("test_mrr20"), 0.0)
            fame_metric_pool = [row for row in route_pool if str(row.get("setting_key", "")) not in exclude_keys]
            fame_metric_pool.sort(
                key=lambda row: (
                    abs(_safe_float(row.get("test_mrr20"), 0.0) - fame_metric_target),
                    _safe_float(row.get("active_params"), default=1e18),
                    _safe_float(row.get("avg_train_epoch_sec"), default=1e18),
                )
            )
            if fame_metric_pool:
                matched.append({**dict(fame_metric_pool[0]), "match_rule": "metric_match_to_fame", "selected_for_main": True})
                exclude_keys.add(str(fame_metric_pool[0].get("setting_key", "")))

        quality_floor = _safe_float(best_route.get("test_mrr20")) * 0.98
        quality_pool = [
            row
            for row in route_pool
            if str(row.get("setting_key", "")) not in exclude_keys and _safe_float(row.get("test_mrr20")) >= quality_floor
        ]
        quality_pool.sort(
            key=lambda row: (
                _safe_float(row.get("active_params"), default=1e18),
                _safe_float(row.get("avg_train_epoch_sec"), default=1e18),
                -_safe_float(row.get("test_mrr20")),
            )
        )
        if quality_pool:
            matched.append({**dict(quality_pool[0]), "match_rule": "quality_retained_min_active", "selected_for_main": True})
    return matched


def _build_compact_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                "question": "q4",
                "dataset_scope": row.get("dataset_scope", ""),
                "dataset": row.get("dataset", ""),
                "model_name": row.get("model_name", ""),
                "model": row.get("model", ""),
                "setting_key": row.get("setting_key", ""),
                "match_rule": row.get("match_rule", row.get("selection_tag", "")),
                "status": row.get("status", ""),
                "run_phase": row.get("run_phase", ""),
                "benchmark_epochs": row.get("benchmark_epochs", ""),
                "chosen_lr": row.get("chosen_lr", ""),
                "test_mrr20": row.get("test_mrr20", ""),
                "test_ndcg10": row.get("test_ndcg10", ""),
                "test_hit10": row.get("test_hit10", ""),
                "total_params": row.get("total_params", ""),
                "active_params": row.get("active_params", ""),
                "train_time_ratio": row.get("train_time_ratio", ""),
                "infer_time_ratio": row.get("infer_time_ratio", ""),
            }
        )
    return compact


def _spec_map_by_key(specs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in specs:
        dataset = str(spec["config"].get("dataset", ""))
        mapping[(dataset, str(spec["setting_key"]))] = spec
    return mapping


def _build_confirm_specs(
    matched_screen_rows: list[dict[str, Any]],
    spec_map: dict[tuple[str, str], dict[str, Any]],
    *,
    confirm_epochs: int,
) -> list[dict[str, Any]]:
    confirm_specs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in matched_screen_rows:
        key = (str(row.get("dataset", "")), str(row.get("setting_key", "")))
        if key in seen:
            continue
        spec = spec_map.get(key)
        if spec is None:
            continue
        confirm_spec = dict(spec)
        confirm_spec["config"] = _set_epoch_budget(spec["config"], int(confirm_epochs))
        confirm_specs.append(confirm_spec)
        seen.add(key)
    return confirm_specs


def _merge_confirm_results(
    matched_screen_rows: list[dict[str, Any]],
    confirm_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    confirm_map = {
        (str(row.get("dataset", "")), str(row.get("setting_key", ""))): row
        for row in confirm_rows
        if str(row.get("status", "")).lower() == "ok"
    }
    merged: list[dict[str, Any]] = []
    for row in matched_screen_rows:
        key = (str(row.get("dataset", "")), str(row.get("setting_key", "")))
        confirmed = confirm_map.get(key)
        if confirmed is None:
            fallback = dict(row)
            fallback["selection_source_phase"] = "screen"
            merged.append(fallback)
            continue
        merged_row = dict(confirmed)
        merged_row["match_rule"] = row.get("match_rule", "")
        merged_row["selected_for_main"] = row.get("selected_for_main", False)
        merged_row["selection_source_phase"] = "screen"
        merged.append(merged_row)
    return merged


def main() -> int:
    parser = common_arg_parser("Q4 efficiency sweep with active-parameter matching", question="q4")
    parser.add_argument("--benchmark-datasets", default="")
    parser.add_argument("--screen-epochs", type=int, default=8)
    parser.add_argument("--confirm-epochs", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--include-fame", action="store_true", default=True)
    parser.add_argument("--no-include-fame", dest="include_fame", action="store_false")
    parser.add_argument("--lr-multipliers", default="0.8,1.0")
    parser.add_argument("--confirm-lr-multipliers", default="0.8,1.0,1.2")
    parser.add_argument("--width-multipliers", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--group-topk-grid", default="1,2,3,4")
    parser.add_argument("--expert-topk-grid", default="1,2")
    parser.add_argument("--max-route-screen-runs", type=int, default=12)
    parser.add_argument("--active-match-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    benchmark_scopes = _resolve_benchmark_datasets(args.datasets, args.benchmark_datasets)
    candidates = selected_candidates_from_args(args)
    by_dataset = {cand.dataset: cand for cand in candidates}
    gpu_id = int((parse_csv_list(args.gpus) or ["0"])[0])
    width_multipliers = [float(token) for token in parse_csv_list(args.width_multipliers)] or [1.0]
    group_topk_grid = parse_csv_ints(args.group_topk_grid) or [1, 2, 3, 4]
    expert_topk_grid = parse_csv_ints(args.expert_topk_grid) or [1, 2]
    lr_multipliers = [float(token) for token in parse_csv_list(args.lr_multipliers)] or [1.0]
    confirm_lr_multipliers = [float(token) for token in parse_csv_list(args.confirm_lr_multipliers)] or [1.0]

    previous_rows = {
        _row_resume_key(row): row
        for row in _read_csv(LOG_ROOT / "q4" / "summary.csv")
        if str(row.get("status", "")).lower() == "ok"
    }

    screen_rows: list[dict[str, Any]] = []
    confirm_rows: list[dict[str, Any]] = []
    spec_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for dataset_scope, dataset_name in benchmark_scopes:
        candidate = by_dataset.get(dataset_name)
        if candidate is None:
            raise RuntimeError(f"No base candidate available for Q4 dataset={dataset_name}")
        route_cfg = build_eval_config_from_result_payload(candidate.payload)
        specs = _portfolio_specs(
            dataset_name,
            route_cfg,
            gpu_id=gpu_id,
            epochs=int(args.screen_epochs),
            train_batch=int(args.train_batch_size),
            eval_batch=int(args.eval_batch_size),
            include_fame=bool(args.include_fame),
            width_multipliers=width_multipliers,
            group_topk_grid=group_topk_grid,
            expert_topk_grid=expert_topk_grid,
            max_route_screen_runs=int(args.max_route_screen_runs),
        )
        spec_maps[dataset_name] = _spec_map_by_key(specs)
        screen_rows.extend(
            _measure_specs(
                specs,
                dataset_scope=dataset_scope,
                dataset_name=dataset_name,
                benchmark_epochs=int(args.screen_epochs),
                lr_multipliers=lr_multipliers,
                run_phase="screen",
                resume_from_logs=bool(args.resume_from_logs),
                previous_rows=previous_rows,
                dry_run=bool(args.dry_run),
            )
        )

    screen_rows = _augment_ratios(screen_rows)
    matched_screen_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        matched_screen_rows = _build_matched_rows(screen_rows, active_match_tolerance=float(args.active_match_tolerance))
        for dataset_scope, dataset_name in benchmark_scopes:
            dataset_matched = [row for row in matched_screen_rows if str(row.get("dataset", "")) == dataset_name]
            confirm_specs = _build_confirm_specs(
                dataset_matched,
                spec_maps.get(dataset_name, {}),
                confirm_epochs=int(args.confirm_epochs),
            )
            confirm_rows.extend(
                _measure_specs(
                    confirm_specs,
                    dataset_scope=dataset_scope,
                    dataset_name=dataset_name,
                    benchmark_epochs=int(args.confirm_epochs),
                    lr_multipliers=confirm_lr_multipliers,
                    run_phase="confirm",
                    resume_from_logs=bool(args.resume_from_logs),
                    previous_rows=previous_rows,
                    dry_run=False,
                )
            )
        confirm_rows = _augment_ratios(confirm_rows)
        matched_rows = _merge_confirm_results(matched_screen_rows, confirm_rows)
        compact_rows = _build_compact_table(matched_rows)
    else:
        matched_screen_rows = _build_matched_rows(screen_rows, active_match_tolerance=float(args.active_match_tolerance)) if screen_rows else []

    summary_rows = screen_rows + confirm_rows

    q4_root = ensure_dir(LOG_ROOT / "q4")
    _write_csv(q4_root / "summary.csv", summary_rows)
    _write_csv(q4_root / "screen.csv", screen_rows)
    _write_csv(q4_root / "confirm.csv", confirm_rows)
    _write_csv(q4_root / "matched_screen.csv", matched_screen_rows)
    _write_csv(q4_root / "matched.csv", matched_rows)
    _write_csv(q4_root / "q4_efficiency_table.csv", compact_rows)
    _write_csv(DATA_ROOT / "q4_efficiency_table.csv", compact_rows)
    _write_csv(DATA_ROOT / "q4_efficiency_screen.csv", screen_rows)
    _write_csv(DATA_ROOT / "q4_efficiency_confirm.csv", confirm_rows)
    _write_csv(DATA_ROOT / "q4_efficiency_matches.csv", matched_rows)
    write_json(
        q4_root / "manifest.json",
        {
            "generated_at": now_utc(),
            "question": "q4",
            "benchmark_scopes": benchmark_scopes,
            "screen_epochs": int(args.screen_epochs),
            "confirm_epochs": int(args.confirm_epochs),
            "include_fame": bool(args.include_fame),
            "lr_multipliers": lr_multipliers,
            "confirm_lr_multipliers": confirm_lr_multipliers,
            "width_multipliers": width_multipliers,
            "group_topk_grid": group_topk_grid,
            "expert_topk_grid": expert_topk_grid,
            "max_route_screen_runs": int(args.max_route_screen_runs),
            "screen_rows": len(screen_rows),
            "confirm_rows": len(confirm_rows),
            "matched_rows": len(matched_rows),
        },
    )
    print(f"[q4] summary -> {q4_root / 'summary.csv'}")
    print(f"[q4] screen  -> {q4_root / 'screen.csv'}")
    print(f"[q4] confirm -> {q4_root / 'confirm.csv'}")
    print(f"[q4] matched -> {q4_root / 'matched.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
