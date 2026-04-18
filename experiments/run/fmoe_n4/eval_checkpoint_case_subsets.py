#!/usr/bin/env python3
"""Eval one exported checkpoint across original and case-eval dataset roots."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from recbole_train import (  # noqa: E402
    _FEATURE_AWARE_MOE_MODELS,
    run_checkpoint_evaluation,
)


DEFAULT_CASE_ROOT = "/workspace/FeaturedMoE/Datasets/processed/feature_added_v4_case_eval_final_v1"
DEFAULT_OUTPUT_ROOT = "/workspace/FeaturedMoE/experiments/run/artifacts/logging/fmoe_n4_case_eval"
DEFAULT_GROUPS = [
    "memory_plus",
    "memory_minus",
    "focus_plus",
    "focus_minus",
    "tempo_plus",
    "tempo_minus",
    "exposure_plus",
    "exposure_minus",
]


def _parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").split(",") if token.strip()]


def _sanitize(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    while "__" in token:
        token = token.replace("__", "_")
    return token or "x"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_inputs(args) -> tuple[Path, Path]:
    if args.source_run_dir:
        run_dir = Path(args.source_run_dir).expanduser().resolve()
        config_json = run_dir / "config.json"
        checkpoint_file = run_dir / "checkpoints" / "best_model_state.pth"
    else:
        config_json = Path(args.config_json).expanduser().resolve()
        checkpoint_file = Path(args.checkpoint_file).expanduser().resolve()

    if not config_json.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_json}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    return config_json, checkpoint_file


def _dataset_exists(root: Path, dataset: str) -> bool:
    dataset_dir = root / dataset
    if dataset_dir.exists():
        return True
    return any((root / f"{dataset}{suffix}").exists() for suffix in (".train.inter", ".valid.inter", ".test.inter"))


def _build_targets(
    *,
    source_cfg: dict,
    dataset: str,
    case_root: Path,
    include_original: bool,
    include_tier_union: bool,
    include_by_group: bool,
    tiers: Iterable[str],
    groups: Iterable[str],
):
    targets = []
    original_root = Path(str(source_cfg.get("data_path", ""))).expanduser().resolve()
    if include_original:
        targets.append(
            {
                "label": "original",
                "scope": "original",
                "tier": "",
                "group": "",
                "data_path": str(original_root),
            }
        )

    for tier in tiers:
        tier_root = case_root / str(tier)
        if include_tier_union and _dataset_exists(tier_root, dataset):
            targets.append(
                {
                    "label": f"{tier}_union",
                    "scope": "tier_union",
                    "tier": str(tier),
                    "group": "",
                    "data_path": str(tier_root),
                }
            )
        if not include_by_group:
            continue
        for group in groups:
            group_root = case_root / "by_tier_group" / str(tier) / str(group)
            if not _dataset_exists(group_root, dataset):
                continue
            targets.append(
                {
                    "label": f"{tier}__{group}",
                    "scope": "tier_group",
                    "tier": str(tier),
                    "group": str(group),
                    "data_path": str(group_root),
                }
            )
    return targets


def _build_eval_cfg(source_cfg: dict, *, target: dict, gpu_id: int | None):
    cfg = deepcopy(source_cfg)
    cfg["data_path"] = str(target["data_path"])
    cfg["dataset"] = str(source_cfg["dataset"])
    cfg["log_wandb"] = False
    cfg["saved"] = False
    cfg["export_best_checkpoint"] = False
    cfg["case_eval_scope"] = str(target["scope"])
    cfg["case_eval_tier"] = str(target["tier"])
    cfg["case_eval_group"] = str(target["group"])
    cfg["case_eval_label"] = str(target["label"])
    cfg["case_eval_source_data_path"] = str(source_cfg.get("data_path", ""))
    cfg["case_eval_selection_rule"] = "overall_seen_target"
    cfg["run_phase"] = f"{source_cfg.get('run_phase', 'CASE')}_CASEEVAL_{_sanitize(target['label'])}"
    cfg["fmoe_phase"] = str(source_cfg.get("fmoe_phase", source_cfg.get("phase", "CASEEVAL")))
    cfg["fmoe_run_id"] = f"caseeval_{_sanitize(target['label'])}_{datetime.now().strftime('%m%d_%H%M%S_%f')[-15:]}"
    if gpu_id is not None:
        cfg["gpu_id"] = int(gpu_id)
        cfg["use_gpu"] = True
    return cfg


def _manifest_row(
    *,
    source_run_dir: str,
    source_result_json: str,
    checkpoint_file: str,
    target: dict,
    logging_dir: str,
    case_eval_result_file: str,
    result: dict | None,
    error: str,
):
    result = result or {}
    best_valid = result.get("best_valid_result", {}) or {}
    test = result.get("test_result", {}) or {}
    return {
        "timestamp_utc": _now_utc(),
        "source_run_dir": str(source_run_dir),
        "source_result_json": str(source_result_json),
        "checkpoint_file": str(checkpoint_file),
        "dataset": str(result.get("dataset", "")),
        "model": str(result.get("model", "")),
        "scope": str(target["scope"]),
        "tier": str(target["tier"]),
        "group": str(target["group"]),
        "label": str(target["label"]),
        "data_path": str(target["data_path"]),
        "logging_dir": str(logging_dir),
        "case_eval_result_file": str(case_eval_result_file),
        "metrics_summary_file": str(Path(logging_dir) / "metrics_summary.json") if logging_dir else "",
        "special_metrics_file": str(Path(logging_dir) / "special_metrics.json") if logging_dir else "",
        "router_diag_file": str(Path(logging_dir) / "router_diag.json") if logging_dir else "",
        "best_valid_mrr20": best_valid.get("mrr@20", ""),
        "test_mrr20": test.get("mrr@20", ""),
        "best_valid_ndcg20": best_valid.get("ndcg@20", ""),
        "test_ndcg20": test.get("ndcg@20", ""),
        "best_valid_hr10": best_valid.get("hit@10", ""),
        "test_hr10": test.get("hit@10", ""),
        "best_valid_seen_mrr20": result.get("valid_seen_target_mrr20", ""),
        "test_seen_mrr20": result.get("test_seen_target_mrr20", ""),
        "status": "ok" if not error else "error",
        "error": str(error),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval-only checkpoint passes for original and case-eval roots.")
    parser.add_argument("--source-run-dir", default="", help="RunLogger output dir containing config.json and checkpoints/best_model_state.pth")
    parser.add_argument("--config-json", default="", help="Explicit config.json path when source-run-dir is unavailable")
    parser.add_argument("--checkpoint-file", default="", help="Explicit checkpoint path when source-run-dir is unavailable")
    parser.add_argument("--source-result-json", default="", help="Optional source result json for provenance")
    parser.add_argument("--case-root", default=DEFAULT_CASE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tiers", default="pure,permissive")
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--dataset", default="", help="Override dataset name; default uses source config dataset")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-tier-union", action="store_true")
    parser.add_argument("--skip-by-group", action="store_true")
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()

    config_json, checkpoint_file = _resolve_inputs(args)
    source_cfg = _load_json(config_json)
    dataset = str(args.dataset or source_cfg.get("dataset", "")).strip()
    if not dataset:
        raise RuntimeError("Dataset is empty after reading source config.")
    source_cfg["dataset"] = dataset

    case_root = Path(args.case_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = output_root / f"{_sanitize(dataset)}_{run_stamp}"
    bundle_root.mkdir(parents=True, exist_ok=True)

    targets = _build_targets(
        source_cfg=source_cfg,
        dataset=dataset,
        case_root=case_root,
        include_original=not args.skip_original,
        include_tier_union=not args.skip_tier_union,
        include_by_group=not args.skip_by_group,
        tiers=_parse_csv(args.tiers),
        groups=_parse_csv(args.groups),
    )
    if args.max_targets and int(args.max_targets) > 0:
        targets = targets[: int(args.max_targets)]
    if not targets:
        raise RuntimeError("No eval targets were resolved.")

    manifest_rows = []
    manifest_path = bundle_root / "case_eval_manifest.csv"

    for idx, target in enumerate(targets, start=1):
        print(f"[{idx}/{len(targets)}] eval target={target['label']} data_path={target['data_path']}")
        eval_cfg = _build_eval_cfg(source_cfg, target=target, gpu_id=args.gpu_id)
        model_name = str(eval_cfg.get("model", "")).lower()

        run_logger = None
        logging_dir = ""
        case_eval_result_file = ""
        try:
            if model_name in _FEATURE_AWARE_MOE_MODELS:
                from models.FeaturedMoE.run_logger import RunLogger

                run_logger = RunLogger(
                    run_name=f"{_sanitize(dataset)}_{_sanitize(target['label'])}",
                    config=eval_cfg,
                    output_root=str(bundle_root / "logging"),
                    debug_logging=False,
                )
                logging_dir = str(run_logger.output_path)

            result = run_checkpoint_evaluation(
                eval_cfg,
                run_name=f"{_sanitize(dataset)}_{_sanitize(target['label'])}",
                checkpoint_path=str(checkpoint_file),
                run_logger=run_logger,
            )
            result["dataset"] = dataset
            result["model"] = str(eval_cfg.get("model", ""))
            result["scope"] = target["scope"]
            result["tier"] = target["tier"]
            result["group"] = target["group"]
            result["data_path"] = target["data_path"]
            if logging_dir:
                case_eval_result_file = str(Path(logging_dir) / "case_eval_result.json")
                with open(case_eval_result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                result_dir = bundle_root / "results"
                result_dir.mkdir(parents=True, exist_ok=True)
                case_eval_result_file = str((result_dir / f"{_sanitize(target['label'])}.json").resolve())
                with open(case_eval_result_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            manifest_rows.append(
                _manifest_row(
                    source_run_dir=str(Path(args.source_run_dir).expanduser().resolve()) if args.source_run_dir else "",
                    source_result_json=str(Path(args.source_result_json).expanduser().resolve()) if args.source_result_json else "",
                    checkpoint_file=str(checkpoint_file),
                    target=target,
                    logging_dir=logging_dir,
                    case_eval_result_file=case_eval_result_file,
                    result=result,
                    error="",
                )
            )
        except Exception as exc:
            manifest_rows.append(
                _manifest_row(
                    source_run_dir=str(Path(args.source_run_dir).expanduser().resolve()) if args.source_run_dir else "",
                    source_result_json=str(Path(args.source_result_json).expanduser().resolve()) if args.source_result_json else "",
                    checkpoint_file=str(checkpoint_file),
                    target=target,
                    logging_dir=logging_dir,
                    case_eval_result_file=case_eval_result_file,
                    result={
                        "dataset": dataset,
                        "model": str(eval_cfg.get("model", "")),
                    },
                    error=str(exc),
                )
            )
            print(f"[WARN] target={target['label']} failed: {exc}")

        if manifest_rows:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                writer.writeheader()
                writer.writerows(manifest_rows)

    print(f"[DONE] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
