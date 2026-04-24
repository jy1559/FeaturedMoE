#!/usr/bin/env python3
"""Run one checkpoint evaluation and materialize a case-eval manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = REPO_ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from recbole_train import _FEATURE_AWARE_MOE_MODELS, run_checkpoint_evaluation  # noqa: E402


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(text: str) -> str:
    out = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or "x"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint and write a case-eval manifest.")
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--source-result-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bundle-name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-by-group", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    args = parser.parse_args()

    config_json = Path(args.config_json).expanduser().resolve()
    checkpoint_file = Path(args.checkpoint_file).expanduser().resolve()
    source_result_json = Path(args.source_result_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    bundle_name = str(args.bundle_name or "").strip()

    cfg = _load_json(config_json)
    source_payload = _load_json(source_result_json)
    dataset = str(cfg.get("dataset", source_payload.get("dataset", ""))).strip()
    model = str(cfg.get("model", source_payload.get("model", ""))).strip()
    if not dataset or not model:
        raise RuntimeError("Both dataset and model must be available for checkpoint case evaluation.")

    if not bundle_name:
        bundle_name = f"{_sanitize(dataset)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_root = output_root / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_root / "case_eval_manifest.csv"

    if args.resume and manifest_path.exists():
        print(f"[resume] case eval manifest already exists: {manifest_path}")
        return 0

    result_file = bundle_root / "results" / "original_eval.json"
    logging_dir = ""
    special_metrics_file = ""
    router_diag_file = ""
    if args.skip_original:
        result_payload = dict(source_payload)
        result_payload["checkpoint_file"] = str(checkpoint_file)
        result_payload["source_result_json"] = str(source_result_json)
        _write_json(result_file, result_payload)
        special_metrics_file = str(source_payload.get("special_log_file", "") or "")
        router_diag_file = str(source_payload.get("diag_raw_test_file", "") or "")
    else:
        if args.gpu_id is not None:
            cfg["gpu_id"] = int(args.gpu_id)
            cfg["use_gpu"] = True
        if str(model).lower() in _FEATURE_AWARE_MOE_MODELS:
            from models.FeaturedMoE.run_logger import RunLogger

            run_logger = RunLogger(
                run_name=bundle_name,
                config=cfg,
                output_root=str(bundle_root / "logging"),
                debug_logging=False,
            )
        else:
            run_logger = None
        result_payload = run_checkpoint_evaluation(
            cfg,
            run_name=bundle_name,
            checkpoint_path=str(checkpoint_file),
            run_logger=run_logger,
        )
        result_payload["source_result_json"] = str(source_result_json)
        result_payload["bundle_name"] = bundle_name
        _write_json(result_file, result_payload)
        logging_dir = str(result_payload.get("logging_dir", "") or "")
        if logging_dir:
            special_metrics_file = str((Path(logging_dir) / "special_metrics.json").resolve())
            router_diag_file = str((Path(logging_dir) / "router_diag.json").resolve())

    row = {
        "timestamp_utc": _now_utc(),
        "bundle_name": bundle_name,
        "source_result_json": str(source_result_json),
        "checkpoint_file": str(checkpoint_file),
        "config_json": str(config_json),
        "dataset": dataset,
        "model": model,
        "scope": "original",
        "group": "original",
        "logging_dir": logging_dir,
        "result_file": str(result_file.resolve()),
        "special_metrics_file": special_metrics_file,
        "router_diag_file": router_diag_file,
        "skip_by_group": int(bool(args.skip_by_group)),
        "status": "ok",
        "error": "",
    }
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"[DONE] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())