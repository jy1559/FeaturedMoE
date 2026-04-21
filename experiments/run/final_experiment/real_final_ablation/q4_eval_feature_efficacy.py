#!/usr/bin/env python3
"""Run eval-only cue-efficacy perturbations for one exported Q4 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = REPO_ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_eval_config_from_result_payload, q4_feature_efficacy_specs  # noqa: E402
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


def _load_result_payload(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval one checkpoint under Q4 cue-efficacy settings.")
    parser.add_argument("--source-result-json", required=True)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpu-id", type=int, default=None)
    args = parser.parse_args()

    source_result = Path(args.source_result_json).expanduser().resolve()
    checkpoint_file = Path(args.checkpoint_file).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_result.is_file():
        raise FileNotFoundError(f"Source result JSON not found: {source_result}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    payload = _load_result_payload(source_result)
    base_cfg = build_eval_config_from_result_payload(payload)
    dataset = str(base_cfg.get("dataset", "")).strip()
    if not dataset:
        raise RuntimeError("Failed to resolve dataset from source result.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_root = output_root / f"{_sanitize(dataset)}_{run_stamp}"
    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_root / "q4_efficacy_manifest.csv"

    manifest_rows: list[dict[str, str]] = []
    for idx, spec in enumerate(q4_feature_efficacy_specs(), start=1):
        intervention = str(spec["intervention"])
        print(f"[{idx}/4] efficacy={intervention}")
        eval_cfg = deepcopy(base_cfg)
        eval_cfg.update(deepcopy(spec.get("overrides") or {}))
        eval_cfg["run_phase"] = f"Q4_EFFICACY_{_sanitize(intervention)}"
        if args.gpu_id is not None:
            eval_cfg["gpu_id"] = int(args.gpu_id)
            eval_cfg["use_gpu"] = True

        logging_dir = ""
        result_file = ""
        error = ""
        result_payload: dict | None = None
        run_logger = None
        try:
            model_name = str(eval_cfg.get("model", "")).lower()
            if model_name in _FEATURE_AWARE_MOE_MODELS:
                from models.FeaturedMoE.run_logger import RunLogger

                run_logger = RunLogger(
                    run_name=f"{_sanitize(dataset)}_{_sanitize(intervention)}",
                    config=eval_cfg,
                    output_root=str(bundle_root / "logging"),
                    debug_logging=False,
                )
                logging_dir = str(run_logger.output_path)

            result_payload = run_checkpoint_evaluation(
                eval_cfg,
                run_name=f"{_sanitize(dataset)}_{_sanitize(intervention)}",
                checkpoint_path=str(checkpoint_file),
                run_logger=run_logger,
            )
            result_payload["intervention"] = intervention
            result_payload["intervention_label"] = str(spec.get("label", intervention))

            if logging_dir:
                result_file = str(Path(logging_dir) / "q4_efficacy_result.json")
            else:
                result_dir = bundle_root / "results"
                result_dir.mkdir(parents=True, exist_ok=True)
                result_file = str((result_dir / f"{_sanitize(intervention)}.json").resolve())
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_payload, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            error = str(exc)
            print(f"[WARN] efficacy={intervention} failed: {exc}")

        manifest_rows.append(
            {
                "timestamp_utc": _now_utc(),
                "source_result_json": str(source_result),
                "checkpoint_file": str(checkpoint_file),
                "dataset": dataset,
                "model": str(base_cfg.get("model", "")),
                "intervention": intervention,
                "intervention_label": str(spec.get("label", intervention)),
                "logging_dir": logging_dir,
                "result_file": result_file,
                "best_valid_mrr20": "" if result_payload is None else str((result_payload.get("best_valid_result") or {}).get("mrr@20", "")),
                "test_mrr20": "" if result_payload is None else str((result_payload.get("test_result") or {}).get("mrr@20", "")),
                "best_valid_seen_mrr20": "" if result_payload is None else str(result_payload.get("valid_seen_target_mrr20", "")),
                "test_seen_mrr20": "" if result_payload is None else str(result_payload.get("test_seen_target_mrr20", "")),
                "status": "ok" if not error else "error",
                "error": error,
            }
        )
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"[DONE] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())