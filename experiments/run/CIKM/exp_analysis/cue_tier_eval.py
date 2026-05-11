#!/usr/bin/env python3
"""Post-hoc cue-tier evaluation for CIKM 2026 experiments.

Reads best checkpoints from P0 result JSONs and evaluates each model on
the 8 cue-tier test splits built by build_cue_case_eval.py.

Usage:
    # Build splits first:
    python build_cue_case_eval.py

    # Then evaluate (reads P0 results automatically):
    python cue_tier_eval.py [--gpu 0]
    python cue_tier_eval.py --result-dir ../../results --gpu 0
    python cue_tier_eval.py --models sasrec featured_moe_n3 --datasets KuaiRec --gpu 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_CIKM_DIR = Path(__file__).resolve().parents[1]
_EXP_DIR  = _CIKM_DIR.parent.parent
_REPO_ROOT = _CIKM_DIR.parents[2]

if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from common import (  # noqa: E402
    ALL_MODELS, DATASETS, RESULT_ROOT,
)

CASE_EVAL_ROOT = _REPO_ROOT / "Datasets" / "processed" / "cikm_case_eval" / "pure"
TIER_GROUPS = [
    "memory_plus",  "memory_minus",
    "focus_plus",   "focus_minus",
    "tempo_plus",   "tempo_minus",
    "exposure_plus","exposure_minus",
]

OUTPUT_CSV = RESULT_ROOT / "cue_tier_metrics.csv"
FIELDS = [
    "dataset", "model", "tier_group",
    "n_sessions", "hit5", "hit10", "hit20",
    "ndcg5", "ndcg10", "ndcg20",
    "mrr5",  "mrr10",  "mrr20",
    "checkpoint_path", "timestamp_utc",
]


def _load_result_summaries(result_dir: Path) -> list[dict]:
    """Load all P0 result row dicts from main_baselines and main_routerec summaries."""
    rows = []
    for csv_name in ["main_baselines_summary.csv", "main_routerec_summary.csv"]:
        f = result_dir / csv_name
        if f.exists():
            rows.extend(list(csv.DictReader(f.open())))
    return rows


def _find_checkpoint(result_path: str | None) -> str | None:
    if not result_path:
        return None
    p = Path(result_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        for key in ("best_checkpoint_file", "artifact_best_checkpoint"):
            ckpt = str(data.get(key, "") or "").strip()
            if ckpt and Path(ckpt).exists():
                return ckpt
    except Exception:
        pass
    return None


def _run_checkpoint_eval(
    *,
    checkpoint_path: str,
    dataset: str,
    tier_group: str,
    gpu_id: str,
) -> dict | None:
    """Run recbole_train.run_checkpoint_evaluation on a tier subset."""
    tier_dir = CASE_EVAL_ROOT / tier_group / dataset
    test_inter = tier_dir / f"{dataset}.test.inter"
    if not test_inter.exists():
        return None

    try:
        import recbole_patch  # noqa: F401 – numpy 2.0 compat

        from recbole_train import run_checkpoint_evaluation  # noqa: E402

        cfg = {
            "dataset": dataset,
            "data_path": str(CASE_EVAL_ROOT / tier_group),
            "eval_mode": "session_fixed",
            "feature_mode": "final",
            "gpu_id": int(gpu_id),
            "use_gpu": True,
            "eval_batch_size": 1024,
            "eval_sampling": {"mode": "full", "auto_full_threshold": 999999999},
            "show_progress": False,
            "state": "INFO",
        }

        result = run_checkpoint_evaluation(
            cfg,
            run_name=f"cue_tier_{tier_group}_{dataset}",
            checkpoint_path=checkpoint_path,
        )
        return result
    except Exception as e:
        print(f"    [WARN] eval failed for {tier_group}/{dataset}: {e}", flush=True)
        return None


def _count_tier_sessions(dataset: str, tier_group: str) -> int:
    test_inter = CASE_EVAL_ROOT / tier_group / dataset / f"{dataset}.test.inter"
    if not test_inter.exists():
        return 0
    try:
        with test_inter.open() as f:
            sessions = set()
            header = f.readline()
            cols = header.strip().split("\t")
            sid_col = next((i for i, c in enumerate(cols) if c.split(":")[0] == "session_id"), 0)
            for line in f:
                parts = line.strip().split("\t")
                if parts and sid_col < len(parts):
                    sessions.add(parts[sid_col])
        return len(sessions)
    except Exception:
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Cue-tier post-hoc evaluation")
    # Default: only FMoE (checkpoint from P0) + sasrec (checkpoint from run_sasrec_cue_baseline.py)
    # This minimises checkpoint storage while still enabling the key CIKM comparison.
    p.add_argument("--result-dir", default=str(RESULT_ROOT), help="CIKM results dir")
    p.add_argument("--gpu", default="0")
    p.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    p.add_argument("--models", nargs="+", default=["featured_moe_n3", "sasrec"],
                   help="Models to evaluate (default: featured_moe_n3 sasrec)")
    p.add_argument("--tiers", nargs="+", default=TIER_GROUPS, choices=TIER_GROUPS)
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    p0_rows = _load_result_summaries(result_dir)

    # Build lookup: (dataset, model) → result_path
    ckpt_map: dict[tuple[str, str], str | None] = {}
    for row in p0_rows:
        key = (row.get("dataset", ""), row.get("model", ""))
        ckpt = _find_checkpoint(row.get("result_path", ""))
        ckpt_map[key] = ckpt

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_CSV.exists()
    out_fh = OUTPUT_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    for dataset in args.datasets:
        for model in args.models:
            ckpt = ckpt_map.get((dataset, model))
            if not ckpt:
                print(f"[SKIP] {dataset}/{model}: no checkpoint found", flush=True)
                continue
            print(f"\n[EVAL] {dataset}/{model}  checkpoint={ckpt[-60:]}", flush=True)
            for tier_group in args.tiers:
                n_sess = _count_tier_sessions(dataset, tier_group)
                if n_sess == 0:
                    print(f"  [SKIP] {tier_group}: no test sessions (run build_cue_case_eval.py first)", flush=True)
                    continue
                print(f"  {tier_group}: {n_sess} sessions ...", end=" ", flush=True)
                result = _run_checkpoint_eval(
                    checkpoint_path=ckpt,
                    dataset=dataset,
                    tier_group=tier_group,
                    gpu_id=args.gpu,
                )
                if result is None:
                    print("FAILED", flush=True)
                    continue
                test_r = result.get("test_result", {}) or {}
                row = {
                    "dataset": dataset,
                    "model": model,
                    "tier_group": tier_group,
                    "n_sessions": n_sess,
                    "hit5":   test_r.get("hit@5", 0.0),
                    "hit10":  test_r.get("hit@10", 0.0),
                    "hit20":  test_r.get("hit@20", 0.0),
                    "ndcg5":  test_r.get("ndcg@5", 0.0),
                    "ndcg10": test_r.get("ndcg@10", 0.0),
                    "ndcg20": test_r.get("ndcg@20", 0.0),
                    "mrr5":   test_r.get("mrr@5", 0.0),
                    "mrr10":  test_r.get("mrr@10", 0.0),
                    "mrr20":  test_r.get("mrr@20", 0.0),
                    "checkpoint_path": ckpt,
                    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                writer.writerow(row)
                out_fh.flush()
                print(f"MRR@20={row['mrr20']:.4f}  HR@10={row['hit10']:.4f}", flush=True)

    out_fh.close()
    print(f"\nResults → {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
