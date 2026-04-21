#!/usr/bin/env python3
"""Run the real-final Q2~Q5 suite and export notebook bundles."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]


def _forward_args(args: argparse.Namespace) -> list[str]:
    pairs = [
        ("--datasets", args.datasets),
        ("--models", args.models),
        ("--top-k-configs", str(args.top_k_configs)),
        ("--seeds", args.seeds),
        ("--gpus", args.gpus),
        ("--base-csv", args.base_csv),
        ("--max-evals", str(args.max_evals)),
        ("--max-run-hours", str(args.max_run_hours)),
        ("--tune-epochs", str(args.tune_epochs)),
        ("--tune-patience", str(args.tune_patience)),
        ("--lr-mode", args.lr_mode),
        ("--search-algo", args.search_algo),
        ("--output-tag", args.output_tag),
    ]
    forwarded: list[str] = []
    for key, value in pairs:
        forwarded.extend([key, value])
    if args.resume_from_logs:
        forwarded.append("--resume-from-logs")
    else:
        forwarded.append("--no-resume-from-logs")
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.smoke_test:
        forwarded.extend(["--smoke-test", "--smoke-max-runs", str(args.smoke_max_runs)])
    if args.case_eval_fast:
        forwarded.append("--case-eval-fast")
    return forwarded


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-final RouteRec Q2~Q5 suite.")
    parser.add_argument("--datasets", default="KuaiRecLargeStrictPosV2_0.2,beauty,foursquare,retail_rocket,movielens1m,lastfm0.03")
    parser.add_argument("--models", default="featured_moe_n3")
    parser.add_argument("--top-k-configs", type=int, default=1)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--base-csv", default=str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "configs" / "base_candidates.csv"))
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--max-run-hours", type=float, default=1.0)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--lr-mode", default="narrow_loguniform")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--case-eval-fast", action="store_true")
    args = parser.parse_args()

    scripts = [
        "q2_routing_control.py",
        "q3_stage_structure.py",
        "q4_portability.py",
        "q5_behavior_semantics.py",
        "export_q2_q5_bundle.py",
    ]
    common_args = _forward_args(args)
    for script in scripts:
        cmd = ["/venv/FMoE/bin/python", str(CODE_DIR / script), *common_args]
        print(f"[suite] running {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
