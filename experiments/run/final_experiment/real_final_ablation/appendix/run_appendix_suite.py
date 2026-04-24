#!/usr/bin/env python3
"""Run the appendix A->K suite in paper order and export notebook bundles."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[5]


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
    forwarded.append("--resume-from-logs" if args.resume_from_logs else "--no-resume-from-logs")
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.smoke_test:
        forwarded.extend(["--smoke-test", "--smoke-max-runs", str(args.smoke_max_runs)])
    return forwarded


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-final appendix suite.")
    parser.add_argument("--datasets", default="KuaiRecLargeStrictPosV2_0.2,foursquare,retail_rocket,lastfm0.03")
    parser.add_argument("--models", default="featured_moe_n3")
    parser.add_argument("--top-k-configs", type=int, default=4)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--base-csv", default=str(REPO_ROOT / "experiments" / "run" / "final_experiment" / "ablation" / "configs" / "base_candidates.csv"))
    parser.add_argument("--max-evals", type=int, default=3)
    parser.add_argument("--max-run-hours", type=float, default=0.5)
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
    parser.add_argument("--sections", default="full_results,special_bins,structural,sparse,objective,cost,diagnostics,behavior_slices,cases")
    parser.add_argument("--include-transfer", action="store_true", help="Opt in to the optional transfer scaffold.")
    args = parser.parse_args()

    common_args = _forward_args(args)
    scripts = {
        "full_results": "full_results_export.py",
        "special_bins": "special_bins.py",
        "structural": "structural_ablation.py",
        "sparse": "sparse_routing.py",
        "objective": "objective_variants.py",
        "cost": "cost_summary.py",
        "diagnostics": "routing_diagnostics.py",
        "behavior_slices": "behavior_slices.py",
        "cases": "targeted_interventions.py",
        "transfer": "optional_transfer.py",
    }
    selected = [token.strip() for token in str(args.sections).split(",") if token.strip()]
    if not args.include_transfer:
        selected = [token for token in selected if token != "transfer"]
    python = "/venv/FMoE/bin/python"
    for key in selected:
        script = scripts[key]
        cmd = [python, str(CODE_DIR / script), *common_args]
        print(f"[appendix-suite] running {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    export_cmd = [python, str(CODE_DIR / "export_appendix_bundle.py"), "--datasets", args.datasets]
    print(f"[appendix-suite] exporting {' '.join(export_cmd)}")
    subprocess.run(export_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
