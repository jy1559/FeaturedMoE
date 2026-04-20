#!/usr/bin/env python3
"""Run A06->A10 appendix ablations and export notebook bundles."""

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
    if args.appendix:
        forwarded.append("--appendix")
    return forwarded


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full A06~A10 appendix ablation suite.")
    parser.add_argument("--datasets", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument("--models", default="featured_moe_n3")
    parser.add_argument("--top-k-configs", type=int, default=1)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--base-csv", default=str(CODE_DIR / "configs" / "base_candidates.csv"))
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--max-run-hours", type=float, default=1.0,
                        help="Wall-clock cap per job in hours (default: 1.0h).")
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--lr-mode", default="narrow_loguniform")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    parser.add_argument("--appendix", action="store_true")
    parser.add_argument("--output-tag", default="")
    # A09-specific
    parser.add_argument("--pairs", default="beauty_to_kuairec,kuairec_to_beauty")
    parser.add_argument("--hparam-presets", default="shared_a")
    # Question selection
    parser.add_argument("--questions", default="a06,a07,a08,a09,a10",
                        help="Comma-separated list of appendix questions to run.")
    # Reuse flags
    parser.add_argument("--reuse-q2-checkpoints", action="store_true", default=False)
    parser.add_argument("--reuse-q4-checkpoints", action="store_true", default=False)
    parser.add_argument("--reuse-q5-interventions", action="store_true", default=False)
    args = parser.parse_args()

    questions = {q.strip() for q in str(args.questions).split(",") if q.strip()}
    common_args = _forward_args(args)
    python = "/venv/FMoE/bin/python"

    script_map = {
        "a06": ("a06_structural_sanity.py", []),
        "a07": ("a07_topk_routing.py", []),
        "a08": ("a08_behavior_cases.py", ["--reuse-q2-checkpoints"] if args.reuse_q2_checkpoints else []),
        "a09": ("a09_transfer_portability.py", ["--pairs", args.pairs, "--hparam-presets", args.hparam_presets]),
        "a10": (
            "a10_cue_semantics.py",
            (["--reuse-q4-checkpoints"] if args.reuse_q4_checkpoints else [])
            + (["--reuse-q5-interventions"] if args.reuse_q5_interventions else []),
        ),
    }

    for q in sorted(questions):
        if q not in script_map:
            print(f"[suite] unknown question: {q}")
            continue
        script, extra = script_map[q]
        cmd = [python, str(CODE_DIR / script), *common_args, *extra]
        print(f"[suite] running {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    # Export
    export_cmd = [python, str(CODE_DIR / "export_a06_a10_bundle.py"), "--questions", ",".join(sorted(questions))]
    print(f"[suite] exporting {' '.join(export_cmd)}")
    subprocess.run(export_cmd, check=True, cwd=str(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
