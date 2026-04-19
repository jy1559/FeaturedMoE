#!/usr/bin/env python3
"""Run Q2~Q5 + A06~A10 ablations dataset-by-dataset.

Execution order per dataset (one dataset at a time):
  q2 → q3 → q4 → q5 → export_q2_q5
  a06 → a07 → a08 → a09 (low_data only) → a10 → export_a06_a10

Use --skip-q2-q5 / --skip-appendix to run only one half.
A09 transfer is handled separately via a09_transfer_portability.sh.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON = "/venv/FMoE/bin/python"


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def _build_common_args(args: argparse.Namespace) -> list[str]:
    """Build the forwarded args that are shared across all child scripts."""
    parts: list[str] = [
        "--models", args.models,
        "--top-k-configs", str(args.top_k_configs),
        "--seeds", args.seeds,
        "--gpus", args.gpus,
        "--base-csv", args.base_csv,
        "--max-evals", str(args.max_evals),
        "--max-run-hours", str(args.max_run_hours),
        "--tune-epochs", str(args.tune_epochs),
        "--tune-patience", str(args.tune_patience),
        "--lr-mode", args.lr_mode,
        "--search-algo", args.search_algo,
        "--output-tag", args.output_tag,
    ]
    if args.resume_from_logs:
        parts.append("--resume-from-logs")
    else:
        parts.append("--no-resume-from-logs")
    if args.dry_run:
        parts.append("--dry-run")
    if args.smoke_test:
        parts.extend(["--smoke-test", "--smoke-max-runs", str(args.smoke_max_runs)])
    if args.appendix:
        parts.append("--appendix")
    return parts


def _run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*70}", flush=True)
    print(f"[full-suite] START {label}", flush=True)
    print(f"[full-suite] CMD  {' '.join(cmd)}", flush=True)
    print(f"{'='*70}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    print(f"[full-suite] DONE  {label}", flush=True)


# ---------------------------------------------------------------------------
# Per-dataset Q2~Q5 block
# ---------------------------------------------------------------------------

def run_q2_q5_for_dataset(dataset: str, common_args: list[str], *, appendix: bool) -> None:
    dataset_args = ["--datasets", dataset, *common_args]
    q2q5_scripts = [
        "q2_routing_control.py",
        "q3_stage_structure.py",
        "q4_cue_ablation.py",
        "q5_behavior_regime.py",
    ]
    for script in q2q5_scripts:
        extra = ["--appendix"] if appendix and script == "q4_cue_ablation.py" else []
        _run([PYTHON, str(CODE_DIR / script), *dataset_args, *extra], f"{script} | {dataset}")
    # Export (export script reads from LOG_ROOT directly; no dataset filter needed)
    _run(
        [PYTHON, str(CODE_DIR / "export_q2_q5_bundle.py"), *dataset_args],
        f"export_q2_q5 | {dataset}",
    )


# ---------------------------------------------------------------------------
# Per-dataset A06~A10 block  (no transfer)
# ---------------------------------------------------------------------------

def run_a06_a10_for_dataset(
    dataset: str,
    common_args: list[str],
    *,
    reuse_q2: bool,
    reuse_q4: bool,
    reuse_q5: bool,
    questions: set[str],
) -> None:
    dataset_args = ["--datasets", dataset, *common_args]

    appendix_script_map: dict[str, tuple[str, list[str]]] = {
        "a06": ("a06_structural_sanity.py", []),
        "a07": ("a07_topk_routing.py", []),
        "a08": (
            "a08_behavior_cases.py",
            ["--reuse-q2-checkpoints"] if reuse_q2 else [],
        ),
        "a09": (
            "a09_transfer_portability.py",
            # low_data only; transfer is handled separately after all datasets
            ["--mode", "low_data"],
        ),
        "a10": (
            "a10_cue_semantics.py",
            (["--reuse-q4-checkpoints"] if reuse_q4 else [])
            + (["--reuse-q5-interventions"] if reuse_q5 else []),
        ),
    }

    ran: list[str] = []
    for q in ("a06", "a07", "a08", "a09", "a10"):
        if q not in questions:
            continue
        script, extra = appendix_script_map[q]
        _run([PYTHON, str(CODE_DIR / script), *dataset_args, *extra], f"{script} | {dataset}")
        ran.append(q)

    if ran:
        _run(
            [
                PYTHON,
                str(CODE_DIR / "export_a06_a10_bundle.py"),
                "--datasets", dataset,
                "--questions", ",".join(ran),
            ],
            f"export_a06_a10 [{','.join(ran)}] | {dataset}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full Q2~Q5 + A06~A10 ablation suite, dataset-by-dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution order:
  For each dataset (one at a time):
    q2 → q3 → q4 → q5 → export_q2_q5
    a06 → a07 → a08 → a09 (low_data) → a10 → export_a06_a10

Use --skip-q2-q5 / --skip-appendix to run only one half.
        """,
    )

    # ---- dataset / model / config ----
    parser.add_argument(
        "--datasets",
        default="KuaiRecLargeStrictPosV2_0.2",
        help="Comma-separated dataset list.  Processed one at a time in order.",
    )
    parser.add_argument("--models", default="featured_moe_n3")
    parser.add_argument(
        "--top-k-configs",
        type=int,
        default=1,
        help="Number of top-k base configs to use per dataset.",
    )
    parser.add_argument("--seeds", default="1", help="Comma-separated seed IDs.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs.")
    parser.add_argument(
        "--base-csv",
        default=str(CODE_DIR / "configs" / "base_candidates.csv"),
    )

    # ---- tuning budget ----
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--max-run-hours", type=float, default=1.0,
                        help="Wall-clock cap per job in hours (default: 1.0h).")
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--lr-mode", default="narrow_loguniform")
    parser.add_argument("--search-algo", default="tpe", choices=["tpe", "random"])

    # ---- resume / dry-run ----
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)
    parser.add_argument("--appendix", action="store_true",
                        help="Also run Q4 appendix settings (per-family isolation).")
    parser.add_argument("--output-tag", default="")

    # ---- question selection ----
    parser.add_argument(
        "--questions",
        default="q2,q3,q4,q5,a06,a07,a08,a09,a10",
        help="Comma-separated questions to run. Transfer is always a09.",
    )

    # ---- A09 transfer ----
    parser.add_argument(
        "--pairs",
        default="beauty_to_kuairec,kuairec_to_beauty",
        help="Transfer pair IDs for A09 (comma-separated). Passed to a09_transfer_portability.py when run separately.",
    )
    parser.add_argument("--hparam-presets", default="shared_a")

    # ---- reuse flags ----
    parser.add_argument(
        "--reuse-q2-checkpoints",
        action="store_true",
        default=False,
        help="A08: reuse Q2 route_rec_full checkpoints instead of training.",
    )
    parser.add_argument(
        "--reuse-q4-checkpoints",
        action="store_true",
        default=False,
        help="A10: reuse Q4 full checkpoints instead of training.",
    )
    parser.add_argument(
        "--reuse-q5-interventions",
        action="store_true",
        default=False,
        help="A10: reuse Q5 intervention results.",
    )

    # ---- skip halves ----
    parser.add_argument("--skip-q2-q5", action="store_true", default=False,
                        help="Skip Q2~Q5 block (run appendix only).")
    parser.add_argument("--skip-appendix", action="store_true", default=False,
                        help="Skip A06~A10 block (run main questions only).")

    args = parser.parse_args()

    questions = {q.strip() for q in str(args.questions).split(",") if q.strip()}
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    if not datasets:
        print("[full-suite] ERROR: no datasets specified.")
        return 1

    common_args = _build_common_args(args)

    main_qs = questions & {"q2", "q3", "q4", "q5"}
    appendix_qs = questions & {"a06", "a07", "a08", "a09", "a10"}

    print(f"[full-suite] datasets ({len(datasets)}): {datasets}", flush=True)
    print(f"[full-suite] main questions : {sorted(main_qs)}", flush=True)
    print(f"[full-suite] appendix questions: {sorted(appendix_qs)}", flush=True)

    # ------------------------------------------------------------------
    # Dataset loop
    # ------------------------------------------------------------------
    for dataset in datasets:
        print(f"\n{'#'*70}", flush=True)
        print(f"[full-suite] ===  DATASET: {dataset}  ===", flush=True)
        print(f"{'#'*70}", flush=True)

        if main_qs and not args.skip_q2_q5:
            run_q2_q5_for_dataset(
                dataset,
                common_args,
                appendix=bool(args.appendix),
            )

        if appendix_qs and not args.skip_appendix:
            run_a06_a10_for_dataset(
                dataset,
                common_args,
                reuse_q2=bool(args.reuse_q2_checkpoints),
                reuse_q4=bool(args.reuse_q4_checkpoints),
                reuse_q5=bool(args.reuse_q5_interventions),
                questions=appendix_qs,
            )

    print(f"\n[full-suite] ALL DONE.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
