#!/usr/bin/env python3
"""Rerun one selected real-final route config with checkpoint export, then case-eval it."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun one q3 hierarchical-sparse config with checkpoint export.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-seed", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--groups", default="memory_plus,focus_plus,tempo_plus")
    parser.add_argument("--max-evals", type=int, default=5)
    parser.add_argument("--max-run-hours", type=float, default=1.0)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--lr-mode", default="narrow_loguniform")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--use-best-fixed-lr", action="store_true")
    parser.add_argument("--search-algo", default="tpe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = common.REPO_ROOT

    candidates = common.load_base_candidates(
        Path(common.DEFAULT_BASE_CSV),
        datasets=[str(args.dataset)],
        models=[common.ROUTE_MODEL],
        top_k_configs=int(args.base_rank),
    )
    try:
        candidate = next(c for c in candidates if c.rank == int(args.base_rank))
    except StopIteration as exc:
        raise RuntimeError(f"No base candidate found for dataset={args.dataset} rank={args.base_rank}") from exc

    setting = next(s for s in common.q3_settings() if s["setting_key"] == "hierarchical_sparse")
    row = common.build_route_row(
        question="q3",
        candidate=candidate,
        setting=setting,
        seed=int(args.seed),
        runtime_seed=int(args.runtime_seed),
        max_evals=int(args.max_evals),
        max_run_hours=float(args.max_run_hours),
        tune_epochs=int(args.tune_epochs),
        tune_patience=int(args.tune_patience),
        lr_mode=str(args.lr_mode),
    )
    row["job_id"] = f"{row['job_id']}_CKPTRERUN"
    row["run_phase"] = row["job_id"]

    fixed_lr = args.learning_rate
    if fixed_lr is None and args.use_best_fixed_lr:
        fixed_lr = candidate.payload.get("best_params", {}).get("learning_rate")
    if fixed_lr is not None:
        row["fixed_context"]["learning_rate"] = float(fixed_lr)
        row["search_space"] = {}
        row["search_space_types"] = {}
        row["max_evals"] = 1
        print(f"[train] using fixed learning_rate={float(fixed_lr):.12g}", flush=True)

    cmd = common.build_route_command(row, str(args.gpu), search_algo=str(args.search_algo))
    cmd.append("++artifact_export_final_checkpoint=true")

    print(f"[train] launching {row['run_phase']}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(repo_root / "experiments"))

    result_path = common._find_result_payload_for_row(row)
    if result_path is None:
        raise RuntimeError(f"No result payload found for run_phase={row['run_phase']}")
    payload = common.read_json(result_path)
    checkpoint_file = str(payload.get("best_checkpoint_file", "") or "").strip()
    if not checkpoint_file:
        raise RuntimeError(f"No exported checkpoint in result payload: {result_path}")
    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"[train] result {result_path}", flush=True)
    print(f"[train] checkpoint {checkpoint_path}", flush=True)

    case_root = common.LOG_ROOT / "q5" / "case_eval_rerun"
    case_root.mkdir(parents=True, exist_ok=True)
    manifest = case_root / str(args.bundle_name) / "case_eval_manifest.csv"

    eval_cmd = [
        common.python_bin(),
        str(repo_root / "experiments" / "run" / "fmoe_n4" / "eval_checkpoint_case_subsets.py"),
        "--source-result-json",
        str(result_path),
        "--checkpoint-file",
        str(checkpoint_path),
        "--output-root",
        str(case_root),
        "--bundle-name",
        str(args.bundle_name),
        "--resume",
        "--groups",
        str(args.groups),
    ]
    print(f"[case-eval] launching {args.bundle_name}", flush=True)
    subprocess.run(eval_cmd, check=True, cwd=str(repo_root))

    if not manifest.exists():
        raise FileNotFoundError(f"Case-eval manifest not found: {manifest}")

    export_dir = case_root / str(args.bundle_name) / "tables"
    export_cmd = [
        common.python_bin(),
        str(repo_root / "experiments" / "run" / "fmoe_n4" / "export_case_eval_tables.py"),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(export_dir),
    ]
    print(f"[export] launching {args.bundle_name}", flush=True)
    subprocess.run(export_cmd, check=True, cwd=str(repo_root))
    print(f"[done] {args.bundle_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())