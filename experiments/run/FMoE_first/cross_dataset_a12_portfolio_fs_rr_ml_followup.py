#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import cross_dataset_a12_portfolio as portfolio
import stage1_a12_broad_templates as stage1


def _interleave_rows(groups: list[list[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    queues = [deque(group) for group in groups if group]
    out: list[Dict[str, Any]] = []
    while queues:
        next_queues: list[deque[Dict[str, Any]]] = []
        for queue in queues:
            if queue:
                out.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return out


def _slice_args(base: argparse.Namespace, *, dataset: str, count: int, start_index: int, max_evals: int, batch_size: int, eval_batch_size: int) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base))
    args.datasets = dataset
    args.dataset_template_counts = f"{dataset}:{int(count)}"
    args.template_start_index = int(start_index)
    args.max_evals = int(max_evals)
    args.batch_size = int(batch_size)
    args.eval_batch_size = int(eval_batch_size)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FMoE_N4 mixed queue follow-up for foursquare, retail_rocket, movielens1m")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--seed-base", type=int, default=264000)
    parser.add_argument("--search-algo", choices=["random", "tpe"], default="tpe")
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=10)
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--resume-from-logs", action="store_true", default=True)
    parser.add_argument("--no-resume-from-logs", dest="resume_from_logs", action="store_false")
    parser.add_argument("--verify-logging", dest="verify_logging", action="store_true")
    parser.add_argument("--no-verify-logging", dest="verify_logging", action="store_false")
    parser.set_defaults(verify_logging=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-max-runs", type=int, default=2)

    parser.add_argument("--foursquare-template-count", type=int, default=8)
    parser.add_argument("--foursquare-template-start-index", type=int, default=20)
    parser.add_argument("--foursquare-max-evals", type=int, default=8)
    parser.add_argument("--foursquare-batch-size", type=int, default=3072)
    parser.add_argument("--foursquare-eval-batch-size", type=int, default=4096)

    parser.add_argument("--retail-template-count", type=int, default=4)
    parser.add_argument("--retail-template-start-index", type=int, default=14)
    parser.add_argument("--retail-max-evals", type=int, default=6)
    parser.add_argument("--retail-batch-size", type=int, default=3072)
    parser.add_argument("--retail-eval-batch-size", type=int, default=4096)

    parser.add_argument("--movielens-template-count", type=int, default=4)
    parser.add_argument("--movielens-template-start-index", type=int, default=6)
    parser.add_argument("--movielens-max-evals", type=int, default=6)
    parser.add_argument("--movielens-batch-size", type=int, default=4096)
    parser.add_argument("--movielens-eval-batch-size", type=int, default=6144)

    args = parser.parse_args()
    if int(args.tune_epochs) < 1:
        raise RuntimeError("--tune-epochs must be >= 1")
    if int(args.tune_patience) < 0:
        raise RuntimeError("--tune-patience must be >= 0")
    return args


def main() -> int:
    args = parse_args()

    base = argparse.Namespace(
        datasets="",
        dataset_template_counts="",
        gpus=args.gpus,
        seeds=args.seeds,
        seed_base=args.seed_base,
        batch_size=0,
        eval_batch_size=0,
        search_algo=args.search_algo,
        max_evals=1,
        tune_epochs=args.tune_epochs,
        tune_patience=args.tune_patience,
        template_start_index=0,
        manifest_out=args.manifest_out,
        resume_from_logs=bool(args.resume_from_logs),
        verify_logging=bool(args.verify_logging),
        dry_run=bool(args.dry_run),
        smoke_test=bool(args.smoke_test),
        smoke_max_runs=int(args.smoke_max_runs),
    )

    fs_args = _slice_args(
        base,
        dataset="foursquare",
        count=args.foursquare_template_count,
        start_index=args.foursquare_template_start_index,
        max_evals=args.foursquare_max_evals,
        batch_size=args.foursquare_batch_size,
        eval_batch_size=args.foursquare_eval_batch_size,
    )
    retail_args = _slice_args(
        base,
        dataset="retail_rocket",
        count=args.retail_template_count,
        start_index=args.retail_template_start_index,
        max_evals=args.retail_max_evals,
        batch_size=args.retail_batch_size,
        eval_batch_size=args.retail_eval_batch_size,
    )
    ml_args = _slice_args(
        base,
        dataset="movielens1m",
        count=args.movielens_template_count,
        start_index=args.movielens_template_start_index,
        max_evals=args.movielens_max_evals,
        batch_size=args.movielens_batch_size,
        eval_batch_size=args.movielens_eval_batch_size,
    )

    row_groups = [
        portfolio.build_rows(fs_args),
        portfolio.build_rows(retail_args),
        portfolio.build_rows(ml_args),
    ]
    rows = _interleave_rows(row_groups)
    rows = portfolio.maybe_limit_smoke(rows, args)

    manifest_args = deepcopy(base)
    if not str(manifest_args.manifest_out).strip():
        manifest_args.manifest_out = str(portfolio.LOG_ROOT / "cross_dataset_fs_rr_ml_followup_manifest.json")
    manifest = portfolio.write_manifest(manifest_args, rows)
    print(f"[cross-dataset-followup] manifest -> {manifest}")

    fieldnames = stage1.build_summary_fieldnames(
        [
            "architecture_id",
            "architecture_name",
            "tuning_stage",
            "family_id",
            "family_group",
            "variant_id",
            "capacity_anchor",
            "selected_from_stage",
            "selection_score",
            "search_algo",
            "source_family_id",
            "stage_group",
            "template_count",
            "aux_route_consistency_lambda",
            "aux_z_loss_lambda",
        ]
    )
    gpus = stage1._parse_csv_strings(args.gpus)
    if not gpus:
        raise RuntimeError("No GPUs selected")

    return int(
        stage1.launch_wide_rows(
            rows=rows,
            gpus=gpus,
            args=args,
            axis=portfolio.AXIS,
            phase_id=portfolio.PHASE_ID,
            phase_name=portfolio.PHASE_NAME,
            log_dir=portfolio.LOG_ROOT,
            summary_path=portfolio.LOG_ROOT / "summary.csv",
            fieldnames=fieldnames,
            extra_cols=[
                col
                for col in fieldnames
                if col
                not in {
                    "global_best_valid_mrr20",
                    "run_best_valid_mrr20",
                    "run_phase",
                    "exp_brief",
                    "stage",
                    "trigger",
                    "dataset",
                    "seed_id",
                    "gpu_id",
                    "status",
                    "test_mrr20",
                    "n_completed",
                    "interrupted",
                    "special_ok",
                    "diag_ok",
                    "result_path",
                    "timestamp_utc",
                }
            ],
            build_command=stage1.build_command,
            build_log_path=stage1.build_log_path,
            verify_logging=bool(args.verify_logging),
            summary_path_for_row=lambda row: portfolio.summary_path(str(row["dataset"])),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())