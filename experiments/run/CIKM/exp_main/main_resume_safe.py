#!/usr/bin/env python3
"""Resume CIKM runs safely across KuaiRec and LastFM.

Policy:
  - skip jobs that already have a clean normal-ending log
  - put all pending jobs from KuaiRec and LastFM into one GPU queue
  - run different models on different GPUs in parallel
  - keep LastFM per-job batches conservative, with OOM retry enabled
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_CIKM_DIR = Path(__file__).resolve().parent.parent
if str(_CIKM_DIR) not in sys.path:
    sys.path.insert(0, str(_CIKM_DIR))

from common import (  # noqa: E402
    BASELINE_MODELS,
    EXP_DIR,
    LIGHT_DATA_ROOT,
    LOG_ROOT,
    RESULT_ROOT,
    ROUTE_MODEL,
    run_jobs_queued,
    run_jobs_resource_aware,
    sanitize,
)

ALL_MODELS = [*BASELINE_MODELS, ROUTE_MODEL]
SUMMARY_CSV = RESULT_ROOT / "resume_safe_summary.csv"
LFM_CACHE_ROOT = EXP_DIR / "saved" / "recbole_cache" / "cikm_final"
LFM_LIGHT_CACHE_ROOT = EXP_DIR / "saved" / "recbole_cache" / "cikm_light"
LFM_LIGHT_NOITEM_CACHE_ROOT = EXP_DIR / "saved" / "recbole_cache" / "cikm_light_noitem"
LFM_SIDEINFO_MODELS = {"difsr", "fdsa"}

BAD_LOG_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"Traceback \(most recent call last\)",
        r"\bFAILED\b",
        r"->\s*FAILED",
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"\[OOM_RETRY\].*exhausted",
        r"KeyError:",
        r"RuntimeError:",
        r"\brc=-?\d+",
    ]
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume CIKM KuaiRec+LastFM jobs safely")
    p.add_argument("gpus", nargs="+", help="GPU IDs, e.g. 0 1 2 3")
    p.add_argument("--datasets", nargs="+", default=["KuaiRec", "lastfm"], choices=["KuaiRec", "lastfm"])
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--lfm-parallel", type=int, default=4, help="Max concurrent LastFM light-data jobs")
    p.add_argument("--max-evals", type=int, default=1, help="Trials for pending jobs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lfm-train-batch-size", type=int, default=1024)
    p.add_argument("--lfm-eval-batch-size", type=int, default=256)
    p.add_argument("--lfm-max-len", type=int, default=10)
    p.add_argument("--lfm-eval-sample-num", type=int, default=1000)
    p.add_argument("--oom-retry-limit", type=int, default=5)
    p.add_argument("--run-axis", default="cikm_resume_safe_parallel")
    p.add_argument("--run-phase", default="P0_RESUME")
    p.add_argument("--dry-run", action="store_true", help="Only print skip/run plan")
    return p.parse_args()


def _log_is_clean_complete(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if "[RUN_STATUS] END status=normal" not in text:
        return False
    return not any(pattern.search(text) for pattern in BAD_LOG_PATTERNS)


def _matching_logs(dataset: str, model: str) -> list[Path]:
    job_id = sanitize(f"{dataset}_{model}")
    return sorted(LOG_ROOT.rglob(f"{job_id}.log"), key=lambda p: p.stat().st_mtime, reverse=True)


def completed_clean(dataset: str, model: str) -> tuple[bool, str]:
    for path in _matching_logs(dataset, model):
        if _log_is_clean_complete(path):
            return True, str(path)
    return False, ""


def stable_lfm_overrides(args: argparse.Namespace, model: str) -> list[str]:
    overrides = [
        f"max_evals={args.max_evals}",
        f"tune_epochs={args.epochs}",
        f"tune_patience={args.patience}",
        f"oom_retry_limit={args.oom_retry_limit}",
        f"train_batch_size={args.lfm_train_batch_size}",
        f"eval_batch_size={args.lfm_eval_batch_size}",
        "enable_data_cache=true",
        "enable_disk_data_cache=true",
        "enable_session_split_cache=true",
        "in_memory_data_cache=false",
        "cache_dataloaders=false",
        "save_dataloaders=false",
        f"large_dataset_cache_anchor_len={args.lfm_max_len}",
        f"++MAX_ITEM_LIST_LENGTH={args.lfm_max_len}",
        "++eval_sampling.mode=auto",
        "++eval_sampling.auto_full_threshold=100000",
        f"++eval_sampling.sample_num={args.lfm_eval_sample_num}",
        "++log_unseen_target_metrics=false",
    ]
    if model == ROUTE_MODEL or model in LFM_SIDEINFO_MODELS:
        if model == ROUTE_MODEL:
            overrides.append(f"data_cache_dir={LFM_CACHE_ROOT}")
        else:
            overrides.extend([
                f"++data_path={LIGHT_DATA_ROOT}",
                f"data_cache_dir={LFM_LIGHT_CACHE_ROOT}",
            ])
    else:
        overrides.extend([
            f"++data_path={LIGHT_DATA_ROOT}",
            f"data_cache_dir={LFM_LIGHT_NOITEM_CACHE_ROOT}",
            "load_col.item=[]",
        ])
    return overrides


def kuai_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"max_evals={args.max_evals}",
        f"tune_epochs={args.epochs}",
        f"tune_patience={args.patience}",
        f"oom_retry_limit={args.oom_retry_limit}",
    ]


def pending_jobs(dataset: str, models: list[str], overrides) -> list[dict]:
    jobs: list[dict] = []
    print(f"\n[{dataset}] skip check", flush=True)
    for model in models:
        ok, path = completed_clean(dataset, model)
        if ok:
            print(f"  [SKIP] {dataset}/{model} clean log: {path}", flush=True)
            continue
        print(f"  [RUN ] {dataset}/{model}", flush=True)
        model_overrides = overrides(model) if callable(overrides) else overrides
        jobs.append({"dataset": dataset, "model": model, "extra_overrides": list(model_overrides)})
    return jobs


def main() -> None:
    args = parse_args()
    gpus = [str(g) for g in args.gpus]
    print("=================================================================", flush=True)
    print("  CIKM resume-safe runner", flush=True)
    print(f"  GPUs          : {' '.join(gpus)}", flush=True)
    print(f"  Scheduler     : one pending-job queue across all GPUs", flush=True)
    print(f"  max_evals     : {args.max_evals}", flush=True)
    print(f"  LastFM batch  : train={args.lfm_train_batch_size} eval={args.lfm_eval_batch_size}", flush=True)
    print(f"  LastFM parallel cap: {args.lfm_parallel}", flush=True)
    print(f"  OOM retry     : {args.oom_retry_limit}", flush=True)
    print(f"  summary       : {SUMMARY_CSV}", flush=True)
    print("=================================================================", flush=True)

    all_jobs: list[dict] = []
    selected_models = [m for m in ALL_MODELS if m in set(args.models)]

    if "KuaiRec" in args.datasets:
        all_jobs.extend(pending_jobs("KuaiRec", selected_models, kuai_overrides(args)))
    if "lastfm" in args.datasets:
        all_jobs.extend(pending_jobs("lastfm", selected_models, lambda model: stable_lfm_overrides(args, model)))

    if args.dry_run:
        print("\n[DRY-RUN] no jobs launched", flush=True)
        print(f"  total pending: {len(all_jobs)}", flush=True)
        return

    route_lfm_jobs = [
        job for job in all_jobs
        if job["dataset"] == "lastfm" and job["model"] == ROUTE_MODEL
    ]
    main_jobs = [
        job for job in all_jobs
        if not (job["dataset"] == "lastfm" and job["model"] == ROUTE_MODEL)
    ]

    capped_jobs = list(main_jobs)

    if capped_jobs:
        print(
            f"\n[RUN] pending capped jobs: {len(capped_jobs)} across {len(gpus)} GPUs "
            f"(LastFM cap={args.lfm_parallel})",
            flush=True,
        )
        run_jobs_resource_aware(
            capped_jobs,
            gpus=gpus,
            summary_path=SUMMARY_CSV,
            run_axis=args.run_axis,
            run_phase=args.run_phase,
            dataset_parallel_limits={"lastfm": args.lfm_parallel},
        )
    if route_lfm_jobs:
        print(
            f"\n[RUN] LastFM RouteRec feature job isolated: {len(route_lfm_jobs)} job on gpu={gpus[0]}",
            flush=True,
        )
        run_jobs_queued(
            route_lfm_jobs,
            gpus=[gpus[0]],
            summary_path=SUMMARY_CSV,
            run_axis=args.run_axis,
            run_phase=args.run_phase,
        )
    if all_jobs:
        pass
    else:
        print("\n[SKIP] no pending jobs", flush=True)

    print(f"\n[DONE] resume-safe runner. Summary -> {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
