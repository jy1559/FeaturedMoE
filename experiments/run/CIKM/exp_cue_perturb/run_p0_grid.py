#!/usr/bin/env python3
"""P0 grid search for exp_cue_perturb — KuaiRec & foursquare.

4개 고정 hparam 후보를 max-evals=1로 병렬 실행하고 test MRR@20 기준으로
최고 세팅을 main_routerec_summary.csv에 기록한다.

hparam 후보는 docs/hparams.md 앵커 기반으로 lr/wd 4가지 조합을 사용한다.

Usage:
    python run_p0_grid.py --gpus 0 1 2 3
    python run_p0_grid.py --gpus 0 1 --datasets KuaiRec
    python run_p0_grid.py --gpus 0 1 2 3 --max-evals 1   # 기본값, 각 후보 단일 실행
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXP_PERTURB = Path(__file__).resolve().parent
_CIKM_DIR  = _EXP_PERTURB.parent
_EXP_DIR   = _CIKM_DIR.parents[1]
_REPO_ROOT = _CIKM_DIR.parents[2]

for _p in [str(_CIKM_DIR), str(_EXP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import (  # noqa: E402
    FIXED_PARAMS, PYTHON_BIN, RESULT_ROOT,
    TUNE_EPOCHS, TUNE_PATIENCE,
)

CIKM_ARTIFACTS_RESULTS = _EXP_PERTURB / "artifacts" / "results"
CIKM_ARTIFACTS_LOGS    = _EXP_PERTURB / "artifacts" / "logs"
LOG_DIR = _EXP_PERTURB / "logs" / "train"
_DATASET_ROOT = _REPO_ROOT / "Datasets" / "processed"

DATASET_TUNE_CFG = {
    "KuaiRec":    "tune_kuai_cikm",
    "foursquare": "tune_foursq_cikm",
}

# ── 4개 후보 hparam 세팅 (docs/hparams.md 앵커 기반) ─────────────────────────
# 각 후보는 lr + wd 조합만 바꾸고, arch는 FIXED_PARAMS에서 가져옴.
# - KuaiRec:    anchor lr≈3.7e-4, 주변 4개 탐색
# - foursquare: anchor lr≈5e-4,   주변 4개 탐색
GRID_CANDIDATES: dict[str, list[dict[str, Any]]] = {
    "KuaiRec": [
        {"learning_rate": 2.5e-4, "weight_decay": 0.0},
        {"learning_rate": 4.0e-4, "weight_decay": 0.0},
        {"learning_rate": 6.0e-4, "weight_decay": 0.0},
        {"learning_rate": 4.0e-4, "weight_decay": 1e-6},
    ],
    "foursquare": [
        {"learning_rate": 3.0e-4, "weight_decay": 0.0},
        {"learning_rate": 5.0e-4, "weight_decay": 0.0},
        {"learning_rate": 7.5e-4, "weight_decay": 0.0},
        {"learning_rate": 5.0e-4, "weight_decay": 5e-7},
    ],
}


def _hydra_singleton(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def build_cmd(dataset: str, candidate: dict[str, Any], gpu_id: str, max_evals: int) -> list[str]:
    cfg_name = DATASET_TUNE_CFG[dataset]
    fixed    = dict(FIXED_PARAMS.get(dataset, {}).get("featured_moe_n3", {}))
    lr       = candidate["learning_rate"]
    wd       = candidate["weight_decay"]

    cmd: list[str] = [
        PYTHON_BIN, "hyperopt_tune.py",
        "--config-name", cfg_name,
        "--search-algo", "tpe",
        "--max-evals", str(max_evals),
        "--tune-epochs", str(TUNE_EPOCHS),
        "--tune-patience", str(TUNE_PATIENCE),
        "--seed", "42",
        "--run-group", "cikm",
        "--run-axis", "cikm_cue_perturb_p0_grid",
        "--run-phase", "P0",
        "model=featured_moe_n3_tune",
        f"dataset={dataset}",
        "eval_mode=session_fixed",
        "feature_mode=final",
        f"++dataset={dataset}",
        f"++dataset_root={_DATASET_ROOT}",
        f"gpu_id={gpu_id}",
        "log_wandb=false",
        "show_progress=false",
        "++seed=42",
        "++eval_sampling.mode=full",
        "++eval_sampling.auto_full_threshold=999999999",
        "++exclude_unseen_target_from_main_eval=true",
        # lr/wd singleton 고정
        f"++search.learning_rate={lr:g}",
        f"++search.weight_decay={wd:g}",
    ]

    # arch params 고정 (MAX_ITEM_LIST_LENGTH은 ++로만, search에는 넣지 않음)
    _SKIP_SEARCH = {"MAX_ITEM_LIST_LENGTH"}
    for k, v in fixed.items():
        sv = _hydra_singleton(v)
        cmd.append(f"++{k}={sv}")
        if k not in _SKIP_SEARCH:
            cmd.append(f"++search.{k}={sv}")

    return cmd


def run_candidate(
    *,
    dataset: str,
    candidate: dict[str, Any],
    candidate_idx: int,
    gpu_id: str,
    max_evals: int,
) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"lr{candidate['learning_rate']:g}_wd{candidate['weight_decay']:g}"
    log_path = LOG_DIR / f"{dataset}_c{candidate_idx}_{tag}.log"
    cmd = build_cmd(dataset, candidate, gpu_id, max_evals)

    label = f"[{dataset}/c{candidate_idx} lr={candidate['learning_rate']:.1e} wd={candidate['weight_decay']:.0e}]"
    print(f"  {label} 시작  gpu={gpu_id}", flush=True)
    start = time.time()

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"   # epoch line이 파일에 즉시 flush되게
    CIKM_ARTIFACTS_RESULTS.mkdir(parents=True, exist_ok=True)
    CIKM_ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)
    env["HYPEROPT_RESULTS_DIR"] = str(CIKM_ARTIFACTS_RESULTS)
    env["RUN_LOGS_DIR"]         = str(CIKM_ARTIFACTS_LOGS)

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {label}  gpu={gpu_id}\n")
        fh.write(f"# cmd: {' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(_EXP_DIR),
            stdout=fh, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        proc.wait()

    elapsed = time.time() - start

    # result JSON 파싱
    result_path_str = ""
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        for pat in [r"Results\s*->\s*(.+\.json)", r"result saved to[:\s]+(.+\.json)"]:
            m = re.search(pat, log_text, re.IGNORECASE)
            if m:
                result_path_str = m.group(1).strip()
                break
    except Exception:
        pass

    test_mrr20 = 0.0
    ckpt_path  = ""
    payload    = {}
    status     = "error"

    if result_path_str and Path(result_path_str).exists():
        try:
            payload = json.loads(Path(result_path_str).read_text(encoding="utf-8"))
            tr = payload.get("test_result", {}) or {}
            test_mrr20 = float(tr.get("mrr@20", 0) or 0)
            for key in ("best_checkpoint_file", "artifact_best_checkpoint"):
                ck = str(payload.get(key, "") or "").strip()
                if ck and Path(ck).exists():
                    ckpt_path = ck
                    break
            status = "ok" if test_mrr20 > 0 else "completed"
        except Exception:
            pass
    elif proc.returncode != 0:
        status = f"error_rc{proc.returncode}"

    print(f"  {label} 완료  MRR@20={test_mrr20:.4f}  elapsed={elapsed/60:.1f}min  rc={proc.returncode}", flush=True)

    return {
        "dataset": dataset,
        "candidate_idx": candidate_idx,
        "learning_rate": candidate["learning_rate"],
        "weight_decay":  candidate["weight_decay"],
        "gpu_id": gpu_id,
        "status": status,
        "test_mrr20": test_mrr20,
        "ckpt_path": ckpt_path,
        "result_path": result_path_str,
        "log_path": str(log_path),
        "elapsed_sec": round(elapsed, 1),
        "payload": payload,
    }


def collect_dataset_results(all_results: list[dict], dataset: str) -> dict[str, Any] | None:
    """dataset별 결과 정렬·출력·기록."""
    results = sorted(
        [r for r in all_results if r["dataset"] == dataset],
        key=lambda r: -r["test_mrr20"],
    )
    if not results:
        return None

    print(f"\n[{dataset}] 결과 순위:", flush=True)
    for i, r in enumerate(results):
        marker = " ★" if i == 0 else ""
        print(f"  [{i+1}] lr={r['learning_rate']:.1e}  wd={r['weight_decay']:.0e}"
              f"  MRR@20={r['test_mrr20']:.4f}  {r['status']}{marker}", flush=True)

    best = results[0]
    if best["test_mrr20"] <= 0:
        print(f"  [WARN] 모든 후보 MRR@20=0, 최고 후보 선택 불가", flush=True)
        return None

    _record_best(dataset=dataset, result=best)
    print(f"\n  → best: lr={best['learning_rate']:.1e}  MRR@20={best['test_mrr20']:.4f}", flush=True)
    print(f"    ckpt: {best['ckpt_path'][-65:] if best['ckpt_path'] else 'NONE'}", flush=True)
    return best


def _record_best(*, dataset: str, result: dict[str, Any]) -> None:
    from common import SUMMARY_FIELDS, append_csv, now_utc, parse_metrics
    payload = result.get("payload", {})
    metrics = parse_metrics(payload)
    row = {
        "dataset": dataset,
        "model": "featured_moe_n3",
        "job_id": f"{dataset}_fmoe_p0_grid_c{result['candidate_idx']}",
        "gpu_id": result["gpu_id"],
        "status": result["status"],
        **metrics,
        "result_path": result["result_path"],
        "log_path": result["log_path"],
        "elapsed_sec": result["elapsed_sec"],
        "error": "",
        "timestamp_utc": now_utc(),
    }
    append_csv(RESULT_ROOT / "main_routerec_summary.csv", SUMMARY_FIELDS, row)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="P0 grid search for exp_cue_perturb")
    parser.add_argument("--gpus", nargs="+", default=["0"], help="GPU IDs")
    parser.add_argument("--datasets", nargs="+", default=["KuaiRec", "foursquare"])
    parser.add_argument("--max-evals", type=int, default=1,
                        help="max-evals per candidate (1 = single fixed run)")
    args = parser.parse_args()

    gpus    = args.gpus
    datasets = args.datasets
    max_evals = args.max_evals

    n_cands = len(GRID_CANDIDATES.get(datasets[0], []))
    print("══════════════════════════════════════════════════════", flush=True)
    print(f"  P0 Grid Search — {datasets}", flush=True)
    print(f"  GPUs={gpus}  후보 {n_cands}개/데이터셋  max-evals={max_evals}", flush=True)
    print(f"  실행 순서: 데이터셋별 순차, 각 데이터셋 내 {len(gpus)}개 GPU 병렬", flush=True)
    print("══════════════════════════════════════════════════════", flush=True)

    # 데이터셋별 순차 실행 (GPU OOM 방지 — 같은 GPU에 2개 job 금지)
    # 각 데이터셋: 후보 4개를 GPU 4개에 1:1 배정하여 동시 실행
    all_results: list[dict] = []
    for ds in datasets:
        candidates = GRID_CANDIDATES.get(ds, [])
        jobs = [
            {
                "dataset": ds,
                "candidate": c,
                "candidate_idx": i,
                "gpu_id": gpus[i % len(gpus)],
                "max_evals": max_evals,
            }
            for i, c in enumerate(candidates)
        ]

        print(f"\n  [{ds}] GPU 배정:", flush=True)
        for j in jobs:
            print(f"    GPU {j['gpu_id']} ← c{j['candidate_idx']}"
                  f"  lr={j['candidate']['learning_rate']:.1e}"
                  f"  wd={j['candidate']['weight_decay']:.0e}", flush=True)
        print(flush=True)

        ds_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futures = {ex.submit(run_candidate, **j): j for j in jobs}
            for fut in as_completed(futures):
                ds_results.append(fut.result())

        all_results.extend(ds_results)
        print("\n" + "="*60, flush=True)
        collect_dataset_results(ds_results, ds)

    print("\n══════════════════════════════════════════════════════", flush=True)
    print("  P0 Grid Search 완료", flush=True)
    print(f"  결과 → {RESULT_ROOT}/main_routerec_summary.csv", flush=True)
    print("══════════════════════════════════════════════════════", flush=True)


if __name__ == "__main__":
    main()
