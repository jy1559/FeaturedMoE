#!/usr/bin/env python3
"""Train-time cue perturbation ablation – CIKM 2026 (A12 architecture).

## 실행 흐름

1. P0 hparam 결정
   - p0_a12_summary.csv에서 test_mrr20 상위 --top-k개 선택 (eval_perturb가 먼저 돌았다면)
   - 없으면 artifact_anchor_catalog.csv top-1 hparam 사용

2. 각 조건 × 데이터셋 × (선택된 hparam)을 A12로 hyperopt_tune.py --max-evals 1 학습
   - GPU global queue로 병렬 실행

3. 결과를 results/train_summary.csv에 누적

Usage:
    python train_perturb.py --gpus 0 1 2 3
    python train_perturb.py --gpus 0 1 2 3 --datasets foursquare
    python train_perturb.py --gpus 0 1 2 3 --top-k 2 --conditions hidden_only feature_only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
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
    NARROW_SEARCH, PYTHON_BIN,
    TUNE_EPOCHS, TUNE_PATIENCE,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_DATASET_ROOT = _REPO_ROOT / "Datasets" / "processed"
_CATALOG_PATH = _CIKM_DIR / "docs" / "artifact_anchor_catalog.csv"

P0_SUMMARY_CSV   = _EXP_PERTURB / "results" / "p0_a12_summary.csv"
TRAIN_OUTPUT_CSV = _EXP_PERTURB / "results" / "train_summary.csv"
LOG_DIR          = _EXP_PERTURB / "logs" / "train"

CIKM_ARTIFACTS_RESULTS = _EXP_PERTURB / "artifacts" / "results"
CIKM_ARTIFACTS_LOGS    = _EXP_PERTURB / "artifacts" / "logs"

# ── 데이터셋 설정 ─────────────────────────────────────────────────────────────
DATASETS = ["foursquare", "KuaiRec"]

DATASET_TUNE_CFG = {
    "KuaiRec":    "tune_kuai_cikm",
    "foursquare": "tune_foursq_cikm",
}

# ── A12 architecture helpers ──────────────────────────────────────────────────

def _all_stage_map(value: Any) -> dict[str, Any]:
    return {"macro": value, "mid": value, "micro": value}


def _hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ",".join(_hydra_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_hydra_literal(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Cannot convert to Hydra literal: {type(value).__name__}")


def _a12_primitives(source: str = "feature") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stage in ("macro", "mid", "micro"):
        stage_cfg: dict[str, Any] = {}
        for prim in ("a_joint", "b_group", "c_shared", "d_cond", "e_scalar"):
            stage_cfg[prim] = {"source": source, "temperature": 1.0, "top_k": 0}
        stage_cfg["wrapper"] = {"alpha_d": 1.0, "alpha_struct": 1.0, "alpha_a": 1.0}
        out[stage] = stage_cfg
    return out


_A12_ARCH_OVERRIDES: list[str] = [
    "++fmoe_architecture_id=A12",
    "++fmoe_architecture_key=A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
    f'++layer_layout={_hydra_literal(["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"])}',
    f'++stage_compute_mode={_hydra_literal(_all_stage_map("moe"))}',
    f'++stage_router_mode={_hydra_literal(_all_stage_map("learned"))}',
    f'++stage_router_wrapper={_hydra_literal(_all_stage_map("w5_exd"))}',
    f'++stage_router_granularity={_hydra_literal({"macro": "session", "mid": "session", "micro": "token"})}',
    f'++stage_feature_injection={_hydra_literal(_all_stage_map("none"))}',
    f'++stage_feature_dropout_scope={_hydra_literal(_all_stage_map("token"))}',
    "++topk_scope_mode=global_flat",
    "++moe_top_k=0",
    "++balance_loss_lambda=0.0",
    "++gate_entropy_lambda=0.0",
    "++route_smoothness_lambda=0.0",
    "++route_sharpness_lambda=0.0",
    "++route_monopoly_lambda=0.0",
    "++route_prior_lambda=0.0",
    "++group_prior_align_lambda=0.0",
    "++factored_group_balance_lambda=0.0",
    "++rule_agreement_lambda=0.0",
    "++group_coverage_lambda=0.0",
    "++feature_group_bias_lambda=0.0",
    "++rule_bias_scale=0.0",
    "++bias_mode=none",
    "++route_consistency_pairs=1",
    "++route_consistency_lambda=0.0",
    "++macro_history_window=5",
]

# ── Train conditions ──────────────────────────────────────────────────────────
# 각 조건:
#   arch_overrides: stage_router_source / primitives source 변경
#   perturb_overrides: feature_perturb 설정

TRAIN_CONDITIONS: dict[str, dict[str, Any]] = {
    "hidden_only": {
        "description": "stage_router_source=hidden → hidden state만으로 routing (A12)",
        "arch_overrides": [
            f'++stage_router_source={_hydra_literal(_all_stage_map("hidden"))}',
            f'++stage_router_primitives={_hydra_literal(_a12_primitives("hidden"))}',
        ],
        "perturb_overrides": [],
    },
    "feature_only": {
        "description": "stage_router_source=feature → cue만으로 routing (A12 기본값)",
        "arch_overrides": [
            f'++stage_router_source={_hydra_literal(_all_stage_map("feature"))}',
            f'++stage_router_primitives={_hydra_literal(_a12_primitives("feature"))}',
        ],
        "perturb_overrides": [],
    },
    "train_zero": {
        "description": "학습 중 cue zero → eval 시 정상 cue",
        "arch_overrides": [
            f'++stage_router_source={_hydra_literal(_all_stage_map("feature"))}',
            f'++stage_router_primitives={_hydra_literal(_a12_primitives("feature"))}',
        ],
        "perturb_overrides": [
            "++feature_perturb_mode=zero",
            "++feature_perturb_apply=train",
        ],
    },
    "both_zero": {
        "description": "학습+eval 모두 cue zero → MoE 구조 자체의 기여",
        "arch_overrides": [
            f'++stage_router_source={_hydra_literal(_all_stage_map("feature"))}',
            f'++stage_router_primitives={_hydra_literal(_a12_primitives("feature"))}',
        ],
        "perturb_overrides": [
            "++feature_perturb_mode=zero",
            "++feature_perturb_apply=both",
        ],
    },
    "train_shuffle": {
        "description": "학습 중 cue shuffle → eval 시 정상",
        "arch_overrides": [
            f'++stage_router_source={_hydra_literal(_all_stage_map("feature"))}',
            f'++stage_router_primitives={_hydra_literal(_a12_primitives("feature"))}',
        ],
        "perturb_overrides": [
            "++feature_perturb_mode=shuffle",
            "++feature_perturb_apply=train",
        ],
    },
}

TRAIN_FIELDS = [
    "dataset", "model", "condition", "hparam_source", "group",
    "valid_mrr20", "test_mrr20",
    "valid_hr1", "test_hr1",
    "valid_hr10", "test_hr10",
    "valid_ndcg1", "test_ndcg1",
    "valid_ndcg10", "test_ndcg10",
    "lr", "hidden_size", "d_expert_hidden",
    "elapsed_sec", "status", "log_path", "result_path", "timestamp_utc",
]

# ── CSV helper ────────────────────────────────────────────────────────────────
_csv_lock = threading.Lock()


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    with _csv_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fv(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


# ── hparam 선택 ───────────────────────────────────────────────────────────────

def get_hparams_for_dataset(dataset: str, top_k: int) -> list[dict[str, Any]]:
    """p0_a12_summary.csv → 없으면 catalog top-1 순으로 hparam 결정.

    각 hparam에 대해 모든 조건을 돌리므로, top_k개가 반환됨.
    """
    # 1) p0_a12_summary.csv에서 best-k
    if P0_SUMMARY_CSV.exists():
        rows = list(csv.DictReader(P0_SUMMARY_CSV.open(encoding="utf-8")))
        ds_rows = [r for r in rows if r.get("dataset") == dataset and
                   _fv(r, "test_mrr20") > 0]
        if ds_rows:
            ds_rows.sort(key=lambda r: _fv(r, "test_mrr20"), reverse=True)
            result = []
            for rank_idx, r in enumerate(ds_rows[:top_k], 1):
                result.append({
                    "hparam_source": f"p0_summary_rank{rank_idx}",
                    "lr":             _fv(r, "lr"),
                    "wd":             _fv(r, "wd"),
                    "hidden_size":    int(_fv(r, "hidden_size", 128)),
                    "d_ff":           int(_fv(r, "hidden_size", 128)) * 2,
                    "d_expert_hidden": int(_fv(r, "d_expert_hidden", 128)),
                    "d_router_hidden": int(_fv(r, "d_router_hidden", 64)),
                    "d_feat_emb":     16,
                    "expert_scale":   4,
                    "MAX_ITEM_LIST_LENGTH": 10,
                    "hidden_dropout_prob": 0.10,
                })
            print(f"  [hparam] {dataset}: p0_a12_summary에서 {len(result)}개 선택", flush=True)
            return result

    # 2) catalog top-1 fallback
    if not _CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found: {_CATALOG_PATH}")

    catalog_rows = list(csv.DictReader(_CATALOG_PATH.open(encoding="utf-8")))
    ds_rows = [
        r for r in catalog_rows
        if r.get("model") == "featured_moe_n3"
        and r.get("family") == "route"
        and r.get("dataset_target") == dataset
    ]
    if not ds_rows:
        raise ValueError(f"No featured_moe_n3 route entries for {dataset} in catalog")

    ds_rows.sort(key=lambda r: _fv(r, "test_mrr20"), reverse=True)
    top = ds_rows[:top_k]
    result = []
    for rank_idx, r in enumerate(top, 1):
        lr = _fv(r, "learning_rate")
        if lr <= 0:
            ns = NARROW_SEARCH.get(dataset, {}).get("featured_moe_n3", {})
            lrs = ns.get("learning_rate", [1e-3])
            lr = lrs[len(lrs) // 2]
        hidden_size = int(_fv(r, "hidden_size", 128))
        result.append({
            "hparam_source": f"catalog_rank{int(_fv(r, 'candidate_rank', rank_idx))}",
            "lr":             lr,
            "wd":             _fv(r, "weight_decay"),
            "hidden_size":    hidden_size,
            "d_ff":           int(_fv(r, "d_ff", hidden_size * 2)),
            "d_expert_hidden": int(_fv(r, "d_expert_hidden", hidden_size)),
            "d_router_hidden": int(_fv(r, "d_router_hidden", 64)),
            "d_feat_emb":     int(_fv(r, "d_feat_emb", 16)),
            "expert_scale":   int(_fv(r, "expert_scale", 4)),
            "MAX_ITEM_LIST_LENGTH": int(_fv(r, "MAX_ITEM_LIST_LENGTH", 10)),
            "hidden_dropout_prob": _fv(r, "hidden_dropout_prob", 0.10),
        })
    print(f"  [hparam] {dataset}: catalog에서 {len(result)}개 fallback 선택", flush=True)
    return result


# ── single condition run ──────────────────────────────────────────────────────

def run_condition_single(
    *,
    dataset: str,
    condition: str,
    condition_cfg: dict[str, Any],
    hparam: dict[str, Any],
    gpu_id: str,
) -> dict[str, Any]:
    hparam_src = hparam["hparam_source"]
    log_name   = f"{dataset}_{condition}_{hparam_src}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path   = LOG_DIR / f"{log_name}.log"
    cfg_name   = DATASET_TUNE_CFG.get(dataset, "tune_kuai_cikm")

    cmd: list[str] = [
        PYTHON_BIN, "hyperopt_tune.py",
        "--config-name", cfg_name,
        "--search-algo", "tpe",
        "--max-evals", "1",
        "--tune-epochs", str(TUNE_EPOCHS),
        "--tune-patience", str(TUNE_PATIENCE),
        "--seed", "42",
        "--run-group", "cikm",
        "--run-axis", "cikm_cue_perturb_train",
        "--run-phase", "P1-cue",
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
        # lr/wd singleton
        f"++search.learning_rate={hparam['lr']:g}",
        f"++search.weight_decay={hparam['wd']:g}",
        # arch size
        f"++MAX_ITEM_LIST_LENGTH={hparam['MAX_ITEM_LIST_LENGTH']}",
        f"++hidden_size={hparam['hidden_size']}",
        f"++d_ff={hparam['d_ff']}",
        f"++d_expert_hidden={hparam['d_expert_hidden']}",
        f"++d_router_hidden={hparam['d_router_hidden']}",
        f"++d_feat_emb={hparam['d_feat_emb']}",
        f"++expert_scale={hparam['expert_scale']}",
        f"++hidden_dropout_prob={hparam['hidden_dropout_prob']:g}",
        # fix search space singletons
        f"++search.hidden_size={hparam['hidden_size']}",
        f"++search.d_ff={hparam['d_ff']}",
        f"++search.d_expert_hidden={hparam['d_expert_hidden']}",
        f"++search.d_router_hidden={hparam['d_router_hidden']}",
        f"++search.d_feat_emb={hparam['d_feat_emb']}",
        f"++search.expert_scale={hparam['expert_scale']}",
        f"++search.hidden_dropout_prob={hparam['hidden_dropout_prob']:g}",
    ]
    # A12 arch
    cmd.extend(_A12_ARCH_OVERRIDES)
    # condition arch (router source)
    cmd.extend(condition_cfg["arch_overrides"])
    # condition perturb
    cmd.extend(condition_cfg["perturb_overrides"])

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    CIKM_ARTIFACTS_RESULTS.mkdir(parents=True, exist_ok=True)
    CIKM_ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)
    env["HYPEROPT_RESULTS_DIR"] = str(CIKM_ARTIFACTS_RESULTS)
    env["RUN_LOGS_DIR"]         = str(CIKM_ARTIFACTS_LOGS)

    print(f"  [{condition}/{hparam_src}] {dataset} gpu={gpu_id}  "
          f"lr={hparam['lr']:.2e} hidden={hparam['hidden_size']}", flush=True)
    start = time.time()

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# condition={condition}  dataset={dataset}  gpu={gpu_id}\n")
        fh.write(f"# hparam={hparam}\n")
        fh.write(f"# {condition_cfg['description']}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(_EXP_DIR),
            stdout=fh, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        proc.wait()

    elapsed = time.time() - start

    # result 파싱
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

    def _f(d: dict, k: str) -> float:
        return float(d.get(k, 0) or 0)

    best_valid, test_r = {}, {}
    status = "error"
    if result_path_str and Path(result_path_str).exists():
        try:
            payload    = json.loads(Path(result_path_str).read_text(encoding="utf-8"))
            best_valid = payload.get("best_valid_result", {}) or {}
            test_r     = payload.get("test_result", {}) or {}
            status = "ok" if _f(test_r, "mrr@20") > 0 else (
                "completed" if proc.returncode == 0 else "error"
            )
        except Exception:
            pass
    elif proc.returncode != 0:
        status = f"error_rc{proc.returncode}"

    row = {
        "dataset": dataset,
        "model": "FeaturedMoE_N3_A12",
        "condition": condition,
        "hparam_source": hparam_src,
        "group": "train_perturb",
        "valid_mrr20":  _f(best_valid, "mrr@20"),
        "test_mrr20":   _f(test_r, "mrr@20"),
        "valid_hr1":    _f(best_valid, "hit@1"),
        "test_hr1":     _f(test_r, "hit@1"),
        "valid_hr10":   _f(best_valid, "hit@10"),
        "test_hr10":    _f(test_r, "hit@10"),
        "valid_ndcg1":  _f(best_valid, "ndcg@1"),
        "test_ndcg1":   _f(test_r, "ndcg@1"),
        "valid_ndcg10": _f(best_valid, "ndcg@10"),
        "test_ndcg10":  _f(test_r, "ndcg@10"),
        "lr":           hparam["lr"],
        "hidden_size":  hparam["hidden_size"],
        "d_expert_hidden": hparam["d_expert_hidden"],
        "elapsed_sec":  round(elapsed, 1),
        "status":       status,
        "log_path":     str(log_path),
        "result_path":  result_path_str,
        "timestamp_utc": _now(),
    }

    print(f"  [{condition}/{hparam_src}] {dataset}  "
          f"test_mrr20={row['test_mrr20']:.4f}  "
          f"elapsed={elapsed/60:.1f}min  status={status}", flush=True)
    return row


# ── main: global queue ────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train-time cue perturbation ablation (A12)")
    p.add_argument("--gpus", nargs="+", default=["0"],
                   help="GPU ID 목록. 전체 job을 global queue로 배분")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--conditions", nargs="+",
                   default=list(TRAIN_CONDITIONS.keys()),
                   help="실행할 조건 (default: all)")
    p.add_argument("--top-k", type=int, default=1,
                   help="p0_a12_summary에서 사용할 hparam 수 (default: 1)")
    args = p.parse_args()

    gpus = [str(g) for g in args.gpus]

    print(f"[train_perturb] GPUs={gpus}  datasets={args.datasets}  top_k={args.top_k}",
          flush=True)

    # 전체 job 구성: (dataset, condition, hparam)
    all_jobs: list[dict] = []
    for ds in args.datasets:
        try:
            hparams = get_hparams_for_dataset(ds, args.top_k)
        except Exception as e:
            print(f"  [SKIP] {ds}: hparam 로드 실패: {e}", flush=True)
            continue
        for cond_name in args.conditions:
            cond_cfg = TRAIN_CONDITIONS.get(cond_name)
            if cond_cfg is None:
                print(f"  [SKIP] 알 수 없는 조건: {cond_name}", flush=True)
                continue
            for hp in hparams:
                all_jobs.append({
                    "dataset": ds,
                    "condition": cond_name,
                    "condition_cfg": cond_cfg,
                    "hparam": hp,
                })

    print(f"\n[train_perturb] 총 {len(all_jobs)}개 job → GPU {len(gpus)}개 global queue",
          flush=True)
    for j in all_jobs:
        print(f"  {j['dataset']:<12} {j['condition']:<15} {j['hparam']['hparam_source']}",
              flush=True)
    print(flush=True)

    job_q: queue.Queue = queue.Queue()
    for job in all_jobs:
        job_q.put(job)

    def worker(gpu: str) -> None:
        while True:
            try:
                job = job_q.get_nowait()
            except queue.Empty:
                return
            try:
                row = run_condition_single(
                    dataset=job["dataset"],
                    condition=job["condition"],
                    condition_cfg=job["condition_cfg"],
                    hparam=job["hparam"],
                    gpu_id=gpu,
                )
                _append_csv(TRAIN_OUTPUT_CSV, TRAIN_FIELDS, row)
            except Exception as e:
                print(f"  [ERROR] {job['dataset']}/{job['condition']}: {e}", flush=True)
            finally:
                job_q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n[train_perturb] Done.  결과 → {TRAIN_OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
