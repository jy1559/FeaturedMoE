#!/usr/bin/env python3
"""Eval-only cue perturbation ablation – CIKM 2026 (A12 architecture).

## 실행 흐름

1. P0 phase  (--p0-candidates N)
   - artifact_anchor_catalog.csv에서 dataset별 featured_moe_n3 top-N hparam 추출
   - 각 hparam 조합을 A12 architecture로 hyperopt_tune.py --max-evals 1 학습
   - GPU global queue로 병렬 실행
   - 결과를 results/p0_a12_summary.csv에 누적

2. Perturb phase  (--top-k K)
   - p0_a12_summary.csv에서 dataset별 test_mrr20 상위 K개 checkpoint 선택
   - 각 checkpoint × 각 eval condition을 GPU global queue로 병렬 실행
   - 결과를 results/eval_summary.csv에 누적

Usage:
    python eval_perturb.py --gpus 0 1 2 3
    python eval_perturb.py --gpus 0 1 2 3 --datasets foursquare
    python eval_perturb.py --gpus 0 1 2 3 --p0-candidates 4 --top-k 2
    python eval_perturb.py --gpus 0 1 2 3 --skip-p0   # P0 건너뛰고 기존 summary 사용
    python eval_perturb.py --gpus 0 1 2 3 --skip-perturb  # P0만 실행
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
    NARROW_SEARCH, PYTHON_BIN, RESULT_ROOT,
    TUNE_EPOCHS, TUNE_PATIENCE,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_DATASET_ROOT  = _REPO_ROOT / "Datasets" / "processed"
_FINAL_DATA    = _DATASET_ROOT / "final_dataset"
_CATALOG_PATH  = _CIKM_DIR / "docs" / "artifact_anchor_catalog.csv"

P0_SUMMARY_CSV  = _EXP_PERTURB / "results" / "p0_a12_summary.csv"
EVAL_OUTPUT_CSV = _EXP_PERTURB / "results" / "eval_summary.csv"
LOG_DIR         = _EXP_PERTURB / "logs" / "eval"

CIKM_ARTIFACTS_RESULTS = _EXP_PERTURB / "artifacts" / "results"
CIKM_ARTIFACTS_LOGS    = _EXP_PERTURB / "artifacts" / "logs"

# ── 데이터셋 설정 ─────────────────────────────────────────────────────────────
DATASETS = ["foursquare", "KuaiRec"]

DATASET_TUNE_CFG = {
    "KuaiRec":    "tune_kuai_cikm",
    "foursquare": "tune_foursq_cikm",
}

DATASET_HAS_ITEM_COL = {
    "KuaiRec":    True,
    "foursquare": False,
}

DATASET_EVAL_CFG: dict[str, dict[str, Any]] = {
    "KuaiRec": {
        "eval_sampling": {"mode": "full", "auto_full_threshold": 999_999_999},
        "eval_batch_size": 2048,
    },
    "foursquare": {
        "eval_sampling": {"mode": "full", "auto_full_threshold": 999_999_999},
        "eval_batch_size": 1024,
    },
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


# A12 Hydra overrides (공통 — architecture 구조 고정)
_A12_ARCH_OVERRIDES: list[str] = [
    "++fmoe_architecture_id=A12",
    "++fmoe_architecture_key=A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
    f'++layer_layout={_hydra_literal(["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"])}',
    f'++stage_compute_mode={_hydra_literal(_all_stage_map("moe"))}',
    f'++stage_router_mode={_hydra_literal(_all_stage_map("learned"))}',
    f'++stage_router_wrapper={_hydra_literal(_all_stage_map("w5_exd"))}',
    f'++stage_router_source={_hydra_literal(_all_stage_map("feature"))}',
    f'++stage_router_primitives={_hydra_literal(_a12_primitives("feature"))}',
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

# A12 cfg dict values (run_checkpoint_evaluation용)
_A12_BASE_CFG: dict[str, Any] = {
    "fmoe_architecture_id": "A12",
    "fmoe_architecture_key": "A12_ATTN_MICRO_BEFORE_NO_BIAS_ALL_W5",
    "layer_layout": ["attn", "macro_ffn", "mid_ffn", "attn", "micro_ffn"],
    "stage_compute_mode": _all_stage_map("moe"),
    "stage_router_mode": _all_stage_map("learned"),
    "stage_router_wrapper": _all_stage_map("w5_exd"),
    "stage_router_source": _all_stage_map("feature"),
    "stage_router_primitives": _a12_primitives("feature"),
    "stage_router_granularity": {"macro": "session", "mid": "session", "micro": "token"},
    "stage_feature_injection": _all_stage_map("none"),
    "stage_feature_dropout_scope": _all_stage_map("token"),
    "topk_scope_mode": "global_flat",
    "moe_top_k": 0,
    "balance_loss_lambda": 0.0,
    "gate_entropy_lambda": 0.0,
    "route_smoothness_lambda": 0.0,
    "route_sharpness_lambda": 0.0,
    "route_monopoly_lambda": 0.0,
    "route_prior_lambda": 0.0,
    "group_prior_align_lambda": 0.0,
    "factored_group_balance_lambda": 0.0,
    "rule_agreement_lambda": 0.0,
    "group_coverage_lambda": 0.0,
    "feature_group_bias_lambda": 0.0,
    "rule_bias_scale": 0.0,
    "bias_mode": "none",
    "route_consistency_pairs": 1,
    "route_consistency_lambda": 0.0,
    "macro_history_window": 5,
}

# ── Eval conditions ────────────────────────────────────────────────────────────
EVAL_CONDITIONS: dict[str, dict[str, Any]] = {
    "eval_zero": {
        "feature_perturb_mode": "zero",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "",
    },
    "eval_shuffle": {
        "feature_perturb_mode": "shuffle",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "",
    },
    "eval_global_permute": {
        "feature_perturb_mode": "global_permute",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "",
    },
    "eval_role_swap": {
        "feature_perturb_mode": "role_swap",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "",
    },
    "eval_stage_mismatch": {
        "feature_perturb_mode": "stage_mismatch",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "",
    },
    "eval_family_memory": {
        "feature_perturb_mode": "zero",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "memory",
    },
    "eval_family_focus": {
        "feature_perturb_mode": "zero",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "focus",
    },
    "eval_family_tempo": {
        "feature_perturb_mode": "zero",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "tempo",
    },
    "eval_family_exposure": {
        "feature_perturb_mode": "zero",
        "feature_perturb_apply": "eval",
        "feature_perturb_family": "exposure",
    },
}

EVAL_FIELDS = [
    "dataset", "model", "p0_rank", "condition", "group",
    "hit1", "hit5", "hit10", "hit20",
    "ndcg1", "ndcg5", "ndcg10", "ndcg20",
    "mrr1", "mrr5", "mrr10", "mrr20",
    "delta_mrr20", "checkpoint_path", "timestamp_utc",
]

P0_FIELDS = [
    "dataset", "model", "candidate_rank", "p0_rank",
    "test_mrr20", "test_hr10", "test_ndcg10",
    "valid_mrr20", "valid_hr10",
    "lr", "wd", "hidden_size", "d_expert_hidden", "d_router_hidden",
    "checkpoint_path", "result_path", "log_path",
    "elapsed_sec", "status", "timestamp_utc",
]

# ── CSV helpers ───────────────────────────────────────────────────────────────
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


# ── Catalog hparam extraction ─────────────────────────────────────────────────

def _fv(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    try:
        return float(v) if v else default
    except (ValueError, TypeError):
        return default


def load_catalog_candidates(dataset: str, n: int) -> list[dict[str, Any]]:
    """artifact_anchor_catalog.csv에서 dataset의 featured_moe_n3 top-N hparam을 반환."""
    if not _CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found: {_CATALOG_PATH}")

    rows = list(csv.DictReader(_CATALOG_PATH.open(encoding="utf-8")))
    ds_rows = [
        r for r in rows
        if r.get("model") == "featured_moe_n3"
        and r.get("family") == "route"
        and r.get("dataset_target") == dataset
    ]
    if not ds_rows:
        raise ValueError(f"No featured_moe_n3 route entries for {dataset} in catalog")

    # sort by test_mrr20 desc, take top-N
    ds_rows.sort(key=lambda r: _fv(r, "test_mrr20"), reverse=True)
    top = ds_rows[:n]

    candidates = []
    for rank_idx, r in enumerate(top, 1):
        # lr/wd: from catalog columns (already extracted by catalog builder)
        lr = _fv(r, "learning_rate")
        wd = _fv(r, "weight_decay")
        if lr <= 0:
            # fallback: NARROW_SEARCH midpoint
            ns = NARROW_SEARCH.get(dataset, {}).get("featured_moe_n3", {})
            lrs = ns.get("learning_rate", [1e-3])
            lr = lrs[len(lrs) // 2]

        hidden_size    = int(_fv(r, "hidden_size", 128))
        d_ff           = int(_fv(r, "d_ff", hidden_size * 2))
        d_expert_hidden = int(_fv(r, "d_expert_hidden", hidden_size))
        d_router_hidden = int(_fv(r, "d_router_hidden", 64))
        d_feat_emb     = int(_fv(r, "d_feat_emb", 16))
        expert_scale   = int(_fv(r, "expert_scale", 4))
        max_seq_len    = int(_fv(r, "MAX_ITEM_LIST_LENGTH", 10))
        hidden_dp      = _fv(r, "hidden_dropout_prob", 0.10)

        candidates.append({
            "catalog_rank": int(r.get("candidate_rank", rank_idx)),
            "p0_rank": rank_idx,           # rank within this top-N selection
            "catalog_mrr20": _fv(r, "test_mrr20"),
            "lr": lr,
            "wd": wd,
            "hidden_size": hidden_size,
            "d_ff": d_ff,
            "d_expert_hidden": d_expert_hidden,
            "d_router_hidden": d_router_hidden,
            "d_feat_emb": d_feat_emb,
            "expert_scale": expert_scale,
            "MAX_ITEM_LIST_LENGTH": max_seq_len,
            "hidden_dropout_prob": hidden_dp,
        })

    print(f"  [catalog] {dataset}: {len(candidates)}개 hparam 후보 선택 "
          f"(top-{n} by test_mrr20)", flush=True)
    for c in candidates:
        print(f"    p0_rank={c['p0_rank']} catalog_rank={c['catalog_rank']} "
              f"catalog_mrr20={c['catalog_mrr20']:.4f}  "
              f"lr={c['lr']:.2e} wd={c['wd']:.1e} "
              f"hidden={c['hidden_size']} d_exp={c['d_expert_hidden']} "
              f"d_rout={c['d_router_hidden']}", flush=True)
    return candidates


# ── P0 single run ─────────────────────────────────────────────────────────────

def run_p0_single(
    *,
    dataset: str,
    candidate: dict[str, Any],
    gpu_id: str,
) -> dict[str, Any]:
    """카탈로그 hparam을 A12로 학습 (max-evals=1). 결과 row 반환."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p0_rank = candidate["p0_rank"]
    log_path = LOG_DIR / f"p0_{dataset}_rank{p0_rank:02d}.log"
    cfg_name = DATASET_TUNE_CFG.get(dataset, "tune_kuai_cikm")

    cmd: list[str] = [
        PYTHON_BIN, "hyperopt_tune.py",
        "--config-name", cfg_name,
        "--search-algo", "tpe",
        "--max-evals", "1",
        "--tune-epochs", str(TUNE_EPOCHS),
        "--tune-patience", str(TUNE_PATIENCE),
        "--seed", "42",
        "--run-group", "cikm",
        "--run-axis", "cikm_cue_perturb_p0_a12",
        "--run-phase", "P0-A12",
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
        f"++search.learning_rate={candidate['lr']:g}",
        f"++search.weight_decay={candidate['wd']:g}",
        # arch size
        f"++MAX_ITEM_LIST_LENGTH={candidate['MAX_ITEM_LIST_LENGTH']}",
        f"++hidden_size={candidate['hidden_size']}",
        f"++d_ff={candidate['d_ff']}",
        f"++d_expert_hidden={candidate['d_expert_hidden']}",
        f"++d_router_hidden={candidate['d_router_hidden']}",
        f"++d_feat_emb={candidate['d_feat_emb']}",
        f"++expert_scale={candidate['expert_scale']}",
        f"++hidden_dropout_prob={candidate['hidden_dropout_prob']:g}",
        # fix search space singletons too (so hyperopt doesn't re-sample arch)
        f"++search.hidden_size={candidate['hidden_size']}",
        f"++search.d_ff={candidate['d_ff']}",
        f"++search.d_expert_hidden={candidate['d_expert_hidden']}",
        f"++search.d_router_hidden={candidate['d_router_hidden']}",
        f"++search.d_feat_emb={candidate['d_feat_emb']}",
        f"++search.expert_scale={candidate['expert_scale']}",
        f"++search.hidden_dropout_prob={candidate['hidden_dropout_prob']:g}",
    ]
    cmd.extend(_A12_ARCH_OVERRIDES)

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    CIKM_ARTIFACTS_RESULTS.mkdir(parents=True, exist_ok=True)
    CIKM_ARTIFACTS_LOGS.mkdir(parents=True, exist_ok=True)
    env["HYPEROPT_RESULTS_DIR"] = str(CIKM_ARTIFACTS_RESULTS)
    env["RUN_LOGS_DIR"]         = str(CIKM_ARTIFACTS_LOGS)

    print(f"  [P0] {dataset} p0_rank={p0_rank} gpu={gpu_id}  "
          f"lr={candidate['lr']:.2e} hidden={candidate['hidden_size']}", flush=True)
    start = time.time()

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# P0 A12  dataset={dataset}  p0_rank={p0_rank}  gpu={gpu_id}\n")
        fh.write(f"# candidate={candidate}\n")
        fh.write(f"# cmd={' '.join(cmd)}\n\n")
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

    def _f(d: dict, k: str) -> float:
        return float(d.get(k, 0) or 0)

    ckpt_path = ""
    test_mrr20 = 0.0
    status = "error"

    if result_path_str and Path(result_path_str).exists():
        try:
            payload = json.loads(Path(result_path_str).read_text(encoding="utf-8"))
            test_r  = payload.get("test_result", {}) or {}
            bval_r  = payload.get("best_valid_result", {}) or {}
            test_mrr20 = _f(test_r, "mrr@20")
            for key in ("best_checkpoint_file", "artifact_best_checkpoint"):
                v = str(payload.get(key, "") or "").strip()
                if v and Path(v).exists():
                    ckpt_path = v
                    break
            status = "ok" if test_mrr20 > 0 else ("completed" if proc.returncode == 0 else "error")
        except Exception:
            pass
    elif proc.returncode != 0:
        status = f"error_rc{proc.returncode}"

    row = {
        "dataset": dataset,
        "model": "FeaturedMoE_N3_A12",
        "candidate_rank": candidate["catalog_rank"],
        "p0_rank": p0_rank,
        "test_mrr20": test_mrr20,
        "test_hr10": 0.0,
        "test_ndcg10": 0.0,
        "valid_mrr20": 0.0,
        "valid_hr10": 0.0,
        "lr": candidate["lr"],
        "wd": candidate["wd"],
        "hidden_size": candidate["hidden_size"],
        "d_expert_hidden": candidate["d_expert_hidden"],
        "d_router_hidden": candidate["d_router_hidden"],
        "checkpoint_path": ckpt_path,
        "result_path": result_path_str,
        "log_path": str(log_path),
        "elapsed_sec": round(elapsed, 1),
        "status": status,
        "timestamp_utc": _now(),
    }

    # fill in more metrics if available
    if result_path_str and Path(result_path_str).exists():
        try:
            payload = json.loads(Path(result_path_str).read_text(encoding="utf-8"))
            test_r  = payload.get("test_result", {}) or {}
            bval_r  = payload.get("best_valid_result", {}) or {}
            row["test_hr10"]   = _f(test_r, "hit@10")
            row["test_ndcg10"] = _f(test_r, "ndcg@10")
            row["valid_mrr20"] = _f(bval_r, "mrr@20")
            row["valid_hr10"]  = _f(bval_r, "hit@10")
        except Exception:
            pass

    print(f"  [P0] {dataset} p0_rank={p0_rank}  "
          f"test_mrr20={test_mrr20:.4f}  elapsed={elapsed/60:.1f}min  "
          f"status={status}  ckpt={'✓' if ckpt_path else '✗'}", flush=True)
    return row


# ── P0 phase: 전체 데이터셋 × candidates → GPU global queue ──────────────────

def run_p0_phase(
    datasets: list[str],
    gpus: list[str],
    n_candidates: int,
) -> None:
    """모든 (dataset, candidate) 조합을 GPU global queue로 병렬 학습."""
    print(f"\n{'='*60}", flush=True)
    print(f"[P0 phase] datasets={datasets}  gpus={gpus}  candidates={n_candidates}", flush=True)

    # 전체 job 구성
    all_jobs: list[dict] = []
    for ds in datasets:
        try:
            candidates = load_catalog_candidates(ds, n_candidates)
        except Exception as e:
            print(f"  [SKIP] {ds}: {e}", flush=True)
            continue
        for cand in candidates:
            all_jobs.append({"dataset": ds, "candidate": cand})

    if not all_jobs:
        print("[P0 phase] 실행할 job 없음.", flush=True)
        return

    print(f"\n[P0 phase] 총 {len(all_jobs)}개 job → GPU {len(gpus)}개 global queue", flush=True)
    for j in all_jobs:
        print(f"  {j['dataset']:<12} p0_rank={j['candidate']['p0_rank']}  "
              f"catalog_rank={j['candidate']['catalog_rank']}  "
              f"catalog_mrr20={j['candidate']['catalog_mrr20']:.4f}", flush=True)

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
                row = run_p0_single(
                    dataset=job["dataset"],
                    candidate=job["candidate"],
                    gpu_id=gpu,
                )
                _append_csv(P0_SUMMARY_CSV, P0_FIELDS, row)
            except Exception as e:
                print(f"  [P0 ERROR] {job['dataset']} rank={job['candidate']['p0_rank']}: {e}",
                      flush=True)
            finally:
                job_q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n[P0 phase] Done.  결과 → {P0_SUMMARY_CSV}", flush=True)


# ── P0 best-K 선택 ────────────────────────────────────────────────────────────

def select_top_k_checkpoints(
    dataset: str,
    k: int,
) -> list[dict[str, Any]]:
    """P0_SUMMARY_CSV에서 dataset 기준 test_mrr20 상위 K개 (ckpt 있는 것만) 반환."""
    if not P0_SUMMARY_CSV.exists():
        return []

    rows = list(csv.DictReader(P0_SUMMARY_CSV.open(encoding="utf-8")))
    ds_rows = [
        r for r in rows
        if r.get("dataset") == dataset
        and r.get("checkpoint_path")
        and Path(r["checkpoint_path"]).exists()
    ]
    if not ds_rows:
        return []

    ds_rows.sort(key=lambda r: float(r.get("test_mrr20", 0) or 0), reverse=True)
    top = ds_rows[:k]

    result = []
    for rank_idx, r in enumerate(top, 1):
        result.append({
            "perturb_rank": rank_idx,
            "p0_rank": r.get("p0_rank", "?"),
            "test_mrr20": float(r.get("test_mrr20", 0) or 0),
            "checkpoint_path": r["checkpoint_path"],
            "hidden_size": int(float(r.get("hidden_size", 128) or 128)),
            "d_expert_hidden": int(float(r.get("d_expert_hidden", 128) or 128)),
            "d_router_hidden": int(float(r.get("d_router_hidden", 64) or 64)),
            "result_path": r.get("result_path", ""),
        })
    return result


# ── build_base_cfg ────────────────────────────────────────────────────────────

def infer_arch_from_checkpoint(ckpt_path: str) -> dict[str, Any]:
    """checkpoint tensor shapes에서 모델 arch 파라미터를 완전히 역산한다.

    추측이나 테이블 참조 없이, 실제 weight shape에서 직접 읽는다.
    run_checkpoint_evaluation이 strict=True로 로드하므로 이 값들이 정확해야 한다.
    """
    import torch as _t
    state: dict[str, Any] = _t.load(ckpt_path, map_location="cpu")

    arch: dict[str, Any] = {}

    # hidden_size, MAX_ITEM_LIST_LENGTH
    for k, v in state.items():
        if k.endswith("item_embedding.weight"):
            arch["hidden_size"] = int(v.shape[1])
            break
    for k, v in state.items():
        if k.endswith("position_embedding.weight"):
            arch["MAX_ITEM_LIST_LENGTH"] = int(v.shape[0])
            break

    # d_ff ← shared_fc1.weight: [d_ff, hidden_size]
    for k, v in state.items():
        if k.endswith("shared_fc1.weight"):
            arch["d_ff"] = int(v.shape[0])
            break

    # d_expert_hidden ← experts.0.backbone.0.weight: [d_expert_hidden, hidden_size]
    for k, v in state.items():
        if ".experts.0.backbone.0.weight" in k:
            arch["d_expert_hidden"] = int(v.shape[0])
            break

    # d_feat_emb ← feature_encoder.net.bias: [d_feat_emb]
    for k, v in state.items():
        if k.endswith("feature_encoder.net.bias"):
            arch["d_feat_emb"] = int(v.shape[0])
            break

    # d_router_hidden ← router_d.group_heads.0.0.bias: [d_router_hidden]
    # (NOT router_a which doesn't exist in this arch)
    for k, v in state.items():
        if "router_d.group_heads.0.0.bias" in k:
            arch["d_router_hidden"] = int(v.shape[0])
            break

    # expert_scale: n_experts = n_base_groups * expert_scale
    # n_experts = count of "experts.N.out.bias" keys in macro stage
    # n_base_groups = count of "group_feature_projections.N.bias" keys in macro stage
    n_experts = sum(
        1 for k in state
        if "stage_blocks.macro.experts." in k and k.endswith(".out.bias")
    )
    n_groups = sum(
        1 for k in state
        if "stage_blocks.macro.group_feature_projections." in k and k.endswith(".bias")
    )
    if n_experts > 0 and n_groups > 0:
        arch["expert_scale"] = max(1, n_experts // n_groups)

    return arch


def build_base_cfg(
    dataset: str,
    gpu_id: int,
    ckpt_info: dict[str, Any],
) -> dict[str, Any]:
    """checkpoint를 직접 열어 arch를 역산하고 run_checkpoint_evaluation cfg를 구성.

    strict=True 로드이므로 cfg의 모든 arch 파라미터가 checkpoint shape와 정확히 일치해야 한다.
    추측/테이블 참조 없이 checkpoint에서 직접 읽는다.
    """
    ckpt_path = ckpt_info.get("checkpoint_path", "")
    if not ckpt_path or not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path!r}")

    # 1. checkpoint에서 arch 완전 역산
    arch = infer_arch_from_checkpoint(ckpt_path)
    print(f"  [build_base_cfg] {dataset} arch from ckpt: {arch}", flush=True)

    missing = [k for k in ("hidden_size", "MAX_ITEM_LIST_LENGTH", "d_ff",
                           "d_expert_hidden", "d_feat_emb", "d_router_hidden",
                           "expert_scale") if k not in arch]
    if missing:
        raise RuntimeError(
            f"Cannot infer arch params from checkpoint {ckpt_path}: missing {missing}"
        )

    # 2. 나머지 cfg 구성
    ds_cfg   = DATASET_EVAL_CFG[dataset]
    has_item = DATASET_HAS_ITEM_COL.get(dataset, False)

    # inter load_col: 필수 필드 + 모든 feature columns (perturbation이 실제 작동하려면 필요)
    # model yaml의 load_col.inter와 동일한 구성
    from models.FeaturedMoE_N3.feature_config import ALL_FEATURE_COLUMNS
    inter_cols = ["session_id", "item_id", "timestamp"] + list(ALL_FEATURE_COLUMNS)
    load_col: dict[str, list[str]] = {"inter": inter_cols}
    if has_item:
        load_col["item"] = ["item_id", "category"]

    cfg: dict[str, Any] = {
        "model": "FeaturedMoE_N3",
        "dataset": dataset,
        "data_path": str(_FINAL_DATA),
        "dataset_root": str(_DATASET_ROOT),
        "eval_mode": "session_fixed",
        "feature_mode": "final",
        # session_fixed: use pre-split files so item vocab matches P0 training
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {"group_by": "user", "order": "TO", "mode": "full"},
        "gpu_id": gpu_id,
        "use_gpu": True,
        "eval_batch_size": ds_cfg["eval_batch_size"],
        "eval_sampling": dict(ds_cfg["eval_sampling"]),
        "show_progress": False,
        "state": "INFO",
        "seed": 42,
        "reproducibility": True,
        "USER_ID_FIELD": "session_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        "field_separator": "\t",
        "train_neg_sample_args": None,
        "loss_type": "CE",
        "enable_data_cache": True,
        "data_cache_dir": "saved/recbole_cache",
        "in_memory_data_cache": True,
        "use_fast_data_augmentation": True,
        "sequence_convert_chunk_size": 16384,
        "fmoe_feature_fp16": True,
        "log_wandb": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "exclude_unseen_target_from_main_eval": True,
        "feature_perturb_mode": "none",
        "feature_perturb_apply": "none",
        "feature_perturb_family": "",
        "topk": [1, 5, 10, 20],
        "metrics": ["Hit", "NDCG", "MRR"],
        "valid_metric": "MRR@20",
    }

    cfg["load_col"] = load_col

    # 3. A12 routing 구조 (학습 때 쓴 구조 그대로)
    cfg.update(_A12_BASE_CFG)

    # 4. checkpoint에서 역산한 arch size (이 값들이 모델 생성에 쓰임)
    cfg.update(arch)

    # FeaturedMoE_N3 reads embedding_size (not hidden_size) to build d_model.
    # _sync_model_dimensions does this in training path; replicate here.
    cfg["embedding_size"] = arch["hidden_size"]

    return cfg


# ── metric helpers ────────────────────────────────────────────────────────────

def _metrics_from_result(test_r: dict) -> dict[str, float]:
    def _f(key: str) -> float:
        return float(test_r.get(key, 0.0) or 0.0)
    return {
        "hit1":   _f("hit@1"),  "hit5":   _f("hit@5"),
        "hit10":  _f("hit@10"), "hit20":  _f("hit@20"),
        "ndcg1":  _f("ndcg@1"), "ndcg5":  _f("ndcg@5"),
        "ndcg10": _f("ndcg@10"),"ndcg20": _f("ndcg@20"),
        "mrr1":   _f("mrr@1"),  "mrr5":   _f("mrr@5"),
        "mrr10":  _f("mrr@10"), "mrr20":  _f("mrr@20"),
    }


# ── single eval condition run ─────────────────────────────────────────────────

def run_eval_condition(
    *,
    condition: str,
    overrides: dict[str, Any],
    base_cfg: dict[str, Any],
    checkpoint_path: str,
    intact_mrr20: float,
    dataset: str,
    p0_rank: int,
    perturb_rank: int,
) -> dict[str, Any] | None:
    import recbole_patch  # noqa: F401
    from recbole_train import run_checkpoint_evaluation  # noqa: E402

    cfg = {**base_cfg, **overrides}
    try:
        result = run_checkpoint_evaluation(
            cfg,
            run_name=f"cue_perturb_{condition}_{dataset}_pr{perturb_rank}",
            checkpoint_path=checkpoint_path,
        )
    except Exception as e:
        print(f"    [ERROR] {condition}: {e}", flush=True)
        return None

    test_r = result.get("test_result", {}) or {}
    m = _metrics_from_result(test_r)
    return {
        "dataset": dataset,
        "model": "FeaturedMoE_N3_A12",
        "p0_rank": f"p0r{p0_rank}_perturb{perturb_rank}",
        "condition": condition,
        "group": "eval_perturb",
        **m,
        "delta_mrr20": round(m["mrr20"] - intact_mrr20, 6),
        "checkpoint_path": checkpoint_path,
        "timestamp_utc": _now(),
    }


# ── Perturb phase: checkpoint × condition → GPU global queue ─────────────────

def run_perturb_phase(
    datasets: list[str],
    gpus: list[str],
    top_k: int,
    conditions: list[str],
) -> None:
    """P0 best-K checkpoint × eval conditions を GPU global queue로 병렬 실행."""
    print(f"\n{'='*60}", flush=True)
    print(f"[Perturb phase] datasets={datasets}  gpus={gpus}  "
          f"top_k={top_k}  conditions={len(conditions)}", flush=True)

    # 데이터셋별 top-K checkpoint 수집
    ds_checkpoints: dict[str, list[dict]] = {}
    for ds in datasets:
        ckpts = select_top_k_checkpoints(ds, top_k)
        if not ckpts:
            print(f"  [SKIP] {ds}: P0 checkpoint 없음 (p0_a12_summary.csv 확인)", flush=True)
            continue
        ds_checkpoints[ds] = ckpts
        print(f"  {ds}: {len(ckpts)}개 checkpoint 선택", flush=True)
        for c in ckpts:
            print(f"    perturb_rank={c['perturb_rank']}  "
                  f"test_mrr20={c['test_mrr20']:.4f}  "
                  f"...{c['checkpoint_path'][-50:]}", flush=True)

    if not ds_checkpoints:
        print("[Perturb phase] 실행할 checkpoint 없음.", flush=True)
        return

    # intact row 먼저 기록 (checkpoint evaluation 없이 P0 결과에서 직접)
    for ds, ckpts in ds_checkpoints.items():
        for ckpt_info in ckpts:
            _append_csv(EVAL_OUTPUT_CSV, EVAL_FIELDS, {
                "dataset": ds,
                "model": "FeaturedMoE_N3_A12",
                "p0_rank": f"p0r{ckpt_info['p0_rank']}_perturb{ckpt_info['perturb_rank']}",
                "condition": "intact",
                "group": "eval_perturb",
                "hit1": 0.0, "hit5": 0.0, "hit10": 0.0, "hit20": 0.0,
                "ndcg1": 0.0, "ndcg5": 0.0, "ndcg10": 0.0, "ndcg20": 0.0,
                "mrr1": 0.0, "mrr5": 0.0, "mrr10": 0.0,
                "mrr20": ckpt_info["test_mrr20"],
                "delta_mrr20": 0.0,
                "checkpoint_path": ckpt_info["checkpoint_path"],
                "timestamp_utc": _now(),
            })

    # 전체 job 목록: (ds, ckpt_info, condition)
    all_jobs: list[dict] = []
    for ds in datasets:
        if ds not in ds_checkpoints:
            continue
        for ckpt_info in ds_checkpoints[ds]:
            for cond_name in conditions:
                cond_overrides = EVAL_CONDITIONS.get(cond_name)
                if cond_overrides is None:
                    continue
                all_jobs.append({
                    "dataset": ds,
                    "ckpt_info": ckpt_info,
                    "condition": cond_name,
                    "overrides": cond_overrides,
                })

    print(f"\n[Perturb phase] 총 {len(all_jobs)}개 eval job → GPU {len(gpus)}개 global queue",
          flush=True)

    job_q: queue.Queue = queue.Queue()
    for job in all_jobs:
        job_q.put(job)

    def worker(gpu: str) -> None:
        import recbole_patch  # noqa: F401
        from recbole_train import run_checkpoint_evaluation  # noqa: E402

        while True:
            try:
                job = job_q.get_nowait()
            except queue.Empty:
                return
            ds        = job["dataset"]
            ckpt_info = job["ckpt_info"]
            cond      = job["condition"]
            try:
                base_cfg = build_base_cfg(ds, int(gpu), ckpt_info)
                cfg = {**base_cfg, **job["overrides"]}
                label = (f"cue_perturb_{cond}_{ds}_"
                         f"p{ckpt_info['p0_rank']}_pr{ckpt_info['perturb_rank']}")
                print(f"  [{ds}/{cond}/pr{ckpt_info['perturb_rank']}] 시작 gpu={gpu}",
                      flush=True)
                result = run_checkpoint_evaluation(
                    cfg,
                    run_name=label,
                    checkpoint_path=ckpt_info["checkpoint_path"],
                )
                test_r = result.get("test_result", {}) or {}
                m = _metrics_from_result(test_r)
                intact = ckpt_info["test_mrr20"]
                row = {
                    "dataset": ds,
                    "model": "FeaturedMoE_N3_A12",
                    "p0_rank": f"p0r{ckpt_info['p0_rank']}_perturb{ckpt_info['perturb_rank']}",
                    "condition": cond,
                    "group": "eval_perturb",
                    **m,
                    "delta_mrr20": round(m["mrr20"] - intact, 6),
                    "checkpoint_path": ckpt_info["checkpoint_path"],
                    "timestamp_utc": _now(),
                }
                _append_csv(EVAL_OUTPUT_CSV, EVAL_FIELDS, row)
                delta_str = f"{row['delta_mrr20']:+.4f}"
                print(f"  [{ds}/{cond}/pr{ckpt_info['perturb_rank']}] "
                      f"MRR@20={m['mrr20']:.4f}  Δ={delta_str}  gpu={gpu}", flush=True)
            except Exception as e:
                print(f"  [{ds}/{cond}/pr{ckpt_info['perturb_rank']}] ERROR: {e}",
                      flush=True)
            finally:
                job_q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"\n[Perturb phase] Done.  결과 → {EVAL_OUTPUT_CSV}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Eval cue perturbation ablation (A12)")
    p.add_argument("--gpus", nargs="+", default=["0"],
                   help="GPU ID 목록. P0/perturb 모두 global queue로 배분")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--conditions", nargs="+",
                   default=["intact"] + list(EVAL_CONDITIONS.keys()),
                   help="실행할 eval 조건 (default: all)")
    p.add_argument("--p0-candidates", type=int, default=4,
                   help="카탈로그에서 가져올 hparam 후보 수 (default: 4)")
    p.add_argument("--top-k", type=int, default=2,
                   help="P0 결과에서 perturb에 사용할 best checkpoint 수 (default: 2)")
    p.add_argument("--skip-p0", action="store_true",
                   help="P0 학습 건너뜀 (기존 p0_a12_summary.csv 사용)")
    p.add_argument("--skip-perturb", action="store_true",
                   help="Perturb 실행 건너뜀 (P0만 실행)")
    args = p.parse_args()

    gpus = [str(g) for g in args.gpus]
    conditions = [c for c in args.conditions if c != "intact"]

    print(f"[eval_perturb] GPUs={gpus}  datasets={args.datasets}", flush=True)
    print(f"[eval_perturb] p0_candidates={args.p0_candidates}  top_k={args.top_k}", flush=True)
    print(f"[eval_perturb] conditions ({len(conditions)}): {conditions}", flush=True)

    if not args.skip_p0:
        run_p0_phase(args.datasets, gpus, args.p0_candidates)
    else:
        print("\n[eval_perturb] --skip-p0: P0 건너뜀", flush=True)

    if not args.skip_perturb:
        run_perturb_phase(args.datasets, gpus, args.top_k, conditions)
    else:
        print("\n[eval_perturb] --skip-perturb: Perturb 건너뜀", flush=True)

    print(f"\n[eval_perturb] All done.", flush=True)


if __name__ == "__main__":
    main()
