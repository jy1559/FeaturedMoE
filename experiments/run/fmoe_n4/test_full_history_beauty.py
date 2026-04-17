#!/usr/bin/env python3
"""
Test: session_only vs full_history_session_targets on beauty (feature_added_v4)

측정 항목:
  1. 시퀀스 길이 통계 (max_seq_length 자르기 전 / 후) — train / valid / test split별
  2. dataset 구성(build) 시간 비교
  3. 짧은 학습 + 추론 속도 비교 (3 epoch, FeaturedMoE_N3)

사용법:
  cd /workspace/FeaturedMoE/experiments
  /venv/FMoE/bin/python run/fmoe_n4/test_full_history_beauty.py
  /venv/FMoE/bin/python run/fmoe_n4/test_full_history_beauty.py --gpu 0 --epochs 3
  /venv/FMoE/bin/python run/fmoe_n4/test_full_history_beauty.py --stats-only
"""

from __future__ import annotations

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = THIS_DIR.parents[1]  # experiments/
REPO_ROOT = EXPERIMENTS_DIR.parent     # FeaturedMoE/

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

DATASET = "beauty"
FEATURE_DIR = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4" / DATASET
MAX_SEQ_LEN = 20          # beauty 기본 len; 통계에서 crop cutoff로 사용
POLICY = "strict_train_prefix"


# ════════════════════════════════════════════════════════════════════════════
# 1. Sequence-length statistics (standalone, no RecBole needed)
# ════════════════════════════════════════════════════════════════════════════

def _load_inter(path: Path) -> dict[str, np.ndarray]:
    """TSV .inter 파일 → field→numpy 딕셔너리"""
    import pandas as pd
    df = pd.read_csv(path, sep="\t")
    # RecBole 스타일 헤더 (field:type) 처리
    rename = {}
    for col in df.columns:
        base = col.split(":")[0]
        rename[col] = base
    df.rename(columns=rename, inplace=True)
    return {col: df[col].to_numpy() for col in df.columns}


def _stats_label(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size:,}  mean={arr.mean():.2f}  "
        f"median={np.median(arr):.1f}  "
        f"p90={np.percentile(arr, 90):.1f}  "
        f"p95={np.percentile(arr, 95):.1f}  "
        f"max={arr.max()}"
    )


def compute_seq_stats(max_seq_len: int = MAX_SEQ_LEN) -> None:
    """두 모드의 input 길이 통계를 계산해 출력한다."""
    print("\n" + "═" * 68)
    print(f"  Sequence-length statistics  |  dataset={DATASET} | max_seq_len={max_seq_len}")
    print("═" * 68)

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    splits = {}
    for split in ("train", "valid", "test"):
        p = FEATURE_DIR / f"{DATASET}.{split}.inter"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        splits[split] = _load_inter(p)
        print(f"  loaded {split}: {len(splits[split]['session_id']):,} rows")

    # ── MODE A: session_only ──────────────────────────────────────────────
    print("\n── MODE A: session_only ──")
    for sname, data in splits.items():
        sess_ids = data["session_id"]
        uniq, counts = np.unique(sess_ids, return_counts=True)
        # 각 세션에서 생성되는 samples의 prefix 길이
        if sname == "train":
            # data augmentation: target = pos 1..n-1 inside session
            lengths_before = []
            for cnt in counts:
                if cnt >= 2:
                    lengths_before.extend(range(1, cnt))  # 1, 2, ..., cnt-1
        else:
            # 마지막 아이템만 target
            lengths_before = [cnt - 1 for cnt in counts if cnt >= 2]
        lengths_before = np.array(lengths_before, dtype=np.int64)
        lengths_after = np.minimum(lengths_before, max_seq_len)
        print(f"  {sname:5s}  pre-crop : {_stats_label(lengths_before)}")
        print(f"         post-crop: {_stats_label(lengths_after)}")

    # ── MODE B: full_history_session_targets ─────────────────────────────
    print("\n── MODE B: full_history_session_targets (strict_train_prefix) ──")
    # Merge all splits to build full user history
    all_user   = np.concatenate([splits[s]["user_id"]   for s in ("train", "valid", "test")])
    all_sess   = np.concatenate([splits[s]["session_id"] for s in ("train", "valid", "test")])
    all_time   = np.concatenate([splits[s]["timestamp"]  for s in ("train", "valid", "test")])
    all_split  = np.concatenate([
        np.full(len(splits["train"]["session_id"]), 0, dtype=np.int8),
        np.full(len(splits["valid"]["session_id"]), 1, dtype=np.int8),
        np.full(len(splits["test"]["session_id"]),  2, dtype=np.int8),
    ])

    # Sort by (user, time, original_order)
    orig_idx = np.arange(len(all_user))
    order = np.lexsort((orig_idx, all_time, all_user))
    all_user  = all_user[order]
    all_sess  = all_sess[order]
    all_time  = all_time[order]
    all_split = all_split[order]

    # user boundaries
    user_changes = np.nonzero(all_user[1:] != all_user[:-1])[0] + 1
    user_starts = np.concatenate(([0], user_changes))
    user_ends   = np.concatenate((user_changes, [len(all_user)]))

    lengths_by_split: dict[str, list] = {"train": [], "valid": [], "test": []}
    split_name_map = {0: "train", 1: "valid", 2: "test"}

    for u_start, u_end in zip(user_starts, user_ends):
        u_sess  = all_sess[u_start:u_end]
        u_split = all_split[u_start:u_end]
        u_pos   = np.arange(u_start, u_end)
        u_train_pos = u_pos[u_split == 0]

        # session boundaries within this user
        sess_changes = np.nonzero(u_sess[1:] != u_sess[:-1])[0] + 1 if u_sess.size > 1 else np.array([], dtype=np.int64)
        s_starts = np.concatenate(([0], sess_changes))
        s_ends   = np.concatenate((sess_changes, [u_sess.size]))

        for ss, se in zip(s_starts, s_ends):
            sess_start = u_start + int(ss)
            sess_end   = u_start + int(se)
            sess_len   = sess_end - sess_start
            if sess_len < 2:
                continue
            sess_split_val = int(u_split[int(ss)])
            sname = split_name_map[sess_split_val]

            # train sessions before this session (causal)
            train_cut = int(np.searchsorted(u_train_pos, sess_start, side="left"))
            cross_base_size = train_cut  # number of cross-session history rows

            if sess_split_val == 0:
                # train: augmented targets
                for rel_t in range(1, sess_len):
                    # full history = cross_base + current_session_prefix up to rel_t
                    full_len = cross_base_size + rel_t
                    lengths_by_split["train"].append(full_len)
            else:
                # valid/test: last item target
                full_len = cross_base_size + (sess_len - 1)
                lengths_by_split[sname].append(full_len)

    for sname in ("train", "valid", "test"):
        arr = np.array(lengths_by_split[sname], dtype=np.int64)
        arr_after = np.minimum(arr, max_seq_len)
        print(f"  {sname:5s}  pre-crop : {_stats_label(arr)}")
        print(f"         post-crop: {_stats_label(arr_after)}")

    # ── Delta summary ─────────────────────────────────────────────────────
    print("\n── Pre-crop delta (full_history − session_only, valid/test) ──")
    for sname in ("valid", "test"):
        # session_only: per session, prefix len before last item
        data = splits[sname]
        sess_ids = data["session_id"]
        _, counts = np.unique(sess_ids, return_counts=True)
        so = np.array([cnt - 1 for cnt in counts if cnt >= 2], dtype=np.float64)
        fh = np.array(lengths_by_split[sname], dtype=np.float64)
        if so.size and fh.size:
            print(f"  {sname}: session_only mean={so.mean():.2f}  →  full_history mean={fh.mean():.2f}  "
                  f"(+{fh.mean() - so.mean():.2f})")


# ════════════════════════════════════════════════════════════════════════════
# 2. RecBole timing test (dataset build + N-epoch train)
# ════════════════════════════════════════════════════════════════════════════

def _build_config(history_mode: str, max_seq_len: int, gpu_id: int, epochs: int,
                  train_batch: int, eval_batch: int) -> dict:
    """recbole_train.py / Config에 전달할 override dict 반환"""
    return {
        # data
        "dataset": DATASET,
        "data_path": str(REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"),
        # eval
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {"group_by": "user", "order": "TO", "mode": "full"},
        # history mode
        "history_input_mode": history_mode,
        "history_group_field": "user_id",
        "target_group_field": "session_id",
        "history_eval_policy": POLICY,
        # seq length
        "MAX_ITEM_LIST_LENGTH": max_seq_len,
        # training (light)
        "epochs": epochs,
        "train_batch_size": train_batch,
        "eval_batch_size": eval_batch,
        "stopping_step": epochs + 10,   # no early stop during timing
        "eval_every": epochs,           # only eval at the end
        # misc
        "gpu_id": gpu_id,
        "enable_tf32": True,
        "log_wandb": False,
        "save_dataset": False,
        "save_dataloaders": False,
        "fmoe_debug_logging": False,
        "fmoe_diag_logging": False,
        "fmoe_special_logging": False,
        "fmoe_feature_ablation_logging": False,
    }


def run_timing_test(gpu_id: int, epochs: int, max_seq_len: int,
                    train_batch: int = 1024, eval_batch: int = 2048) -> None:
    """두 모드 각각 RecBole 학습 실행 → 시간 측정"""

    # ── heavy imports (only when needed) ──────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "4"

    import torch
    torch.set_num_threads(4)

    import recbole_patch  # noqa: F401 – apply all patches

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, get_trainer
    from recbole.utils import utils as recbole_utils
    from hydra_utils import load_hydra_config

    PYTHON = str(Path("/venv/FMoE/bin/python"))
    if not Path(PYTHON).exists():
        PYTHON = sys.executable

    results: list[dict] = []

    for mode in ("session_only", "full_history_session_targets"):
        print(f"\n{'─'*60}")
        print(f"  MODE: {mode}")
        print(f"{'─'*60}")

        # build Hydra config: model=featured_moe_n3_tune, eval_mode=session_fixed
        overrides = [
            "model=featured_moe_n3_tune",
            f"dataset={DATASET}",
            "eval_mode=session_fixed",
            "feature_mode=full_v4",
            f"history_input_mode={mode}",
            f"history_group_field=user_id",
            f"target_group_field=session_id",
            f"history_eval_policy={POLICY}",
            f"MAX_ITEM_LIST_LENGTH={max_seq_len}",
            f"epochs={epochs}",
            f"train_batch_size={train_batch}",
            f"eval_batch_size={eval_batch}",
            f"stopping_step={epochs + 99}",
            f"eval_every={epochs}",
            "gpu_id=0",   # CUDA_VISIBLE_DEVICES already set; use cuda:0
            "log_wandb=false",
            "save_dataset=false",
            "enable_tf32=true",
            "fmoe_debug_logging=false",
            "fmoe_diag_logging=false",
            "fmoe_special_logging=false",
            "fmoe_feature_ablation_logging=false",
            "++eval_sampling.mode=full",
            "++eval_sampling.auto_full_threshold=999999999",
            "++exclude_unseen_target_from_main_eval=true",
        ]

        cfg_dict = load_hydra_config(
            config_dir=EXPERIMENTS_DIR / "configs",
            config_name="config",
            overrides=overrides,
        )
        cfg_dict["data_path"] = str(REPO_ROOT / "Datasets" / "processed" / "feature_added_v4")
        # Disable split cache so we always measure real build time
        cfg_dict["enable_session_split_cache"] = False

        config = Config(
            model="FeaturedMoE_N3",
            dataset=DATASET,
            config_dict=cfg_dict,
        )
        init_seed(cfg_dict.get("seed", 42), cfg_dict.get("reproducibility", True))

        # ── dataset build ─────────────────────────────────────────────────
        t0_build = time.perf_counter()
        dataset = create_dataset(config)
        t_build = time.perf_counter() - t0_build
        print(f"  dataset build time: {t_build:.2f}s")

        # ── data preparation ──────────────────────────────────────────────
        t0_prep = time.perf_counter()
        train_data, valid_data, test_data = data_preparation(config, dataset)
        t_prep = time.perf_counter() - t0_prep
        print(f"  data_preparation time: {t_prep:.2f}s")

        # ── model init ────────────────────────────────────────────────────
        # CUDA_VISIBLE_DEVICES is already set → physical GPU maps to cuda:0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_cls = recbole_utils.get_model(config["model"])
        model = model_cls(config, train_data.dataset).to(device)

        # ── trainer ───────────────────────────────────────────────────────
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

        # ── training ──────────────────────────────────────────────────────
        t0_train = time.perf_counter()
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, verbose=False, saved=False, show_progress=False
        )
        t_train = time.perf_counter() - t0_train
        train_ms_per_epoch = t_train / epochs * 1000
        print(f"  training time ({epochs} ep): {t_train:.2f}s  ({train_ms_per_epoch:.0f} ms/ep)")

        # ── inference (test) ──────────────────────────────────────────────
        t0_test = time.perf_counter()
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        t_test = time.perf_counter() - t0_test
        print(f"  test eval time: {t_test:.2f}s")

        results.append({
            "mode": mode,
            "build_s": round(t_build, 2),
            "prep_s":  round(t_prep, 2),
            "train_s": round(t_train, 2),
            "ms_per_epoch": round(train_ms_per_epoch, 1),
            "test_s": round(t_test, 2),
            "best_valid": best_valid_result,
        })

        del model, trainer, dataset, train_data, valid_data, test_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    print(f"  Speed comparison  |  dataset={DATASET}  max_seq_len={max_seq_len}")
    print(f"  {'mode':<38}  {'build':>7}  {'ms/ep':>7}  {'test':>7}")
    print("─" * 68)
    for r in results:
        print(f"  {r['mode']:<38}  {r['build_s']:>6.1f}s  {r['ms_per_epoch']:>6.0f}ms  {r['test_s']:>6.2f}s")
    print("═" * 68)

    if len(results) == 2:
        so = results[0]
        fh = results[1]
        print(f"\n  build   ratio (full/session): {fh['build_s'] / max(so['build_s'], 1e-3):.2f}x")
        print(f"  ms/ep   ratio (full/session): {fh['ms_per_epoch'] / max(so['ms_per_epoch'], 1e-3):.2f}x")
        print(f"  test    ratio (full/session): {fh['test_s'] / max(so['test_s'], 1e-3):.2f}x")


# ════════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test full_history_session_targets on beauty v4")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help="MAX_ITEM_LIST_LENGTH")
    p.add_argument("--gpu", type=int, default=0, help="GPU id")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs for speed test")
    p.add_argument("--train-batch", type=int, default=1024)
    p.add_argument("--eval-batch",  type=int, default=2048)
    p.add_argument("--stats-only", action="store_true", help="Only compute length stats, skip RecBole timing")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    compute_seq_stats(max_seq_len=args.max_seq_len)

    if not args.stats_only:
        run_timing_test(
            gpu_id=args.gpu,
            epochs=args.epochs,
            max_seq_len=args.max_seq_len,
            train_batch=args.train_batch,
            eval_batch=args.eval_batch,
        )
