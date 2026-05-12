#!/usr/bin/env python3
"""Cue perturbation 결과 통합 → 논문용 테이블 생성.

eval_perturb (그룹 A) + train_perturb (그룹 B) 두 CSV를 병합.
delta_mrr20 계산, 조건 순서 정렬, 터미널 요약 출력.

Usage:
    python collect_results.py
    python collect_results.py --out results/cue_perturb_final.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_EXP_PERTURB = Path(__file__).resolve().parent
_CIKM_DIR = _EXP_PERTURB.parent
sys.path.insert(0, str(_CIKM_DIR))

from common import RESULT_ROOT  # noqa: E402

EVAL_CSV  = _EXP_PERTURB / "results" / "eval_summary.csv"
TRAIN_CSV = _EXP_PERTURB / "results" / "train_summary.csv"
DEFAULT_OUT = _EXP_PERTURB / "results" / "cue_perturb_summary.csv"

# 논문 테이블 조건 순서 (그룹 A → 그룹 B)
CONDITION_ORDER = [
    "intact",
    # Group A: eval-only
    "eval_zero",
    "eval_shuffle",
    "eval_global_permute",
    "eval_role_swap",
    "eval_stage_mismatch",
    "eval_family_memory",
    "eval_family_focus",
    "eval_family_tempo",
    "eval_family_exposure",
    # Group B: train-time
    "hidden_only",
    "feature_only",
    "train_zero",
    "both_zero",
    "train_shuffle",
]

OUT_FIELDS = [
    "condition", "group",
    "KuaiRec_mrr20", "KuaiRec_hr1", "KuaiRec_hr10",
    "KuaiRec_ndcg1", "KuaiRec_ndcg10", "KuaiRec_delta_mrr20",
    "foursquare_mrr20", "foursquare_hr1", "foursquare_hr10",
    "foursquare_ndcg1", "foursquare_ndcg10", "foursquare_delta_mrr20",
    "status_kuai", "status_foursq",
]


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pick(rows: list[dict], dataset: str, condition: str, key: str) -> str:
    for r in rows:
        if r.get("dataset") == dataset and r.get("condition") == condition:
            v = r.get(key, "")
            if v:
                return v
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    eval_rows  = _load(EVAL_CSV)
    train_rows = _load(TRAIN_CSV)
    all_rows   = eval_rows + train_rows

    if not all_rows:
        print("[collect] 결과 없음. eval_perturb.py / train_perturb.py 먼저 실행하세요.", flush=True)
        print(f"  Expected: {EVAL_CSV}", flush=True)
        print(f"           {TRAIN_CSV}", flush=True)
        sys.exit(0)

    print(f"[collect] eval rows={len(eval_rows)}  train rows={len(train_rows)}", flush=True)

    # intact MRR@20 기준점 수집
    intact_mrr: dict[str, float] = {}
    for r in all_rows:
        if r.get("condition") == "intact":
            ds  = r.get("dataset", "")
            mrr = float(r.get("mrr20") or r.get("test_mrr20") or 0)
            if mrr > 0:
                intact_mrr[ds] = mrr

    # 발견된 조건 집합
    seen = {r.get("condition", "") for r in all_rows}
    ordered = [c for c in CONDITION_ORDER if c in seen]
    ordered += sorted(seen - set(CONDITION_ORDER))  # 목록 외 조건 추가

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _get(ds: str, cond: str, key: str) -> str:
        # eval 우선, 없으면 train
        for pool in (eval_rows, train_rows):
            v = _pick(pool, ds, cond, key)
            if v:
                return v
        return ""

    def _mrr(ds: str, cond: str) -> str:
        return _get(ds, cond, "mrr20") or _get(ds, cond, "test_mrr20")

    def _hr1(ds: str, cond: str) -> str:
        return _get(ds, cond, "hit1") or _get(ds, cond, "test_hr1")

    def _hr10(ds: str, cond: str) -> str:
        return _get(ds, cond, "hit10") or _get(ds, cond, "test_hr10")

    def _ndcg1(ds: str, cond: str) -> str:
        return _get(ds, cond, "ndcg1") or _get(ds, cond, "test_ndcg1")

    def _ndcg10(ds: str, cond: str) -> str:
        return _get(ds, cond, "ndcg10") or _get(ds, cond, "test_ndcg10")

    def _delta(ds: str, cond: str) -> str:
        v = _get(ds, cond, "delta_mrr20")
        if v:
            return v
        # train 그룹: intact 대비 직접 계산
        mrr_s = _mrr(ds, cond)
        if mrr_s and ds in intact_mrr:
            try:
                return f"{float(mrr_s) - intact_mrr[ds]:+.6f}"
            except ValueError:
                pass
        return ""

    def _status(ds: str, cond: str) -> str:
        return _get(ds, cond, "status")

    def _group(cond: str) -> str:
        for r in all_rows:
            if r.get("condition") == cond:
                return r.get("group", "")
        return ""

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for cond in ordered:
            writer.writerow({
                "condition":                cond,
                "group":                    _group(cond),
                "KuaiRec_mrr20":            _mrr("KuaiRec", cond),
                "KuaiRec_hr1":              _hr1("KuaiRec", cond),
                "KuaiRec_hr10":             _hr10("KuaiRec", cond),
                "KuaiRec_ndcg1":            _ndcg1("KuaiRec", cond),
                "KuaiRec_ndcg10":           _ndcg10("KuaiRec", cond),
                "KuaiRec_delta_mrr20":      _delta("KuaiRec", cond),
                "foursquare_mrr20":         _mrr("foursquare", cond),
                "foursquare_hr1":           _hr1("foursquare", cond),
                "foursquare_hr10":          _hr10("foursquare", cond),
                "foursquare_ndcg1":         _ndcg1("foursquare", cond),
                "foursquare_ndcg10":        _ndcg10("foursquare", cond),
                "foursquare_delta_mrr20":   _delta("foursquare", cond),
                "status_kuai":              _status("KuaiRec", cond),
                "status_foursq":            _status("foursquare", cond),
            })

    print(f"[collect] → {out_path}", flush=True)

    # 터미널 요약
    print(f"\n{'─'*80}", flush=True)
    print(f"{'조건':<28} {'KuaiRec MRR@20':>14} {'Δ':>9}  {'Foursq MRR@20':>14} {'Δ':>9}", flush=True)
    print(f"{'─'*80}", flush=True)
    prev_group = ""
    with out_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            grp  = row["group"]
            if grp != prev_group and prev_group:
                print(f"{'─'*80}", flush=True)
            prev_group = grp
            cond = row["condition"]
            k_m  = row["KuaiRec_mrr20"]
            k_d  = row["KuaiRec_delta_mrr20"]
            f_m  = row["foursquare_mrr20"]
            f_d  = row["foursquare_delta_mrr20"]
            print(f"  {cond:<26} {k_m:>14}  {k_d:>9}  {f_m:>14}  {f_d:>9}", flush=True)
    print(f"{'─'*80}", flush=True)


if __name__ == "__main__":
    main()
