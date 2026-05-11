#!/usr/bin/env python3
"""Build phase9~13 + legacy wrap-up package under docs/final.

Design goals:
- Keep existing docs/results/visualization/data untouched.
- Create new wrap-up artifacts only under docs/final.
- Integrate phase9~13 (wide + verification) and pre9 timeline digest.
- Emit reproducible CSV tables, one markdown report, one notebook.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import nbformat as nbf


# ----------------------------
# constants
# ----------------------------

DEFAULT_REPO_ROOT = Path("/workspace/jy1559/FMoE")
DEFAULT_DATASET = "KuaiRecLargeStrictPosV2_0.2"

PHASE_ORDER = ["P9", "P9_2", "P10", "P11", "P12", "P13"]
PHASE_RANK = {p: i for i, p in enumerate(PHASE_ORDER)}

PHASE_TITLE = {
    "P9": "Aux Loss Concept Study",
    "P9_2": "Aux Loss Verification",
    "P10": "Feature Portability / Compactness",
    "P11": "Stage Semantics / Necessity / Granularity",
    "P12": "Layout Composition / Attention Placement",
    "P13": "Feature Sanity / Alignment Checks",
}

INPUT_REL = {
    "phase9_main": "experiments/run/fmoe_n3/docs/data/phase8_9/phase9_main.csv",
    "phase9_2_main": "experiments/run/fmoe_n3/docs/data/phase8_9/phase9_2_main.csv",
    "phase9_2_pending": "experiments/run/fmoe_n3/docs/data/phase8_9/phase9_2_pending.csv",
    "diag_special_join": "experiments/run/fmoe_n3/docs/data/phase8_9/diag_special_join.csv",
    "family_expert_pca_points": "experiments/run/fmoe_n3/docs/data/phase8_9/family_expert_pca_points.csv",
    "wide_all_dedup": "experiments/run/fmoe_n3/docs/data/phase10_13/wide_all_dedup.csv",
    "verification_all_dedup": "experiments/run/fmoe_n3/docs/data/phase10_13/verification_all_dedup.csv",
    "verification_main_h3_n20": "experiments/run/fmoe_n3/docs/data/phase10_13/verification_main_h3_n20.csv",
    "router_stage_scalar": "experiments/run/fmoe_n3/docs/data/phase10_13/router_stage_scalar.csv",
    "router_family_expert_long": "experiments/run/fmoe_n3/docs/data/phase10_13/router_family_expert_long.csv",
    "router_position_expert_long": "experiments/run/fmoe_n3/docs/data/phase10_13/router_position_expert_long.csv",
    "intent_vs_observed_summary": "experiments/run/fmoe_n3/docs/data/phase10_13/intent_vs_observed_summary.csv",
    "phase9_result_md": "experiments/run/fmoe_n3/docs/results/phase9_auxloss_and_verification.md",
    "phase10_13_wide_md": "experiments/run/fmoe_n3/docs/results/phase10_13_wide.md",
    "phase10_13_verif_md": "experiments/run/fmoe_n3/docs/results/phase10_13_verification.md",
    "master_handoff_md": "experiments/run/fmoe_n3/docs/FMOE_N3_KuaiRec_Strict_Master_Handoff.md",
    "phase9_plan_md": "experiments/run/fmoe_n3/docs/plans/phase9_auxloss.md",
    "phase10_plan_md": "experiments/run/fmoe_n3/docs/plans/phase_10_13_specified/phase10_feature_portability.md",
    "phase11_plan_md": "experiments/run/fmoe_n3/docs/plans/phase_10_13_specified/phase11_stage_semantics.md",
    "phase12_plan_md": "experiments/run/fmoe_n3/docs/plans/phase_10_13_specified/phase12_layout_composition.md",
    "phase13_plan_md": "experiments/run/fmoe_n3/docs/plans/phase_10_13_specified/phase13_feature_sanity.md",
    "phase10_13_verif_plan_md": "experiments/run/fmoe_n3/docs/plans/phase_10_13_verification.md",
}

OUTPUT_DATA_FILES = {
    "wide_all_9_13": "wide_all_9_13.csv",
    "verification_all_9_13": "verification_all_9_13.csv",
    "verification_main_fair_h3_n20": "verification_main_fair_h3_n20.csv",
    "verification_support_coverage": "verification_support_coverage.csv",
    "verification_seed_stats_9_13": "verification_seed_stats_9_13.csv",
    "diag_special_long_9_13": "diag_special_long_9_13.csv",
    "diag_corr_by_phase": "diag_corr_by_phase.csv",
    "diag_quantile_profile": "diag_quantile_profile.csv",
    "router_stage_scalar_9_13": "router_stage_scalar_9_13.csv",
    "router_family_long_10_13": "router_family_long_10_13.csv",
    "router_position_long_10_13": "router_position_long_10_13.csv",
    "family_pca_points_9_13": "family_pca_points_9_13.csv",
    "intent_claim_evidence_map_9_13": "intent_claim_evidence_map_9_13.csv",
    "legacy_timeline_pre9_summary": "legacy_timeline_pre9_summary.csv",
    "source_manifest": "source_manifest.csv",
}

EXPECTED_COUNTS = {
    "wide_all_9_13": 163,
    "verification_all_9_13": 239,
    "verification_main_fair_h3_n20": 108,
    "verification_support_coverage": 131,
}

DIAG_METRICS = [
    "diag_n_eff",
    "diag_cv_usage",
    "diag_top1_max_frac",
    "diag_entropy_mean",
    "diag_route_jitter_adjacent",
    "diag_route_consistency_knn_score",
    "diag_route_consistency_group_knn_score",
    "diag_route_consistency_intra_group_knn_mean_score",
    "diag_family_top_expert_mean_share",
]

TARGET_METRICS = [
    "best_valid_mrr20",
    "test_mrr20",
    "cold_item_mrr20",
    "long_session_mrr20",
    "sess_3_5_mrr20",
]

P9_CONCEPT_INTENT = {
    "C0": {
        "setting_group": "C0_Natural",
        "axis_label": "Natural",
        "plan_intent": "aux 최소/약한 안정화로 baseline 대비 변화 확인",
        "expected_pattern": "보수적 개선 또는 유지, 큰 붕괴 없이 자연 분포",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "약한 정규화만으로도 안정적인 성능대를 유지할 수 있다.",
    },
    "C1": {
        "setting_group": "C1_CanonicalBalance",
        "axis_label": "CanonicalBalance",
        "plan_intent": "expert/group usage 균형 유도로 안정성 향상 검증",
        "expected_pattern": "valid 안정성 개선 가능, 강도 과다 시 test 손실 위험",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "균형 유도는 성능/안정성 trade-off를 조절하는 핵심 축이다.",
    },
    "C2": {
        "setting_group": "C2_Specialization",
        "axis_label": "Specialization",
        "plan_intent": "route 집중(특화)과 일반화 손실 간 균형 확인",
        "expected_pattern": "특정 base에서 고점, 과도 특화는 일반화 하락 가능",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "적절한 특화는 유효하지만 과도 집중은 일반화 리스크를 만든다.",
    },
    "C3": {
        "setting_group": "C3_FeatureAlignment",
        "axis_label": "FeatureAlignment",
        "plan_intent": "feature prior와 routing 정렬 효과 검증",
        "expected_pattern": "feature-rich 조건에서 상대 이득 가능",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "feature-aligned routing은 성능 분포를 유의미하게 이동시킨다.",
    },
}


# ----------------------------
# helpers
# ----------------------------


def parse_bool(text: str) -> bool:
    v = str(text).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool: {text}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return [dict(r) for r in csv.DictReader(fp)]


def write_csv(path: Path, rows: List[Dict[str, Any]], field_order: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        rows = [{"note": "no rows"}]

    if field_order:
        ordered = list(field_order)
        seen = set(ordered)
        rest = sorted({k for row in rows for k in row.keys() if k not in seen})
        fieldnames = ordered + rest
    else:
        fieldnames = sorted({k for row in rows for k in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        s = str(value).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def to_int(value: Any, default: int = 0) -> int:
    v = to_float(value)
    if v is None:
        return default
    return int(v)


def normalize_hparam_id(value: Any) -> str:
    s = str(value or "").strip().upper()
    if not s:
        return ""
    if s.startswith("H"):
        return s
    return f"H{s}"


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [v for v in values if v is not None and math.isfinite(v)]
    if not arr:
        return None
    return float(sum(arr) / len(arr))


def safe_stdev(values: Iterable[Optional[float]]) -> Optional[float]:
    arr = [v for v in values if v is not None and math.isfinite(v)]
    if not arr:
        return None
    if len(arr) == 1:
        return 0.0
    return float(statistics.stdev(arr))


def fmt_num(value: Any, nd: int = 4, dash: str = "-") -> str:
    v = to_float(value)
    if v is None:
        return dash
    return f"{v:.{nd}f}"


def md_table(rows: List[Dict[str, Any]], columns: List[str], headers: Optional[List[str]] = None) -> List[str]:
    if headers is None:
        headers = columns
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = [str(row.get(c, "")) for c in columns]
        out.append("| " + " | ".join(vals) + " |")
    return out


def rank_average(values: List[float]) -> List[float]:
    indexed = sorted([(v, i) for i, v in enumerate(values)], key=lambda x: x[0])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def spearman_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx = rank_average(xs)
    ry = rank_average(ys)
    return pearson_corr(rx, ry)


def quantile_index(idx: int, n: int, q: int = 5) -> int:
    if n <= 0:
        return 1
    return min(q, int(idx * q / n) + 1)


def phase_sort_key(phase: str) -> Tuple[int, str]:
    return (PHASE_RANK.get(phase, 999), phase)


# ----------------------------
# normalization
# ----------------------------


def normalize_phase9_wide_row(raw: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    concept = str(raw.get("concept_id", "")).strip()
    combo = str(raw.get("combo_id", "")).strip()
    setting_key = f"P9-{concept}-{combo}" if concept and combo else str(raw.get("setting_id", ""))
    return {
        **raw,
        "dataset": raw.get("dataset", dataset),
        "source_phase": "P9",
        "phase_title": PHASE_TITLE["P9"],
        "source_axis": "phase9_auxloss_v1",
        "split": "wide",
        "setting_group": concept,
        "setting_key": setting_key,
        "setting_short": combo,
        "setting_uid": raw.get("setting_id", raw.get("run_phase", "")),
        "setting_idx": raw.get("candidate_id", ""),
        "hparam_id": normalize_hparam_id(raw.get("hvar_id")),
        "seed_id": raw.get("seed_id", ""),
        "stage": raw.get("stage", ""),
        "long_session_mrr20": raw.get("long_session_mrr20", ""),
        "sess_6_10_mrr20": raw.get("sess_6_10_mrr20", ""),
        "timestamp_utc": raw.get("timestamp", ""),
        "record_origin": "phase8_9/phase9_main.csv",
    }


def normalize_phase9_2_verification_row(raw: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    concept = str(raw.get("concept_id", "")).strip()
    combo = str(raw.get("combo_id", "")).strip()
    base = str(raw.get("base_id", "")).strip()
    setting_key = f"P9_2-{base}-{concept}-{combo}" if concept and combo else str(raw.get("setting_id", ""))
    return {
        **raw,
        "dataset": raw.get("dataset", dataset),
        "source_phase": "P9_2",
        "phase_title": PHASE_TITLE["P9_2"],
        "source_axis": "phase9_2_verification_v2",
        "split": "verification",
        "setting_group": concept,
        "setting_key": setting_key,
        "setting_short": combo,
        "setting_uid": raw.get("setting_id", raw.get("run_phase", "")),
        "setting_idx": raw.get("candidate_id", ""),
        "hparam_id": normalize_hparam_id(raw.get("hvar_id")),
        "seed_id": raw.get("seed_id", ""),
        "stage": raw.get("stage", ""),
        "long_session_mrr20": raw.get("long_session_mrr20", ""),
        "sess_6_10_mrr20": raw.get("sess_6_10_mrr20", ""),
        "timestamp_utc": raw.get("timestamp", ""),
        "record_origin": "phase8_9/phase9_2_main.csv",
    }


def normalize_phase10_13_row(raw: Dict[str, Any], origin: str) -> Dict[str, Any]:
    row = dict(raw)
    row["hparam_id"] = normalize_hparam_id(row.get("hparam_id"))
    row["record_origin"] = origin
    return row


# ----------------------------
# table builders
# ----------------------------


def build_wide_all_9_13(
    phase9_main: List[Dict[str, Any]],
    wide_10_13: List[Dict[str, Any]],
    dataset: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(normalize_phase9_wide_row(r, dataset) for r in phase9_main)
    rows.extend(normalize_phase10_13_row(r, "phase10_13/wide_all_dedup.csv") for r in wide_10_13)
    rows.sort(key=lambda r: (phase_sort_key(str(r.get("source_phase", ""))), str(r.get("run_phase", ""))))
    return rows


def build_verification_all_9_13(
    phase9_2_main: List[Dict[str, Any]],
    ver_10_13: List[Dict[str, Any]],
    dataset: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(normalize_phase9_2_verification_row(r, dataset) for r in phase9_2_main)
    rows.extend(normalize_phase10_13_row(r, "phase10_13/verification_all_dedup.csv") for r in ver_10_13)
    rows.sort(key=lambda r: (phase_sort_key(str(r.get("source_phase", ""))), str(r.get("run_phase", ""))))
    return rows


def build_verification_main_fair_h3_n20(
    verification_all: List[Dict[str, Any]],
    main_hparam: str,
    main_min_completed: int,
) -> List[Dict[str, Any]]:
    hmain = normalize_hparam_id(main_hparam)
    out: List[Dict[str, Any]] = []
    for r in verification_all:
        phase = str(r.get("source_phase", "")).strip()
        hparam = normalize_hparam_id(r.get("hparam_id"))
        n_completed = to_int(r.get("n_completed"), 0)

        is_main_p11_13 = phase in {"P11", "P12", "P13"} and hparam == hmain and n_completed >= main_min_completed
        is_main_p9_2 = phase == "P9_2" and hparam == hmain and n_completed >= main_min_completed
        if is_main_p11_13 or is_main_p9_2:
            out.append(dict(r))

    out.sort(key=lambda r: (phase_sort_key(str(r.get("source_phase", ""))), str(r.get("setting_key", "")), str(r.get("seed_id", ""))))
    return out


def build_verification_support_coverage(
    verification_all: List[Dict[str, Any]],
    main_hparam: str,
    main_min_completed: int,
) -> List[Dict[str, Any]]:
    hmain = normalize_hparam_id(main_hparam)
    out: List[Dict[str, Any]] = []

    for r in verification_all:
        phase = str(r.get("source_phase", "")).strip()
        hparam = normalize_hparam_id(r.get("hparam_id"))
        n_completed = to_int(r.get("n_completed"), 0)

        # coverage bucket 1: P10 support (H1/H3, n=10)
        if phase == "P10" and hparam in {"H1", "H3"} and n_completed == 10:
            out.append(dict(r))
            continue

        # coverage bucket 2: P9_2 non-main hparams (H1/H2/H4, n>=20)
        if phase == "P9_2" and hparam and hparam != hmain and n_completed >= main_min_completed:
            out.append(dict(r))
            continue

    out.sort(key=lambda r: (phase_sort_key(str(r.get("source_phase", ""))), str(r.get("setting_key", "")), str(r.get("hparam_id", "")), str(r.get("seed_id", ""))))
    return out


def summarize_verification_seed_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = [
        "best_valid_mrr20",
        "test_mrr20",
        "cold_item_mrr20",
        "long_session_mrr20",
        "sess_3_5_mrr20",
        "diag_top1_max_frac",
        "diag_cv_usage",
        "diag_n_eff",
        "diag_entropy_mean",
    ]

    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get("source_phase", "")),
            str(r.get("split", "")),
            str(r.get("setting_group", "")),
            str(r.get("setting_key", "")),
            normalize_hparam_id(r.get("hparam_id")),
        )
        groups[key].append(r)

    out: List[Dict[str, Any]] = []
    for key, members in groups.items():
        source_phase, split, setting_group, setting_key, hparam_id = key

        row: Dict[str, Any] = {
            "source_phase": source_phase,
            "phase_title": PHASE_TITLE.get(source_phase, source_phase),
            "split": split,
            "setting_group": setting_group,
            "setting_key": setting_key,
            "hparam_id": hparam_id,
            "seed_n": len({str(m.get("seed_id", "")) for m in members if str(m.get("seed_id", "")).strip()}),
            "row_n": len(members),
        }

        for m in metrics:
            vals = [to_float(x.get(m)) for x in members]
            row[f"{m}_mean"] = safe_mean(vals)
            row[f"{m}_std"] = safe_stdev(vals)

        # stability helper
        row["stability_score"] = (
            (to_float(row.get("best_valid_mrr20_mean")) or 0.0)
            - (to_float(row.get("best_valid_mrr20_std")) or 0.0)
        )

        out.append(row)

    out.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("setting_group", "")),
            -(to_float(r.get("best_valid_mrr20_mean")) or -1e12),
        )
    )
    return out


def build_diag_special_long_9_13(
    wide_rows: List[Dict[str, Any]],
    verification_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for r in (list(wide_rows) + list(verification_rows)):
        base = {
            "source_phase": r.get("source_phase", ""),
            "phase_title": PHASE_TITLE.get(str(r.get("source_phase", "")), ""),
            "split": r.get("split", ""),
            "run_phase": r.get("run_phase", ""),
            "setting_group": r.get("setting_group", ""),
            "setting_key": r.get("setting_key", ""),
            "hparam_id": normalize_hparam_id(r.get("hparam_id")),
            "seed_id": r.get("seed_id", ""),
        }

        for t in TARGET_METRICS:
            tv = to_float(r.get(t))
            if tv is None:
                continue
            for d in DIAG_METRICS:
                dv = to_float(r.get(d))
                if dv is None:
                    continue
                rows.append(
                    {
                        **base,
                        "target_metric": t,
                        "target_value": tv,
                        "diag_metric": d,
                        "diag_value": dv,
                    }
                )

    rows.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("split", "")),
            str(r.get("target_metric", "")),
            str(r.get("diag_metric", "")),
            str(r.get("run_phase", "")),
        )
    )
    return rows


def summarize_diag_corr_by_phase(diag_long: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = defaultdict(list)

    for r in diag_long:
        key = (
            str(r.get("source_phase", "")),
            str(r.get("split", "")),
            str(r.get("target_metric", "")),
            str(r.get("diag_metric", "")),
        )
        x = to_float(r.get("diag_value"))
        y = to_float(r.get("target_value"))
        if x is None or y is None:
            continue
        groups[key].append((x, y))

    out: List[Dict[str, Any]] = []
    for key, pairs in groups.items():
        source_phase, split, target_metric, diag_metric = key
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        out.append(
            {
                "source_phase": source_phase,
                "phase_title": PHASE_TITLE.get(source_phase, source_phase),
                "split": split,
                "target_metric": target_metric,
                "diag_metric": diag_metric,
                "n_pairs": len(pairs),
                "pearson": pearson_corr(xs, ys),
                "spearman": spearman_corr(xs, ys),
                "abs_spearman": abs(spearman_corr(xs, ys) or 0.0),
            }
        )

    out.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("split", "")),
            str(r.get("target_metric", "")),
            -(to_float(r.get("abs_spearman")) or 0.0),
        )
    )
    return out


def summarize_diag_quantile_profile(diag_long: List[Dict[str, Any]], q: int = 5) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = defaultdict(list)

    for r in diag_long:
        target_metric = str(r.get("target_metric", ""))
        if target_metric not in {"best_valid_mrr20", "test_mrr20"}:
            continue
        key = (
            str(r.get("source_phase", "")),
            str(r.get("split", "")),
            target_metric,
            str(r.get("diag_metric", "")),
        )
        x = to_float(r.get("diag_value"))
        y = to_float(r.get("target_value"))
        if x is None or y is None:
            continue
        groups[key].append((x, y))

    out: List[Dict[str, Any]] = []

    for key, pairs in groups.items():
        if len(pairs) < q + 1:
            continue

        source_phase, split, target_metric, diag_metric = key
        pairs_sorted = sorted(pairs, key=lambda p: p[0])

        buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        n = len(pairs_sorted)
        for idx, pair in enumerate(pairs_sorted):
            qb = quantile_index(idx, n, q=q)
            buckets[qb].append(pair)

        for qb in sorted(buckets.keys()):
            bucket = buckets[qb]
            out.append(
                {
                    "source_phase": source_phase,
                    "phase_title": PHASE_TITLE.get(source_phase, source_phase),
                    "split": split,
                    "target_metric": target_metric,
                    "diag_metric": diag_metric,
                    "diag_quantile": qb,
                    "n_pairs": len(bucket),
                    "diag_min": min(x for x, _ in bucket),
                    "diag_max": max(x for x, _ in bucket),
                    "diag_mean": safe_mean([x for x, _ in bucket]),
                    "target_mean": safe_mean([y for _, y in bucket]),
                    "target_std": safe_stdev([y for _, y in bucket]),
                }
            )

    out.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("split", "")),
            str(r.get("target_metric", "")),
            str(r.get("diag_metric", "")),
            to_int(r.get("diag_quantile"), 0),
        )
    )
    return out


def build_router_stage_scalar_9_13(
    router_stage_10_13: List[Dict[str, Any]],
    phase9_main: List[Dict[str, Any]],
    phase9_2_main: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [dict(r) for r in router_stage_10_13]

    def add_phase9_row(raw: Dict[str, Any], phase: str, split: str) -> None:
        concept = str(raw.get("concept_id", "")).strip()
        combo = str(raw.get("combo_id", "")).strip()
        base = str(raw.get("base_id", "")).strip()
        setting_key = f"{phase}-{base}-{concept}-{combo}" if base or concept or combo else str(raw.get("setting_id", ""))
        out.append(
            {
                "source_phase": phase,
                "phase_title": PHASE_TITLE.get(phase, phase),
                "source_axis": "phase9_auxloss_v1" if phase == "P9" else "phase9_2_verification_v2",
                "split": split,
                "stage_name": "all@1",
                "mode": "summary",
                "run_phase": raw.get("run_phase", ""),
                "seed_id": raw.get("seed_id", ""),
                "hparam_id": normalize_hparam_id(raw.get("hvar_id")),
                "setting_group": concept,
                "setting_key": setting_key,
                "setting_short": combo,
                "setting_uid": raw.get("setting_id", raw.get("run_phase", "")),
                "best_valid_mrr20": raw.get("best_valid_mrr20", ""),
                "test_mrr20": raw.get("test_mrr20", ""),
                "n_eff": raw.get("diag_n_eff", ""),
                "cv_usage": raw.get("diag_cv_usage", ""),
                "top1_max_frac": raw.get("diag_top1_max_frac", ""),
                "entropy_mean": raw.get("diag_entropy_mean", ""),
                "route_jitter_adjacent": raw.get("diag_route_jitter_adjacent", ""),
                "route_consistency_knn_score": raw.get("diag_route_consistency_knn_score", ""),
                "route_consistency_group_knn_score": raw.get("diag_route_consistency_group_knn_score", ""),
                "route_consistency_group_knn_js": raw.get("diag_route_consistency_group_knn_js", ""),
                "route_consistency_feature_group_knn_mean_score": "",
                "route_consistency_feature_group_knn_mean_js": "",
                "usage_share_max": raw.get("diag_top1_max_frac", ""),
                "usage_share_entropy": raw.get("diag_entropy_mean", ""),
                "usage_share_mean": "",
                "usage_share_min": "",
                "usage_share_std": "",
                "dataset": raw.get("dataset", DEFAULT_DATASET),
                "record_origin": "phase8_9",
            }
        )

    for r in phase9_main:
        add_phase9_row(r, phase="P9", split="wide")
    for r in phase9_2_main:
        add_phase9_row(r, phase="P9_2", split="verification")

    out.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("split", "")),
            str(r.get("run_phase", "")),
            str(r.get("stage_name", "")),
        )
    )
    return out


def build_family_pca_points_9_13(
    legacy_pca_rows: List[Dict[str, Any]],
    router_family_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for r in legacy_pca_rows:
        tag = str(r.get("phase_tag", "")).strip().lower()
        if tag not in {"phase9", "phase9_2"}:
            continue
        row = dict(r)
        row["source"] = "legacy_phase8_9_pca"
        out.append(row)

    for r in router_family_rows:
        phase = str(r.get("source_phase", "")).strip()
        if phase not in {"P10", "P11", "P12", "P13"}:
            continue

        share = to_float(r.get("family_expert_share_norm"))
        eg = to_float(r.get("expert_global_share"))
        if share is None or eg is None:
            continue

        out.append(
            {
                "axis_group": r.get("source_axis", ""),
                "axis_label_x": "router_family_share",
                "axis_label_y": "expert_global_share",
                "best_valid_mrr20": r.get("best_valid_mrr20", ""),
                "expert": r.get("expert", ""),
                "family": r.get("family", ""),
                "loading_pc1": "",
                "loading_pc2": "",
                "pc1": share,
                "pc1_explained_var_ratio": "",
                "pc2": eg,
                "pc2_explained_var_ratio": "",
                "phase_tag": phase.lower(),
                "run_phase": r.get("run_phase", ""),
                "setting_id": r.get("setting_uid", r.get("setting_key", "")),
                "test_mrr20": r.get("test_mrr20", ""),
                "weight_norm": share,
                "source": "router_family_proxy_points",
            }
        )

    out.sort(key=lambda r: (str(r.get("phase_tag", "")), str(r.get("run_phase", "")), str(r.get("family", "")), str(r.get("expert", ""))))
    return out


def summarize_phase9_concept_observed(phase9_main: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in phase9_main:
        concept = str(r.get("concept_id", "")).strip()
        if concept:
            groups[concept].append(r)

    rows: List[Dict[str, Any]] = []
    for concept, members in sorted(groups.items()):
        meta = P9_CONCEPT_INTENT.get(concept, {})
        best_row = sorted(
            members,
            key=lambda r: (
                to_float(r.get("best_valid_mrr20")) or -1e12,
                to_float(r.get("test_mrr20")) or -1e12,
            ),
            reverse=True,
        )[0]
        valid_mean = safe_mean(to_float(r.get("best_valid_mrr20")) for r in members)
        test_mean = safe_mean(to_float(r.get("test_mrr20")) for r in members)
        best_valid = to_float(best_row.get("best_valid_mrr20"))
        best_test = to_float(best_row.get("test_mrr20"))
        delta_valid = (best_valid - valid_mean) if (best_valid is not None and valid_mean is not None) else None
        delta_test = (best_test - test_mean) if (best_test is not None and test_mean is not None) else None

        rows.append(
            {
                "source_phase": "P9",
                "phase_title": PHASE_TITLE["P9"],
                "setting_group": meta.get("setting_group", concept),
                "axis_label": meta.get("axis_label", concept),
                "plan_intent": meta.get("plan_intent", ""),
                "expected_pattern": meta.get("expected_pattern", ""),
                "expectation_type": meta.get("expectation_type", ""),
                "paper_claim_template": meta.get("paper_claim_template", ""),
                "n_rows_dedup": len(members),
                "n_rows_main": len(members),
                "best_run_phase": best_row.get("run_phase", ""),
                "best_setting_key": best_row.get("setting_id", ""),
                "best_valid_mrr20": best_valid,
                "best_test_mrr20": best_test,
                "mean_valid_main": valid_mean,
                "mean_test_main": test_mean,
                "mean_cold_main": safe_mean(to_float(r.get("cold_item_mrr20")) for r in members),
                "mean_long_session_main": safe_mean(to_float(r.get("long_session_mrr20")) for r in members),
                "mean_diag_top1_main": safe_mean(to_float(r.get("diag_top1_max_frac")) for r in members),
                "mean_diag_cv_main": safe_mean(to_float(r.get("diag_cv_usage")) for r in members),
                "mean_diag_n_eff_main": safe_mean(to_float(r.get("diag_n_eff")) for r in members),
                "anchor_setting_key": f"{concept}-mean",
                "anchor_best_valid_mrr20": valid_mean,
                "anchor_test_mrr20": test_mean,
                "delta_best_valid_vs_anchor": delta_valid,
                "delta_best_test_vs_anchor": delta_test,
                "observed_tag": "observed_phase9_concept",
                "match_flag": 1,
                "diag_missing_count": sum(1 for r in members if to_int(r.get("diag_available"), 0) == 0),
                "special_missing_count": sum(1 for r in members if to_int(r.get("special_available"), 0) == 0),
                "evidence_source": "phase8_9/phase9_main.csv",
            }
        )

    return rows


def build_intent_claim_evidence_map_9_13(
    intent_10_13: List[Dict[str, Any]],
    phase9_main: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []

    for r in intent_10_13:
        row = dict(r)
        row["evidence_source"] = "phase10_13/intent_vs_observed_summary.csv"
        row["claim_bucket"] = f"{row.get('source_phase', '')}:{row.get('setting_group', '')}"
        out.append(row)

    out.extend(summarize_phase9_concept_observed(phase9_main))

    out.sort(
        key=lambda r: (
            phase_sort_key(str(r.get("source_phase", ""))),
            str(r.get("setting_group", "")),
        )
    )
    return out


def parse_handoff_timeline(master_handoff_text: str) -> List[Dict[str, Any]]:
    lines = master_handoff_text.splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("| Phase | Goal | What changed | Key result"):
            start_idx = i
            break

    if start_idx is None:
        return []

    # header, separator, then rows
    rows: List[Dict[str, Any]] = []
    for line in lines[start_idx + 2 :]:
        s = line.strip()
        if not s.startswith("|"):
            break
        if s.startswith("| ---"):
            continue
        parts = [p.strip() for p in s.strip("|").split("|")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "phase_label": parts[0],
                "goal": parts[1],
                "what_changed": parts[2],
                "key_result": parts[3],
                "interpretation": parts[4],
                "source_doc": "docs/FMOE_N3_KuaiRec_Strict_Master_Handoff.md",
            }
        )

    # pre9 only
    keep = {"Baseline", "Core", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
    rows = [r for r in rows if str(r.get("phase_label", "")) in keep]
    return rows


def build_source_manifest(repo_root: Path, out_data_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for key, rel in INPUT_REL.items():
        p = repo_root / rel
        ext = p.suffix.lower()
        row_count: Optional[int] = None
        col_count: Optional[int] = None
        if p.exists() and ext == ".csv":
            with p.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.reader(fp)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []
                    row_count = 0
                else:
                    row_count = sum(1 for _ in reader)
                col_count = len(header)

        rows.append(
            {
                "kind": "input",
                "key": key,
                "path": str(p),
                "exists": int(p.exists()),
                "row_count": row_count if row_count is not None else "",
                "col_count": col_count if col_count is not None else "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    for key, name in OUTPUT_DATA_FILES.items():
        p = out_data_dir / name
        rows.append(
            {
                "kind": "output",
                "key": key,
                "path": str(p),
                "exists": int(p.exists()),
                "row_count": "",
                "col_count": "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    rows.sort(key=lambda r: (str(r.get("kind", "")), str(r.get("key", ""))))
    return rows


# ----------------------------
# markdown generation
# ----------------------------


def phase_summary(rows: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    phase_rows = [r for r in rows if str(r.get("source_phase", "")) == phase]
    if not phase_rows:
        return {
            "source_phase": phase,
            "rows": 0,
            "best_valid_setting": "",
            "best_valid": None,
            "best_test_setting": "",
            "best_test": None,
            "mean_valid": None,
            "mean_test": None,
            "mean_cold": None,
            "mean_long": None,
            "mean_top1": None,
            "mean_cv": None,
            "mean_n_eff": None,
        }

    by_valid = sorted(phase_rows, key=lambda r: (to_float(r.get("best_valid_mrr20")) or -1e12), reverse=True)
    by_test = sorted(phase_rows, key=lambda r: (to_float(r.get("test_mrr20")) or -1e12), reverse=True)

    return {
        "source_phase": phase,
        "rows": len(phase_rows),
        "best_valid_setting": by_valid[0].get("setting_key", by_valid[0].get("run_phase", "")),
        "best_valid": to_float(by_valid[0].get("best_valid_mrr20")),
        "best_test_setting": by_test[0].get("setting_key", by_test[0].get("run_phase", "")),
        "best_test": to_float(by_test[0].get("test_mrr20")),
        "mean_valid": safe_mean(to_float(r.get("best_valid_mrr20")) for r in phase_rows),
        "mean_test": safe_mean(to_float(r.get("test_mrr20")) for r in phase_rows),
        "mean_cold": safe_mean(to_float(r.get("cold_item_mrr20")) for r in phase_rows),
        "mean_long": safe_mean(to_float(r.get("long_session_mrr20")) for r in phase_rows),
        "mean_top1": safe_mean(to_float(r.get("diag_top1_max_frac")) for r in phase_rows),
        "mean_cv": safe_mean(to_float(r.get("diag_cv_usage")) for r in phase_rows),
        "mean_n_eff": safe_mean(to_float(r.get("diag_n_eff")) for r in phase_rows),
    }


def top_settings_table(
    rows: List[Dict[str, Any]],
    phase: str,
    topn: int = 8,
    split_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    phase_rows = [r for r in rows if str(r.get("source_phase", "")) == phase]
    if split_filter is not None:
        phase_rows = [r for r in phase_rows if str(r.get("split", "")) == split_filter]
    phase_rows = sorted(
        phase_rows,
        key=lambda r: (
            to_float(r.get("best_valid_mrr20")) or -1e12,
            to_float(r.get("test_mrr20")) or -1e12,
        ),
        reverse=True,
    )
    out = []
    for r in phase_rows[:topn]:
        out.append(
            {
                "setting": r.get("setting_key", r.get("run_phase", "")),
                "group": r.get("setting_group", ""),
                "hparam": normalize_hparam_id(r.get("hparam_id")),
                "valid": fmt_num(r.get("best_valid_mrr20")),
                "test": fmt_num(r.get("test_mrr20")),
                "cold": fmt_num(r.get("cold_item_mrr20")),
                "long": fmt_num(r.get("long_session_mrr20")),
                "top1": fmt_num(r.get("diag_top1_max_frac")),
                "cv": fmt_num(r.get("diag_cv_usage")),
                "n_eff": fmt_num(r.get("diag_n_eff"), nd=3),
                "n_completed": str(r.get("n_completed", "")),
            }
        )
    return out


def top_seed_stats_table(rows: List[Dict[str, Any]], phase: str, topn: int = 8) -> List[Dict[str, Any]]:
    phase_rows = [r for r in rows if str(r.get("source_phase", "")) == phase]
    phase_rows = sorted(
        phase_rows,
        key=lambda r: (to_float(r.get("best_valid_mrr20_mean")) or -1e12),
        reverse=True,
    )
    out = []
    for r in phase_rows[:topn]:
        out.append(
            {
                "setting": r.get("setting_key", ""),
                "group": r.get("setting_group", ""),
                "hparam": r.get("hparam_id", ""),
                "seed_n": str(r.get("seed_n", "")),
                "valid_mean_std": f"{fmt_num(r.get('best_valid_mrr20_mean'))} +/- {fmt_num(r.get('best_valid_mrr20_std'))}",
                "test_mean_std": f"{fmt_num(r.get('test_mrr20_mean'))} +/- {fmt_num(r.get('test_mrr20_std'))}",
                "cold_mean": fmt_num(r.get("cold_item_mrr20_mean")),
                "long_mean": fmt_num(r.get("long_session_mrr20_mean")),
                "top1_mean": fmt_num(r.get("diag_top1_max_frac_mean")),
                "cv_mean": fmt_num(r.get("diag_cv_usage_mean")),
            }
        )
    return out


def write_markdown_report(
    path: Path,
    dataset: str,
    wide_all: List[Dict[str, Any]],
    verification_all: List[Dict[str, Any]],
    verification_main: List[Dict[str, Any]],
    verification_support: List[Dict[str, Any]],
    seed_stats: List[Dict[str, Any]],
    diag_corr: List[Dict[str, Any]],
    intent_map: List[Dict[str, Any]],
    legacy_timeline: List[Dict[str, Any]],
) -> None:
    ensure_dir(path.parent)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: List[str] = []
    lines.append("# Phase9~13 + Legacy Integrated Wrap-up")
    lines.append("")
    lines.append(f"작성일: {now}")
    lines.append(f"데이터셋: `{dataset}`")
    lines.append("")

    lines.append("## 1) FMoE 배경 / 읽는 법")
    lines.append("- FMoE_N3는 stage(macro/mid/micro)와 feature family 힌트를 함께 사용해 routing을 구성한다.")
    lines.append("- 본 문서는 기존 결과를 보존한 상태에서 `phase9~13`을 전수 통합하고, `phase1~8`은 타임라인 맥락으로 연결한다.")
    lines.append("- main metric은 `best valid MRR@20`, sub metric은 `test MRR@20`, 보조로 special(cold/long)과 diag(router dynamics)를 같이 해석한다.")
    lines.append("")

    lines.append("## 2) 집계 범위 / 정책")
    lines.append("- 기존 `docs/results`, `docs/visualization`, `docs/data`, logs/artifacts는 수정하지 않았다.")
    lines.append("- 신규 산출물은 `docs/final` 하위에만 생성했다.")
    lines.append("- 통합 row 수:")
    lines.extend(
        md_table(
            [
                {
                    "table": "wide_all_9_13",
                    "rows": len(wide_all),
                    "note": "P9(63) + P10~P13(100)",
                },
                {
                    "table": "verification_all_9_13",
                    "rows": len(verification_all),
                    "note": "P9_2(47) + P10~P13 verification(192)",
                },
                {
                    "table": "verification_main_fair_h3_n20",
                    "rows": len(verification_main),
                    "note": "P9_2(H3,n>=20) + P11~P13(H3,n>=20)",
                },
                {
                    "table": "verification_support_coverage",
                    "rows": len(verification_support),
                    "note": "P10(H1/H3,n=10) + P9_2(non-H3,n>=20)",
                },
            ],
            columns=["table", "rows", "note"],
            headers=["Table", "Rows", "Definition"],
        )
    )
    lines.append("")

    diag_missing_wide = sorted({str(r.get("run_phase", "")) for r in wide_all if to_int(r.get("diag_available"), 0) == 0})
    diag_missing_ver = sorted({str(r.get("run_phase", "")) for r in verification_all if to_int(r.get("diag_available"), 0) == 0})
    lines.append("- diag 누락 run:")
    lines.append(f"  - wide: `{', '.join(diag_missing_wide) if diag_missing_wide else 'none'}`")
    lines.append(f"  - verification: `{', '.join(diag_missing_ver) if diag_missing_ver else 'none'}`")
    lines.append("")

    lines.append("## 3) Legacy 타임라인 (Phase1~8 요약)")
    lines.append("- 동일 스키마 raw가 완전하지 않은 구간은 기존 handoff/result 문서를 근거로 요약했다.")
    if legacy_timeline:
        lines.extend(
            md_table(
                legacy_timeline,
                columns=["phase_label", "goal", "key_result", "interpretation"],
                headers=["Phase", "Goal", "Key Result", "Interpretation"],
            )
        )
    else:
        lines.append("- timeline table parsing failed; source 문서를 직접 참조해야 한다.")
    lines.append("")

    lines.append("## 4) Phase9 상세 (Concept/Setting 중심)")
    p9_rows = [r for r in wide_all if str(r.get("source_phase", "")) == "P9"]
    concept_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in p9_rows:
        concept_group[str(r.get("setting_group", ""))].append(r)

    concept_table = []
    for concept in sorted(concept_group.keys()):
        members = concept_group[concept]
        bv = sorted(members, key=lambda x: to_float(x.get("best_valid_mrr20")) or -1e12, reverse=True)[0]
        bt = sorted(members, key=lambda x: to_float(x.get("test_mrr20")) or -1e12, reverse=True)[0]
        concept_table.append(
            {
                "concept": concept,
                "rows": len(members),
                "best_valid_setting": bv.get("run_phase", ""),
                "best_valid": fmt_num(bv.get("best_valid_mrr20")),
                "best_test_setting": bt.get("run_phase", ""),
                "best_test": fmt_num(bt.get("test_mrr20")),
                "mean_top1": fmt_num(safe_mean(to_float(x.get("diag_top1_max_frac")) for x in members)),
                "mean_cv": fmt_num(safe_mean(to_float(x.get("diag_cv_usage")) for x in members)),
            }
        )

    lines.extend(
        md_table(
            concept_table,
            columns=["concept", "rows", "best_valid_setting", "best_valid", "best_test_setting", "best_test", "mean_top1", "mean_cv"],
            headers=["Concept", "Rows", "Best Valid Setting", "Best Valid", "Best Test Setting", "Best Test", "Mean Top1", "Mean CV"],
        )
    )
    lines.append("- 해석: Phase9는 concept별로 성능 고점과 router 분포가 다르게 움직이며, 단일 aux 우열보다 concept-conditional 비교가 더 설득력 있다.")
    lines.append("")

    lines.append("## 5) Phase10~13 Wide (세팅 중심)")
    for phase in ["P10", "P11", "P12", "P13"]:
        lines.append(f"### {phase} — {PHASE_TITLE.get(phase, phase)}")
        table = top_settings_table(wide_all, phase=phase, topn=8, split_filter="wide")
        lines.extend(
            md_table(
                table,
                columns=["setting", "group", "valid", "test", "cold", "long", "top1", "cv", "n_eff", "n_completed"],
                headers=["Setting", "Group", "Valid", "Test", "Cold", "Long", "Top1", "CV", "n_eff", "n_completed"],
            )
        )
        ps = phase_summary(wide_all, phase)
        lines.append(
            "- 해석: "
            f"valid winner는 `{ps['best_valid_setting']}` ({fmt_num(ps['best_valid'])}), "
            f"test winner는 `{ps['best_test_setting']}` ({fmt_num(ps['best_test'])})이며, "
            f"phase 평균은 valid {fmt_num(ps['mean_valid'])} / test {fmt_num(ps['mean_test'])}로 수렴했다."
        )
        lines.append("")

    lines.append("## 6) Verification 통합 (P9_2 + P10~13)")
    lines.append("### 6.1 Main Fair Table (H3 + n>=20)")
    lines.extend(
        md_table(
            [phase_summary(verification_main, p) for p in ["P9_2", "P11", "P12", "P13"]],
            columns=["source_phase", "rows", "best_valid_setting", "best_valid", "best_test_setting", "best_test", "mean_valid", "mean_test"],
            headers=["Phase", "Rows", "Best Valid Setting", "Best Valid", "Best Test Setting", "Best Test", "Mean Valid", "Mean Test"],
        )
    )
    lines.append("")

    lines.append("### 6.2 Setting별 Seed Mean/Std (Top)")
    seed_rows = []
    for phase in ["P9_2", "P11", "P12", "P13"]:
        seed_rows.extend(top_seed_stats_table(seed_stats, phase, topn=5))
    lines.extend(
        md_table(
            seed_rows,
            columns=["setting", "group", "hparam", "seed_n", "valid_mean_std", "test_mean_std", "cold_mean", "long_mean", "top1_mean", "cv_mean"],
            headers=["Setting", "Group", "Hparam", "SeedN", "Valid mean+-std", "Test mean+-std", "Cold mean", "Long mean", "Top1 mean", "CV mean"],
        )
    )
    lines.append("- 해석: fair 본표에서는 phase별로 valid/test winner가 분리되는 구간이 있어, 논문 본문에선 winner 1개가 아니라 valid/test/stability 3점 제시가 안전하다.")
    lines.append("")

    lines.append("## 7) 통합 Diag/Special/Router")
    phase_rollup = [phase_summary(wide_all + verification_all, p) for p in ["P9", "P9_2", "P10", "P11", "P12", "P13"]]
    lines.extend(
        md_table(
            [
                {
                    "phase": r["source_phase"],
                    "rows": r["rows"],
                    "mean_valid": fmt_num(r["mean_valid"]),
                    "mean_test": fmt_num(r["mean_test"]),
                    "mean_cold": fmt_num(r["mean_cold"]),
                    "mean_long": fmt_num(r["mean_long"]),
                    "mean_top1": fmt_num(r["mean_top1"]),
                    "mean_cv": fmt_num(r["mean_cv"]),
                    "mean_n_eff": fmt_num(r["mean_n_eff"], nd=3),
                }
                for r in phase_rollup
            ],
            columns=["phase", "rows", "mean_valid", "mean_test", "mean_cold", "mean_long", "mean_top1", "mean_cv", "mean_n_eff"],
            headers=["Phase", "Rows", "Mean Valid", "Mean Test", "Mean Cold", "Mean Long", "Mean Top1", "Mean CV", "Mean n_eff"],
        )
    )
    lines.append("")

    # top diag correlations by phase and target=best_valid
    corr_focus = [
        r
        for r in diag_corr
        if str(r.get("target_metric", "")) == "best_valid_mrr20" and (to_int(r.get("n_pairs"), 0) >= 10)
    ]
    corr_focus = sorted(corr_focus, key=lambda r: (phase_sort_key(str(r.get("source_phase", ""))), -(to_float(r.get("abs_spearman")) or 0.0)))

    corr_top = []
    seen = set()
    for r in corr_focus:
        key = (r.get("source_phase", ""), r.get("split", ""))
        if key in seen:
            continue
        seen.add(key)
        corr_top.append(
            {
                "phase": r.get("source_phase", ""),
                "split": r.get("split", ""),
                "diag_metric": r.get("diag_metric", ""),
                "target": r.get("target_metric", ""),
                "spearman": fmt_num(r.get("spearman"), nd=3),
                "pearson": fmt_num(r.get("pearson"), nd=3),
                "n": str(r.get("n_pairs", "")),
            }
        )

    lines.extend(
        md_table(
            corr_top,
            columns=["phase", "split", "diag_metric", "target", "spearman", "pearson", "n"],
            headers=["Phase", "Split", "Diag Metric", "Target", "Spearman", "Pearson", "N"],
        )
    )
    lines.append("- 해석: phase마다 유효한 diag 지표가 다르므로, 단일 router metric으로 전 phase를 설명하기보다 phase-conditioned 해석이 필요하다.")
    lines.append("")

    lines.append("## 8) Plan 가설 대비 관찰 / Claim Bank")
    claim_rows = []
    for r in intent_map:
        claim_rows.append(
            {
                "phase": r.get("source_phase", ""),
                "group": r.get("setting_group", ""),
                "observed": r.get("observed_tag", ""),
                "match": str(r.get("match_flag", "")),
                "best_setting": r.get("best_setting_key", r.get("best_run_phase", "")),
                "delta_valid": fmt_num(r.get("delta_best_valid_vs_anchor")),
                "delta_test": fmt_num(r.get("delta_best_test_vs_anchor")),
                "claim_template": r.get("paper_claim_template", ""),
            }
        )

    lines.extend(
        md_table(
            claim_rows,
            columns=["phase", "group", "observed", "match", "best_setting", "delta_valid", "delta_test", "claim_template"],
            headers=["Phase", "Group", "Observed", "Match", "Best Setting", "dValid", "dTest", "Claim Template"],
        )
    )
    lines.append("- 해석: P9~P13 전체를 연결하면, `feature 선택(P9/P10) -> stage 의미(P11) -> composition(P12) -> sanity counterfactual(P13)`의 논문 스토리라인이 자연스럽게 완성된다.")
    lines.append("")

    lines.append("## 9) 참고")
    lines.append("- 상세 figure는 `docs/final/visualization/phase9_13_wrapup.ipynb`를 참조한다.")
    lines.append("- 원본 근거 파일 목록은 `docs/final/data/phase9_13_wrapup/source_manifest.csv`를 참조한다.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# notebook generation
# ----------------------------


def mcell(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def ccell(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def write_wrapup_notebook(path: Path) -> None:
    ensure_dir(path.parent)

    cells = []
    cells.append(
        mcell(
            """
# Phase9~13 + Legacy Wrap-up Visualization

- Figure text: English only
- Explanations: Korean `print()`
- Focus: setting-level comparison, special slices, diag/router behavior
            """
        )
    )

    cells.append(
        ccell(
            """
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

DATA_DIR = Path("../data/phase9_13_wrapup")

wide = pd.read_csv(DATA_DIR / "wide_all_9_13.csv")
ver_all = pd.read_csv(DATA_DIR / "verification_all_9_13.csv")
ver_main = pd.read_csv(DATA_DIR / "verification_main_fair_h3_n20.csv")
ver_support = pd.read_csv(DATA_DIR / "verification_support_coverage.csv")
seed_stats = pd.read_csv(DATA_DIR / "verification_seed_stats_9_13.csv")
diag_long = pd.read_csv(DATA_DIR / "diag_special_long_9_13.csv")
diag_corr = pd.read_csv(DATA_DIR / "diag_corr_by_phase.csv")
diag_q = pd.read_csv(DATA_DIR / "diag_quantile_profile.csv")
router_stage = pd.read_csv(DATA_DIR / "router_stage_scalar_9_13.csv")
router_family = pd.read_csv(DATA_DIR / "router_family_long_10_13.csv")
family_pca = pd.read_csv(DATA_DIR / "family_pca_points_9_13.csv")
intent_map = pd.read_csv(DATA_DIR / "intent_claim_evidence_map_9_13.csv")
legacy = pd.read_csv(DATA_DIR / "legacy_timeline_pre9_summary.csv")

num_cols = [
    "best_valid_mrr20", "test_mrr20", "cold_item_mrr20", "long_session_mrr20", "sess_3_5_mrr20",
    "diag_top1_max_frac", "diag_cv_usage", "diag_n_eff", "diag_entropy_mean",
    "best_valid_mrr20_mean", "best_valid_mrr20_std", "test_mrr20_mean", "test_mrr20_std",
    "cold_item_mrr20_mean", "long_session_mrr20_mean", "diag_top1_max_frac_mean", "diag_cv_usage_mean",
    "pearson", "spearman", "target_mean", "diag_mean", "diag_quantile",
    "n_eff", "cv_usage", "top1_max_frac", "entropy_mean",
    "family_expert_share_norm", "expert_global_share", "pc1", "pc2"
]

for df in [wide, ver_all, ver_main, ver_support, seed_stats, diag_long, diag_corr, diag_q, router_stage, router_family, family_pca, intent_map]:
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

print("[설명] 통합 wrap-up 데이터셋을 로드했습니다.")
print("[설명] wide/ver/main/support/diag/router/claim 테이블을 모두 메모리에 올렸습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[요약] 테이블 row 수를 먼저 확인합니다.")
summary = pd.DataFrame([
    {"table": "wide_all_9_13", "rows": len(wide)},
    {"table": "verification_all_9_13", "rows": len(ver_all)},
    {"table": "verification_main_fair_h3_n20", "rows": len(ver_main)},
    {"table": "verification_support_coverage", "rows": len(ver_support)},
    {"table": "verification_seed_stats_9_13", "rows": len(seed_stats)},
    {"table": "diag_corr_by_phase", "rows": len(diag_corr)},
    {"table": "router_stage_scalar_9_13", "rows": len(router_stage)},
])
display(summary)

print("[설명] phase별 row 분포를 확인해서 누락 phase가 없는지 확인합니다.")
phase_counts = pd.concat([
    wide.assign(table="wide")[ ["table", "source_phase"] ],
    ver_all.assign(table="verification")[ ["table", "source_phase"] ],
]).groupby(["table", "source_phase"]).size().reset_index(name="rows")
display(phase_counts)
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] Legacy(pre9) 타임라인 요약을 출력해 wrap-up 맥락을 먼저 잡습니다.")
display(legacy)

if not legacy.empty:
    plt.figure(figsize=(8, 3.8))
    order = legacy["phase_label"].tolist()
    y = np.arange(len(order))
    plt.barh(y, np.arange(1, len(order)+1), color="#4c78a8")
    plt.yticks(y, order)
    plt.xlabel("Timeline Index")
    plt.title("Legacy Phase Timeline (Pre-9)")
    plt.tight_layout()
    plt.show()
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] Phase9는 concept 내부 세팅 비교가 핵심이므로 setting-level scatter를 먼저 봅니다.")
p9 = wide[wide["source_phase"] == "P9"].copy()

if p9.empty:
    print("P9 rows are empty")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    sns.scatterplot(
        data=p9,
        x="best_valid_mrr20",
        y="test_mrr20",
        hue="setting_group",
        style="setting_short" if "setting_short" in p9.columns else None,
        s=55,
        alpha=0.9,
        ax=axes[0],
    )
    axes[0].set_title("P9 Setting Scatter (Valid vs Test)")
    axes[0].set_xlabel("best_valid_mrr20")
    axes[0].set_ylabel("test_mrr20")
    leg = axes[0].get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1.0))
        leg._loc = 2

    agg = p9.groupby("setting_group", as_index=False).agg(
        valid_mean=("best_valid_mrr20", "mean"),
        test_mean=("test_mrr20", "mean"),
        top1_mean=("diag_top1_max_frac", "mean"),
        cv_mean=("diag_cv_usage", "mean"),
    )
    sns.scatterplot(data=agg, x="top1_mean", y="valid_mean", hue="setting_group", s=120, ax=axes[1])
    axes[1].set_title("P9 Concept Router Signal vs Valid")
    axes[1].set_xlabel("diag_top1_max_frac (mean)")
    axes[1].set_ylabel("best_valid_mrr20 (mean)")
    leg2 = axes[1].get_legend()
    if leg2 is not None:
        leg2.set_bbox_to_anchor((1.02, 1.0))
        leg2._loc = 2

    plt.tight_layout()
    plt.show()

    print("[설명] 좌측은 concept 내 세팅 분포, 우측은 concept 평균 router 신호(top1)와 valid 관계를 보여줍니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] P10~P13 wide는 축 비교보다 세팅 내부 비교에 집중해 phase별 top setting을 시각화합니다.")

def add_bar_labels(ax, fmt="{:.4f}"):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

def set_tight_ylim(ax, series, pad=0.0006):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        return
    lo = float(vals.min())
    hi = float(vals.max())
    if np.isclose(lo, hi):
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(lo - max((hi - lo) * 0.2, pad), hi + max((hi - lo) * 0.2, pad))

for phase in ["P10", "P11", "P12", "P13"]:
    sub = wide[(wide["source_phase"] == phase)].copy()
    if sub.empty:
        continue
    sub = sub.sort_values("best_valid_mrr20", ascending=False).head(8)
    sub["setting_plot"] = sub["setting_key"].fillna(sub["run_phase"]).astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    sns.barplot(data=sub, x="setting_plot", y="best_valid_mrr20", hue="setting_group", dodge=False, ax=axes[0])
    axes[0].set_title(f"{phase} Wide Top Settings (Valid)")
    axes[0].set_xlabel("setting")
    axes[0].set_ylabel("best_valid_mrr20")
    axes[0].tick_params(axis="x", rotation=60)
    set_tight_ylim(axes[0], sub["best_valid_mrr20"])
    add_bar_labels(axes[0])
    leg = axes[0].get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1.0))
        leg._loc = 2

    sns.barplot(data=sub, x="setting_plot", y="test_mrr20", hue="setting_group", dodge=False, ax=axes[1])
    axes[1].set_title(f"{phase} Wide Top Settings (Test)")
    axes[1].set_xlabel("setting")
    axes[1].set_ylabel("test_mrr20")
    axes[1].tick_params(axis="x", rotation=60)
    set_tight_ylim(axes[1], sub["test_mrr20"])
    add_bar_labels(axes[1])
    leg2 = axes[1].get_legend()
    if leg2 is not None:
        leg2.set_bbox_to_anchor((1.02, 1.0))
        leg2._loc = 2

    plt.tight_layout()
    plt.show()

print("[설명] phase별로 valid/test winner가 달라지는 지점을 세팅 단위로 확인할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] verification는 mean-std 안정성 관점으로 봅니다. legend는 그래프 밖으로 배치합니다.")
main_phase = seed_stats[seed_stats["source_phase"].isin(["P9_2", "P11", "P12", "P13"])].copy()
main_phase = main_phase[main_phase["hparam_id"].astype(str).str.upper().eq("H3") | main_phase["source_phase"].eq("P9_2")]

if main_phase.empty:
    print("seed_stats for verification focus is empty")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    sns.scatterplot(
        data=main_phase,
        x="best_valid_mrr20_mean",
        y="best_valid_mrr20_std",
        hue="source_phase",
        style="setting_group",
        s=70,
        alpha=0.9,
        ax=axes[0],
    )
    axes[0].set_title("Verification Stability (Valid Mean vs Std)")
    axes[0].set_xlabel("best_valid_mrr20_mean")
    axes[0].set_ylabel("best_valid_mrr20_std")
    leg = axes[0].get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.03, 1.0))
        leg._loc = 2

    sns.scatterplot(
        data=main_phase,
        x="test_mrr20_mean",
        y="test_mrr20_std",
        hue="source_phase",
        style="setting_group",
        s=70,
        alpha=0.9,
        ax=axes[1],
    )
    axes[1].set_title("Verification Stability (Test Mean vs Std)")
    axes[1].set_xlabel("test_mrr20_mean")
    axes[1].set_ylabel("test_mrr20_std")
    leg2 = axes[1].get_legend()
    if leg2 is not None:
        leg2.set_bbox_to_anchor((1.03, 1.0))
        leg2._loc = 2

    plt.tight_layout()
    plt.show()

print("[설명] 평균이 높고 표준편차가 낮은 영역이 배포 관점에서 유리한 후보입니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] test/cold/long metric은 스케일이 다르므로 분리된 plot으로 비교합니다.")
focus = seed_stats[(seed_stats["source_phase"].isin(["P9_2", "P11", "P12", "P13"]))].copy()
focus = focus.sort_values("best_valid_mrr20_mean", ascending=False).head(12).copy()
focus["setting_plot"] = focus["setting_key"].fillna("na")

metric_specs = [
    ("test_mrr20_mean", "Verification Test Mean"),
    ("cold_item_mrr20_mean", "Verification Cold Mean"),
    ("long_session_mrr20_mean", "Verification Long-session Mean"),
]

for metric, title in metric_specs:
    if metric not in focus.columns:
        continue
    sub = focus[["setting_plot", metric, "source_phase"]].dropna()
    if sub.empty:
        continue
    plt.figure(figsize=(13, 4.6))
    ax = sns.barplot(data=sub, x="setting_plot", y=metric, hue="source_phase")
    ax.set_title(title)
    ax.set_xlabel("setting")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=65)

    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if len(vals) > 0:
        lo, hi = vals.min(), vals.max()
        pad = max((hi - lo) * 0.2, 0.0005)
        ax.set_ylim(lo - pad, hi + pad)

    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width()/2, h),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 1), textcoords="offset points")

    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1.0))
        leg._loc = 2

    plt.tight_layout()
    plt.show()

print("[설명] 분리 플롯으로 metric별 유리 세팅이 어떻게 달라지는지 명확히 볼 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] phase/metric별 diag 상관 히트맵으로 router 동작과 성능의 연결을 확인합니다.")
sub = diag_corr[diag_corr["target_metric"].isin(["best_valid_mrr20", "test_mrr20"])].copy()
sub = sub[sub["n_pairs"] >= 8]

if sub.empty:
    print("diag_corr has no sufficient rows")
else:
    for target in ["best_valid_mrr20", "test_mrr20"]:
        ss = sub[sub["target_metric"] == target].copy()
        if ss.empty:
            continue
        # phase x diag metric using spearman over all splits (mean)
        pivot = ss.pivot_table(index="source_phase", columns="diag_metric", values="spearman", aggfunc="mean")
        pivot = pivot.reindex(index=[p for p in ["P9", "P9_2", "P10", "P11", "P12", "P13"] if p in pivot.index])

        w = min(16.0, max(9.5, 0.85 * len(pivot.columns) + 2.2))
        h = max(3.6, 0.7 * len(pivot.index) + 1.4)
        plt.figure(figsize=(w, h))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            annot_kws={"size": 8},
            cbar_kws={"label": "Spearman"},
        )
        plt.title(f"Diag vs {target} (Phase-level Spearman)")
        plt.xlabel("diag_metric")
        plt.ylabel("source_phase")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

print("[설명] phase마다 sign/strength가 달라지는 metric이 있는지 확인해 주장 범위를 phase-conditioned로 제한할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] diag quantile trend로 단조/비단조 패턴을 확인합니다.")
qsub = diag_q[(diag_q["target_metric"] == "best_valid_mrr20") & (diag_q["diag_quantile"].between(1, 5))].copy()

if qsub.empty:
    print("diag_quantile_profile is empty")
else:
    # choose representative metrics by coverage
    metric_order = (
        qsub.groupby("diag_metric")["n_pairs"].sum().sort_values(ascending=False).head(6).index.tolist()
    )
    qsub = qsub[qsub["diag_metric"].isin(metric_order)]

    g = sns.FacetGrid(
        qsub,
        row="source_phase",
        col="diag_metric",
        margin_titles=True,
        sharex=True,
        sharey=False,
        height=2.1,
        aspect=1.25,
    )
    g.map_dataframe(sns.lineplot, x="diag_quantile", y="target_mean", marker="o", color="#1f77b4")
    g.set_axis_labels("diag quantile", "best_valid_mrr20")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    for ax in g.axes.flat:
        if ax is None:
            continue
        for line in ax.lines:
            xd = line.get_xdata()
            yd = line.get_ydata()
            for x, y in zip(xd, yd):
                if np.isfinite(y):
                    ax.annotate(f"{y:.4f}", (x, y), fontsize=7, xytext=(0, 2), textcoords="offset points", ha="center")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle("Diag Quantile Trend (Valid)")
    plt.show()

print("[설명] quantile 기반으로 어떤 diag 구간에서 성능이 올라가는지, 혹은 U-shape인지 확인할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] router stage scalar를 이용해 n_eff/cv/top1 분포를 phase별로 비교합니다.")
rs = router_stage.copy()
rs = rs[rs["source_phase"].isin(["P9", "P9_2", "P10", "P11", "P12", "P13"])]

metrics = [
    ("n_eff", "Router n_eff Distribution"),
    ("cv_usage", "Router cv_usage Distribution"),
    ("top1_max_frac", "Router top1_max_frac Distribution"),
]

for metric, title in metrics:
    if metric not in rs.columns:
        continue
    sub = rs[["source_phase", "split", metric]].dropna().copy()
    if sub.empty:
        continue
    plt.figure(figsize=(11, 4.3))
    ax = sns.boxplot(data=sub, x="source_phase", y=metric, hue="split")
    ax.set_title(title)
    ax.set_xlabel("source_phase")
    ax.set_ylabel(metric)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_bbox_to_anchor((1.02, 1.0))
        leg._loc = 2
    plt.tight_layout()
    plt.show()

print("[설명] phase가 진행되며 router 분포가 균형/집중 방향 중 어디로 이동하는지 확인할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] router family heatmap으로 세팅별 feature-family expert 사용 패턴을 봅니다.")
rf = router_family.copy()
rf = rf[rf["source_phase"].isin(["P10", "P11", "P12", "P13"])].copy()

if rf.empty:
    print("router_family is empty")
else:
    top_settings = (
        rf.groupby("setting_key", as_index=False)["best_valid_mrr20"]
        .mean()
        .sort_values("best_valid_mrr20", ascending=False)
        .head(8)["setting_key"]
        .tolist()
    )
    sub = rf[rf["setting_key"].isin(top_settings)].copy()
    if not sub.empty:
        pivot = sub.pivot_table(
            index="setting_key",
            columns="family",
            values="family_expert_share_norm",
            aggfunc="mean",
        )
        plt.figure(figsize=(8.5, max(3.8, 0.45 * len(pivot.index) + 1.6)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            annot_kws={"size": 8},
            cbar_kws={"label": "Mean family_expert_share_norm"},
        )
        plt.title("Family-level Expert Usage (Top Settings)")
        plt.xlabel("family")
        plt.ylabel("setting")
        plt.tight_layout()
        plt.show()

print("[설명] 어떤 세팅이 특정 family 사용을 강화/완화하는지 확인하고, 계획 의도와 연결할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] phase9~13 PCA/proxy scatter로 family-expert 분포 지형을 봅니다.")
fp = family_pca.copy()
fp = fp[fp["phase_tag"].isin(["phase9", "phase9_2", "p10", "p11", "p12", "p13"])].copy()

if fp.empty:
    print("family_pca_points_9_13 is empty")
else:
    # normalize phase tags
    fp["phase_tag"] = fp["phase_tag"].str.upper()
    fp["pc1"] = pd.to_numeric(fp["pc1"], errors="coerce")
    fp["pc2"] = pd.to_numeric(fp["pc2"], errors="coerce")
    sub = fp.dropna(subset=["pc1", "pc2"]).copy()

    if not sub.empty:
        plt.figure(figsize=(9.2, 6.0))
        ax = sns.scatterplot(
            data=sub.sample(min(5000, len(sub)), random_state=42),
            x="pc1",
            y="pc2",
            hue="phase_tag",
            style="family" if "family" in sub.columns else None,
            s=18,
            alpha=0.65,
        )
        ax.set_title("Feature-family Embedding View (P9~P13)")
        ax.set_xlabel("pc1")
        ax.set_ylabel("pc2")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_bbox_to_anchor((1.02, 1.0))
            leg._loc = 2
        plt.tight_layout()
        plt.show()

print("[설명] phase별 cluster 이동이 보이면 router가 family 표현을 다르게 사용하는 근거로 제시할 수 있습니다.")
            """
        )
    )

    cells.append(
        ccell(
            """
print("[설명] 최종 claim-evidence 맵을 출력합니다.")
cols = [
    c for c in [
        "source_phase", "setting_group", "observed_tag", "match_flag",
        "best_setting_key", "delta_best_valid_vs_anchor", "delta_best_test_vs_anchor", "paper_claim_template"
    ] if c in intent_map.columns
]

display(intent_map[cols].sort_values(["source_phase", "setting_group"]))

print("[결론] 이 노트북은 setting-level 비교 + diag/special 증거를 기반으로 논문 주장 문장을 연결하는 데 초점을 둡니다.")
            """
        )
    )

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nbf.write(nb, path)


# ----------------------------
# orchestration
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase9~13 + legacy integrated wrap-up package.")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out-root", default="/workspace/jy1559/FMoE/experiments/run/fmoe_n3/docs/final")
    parser.add_argument("--main-hparam", default="H3")
    parser.add_argument("--main-min-completed", type=int, default=20)
    parser.add_argument("--include-phase9-2", type=parse_bool, default=True)
    parser.add_argument("--include-pre9-doc-digest", type=parse_bool, default=True)
    parser.add_argument("--emit-markdown", type=parse_bool, default=True)
    parser.add_argument("--emit-notebook", type=parse_bool, default=True)
    return parser.parse_args()


def run_build(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    dataset = str(args.dataset)
    out_root = Path(args.out_root)

    out_data = out_root / "data" / "phase9_13_wrapup"
    out_results = out_root / "results"
    out_viz = out_root / "visualization"

    ensure_dir(out_data)
    ensure_dir(out_results)
    ensure_dir(out_viz)

    # load inputs
    phase9_main = read_csv_rows(repo_root / INPUT_REL["phase9_main"])
    phase9_2_main = read_csv_rows(repo_root / INPUT_REL["phase9_2_main"])
    phase9_2_pending = read_csv_rows(repo_root / INPUT_REL["phase9_2_pending"])

    wide_10_13 = read_csv_rows(repo_root / INPUT_REL["wide_all_dedup"])
    ver_10_13 = read_csv_rows(repo_root / INPUT_REL["verification_all_dedup"])

    router_stage_10_13 = read_csv_rows(repo_root / INPUT_REL["router_stage_scalar"])
    router_family_10_13 = read_csv_rows(repo_root / INPUT_REL["router_family_expert_long"])
    router_pos_10_13 = read_csv_rows(repo_root / INPUT_REL["router_position_expert_long"])

    intent_10_13 = read_csv_rows(repo_root / INPUT_REL["intent_vs_observed_summary"])
    legacy_pca_rows = read_csv_rows(repo_root / INPUT_REL["family_expert_pca_points"])

    # phase9_2 inclusion toggle
    phase9_2_for_build = phase9_2_main if bool(args.include_phase9_2) else []

    # main tables
    wide_all = build_wide_all_9_13(phase9_main=phase9_main, wide_10_13=wide_10_13, dataset=dataset)
    verification_all = build_verification_all_9_13(
        phase9_2_main=phase9_2_for_build,
        ver_10_13=ver_10_13,
        dataset=dataset,
    )

    verification_main = build_verification_main_fair_h3_n20(
        verification_all=verification_all,
        main_hparam=str(args.main_hparam),
        main_min_completed=int(args.main_min_completed),
    )

    verification_support = build_verification_support_coverage(
        verification_all=verification_all,
        main_hparam=str(args.main_hparam),
        main_min_completed=int(args.main_min_completed),
    )

    verification_seed_stats = summarize_verification_seed_stats(verification_all)

    # diag tables
    diag_special_long = build_diag_special_long_9_13(wide_all, verification_all)
    diag_corr = summarize_diag_corr_by_phase(diag_special_long)
    diag_quantile = summarize_diag_quantile_profile(diag_special_long, q=5)

    # router tables
    router_stage_9_13 = build_router_stage_scalar_9_13(
        router_stage_10_13=router_stage_10_13,
        phase9_main=phase9_main,
        phase9_2_main=phase9_2_for_build,
    )

    router_family_long_10_13 = [dict(r) for r in router_family_10_13]
    router_position_long_10_13 = [dict(r) for r in router_pos_10_13]

    family_pca_points = build_family_pca_points_9_13(
        legacy_pca_rows=legacy_pca_rows,
        router_family_rows=router_family_long_10_13,
    )

    intent_claim_map = build_intent_claim_evidence_map_9_13(
        intent_10_13=intent_10_13,
        phase9_main=phase9_main,
    )

    # legacy timeline
    legacy_timeline: List[Dict[str, Any]] = []
    if bool(args.include_pre9_doc_digest):
        master_handoff_text = read_text(repo_root / INPUT_REL["master_handoff_md"])
        legacy_timeline = parse_handoff_timeline(master_handoff_text)

    # write CSV outputs
    write_csv(out_data / OUTPUT_DATA_FILES["wide_all_9_13"], wide_all)
    write_csv(out_data / OUTPUT_DATA_FILES["verification_all_9_13"], verification_all)
    write_csv(out_data / OUTPUT_DATA_FILES["verification_main_fair_h3_n20"], verification_main)
    write_csv(out_data / OUTPUT_DATA_FILES["verification_support_coverage"], verification_support)
    write_csv(out_data / OUTPUT_DATA_FILES["verification_seed_stats_9_13"], verification_seed_stats)

    write_csv(out_data / OUTPUT_DATA_FILES["diag_special_long_9_13"], diag_special_long)
    write_csv(out_data / OUTPUT_DATA_FILES["diag_corr_by_phase"], diag_corr)
    write_csv(out_data / OUTPUT_DATA_FILES["diag_quantile_profile"], diag_quantile)

    write_csv(out_data / OUTPUT_DATA_FILES["router_stage_scalar_9_13"], router_stage_9_13)
    write_csv(out_data / OUTPUT_DATA_FILES["router_family_long_10_13"], router_family_long_10_13)
    write_csv(out_data / OUTPUT_DATA_FILES["router_position_long_10_13"], router_position_long_10_13)
    write_csv(out_data / OUTPUT_DATA_FILES["family_pca_points_9_13"], family_pca_points)

    write_csv(out_data / OUTPUT_DATA_FILES["intent_claim_evidence_map_9_13"], intent_claim_map)
    write_csv(out_data / OUTPUT_DATA_FILES["legacy_timeline_pre9_summary"], legacy_timeline)

    # source manifest after outputs exist
    manifest = build_source_manifest(repo_root=repo_root, out_data_dir=out_data)
    write_csv(out_data / OUTPUT_DATA_FILES["source_manifest"], manifest)

    # markdown
    md_path = out_results / "phase9_13_wrapup.md"
    if bool(args.emit_markdown):
        write_markdown_report(
            path=md_path,
            dataset=dataset,
            wide_all=wide_all,
            verification_all=verification_all,
            verification_main=verification_main,
            verification_support=verification_support,
            seed_stats=verification_seed_stats,
            diag_corr=diag_corr,
            intent_map=intent_claim_map,
            legacy_timeline=legacy_timeline,
        )

    # notebook
    nb_path = out_viz / "phase9_13_wrapup.ipynb"
    if bool(args.emit_notebook):
        write_wrapup_notebook(nb_path)

    # validation prints
    print("[validation] row counts")
    count_map = {
        "wide_all_9_13": len(wide_all),
        "verification_all_9_13": len(verification_all),
        "verification_main_fair_h3_n20": len(verification_main),
        "verification_support_coverage": len(verification_support),
    }
    for key, expected in EXPECTED_COUNTS.items():
        actual = count_map.get(key, -1)
        mark = "OK" if actual == expected else "MISMATCH"
        print(f"  - {key}: {actual} (expected {expected}) [{mark}]")

    # special/diag missing checks
    special_missing = {
        "wide": sum(1 for r in wide_all if to_int(r.get("special_available"), 0) == 0),
        "verification": sum(1 for r in verification_all if to_int(r.get("special_available"), 0) == 0),
    }
    diag_missing = {
        "wide": sorted({str(r.get("run_phase", "")) for r in wide_all if to_int(r.get("diag_available"), 0) == 0}),
        "verification": sorted({str(r.get("run_phase", "")) for r in verification_all if to_int(r.get("diag_available"), 0) == 0}),
    }

    print("[validation] special missing:", special_missing)
    print("[validation] diag missing wide:", ",".join(diag_missing["wide"]) if diag_missing["wide"] else "none")
    print("[validation] diag missing verification:", ",".join(diag_missing["verification"]) if diag_missing["verification"] else "none")

    print("[done] outputs")
    print("  - data:", out_data)
    print("  - markdown:", md_path if bool(args.emit_markdown) else "(skipped)")
    print("  - notebook:", nb_path if bool(args.emit_notebook) else "(skipped)")
    print("  - phase9_2_pending rows (reference only):", len(phase9_2_pending))

    return 0


def main() -> int:
    args = parse_args()
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
