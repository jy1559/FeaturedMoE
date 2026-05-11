#!/usr/bin/env python3
"""Build phase10~13 wide/verification report data tables for KuaiRec FMoE_N3.

Outputs (default: docs/data/phase10_13):
- wide_all_dedup.csv
- wide_phase_summary.csv
- wide_axis_summary.csv
- verification_all_dedup.csv
- verification_main_h3_n20.csv
- verification_support_p10_n10.csv
- verification_setting_seed_stats.csv
- router_family_expert_long.csv
- router_position_expert_long.csv
- router_stage_scalar.csv
- intent_expectation_map.csv
- intent_vs_observed_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE_ORDER: List[str] = ["P10", "P11", "P12", "P13"]
PHASE_RANK: Dict[str, int] = {p: i for i, p in enumerate(PHASE_ORDER)}
PHASE_TITLE: Dict[str, str] = {
    "P10": "Feature Portability / Compactness",
    "P11": "Stage Semantics / Necessity / Granularity",
    "P12": "Layout Composition / Attention Placement",
    "P13": "Feature Sanity / Corruption Checks",
}
PHASE_ANCHOR_SETTING: Dict[str, str] = {
    "P10": "P10-00_FULL",
    "P11": "P11-00_MACRO_MID_MICRO",
    "P12": "P12-00_ATTN_ONESHOT",
    "P13": "P13-00_FULL_DATA",
}

PHASE_WIDE_SUMMARY_REL: Dict[str, str] = {
    "P10": "experiments/run/artifacts/logs/fmoe_n3/phase10_feature_portability_v1/P10/{dataset}/summary.csv",
    "P11": "experiments/run/artifacts/logs/fmoe_n3/phase11_stage_semantics_v1/{dataset}/summary.csv",
    "P12": "experiments/run/artifacts/logs/fmoe_n3/phase12_layout_composition_v1/{dataset}/summary.csv",
    "P13": "experiments/run/artifacts/logs/fmoe_n3/phase13_feature_sanity_v1/{dataset}/summary.csv",
}

PHASE_AXIS_NAME: Dict[str, str] = {
    "P10": "phase10_feature_portability_v1",
    "P11": "phase11_stage_semantics_v1",
    "P12": "phase12_layout_composition_v1",
    "P13": "phase13_feature_sanity_v1",
}

VERIFICATION_SUMMARY_REL = (
    "experiments/run/artifacts/logs/fmoe_n3/phase10_13_verification_wrapper_v1/{dataset}/summary.csv"
)

EXPECTED_WIDE_COUNTS = {"P10": 22, "P11": 24, "P12": 32, "P13": 22}
EXPECTED_VERIFICATION_DEDUP = 192
EXPECTED_VERIFICATION_MAIN = 96
EXPECTED_VERIFICATION_SUPPORT = 96


INTENT_EXPECTATION_MAP: List[Dict[str, str]] = [
    {
        "source_phase": "P10",
        "setting_group": "group_subset",
        "axis_label": "Family subset lattice",
        "plan_intent": "Identify which feature families are necessary and whether 2~3-family subsets stay near full setting.",
        "expected_pattern": "Competitive or near-anchor valid/test with interpretable family sensitivity.",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "A compact subset of feature families can preserve most of the gain, indicating portability beyond large handcrafted banks.",
    },
    {
        "source_phase": "P10",
        "setting_group": "compactness",
        "axis_label": "Intra-group compactness",
        "plan_intent": "Reduce feature count within each family (TOP2/TOP1/common template) to test reusable compact templates.",
        "expected_pattern": "Small-to-moderate drop from anchor while retaining strong test behavior.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Few representative signals per family are sufficient for robust feature-aware routing.",
    },
    {
        "source_phase": "P10",
        "setting_group": "availability",
        "axis_label": "Availability ablation",
        "plan_intent": "Drop category/timestamp families structurally to test portability under missing common metadata.",
        "expected_pattern": "Moderate degradation, not catastrophic collapse.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "The framework remains functional even under partial feature availability constraints.",
    },
    {
        "source_phase": "P10",
        "setting_group": "stochastic",
        "axis_label": "Stochastic usage",
        "plan_intent": "Apply feature/family dropout during training to probe robustness to feature availability variation.",
        "expected_pattern": "Comparable or slightly better test generalization with stable router diagnostics.",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "Stochastic feature usage improves robustness without requiring larger feature banks.",
    },
    {
        "source_phase": "P10",
        "setting_group": "availability_plus",
        "axis_label": "Availability plus",
        "plan_intent": "Stress missing-signal conditions with combined category+timestamp removals.",
        "expected_pattern": "Noticeable degradation; useful as stress-control evidence.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Severe multi-signal removal reveals the boundary of portability.",
    },
    {
        "source_phase": "P10",
        "setting_group": "compactness_plus",
        "axis_label": "Compactness plus",
        "plan_intent": "Combine compact template with category removal for stricter portability checks.",
        "expected_pattern": "Moderate drop but non-trivial residual performance.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Even stricter compactness-plus settings preserve a meaningful portion of gains.",
    },
    {
        "source_phase": "P11",
        "setting_group": "base_ablation",
        "axis_label": "Base stage ablation",
        "plan_intent": "Test whether macro/mid/micro decomposition is meaningful versus reduced stage sets.",
        "expected_pattern": "Full 3-stage or select 2-stage settings should lead; single-stage likely weaker.",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "Performance depends on temporal-horizon decomposition, not just MoE presence.",
    },
    {
        "source_phase": "P11",
        "setting_group": "prepend_layer",
        "axis_label": "Prepend dense layer",
        "plan_intent": "Check if gains are just from extra dense depth before stage routing.",
        "expected_pattern": "No consistent domination over base-stage variants.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Stage decomposition contributes beyond simply adding dense contextualization depth.",
    },
    {
        "source_phase": "P11",
        "setting_group": "order_permutation",
        "axis_label": "Stage order permutation",
        "plan_intent": "Measure sensitivity to stage order to validate coarse-to-fine rationale.",
        "expected_pattern": "Order differences are visible; a few permutations may compete strongly.",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "Ordering influences routing behavior, supporting stage-semantic interpretation.",
    },
    {
        "source_phase": "P11",
        "setting_group": "routing_granularity",
        "axis_label": "Routing granularity",
        "plan_intent": "Compare session-level and token-level routing for macro/mid stages.",
        "expected_pattern": "Session-aware variants should remain competitive and interpretable.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Macro/mid routing behaves naturally as session-regime routing.",
    },
    {
        "source_phase": "P11",
        "setting_group": "extra_alignment",
        "axis_label": "Depth controls",
        "plan_intent": "Use layer-only controls to reject purely depth-driven explanations.",
        "expected_pattern": "Control settings should not dominate best stage-semantic variants.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Improvements are not explained by generic depth increases alone.",
    },
    {
        "source_phase": "P12",
        "setting_group": "layout_variants",
        "axis_label": "Layout variants",
        "plan_intent": "Change attention placement and local refinement while keeping stage set.",
        "expected_pattern": "Several layout variants should be strong and stable.",
        "expectation_type": "target_near_or_better",
        "paper_claim_template": "Composition details matter: identical stage sets can differ by layout quality.",
    },
    {
        "source_phase": "P12",
        "setting_group": "bundle_pair_then_follow",
        "axis_label": "Pair bundle then follow",
        "plan_intent": "Test selected horizon interactions via 2-stage bundles before follow-up stage.",
        "expected_pattern": "Some competitive cases, but mixed stability.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Selected horizon interaction can help, but not all bundling is beneficial.",
    },
    {
        "source_phase": "P12",
        "setting_group": "bundle_all",
        "axis_label": "All-stage bundle",
        "plan_intent": "Use all horizons simultaneously as aggressive alternative to serial decomposition.",
        "expected_pattern": "Likely weaker than top serial/layout variants; acts as negative control.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Serial horizon separation is often preferable to collapsing all horizons at once.",
    },
    {
        "source_phase": "P12",
        "setting_group": "bundle_chain",
        "axis_label": "Bundle chain",
        "plan_intent": "Chain multiple bundles to test deeper interaction composition.",
        "expected_pattern": "Generally lower stability than top layout variants.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "More complex bundle chains do not automatically improve routing quality.",
    },
    {
        "source_phase": "P12",
        "setting_group": "bundle_router",
        "axis_label": "Router-conditioned bundle",
        "plan_intent": "Add router-conditioned aggregation for adaptive bundle composition.",
        "expected_pattern": "Can recover part of bundle gap with better adaptivity.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Adaptive aggregation helps, but composition bias remains a critical factor.",
    },
    {
        "source_phase": "P13",
        "setting_group": "data_condition",
        "axis_label": "Data condition",
        "plan_intent": "Simulate category-zero data conditions without structural feature removal.",
        "expected_pattern": "Moderate drop relative to clean full-data anchor.",
        "expectation_type": "tradeoff_or_stability",
        "paper_claim_template": "Category signals help, but the framework can remain functional under weakened category cues.",
    },
    {
        "source_phase": "P13",
        "setting_group": "eval_perturb",
        "axis_label": "Eval perturbation",
        "plan_intent": "Perturb features only at evaluation to test direct feature usage at inference.",
        "expected_pattern": "Clear drop under all-zero/all-shuffle controls.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Inference-time perturbation degrades performance, indicating active feature usage.",
    },
    {
        "source_phase": "P13",
        "setting_group": "train_corruption",
        "axis_label": "Train corruption",
        "plan_intent": "Break feature-sequence alignment during training to separate parameter count from aligned signal learning.",
        "expected_pattern": "Corrupted training should underperform clean alignment.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Aligned feature guidance, not branch size alone, explains gains.",
    },
    {
        "source_phase": "P13",
        "setting_group": "semantic_mismatch",
        "axis_label": "Semantic mismatch",
        "plan_intent": "Introduce role/stage/position mismatch for stronger sanity checks beyond zero/shuffle.",
        "expected_pattern": "Meaningful drop or instability under semantic mismatch.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Semantic and temporal alignment of features is crucial for routing quality.",
    },
    {
        "source_phase": "P13",
        "setting_group": "eval_perturb_extra",
        "axis_label": "Eval perturb extra",
        "plan_intent": "Supplement perturbation controls with stronger family-focused stress variants.",
        "expected_pattern": "Should behave as negative controls with visible degradation.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Additional perturb controls reinforce that routing depends on aligned feature signals.",
    },
    {
        "source_phase": "P13",
        "setting_group": "train_shift_extra",
        "axis_label": "Train shift extra",
        "plan_intent": "Shift-based corruption for graded alignment stress at train time.",
        "expected_pattern": "Shift stress should reduce performance versus clean anchor.",
        "expectation_type": "control_expected_drop",
        "paper_claim_template": "Temporal misalignment during training directly weakens downstream ranking quality.",
    },
]


@dataclass
class ArtifactPaths:
    result_json_path: Optional[Path]
    logging_bundle_dir: Optional[Path]
    result_payload: Optional[Dict[str, Any]]


def parse_bool(text: str) -> bool:
    v = str(text).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool value: {text}")


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
    try:
        return int(v)
    except Exception:
        return default


def parse_timestamp_utc(text: Any) -> float:
    raw = str(text or "").strip()
    if not raw:
        return 0.0
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw).timestamp()
    except Exception:
        return 0.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: List[Dict[str, Any]], field_order: Optional[Sequence[str]] = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        rows = [{"note": "no rows"}]

    keys: List[str]
    if field_order:
        seen = set(field_order)
        extra = sorted({k for row in rows for k in row.keys() if k not in seen})
        keys = list(field_order) + extra
    else:
        keys = sorted({k for row in rows for k in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dedup_by_run_phase(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    score: Dict[str, Tuple[float, int, float, str]] = {}

    for row in rows:
        run_phase = str(row.get("run_phase", "") or "").strip()
        if not run_phase:
            continue

        ts = parse_timestamp_utc(row.get("timestamp_utc", ""))
        n_completed = to_int(row.get("n_completed", 0), default=0)
        best_valid = to_float(row.get("run_best_valid_mrr20"))
        best_valid_score = best_valid if best_valid is not None else -1e12
        tie_tail = str(row.get("run_id", "") or "")
        sc = (ts, n_completed, best_valid_score, tie_tail)

        prev = score.get(run_phase)
        if prev is None or sc > prev:
            score[run_phase] = sc
            latest[run_phase] = dict(row)

    return [latest[k] for k in sorted(latest.keys())]


def normalize_hparam_id(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    s_up = s.upper()
    if s_up.startswith("H"):
        return s_up
    if s.isdigit():
        return f"H{int(s)}"
    return s


def parse_setting_idx(setting_key: str, setting_id: str, setting_idx_raw: Any) -> int:
    if str(setting_idx_raw or "").strip() != "":
        return to_int(setting_idx_raw, default=999)

    sid = str(setting_id or "").strip()
    if sid.isdigit():
        return int(sid)

    key = str(setting_key or "").strip() or sid
    m = re.search(r"P\d{2}-(\d+)", key)
    if m:
        return int(m.group(1))

    m2 = re.search(r"_(\d{2})_", sid)
    if m2:
        return int(m2.group(1))

    return 999


def canonical_setting_key(source_phase: str, row: Dict[str, Any]) -> str:
    key = str(row.get("setting_key", "") or "").strip()
    sid = str(row.get("setting_id", "") or "").strip()

    if key.startswith(f"{source_phase}-"):
        return key
    if sid.startswith(f"{source_phase}-"):
        return sid

    if source_phase == "P10":
        # P10 wide summary stores canonical key in setting_id, short key in setting_key.
        if sid:
            return sid
        if key:
            idx = parse_setting_idx(key, sid, row.get("setting_idx"))
            return f"P10-{idx:02d}_{key}"

    if key:
        return key
    return sid


def canonical_setting_uid(source_phase: str, row: Dict[str, Any], setting_key: str) -> str:
    uid = str(row.get("setting_uid", "") or "").strip()
    if uid:
        return uid

    sid = str(row.get("setting_id", "") or "").strip()
    if sid and sid.startswith(f"{source_phase}_"):
        return sid

    idx = parse_setting_idx(setting_key, sid, row.get("setting_idx"))
    if idx < 999:
        return f"{source_phase}_{idx:02d}"

    return f"{source_phase}_UNK"


def canonical_setting_short(row: Dict[str, Any], setting_key: str) -> str:
    short = str(row.get("setting_short", "") or "").strip()
    if short:
        return short

    m = re.search(r"P\d{2}-\d+_(.+)", setting_key)
    if m:
        return m.group(1)

    key_raw = str(row.get("setting_key", "") or "").strip()
    if key_raw:
        return key_raw

    return setting_key


def resolve_path_with_repo_root(path_text: str, repo_root: Path) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return repo_root / p


def load_result_artifacts(result_path_text: str, repo_root: Path) -> ArtifactPaths:
    text = str(result_path_text or "").strip()
    if not text:
        return ArtifactPaths(result_json_path=None, logging_bundle_dir=None, result_payload=None)

    payload: Optional[Dict[str, Any]] = None
    result_json_path: Optional[Path] = None

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
    else:
        candidate = resolve_path_with_repo_root(text, repo_root)
        result_json_path = candidate
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                payload = None

    logging_bundle_dir: Optional[Path] = None
    if payload and isinstance(payload, dict):
        b = str(payload.get("logging_bundle_dir", "") or "").strip()
        if b:
            bp = resolve_path_with_repo_root(b, repo_root)
            logging_bundle_dir = bp

    return ArtifactPaths(
        result_json_path=result_json_path,
        logging_bundle_dir=logging_bundle_dir,
        result_payload=payload,
    )


def extract_special_metrics(bundle_dir: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "special_available": 0,
        "special_path": "",
        "special_test_overall_mrr20": None,
        "special_test_overall_hit20": None,
        "cold_item_mrr20": None,
        "cold_item_count": None,
        "long_session_mrr20": None,
        "long_session_count": None,
        "sess_1_2_mrr20": None,
        "sess_3_5_mrr20": None,
        "sess_6_10_mrr20": None,
    }

    if bundle_dir is None:
        return out

    path = bundle_dir / "special_metrics.json"
    out["special_path"] = str(path)
    if not path.exists():
        return out

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out

    test = obj.get("test_special_metrics", {}) or {}
    overall = test.get("overall", {}) or {}
    slices = test.get("slices", {}) or {}
    counts = test.get("counts", {}) or {}

    target_pop = slices.get("target_popularity_abs", {}) or {}
    sess = slices.get("session_len", {}) or {}

    out.update(
        {
            "special_available": 1,
            "special_test_overall_mrr20": to_float(overall.get("mrr@20")),
            "special_test_overall_hit20": to_float(overall.get("hit@20")),
            "cold_item_mrr20": to_float(((target_pop.get("<=5") or {}).get("mrr@20"))),
            "cold_item_count": to_int(((counts.get("target_popularity_abs", {}) or {}).get("<=5")), default=0)
            if ((counts.get("target_popularity_abs", {}) or {}).get("<=5")) is not None
            else None,
            "long_session_mrr20": to_float(((sess.get("11+") or {}).get("mrr@20"))),
            "long_session_count": to_int(((counts.get("session_len", {}) or {}).get("11+")), default=0)
            if ((counts.get("session_len", {}) or {}).get("11+")) is not None
            else None,
            "sess_1_2_mrr20": to_float(((sess.get("1-2") or {}).get("mrr@20"))),
            "sess_3_5_mrr20": to_float(((sess.get("3-5") or {}).get("mrr@20"))),
            "sess_6_10_mrr20": to_float(((sess.get("6-10") or {}).get("mrr@20"))),
        }
    )
    return out


def extract_diag_overview(bundle_dir: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "diag_available": 0,
        "diag_overview_path": "",
        "diag_source_row_count": 0,
        "diag_stage": "",
        "diag_n_eff": None,
        "diag_cv_usage": None,
        "diag_top1_max_frac": None,
        "diag_entropy_mean": None,
        "diag_route_jitter_adjacent": None,
        "diag_route_consistency_knn_score": None,
        "diag_route_consistency_knn_js": None,
        "diag_route_consistency_group_knn_score": None,
        "diag_route_consistency_group_knn_js": None,
        "diag_route_consistency_feature_group_knn_mean_score": None,
        "diag_route_consistency_feature_group_knn_mean_js": None,
        "diag_family_top_expert_mean_share": None,
    }

    if bundle_dir is None:
        return out

    path = bundle_dir / "diag" / "raw" / "overview_table.csv"
    out["diag_overview_path"] = str(path)
    if not path.exists():
        return out

    rows = read_csv_rows(path)
    if not rows:
        return out

    row = rows[0]
    out["diag_available"] = 1
    out["diag_source_row_count"] = len(rows)

    for k, v in row.items():
        key = f"diag_{k}"
        if k == "stage":
            out[key] = str(v or "")
            continue
        fv = to_float(v)
        out[key] = fv if fv is not None else v

    for k in list(out.keys()):
        out.setdefault(k, out[k])

    return out


def load_best_valid_diag(bundle_dir: Optional[Path]) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if bundle_dir is None:
        return None, None
    path = bundle_dir / "diag" / "raw" / "best_valid_diag.json"
    if not path.exists():
        return path, None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path, None
    if not isinstance(obj, dict):
        return path, None
    return path, obj


def stage_sort_key(stage_name: str) -> Tuple[int, str]:
    m = re.search(r"@(\d+)", str(stage_name))
    if m:
        return (int(m.group(1)), str(stage_name))
    return (10**6, str(stage_name))


def safe_float_list(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    out: List[float] = []
    for x in values:
        v = to_float(x)
        out.append(v if v is not None else 0.0)
    return out


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var))


def usage_entropy(shares: List[float]) -> Optional[float]:
    if not shares:
        return None
    total = sum(shares)
    if total <= 0:
        return 0.0
    p = [max(x, 0.0) / total for x in shares]
    eps = 1e-12
    return float(-sum(pi * math.log(pi + eps) for pi in p if pi > 0))


def gini(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(max(v, 0.0) for v in values)
    n = len(xs)
    total = sum(xs)
    if total <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return float((2.0 * cum) / (n * total) - (n + 1) / n)


def enrich_row(
    raw: Dict[str, Any],
    split: str,
    source_phase: str,
    source_axis: str,
    repo_root: Path,
    wide_min_completed: int,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Path], Optional[Path]]:
    setting_key = canonical_setting_key(source_phase, raw)
    setting_uid = canonical_setting_uid(source_phase, raw, setting_key)
    setting_short = canonical_setting_short(raw, setting_key)
    setting_id = str(raw.get("setting_id", "") or "").strip()
    setting_group = str(raw.get("setting_group", "") or "").strip()
    setting_desc = str(raw.get("setting_desc", "") or "").strip()
    setting_detail = str(raw.get("setting_detail", "") or "").strip()
    if not setting_desc:
        setting_desc = str(raw.get("exp_brief", "") or "").strip()

    setting_idx = parse_setting_idx(setting_key, setting_id, raw.get("setting_idx"))

    n_completed = to_int(raw.get("n_completed", 0), default=0)
    best_valid = to_float(raw.get("run_best_valid_mrr20"))
    test_mrr = to_float(raw.get("test_mrr20"))

    result_path_raw = str(raw.get("result_path", "") or "").strip()
    artifacts = load_result_artifacts(result_path_raw, repo_root)

    special = extract_special_metrics(artifacts.logging_bundle_dir)
    diag = extract_diag_overview(artifacts.logging_bundle_dir)
    best_diag_path, best_diag_obj = load_best_valid_diag(artifacts.logging_bundle_dir)

    phase_title = PHASE_TITLE.get(source_phase, "")

    status = str(raw.get("status", "") or "").strip().lower()
    if not status:
        if n_completed >= wide_min_completed:
            status = "completed"
        elif n_completed > 0:
            status = "partial"
        else:
            status = "pending"

    out: Dict[str, Any] = {
        "split": split,
        "source_phase": source_phase,
        "source_axis": source_axis,
        "phase_title": phase_title,
        "dataset": str(raw.get("dataset", "") or "").strip(),
        "run_phase": str(raw.get("run_phase", "") or "").strip(),
        "run_id": str(raw.get("run_id", "") or "").strip(),
        "timestamp_utc": str(raw.get("timestamp_utc", "") or "").strip(),
        "stage": str(raw.get("stage", "") or "").strip(),
        "trigger": str(raw.get("trigger", "") or "").strip(),
        "status": status,
        "setting_uid": setting_uid,
        "setting_id": setting_id,
        "setting_idx": setting_idx,
        "setting_key": setting_key,
        "setting_short": setting_short,
        "setting_group": setting_group,
        "setting_desc": setting_desc,
        "setting_detail": setting_detail,
        "best_valid_mrr20": best_valid,
        "test_mrr20": test_mrr,
        "n_completed": n_completed,
        "seed_id": to_int(raw.get("seed_id"), default=0) if str(raw.get("seed_id", "")).strip() else None,
        "gpu_id": str(raw.get("gpu_id", "") or "").strip(),
        "hparam_id": normalize_hparam_id(raw.get("hparam_id")),
        "global_best_valid_mrr20": to_float(raw.get("global_best_valid_mrr20")),
        "special_ok_flag": str(raw.get("special_ok", "") or "").strip(),
        "diag_ok_flag": str(raw.get("diag_ok", "") or "").strip(),
        "interrupted": str(raw.get("interrupted", "") or "").strip(),
        "result_path_raw": result_path_raw,
        "result_json_path": str(artifacts.result_json_path) if artifacts.result_json_path else "",
        "result_payload_ok": 1 if artifacts.result_payload else 0,
        "logging_bundle_dir": str(artifacts.logging_bundle_dir) if artifacts.logging_bundle_dir else "",
        "best_valid_diag_json_path": str(best_diag_path) if best_diag_path else "",
        "is_main_wide": 1 if n_completed >= wide_min_completed else 0,
        "anchor_setting_key": PHASE_ANCHOR_SETTING.get(source_phase, ""),
    }

    out.update(special)
    out.update(diag)

    return out, best_diag_obj, artifacts.logging_bundle_dir, best_diag_path


def build_router_rows(row: Dict[str, Any], best_diag_obj: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    family_rows: List[Dict[str, Any]] = []
    position_rows: List[Dict[str, Any]] = []
    scalar_rows: List[Dict[str, Any]] = []

    if not best_diag_obj:
        return family_rows, position_rows, scalar_rows

    stage_metrics = best_diag_obj.get("stage_metrics", {})
    if not isinstance(stage_metrics, dict) or not stage_metrics:
        return family_rows, position_rows, scalar_rows

    base_meta = {
        "split": row.get("split", ""),
        "source_phase": row.get("source_phase", ""),
        "source_axis": row.get("source_axis", ""),
        "dataset": row.get("dataset", ""),
        "run_phase": row.get("run_phase", ""),
        "setting_uid": row.get("setting_uid", ""),
        "setting_key": row.get("setting_key", ""),
        "setting_short": row.get("setting_short", ""),
        "setting_group": row.get("setting_group", ""),
        "hparam_id": row.get("hparam_id", ""),
        "seed_id": row.get("seed_id", None),
        "best_valid_mrr20": row.get("best_valid_mrr20", None),
        "test_mrr20": row.get("test_mrr20", None),
    }

    scalar_keys = [
        "n_slots",
        "entropy_mean",
        "route_jitter_adjacent",
        "route_jitter_session",
        "route_consistency_knn_js",
        "route_consistency_knn_score",
        "route_consistency_group_knn_js",
        "route_consistency_group_knn_score",
        "condition_norm",
        "stage_delta_norm",
        "shared_delta_norm",
        "moe_delta_norm",
        "residual_delta_norm",
        "alpha_value",
        "alpha_effective",
        "n_eff",
        "cv_usage",
        "dead_expert_frac",
        "top1_max_frac",
        "expert_similarity_mean",
        "expert_similarity_max",
    ]

    for stage_name in sorted(stage_metrics.keys(), key=stage_sort_key):
        st = stage_metrics.get(stage_name)
        if not isinstance(st, dict):
            continue

        expert_names = [str(x) for x in (st.get("expert_names") or [])] if isinstance(st.get("expert_names"), list) else []
        usage_share = safe_float_list(st.get("usage_share"))
        if expert_names and len(usage_share) < len(expert_names):
            usage_share = usage_share + [0.0] * (len(expert_names) - len(usage_share))

        scalar_row: Dict[str, Any] = dict(base_meta)
        scalar_row.update(
            {
                "stage_name": str(stage_name),
                "mode": str(st.get("mode", "") or ""),
                "expert_count": len(expert_names),
                "family_count": len(st.get("family_names") or []) if isinstance(st.get("family_names"), list) else None,
                "usage_share_max": max(usage_share) if usage_share else None,
                "usage_share_min": min(usage_share) if usage_share else None,
                "usage_share_mean": mean(usage_share),
                "usage_share_std": std(usage_share),
                "usage_share_entropy": usage_entropy(usage_share),
                "usage_share_gini": gini(usage_share),
            }
        )
        for k in scalar_keys:
            scalar_row[k] = to_float(st.get(k))

        knn = st.get("route_consistency_feature_group_knn", {})
        if isinstance(knn, dict):
            group_names = [str(x) for x in (knn.get("group_names") or [])] if isinstance(knn.get("group_names"), list) else []
            score_by_group = safe_float_list(knn.get("score_by_group"))
            js_by_group = safe_float_list(knn.get("js_by_group"))
            scalar_row["route_consistency_feature_group_knn_mean_score"] = to_float(knn.get("mean_score"))
            scalar_row["route_consistency_feature_group_knn_mean_js"] = to_float(knn.get("mean_js"))
            for i, gname in enumerate(group_names):
                g = re.sub(r"[^a-zA-Z0-9_]+", "_", gname.strip().lower()) or f"group{i}"
                scalar_row[f"route_consistency_feature_group_knn_{g}_score"] = (
                    score_by_group[i] if i < len(score_by_group) else None
                )
                scalar_row[f"route_consistency_feature_group_knn_{g}_js"] = js_by_group[i] if i < len(js_by_group) else None
        scalar_rows.append(scalar_row)

        # Family-expert heatmap long table
        heat = st.get("feature_family_expert_heatmap", {})
        if isinstance(heat, dict):
            family_names = heat.get("family_names")
            values = heat.get("values")
            if not isinstance(family_names, list):
                family_names = st.get("family_names") if isinstance(st.get("family_names"), list) else []
            if isinstance(values, list):
                for fi, fam in enumerate(family_names):
                    vec_raw = values[fi] if fi < len(values) and isinstance(values[fi], list) else []
                    vec = safe_float_list(vec_raw)
                    if expert_names and len(vec) < len(expert_names):
                        vec = vec + [0.0] * (len(expert_names) - len(vec))
                    row_sum = sum(vec)
                    for ei, expert in enumerate(expert_names):
                        val = vec[ei] if ei < len(vec) else 0.0
                        family_rows.append(
                            {
                                **base_meta,
                                "stage_name": str(stage_name),
                                "family": str(fam),
                                "expert": expert,
                                "family_expert_value": val,
                                "family_expert_share_norm": (val / row_sum) if row_sum > 0 else 0.0,
                                "expert_global_share": usage_share[ei] if ei < len(usage_share) else None,
                            }
                        )

        # Position-expert usage long table
        pos = st.get("position_expert_usage", {})
        if isinstance(pos, dict):
            values = pos.get("values")
            if isinstance(values, list):
                for pi, vec_raw in enumerate(values):
                    vec = safe_float_list(vec_raw if isinstance(vec_raw, list) else [])
                    if expert_names and len(vec) < len(expert_names):
                        vec = vec + [0.0] * (len(expert_names) - len(vec))
                    row_sum = sum(vec)
                    for ei, expert in enumerate(expert_names):
                        val = vec[ei] if ei < len(vec) else 0.0
                        position_rows.append(
                            {
                                **base_meta,
                                "stage_name": str(stage_name),
                                "position_index": pi + 1,
                                "expert": expert,
                                "position_expert_value": val,
                                "position_expert_share_norm": (val / row_sum) if row_sum > 0 else 0.0,
                                "expert_global_share": usage_share[ei] if ei < len(usage_share) else None,
                            }
                        )

    return family_rows, position_rows, scalar_rows


def sort_rows_common(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def hparam_rank(h: str) -> int:
        m = re.search(r"(\d+)", str(h or ""))
        if m:
            return int(m.group(1))
        return 999

    return sorted(
        rows,
        key=lambda r: (
            PHASE_RANK.get(str(r.get("source_phase", "")), 999),
            to_int(r.get("setting_idx"), default=999),
            str(r.get("setting_uid", "")),
            hparam_rank(str(r.get("hparam_id", ""))),
            to_int(r.get("seed_id"), default=0),
            str(r.get("run_phase", "")),
        ),
    )


def find_phase_anchor(rows: List[Dict[str, Any]], source_phase: str, wide_min_completed: int) -> Optional[Dict[str, Any]]:
    anchor_key = PHASE_ANCHOR_SETTING.get(source_phase, "")
    if not anchor_key:
        return None

    candidates = [
        r
        for r in rows
        if str(r.get("source_phase", "")) == source_phase and str(r.get("setting_key", "")) == anchor_key
    ]
    if not candidates:
        return None

    # Prefer main-eligible row if present.
    main_candidates = [r for r in candidates if to_int(r.get("n_completed"), 0) >= wide_min_completed]
    target = main_candidates if main_candidates else candidates
    target = sorted(
        target,
        key=lambda r: (
            to_int(r.get("n_completed"), 0),
            to_float(r.get("best_valid_mrr20")) or -1e12,
            str(r.get("timestamp_utc", "")),
        ),
        reverse=True,
    )
    return target[0] if target else None


def summarize_wide_phase(rows: List[Dict[str, Any]], wide_min_completed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for phase in PHASE_ORDER:
        phase_rows = [r for r in rows if str(r.get("source_phase", "")) == phase]
        if not phase_rows:
            continue

        main_rows = [r for r in phase_rows if to_int(r.get("n_completed"), 0) >= wide_min_completed]
        target_rows = main_rows if main_rows else phase_rows

        best_row = sorted(
            target_rows,
            key=lambda r: (to_float(r.get("best_valid_mrr20")) or -1e12, to_float(r.get("test_mrr20")) or -1e12),
            reverse=True,
        )[0]

        anchor = find_phase_anchor(phase_rows, phase, wide_min_completed)
        anchor_valid = to_float(anchor.get("best_valid_mrr20")) if anchor else None
        anchor_test = to_float(anchor.get("test_mrr20")) if anchor else None

        diag_missing = [r for r in phase_rows if to_int(r.get("diag_available"), 0) == 0]
        special_missing = [r for r in phase_rows if to_int(r.get("special_available"), 0) == 0]

        best_valid = to_float(best_row.get("best_valid_mrr20"))
        best_test = to_float(best_row.get("test_mrr20"))

        out.append(
            {
                "source_phase": phase,
                "phase_title": PHASE_TITLE.get(phase, ""),
                "source_axis": PHASE_AXIS_NAME.get(phase, ""),
                "n_rows_dedup": len(phase_rows),
                "n_rows_main": len(main_rows),
                "main_min_completed": wide_min_completed,
                "anchor_setting_key": PHASE_ANCHOR_SETTING.get(phase, ""),
                "anchor_best_valid_mrr20": anchor_valid,
                "anchor_test_mrr20": anchor_test,
                "best_run_phase": best_row.get("run_phase", ""),
                "best_setting_key": best_row.get("setting_key", ""),
                "best_valid_mrr20": best_valid,
                "best_test_mrr20": best_test,
                "best_valid_minus_anchor": (best_valid - anchor_valid) if (best_valid is not None and anchor_valid is not None) else None,
                "best_test_minus_anchor": (best_test - anchor_test) if (best_test is not None and anchor_test is not None) else None,
                "mean_valid_main": mean(to_float(r.get("best_valid_mrr20")) for r in main_rows),
                "std_valid_main": std(to_float(r.get("best_valid_mrr20")) for r in main_rows),
                "mean_test_main": mean(to_float(r.get("test_mrr20")) for r in main_rows),
                "std_test_main": std(to_float(r.get("test_mrr20")) for r in main_rows),
                "mean_cold_main": mean(to_float(r.get("cold_item_mrr20")) for r in main_rows),
                "mean_long_session_main": mean(to_float(r.get("long_session_mrr20")) for r in main_rows),
                "mean_diag_top1_main": mean(to_float(r.get("diag_top1_max_frac")) for r in main_rows),
                "mean_diag_cv_main": mean(to_float(r.get("diag_cv_usage")) for r in main_rows),
                "mean_diag_n_eff_main": mean(to_float(r.get("diag_n_eff")) for r in main_rows),
                "special_missing_count": len(special_missing),
                "diag_missing_count": len(diag_missing),
            }
        )

    return out


def summarize_wide_axis(rows: List[Dict[str, Any]], wide_min_completed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r.get("source_phase", "")), str(r.get("setting_group", "")))].append(r)

    for (phase, group), g_rows in sorted(grouped.items(), key=lambda kv: (PHASE_RANK.get(kv[0][0], 999), kv[0][1])):
        main_rows = [r for r in g_rows if to_int(r.get("n_completed"), 0) >= wide_min_completed]
        target_rows = main_rows if main_rows else g_rows
        best_row = sorted(
            target_rows,
            key=lambda r: (to_float(r.get("best_valid_mrr20")) or -1e12, to_float(r.get("test_mrr20")) or -1e12),
            reverse=True,
        )[0]

        anchor = find_phase_anchor(rows, phase, wide_min_completed)
        anchor_valid = to_float(anchor.get("best_valid_mrr20")) if anchor else None
        anchor_test = to_float(anchor.get("test_mrr20")) if anchor else None
        anchor_top1 = to_float(anchor.get("diag_top1_max_frac")) if anchor else None

        best_valid = to_float(best_row.get("best_valid_mrr20"))
        best_test = to_float(best_row.get("test_mrr20"))
        mean_top1 = mean(to_float(r.get("diag_top1_max_frac")) for r in main_rows)
        mean_cv = mean(to_float(r.get("diag_cv_usage")) for r in main_rows)

        delta_valid = (best_valid - anchor_valid) if (best_valid is not None and anchor_valid is not None) else None
        delta_test = (best_test - anchor_test) if (best_test is not None and anchor_test is not None) else None

        concentration_flag = ""
        if mean_top1 is not None and mean_top1 >= 0.80:
            concentration_flag = "high_top1"
        elif mean_cv is not None and mean_cv >= 1.00:
            concentration_flag = "high_cv"
        else:
            concentration_flag = "stable_or_mixed"

        out.append(
            {
                "source_phase": phase,
                "phase_title": PHASE_TITLE.get(phase, ""),
                "setting_group": group,
                "n_rows_dedup": len(g_rows),
                "n_rows_main": len(main_rows),
                "main_min_completed": wide_min_completed,
                "best_run_phase": best_row.get("run_phase", ""),
                "best_setting_key": best_row.get("setting_key", ""),
                "best_valid_mrr20": best_valid,
                "best_test_mrr20": best_test,
                "mean_valid_main": mean(to_float(r.get("best_valid_mrr20")) for r in main_rows),
                "std_valid_main": std(to_float(r.get("best_valid_mrr20")) for r in main_rows),
                "mean_test_main": mean(to_float(r.get("test_mrr20")) for r in main_rows),
                "std_test_main": std(to_float(r.get("test_mrr20")) for r in main_rows),
                "mean_cold_main": mean(to_float(r.get("cold_item_mrr20")) for r in main_rows),
                "mean_long_session_main": mean(to_float(r.get("long_session_mrr20")) for r in main_rows),
                "mean_diag_top1_main": mean_top1,
                "mean_diag_cv_main": mean_cv,
                "mean_diag_n_eff_main": mean(to_float(r.get("diag_n_eff")) for r in main_rows),
                "diag_high_concentration_frac_main": mean(
                    1.0 if (to_float(r.get("diag_top1_max_frac")) is not None and to_float(r.get("diag_top1_max_frac")) >= 0.80) else 0.0
                    for r in main_rows
                ),
                "anchor_setting_key": PHASE_ANCHOR_SETTING.get(phase, ""),
                "anchor_best_valid_mrr20": anchor_valid,
                "anchor_test_mrr20": anchor_test,
                "anchor_diag_top1_max_frac": anchor_top1,
                "best_valid_minus_anchor": delta_valid,
                "best_test_minus_anchor": delta_test,
                "concentration_tag": concentration_flag,
                "special_missing_count": sum(1 for r in g_rows if to_int(r.get("special_available"), 0) == 0),
                "diag_missing_count": sum(1 for r in g_rows if to_int(r.get("diag_available"), 0) == 0),
            }
        )

    return out


def summarize_verification_seed_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[
            (
                str(r.get("source_phase", "")),
                str(r.get("setting_uid", "")),
                str(r.get("setting_key", "")),
                str(r.get("setting_group", "")),
                str(r.get("hparam_id", "")),
            )
        ].append(r)

    out: List[Dict[str, Any]] = []
    for key, g_rows in sorted(grouped.items(), key=lambda kv: (PHASE_RANK.get(kv[0][0], 999), kv[0][1], kv[0][4])):
        source_phase, setting_uid, setting_key, setting_group, hparam_id = key
        seeds = sorted({to_int(r.get("seed_id"), default=0) for r in g_rows if r.get("seed_id") is not None})
        out.append(
            {
                "source_phase": source_phase,
                "phase_title": PHASE_TITLE.get(source_phase, ""),
                "setting_uid": setting_uid,
                "setting_key": setting_key,
                "setting_group": setting_group,
                "hparam_id": hparam_id,
                "n_seed": len(g_rows),
                "seed_ids": ",".join(str(s) for s in seeds if s > 0),
                "n_completed_min": min(to_int(r.get("n_completed"), 0) for r in g_rows) if g_rows else None,
                "n_completed_max": max(to_int(r.get("n_completed"), 0) for r in g_rows) if g_rows else None,
                "best_valid_mean": mean(to_float(r.get("best_valid_mrr20")) for r in g_rows),
                "best_valid_std": std(to_float(r.get("best_valid_mrr20")) for r in g_rows),
                "best_valid_min": min(
                    (to_float(r.get("best_valid_mrr20")) for r in g_rows if to_float(r.get("best_valid_mrr20")) is not None),
                    default=None,
                ),
                "best_valid_max": max(
                    (to_float(r.get("best_valid_mrr20")) for r in g_rows if to_float(r.get("best_valid_mrr20")) is not None),
                    default=None,
                ),
                "test_mean": mean(to_float(r.get("test_mrr20")) for r in g_rows),
                "test_std": std(to_float(r.get("test_mrr20")) for r in g_rows),
                "cold_mean": mean(to_float(r.get("cold_item_mrr20")) for r in g_rows),
                "cold_std": std(to_float(r.get("cold_item_mrr20")) for r in g_rows),
                "long_session_mean": mean(to_float(r.get("long_session_mrr20")) for r in g_rows),
                "long_session_std": std(to_float(r.get("long_session_mrr20")) for r in g_rows),
                "diag_top1_mean": mean(to_float(r.get("diag_top1_max_frac")) for r in g_rows),
                "diag_top1_std": std(to_float(r.get("diag_top1_max_frac")) for r in g_rows),
                "diag_cv_mean": mean(to_float(r.get("diag_cv_usage")) for r in g_rows),
                "diag_cv_std": std(to_float(r.get("diag_cv_usage")) for r in g_rows),
                "diag_n_eff_mean": mean(to_float(r.get("diag_n_eff")) for r in g_rows),
                "diag_n_eff_std": std(to_float(r.get("diag_n_eff")) for r in g_rows),
                "diag_available_frac": mean(float(to_int(r.get("diag_available"), 0)) for r in g_rows),
                "special_available_frac": mean(float(to_int(r.get("special_available"), 0)) for r in g_rows),
            }
        )

    return out


def judge_expectation(expectation_type: str, delta_best_valid: Optional[float], mean_top1: Optional[float], mean_cv: Optional[float]) -> Tuple[str, int]:
    if delta_best_valid is None:
        return "no_anchor", 0

    if expectation_type == "control_expected_drop":
        if delta_best_valid < -0.0010:
            return "matched_control_drop", 1
        if delta_best_valid < 0.0:
            return "weak_control_drop", 1
        return "unexpected_non_drop", 0

    if expectation_type == "target_near_or_better":
        if delta_best_valid >= 0.0:
            return "matched_gain", 1
        if delta_best_valid >= -0.0015:
            return "near_anchor", 1
        return "below_expected", 0

    if expectation_type == "tradeoff_or_stability":
        stable = True
        if mean_top1 is not None and mean_top1 >= 0.90:
            stable = False
        if mean_cv is not None and mean_cv >= 1.5:
            stable = False

        if delta_best_valid >= -0.0015 and stable:
            return "matched_tradeoff", 1
        if delta_best_valid >= -0.0025:
            return "partial_tradeoff", 1
        return "underperformed_tradeoff", 0

    return "unknown_expectation", 0


def summarize_intent_vs_observed(
    wide_rows: List[Dict[str, Any]],
    wide_min_completed: int,
    intent_map_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for im in intent_map_rows:
        phase = str(im.get("source_phase", ""))
        group = str(im.get("setting_group", ""))
        group_rows = [
            r
            for r in wide_rows
            if str(r.get("source_phase", "")) == phase and str(r.get("setting_group", "")) == group
        ]
        main_rows = [r for r in group_rows if to_int(r.get("n_completed"), 0) >= wide_min_completed]
        target_rows = main_rows if main_rows else group_rows

        anchor = find_phase_anchor(wide_rows, phase, wide_min_completed)
        anchor_valid = to_float(anchor.get("best_valid_mrr20")) if anchor else None
        anchor_test = to_float(anchor.get("test_mrr20")) if anchor else None

        if target_rows:
            best_row = sorted(
                target_rows,
                key=lambda r: (to_float(r.get("best_valid_mrr20")) or -1e12, to_float(r.get("test_mrr20")) or -1e12),
                reverse=True,
            )[0]
            best_valid = to_float(best_row.get("best_valid_mrr20"))
            best_test = to_float(best_row.get("test_mrr20"))
            best_run_phase = str(best_row.get("run_phase", ""))
            best_setting_key = str(best_row.get("setting_key", ""))
        else:
            best_row = None
            best_valid = None
            best_test = None
            best_run_phase = ""
            best_setting_key = ""

        delta_best_valid = (best_valid - anchor_valid) if (best_valid is not None and anchor_valid is not None) else None
        delta_best_test = (best_test - anchor_test) if (best_test is not None and anchor_test is not None) else None

        mean_top1 = mean(to_float(r.get("diag_top1_max_frac")) for r in main_rows)
        mean_cv = mean(to_float(r.get("diag_cv_usage")) for r in main_rows)

        observed_tag, match_flag = judge_expectation(str(im.get("expectation_type", "")), delta_best_valid, mean_top1, mean_cv)

        out.append(
            {
                "source_phase": phase,
                "phase_title": PHASE_TITLE.get(phase, ""),
                "setting_group": group,
                "axis_label": im.get("axis_label", ""),
                "plan_intent": im.get("plan_intent", ""),
                "expected_pattern": im.get("expected_pattern", ""),
                "expectation_type": im.get("expectation_type", ""),
                "paper_claim_template": im.get("paper_claim_template", ""),
                "n_rows_dedup": len(group_rows),
                "n_rows_main": len(main_rows),
                "best_run_phase": best_run_phase,
                "best_setting_key": best_setting_key,
                "best_valid_mrr20": best_valid,
                "best_test_mrr20": best_test,
                "mean_valid_main": mean(to_float(r.get("best_valid_mrr20")) for r in main_rows),
                "mean_test_main": mean(to_float(r.get("test_mrr20")) for r in main_rows),
                "mean_cold_main": mean(to_float(r.get("cold_item_mrr20")) for r in main_rows),
                "mean_long_session_main": mean(to_float(r.get("long_session_mrr20")) for r in main_rows),
                "mean_diag_top1_main": mean_top1,
                "mean_diag_cv_main": mean_cv,
                "mean_diag_n_eff_main": mean(to_float(r.get("diag_n_eff")) for r in main_rows),
                "anchor_setting_key": PHASE_ANCHOR_SETTING.get(phase, ""),
                "anchor_best_valid_mrr20": anchor_valid,
                "anchor_test_mrr20": anchor_test,
                "delta_best_valid_vs_anchor": delta_best_valid,
                "delta_best_test_vs_anchor": delta_best_test,
                "observed_tag": observed_tag,
                "match_flag": match_flag,
                "diag_missing_count": sum(1 for r in group_rows if to_int(r.get("diag_available"), 0) == 0),
                "special_missing_count": sum(1 for r in group_rows if to_int(r.get("special_available"), 0) == 0),
            }
        )

    return out


def build_data_tables(
    repo_root: Path,
    dataset: str,
    out_dir: Path,
    wide_min_completed: int,
    verification_main_hparam: str,
    verification_main_min_completed: int,
    include_support_p10: bool,
) -> int:
    wide_rows: List[Dict[str, Any]] = []
    verification_rows: List[Dict[str, Any]] = []

    router_family_rows: List[Dict[str, Any]] = []
    router_position_rows: List[Dict[str, Any]] = []
    router_scalar_rows: List[Dict[str, Any]] = []

    diag_missing_wide: List[str] = []
    diag_missing_verif: List[str] = []
    diag_json_missing_verif: List[str] = []

    # -------- wide --------
    for phase in PHASE_ORDER:
        summary_rel = PHASE_WIDE_SUMMARY_REL[phase].format(dataset=dataset)
        summary_path = repo_root / summary_rel
        rows_raw = read_csv_rows(summary_path)
        dedup_rows = dedup_by_run_phase(rows_raw)

        for raw in dedup_rows:
            row, _best_diag_obj, _bundle_dir, _best_diag_path = enrich_row(
                raw=raw,
                split="wide",
                source_phase=phase,
                source_axis=PHASE_AXIS_NAME[phase],
                repo_root=repo_root,
                wide_min_completed=wide_min_completed,
            )
            if to_int(row.get("diag_available"), 0) == 0:
                diag_missing_wide.append(str(row.get("run_phase", "")))
            wide_rows.append(row)

    wide_rows = sort_rows_common(wide_rows)

    # -------- verification --------
    verification_summary_path = repo_root / VERIFICATION_SUMMARY_REL.format(dataset=dataset)
    verification_raw = read_csv_rows(verification_summary_path)
    verification_dedup = dedup_by_run_phase(verification_raw)

    for raw in verification_dedup:
        source_phase = str(raw.get("source_phase", "") or "").strip()
        source_axis = str(raw.get("source_axis", "") or "").strip()

        row, best_diag_obj, _bundle_dir, best_diag_path = enrich_row(
            raw=raw,
            split="verification",
            source_phase=source_phase,
            source_axis=source_axis,
            repo_root=repo_root,
            wide_min_completed=wide_min_completed,
        )
        verification_rows.append(row)

        if to_int(row.get("diag_available"), 0) == 0:
            diag_missing_verif.append(str(row.get("run_phase", "")))

        if best_diag_obj is None:
            # overview may exist while raw diag json is missing; keep separate report.
            if best_diag_path and best_diag_path.exists() is False:
                diag_json_missing_verif.append(str(row.get("run_phase", "")))
            continue

        fam_rows, pos_rows, stg_rows = build_router_rows(row, best_diag_obj)
        router_family_rows.extend(fam_rows)
        router_position_rows.extend(pos_rows)
        router_scalar_rows.extend(stg_rows)

    verification_rows = sort_rows_common(verification_rows)
    router_family_rows = sort_rows_common(router_family_rows)
    router_position_rows = sort_rows_common(router_position_rows)
    router_scalar_rows = sort_rows_common(router_scalar_rows)

    # -------- derived tables --------
    wide_phase_summary = summarize_wide_phase(wide_rows, wide_min_completed=wide_min_completed)
    wide_axis_summary = summarize_wide_axis(wide_rows, wide_min_completed=wide_min_completed)

    verification_main = [
        r
        for r in verification_rows
        if normalize_hparam_id(r.get("hparam_id")) == normalize_hparam_id(verification_main_hparam)
        and to_int(r.get("n_completed"), 0) >= verification_main_min_completed
    ]

    verification_support = (
        [
            r
            for r in verification_rows
            if str(r.get("source_phase", "")) == "P10"
            and to_int(r.get("n_completed"), 0) == 10
            and normalize_hparam_id(r.get("hparam_id")) in {"H1", "H3"}
        ]
        if include_support_p10
        else []
    )

    verification_setting_seed_stats = summarize_verification_seed_stats(verification_rows)

    intent_map_rows = list(INTENT_EXPECTATION_MAP)
    intent_vs_observed = summarize_intent_vs_observed(
        wide_rows=wide_rows,
        wide_min_completed=wide_min_completed,
        intent_map_rows=intent_map_rows,
    )

    # -------- write --------
    ensure_dir(out_dir)

    write_csv(out_dir / "wide_all_dedup.csv", wide_rows)
    write_csv(out_dir / "wide_phase_summary.csv", wide_phase_summary)
    write_csv(out_dir / "wide_axis_summary.csv", wide_axis_summary)

    write_csv(out_dir / "verification_all_dedup.csv", verification_rows)
    write_csv(out_dir / "verification_main_h3_n20.csv", verification_main)
    write_csv(out_dir / "verification_support_p10_n10.csv", verification_support)
    write_csv(out_dir / "verification_setting_seed_stats.csv", verification_setting_seed_stats)

    write_csv(out_dir / "router_family_expert_long.csv", router_family_rows)
    write_csv(out_dir / "router_position_expert_long.csv", router_position_rows)
    write_csv(out_dir / "router_stage_scalar.csv", router_scalar_rows)

    write_csv(out_dir / "intent_expectation_map.csv", intent_map_rows)
    write_csv(out_dir / "intent_vs_observed_summary.csv", intent_vs_observed)

    # -------- validation prints --------
    def count_by_phase(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for r in rows:
            counts[str(r.get("source_phase", ""))] += 1
        return {k: counts[k] for k in sorted(counts.keys(), key=lambda p: PHASE_RANK.get(p, 999))}

    wide_counts = count_by_phase(wide_rows)
    print("[validation] wide dedup counts:", wide_counts)
    print("[validation] expected wide counts:", EXPECTED_WIDE_COUNTS)
    for phase, expected in EXPECTED_WIDE_COUNTS.items():
        actual = wide_counts.get(phase, 0)
        mark = "OK" if actual == expected else "MISMATCH"
        print(f"  - {phase}: {actual} (expected {expected}) [{mark}]")

    print("[validation] verification dedup rows:", len(verification_rows), "expected:", EXPECTED_VERIFICATION_DEDUP)
    print("[validation] verification main rows:", len(verification_main), "expected:", EXPECTED_VERIFICATION_MAIN)
    if include_support_p10:
        print("[validation] verification support rows:", len(verification_support), "expected:", EXPECTED_VERIFICATION_SUPPORT)

    wide_special_missing = sum(1 for r in wide_rows if to_int(r.get("special_available"), 0) == 0)
    ver_special_missing = sum(1 for r in verification_rows if to_int(r.get("special_available"), 0) == 0)
    print(
        "[validation] special metric missing rows:",
        {"wide": wide_special_missing, "verification": ver_special_missing, "total": wide_special_missing + ver_special_missing},
    )

    print("[validation] diag missing (wide) count:", len(diag_missing_wide))
    if diag_missing_wide:
        print("[validation] diag missing (wide) run_phase:", ",".join(sorted(diag_missing_wide)))

    print("[validation] diag missing (verification overview) count:", len(diag_missing_verif))
    if diag_missing_verif:
        print("[validation] diag missing (verification overview) run_phase:", ",".join(sorted(diag_missing_verif)))

    if diag_json_missing_verif:
        print("[validation] diag raw json missing (verification) run_phase:", ",".join(sorted(set(diag_json_missing_verif))))

    print("[done] wrote phase10_13 data tables to:", out_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phase10~13 wide and verification report data tables.")
    parser.add_argument("--repo-root", default="/workspace/jy1559/FMoE")
    parser.add_argument("--dataset", default="KuaiRecLargeStrictPosV2_0.2")
    parser.add_argument(
        "--out-dir",
        default="/workspace/jy1559/FMoE/experiments/run/fmoe_n3/docs/data/phase10_13",
    )
    parser.add_argument("--wide-min-completed", type=int, default=20)
    parser.add_argument("--verification-main-hparam", default="H3")
    parser.add_argument("--verification-main-min-completed", type=int, default=20)
    parser.add_argument("--include-support-p10", type=parse_bool, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return build_data_tables(
        repo_root=Path(args.repo_root),
        dataset=str(args.dataset),
        out_dir=Path(args.out_dir),
        wide_min_completed=int(args.wide_min_completed),
        verification_main_hparam=str(args.verification_main_hparam),
        verification_main_min_completed=int(args.verification_main_min_completed),
        include_support_p10=bool(args.include_support_p10),
    )


if __name__ == "__main__":
    raise SystemExit(main())
