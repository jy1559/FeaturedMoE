#!/usr/bin/env python3
"""Build beauty feature_added_v3 dataset from basic and run split.

Stage 2 output format:
  Datasets/processed/feature_added_v3/beauty/beauty.inter
  Datasets/processed/feature_added_v3/beauty/beauty.item
  Datasets/processed/feature_added_v3/beauty/feature_meta_v3.json
  Datasets/processed/feature_added_v3/beauty/beauty.train.inter
  Datasets/processed/feature_added_v3/beauty/beauty.valid.inter
  Datasets/processed/feature_added_v3/beauty/beauty.test.inter
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "Datasets" / "processed"


FAMILIES = {
    "macro5": {
        "Tempo": ["mac5_ctx_valid_r", "mac5_gap_last", "mac5_pace_mean", "mac5_pace_trend"],
        "Focus": ["mac5_theme_ent_mean", "mac5_theme_top1_mean", "mac5_theme_repeat_r", "mac5_theme_shift_r"],
        "Memory": ["mac5_repeat_mean", "mac5_adj_cat_overlap_mean", "mac5_adj_item_overlap_mean", "mac5_repeat_trend"],
        "Exposure": ["mac5_pop_mean", "mac5_pop_std_mean", "mac5_pop_ent_mean", "mac5_pop_trend"],
    },
    "macro10": {
        "Tempo": ["mac10_ctx_valid_r", "mac10_gap_last", "mac10_pace_mean", "mac10_pace_trend"],
        "Focus": ["mac10_theme_ent_mean", "mac10_theme_top1_mean", "mac10_theme_repeat_r", "mac10_theme_shift_r"],
        "Memory": ["mac10_repeat_mean", "mac10_adj_cat_overlap_mean", "mac10_adj_item_overlap_mean", "mac10_repeat_trend"],
        "Exposure": ["mac10_pop_mean", "mac10_pop_std_mean", "mac10_pop_ent_mean", "mac10_pop_trend"],
    },
    "mid": {
        "Tempo": ["mid_valid_r", "mid_int_mean", "mid_int_std", "mid_sess_age"],
        "Focus": ["mid_cat_ent", "mid_cat_top1", "mid_cat_switch_r", "mid_cat_uniq_r"],
        "Memory": ["mid_item_uniq_r", "mid_repeat_r", "mid_novel_r", "mid_max_run_i"],
        "Exposure": ["mid_pop_mean", "mid_pop_std", "mid_pop_ent", "mid_pop_trend"],
    },
    "micro": {
        "Tempo": ["mic_valid_r", "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid"],
        "Focus": ["mic_cat_switch_now", "mic_last_cat_mismatch_r", "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r"],
        "Memory": ["mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i", "mic_suffix_max_run_i"],
        "Exposure": ["mic_last_pop", "mic_suffix_pop_std", "mic_suffix_pop_ent", "mic_pop_delta_vs_mid"],
    },
}

ALL_FEATURES: List[str] = []
for scope in ("macro5", "macro10", "mid", "micro"):
    for family in ("Tempo", "Focus", "Memory", "Exposure"):
        ALL_FEATURES.extend(FAMILIES[scope][family])

MID_FEATURES = [name for name in ALL_FEATURES if name.startswith("mid_")]

CONTINUOUS_FEATURES = {
    "mac5_gap_last",
    "mac5_pace_mean",
    "mac10_gap_last",
    "mac10_pace_mean",
    "mid_int_mean",
    "mid_int_std",
    "mid_sess_age",
    "mic_last_gap",
    "mic_gap_mean",
}

PAIR_FEATURES = {
    "mac5_pace_trend",
    "mac5_repeat_trend",
    "mac5_pop_trend",
    "mac10_pace_trend",
    "mac10_repeat_trend",
    "mac10_pop_trend",
    "mid_pop_trend",
    "mic_gap_delta_vs_mid",
    "mic_pop_delta_vs_mid",
}

DEFAULT_VALUE = {name: 0.5 for name in ALL_FEATURES}
for name in (
    "mac5_ctx_valid_r",
    "mac10_ctx_valid_r",
    "mid_valid_r",
    "mic_valid_r",
    "mic_cat_switch_now",
    "mic_last_cat_mismatch_r",
    "mic_suffix_cat_ent",
    "mic_suffix_cat_uniq_r",
    "mic_is_recons",
    "mic_suffix_recons_r",
    "mic_suffix_uniq_i",
    "mic_suffix_max_run_i",
    "mic_suffix_pop_ent",
):
    DEFAULT_VALUE[name] = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--source-subdir", type=str, default="basic")
    p.add_argument("--output-subdir", type=str, default="feature_added_v3")
    p.add_argument("--fit-session-ratio", type=float, default=0.7)
    p.add_argument("--micro-window", type=int, default=5)
    p.add_argument("--mid-valid-cap", type=int, default=10)
    p.add_argument("--n-pop-bins", type=int, default=10)
    p.add_argument("--split-ratios", type=str, default="0.7,0.15,0.15")
    p.add_argument("--split-strategy", type=str, default="tail_stratified", choices=["contiguous", "tail_stratified"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@dataclass
class Event:
    row_idx: int
    item: str
    ts_ms: float
    user: str


@dataclass
class SessionSummary:
    start_ts: float
    end_ts: float
    pace_mean: float | None
    cat_ent: float
    cat_top1: float
    cat_switch: float
    cat_repeat: float
    item_repeat: float
    pop_mean: float
    pop_std: float
    pop_ent: float
    item_set: set[str]
    cat_set: set[str]


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def pick_column(header: List[str], plain_name: str) -> str:
    for col in header:
        if strip_type(col) == plain_name:
            return col
    raise KeyError(f"Missing column: {plain_name}")


def safe_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return sum(vals) / len(vals)


def safe_std(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    n = len(vals)
    if n <= 1:
        return None
    mu = sum(vals) / n
    var = sum((x - mu) * (x - mu) for x in vals) / n
    return math.sqrt(max(0.0, var))


def entropy_ratio(tokens: Iterable[str]) -> float:
    vals = [str(x) for x in tokens]
    n = len(vals)
    if n <= 1:
        return 0.0
    c = Counter(vals)
    h = 0.0
    for cnt in c.values():
        p = cnt / n
        if p > 0:
            h -= p * math.log(p)
    denom = math.log(max(2, len(c)))
    return 0.0 if denom <= 0 else min(1.0, max(0.0, h / denom))


def top1_ratio(tokens: Iterable[str]) -> float:
    vals = [str(x) for x in tokens]
    n = len(vals)
    if n <= 0:
        return 0.0
    c = Counter(vals)
    return max(c.values()) / n


def uniq_ratio(tokens: Iterable[str]) -> float:
    vals = [str(x) for x in tokens]
    if not vals:
        return 0.0
    return len(set(vals)) / len(vals)


def switch_ratio(tokens: Iterable[str]) -> float:
    vals = [str(x) for x in tokens]
    if len(vals) <= 1:
        return 0.0
    switches = 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            switches += 1
    return switches / (len(vals) - 1)


def max_run_ratio(tokens: Iterable[str]) -> float:
    vals = [str(x) for x in tokens]
    if not vals:
        return 0.0
    best = 1
    run = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            run += 1
        else:
            run = 1
        if run > best:
            best = run
    return best / len(vals)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    den = len(a | b)
    if den <= 0:
        return 0.0
    return len(a & b) / den


def clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = (len(sorted_vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def load_basic_data(inter_path: Path, item_path: Path):
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        if not header:
            raise RuntimeError(f"Invalid inter file: {inter_path}")
        sid_col = pick_column(header, "session_id")
        item_col = pick_column(header, "item_id")
        ts_col = pick_column(header, "timestamp")
        user_col = pick_column(header, "user_id")

        by_sid: Dict[str, List[Event]] = defaultdict(list)
        for row_idx, row in enumerate(reader):
            sid = str(row[sid_col])
            by_sid[sid].append(
                Event(
                    row_idx=row_idx,
                    item=str(row[item_col]),
                    ts_ms=float(row[ts_col]),
                    user=str(row[user_col]),
                )
            )

    user_by_sid: Dict[str, str] = {}
    start_ts: Dict[str, float] = {}
    first_row: Dict[str, int] = {}
    for sid, events in by_sid.items():
        ordered = sorted(events, key=lambda e: (e.ts_ms, e.row_idx))
        by_sid[sid] = ordered
        user_by_sid[sid] = ordered[0].user
        start_ts[sid] = ordered[0].ts_ms
        first_row[sid] = ordered[0].row_idx

    session_order = sorted(by_sid.keys(), key=lambda sid: (start_ts[sid], first_row[sid], sid))
    sessions_by_user: Dict[str, List[str]] = defaultdict(list)
    for sid in session_order:
        sessions_by_user[user_by_sid[sid]].append(sid)

    item_cat: Dict[str, str] = {}
    with item_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header_item = list(reader.fieldnames or [])
        if not header_item:
            raise RuntimeError(f"Invalid item file: {item_path}")
        item_col = pick_column(header_item, "item_id")
        cat_col = pick_column(header_item, "category")
        for row in reader:
            item_cat[str(row[item_col])] = str(row[cat_col])

    return by_sid, sessions_by_user, session_order, item_cat


def build_popularity_scores(
    by_sid: Dict[str, List[Event]],
    fit_sessions: set[str],
    n_pop_bins: int,
) -> tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    item_counts = Counter()
    for sid in fit_sessions:
        for e in by_sid[sid]:
            item_counts[e.item] += 1
    if not item_counts:
        return {}, {}, {}

    uniq_counts = sorted(set(item_counts.values()))
    cdf = {}
    if len(uniq_counts) == 1:
        cdf[uniq_counts[0]] = 1.0
    else:
        denom = len(uniq_counts) - 1
        for i, c in enumerate(uniq_counts):
            cdf[c] = i / denom

    pop_score = {item: cdf[cnt] for item, cnt in item_counts.items()}
    pop_bin = {}
    for item, score in pop_score.items():
        b = int(math.floor(score * max(1, n_pop_bins - 1)))
        b = max(0, min(max(0, n_pop_bins - 1), b))
        pop_bin[item] = b
    return pop_score, dict(item_counts), pop_bin


def summarize_sessions(
    by_sid: Dict[str, List[Event]],
    item_cat: Dict[str, str],
    pop_score: Dict[str, float],
    pop_bin: Dict[str, int],
    n_pop_bins: int,
) -> Dict[str, SessionSummary]:
    out: Dict[str, SessionSummary] = {}
    for sid, events in by_sid.items():
        items = [e.item for e in events]
        cats = [item_cat.get(it, "Unknown") for it in items]
        pops = [pop_score.get(it, 0.0) for it in items]
        bins = [pop_bin.get(it, 0) for it in items]

        gaps = []
        for i in range(1, len(events)):
            dt_sec = max(0.0, (events[i].ts_ms - events[i - 1].ts_ms) / 1000.0)
            gaps.append(math.log1p(dt_sec))
        pace_mean = safe_mean(gaps)

        bin_tokens = [str(x) for x in bins]
        out[sid] = SessionSummary(
            start_ts=events[0].ts_ms,
            end_ts=events[-1].ts_ms,
            pace_mean=pace_mean,
            cat_ent=entropy_ratio(cats),
            cat_top1=top1_ratio(cats),
            cat_switch=switch_ratio(cats),
            cat_repeat=1.0 - uniq_ratio(cats),
            item_repeat=1.0 - uniq_ratio(items),
            pop_mean=safe_mean(pops) or 0.0,
            pop_std=safe_std(pops) or 0.0,
            pop_ent=entropy_ratio(bin_tokens),
            item_set=set(items),
            cat_set=set(cats),
        )
    return out


def macro_features_for_session(
    *,
    current_start: float,
    ctx_sids: List[str],
    summaries: Dict[str, SessionSummary],
    window: int,
    prefix: str,
) -> Dict[str, float | None]:
    out = {
        f"{prefix}_ctx_valid_r": min(len(ctx_sids), window) / float(window),
        f"{prefix}_gap_last": None,
        f"{prefix}_pace_mean": None,
        f"{prefix}_pace_trend": None,
        f"{prefix}_theme_ent_mean": None,
        f"{prefix}_theme_top1_mean": None,
        f"{prefix}_theme_repeat_r": None,
        f"{prefix}_theme_shift_r": None,
        f"{prefix}_repeat_mean": None,
        f"{prefix}_adj_cat_overlap_mean": None,
        f"{prefix}_adj_item_overlap_mean": None,
        f"{prefix}_repeat_trend": None,
        f"{prefix}_pop_mean": None,
        f"{prefix}_pop_std_mean": None,
        f"{prefix}_pop_ent_mean": None,
        f"{prefix}_pop_trend": None,
    }
    if not ctx_sids:
        return out

    ctx = [summaries[sid] for sid in ctx_sids]
    last = ctx[-1]
    first = ctx[0]
    out[f"{prefix}_gap_last"] = max(0.0, (current_start - last.end_ts) / 1000.0)
    out[f"{prefix}_pace_mean"] = safe_mean([x.pace_mean for x in ctx if x.pace_mean is not None])
    if len(ctx) >= 2 and first.pace_mean is not None and last.pace_mean is not None:
        out[f"{prefix}_pace_trend"] = last.pace_mean - first.pace_mean
    else:
        out[f"{prefix}_pace_trend"] = 0.0

    out[f"{prefix}_theme_ent_mean"] = safe_mean([x.cat_ent for x in ctx])
    out[f"{prefix}_theme_top1_mean"] = safe_mean([x.cat_top1 for x in ctx])
    out[f"{prefix}_theme_repeat_r"] = safe_mean([x.cat_repeat for x in ctx])
    out[f"{prefix}_theme_shift_r"] = safe_mean([x.cat_switch for x in ctx])

    out[f"{prefix}_repeat_mean"] = safe_mean([x.item_repeat for x in ctx])
    if len(ctx) >= 2:
        out[f"{prefix}_repeat_trend"] = ctx[-1].item_repeat - ctx[0].item_repeat
    else:
        out[f"{prefix}_repeat_trend"] = 0.0

    cat_j = []
    item_j = []
    for i in range(1, len(ctx)):
        cat_j.append(jaccard(ctx[i - 1].cat_set, ctx[i].cat_set))
        item_j.append(jaccard(ctx[i - 1].item_set, ctx[i].item_set))
    out[f"{prefix}_adj_cat_overlap_mean"] = safe_mean(cat_j) if cat_j else None
    out[f"{prefix}_adj_item_overlap_mean"] = safe_mean(item_j) if item_j else None

    out[f"{prefix}_pop_mean"] = safe_mean([x.pop_mean for x in ctx])
    out[f"{prefix}_pop_std_mean"] = safe_mean([x.pop_std for x in ctx])
    out[f"{prefix}_pop_ent_mean"] = safe_mean([x.pop_ent for x in ctx])
    if len(ctx) >= 2:
        out[f"{prefix}_pop_trend"] = ctx[-1].pop_mean - ctx[0].pop_mean
    else:
        out[f"{prefix}_pop_trend"] = 0.0
    return out


def build_macro_by_session(
    *,
    sessions_by_user: Dict[str, List[str]],
    summaries: Dict[str, SessionSummary],
) -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for _user, sids in sessions_by_user.items():
        for i, sid in enumerate(sids):
            current_start = summaries[sid].start_ts
            ctx5 = sids[max(0, i - 5):i]
            ctx10 = sids[max(0, i - 10):i]
            f5 = macro_features_for_session(
                current_start=current_start,
                ctx_sids=ctx5,
                summaries=summaries,
                window=5,
                prefix="mac5",
            )
            f10 = macro_features_for_session(
                current_start=current_start,
                ctx_sids=ctx10,
                summaries=summaries,
                window=10,
                prefix="mac10",
            )
            merged = dict(f5)
            merged.update(f10)
            out[sid] = merged
    return out


def compute_session_raw_rows(
    *,
    events: List[Event],
    macro: Dict[str, float | None],
    item_cat: Dict[str, str],
    pop_score: Dict[str, float],
    pop_bin: Dict[str, int],
    micro_window: int,
    mid_valid_cap: int,
    n_pop_bins: int,
) -> List[Dict[str, float | None]]:
    rows: List[Dict[str, float | None]] = []

    items = [e.item for e in events]
    cats = [item_cat.get(it, "Unknown") for it in items]
    pops = [pop_score.get(it, 0.0) for it in items]
    bins = [pop_bin.get(it, 0) for it in items]
    ts = [e.ts_ms for e in events]

    for i in range(len(events)):
        feat = dict(macro)

        hist_items = items[:i]
        prefix_items = items[: i + 1]
        prefix_cats = cats[: i + 1]
        prefix_pops = pops[: i + 1]
        prefix_bins = bins[: i + 1]

        feat["mid_valid_r"] = min(i, mid_valid_cap) / float(mid_valid_cap)
        gap_logs = []
        for j in range(1, i + 1):
            dt = max(0.0, (ts[j] - ts[j - 1]) / 1000.0)
            gap_logs.append(math.log1p(dt))
        feat["mid_int_mean"] = safe_mean(gap_logs)
        feat["mid_int_std"] = safe_std(gap_logs)
        feat["mid_sess_age"] = max(0.0, (ts[i] - ts[0]) / 1000.0)

        feat["mid_cat_ent"] = entropy_ratio(prefix_cats)
        feat["mid_cat_top1"] = top1_ratio(prefix_cats)
        feat["mid_cat_switch_r"] = switch_ratio(prefix_cats)
        feat["mid_cat_uniq_r"] = uniq_ratio(prefix_cats)

        feat["mid_item_uniq_r"] = uniq_ratio(prefix_items)
        feat["mid_repeat_r"] = 1.0 - feat["mid_item_uniq_r"]
        feat["mid_novel_r"] = feat["mid_item_uniq_r"]
        feat["mid_max_run_i"] = max_run_ratio(prefix_items)

        feat["mid_pop_mean"] = safe_mean(prefix_pops)
        feat["mid_pop_std"] = safe_std(prefix_pops)
        feat["mid_pop_ent"] = entropy_ratio([str(x) for x in prefix_bins])
        if len(prefix_pops) > 1:
            feat["mid_pop_trend"] = prefix_pops[-1] - (sum(prefix_pops[:-1]) / len(prefix_pops[:-1]))
        else:
            feat["mid_pop_trend"] = 0.0

        win = hist_items[-micro_window:]
        win_cats = cats[max(0, i - micro_window):i]
        win_pops = pops[max(0, i - micro_window):i]
        win_bins = bins[max(0, i - micro_window):i]
        win_ts = ts[max(0, i - micro_window):i]
        w = len(win)

        feat["mic_valid_r"] = w / float(max(1, micro_window))
        if w <= 0:
            feat["mic_last_gap"] = None
            feat["mic_gap_mean"] = None
            feat["mic_gap_delta_vs_mid"] = None
            feat["mic_cat_switch_now"] = 0.0
            feat["mic_last_cat_mismatch_r"] = 0.0
            feat["mic_suffix_cat_ent"] = 0.0
            feat["mic_suffix_cat_uniq_r"] = 0.0
            feat["mic_is_recons"] = 0.0
            feat["mic_suffix_recons_r"] = 0.0
            feat["mic_suffix_uniq_i"] = 0.0
            feat["mic_suffix_max_run_i"] = 0.0
            feat["mic_last_pop"] = None
            feat["mic_suffix_pop_std"] = None
            feat["mic_suffix_pop_ent"] = 0.0
            feat["mic_pop_delta_vs_mid"] = None
        else:
            feat["mic_last_gap"] = max(0.0, (ts[i] - win_ts[-1]) / 1000.0)

            local_gap_logs = []
            local_ts = win_ts + [ts[i]]
            for j in range(1, len(local_ts)):
                dt = max(0.0, (local_ts[j] - local_ts[j - 1]) / 1000.0)
                local_gap_logs.append(math.log1p(dt))
            gap_mean = safe_mean(local_gap_logs)
            feat["mic_gap_mean"] = gap_mean
            if gap_mean is not None and feat["mid_int_mean"] is not None:
                feat["mic_gap_delta_vs_mid"] = gap_mean - float(feat["mid_int_mean"])
            else:
                feat["mic_gap_delta_vs_mid"] = None

            cur_cat = cats[i]
            feat["mic_cat_switch_now"] = 1.0 if cur_cat != win_cats[-1] else 0.0
            mismatch = sum(1 for c in win_cats if c != cur_cat)
            feat["mic_last_cat_mismatch_r"] = mismatch / float(w)
            feat["mic_suffix_cat_ent"] = entropy_ratio(win_cats)
            feat["mic_suffix_cat_uniq_r"] = uniq_ratio(win_cats)

            feat["mic_is_recons"] = 1.0 if items[i] in win else 0.0
            feat["mic_suffix_recons_r"] = 1.0 - uniq_ratio(win)
            feat["mic_suffix_uniq_i"] = uniq_ratio(win)
            feat["mic_suffix_max_run_i"] = max_run_ratio(win)

            feat["mic_last_pop"] = pops[i]
            feat["mic_suffix_pop_std"] = safe_std(win_pops)
            feat["mic_suffix_pop_ent"] = entropy_ratio([str(x) for x in win_bins])
            if feat["mid_pop_mean"] is not None:
                feat["mic_pop_delta_vs_mid"] = pops[i] - float(feat["mid_pop_mean"])
            else:
                feat["mic_pop_delta_vs_mid"] = None

        rows.append(feat)

    if rows:
        last_mid = {k: rows[-1].get(k) for k in MID_FEATURES}
        for r in rows:
            for k in MID_FEATURES:
                r[k] = last_mid[k]

    return rows


def compute_continuous_stats(values_by_feature: Dict[str, List[float]]) -> Dict[str, dict]:
    stats = {}
    for name in CONTINUOUS_FEATURES:
        vals = sorted(values_by_feature.get(name, []))
        if not vals:
            stats[name] = {
                "type": "continuous",
                "transform": "log1p_winsorize_z_phi",
                "mean": 0.0,
                "std": 1.0,
                "q01": 0.0,
                "q99": 0.0,
                "missing_default": 0.5,
            }
            continue
        q01 = quantile(vals, 0.01)
        q99 = quantile(vals, 0.99)
        mu = safe_mean(vals) or 0.0
        sd = safe_std(vals) or 1.0
        if sd <= 1e-12:
            sd = 1.0
        stats[name] = {
            "type": "continuous",
            "transform": "log1p_winsorize_z_phi",
            "mean": float(mu),
            "std": float(sd),
            "q01": float(q01),
            "q99": float(q99),
            "missing_default": 0.5,
        }
    return stats


def normalize_feature(name: str, value: float | None, stats: Dict[str, dict]) -> float:
    default = DEFAULT_VALUE.get(name, 0.5)
    if name in CONTINUOUS_FEATURES:
        if value is None:
            return default
        cfg = stats[name]
        x = math.log1p(max(0.0, float(value)))
        x = min(max(x, cfg["q01"]), cfg["q99"])
        z = (x - cfg["mean"]) / max(1e-12, cfg["std"])
        return clip01(phi(z))

    if name in PAIR_FEATURES:
        if value is None:
            return default
        return clip01(0.5 + 0.5 * math.tanh(float(value)))

    if value is None:
        return default
    return clip01(float(value))


def load_split_module():
    split_path = Path(__file__).resolve().parent / "split_feature_v3_by_session_time.py"
    spec = importlib.util.spec_from_file_location("split_feature_v3_by_session_time", split_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load split_feature_v3_by_session_time.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset)

    source_dir = Path(args.processed_root) / str(args.source_subdir) / dataset
    out_root = Path(args.processed_root) / str(args.output_subdir)
    out_dir = out_root / dataset
    inter_in = source_dir / f"{dataset}.inter"
    item_in = source_dir / f"{dataset}.item"
    inter_out = out_dir / f"{dataset}.inter"
    item_out = out_dir / f"{dataset}.item"

    if not inter_in.exists() or not item_in.exists():
        raise SystemExit(f"Missing source files under: {source_dir}")
    if inter_out.exists() and not bool(args.overwrite):
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading basic dataset")
    by_sid, sessions_by_user, session_order, item_cat = load_basic_data(inter_in, item_in)
    total_sessions = len(session_order)
    total_rows = sum(len(by_sid[sid]) for sid in session_order)
    print(f"  sessions={total_sessions} rows={total_rows} items={len(item_cat)}")

    fit_n = max(1, int(math.floor(total_sessions * float(args.fit_session_ratio))))
    fit_sids = set(session_order[:fit_n])
    fit_last_start = by_sid[session_order[fit_n - 1]][0].ts_ms if fit_n > 0 else None

    print("[2/7] popularity fit stats")
    pop_score, pop_counts, pop_bin = build_popularity_scores(
        by_sid=by_sid,
        fit_sessions=fit_sids,
        n_pop_bins=int(args.n_pop_bins),
    )
    print(f"  fit_sessions={fit_n} fit_items={len(pop_counts)}")

    print("[3/7] session summaries + macro context")
    summaries = summarize_sessions(
        by_sid=by_sid,
        item_cat=item_cat,
        pop_score=pop_score,
        pop_bin=pop_bin,
        n_pop_bins=int(args.n_pop_bins),
    )
    macro_by_sid = build_macro_by_session(sessions_by_user=sessions_by_user, summaries=summaries)

    print("[4/7] fit-time normalization stats")
    cont_values: Dict[str, List[float]] = defaultdict(list)
    for sid in session_order:
        rows = compute_session_raw_rows(
            events=by_sid[sid],
            macro=macro_by_sid[sid],
            item_cat=item_cat,
            pop_score=pop_score,
            pop_bin=pop_bin,
            micro_window=int(args.micro_window),
            mid_valid_cap=int(args.mid_valid_cap),
            n_pop_bins=int(args.n_pop_bins),
        )
        if sid not in fit_sids:
            continue
        for r in rows:
            for name in CONTINUOUS_FEATURES:
                v = r.get(name)
                if v is None:
                    continue
                cont_values[name].append(math.log1p(max(0.0, float(v))))

    norm_stats = compute_continuous_stats(cont_values)

    print("[5/7] writing feature_added_v3 inter/item")
    with inter_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        header = ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"] + [f"{n}:float" for n in ALL_FEATURES]
        writer.writerow(header)

        for sid in session_order:
            events = by_sid[sid]
            raw_rows = compute_session_raw_rows(
                events=events,
                macro=macro_by_sid[sid],
                item_cat=item_cat,
                pop_score=pop_score,
                pop_bin=pop_bin,
                micro_window=int(args.micro_window),
                mid_valid_cap=int(args.mid_valid_cap),
                n_pop_bins=int(args.n_pop_bins),
            )
            for e, raw in zip(events, raw_rows):
                values = [normalize_feature(name, raw.get(name), norm_stats) for name in ALL_FEATURES]
                writer.writerow([sid, e.item, int(e.ts_ms), e.user] + values)

    shutil.copy2(item_in, item_out)

    print("[6/7] writing feature meta")
    normalization_stats = {}
    for name in ALL_FEATURES:
        if name in CONTINUOUS_FEATURES:
            normalization_stats[name] = dict(norm_stats[name])
        elif name in PAIR_FEATURES:
            normalization_stats[name] = {
                "type": "pair_bounded",
                "transform": "0.5 + 0.5 * tanh(delta)",
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }
        else:
            normalization_stats[name] = {
                "type": "bounded",
                "transform": "clip01",
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }

    meta = {
        "dataset": dataset,
        "all_features": ALL_FEATURES,
        "families": FAMILIES,
        "macro_windows": [5, 10],
        "micro_window": int(args.micro_window),
        "mid_valid_cap": int(args.mid_valid_cap),
        "mid_scope_original": "session_full",
        "mid_scope": "session_constant_last",
        "mid_constant_source": "last_interaction_mid_features",
        "mid_constant_postprocess": {
            "applied": True,
            "dataset": dataset,
            "mid_columns": len(MID_FEATURES),
            "rows": total_rows,
            "sessions": total_sessions,
        },
        "fit_sessions": {
            "fit_session_ratio": float(args.fit_session_ratio),
            "fit_session_count": fit_n,
            "total_session_count": total_sessions,
            "fit_last_session_start_timestamp": float(fit_last_start) if fit_last_start is not None else None,
        },
        "n_pop_bins": int(args.n_pop_bins),
        "macro_missing_policy": {
            "ctx_valid_r": "min(n_ctx, K) / K",
            "gap_last": "neutral 0.5 when no completed prior session exists",
            "average_features": "computed from available prior sessions only",
            "pairwise_and_trend": "neutral 0.5 when insufficient context",
            "n_ctx_eq_0": "all non-valid macro features set to neutral 0.5 after normalization",
        },
        "normalization": {
            "bounded": "clip01",
            "continuous": "log1p -> winsorize(p01,p99) -> zscore -> phi",
            "pair_bounded": "0.5 + 0.5 * tanh(delta)",
            "continuous_missing_default": 0.5,
        },
        "normalization_stats": normalization_stats,
    }
    meta_path = out_dir / "feature_meta_v3.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[7/7] split train/valid/test")
    split_mod = load_split_module()
    ratios = split_mod.parse_ratios(str(args.split_ratios))
    split_summary = split_mod.process_dataset(
        processed_root=out_root,
        dataset=dataset,
        ratios=ratios,
        strategy=str(args.split_strategy),
        overwrite=bool(args.overwrite),
        dry_run=False,
    )

    build_summary = {
        "dataset": dataset,
        "source": str(source_dir),
        "target": str(out_dir),
        "rows": total_rows,
        "sessions": total_sessions,
        "features": len(ALL_FEATURES),
        "split_summary": split_summary,
        "meta_path": str(meta_path),
    }
    summary_path = out_dir / f"{dataset}.v3_build_summary.json"
    summary_path.write_text(json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: {summary_path}")


if __name__ == "__main__":
    main()
