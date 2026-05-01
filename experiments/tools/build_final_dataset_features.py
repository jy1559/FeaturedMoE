#!/usr/bin/env python3
"""Build final dataset with behavioral features from a basic inter/item file.

Reads from:
  Datasets/processed/basic/{dataset}/{dataset}.inter
  Datasets/processed/basic/{dataset}/{dataset}.item

Writes to:
  Datasets/processed/final_dataset/{dataset}/{dataset}.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.item
  Datasets/processed/final_dataset/{dataset}/{dataset}.train.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.valid.inter
  Datasets/processed/final_dataset/{dataset}/{dataset}.test.inter
  Datasets/processed/final_dataset/{dataset}/feature_meta_v3.json

Feature structure (64 features):
  macro (mac5 + mac10): 16 each — Tempo / Focus / Memory / Exposure
  mid:                  16      — session-level (session_constant_last)
  micro (mic):          16      — recent-5-interaction level

This script is a direct port of build_beauty_feature_v3.py, parameterised for
any basic dataset. The normalization statistics are fit on the first
--fit-session-ratio fraction of sessions (default 0.7) ordered by start time.
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
DEFAULT_BASIC_ROOT = REPO_ROOT / "Datasets" / "processed" / "basic"
DEFAULT_OUT_ROOT   = REPO_ROOT / "Datasets" / "processed" / "final_dataset"


# ---------------------------------------------------------------------------
# Feature / normalization definitions (identical to build_beauty_feature_v3)
# ---------------------------------------------------------------------------

FAMILIES = {
    "macro5": {
        "Tempo":    ["mac5_ctx_valid_r", "mac5_gap_last", "mac5_pace_mean", "mac5_pace_trend"],
        "Focus":    ["mac5_theme_ent_mean", "mac5_theme_top1_mean", "mac5_theme_repeat_r", "mac5_theme_shift_r"],
        "Memory":   ["mac5_repeat_mean", "mac5_adj_cat_overlap_mean", "mac5_adj_item_overlap_mean", "mac5_repeat_trend"],
        "Exposure": ["mac5_pop_mean", "mac5_pop_std_mean", "mac5_pop_ent_mean", "mac5_pop_trend"],
    },
    "macro10": {
        "Tempo":    ["mac10_ctx_valid_r", "mac10_gap_last", "mac10_pace_mean", "mac10_pace_trend"],
        "Focus":    ["mac10_theme_ent_mean", "mac10_theme_top1_mean", "mac10_theme_repeat_r", "mac10_theme_shift_r"],
        "Memory":   ["mac10_repeat_mean", "mac10_adj_cat_overlap_mean", "mac10_adj_item_overlap_mean", "mac10_repeat_trend"],
        "Exposure": ["mac10_pop_mean", "mac10_pop_std_mean", "mac10_pop_ent_mean", "mac10_pop_trend"],
    },
    "mid": {
        "Tempo":    ["mid_valid_r", "mid_int_mean", "mid_int_std", "mid_sess_age"],
        "Focus":    ["mid_cat_ent", "mid_cat_top1", "mid_cat_switch_r", "mid_cat_uniq_r"],
        "Memory":   ["mid_item_uniq_r", "mid_repeat_r", "mid_novel_r", "mid_max_run_i"],
        "Exposure": ["mid_pop_mean", "mid_pop_std", "mid_pop_ent", "mid_pop_trend"],
    },
    "micro": {
        "Tempo":    ["mic_valid_r", "mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid"],
        "Focus":    ["mic_cat_switch_now", "mic_last_cat_mismatch_r", "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r"],
        "Memory":   ["mic_is_recons", "mic_suffix_recons_r", "mic_suffix_uniq_i", "mic_suffix_max_run_i"],
        "Exposure": ["mic_last_pop", "mic_suffix_pop_std", "mic_suffix_pop_ent", "mic_pop_delta_vs_mid"],
    },
}

ALL_FEATURES: List[str] = []
for _scope in ("macro5", "macro10", "mid", "micro"):
    for _fam in ("Tempo", "Focus", "Memory", "Exposure"):
        ALL_FEATURES.extend(FAMILIES[_scope][_fam])

MID_FEATURES = [n for n in ALL_FEATURES if n.startswith("mid_")]

CONTINUOUS_FEATURES = {
    "mac5_gap_last", "mac5_pace_mean",
    "mac10_gap_last", "mac10_pace_mean",
    "mid_int_mean", "mid_int_std", "mid_sess_age",
    "mic_last_gap", "mic_gap_mean",
    "mac5_pop_mean", "mac5_pop_std_mean",
    "mac10_pop_mean", "mac10_pop_std_mean",
    "mid_pop_mean", "mid_pop_std",
    "mic_last_pop", "mic_suffix_pop_std",
}

PAIR_FEATURES = {
    "mac5_pace_trend", "mac5_repeat_trend", "mac5_pop_trend",
    "mac10_pace_trend", "mac10_repeat_trend", "mac10_pop_trend",
    "mid_pop_trend",
    "mic_gap_delta_vs_mid", "mic_pop_delta_vs_mid",
}

# Default missing value for each feature
DEFAULT_VALUE: Dict[str, float] = {n: 0.5 for n in ALL_FEATURES}
for _n in (
    "mac5_ctx_valid_r", "mac10_ctx_valid_r",
    "mid_valid_r", "mic_valid_r",
    "mic_cat_switch_now", "mic_last_cat_mismatch_r",
    "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
    "mic_is_recons", "mic_suffix_recons_r",
    "mic_suffix_uniq_i", "mic_suffix_max_run_i",
    "mic_suffix_pop_ent",
):
    DEFAULT_VALUE[_n] = 0.0


# ---------------------------------------------------------------------------
# arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--basic-root",      type=Path, default=DEFAULT_BASIC_ROOT)
    p.add_argument("--output-root",     type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dataset",         type=str,  required=True)
    p.add_argument("--fit-session-ratio", type=float, default=0.7)
    p.add_argument("--micro-window",    type=int,  default=5)
    p.add_argument("--mid-valid-cap",   type=int,  default=10)
    p.add_argument("--n-pop-bins",      type=int,  default=10)
    p.add_argument("--split-ratios",    type=str,  default="0.7,0.15,0.15")
    p.add_argument("--split-strategy",  type=str,  default="tail_stratified",
                   choices=["contiguous", "tail_stratified"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Event:
    row_idx: int
    item:    str
    ts_ms:   float
    user:    str


@dataclass
class SessionSummary:
    start_ts:    float
    end_ts:      float
    pace_mean:   float | None
    cat_ent:     float
    cat_top1:    float
    cat_switch:  float
    cat_repeat:  float
    item_repeat: float
    pop_mean:    float
    pop_std:     float
    pop_ent:     float
    item_set:    set[str]
    cat_set:     set[str]


# ---------------------------------------------------------------------------
# Helper math functions
# ---------------------------------------------------------------------------

def safe_mean(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    return (sum(vals) / len(vals)) if vals else None


def safe_std(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs]
    n = len(vals)
    if n <= 1:
        return None
    mu = sum(vals) / n
    return math.sqrt(max(0.0, sum((x - mu) ** 2 for x in vals) / n))


def entropy_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    n = len(vals)
    if n <= 1:
        return 0.0
    c = Counter(vals)
    h = -sum((cnt / n) * math.log(cnt / n) for cnt in c.values() if cnt > 0)
    denom = math.log(max(2, len(c)))
    return 0.0 if denom <= 0 else min(1.0, max(0.0, h / denom))


def top1_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    return max(Counter(vals).values()) / len(vals) if vals else 0.0


def uniq_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    return len(set(vals)) / len(vals) if vals else 0.0


def switch_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    if len(vals) <= 1:
        return 0.0
    return sum(1 for i in range(1, len(vals)) if vals[i] != vals[i - 1]) / (len(vals) - 1)


def max_run_ratio(tokens: Iterable[str]) -> float:
    vals = list(tokens)
    if not vals:
        return 0.0
    best = run = 1
    for i in range(1, len(vals)):
        run = (run + 1) if vals[i] == vals[i - 1] else 1
        best = max(best, run)
    return best / len(vals)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    den = len(a | b)
    return len(a & b) / den if den else 0.0


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def quantile_sorted(sv: List[float], q: float) -> float:
    if not sv:
        return 0.0
    if q <= 0:
        return sv[0]
    if q >= 1:
        return sv[-1]
    pos = (len(sv) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return sv[lo] if lo == hi else sv[lo] * (1 - (pos - lo)) + sv[hi] * (pos - lo)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def load_basic_data(
    inter_path: Path, item_path: Path
) -> tuple[Dict[str, List[Event]], Dict[str, List[str]], List[str], Dict[str, str]]:
    by_sid: Dict[str, List[Event]] = defaultdict(list)
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        hdr = [strip_type(c) for c in (reader.fieldnames or [])]
        for row_idx, row in enumerate(reader):
            # map stripped names
            plain = {strip_type(k): v for k, v in row.items()}
            by_sid[plain["session_id"]].append(Event(
                row_idx=row_idx,
                item=plain["item_id"],
                ts_ms=float(plain["timestamp"]),
                user=plain["user_id"],
            ))

    user_by_sid: Dict[str, str] = {}
    start_ts:    Dict[str, float] = {}
    first_row:   Dict[str, int]   = {}
    for sid, evts in by_sid.items():
        evts.sort(key=lambda e: (e.ts_ms, e.row_idx))
        by_sid[sid] = evts
        user_by_sid[sid] = evts[0].user
        start_ts[sid]    = evts[0].ts_ms
        first_row[sid]   = evts[0].row_idx

    session_order = sorted(by_sid.keys(), key=lambda s: (start_ts[s], first_row[s], s))
    sessions_by_user: Dict[str, List[str]] = defaultdict(list)
    for sid in session_order:
        sessions_by_user[user_by_sid[sid]].append(sid)

    item_cat: Dict[str, str] = {}
    with item_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            plain = {strip_type(k): v for k, v in row.items()}
            item_cat[plain["item_id"]] = plain["category"]

    return by_sid, sessions_by_user, session_order, item_cat


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------

def build_popularity(
    by_sid: Dict[str, List[Event]],
    fit_sessions: set[str],
    n_bins: int,
) -> tuple[Dict[str, float], Dict[str, int]]:
    item_counts: Counter = Counter()
    for sid in fit_sessions:
        for e in by_sid[sid]:
            item_counts[e.item] += 1
    if not item_counts:
        return {}, {}
    uniq = sorted(set(item_counts.values()))
    denom = max(1, len(uniq) - 1)
    cdf = {c: i / denom for i, c in enumerate(uniq)}
    pop_score = {item: cdf[cnt] for item, cnt in item_counts.items()}
    pop_bin   = {
        item: max(0, min(n_bins - 1, int(math.floor(s * max(1, n_bins - 1)))))
        for item, s in pop_score.items()
    }
    return pop_score, pop_bin


# ---------------------------------------------------------------------------
# Session summaries
# ---------------------------------------------------------------------------

def summarize_sessions(
    by_sid: Dict[str, List[Event]],
    item_cat: Dict[str, str],
    pop_score: Dict[str, float],
    pop_bin: Dict[str, int],
    n_bins: int,
) -> Dict[str, SessionSummary]:
    out: Dict[str, SessionSummary] = {}
    for sid, evts in by_sid.items():
        items = [e.item for e in evts]
        cats  = [item_cat.get(it, "Unknown") for it in items]
        pops  = [pop_score.get(it, 0.0) for it in items]
        bins  = [pop_bin.get(it, 0) for it in items]
        gaps  = [
            math.log1p(max(0.0, (evts[i].ts_ms - evts[i - 1].ts_ms) / 1000.0))
            for i in range(1, len(evts))
        ]
        out[sid] = SessionSummary(
            start_ts=evts[0].ts_ms,
            end_ts=evts[-1].ts_ms,
            pace_mean=safe_mean(gaps),
            cat_ent=entropy_ratio(cats),
            cat_top1=top1_ratio(cats),
            cat_switch=switch_ratio(cats),
            cat_repeat=1.0 - uniq_ratio(cats),
            item_repeat=1.0 - uniq_ratio(items),
            pop_mean=safe_mean(pops) or 0.0,
            pop_std=safe_std(pops)   or 0.0,
            pop_ent=entropy_ratio([str(b) for b in bins]),
            item_set=set(items),
            cat_set=set(cats),
        )
    return out


# ---------------------------------------------------------------------------
# Macro features per session
# ---------------------------------------------------------------------------

def macro_features(
    *,
    current_start: float,
    ctx_sids: List[str],
    summaries: Dict[str, SessionSummary],
    window: int,
    prefix: str,
) -> Dict[str, float | None]:
    out: Dict[str, float | None] = {
        f"{prefix}_ctx_valid_r": min(len(ctx_sids), window) / float(window),
        f"{prefix}_gap_last": None,
        f"{prefix}_pace_mean": None, f"{prefix}_pace_trend": None,
        f"{prefix}_theme_ent_mean": None, f"{prefix}_theme_top1_mean": None,
        f"{prefix}_theme_repeat_r": None, f"{prefix}_theme_shift_r": None,
        f"{prefix}_repeat_mean": None,
        f"{prefix}_adj_cat_overlap_mean": None, f"{prefix}_adj_item_overlap_mean": None,
        f"{prefix}_repeat_trend": None,
        f"{prefix}_pop_mean": None, f"{prefix}_pop_std_mean": None,
        f"{prefix}_pop_ent_mean": None, f"{prefix}_pop_trend": None,
    }
    if not ctx_sids:
        return out
    ctx = [summaries[s] for s in ctx_sids]
    last, first = ctx[-1], ctx[0]
    out[f"{prefix}_gap_last"]       = max(0.0, (current_start - last.end_ts) / 1000.0)
    out[f"{prefix}_pace_mean"]      = safe_mean([x.pace_mean for x in ctx if x.pace_mean is not None])
    out[f"{prefix}_pace_trend"]     = (last.pace_mean - first.pace_mean) if (len(ctx) >= 2 and first.pace_mean is not None and last.pace_mean is not None) else 0.0
    out[f"{prefix}_theme_ent_mean"] = safe_mean([x.cat_ent   for x in ctx])
    out[f"{prefix}_theme_top1_mean"]= safe_mean([x.cat_top1  for x in ctx])
    out[f"{prefix}_theme_repeat_r"] = safe_mean([x.cat_repeat for x in ctx])
    out[f"{prefix}_theme_shift_r"]  = safe_mean([x.cat_switch for x in ctx])
    out[f"{prefix}_repeat_mean"]    = safe_mean([x.item_repeat for x in ctx])
    out[f"{prefix}_repeat_trend"]   = (ctx[-1].item_repeat - ctx[0].item_repeat) if len(ctx) >= 2 else 0.0
    adj_cat  = [jaccard(ctx[i-1].cat_set,  ctx[i].cat_set)  for i in range(1, len(ctx))]
    adj_item = [jaccard(ctx[i-1].item_set, ctx[i].item_set) for i in range(1, len(ctx))]
    out[f"{prefix}_adj_cat_overlap_mean"]  = safe_mean(adj_cat)  if adj_cat  else None
    out[f"{prefix}_adj_item_overlap_mean"] = safe_mean(adj_item) if adj_item else None
    out[f"{prefix}_pop_mean"]      = safe_mean([x.pop_mean for x in ctx])
    out[f"{prefix}_pop_std_mean"]  = safe_mean([x.pop_std  for x in ctx])
    out[f"{prefix}_pop_ent_mean"]  = safe_mean([x.pop_ent  for x in ctx])
    out[f"{prefix}_pop_trend"]     = (ctx[-1].pop_mean - ctx[0].pop_mean) if len(ctx) >= 2 else 0.0
    return out


def build_macro_by_session(
    sessions_by_user: Dict[str, List[str]],
    summaries: Dict[str, SessionSummary],
) -> Dict[str, Dict[str, float | None]]:
    out: Dict[str, Dict[str, float | None]] = {}
    for _user, sids in sessions_by_user.items():
        for i, sid in enumerate(sids):
            start = summaries[sid].start_ts
            f5  = macro_features(current_start=start, ctx_sids=sids[max(0, i-5):i],  summaries=summaries, window=5,  prefix="mac5")
            f10 = macro_features(current_start=start, ctx_sids=sids[max(0, i-10):i], summaries=summaries, window=10, prefix="mac10")
            merged = dict(f5)
            merged.update(f10)
            out[sid] = merged
    return out


# ---------------------------------------------------------------------------
# Per-row feature computation (mid + micro)
# ---------------------------------------------------------------------------

def compute_session_rows(
    *,
    events:        List[Event],
    macro:         Dict[str, float | None],
    item_cat:      Dict[str, str],
    pop_score:     Dict[str, float],
    pop_bin:       Dict[str, int],
    micro_window:  int,
    mid_valid_cap: int,
    n_pop_bins:    int,
) -> List[Dict[str, float | None]]:
    rows: List[Dict[str, float | None]] = []
    items = [e.item for e in events]
    cats  = [item_cat.get(it, "Unknown") for it in items]
    pops  = [pop_score.get(it, 0.0) for it in items]
    bins  = [pop_bin.get(it, 0)     for it in items]
    ts    = [e.ts_ms for e in events]

    for i in range(len(events)):
        feat = dict(macro)
        prefix_items = items[:i + 1]
        prefix_cats  = cats[:i + 1]
        prefix_pops  = pops[:i + 1]
        prefix_bins  = bins[:i + 1]

        # ── mid features ────────────────────────────────────────────────────
        feat["mid_valid_r"]    = min(i, mid_valid_cap) / float(mid_valid_cap)
        gap_logs = [math.log1p(max(0.0, (ts[j] - ts[j-1]) / 1000.0)) for j in range(1, i+1)]
        feat["mid_int_mean"]   = safe_mean(gap_logs)
        feat["mid_int_std"]    = safe_std(gap_logs)
        feat["mid_sess_age"]   = max(0.0, (ts[i] - ts[0]) / 1000.0)
        feat["mid_cat_ent"]    = entropy_ratio(prefix_cats)
        feat["mid_cat_top1"]   = top1_ratio(prefix_cats)
        feat["mid_cat_switch_r"] = switch_ratio(prefix_cats)
        feat["mid_cat_uniq_r"] = uniq_ratio(prefix_cats)
        feat["mid_item_uniq_r"]= uniq_ratio(prefix_items)
        feat["mid_repeat_r"]   = 1.0 - feat["mid_item_uniq_r"]
        feat["mid_novel_r"]    = feat["mid_item_uniq_r"]
        feat["mid_max_run_i"]  = max_run_ratio(prefix_items)
        feat["mid_pop_mean"]   = safe_mean(prefix_pops)
        feat["mid_pop_std"]    = safe_std(prefix_pops)
        feat["mid_pop_ent"]    = entropy_ratio([str(b) for b in prefix_bins])
        feat["mid_pop_trend"]  = (prefix_pops[-1] - sum(prefix_pops[:-1]) / len(prefix_pops[:-1])) if len(prefix_pops) > 1 else 0.0

        # ── micro features ───────────────────────────────────────────────────
        win      = items[max(0, i - micro_window):i]
        win_cats = cats[max(0,  i - micro_window):i]
        win_pops = pops[max(0,  i - micro_window):i]
        win_bins = bins[max(0,  i - micro_window):i]
        win_ts   = ts[max(0,    i - micro_window):i]
        w = len(win)

        feat["mic_valid_r"] = w / float(max(1, micro_window))
        if w <= 0:
            for k in ("mic_last_gap", "mic_gap_mean", "mic_gap_delta_vs_mid",
                      "mic_last_pop", "mic_suffix_pop_std", "mic_pop_delta_vs_mid"):
                feat[k] = None
            for k in ("mic_cat_switch_now", "mic_last_cat_mismatch_r",
                      "mic_suffix_cat_ent", "mic_suffix_cat_uniq_r",
                      "mic_is_recons", "mic_suffix_recons_r",
                      "mic_suffix_uniq_i", "mic_suffix_max_run_i", "mic_suffix_pop_ent"):
                feat[k] = 0.0
        else:
            feat["mic_last_gap"] = max(0.0, (ts[i] - win_ts[-1]) / 1000.0)
            loc_ts  = win_ts + [ts[i]]
            loc_gaps = [math.log1p(max(0.0, (loc_ts[j] - loc_ts[j-1]) / 1000.0)) for j in range(1, len(loc_ts))]
            gm = safe_mean(loc_gaps)
            feat["mic_gap_mean"] = gm
            feat["mic_gap_delta_vs_mid"] = (gm - float(feat["mid_int_mean"])) if (gm is not None and feat["mid_int_mean"] is not None) else None
            cur_cat = cats[i]
            feat["mic_cat_switch_now"]       = 1.0 if cur_cat != win_cats[-1] else 0.0
            feat["mic_last_cat_mismatch_r"]  = sum(1 for c in win_cats if c != cur_cat) / float(w)
            feat["mic_suffix_cat_ent"]       = entropy_ratio(win_cats)
            feat["mic_suffix_cat_uniq_r"]    = uniq_ratio(win_cats)
            feat["mic_is_recons"]            = 1.0 if items[i] in set(win) else 0.0
            feat["mic_suffix_recons_r"]      = 1.0 - uniq_ratio(win)
            feat["mic_suffix_uniq_i"]        = uniq_ratio(win)
            feat["mic_suffix_max_run_i"]     = max_run_ratio(win)
            feat["mic_last_pop"]             = pops[i]
            feat["mic_suffix_pop_std"]       = safe_std(win_pops)
            feat["mic_suffix_pop_ent"]       = entropy_ratio([str(b) for b in win_bins])
            feat["mic_pop_delta_vs_mid"]     = (pops[i] - float(feat["mid_pop_mean"])) if feat["mid_pop_mean"] is not None else None

        rows.append(feat)

    # session_constant_last: overwrite all mid features with last row's values
    if rows:
        last_mid = {k: rows[-1].get(k) for k in MID_FEATURES}
        for r in rows:
            for k in MID_FEATURES:
                r[k] = last_mid[k]

    return rows


# ---------------------------------------------------------------------------
# Normalization stat fitting
# ---------------------------------------------------------------------------

def fit_norm_stats(
    cont_values: Dict[str, List[float]],
) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    for name in CONTINUOUS_FEATURES:
        vals = sorted(cont_values.get(name, []))
        if not vals:
            stats[name] = {
                "type": "continuous", "transform": "log1p_winsorize_z_phi",
                "mean": 0.0, "std": 1.0, "q01": 0.0, "q99": 0.0, "missing_default": 0.5,
            }
            continue
        q01 = quantile_sorted(vals, 0.01)
        q99 = quantile_sorted(vals, 0.99)
        mu  = safe_mean(vals) or 0.0
        sd  = safe_std(vals)  or 1.0
        sd  = max(sd, 1e-12)
        stats[name] = {
            "type": "continuous", "transform": "log1p_winsorize_z_phi",
            "mean": float(mu), "std": float(sd),
            "q01": float(q01), "q99": float(q99), "missing_default": 0.5,
        }
    return stats


def normalize(name: str, value: float | None, stats: Dict[str, dict]) -> float:
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


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------

def _load_module(filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_split_module():
    return _load_module("split_feature_v3_by_session_time.py")


def load_v4_module():
    return _load_module("build_feature_v4_from_v3.py")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dataset    = str(args.dataset)
    source_dir = Path(args.basic_root) / dataset
    out_dir    = Path(args.output_root) / dataset
    inter_in   = source_dir / f"{dataset}.inter"
    item_in    = source_dir / f"{dataset}.item"
    inter_out  = out_dir / f"{dataset}.inter"
    item_out   = out_dir / f"{dataset}.item"

    if not inter_in.exists() or not item_in.exists():
        raise SystemExit(f"Missing source files in: {source_dir}")
    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] loading basic dataset: {source_dir}")
    by_sid, sessions_by_user, session_order, item_cat = load_basic_data(inter_in, item_in)
    total_sessions = len(session_order)
    total_rows     = sum(len(by_sid[sid]) for sid in session_order)
    print(f"  sessions={total_sessions:,}  rows={total_rows:,}  items={len(item_cat):,}")

    fit_n   = max(1, int(math.floor(total_sessions * float(args.fit_session_ratio))))
    fit_sids = set(session_order[:fit_n])
    fit_last_start = by_sid[session_order[fit_n - 1]][0].ts_ms if fit_n > 0 else None
    print(f"  fit_sessions={fit_n}  (ratio={args.fit_session_ratio})")

    print("[2/7] building popularity scores")
    pop_score, pop_bin = build_popularity(by_sid, fit_sids, int(args.n_pop_bins))
    print(f"  pop items in fit={len(pop_score):,}")

    print("[3/7] computing session summaries + macro context")
    summaries      = summarize_sessions(by_sid, item_cat, pop_score, pop_bin, int(args.n_pop_bins))
    macro_by_sid   = build_macro_by_session(sessions_by_user, summaries)

    print("[4/7] fitting normalization stats (fit sessions only)")
    cont_values: Dict[str, List[float]] = defaultdict(list)
    for sid in fit_sids:
        raw_rows = compute_session_rows(
            events=by_sid[sid], macro=macro_by_sid[sid],
            item_cat=item_cat, pop_score=pop_score, pop_bin=pop_bin,
            micro_window=int(args.micro_window), mid_valid_cap=int(args.mid_valid_cap),
            n_pop_bins=int(args.n_pop_bins),
        )
        for r in raw_rows:
            for name in CONTINUOUS_FEATURES:
                v = r.get(name)
                if v is not None:
                    cont_values[name].append(math.log1p(max(0.0, float(v))))
    norm_stats = fit_norm_stats(cont_values)

    print("[5/7] writing featured inter file")
    with inter_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"]
            + [f"{n}:float" for n in ALL_FEATURES]
        )
        for sid in session_order:
            raw_rows = compute_session_rows(
                events=by_sid[sid], macro=macro_by_sid[sid],
                item_cat=item_cat, pop_score=pop_score, pop_bin=pop_bin,
                micro_window=int(args.micro_window), mid_valid_cap=int(args.mid_valid_cap),
                n_pop_bins=int(args.n_pop_bins),
            )
            for e, raw in zip(by_sid[sid], raw_rows):
                vals = [normalize(n, raw.get(n), norm_stats) for n in ALL_FEATURES]
                writer.writerow([sid, e.item, int(e.ts_ms), e.user] + vals)

    shutil.copy2(item_in, item_out)

    print("[6/7] writing feature_meta_v3.json")
    norm_meta: Dict[str, dict] = {}
    for name in ALL_FEATURES:
        if name in CONTINUOUS_FEATURES:
            norm_meta[name] = dict(norm_stats[name])
        elif name in PAIR_FEATURES:
            norm_meta[name] = {
                "type": "pair_bounded", "transform": "0.5 + 0.5 * tanh(delta)",
                "missing_default": DEFAULT_VALUE.get(name, 0.5),
            }
        else:
            norm_meta[name] = {
                "type": "bounded", "transform": "clip01",
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
            "applied": True, "dataset": dataset,
            "mid_columns": len(MID_FEATURES),
            "rows": total_rows, "sessions": total_sessions,
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
            "gap_last": "neutral 0.5 when no prior session",
            "average_features": "computed from available prior sessions",
            "pairwise_and_trend": "neutral 0.5 when insufficient context",
        },
        "normalization": {
            "bounded": "clip01",
            "continuous": "log1p -> winsorize(p01,p99) -> zscore -> phi",
            "pair_bounded": "0.5 + 0.5 * tanh(delta)",
            "continuous_missing_default": 0.5,
        },
        "normalization_stats": norm_meta,
        "timestamp_unit": "milliseconds",
        "source_root": str(source_dir),
        "output_root": str(out_dir),
    }
    (out_dir / "feature_meta_v3.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("[7/7] splitting train/valid/test")
    split_mod = load_split_module()
    ratios = split_mod.parse_ratios(str(args.split_ratios))
    split_summary = split_mod.process_dataset(
        processed_root=Path(args.output_root),
        dataset=dataset,
        ratios=ratios,
        strategy=str(args.split_strategy),
        overwrite=bool(args.overwrite),
        dry_run=False,
    )

    print("[7b/7] applying v4 re-split (valid+test 50:50 by session start time)")
    v4_mod = load_v4_module()
    v4_summary = v4_mod.process_dataset(
        src_root=Path(args.output_root),
        dst_root=Path(args.output_root),
        dataset=dataset,
        overwrite=bool(args.overwrite),
    )

    build_summary = {
        "dataset": dataset,
        "source": str(source_dir),
        "target": str(out_dir),
        "rows": total_rows, "sessions": total_sessions, "features": len(ALL_FEATURES),
        "split_summary": split_summary,
        "v4_split_summary": v4_summary,
    }
    (out_dir / f"{dataset}.build_summary.json").write_text(
        json.dumps(build_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"done → {out_dir}")
    print(f"  sessions={total_sessions:,}  rows={total_rows:,}  features={len(ALL_FEATURES)}")


if __name__ == "__main__":
    main()
