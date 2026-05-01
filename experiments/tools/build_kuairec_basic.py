#!/usr/bin/env python3
"""Build KuaiRec basic dataset from raw big_matrix.csv.

Pipeline (in order):
  1. Sessionize all rows with 30-min inactivity gap.
  2. Iterative k-core until convergence:
       - remove sessions with len < min_session_len  (default 5)
       - remove items with session_freq < min_item_freq  (default 3)
  3. watch_ratio filter: keep rows where
       watch_ratio > 1.0  OR  watch_ratio > per-user mean watch_ratio
     (applied on k-core stabilised data, then re-run one more k-core pass)
  4. Write basic inter + item files.

Output:
  Datasets/processed/basic/KuaiRec/KuaiRec.inter
  Datasets/processed/basic/KuaiRec/KuaiRec.item
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIG_MATRIX    = REPO_ROOT / "Datasets" / "raw" / "KuaiRec" / "KuaiRec 2.0" / "data" / "big_matrix.csv"
DEFAULT_ITEM_CATS     = REPO_ROOT / "Datasets" / "raw" / "KuaiRec" / "KuaiRec 2.0" / "data" / "item_categories.csv"
DEFAULT_CAPTION_CATS  = REPO_ROOT / "Datasets" / "raw" / "KuaiRec" / "KuaiRec 2.0" / "data" / "kuairec_caption_category.csv"
DEFAULT_OUT_ROOT      = REPO_ROOT / "Datasets" / "processed" / "basic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--big-matrix",    type=Path, default=DEFAULT_BIG_MATRIX)
    p.add_argument("--item-cats",     type=Path, default=DEFAULT_ITEM_CATS)
    p.add_argument("--caption-cats",  type=Path, default=DEFAULT_CAPTION_CATS)
    p.add_argument("--output-root",   type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dataset",       type=str,  default="KuaiRec")
    p.add_argument("--session-gap",   type=int,  default=1800,
                   help="Inactivity gap in seconds to split sessions (default: 1800 = 30 min)")
    p.add_argument("--min-session-len", type=int, default=5)
    p.add_argument("--min-item-freq",   type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Load raw interactions
# ---------------------------------------------------------------------------

def load_raw(big_matrix: Path) -> List[Tuple[int, int, float, float]]:
    """Return list of (user_id, video_id, timestamp, watch_ratio)."""
    rows: List[Tuple[int, int, float, float]] = []
    with big_matrix.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                uid  = int(row["user_id"])
                vid  = int(row["video_id"])
                ts   = float(row["timestamp"])
                wr   = float(row["watch_ratio"])
            except (KeyError, ValueError):
                continue
            rows.append((uid, vid, ts, wr))
    return rows


# ---------------------------------------------------------------------------
# Step 2: Sessionize
# ---------------------------------------------------------------------------

def sessionize(
    rows: List[Tuple[int, int, float, float]],
    gap_sec: int,
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Return {session_id: [(video_id, timestamp, watch_ratio), ...]}."""
    by_user: Dict[int, List[Tuple[float, int, float]]] = defaultdict(list)
    for uid, vid, ts, wr in rows:
        by_user[uid].append((ts, vid, wr))

    sessions: Dict[str, List[Tuple[int, float, float]]] = {}
    for uid, events in by_user.items():
        events.sort(key=lambda x: x[0])
        sess_idx = 0
        prev_ts: float | None = None
        curr: List[Tuple[int, float, float]] = []
        for ts, vid, wr in events:
            if prev_ts is not None and (ts - prev_ts) > gap_sec:
                if curr:
                    sid = f"{uid}_s{sess_idx}"
                    sessions[sid] = curr
                    sess_idx += 1
                curr = []
            curr.append((vid, ts, wr))
            prev_ts = ts
        if curr:
            sessions[f"{uid}_s{sess_idx}"] = curr
    return sessions


# ---------------------------------------------------------------------------
# Step 3: Iterative k-core
# ---------------------------------------------------------------------------

def iterative_kcore(
    sessions: Dict[str, List[Tuple[int, float, float]]],
    min_session_len: int,
    min_item_freq: int,
) -> tuple[Dict[str, List[Tuple[int, float, float]]], dict]:
    history = []
    work = dict(sessions)
    while True:
        prev_n = sum(len(v) for v in work.values())

        # drop short sessions
        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}

        # count item frequency (by number of sessions containing it)
        item_sess_freq: Counter = Counter()
        for evts in work.values():
            for vid, _ts, _wr in evts:
                item_sess_freq[vid] += 1

        # drop rare items
        rare = {vid for vid, cnt in item_sess_freq.items() if cnt < min_item_freq}
        if rare:
            new_work: Dict[str, List[Tuple[int, float, float]]] = {}
            for sid, evts in work.items():
                kept = [(vid, ts, wr) for vid, ts, wr in evts if vid not in rare]
                if kept:
                    new_work[sid] = kept
            work = new_work

        # drop short sessions again after item removal
        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}

        now_n = sum(len(v) for v in work.values())
        history.append({"rows_before": prev_n, "rows_after": now_n, "rare_items": len(rare)})
        if now_n == prev_n:
            break
    return work, {"iterations": len(history), "history": history}


# ---------------------------------------------------------------------------
# Step 4: watch_ratio filter
# ---------------------------------------------------------------------------

def filter_watch_ratio(
    sessions: Dict[str, List[Tuple[int, float, float]]],
) -> Dict[str, List[Tuple[int, float, float]]]:
    """Keep rows where watch_ratio > 1.0 OR watch_ratio > user mean watch_ratio.

    User mean is computed from all rows *within k-core stabilised sessions*.
    After filtering, sessions that drop below min_session_len will be caught
    by a subsequent k-core pass in the caller.
    """
    # compute per-user mean watch_ratio from all k-core rows
    user_wr_sum: Dict[int, float] = defaultdict(float)
    user_wr_cnt: Dict[int, int]   = defaultdict(int)
    for sid, evts in sessions.items():
        uid = int(sid.split("_s")[0])
        for _vid, _ts, wr in evts:
            user_wr_sum[uid] += wr
            user_wr_cnt[uid] += 1
    user_mean_wr: Dict[int, float] = {
        uid: user_wr_sum[uid] / user_wr_cnt[uid]
        for uid in user_wr_sum
    }

    filtered: Dict[str, List[Tuple[int, float, float]]] = {}
    for sid, evts in sessions.items():
        uid = int(sid.split("_s")[0])
        mean_wr = user_mean_wr.get(uid, 0.0)
        kept = [
            (vid, ts, wr) for vid, ts, wr in evts
            if wr > 1.0 or wr > mean_wr
        ]
        if kept:
            filtered[sid] = kept
    return filtered


# ---------------------------------------------------------------------------
# Step 5: Load item categories
# ---------------------------------------------------------------------------

def load_item_categories(item_cats: Path, caption_cats: Path) -> Dict[int, str]:
    """Return {video_id: category_id_str}.

    Primary source: kuairec_caption_category.csv (first_level_category_id).
    Fallback: item_categories.csv feat[0] for items not in caption file or with id=-124.
    """
    # Fallback: item_categories.csv feat[0]
    fallback_map: Dict[int, str] = {}
    with item_cats.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vid  = int(row["video_id"])
                feat = ast.literal_eval(row["feat"])
                cat_id = int(feat[0])
                # remap placeholder IDs 0 and 30 to -1 so caption takes priority
                if cat_id not in (0, 30):
                    fallback_map[vid] = str(cat_id)
            except (KeyError, ValueError, IndexError, SyntaxError):
                continue

    # Primary: kuairec_caption_category.csv first_level_category_id
    caption_map: Dict[int, str] = {}
    with caption_cats.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vid    = int(row["video_id"])
                raw_cat = row.get("first_level_category_id") or ""
                cat_id = raw_cat.strip()
                if cat_id and cat_id != "-124":
                    caption_map[vid] = cat_id
            except (KeyError, ValueError):
                continue

    # caption_map takes priority over fallback_map
    return {**fallback_map, **caption_map}


# ---------------------------------------------------------------------------
# Step 6: Write output
# ---------------------------------------------------------------------------

def write_basic(
    *,
    out_dir: Path,
    dataset: str,
    sessions: Dict[str, List[Tuple[int, float, float]]],
    item_cat: Dict[int, str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all items; assign fallback category 0 if missing
    all_items: set[int] = set()
    for evts in sessions.values():
        for vid, _ts, _wr in evts:
            all_items.add(vid)

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        # sort sessions by start timestamp for stable ordering
        sid_order = sorted(sessions.keys(), key=lambda s: sessions[s][0][1])
        for sid in sid_order:
            uid = sid.split("_s")[0]
            for vid, ts, _wr in sessions[sid]:
                w.writerow([sid, vid, int(ts), uid])

    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for vid in sorted(all_items):
            w.writerow([vid, item_cat.get(vid, 0)])

    item_freq: Counter = Counter()
    for evts in sessions.values():
        for vid, _ts, _wr in evts:
            item_freq[vid] += 1

    return {
        "inter_path": str(inter_path),
        "item_path":  str(item_path),
        "rows":     sum(len(v) for v in sessions.values()),
        "sessions": len(sessions),
        "users":    len({s.split("_s")[0] for s in sessions}),
        "items":    len(all_items),
        "min_session_len": min((len(v) for v in sessions.values()), default=0),
        "min_item_freq":   min(item_freq.values()) if item_freq else 0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_root) / str(args.dataset)
    inter_out = out_dir / f"{args.dataset}.inter"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    print(f"[1/6] loading raw interactions: {args.big_matrix}")
    raw = load_raw(Path(args.big_matrix))
    print(f"  rows={len(raw):,}  users={len({r[0] for r in raw}):,}")

    print(f"[2/6] sessionizing (gap={args.session_gap}s = {args.session_gap//60} min)")
    sessions = sessionize(raw, gap_sec=int(args.session_gap))
    print(f"  raw sessions={len(sessions):,}")

    print(f"[3/6] iterative k-core (min_session_len={args.min_session_len}, min_item_freq={args.min_item_freq})")
    sessions, kcore_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    users_kc = len({s.split("_s")[0] for s in sessions})
    rows_kc  = sum(len(v) for v in sessions.values())
    print(f"  after k-core: sessions={len(sessions):,}  users={users_kc:,}  rows={rows_kc:,}  iters={kcore_stats['iterations']}")

    print("[4/6] watch_ratio filter (wr > 1.0 OR wr > user_mean_wr)")
    sessions = filter_watch_ratio(sessions)
    print(f"  after wr filter: sessions={len(sessions):,}  rows={sum(len(v) for v in sessions.values()):,}")

    print(f"[4b/6] re-running k-core after wr filter")
    sessions, kcore2_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    users_final = len({s.split("_s")[0] for s in sessions})
    rows_final  = sum(len(v) for v in sessions.values())
    print(f"  final: sessions={len(sessions):,}  users={users_final:,}  items={len({vid for evts in sessions.values() for vid,_,_ in evts}):,}  rows={rows_final:,}")

    print(f"[5/6] loading item categories: {args.item_cats} + {args.caption_cats}")
    item_cat = load_item_categories(Path(args.item_cats), Path(args.caption_cats))
    print(f"  categories loaded: {len(item_cat):,} videos  unique_cats={len(set(item_cat.values()))}")

    print("[6/6] writing basic dataset")
    write_stats = write_basic(
        out_dir=out_dir,
        dataset=str(args.dataset),
        sessions=sessions,
        item_cat=item_cat,
    )

    summary = {
        "dataset": str(args.dataset),
        "source": {
            "big_matrix": str(Path(args.big_matrix).resolve()),
            "item_cats":  str(Path(args.item_cats).resolve()),
        },
        "params": {
            "session_gap_sec":   int(args.session_gap),
            "min_session_len":   int(args.min_session_len),
            "min_item_freq":     int(args.min_item_freq),
            "wr_filter":         "wr > 1.0 OR wr > user_mean_wr (computed from k-core data)",
        },
        "kcore_stats":  kcore_stats,
        "kcore2_stats": kcore2_stats,
        "write_stats":  write_stats,
    }
    summary_path = out_dir / f"{args.dataset}.basic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done → {summary_path}")
    print(f"  users={write_stats['users']:,}  sessions={write_stats['sessions']:,}  items={write_stats['items']:,}  rows={write_stats['rows']:,}")


if __name__ == "__main__":
    main()
