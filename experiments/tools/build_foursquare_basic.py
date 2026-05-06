#!/usr/bin/env python3
"""Build Foursquare-NYC basic dataset from raw dataset_TSMC2014_NYC.txt.

Pipeline:
  1. Parse TSV (no header): user_id, venue_id, venue_category_id, venue_category_name,
                             latitude, longitude, timezone_offset, UTC_time
  2. Parse UTC_time → Unix seconds.
  3. Sort each user's check-ins by timestamp, chunk into sessions of chunk_size (default 50).
     (Same strategy as MovieLens: fixed-size chronological chunks.)
  4. Iterative k-core until convergence:
       - remove sessions with len < min_session_len  (default 5)
       - remove items (venues) with session_freq < min_item_freq  (default 3)
  5. Map venue_category_name → 28 canonical category labels.
  6. Write basic inter + item files.

item     = venue_id
category = canonical label (Food_Specialty, Travel_Other, …)

Output:
  Datasets/processed/basic/foursquare/foursquare.inter
  Datasets/processed/basic/foursquare/foursquare.item
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT   = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "Datasets" / "raw" / "Foursquare" / "dataset_TSMC2014_NYC.txt"
DEFAULT_OUT = REPO_ROOT / "Datasets" / "processed" / "basic"

# 12-hour session gap (seconds)
DEFAULT_CHUNK_SIZE = 50

# Venue category name → canonical label
# Derived from grouping 251 raw Foursquare category names into 28 groups.
CATEGORY_MAP: Dict[str, str] = {}

_RAW_TO_LABEL: List[Tuple[str, str]] = [
    # Food
    ("restaurant", "Food_Casual"),
    ("diner", "Food_Casual"),
    ("deli", "Food_Casual"),
    ("sandwich", "Food_Casual"),
    ("pizza", "Food_Casual"),
    ("burger", "Food_Casual"),
    ("fast food", "Food_Casual"),
    ("food truck", "Food_Casual"),
    ("wings", "Food_Casual"),
    ("fried chicken", "Food_Casual"),
    ("hot dog", "Food_Casual"),
    ("taco", "Food_Casual"),
    ("bbq", "Food_Casual"),
    ("soup", "Food_Casual"),
    ("food & drink shop", "Food_Specialty"),
    ("food", "Food_Specialty"),
    ("grocery", "Food_Specialty"),
    ("bakery", "Food_Cafe_Bakery"),
    ("bagel", "Food_Cafe_Bakery"),
    ("cupcake", "Food_Cafe_Bakery"),
    ("coffee", "Food_Cafe_Bakery"),
    ("café", "Food_Cafe_Bakery"),
    ("cafe", "Food_Cafe_Bakery"),
    ("tea", "Food_Cafe_Bakery"),
    ("juice", "Food_Cafe_Bakery"),
    ("ice cream", "Food_Cafe_Bakery"),
    ("dessert", "Food_Cafe_Bakery"),
    ("donut", "Food_Cafe_Bakery"),
    ("sushi", "Food_International"),
    ("japanese", "Food_International"),
    ("chinese", "Food_International"),
    ("thai", "Food_International"),
    ("indian", "Food_International"),
    ("mexican", "Food_International"),
    ("french", "Food_International"),
    ("italian", "Food_International"),
    ("mediterranean", "Food_International"),
    ("greek", "Food_International"),
    ("middle eastern", "Food_International"),
    ("korean", "Food_International"),
    ("latin", "Food_International"),
    ("caribbean", "Food_International"),
    ("vietnamese", "Food_International"),
    ("bar", "Food_Nightlife"),
    ("pub", "Food_Nightlife"),
    ("lounge", "Food_Nightlife"),
    ("nightclub", "Food_Nightlife"),
    ("club", "Food_Nightlife"),
    ("brewery", "Food_Nightlife"),
    ("wine", "Food_Nightlife"),
    ("cocktail", "Food_Nightlife"),
    # Arts & Entertainment
    ("museum", "Arts_Museum"),
    ("gallery", "Arts_Gallery"),
    ("art", "Arts_Gallery"),
    ("theater", "Arts_Performance"),
    ("theatre", "Arts_Performance"),
    ("concert", "Arts_Performance"),
    ("music venue", "Arts_Performance"),
    ("comedy", "Arts_Performance"),
    ("cinema", "Arts_Performance"),
    ("movie", "Arts_Performance"),
    ("bowling", "Arts_Entertainment"),
    ("arcade", "Arts_Entertainment"),
    ("casino", "Arts_Entertainment"),
    ("entertainment", "Arts_Entertainment"),
    ("amusement", "Arts_Entertainment"),
    # Recreation
    ("park", "Recreation_Outdoor"),
    ("playground", "Recreation_Outdoor"),
    ("beach", "Recreation_Outdoor"),
    ("trail", "Recreation_Outdoor"),
    ("garden", "Recreation_Outdoor"),
    ("river", "Recreation_Outdoor"),
    ("lake", "Recreation_Outdoor"),
    ("outdoors", "Recreation_Outdoor"),
    ("gym", "Recreation_Sport"),
    ("fitness", "Recreation_Sport"),
    ("sport", "Recreation_Sport"),
    ("pool", "Recreation_Sport"),
    ("stadium", "Recreation_Sport"),
    ("court", "Recreation_Sport"),
    ("field", "Recreation_Sport"),
    ("golf", "Recreation_Sport"),
    ("yoga", "Recreation_Sport"),
    ("martial arts", "Recreation_Sport"),
    # Education
    ("university", "Education_Higher"),
    ("college", "Education_Higher"),
    ("school", "Education_Elementary"),
    ("library", "Education_Elementary"),
    ("elementary", "Education_Elementary"),
    ("high school", "Education_Elementary"),
    # Services
    ("auto", "Services_Auto"),
    ("car", "Services_Auto"),
    ("parking", "Services_Auto"),
    ("gas", "Services_Auto"),
    ("repair", "Services_Auto"),
    # Hotels & Travel
    ("hotel", "Hotel_Lodging"),
    ("hostel", "Hotel_Lodging"),
    ("motel", "Hotel_Lodging"),
    ("inn", "Hotel_Lodging"),
    ("bed and breakfast", "Hotel_Lodging"),
    ("resort", "Hotel_Lodging"),
    ("airport", "Travel_Other"),
    ("train station", "Travel_Other"),
    ("subway", "Travel_Other"),
    ("bus", "Travel_Other"),
    ("ferry", "Travel_Other"),
    ("taxi", "Travel_Other"),
    ("transport", "Travel_Other"),
    ("travel", "Travel_Other"),
    ("bridge", "Travel_Other"),
    ("tunnel", "Travel_Other"),
    # Buildings & Services
    ("office", "Building_General"),
    ("building", "Building_General"),
    ("home", "Building_General"),
    ("apartment", "Building_General"),
    ("residential", "Building_General"),
    ("neighborhood", "Building_General"),
    ("hospital", "Building_Institutional"),
    ("medical", "Building_Institutional"),
    ("pharmacy", "Building_Institutional"),
    ("clinic", "Building_Institutional"),
    ("government", "Building_Institutional"),
    ("post office", "Building_Institutional"),
    ("bank", "Building_Institutional"),
    ("church", "Religious"),
    ("mosque", "Religious"),
    ("temple", "Religious"),
    ("synagogue", "Religious"),
    ("religious", "Religious"),
    ("historic", "Historical_Cultural"),
    ("monument", "Historical_Cultural"),
    ("memorial", "Historical_Cultural"),
    ("landmark", "Historical_Cultural"),
]


def _build_category_map() -> Dict[str, str]:
    cache: Dict[str, str] = {}

    def lookup(name: str) -> str:
        n = name.lower().strip()
        if n in cache:
            return cache[n]
        for keyword, label in _RAW_TO_LABEL:
            if keyword in n:
                cache[n] = label
                return label
        cache[n] = "Building_Other"
        return "Building_Other"

    return lookup


_lookup_category = _build_category_map()


def parse_utc_time(s: str) -> int | None:
    """Parse Foursquare UTC_time string to Unix seconds."""
    s = s.strip()
    for fmt in ("%a %b %d %H:%M:%S +0000 %Y",):
        try:
            dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def load_raw(raw_path: Path) -> List[Tuple[str, str, str, int]]:
    """Return list of (user_id, venue_id, venue_category_name, timestamp_sec)."""
    rows: List[Tuple[str, str, str, int]] = []
    skipped = 0
    with raw_path.open("r", encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for cols in reader:
            if len(cols) < 8:
                skipped += 1
                continue
            user_id      = cols[0].strip()
            venue_id     = cols[1].strip()
            cat_name     = cols[3].strip()
            utc_str      = cols[7].strip()
            ts = parse_utc_time(utc_str)
            if ts is None:
                skipped += 1
                continue
            rows.append((user_id, venue_id, cat_name, ts))
    print(f"  parsed={len(rows):,}  skipped={skipped:,}")
    return rows


def build_chunk_sessions(
    rows: List[Tuple[str, str, str, int]],
    chunk_size: int,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """Sort each user's check-ins by timestamp, chunk into fixed-size sessions."""
    by_user: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for user_id, venue_id, cat_name, ts in rows:
        by_user[user_id].append((ts, venue_id, cat_name))

    sessions: Dict[str, List[Tuple[str, str, int]]] = {}
    for user_id, events in by_user.items():
        events.sort(key=lambda x: x[0])
        for chunk_idx in range(0, len(events), chunk_size):
            chunk = events[chunk_idx: chunk_idx + chunk_size]
            sid = f"{user_id}_c{chunk_idx // chunk_size}"
            sessions[sid] = [(vid, cat, ts) for ts, vid, cat in chunk]
    return sessions


def iterative_kcore(
    sessions: Dict[str, List[Tuple[str, str, int]]],
    min_session_len: int,
    min_item_freq: int,
) -> tuple[Dict[str, List[Tuple[str, str, int]]], dict]:
    history = []
    work = dict(sessions)
    while True:
        prev_n = sum(len(v) for v in work.values())
        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}
        freq: Counter = Counter()
        for evts in work.values():
            for vid, _cat, _ts in evts:
                freq[vid] += 1
        rare = {vid for vid, cnt in freq.items() if cnt < min_item_freq}
        if rare:
            new_work: Dict[str, List[Tuple[str, str, int]]] = {}
            for sid, evts in work.items():
                kept = [(v, c, t) for v, c, t in evts if v not in rare]
                if kept:
                    new_work[sid] = kept
            work = new_work
        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}
        now_n = sum(len(v) for v in work.values())
        history.append({"rows_before": prev_n, "rows_after": now_n, "rare_items": len(rare)})
        if now_n == prev_n:
            break
    return work, {"iterations": len(history), "history": history}


def write_basic(
    *,
    out_dir: Path,
    dataset: str,
    sessions: Dict[str, List[Tuple[str, str, int]]],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # venue → canonical category (most frequent raw cat name)
    venue_cat_cnt: Dict[str, Counter] = defaultdict(Counter)
    for evts in sessions.values():
        for vid, cat_name, _ts in evts:
            venue_cat_cnt[vid][cat_name] += 1
    venue_label: Dict[str, str] = {
        vid: _lookup_category(cnt.most_common(1)[0][0])
        for vid, cnt in venue_cat_cnt.items()
    }

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        sid_order = sorted(sessions.keys(), key=lambda s: sessions[s][0][2])
        for sid in sid_order:
            uid = sid.rsplit("_c", 1)[0]
            for vid, _cat, ts in sessions[sid]:
                w.writerow([sid, vid, ts, uid])

    all_items = {vid for evts in sessions.values() for vid, _, _ in evts}
    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for vid in sorted(all_items):
            w.writerow([vid, venue_label.get(vid, "Building_Other")])

    item_freq: Counter = Counter()
    for evts in sessions.values():
        for vid, _, _ in evts:
            item_freq[vid] += 1

    cats = set(venue_label.values())
    return {
        "inter_path":     str(inter_path),
        "item_path":      str(item_path),
        "rows":           sum(len(v) for v in sessions.values()),
        "sessions":       len(sessions),
        "users":          len({s.rsplit("_c", 1)[0] for s in sessions}),
        "items":          len(all_items),
        "categories":     len(cats),
        "min_session_len": min((len(v) for v in sessions.values()), default=0),
        "min_item_freq":   min(item_freq.values()) if item_freq else 0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw",             type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-root",     type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset",         type=str,  default="foursquare")
    p.add_argument("--chunk-size",      type=int,  default=DEFAULT_CHUNK_SIZE,
                   help="Fixed-size session chunk (chronological, default: 50)")
    p.add_argument("--min-session-len", type=int,  default=5)
    p.add_argument("--min-item-freq",   type=int,  default=3)
    p.add_argument("--overwrite",       action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir   = Path(args.output_root) / args.dataset
    inter_out = out_dir / f"{args.dataset}.inter"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    print(f"[1/5] loading raw: {args.raw}")
    raw = load_raw(Path(args.raw))
    print(f"  rows={len(raw):,}  users={len({r[0] for r in raw}):,}")

    print(f"[2/5] building chunk sessions (chunk_size={args.chunk_size})")
    sessions = build_chunk_sessions(raw, chunk_size=args.chunk_size)
    print(f"  raw sessions={len(sessions):,}")

    print(f"[3/5] iterative k-core (min_session_len={args.min_session_len}, min_item_freq={args.min_item_freq})")
    sessions, kcore_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    print(f"  after k-core: sessions={len(sessions):,}  users={len({s.rsplit('_c',1)[0] for s in sessions}):,}  rows={sum(len(v) for v in sessions.values()):,}  iters={kcore_stats['iterations']}")

    print("[4/5] mapping venue categories")
    print("[5/5] writing basic dataset")
    stats = write_basic(out_dir=out_dir, dataset=args.dataset, sessions=sessions)

    summary = {
        "dataset": args.dataset,
        "source": str(Path(args.raw).resolve()),
        "params": {
            "chunk_size":       args.chunk_size,
            "min_session_len":  args.min_session_len,
            "min_item_freq":    args.min_item_freq,
        },
        "kcore_stats": kcore_stats,
        "write_stats": stats,
    }
    sp = out_dir / f"{args.dataset}.basic_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done → {sp}")
    print(f"  users={stats['users']:,}  sessions={stats['sessions']:,}  items={stats['items']:,}  categories={stats['categories']}  rows={stats['rows']:,}")


if __name__ == "__main__":
    main()
