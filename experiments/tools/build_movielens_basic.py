#!/usr/bin/env python3
"""Build MovieLens-1M basic dataset from raw ratings.dat + movies.dat.

Pipeline:
  1. Parse ratings.dat (UserID::MovieID::Rating::Timestamp).
  2. Filter ratings with rating >= min_rating (default 3).
  3. Sort each user's ratings by timestamp, chunk into sessions of chunk_size (default 50).
     (MovieLens ratings are spread over months/years with no natural session boundaries,
     so fixed-size chronological chunks are used to create session structure.)
  4. Iterative k-core until convergence:
       - remove sessions with len < min_session_len  (default 5)
       - remove items with session_freq < min_item_freq  (default 3)
  5. Write basic inter + item files.

item     = movie_id
category = first genre from movies.dat pipe-separated genre list

Output:
  Datasets/processed/basic/movielens1m/movielens1m.inter
  Datasets/processed/basic/movielens1m/movielens1m.item
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT        = Path(__file__).resolve().parents[2]
DEFAULT_RATINGS  = REPO_ROOT / "Datasets" / "raw" / "MovieLens" / "ratings.dat"
DEFAULT_MOVIES   = REPO_ROOT / "Datasets" / "raw" / "MovieLens" / "movies.dat"
DEFAULT_OUT      = REPO_ROOT / "Datasets" / "processed" / "basic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratings",         type=Path, default=DEFAULT_RATINGS)
    p.add_argument("--movies",          type=Path, default=DEFAULT_MOVIES)
    p.add_argument("--output-root",     type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset",         type=str,  default="movielens1m")
    p.add_argument("--min-rating",      type=float, default=3.0,
                   help="Minimum rating to keep (default: 3.0)")
    p.add_argument("--chunk-size",      type=int,  default=50,
                   help="Fixed-size session chunk (chronological, default: 50)")
    p.add_argument("--min-session-len", type=int,  default=5)
    p.add_argument("--min-item-freq",   type=int,  default=3)
    p.add_argument("--overwrite",       action="store_true")
    return p.parse_args()


def load_ratings(ratings_path: Path, min_rating: float) -> List[Tuple[str, str, int]]:
    """Return list of (user_id, movie_id, timestamp) filtered by min_rating."""
    rows: List[Tuple[str, str, int]] = []
    skipped = 0
    with ratings_path.open("r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("::")
            if len(parts) < 4:
                continue
            try:
                uid   = parts[0].strip()
                mid   = parts[1].strip()
                rating = float(parts[2].strip())
                ts    = int(parts[3].strip())
            except ValueError:
                skipped += 1
                continue
            if rating >= min_rating:
                rows.append((uid, mid, ts))
            else:
                skipped += 1
    print(f"  parsed={len(rows):,}  skipped(low_rating)={skipped:,}")
    return rows


def load_movies(movies_path: Path) -> Dict[str, str]:
    """Return {movie_id: first_genre} from movies.dat."""
    genre_map: Dict[str, str] = {}
    with movies_path.open("r", encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            mid    = parts[0].strip()
            genres = parts[2].strip().split("|")
            genre_map[mid] = genres[0] if genres and genres[0] != "(no genres listed)" else "Unknown"
    return genre_map


def build_chunk_sessions(
    rows: List[Tuple[str, str, int]],
    chunk_size: int,
) -> Dict[str, List[Tuple[str, int]]]:
    """Sort each user's ratings by timestamp, chunk into sessions of chunk_size.

    Session id format: user_{user_id}_c{chunk_idx}
    """
    by_user: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for uid, mid, ts in rows:
        by_user[uid].append((ts, mid))

    sessions: Dict[str, List[Tuple[str, int]]] = {}
    for uid, events in by_user.items():
        events.sort(key=lambda x: x[0])
        for chunk_idx in range(0, len(events), chunk_size):
            chunk = events[chunk_idx: chunk_idx + chunk_size]
            sid = f"user_{uid}_c{chunk_idx // chunk_size}"
            sessions[sid] = [(mid, ts) for ts, mid in chunk]
    return sessions


def iterative_kcore(
    sessions: Dict[str, List[Tuple[str, int]]],
    min_session_len: int,
    min_item_freq: int,
) -> tuple[Dict[str, List[Tuple[str, int]]], dict]:
    history = []
    work = dict(sessions)
    while True:
        prev_n = sum(len(v) for v in work.values())
        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}
        freq: Counter = Counter()
        for evts in work.values():
            for mid, _ts in evts:
                freq[mid] += 1
        rare = {mid for mid, cnt in freq.items() if cnt < min_item_freq}
        if rare:
            new_work: Dict[str, List[Tuple[str, int]]] = {}
            for sid, evts in work.items():
                kept = [(m, t) for m, t in evts if m not in rare]
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
    sessions: Dict[str, List[Tuple[str, int]]],
    genre_map: Dict[str, str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        sid_order = sorted(sessions.keys(), key=lambda s: sessions[s][0][1])
        for sid in sid_order:
            # user_id embedded in session id: user_{uid}_c{idx}
            uid = sid[5:].rsplit("_c", 1)[0]
            for mid, ts in sorted(sessions[sid], key=lambda x: x[1]):
                w.writerow([sid, mid, ts, uid])

    all_items = {mid for evts in sessions.values() for mid, _ in evts}
    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for mid in sorted(all_items, key=lambda x: int(x) if x.isdigit() else x):
            w.writerow([mid, genre_map.get(mid, "Unknown")])

    item_freq: Counter = Counter()
    for evts in sessions.values():
        for mid, _ in evts:
            item_freq[mid] += 1

    return {
        "inter_path":     str(inter_path),
        "item_path":      str(item_path),
        "rows":           sum(len(v) for v in sessions.values()),
        "sessions":       len(sessions),
        "users":          len({s[5:].rsplit("_c", 1)[0] for s in sessions}),
        "items":          len(all_items),
        "categories":     len({genre_map.get(m, "Unknown") for m in all_items}),
        "min_session_len": min((len(v) for v in sessions.values()), default=0),
        "min_item_freq":   min(item_freq.values()) if item_freq else 0,
    }


def main() -> None:
    args = parse_args()
    out_dir   = Path(args.output_root) / args.dataset
    inter_out = out_dir / f"{args.dataset}.inter"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    print(f"[1/5] loading ratings: {args.ratings}")
    raw = load_ratings(Path(args.ratings), min_rating=args.min_rating)
    print(f"  rows={len(raw):,}  users={len({r[0] for r in raw}):,}")

    print(f"[2/5] building chunk sessions (chunk_size={args.chunk_size})")
    sessions = build_chunk_sessions(raw, chunk_size=args.chunk_size)
    print(f"  raw sessions={len(sessions):,}")

    print(f"[3/5] iterative k-core (min_session_len={args.min_session_len}, min_item_freq={args.min_item_freq})")
    sessions, kcore_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    print(f"  after k-core: sessions={len(sessions):,}  rows={sum(len(v) for v in sessions.values()):,}  iters={kcore_stats['iterations']}")

    print(f"[4/5] loading movie genres: {args.movies}")
    genre_map = load_movies(Path(args.movies))
    print(f"  movies loaded: {len(genre_map):,}")

    print("[5/5] writing basic dataset")
    stats = write_basic(out_dir=out_dir, dataset=args.dataset, sessions=sessions, genre_map=genre_map)

    summary = {
        "dataset": args.dataset,
        "source": {"ratings": str(Path(args.ratings).resolve()), "movies": str(Path(args.movies).resolve())},
        "params": {
            "min_rating":       args.min_rating,
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
