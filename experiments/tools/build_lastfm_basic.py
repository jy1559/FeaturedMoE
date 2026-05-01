#!/usr/bin/env python3
"""Build LastFM basic dataset from raw userid-timestamp-artid-artname-traid-traname.tsv.

Pipeline:
  1. Parse TSV (no header): user_id, timestamp(ISO8601), artist_id, artist_name, track_id, track_name
  2. Drop rows with missing track_id.
  3. Sessionize per user with 30-min inactivity gap (timestamps in ms).
  4. Iterative k-core until convergence:
       - remove sessions with len < min_session_len  (default 5)
       - remove items with session_freq < min_item_freq  (default 3)
  5. Write basic inter + item files.

item  = track_id   (song level)
category = artist_id

Output:
  Datasets/processed/basic/lastfm/lastfm.inter
  Datasets/processed/basic/lastfm/lastfm.item
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT  = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "Datasets" / "raw" / "lastfm-dataset-1K" / "userid-timestamp-artid-artname-traid-traname.tsv"
DEFAULT_OUT = REPO_ROOT / "Datasets" / "processed" / "basic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",           type=Path, default=DEFAULT_RAW)
    p.add_argument("--output-root",   type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset",       type=str,  default="lastfm")
    p.add_argument("--session-gap-ms", type=int, default=1_800_000,
                   help="Inactivity gap in milliseconds to split sessions (default: 1800000 = 30 min)")
    p.add_argument("--min-session-len", type=int, default=5)
    p.add_argument("--min-item-freq",   type=int, default=3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Parse raw TSV
# ---------------------------------------------------------------------------

def parse_iso8601_ms(ts_str: str) -> int | None:
    """Parse ISO8601 timestamp string to Unix milliseconds."""
    ts_str = ts_str.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def load_raw(raw_path: Path) -> List[Tuple[str, str, str, int]]:
    """Return list of (user_id, track_id, artist_id, timestamp_ms).

    TSV columns (no header):
      0: user_id  1: timestamp  2: artist_id  3: artist_name  4: track_id  5: track_name
    """
    rows: List[Tuple[str, str, str, int]] = []
    skipped_no_track = 0
    skipped_no_ts    = 0
    csv.field_size_limit(10_000_000)
    with raw_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for cols in reader:
            if len(cols) < 5:
                continue
            user_id   = cols[0].strip()
            ts_str    = cols[1].strip()
            artist_id = cols[2].strip()
            track_id  = cols[4].strip() if len(cols) > 4 else ""

            if not track_id:
                skipped_no_track += 1
                continue
            ts_ms = parse_iso8601_ms(ts_str)
            if ts_ms is None:
                skipped_no_ts += 1
                continue
            rows.append((user_id, track_id, artist_id, ts_ms))

    print(f"  parsed={len(rows):,}  skipped_no_track={skipped_no_track:,}  skipped_no_ts={skipped_no_ts:,}")
    return rows


# ---------------------------------------------------------------------------
# Step 2: Sessionize
# ---------------------------------------------------------------------------

def sessionize(
    rows: List[Tuple[str, str, str, int]],
    gap_ms: int,
) -> Dict[str, List[Tuple[str, str, int]]]:
    """Return {session_id: [(track_id, artist_id, timestamp_ms), ...]}."""
    by_user: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
    for user_id, track_id, artist_id, ts_ms in rows:
        by_user[user_id].append((ts_ms, track_id, artist_id))

    sessions: Dict[str, List[Tuple[str, str, int]]] = {}
    global_idx = 0
    for user_id, events in by_user.items():
        events.sort(key=lambda x: x[0])
        sess_idx = 0
        prev_ts: int | None = None
        curr: List[Tuple[str, str, int]] = []
        for ts_ms, track_id, artist_id in events:
            if prev_ts is not None and (ts_ms - prev_ts) > gap_ms:
                if curr:
                    sid = f"{user_id}_s{sess_idx}"
                    sessions[sid] = curr
                    sess_idx += 1
                curr = []
            curr.append((track_id, artist_id, ts_ms))
            prev_ts = ts_ms
        if curr:
            sessions[f"{user_id}_s{sess_idx}"] = curr
        global_idx += 1
    return sessions


# ---------------------------------------------------------------------------
# Step 3: Iterative k-core
# ---------------------------------------------------------------------------

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

        item_sess_freq: Counter = Counter()
        for evts in work.values():
            for track_id, _artist, _ts in evts:
                item_sess_freq[track_id] += 1

        rare = {t for t, cnt in item_sess_freq.items() if cnt < min_item_freq}
        if rare:
            new_work: Dict[str, List[Tuple[str, str, int]]] = {}
            for sid, evts in work.items():
                kept = [(t, a, ts) for t, a, ts in evts if t not in rare]
                if kept:
                    new_work[sid] = kept
            work = new_work

        work = {sid: evts for sid, evts in work.items() if len(evts) >= min_session_len}

        now_n = sum(len(v) for v in work.values())
        history.append({"rows_before": prev_n, "rows_after": now_n, "rare_items": len(rare)})
        if now_n == prev_n:
            break
    return work, {"iterations": len(history), "history": history}


# ---------------------------------------------------------------------------
# Step 4: Collect item→artist mapping and write output
# ---------------------------------------------------------------------------

def build_item_artist(
    sessions: Dict[str, List[Tuple[str, str, int]]],
) -> Dict[str, str]:
    """Map track_id → most frequent artist_id across all events."""
    track_artist_cnt: Dict[str, Counter] = defaultdict(Counter)
    for evts in sessions.values():
        for track_id, artist_id, _ts in evts:
            track_artist_cnt[track_id][artist_id] += 1
    return {track: cnt.most_common(1)[0][0] for track, cnt in track_artist_cnt.items()}


def write_basic(
    *,
    out_dir: Path,
    dataset: str,
    sessions: Dict[str, List[Tuple[str, str, int]]],
    item_artist: Dict[str, str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        sid_order = sorted(sessions.keys(), key=lambda s: sessions[s][0][2])
        for sid in sid_order:
            uid = "_s".join(sid.split("_s")[:-1]) if "_s" in sid else sid
            for track_id, _artist, ts_ms in sessions[sid]:
                w.writerow([sid, track_id, ts_ms, uid])

    all_items = {t for evts in sessions.values() for t, _, _ in evts}
    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for track_id in sorted(all_items):
            w.writerow([track_id, item_artist.get(track_id, "Unknown")])

    item_freq: Counter = Counter()
    for evts in sessions.values():
        for t, _, _ in evts:
            item_freq[t] += 1

    return {
        "inter_path": str(inter_path),
        "item_path":  str(item_path),
        "rows":     sum(len(v) for v in sessions.values()),
        "sessions": len(sessions),
        "users":    len({s.rsplit("_s", 1)[0] for s in sessions}),
        "items":    len(all_items),
        "min_session_len": min((len(v) for v in sessions.values()), default=0),
        "min_item_freq":   min(item_freq.values()) if item_freq else 0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir  = Path(args.output_root) / str(args.dataset)
    inter_out = out_dir / f"{args.dataset}.inter"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    print(f"[1/5] loading raw TSV: {args.raw}")
    raw = load_raw(Path(args.raw))
    users_raw = len({r[0] for r in raw})
    print(f"  rows={len(raw):,}  users={users_raw:,}")

    print(f"[2/5] sessionizing (gap={args.session_gap_ms}ms = {args.session_gap_ms//60000} min)")
    sessions = sessionize(raw, gap_ms=int(args.session_gap_ms))
    print(f"  raw sessions={len(sessions):,}")

    print(f"[3/5] iterative k-core (min_session_len={args.min_session_len}, min_item_freq={args.min_item_freq})")
    sessions, kcore_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    users_kc = len({s.rsplit("_s", 1)[0] for s in sessions})
    print(f"  after k-core: sessions={len(sessions):,}  users={users_kc:,}  rows={sum(len(v) for v in sessions.values()):,}  iters={kcore_stats['iterations']}")

    print("[4/5] building item→artist mapping")
    item_artist = build_item_artist(sessions)
    print(f"  unique tracks={len(item_artist):,}")

    print("[5/5] writing basic dataset")
    write_stats = write_basic(
        out_dir=out_dir,
        dataset=str(args.dataset),
        sessions=sessions,
        item_artist=item_artist,
    )

    summary = {
        "dataset": str(args.dataset),
        "source": str(Path(args.raw).resolve()),
        "params": {
            "session_gap_ms":  int(args.session_gap_ms),
            "min_session_len": int(args.min_session_len),
            "min_item_freq":   int(args.min_item_freq),
        },
        "kcore_stats": kcore_stats,
        "write_stats": write_stats,
    }
    summary_path = out_dir / f"{args.dataset}.basic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done → {summary_path}")
    print(f"  users={write_stats['users']:,}  sessions={write_stats['sessions']:,}  items={write_stats['items']:,}  rows={write_stats['rows']:,}")


if __name__ == "__main__":
    main()
