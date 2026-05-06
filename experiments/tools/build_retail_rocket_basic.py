#!/usr/bin/env python3
"""Build Retail Rocket basic dataset from raw events.csv + item_properties + category_tree.csv.

Pipeline:
  1. Parse events.csv (timestamp_ms, visitorid, event, itemid, transactionid).
  2. Keep all event types (view / addtocart / transaction).
  3. Sessionize per visitor with 30-min inactivity gap (timestamps are Unix ms).
  4. Iterative k-core until convergence:
       - remove sessions with len < min_session_len  (default 5)
       - remove items with session_freq < min_item_freq  (default 3)
  5. Map item → root category via category_tree.csv (walk to root, re-index 0-based).
  6. Write basic inter + item files.

item     = itemid
category = root-level category index (0-based integer)

Output:
  Datasets/processed/basic/retail_rocket/retail_rocket.inter
  Datasets/processed/basic/retail_rocket/retail_rocket.item
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT           = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS      = REPO_ROOT / "Datasets" / "raw" / "retail_rocket" / "events.csv"
DEFAULT_PROPS1      = REPO_ROOT / "Datasets" / "raw" / "retail_rocket" / "item_properties_part1.csv"
DEFAULT_PROPS2      = REPO_ROOT / "Datasets" / "raw" / "retail_rocket" / "item_properties_part2.csv"
DEFAULT_CAT_TREE    = REPO_ROOT / "Datasets" / "raw" / "retail_rocket" / "category_tree.csv"
DEFAULT_OUT         = REPO_ROOT / "Datasets" / "processed" / "basic"

DEFAULT_SESSION_GAP_MS = 30 * 60 * 1000  # 30 minutes in ms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events",          type=Path, default=DEFAULT_EVENTS)
    p.add_argument("--props1",          type=Path, default=DEFAULT_PROPS1)
    p.add_argument("--props2",          type=Path, default=DEFAULT_PROPS2)
    p.add_argument("--cat-tree",        type=Path, default=DEFAULT_CAT_TREE)
    p.add_argument("--output-root",     type=Path, default=DEFAULT_OUT)
    p.add_argument("--dataset",         type=str,  default="retail_rocket")
    p.add_argument("--session-gap-ms",  type=int,  default=DEFAULT_SESSION_GAP_MS,
                   help="Inactivity gap in ms (default: 1800000 = 30 min)")
    p.add_argument("--min-session-len", type=int,  default=5)
    p.add_argument("--min-item-freq",   type=int,  default=3)
    p.add_argument("--overwrite",       action="store_true")
    return p.parse_args()


def load_category_tree(cat_tree: Path) -> Dict[str, Optional[str]]:
    """Return {categoryid: parentid} (root nodes have parentid=None)."""
    parent: Dict[str, Optional[str]] = {}
    with cat_tree.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["categoryid"].strip()
            pid = row["parentid"].strip()
            parent[cid] = pid if pid else None
    return parent


def get_root(cid: str, parent: Dict[str, Optional[str]]) -> str:
    visited: set = set()
    while cid in parent and parent[cid] and cid not in visited:
        visited.add(cid)
        cid = parent[cid]
    return cid


def load_item_categories(
    props1: Path,
    props2: Optional[Path],
    parent: Dict[str, Optional[str]],
) -> Dict[str, str]:
    """Return {item_id: root_category_id} using most recent categoryid property."""
    item_cat_ts: Dict[str, Tuple[int, str]] = {}  # item_id → (ts, cat_id)

    def _process(path: Path) -> None:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("property", "").strip() != "categoryid":
                    continue
                iid = row["itemid"].strip()
                cat = row["value"].strip()
                ts  = int(row["timestamp"].strip())
                if iid not in item_cat_ts or ts > item_cat_ts[iid][0]:
                    item_cat_ts[iid] = (ts, cat)

    _process(props1)
    if props2 and props2.exists():
        _process(props2)

    return {iid: get_root(cat, parent) for iid, (_, cat) in item_cat_ts.items()}


def load_events(events_path: Path) -> List[Tuple[str, str, int]]:
    """Return list of (visitor_id, item_id, timestamp_ms)."""
    rows: List[Tuple[str, str, int]] = []
    skipped = 0
    with events_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                vis  = row["visitorid"].strip()
                item = row["itemid"].strip()
                ts   = int(row["timestamp"].strip())
            except (KeyError, ValueError):
                skipped += 1
                continue
            if vis and item:
                rows.append((vis, item, ts))
            else:
                skipped += 1
    print(f"  parsed={len(rows):,}  skipped={skipped:,}")
    return rows


def sessionize(
    rows: List[Tuple[str, str, int]],
    gap_ms: int,
) -> Dict[str, List[Tuple[str, int]]]:
    """Return {session_id: [(item_id, timestamp_ms), ...]}."""
    by_user: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for vis, item, ts in rows:
        by_user[vis].append((ts, item))

    sessions: Dict[str, List[Tuple[str, int]]] = {}
    for vis, events in by_user.items():
        events.sort(key=lambda x: x[0])
        sess_idx = 0
        prev_ts: int | None = None
        curr: List[Tuple[str, int]] = []
        for ts, item in events:
            if prev_ts is not None and (ts - prev_ts) > gap_ms:
                if curr:
                    sessions[f"{vis}_s{sess_idx}"] = curr
                    sess_idx += 1
                curr = []
            curr.append((item, ts))
            prev_ts = ts
        if curr:
            sessions[f"{vis}_s{sess_idx}"] = curr
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
            for item, _ts in evts:
                freq[item] += 1
        rare = {item for item, cnt in freq.items() if cnt < min_item_freq}
        if rare:
            new_work: Dict[str, List[Tuple[str, int]]] = {}
            for sid, evts in work.items():
                kept = [(i, t) for i, t in evts if i not in rare]
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
    item_root_cat: Dict[str, str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items = {item for evts in sessions.values() for item, _ in evts}

    # Build sorted root category → 0-based index
    root_cats = sorted(
        {item_root_cat.get(item, "0") for item in all_items},
        key=lambda x: int(x) if x.isdigit() else x,
    )
    cat_idx: Dict[str, str] = {cat: str(i) for i, cat in enumerate(root_cats)}

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        sid_order = sorted(sessions.keys(), key=lambda s: sessions[s][0][1])
        for sid in sid_order:
            uid = sid.rsplit("_s", 1)[0]
            for item, ts in sessions[sid]:
                w.writerow([sid, item, ts, uid])

    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for item in sorted(all_items, key=lambda x: int(x) if x.isdigit() else x):
            raw_root = item_root_cat.get(item, root_cats[0] if root_cats else "0")
            w.writerow([item, cat_idx.get(raw_root, "0")])

    item_freq: Counter = Counter()
    for evts in sessions.values():
        for item, _ in evts:
            item_freq[item] += 1

    return {
        "inter_path":     str(inter_path),
        "item_path":      str(item_path),
        "rows":           sum(len(v) for v in sessions.values()),
        "sessions":       len(sessions),
        "users":          len({s.rsplit("_s", 1)[0] for s in sessions}),
        "items":          len(all_items),
        "categories":     len(root_cats),
        "min_session_len": min((len(v) for v in sessions.values()), default=0),
        "min_item_freq":   min(item_freq.values()) if item_freq else 0,
    }


def main() -> None:
    args = parse_args()
    out_dir   = Path(args.output_root) / args.dataset
    inter_out = out_dir / f"{args.dataset}.inter"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    print(f"[1/6] loading category tree: {args.cat_tree}")
    parent = load_category_tree(Path(args.cat_tree))
    print(f"  categories in tree: {len(parent):,}")

    print(f"[2/6] loading item categories: {args.props1}")
    item_root_cat = load_item_categories(Path(args.props1), Path(args.props2) if args.props2 else None, parent)
    print(f"  items with category: {len(item_root_cat):,}")

    print(f"[3/6] loading events: {args.events}")
    raw = load_events(Path(args.events))
    print(f"  rows={len(raw):,}  users={len({r[0] for r in raw}):,}")

    print(f"[4/6] sessionizing (gap={args.session_gap_ms}ms = {args.session_gap_ms//60000}min)")
    sessions = sessionize(raw, gap_ms=args.session_gap_ms)
    print(f"  raw sessions={len(sessions):,}")

    print(f"[5/6] iterative k-core (min_session_len={args.min_session_len}, min_item_freq={args.min_item_freq})")
    sessions, kcore_stats = iterative_kcore(sessions, args.min_session_len, args.min_item_freq)
    print(f"  after k-core: sessions={len(sessions):,}  users={len({s.rsplit('_s',1)[0] for s in sessions}):,}  rows={sum(len(v) for v in sessions.values()):,}  iters={kcore_stats['iterations']}")

    print("[6/6] writing basic dataset")
    stats = write_basic(out_dir=out_dir, dataset=args.dataset, sessions=sessions, item_root_cat=item_root_cat)

    summary = {
        "dataset": args.dataset,
        "source": {
            "events":   str(Path(args.events).resolve()),
            "cat_tree": str(Path(args.cat_tree).resolve()),
        },
        "params": {
            "session_gap_ms":   args.session_gap_ms,
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
