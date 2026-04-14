#!/usr/bin/env python3
"""Build beauty basic dataset from raw Amazon Beauty dumps.

Stage 1 output format:
  Datasets/processed/basic/beauty/beauty.inter
  Datasets/processed/basic/beauty/beauty.item

Rules:
- Keep review if overall >= 3.0 (or missing overall).
- Sessionize per user with gap threshold (default 14 days).
- Iterative filtering until convergence:
  1) remove sessions with len < min_session_len (default 5)
  2) remove items with freq < min_item_freq (default 3)
  3) re-check session length
- Session id format: "{user_id}_c{idx}".
- Timestamps are written in milliseconds.

Category policy (distribution-aware):
- Prefer metadata category paths containing "Beauty".
- Base labels are level2::level3 where useful, otherwise level2.
- Sparse labels (< min_category_interactions) are merged to parent or Other.
- Automatically tune split depth to keep cardinality roughly 20-30.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWS = REPO_ROOT / "Datasets" / "raw" / "reviews_Beauty.json.gz"
DEFAULT_META = REPO_ROOT / "Datasets" / "raw" / "meta_Beauty.json.gz"
DEFAULT_OUT_ROOT = REPO_ROOT / "Datasets" / "processed" / "basic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--session-gap-days", type=float, default=14.0)
    p.add_argument("--min-session-len", type=int, default=5)
    p.add_argument("--min-item-freq", type=int, default=3)
    p.add_argument("--min-category-interactions", type=int, default=100)
    p.add_argument("--target-category-min", type=int, default=20)
    p.add_argument("--target-category-max", type=int, default=30)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


@dataclass
class Session:
    user_id: str
    events: List[Tuple[str, int]]  # (item_id, unix_sec)

    @property
    def start_ts(self) -> int:
        return self.events[0][1]


def _parse_json_or_literal(line: str) -> dict:
    text = line.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            obj = ast.literal_eval(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}


def _parse_ts(record: dict) -> int | None:
    raw = record.get("unixReviewTime")
    if raw is not None:
        try:
            return int(float(raw))
        except Exception:
            pass
    review_time = record.get("reviewTime")
    if review_time:
        for fmt in ("%m %d, %Y", "%m %d, %y"):
            try:
                return int(datetime.strptime(str(review_time), fmt).timestamp())
            except Exception:
                continue
    return None


def _clean_token(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\t", " ").replace("\n", " ")
    s = s.strip().replace(" ", "_")
    s = s.replace("/", "_").replace("\\", "_")
    return s if s else "Unknown"


def read_user_events(reviews_path: Path) -> tuple[Dict[str, List[Tuple[int, int, str]]], dict]:
    user_events: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    stats = {
        "raw_lines": 0,
        "parsed_rows": 0,
        "kept_rows": 0,
        "dropped_low_rating": 0,
        "dropped_no_user_or_item": 0,
        "dropped_no_timestamp": 0,
    }
    with gzip.open(reviews_path, "rt", encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            stats["raw_lines"] += 1
            row = _parse_json_or_literal(line)
            if not row:
                continue
            stats["parsed_rows"] += 1
            user = str(row.get("reviewerID") or "").strip()
            item = str(row.get("asin") or "").strip()
            if not user or not item:
                stats["dropped_no_user_or_item"] += 1
                continue

            overall = row.get("overall")
            if overall is not None:
                try:
                    if float(overall) < 3.0:
                        stats["dropped_low_rating"] += 1
                        continue
                except Exception:
                    pass

            ts = _parse_ts(row)
            if ts is None:
                stats["dropped_no_timestamp"] += 1
                continue

            user_events[user].append((int(ts), int(line_idx), item))
            stats["kept_rows"] += 1
    return user_events, stats


def build_sessions(user_events: Dict[str, List[Tuple[int, int, str]]], gap_days: float) -> List[Session]:
    gap_sec = int(gap_days * 86400)
    sessions: List[Session] = []
    for user, events in user_events.items():
        ordered = sorted(events, key=lambda x: (x[0], x[1]))
        curr: List[Tuple[str, int]] = []
        prev_ts: int | None = None
        for ts, _row_idx, item in ordered:
            if prev_ts is not None and (ts - prev_ts) > gap_sec:
                if curr:
                    sessions.append(Session(user_id=user, events=curr))
                curr = []
            curr.append((item, ts))
            prev_ts = ts
        if curr:
            sessions.append(Session(user_id=user, events=curr))
    return sessions


def filter_sessions_iterative(
    sessions: List[Session],
    min_session_len: int,
    min_item_freq: int,
) -> tuple[List[Session], dict]:
    iters = 0
    history = []
    work = sessions
    while True:
        iters += 1
        prev_n_sessions = len(work)
        prev_n_events = sum(len(s.events) for s in work)

        work = [s for s in work if len(s.events) >= min_session_len]

        item_counts = Counter()
        for s in work:
            for item, _ts in s.events:
                item_counts[item] += 1

        low_items = {item for item, cnt in item_counts.items() if cnt < min_item_freq}
        if low_items:
            new_work: List[Session] = []
            for s in work:
                filtered = [(item, ts) for item, ts in s.events if item not in low_items]
                if filtered:
                    new_work.append(Session(user_id=s.user_id, events=filtered))
            work = new_work

        work = [s for s in work if len(s.events) >= min_session_len]

        now_n_sessions = len(work)
        now_n_events = sum(len(s.events) for s in work)
        history.append(
            {
                "iter": iters,
                "sessions": now_n_sessions,
                "events": now_n_events,
                "low_item_count": len(low_items),
            }
        )
        if now_n_sessions == prev_n_sessions and now_n_events == prev_n_events:
            break

    return work, {"iterations": iters, "history": history}


def _extract_levels(paths: object) -> tuple[str, str]:
    if not isinstance(paths, list):
        return "Unknown", ""
    norm_paths: List[List[str]] = []
    for p in paths:
        if isinstance(p, list):
            vals = [str(x).strip() for x in p if str(x).strip()]
            if vals:
                norm_paths.append(vals)
    if not norm_paths:
        return "Unknown", ""

    chosen = None
    for p in norm_paths:
        if any(str(x).strip().lower() == "beauty" for x in p):
            chosen = p
            break
    if chosen is None:
        chosen = norm_paths[0]

    beauty_idx = -1
    for i, token in enumerate(chosen):
        if token.strip().lower() == "beauty":
            beauty_idx = i
            break

    if beauty_idx >= 0:
        l2 = chosen[beauty_idx + 1] if beauty_idx + 1 < len(chosen) else "Unknown"
        l3 = chosen[beauty_idx + 2] if beauty_idx + 2 < len(chosen) else ""
    else:
        l2 = chosen[1] if len(chosen) > 1 else chosen[0]
        l3 = chosen[2] if len(chosen) > 2 else ""

    return _clean_token(l2), _clean_token(l3)


def load_item_levels(meta_path: Path, target_items: set[str]) -> Dict[str, tuple[str, str]]:
    out: Dict[str, tuple[str, str]] = {}
    with gzip.open(meta_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = _parse_json_or_literal(line)
            if not row:
                continue
            asin = str(row.get("asin") or "").strip()
            if not asin or asin not in target_items:
                continue
            out[asin] = _extract_levels(row.get("categories"))
    return out


def _is_generic_level3(level3: str) -> bool:
    s = str(level3 or "").strip().lower()
    if not s:
        return True
    bad = {
        "other",
        "others",
        "general",
        "misc",
        "miscellaneous",
        "accessories",
        "kits",
        "sets",
        "unknown",
    }
    return s in bad


def choose_category_labels(
    *,
    item_interactions: Dict[str, int],
    item_levels: Dict[str, tuple[str, str]],
    min_category_interactions: int,
    target_min: int,
    target_max: int,
) -> tuple[Dict[str, str], dict]:
    l2_counts = Counter()
    l23_counts = Counter()
    for item, cnt in item_interactions.items():
        l2, l3 = item_levels.get(item, ("Unknown", ""))
        l2_counts[l2] += cnt
        if l3 and not _is_generic_level3(l3):
            l23_counts[(l2, l3)] += cnt

    candidate_thresholds = [500, 400, 300, 250, 200, 150, 120, 100, 80, 60, 40, 30]
    target_mid = (target_min + target_max) / 2.0

    def simulate(split_min: int) -> tuple[Dict[str, str], dict]:
        labels: Dict[str, str] = {}
        for item, cnt in item_interactions.items():
            l2, l3 = item_levels.get(item, ("Unknown", ""))
            if l3 and not _is_generic_level3(l3):
                sub_cnt = l23_counts.get((l2, l3), 0)
                if sub_cnt >= split_min and l2_counts[l2] >= split_min * 2:
                    labels[item] = f"{l2}::{l3}"
                else:
                    labels[item] = l2
            else:
                labels[item] = l2

        def count_labels(cur: Dict[str, str]) -> Counter:
            c = Counter()
            for it, lb in cur.items():
                c[lb] += item_interactions[it]
            return c

        for _ in range(2):
            counts = count_labels(labels)
            remap = {}
            for lb, cnt in counts.items():
                if cnt >= min_category_interactions:
                    continue
                if "::" in lb:
                    parent = lb.split("::", 1)[0]
                    if l2_counts.get(parent, 0) >= min_category_interactions:
                        remap[lb] = parent
                    else:
                        remap[lb] = "Other"
                else:
                    remap[lb] = "Other"
            if not remap:
                break
            for item, lb in list(labels.items()):
                if lb in remap:
                    labels[item] = remap[lb]

        while True:
            counts = count_labels(labels)
            non_other = [k for k in counts if k != "Other"]
            if len(non_other) <= target_max:
                break
            smallest = min(non_other, key=lambda k: (counts[k], k))
            if "::" in smallest:
                parent = smallest.split("::", 1)[0]
                merged_to = parent if l2_counts.get(parent, 0) >= min_category_interactions else "Other"
            else:
                merged_to = "Other"
            for item, lb in list(labels.items()):
                if lb == smallest:
                    labels[item] = merged_to

        counts = count_labels(labels)
        cardinality = len([k for k in counts.keys() if k])
        other_ratio = float(counts.get("Other", 0)) / max(1, sum(counts.values()))
        in_range = target_min <= cardinality <= target_max
        score = abs(cardinality - target_mid) + (0.0 if in_range else 10.0) + max(0.0, other_ratio - 0.35) * 30.0
        return labels, {
            "split_min": split_min,
            "category_cardinality": cardinality,
            "other_ratio": other_ratio,
            "in_target_range": in_range,
            "score": score,
            "label_counts": dict(counts),
        }

    best_labels = {}
    best_report = None
    for split_min in candidate_thresholds:
        labels, report = simulate(split_min)
        if best_report is None or report["score"] < best_report["score"]:
            best_labels = labels
            best_report = report

    assert best_report is not None
    return best_labels, best_report


def write_basic_dataset(
    *,
    out_dir: Path,
    dataset: str,
    sessions: List[Session],
    item_category: Dict[str, str],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions_by_user: Dict[str, List[Session]] = defaultdict(list)
    for s in sessions:
        sessions_by_user[s.user_id].append(s)
    for user in sessions_by_user:
        sessions_by_user[user].sort(key=lambda s: (s.start_ts, len(s.events)))

    inter_rows: List[Tuple[str, str, int, str]] = []
    for user, user_sessions in sessions_by_user.items():
        for idx, sess in enumerate(user_sessions):
            sid = f"{user}_c{idx}"
            for item, ts_sec in sess.events:
                inter_rows.append((sid, item, int(ts_sec) * 1000, user))

    inter_rows.sort(key=lambda x: (x[2], x[0], x[1]))

    inter_path = out_dir / f"{dataset}.inter"
    with inter_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["session_id:token", "item_id:token", "timestamp:float", "user_id:token"])
        for row in inter_rows:
            w.writerow(row)

    item_path = out_dir / f"{dataset}.item"
    with item_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["item_id:token", "category:token"])
        for item in sorted(item_category.keys()):
            w.writerow([item, item_category[item]])

    session_lengths = Counter()
    item_freq = Counter()
    for s in sessions:
        session_lengths[len(s.events)] += 1
        for item, _ts in s.events:
            item_freq[item] += 1

    return {
        "inter_path": str(inter_path),
        "item_path": str(item_path),
        "rows": len(inter_rows),
        "sessions": sum(len(v) for v in sessions_by_user.values()),
        "users": len(sessions_by_user),
        "items": len(item_category),
        "min_session_len": min((len(s.events) for s in sessions), default=0),
        "min_item_freq": min(item_freq.values()) if item_freq else 0,
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_root) / str(args.dataset)
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"Output dir is not empty (use --overwrite): {out_dir}")

    print(f"[1/6] reading reviews: {args.reviews}")
    user_events, read_stats = read_user_events(Path(args.reviews))
    print(
        f"  kept={read_stats['kept_rows']} users={len(user_events)} "
        f"dropped_low_rating={read_stats['dropped_low_rating']}"
    )

    print("[2/6] sessionizing")
    sessions0 = build_sessions(user_events, gap_days=float(args.session_gap_days))
    print(f"  sessions_before_filter={len(sessions0)}")

    print("[3/6] iterative filtering")
    sessions, filter_stats = filter_sessions_iterative(
        sessions0,
        min_session_len=int(args.min_session_len),
        min_item_freq=int(args.min_item_freq),
    )
    if not sessions:
        raise SystemExit("All sessions were removed by filtering rules.")
    item_interactions = Counter()
    for s in sessions:
        for item, _ts in s.events:
            item_interactions[item] += 1
    print(
        f"  sessions_after_filter={len(sessions)} items={len(item_interactions)} "
        f"events={sum(item_interactions.values())}"
    )

    print("[4/6] loading metadata categories")
    item_levels = load_item_levels(Path(args.meta), set(item_interactions.keys()))

    print("[5/6] adaptive category grouping")
    labels_by_item, cat_report = choose_category_labels(
        item_interactions=dict(item_interactions),
        item_levels=item_levels,
        min_category_interactions=int(args.min_category_interactions),
        target_min=int(args.target_category_min),
        target_max=int(args.target_category_max),
    )
    for item in item_interactions.keys():
        if item not in labels_by_item:
            labels_by_item[item] = "Unknown"

    counts = Counter(labels_by_item.values())
    print(
        f"  category_cardinality={cat_report['category_cardinality']} "
        f"other_ratio={cat_report['other_ratio']:.4f} split_min={cat_report['split_min']}"
    )

    print("[6/6] writing basic dataset")
    write_stats = write_basic_dataset(
        out_dir=out_dir,
        dataset=str(args.dataset),
        sessions=sessions,
        item_category=labels_by_item,
    )

    summary = {
        "dataset": str(args.dataset),
        "source": {
            "reviews": str(Path(args.reviews).resolve()),
            "meta": str(Path(args.meta).resolve()),
        },
        "params": {
            "session_gap_days": float(args.session_gap_days),
            "min_session_len": int(args.min_session_len),
            "min_item_freq": int(args.min_item_freq),
            "min_category_interactions": int(args.min_category_interactions),
            "target_category_min": int(args.target_category_min),
            "target_category_max": int(args.target_category_max),
        },
        "read_stats": read_stats,
        "filter_stats": filter_stats,
        "write_stats": write_stats,
        "category": {
            "selected_split_min": cat_report["split_min"],
            "category_cardinality": cat_report["category_cardinality"],
            "other_ratio": cat_report["other_ratio"],
            "label_counts": cat_report["label_counts"],
            "distinct_labels": sorted(counts.keys()),
        },
    }
    summary_path = out_dir / f"{args.dataset}.basic_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done: {summary_path}")


if __name__ == "__main__":
    main()
