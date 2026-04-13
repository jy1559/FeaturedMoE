#!/usr/bin/env python3
"""Build feature_added_v4 from feature_added_v3 with strict contiguous v/t re-split.

Rules:
- Keep train split as-is (copied from v3).
- Merge valid+test sessions from v3, sort sessions by session start time, split 50/50.
- For valid/test rows:
  - If item is unseen in train and not session target(last interaction), drop row.
  - If unseen item is session target, keep row.
- Save per-dataset summary json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_ROOT = REPO_ROOT / "Datasets" / "processed" / "feature_added_v3"
DEFAULT_DST_ROOT = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def pick_column(header: List[str], plain_name: str) -> str:
    for col in header:
        if strip_type(col) == plain_name:
            return col
    raise KeyError(f"Missing required column `{plain_name}`")


def to_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def discover_datasets(root: Path) -> List[str]:
    out: List[str] = []
    if not root.exists():
        return out
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        ds = p.name
        if (p / f"{ds}.train.inter").exists() and (p / f"{ds}.valid.inter").exists() and (p / f"{ds}.test.inter").exists():
            out.append(ds)
    return out


def parse_datasets(raw: str, src_root: Path) -> List[str]:
    if str(raw).strip():
        return [x.strip() for x in str(raw).split(",") if x.strip()]
    return discover_datasets(src_root)


@dataclass
class Event:
    row: Dict[str, str]
    ts: float
    source_order: int
    source_split: str


@dataclass
class SessionPack:
    sid: str
    events: List[Event]
    start_ts: float
    first_order: int


def read_inter_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not header:
        raise RuntimeError(f"Empty or invalid inter file: {path}")
    return header, rows


def write_inter_rows(path: Path, header: List[str], rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def _sort_events(events: List[Event]) -> List[Event]:
    return sorted(
        events,
        key=lambda e: (
            math.inf if math.isnan(e.ts) else e.ts,
            e.source_order,
        ),
    )


def _copy_metadata_files(src_dir: Path, dst_dir: Path, dataset: str) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    split_names = {
        f"{dataset}.train.inter",
        f"{dataset}.valid.inter",
        f"{dataset}.test.inter",
        f"{dataset}.v4_split_summary.json",
    }
    for src in src_dir.iterdir():
        if not src.is_file():
            continue
        if src.name in split_names:
            continue
        shutil.copy2(src, dst_dir / src.name)


def process_dataset(*, src_root: Path, dst_root: Path, dataset: str, overwrite: bool) -> dict:
    src_dir = src_root / dataset
    dst_dir = dst_root / dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    train_path = src_dir / f"{dataset}.train.inter"
    valid_path = src_dir / f"{dataset}.valid.inter"
    test_path = src_dir / f"{dataset}.test.inter"
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing split files for dataset={dataset} under {src_dir}")

    train_header, train_rows = read_inter_rows(train_path)
    valid_header, valid_rows = read_inter_rows(valid_path)
    test_header, test_rows = read_inter_rows(test_path)
    if train_header != valid_header or train_header != test_header:
        raise RuntimeError(f"Header mismatch among split files for dataset={dataset}")

    session_col = pick_column(train_header, "session_id")
    ts_col = pick_column(train_header, "timestamp")
    item_col = pick_column(train_header, "item_id")

    # Copy metadata and keep train split intact.
    _copy_metadata_files(src_dir, dst_dir, dataset)
    dst_train_path = dst_dir / f"{dataset}.train.inter"
    if dst_train_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {dst_train_path}")
    shutil.copy2(train_path, dst_train_path)

    train_items = set(str(row[item_col]) for row in train_rows)

    merged_rows: List[Event] = []
    global_order = 0
    for split_name, rows in (("valid", valid_rows), ("test", test_rows)):
        for row in rows:
            merged_rows.append(
                Event(
                    row=row,
                    ts=to_float(str(row[ts_col])),
                    source_order=global_order,
                    source_split=split_name,
                )
            )
            global_order += 1

    events_by_sid: Dict[str, List[Event]] = {}
    for ev in merged_rows:
        sid = str(ev.row[session_col])
        events_by_sid.setdefault(sid, []).append(ev)

    kept_sessions: List[SessionPack] = []
    dropped_non_target_unseen_rows = 0
    target_unseen_total = 0
    target_seen_total = 0
    source_valid_sessions = set(str(r[session_col]) for r in valid_rows)
    source_test_sessions = set(str(r[session_col]) for r in test_rows)

    for sid, raw_events in events_by_sid.items():
        ordered = _sort_events(raw_events)
        target_event = ordered[-1]
        filtered: List[Event] = []
        for idx, ev in enumerate(ordered):
            item = str(ev.row[item_col])
            is_target = (idx == len(ordered) - 1)
            unseen = item not in train_items
            if unseen and not is_target:
                dropped_non_target_unseen_rows += 1
                continue
            filtered.append(ev)
        if not filtered:
            # Should not happen because target is always kept.
            continue
        target_item = str(target_event.row[item_col])
        target_unseen = target_item not in train_items
        if target_unseen:
            target_unseen_total += 1
        else:
            target_seen_total += 1
        start_ts_vals = [ev.ts for ev in filtered if not math.isnan(ev.ts)]
        start_ts = min(start_ts_vals) if start_ts_vals else float("nan")
        first_order = min(ev.source_order for ev in filtered)
        kept_sessions.append(SessionPack(sid=sid, events=filtered, start_ts=start_ts, first_order=first_order))

    kept_sessions.sort(
        key=lambda s: (
            math.inf if math.isnan(s.start_ts) else s.start_ts,
            s.first_order,
            s.sid,
        )
    )

    total_sessions = len(kept_sessions)
    valid_n = total_sessions // 2
    test_n = total_sessions - valid_n
    valid_sessions = kept_sessions[:valid_n]
    test_sessions = kept_sessions[valid_n:]
    valid_sid_set = {s.sid for s in valid_sessions}
    test_sid_set = {s.sid for s in test_sessions}

    dst_valid_rows: List[Dict[str, str]] = []
    dst_test_rows: List[Dict[str, str]] = []
    valid_unseen_target = 0
    test_unseen_target = 0
    valid_seen_target = 0
    test_seen_target = 0

    for pack in valid_sessions:
        ordered = _sort_events(pack.events)
        dst_valid_rows.extend(ev.row for ev in ordered)
        target_item = str(ordered[-1].row[item_col])
        if target_item not in train_items:
            valid_unseen_target += 1
        else:
            valid_seen_target += 1
    for pack in test_sessions:
        ordered = _sort_events(pack.events)
        dst_test_rows.extend(ev.row for ev in ordered)
        target_item = str(ordered[-1].row[item_col])
        if target_item not in train_items:
            test_unseen_target += 1
        else:
            test_seen_target += 1

    dst_valid_path = dst_dir / f"{dataset}.valid.inter"
    dst_test_path = dst_dir / f"{dataset}.test.inter"
    if (dst_valid_path.exists() or dst_test_path.exists()) and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {dst_valid_path} / {dst_test_path}")
    write_inter_rows(dst_valid_path, train_header, dst_valid_rows)
    write_inter_rows(dst_test_path, train_header, dst_test_rows)

    summary = {
        "dataset": dataset,
        "source_root": str(src_root.resolve()),
        "target_root": str(dst_root.resolve()),
        "source": {
            "train_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "test_rows": len(test_rows),
            "valid_sessions": len(source_valid_sessions),
            "test_sessions": len(source_test_sessions),
        },
        "rule": {
            "train_unchanged": True,
            "valid_test_resplit": "strict_contiguous_50_50_by_session_start",
            "drop_non_target_unseen_in_valid_test": True,
            "keep_target_unseen_in_valid_test": True,
        },
        "processing": {
            "merged_valid_test_sessions": len(events_by_sid),
            "kept_sessions_total": total_sessions,
            "dropped_non_target_unseen_rows": int(dropped_non_target_unseen_rows),
            "target_unseen_total": int(target_unseen_total),
            "target_seen_total": int(target_seen_total),
        },
        "output": {
            "train_rows": len(train_rows),
            "valid_rows": len(dst_valid_rows),
            "test_rows": len(dst_test_rows),
            "valid_sessions": len(valid_sid_set),
            "test_sessions": len(test_sid_set),
            "valid_unseen_target_sessions": int(valid_unseen_target),
            "test_unseen_target_sessions": int(test_unseen_target),
            "valid_seen_target_sessions": int(valid_seen_target),
            "test_seen_target_sessions": int(test_seen_target),
        },
        "integrity": {
            "valid_test_session_overlap": int(len(valid_sid_set & test_sid_set)),
            "train_item_count": int(len(train_items)),
        },
    }
    summary_path = dst_dir / f"{dataset}.v4_split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SRC_ROOT)
    p.add_argument("--target-root", type=Path, default=DEFAULT_DST_ROOT)
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.source_root)
    dst_root = Path(args.target_root)
    datasets = parse_datasets(args.datasets, src_root)
    if not datasets:
        raise SystemExit(f"No datasets found under {src_root}")

    print(f"source_root={src_root}")
    print(f"target_root={dst_root}")
    print(f"datasets={datasets}")
    print(f"overwrite={bool(args.overwrite)}")
    all_summaries = []
    for ds in datasets:
        print(f"\n[dataset] {ds}")
        summary = process_dataset(
            src_root=src_root,
            dst_root=dst_root,
            dataset=ds,
            overwrite=bool(args.overwrite),
        )
        all_summaries.append(summary)
        print(
            "  -> done "
            f"valid_sessions={summary['output']['valid_sessions']} "
            f"test_sessions={summary['output']['test_sessions']} "
            f"dropped_non_target_unseen_rows={summary['processing']['dropped_non_target_unseen_rows']}"
        )

    print("\n[done] feature_added_v4 build completed")
    print(f"processed={len(all_summaries)} datasets")


if __name__ == "__main__":
    main()
