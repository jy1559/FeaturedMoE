#!/usr/bin/env python3
"""Validate rebuilt beauty dataset artifacts for basic/v3/v4."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED = REPO_ROOT / "Datasets" / "processed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--report", type=Path, default=REPO_ROOT / "outputs" / "beauty_validation_report.json")
    return p.parse_args()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        row = next(reader, None)
    return list(row or [])


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def pick_col(header: list[str], plain: str) -> str:
    for c in header:
        if strip_type(c) == plain:
            return c
    raise KeyError(plain)


def load_sessions_and_items(inter_path: Path) -> tuple[dict[str, int], Counter]:
    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        sid_col = pick_col(header, "session_id")
        item_col = pick_col(header, "item_id")
        sess_len = Counter()
        item_freq = Counter()
        for row in reader:
            sid = str(row[sid_col])
            item = str(row[item_col])
            sess_len[sid] += 1
            item_freq[item] += 1
    return dict(sess_len), item_freq


def load_split_session_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        sid_col = pick_col(header, "session_id")
        return {str(row[sid_col]) for row in reader}


def load_item_categories(item_path: Path) -> dict[str, str]:
    out = {}
    with item_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        item_col = pick_col(header, "item_id")
        cat_col = pick_col(header, "category")
        for row in reader:
            out[str(row[item_col])] = str(row[cat_col])
    return out


def main() -> None:
    args = parse_args()
    ds = str(args.dataset)
    root = Path(args.processed_root)

    basic_dir = root / "basic" / ds
    v3_dir = root / "feature_added_v3" / ds
    v4_dir = root / "feature_added_v4" / ds

    basic_inter = basic_dir / f"{ds}.inter"
    basic_item = basic_dir / f"{ds}.item"

    v3_inter = v3_dir / f"{ds}.inter"
    v3_item = v3_dir / f"{ds}.item"
    v3_train = v3_dir / f"{ds}.train.inter"
    v3_valid = v3_dir / f"{ds}.valid.inter"
    v3_test = v3_dir / f"{ds}.test.inter"
    v3_meta = v3_dir / "feature_meta_v3.json"

    v4_train = v4_dir / f"{ds}.train.inter"
    v4_valid = v4_dir / f"{ds}.valid.inter"
    v4_test = v4_dir / f"{ds}.test.inter"

    files = {
        "basic_inter": basic_inter,
        "basic_item": basic_item,
        "v3_inter": v3_inter,
        "v3_item": v3_item,
        "v3_train": v3_train,
        "v3_valid": v3_valid,
        "v3_test": v3_test,
        "v3_meta": v3_meta,
        "v4_train": v4_train,
        "v4_valid": v4_valid,
        "v4_test": v4_test,
    }
    exists = {k: bool(p.exists()) for k, p in files.items()}

    basic_header = read_header(basic_inter) if basic_inter.exists() else []
    v3_header = read_header(v3_inter) if v3_inter.exists() else []
    v4_train_header = read_header(v4_train) if v4_train.exists() else []

    sess_len, item_freq = load_sessions_and_items(basic_inter) if basic_inter.exists() else ({}, Counter())

    v3_sids = {
        "train": load_split_session_ids(v3_train) if v3_train.exists() else set(),
        "valid": load_split_session_ids(v3_valid) if v3_valid.exists() else set(),
        "test": load_split_session_ids(v3_test) if v3_test.exists() else set(),
    }
    v4_sids = {
        "train": load_split_session_ids(v4_train) if v4_train.exists() else set(),
        "valid": load_split_session_ids(v4_valid) if v4_valid.exists() else set(),
        "test": load_split_session_ids(v4_test) if v4_test.exists() else set(),
    }

    v3_overlap = {
        "train_valid": len(v3_sids["train"] & v3_sids["valid"]),
        "train_test": len(v3_sids["train"] & v3_sids["test"]),
        "valid_test": len(v3_sids["valid"] & v3_sids["test"]),
    }
    v4_overlap = {
        "train_valid": len(v4_sids["train"] & v4_sids["valid"]),
        "train_test": len(v4_sids["train"] & v4_sids["test"]),
        "valid_test": len(v4_sids["valid"] & v4_sids["test"]),
    }

    feature_cols = [c for c in v3_header if strip_type(c).startswith(("mac", "mid", "mic"))]
    meta = json.loads(v3_meta.read_text(encoding="utf-8")) if v3_meta.exists() else {}

    item_cats = load_item_categories(basic_item) if basic_item.exists() else {}
    cat_counts = Counter(item_cats.values())
    interaction_cat_counts = Counter()
    if basic_inter.exists() and item_cats:
        with basic_inter.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            h = list(reader.fieldnames or [])
            item_col = pick_col(h, "item_id")
            for row in reader:
                cat = item_cats.get(str(row[item_col]), "Unknown")
                interaction_cat_counts[cat] += 1
    inter_total = sum(interaction_cat_counts.values())
    other_ratio = (interaction_cat_counts.get("Other", 0) / inter_total) if inter_total > 0 else 0.0

    report = {
        "dataset": ds,
        "exists": exists,
        "headers": {
            "basic": basic_header,
            "v3_ncols": len(v3_header),
            "v4_train_ncols": len(v4_train_header),
        },
        "cleaning_rules": {
            "min_session_len_observed": min(sess_len.values()) if sess_len else 0,
            "min_item_freq_observed": min(item_freq.values()) if item_freq else 0,
        },
        "split_overlap": {
            "v3": v3_overlap,
            "v4": v4_overlap,
        },
        "feature_v3": {
            "feature_column_count": len(feature_cols),
            "meta_macro_windows": meta.get("macro_windows"),
            "meta_micro_window": meta.get("micro_window"),
        },
        "category": {
            "cardinality": len(cat_counts),
            "other_ratio": other_ratio,
            "top10_interaction_categories": dict(interaction_cat_counts.most_common(10)),
        },
        "checks": {
            "all_required_files_exist": all(exists.values()),
            "basic_header_ok": basic_header == ["session_id:token", "item_id:token", "timestamp:float", "user_id:token"],
            "split_overlap_zero_v3": all(v == 0 for v in v3_overlap.values()),
            "split_overlap_zero_v4": all(v == 0 for v in v4_overlap.values()),
            "min_session_len_ge_5": (min(sess_len.values()) if sess_len else 0) >= 5,
            "min_item_freq_ge_3": (min(item_freq.values()) if item_freq else 0) >= 3,
            "feature_cols_eq_64": len(feature_cols) == 64,
            "meta_macro_windows_5_10": sorted(meta.get("macro_windows", [])) == [5, 10],
            "meta_micro_window_5": int(meta.get("micro_window", -1)) == 5,
            "category_not_too_many": len(cat_counts) < 100,
            "category_not_too_few": len(cat_counts) > 5,
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.report)


if __name__ == "__main__":
    main()
