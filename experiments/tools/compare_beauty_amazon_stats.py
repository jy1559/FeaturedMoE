#!/usr/bin/env python3
"""Compare key stats between amazon_beauty and beauty datasets."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "Datasets" / "processed"
DATASETS = ("amazon_beauty", "beauty")


def strip_type(col: str) -> str:
    return col.split(":", 1)[0] if ":" in col else col


def pick_col(header: list[str], plain_name: str) -> str:
    for col in header:
        if strip_type(col) == plain_name:
            return col
    raise KeyError(f"Missing column: {plain_name}")


def basic_stats(dataset: str) -> dict:
    inter_path = PROCESSED_ROOT / "basic" / dataset / f"{dataset}.inter"
    item_path = PROCESSED_ROOT / "basic" / dataset / f"{dataset}.item"
    source_mode = "basic"
    if not inter_path.exists() or not item_path.exists():
        inter_path = PROCESSED_ROOT / "feature_added_v3" / dataset / f"{dataset}.inter"
        item_path = PROCESSED_ROOT / "feature_added_v3" / dataset / f"{dataset}.item"
        source_mode = "feature_added_v3"

    session_len = Counter()
    item_freq = Counter()
    users = set()

    with inter_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        sid_col = pick_col(header, "session_id")
        item_col = pick_col(header, "item_id")
        user_col = pick_col(header, "user_id")
        for row in reader:
            sid = str(row[sid_col])
            item = str(row[item_col])
            user = str(row[user_col])
            session_len[sid] += 1
            item_freq[item] += 1
            users.add(user)

    item_cat = {}
    with item_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        item_col = pick_col(header, "item_id")
        cat_col = pick_col(header, "category")
        for row in reader:
            item_cat[str(row[item_col])] = str(row[cat_col])

    cat_interactions = Counter()
    for item, cnt in item_freq.items():
        cat_interactions[item_cat.get(item, "Unknown")] += cnt

    lengths = sorted(session_len.values())
    p90 = lengths[int(0.9 * (len(lengths) - 1))] if lengths else 0
    return {
        "source_mode": source_mode,
        "rows": int(sum(lengths)),
        "sessions": int(len(lengths)),
        "users": int(len(users)),
        "items": int(len(item_freq)),
        "avg_session_len": (float(sum(lengths)) / len(lengths)) if lengths else 0.0,
        "p90_session_len": float(p90),
        "min_session_len": int(min(lengths) if lengths else 0),
        "min_item_freq": int(min(item_freq.values()) if item_freq else 0),
        "category_cardinality": int(len(cat_interactions)),
        "other_ratio": float(cat_interactions.get("Other", 0) / max(1, sum(cat_interactions.values()))),
        "top3_categories": cat_interactions.most_common(3),
    }


def split_stats(dataset: str, mode: str) -> dict:
    base = PROCESSED_ROOT / mode / dataset
    split_sids = {}
    out = {}
    for split in ("train", "valid", "test"):
        path = base / f"{dataset}.{split}.inter"
        rows = 0
        sids = set()
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            header = list(reader.fieldnames or [])
            sid_col = pick_col(header, "session_id")
            for row in reader:
                rows += 1
                sids.add(str(row[sid_col]))
        out[split] = {"rows": int(rows), "sessions": int(len(sids))}
        split_sids[split] = sids

    out["overlap"] = {
        "train_valid": int(len(split_sids["train"] & split_sids["valid"])),
        "train_test": int(len(split_sids["train"] & split_sids["test"])),
        "valid_test": int(len(split_sids["valid"] & split_sids["test"])),
    }
    return out


def main() -> None:
    result = {}
    for ds in DATASETS:
        result[ds] = {
            "basic": basic_stats(ds),
            "v3_split": split_stats(ds, "feature_added_v3"),
            "v4_split": split_stats(ds, "feature_added_v4"),
        }

    json_path = REPO_ROOT / "outputs" / "beauty_vs_amazon_stats.json"
    md_path = REPO_ROOT / "outputs" / "beauty_vs_amazon_stats.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("| metric | amazon_beauty | beauty |")
    md.append("|---|---:|---:|")
    for key in (
        "rows",
        "sessions",
        "users",
        "items",
        "avg_session_len",
        "p90_session_len",
        "min_session_len",
        "min_item_freq",
        "category_cardinality",
        "other_ratio",
    ):
        a = result["amazon_beauty"]["basic"][key]
        b = result["beauty"]["basic"][key]
        if isinstance(a, float):
            md.append(f"| basic.{key} | {a:.6g} | {b:.6g} |")
        else:
            md.append(f"| basic.{key} | {a} | {b} |")

    for mode in ("v3_split", "v4_split"):
        for split in ("train", "valid", "test"):
            for key in ("rows", "sessions"):
                a = result["amazon_beauty"][mode][split][key]
                b = result["beauty"][mode][split][key]
                md.append(f"| {mode}.{split}.{key} | {a} | {b} |")
        for key in ("train_valid", "train_test", "valid_test"):
            a = result["amazon_beauty"][mode]["overlap"][key]
            b = result["beauty"][mode]["overlap"][key]
            md.append(f"| {mode}.overlap.{key} | {a} | {b} |")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
