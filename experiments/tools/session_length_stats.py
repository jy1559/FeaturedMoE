#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def percentile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def read_session_lengths(inter_path: Path) -> list[int]:
    counts: Counter[str] = Counter()
    with inter_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        session_idx = None
        for index, column in enumerate(header):
            if column.split(":", 1)[0] == "session_id":
                session_idx = index
                break
        if session_idx is None:
            raise ValueError(f"session_id column not found: {inter_path}")
        for row in reader:
            if not row:
                continue
            counts[row[session_idx]] += 1
    return list(counts.values())


def summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    values = sorted(lengths)
    sessions = len(values)
    interactions = sum(values)
    mean = interactions / sessions if sessions else 0.0
    variance = sum((value - mean) ** 2 for value in values) / sessions if sessions else 0.0
    stddev = math.sqrt(variance)
    return {
        "sessions": sessions,
        "interactions": interactions,
        "avg_len": mean,
        "variance": variance,
        "stddev": stddev,
        "min_len": values[0] if values else 0,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max_len": values[-1] if values else 0,
    }


def find_inter_files(dataset_dir: Path) -> list[tuple[str, Path]]:
    dataset_name = dataset_dir.name
    candidates = [
        ("all", dataset_dir / f"{dataset_name}.inter"),
        ("train", dataset_dir / f"{dataset_name}.train.inter"),
        ("valid", dataset_dir / f"{dataset_name}.valid.inter"),
        ("test", dataset_dir / f"{dataset_name}.test.inter"),
    ]
    return [(split, path) for split, path in candidates if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute session length stats for processed datasets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/workspace/FeaturedMoE/Datasets/processed/feature_added_v4"),
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/workspace/FeaturedMoE/outputs/_tmp_session_length_stats.csv"),
        help="CSV file to write.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("/workspace/FeaturedMoE/outputs/_tmp_session_length_stats.md"),
        help="Markdown file to write.",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for dataset_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for split, inter_path in find_inter_files(dataset_dir):
            lengths = read_session_lengths(inter_path)
            stats = summarize_lengths(lengths)
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "split": split,
                    "sessions": str(stats["sessions"]),
                    "interactions": str(stats["interactions"]),
                    "avg_len": f"{stats['avg_len']:.4f}",
                    "variance": f"{stats['variance']:.4f}",
                    "stddev": f"{stats['stddev']:.4f}",
                    "min_len": str(stats["min_len"]),
                    "p25": f"{stats['p25']:.2f}",
                    "median": f"{stats['median']:.2f}",
                    "p75": f"{stats['p75']:.2f}",
                    "p90": f"{stats['p90']:.2f}",
                    "p95": f"{stats['p95']:.2f}",
                    "max_len": str(stats["max_len"]),
                    "path": str(inter_path),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "split",
            "sessions",
            "interactions",
            "avg_len",
            "variance",
            "stddev",
            "min_len",
            "p25",
            "median",
            "p75",
            "p90",
            "p95",
            "max_len",
            "path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Dataset | Split | Sessions | Interactions | Avg Len | Variance | Stddev | P50 | P90 | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['split']} | {row['sessions']} | {row['interactions']} | {row['avg_len']} | {row['variance']} | {row['stddev']} | {row['median']} | {row['p90']} | {row['p95']} | {row['max_len']} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()