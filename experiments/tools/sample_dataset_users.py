#!/usr/bin/env python3
"""Sample a fraction of users from a basic dataset inter/item file.

Creates a sub-dataset by randomly sampling users and keeping only their sessions.
Item file is filtered to only items that appear in the sampled sessions.

Usage examples:
  # 3% of lastfm users (reproduces lastfm0.03)
  python sample_dataset_users.py \\
      --source-dir Datasets/processed/basic/lastfm \\
      --out-dir    Datasets/processed/basic/lastfm0.03 \\
      --dataset    lastfm0.03 \\
      --fraction   0.03 \\
      --seed       42

  # Explicit user count
  python sample_dataset_users.py \\
      --source-dir Datasets/processed/basic/lastfm \\
      --out-dir    Datasets/processed/basic/lastfm_100users \\
      --dataset    lastfm_100users \\
      --n-users    100 \\
      --seed       42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, required=True,
                   help="Directory containing {dataset}.inter and {dataset}.item")
    p.add_argument("--out-dir",    type=Path, required=True,
                   help="Output directory")
    p.add_argument("--dataset",    type=str,  required=True,
                   help="Source dataset name (for finding .inter/.item files)")
    p.add_argument("--out-dataset", type=str, default=None,
                   help="Output dataset name (default: same as --dataset)")
    p.add_argument("--fraction",   type=float, default=None,
                   help="Fraction of users to sample, e.g. 0.03 for 3%%")
    p.add_argument("--n-users",    type=int,   default=None,
                   help="Exact number of users to sample (overrides --fraction)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--overwrite",  action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_dir   = Path(args.source_dir)
    out_dir   = Path(args.out_dir)
    src_name  = args.dataset
    out_name  = args.out_dataset or src_name

    inter_src = src_dir / f"{src_name}.inter"
    item_src  = src_dir / f"{src_name}.item"
    inter_out = out_dir / f"{out_name}.inter"
    item_out  = out_dir / f"{out_name}.item"

    if inter_out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists (use --overwrite): {inter_out}")

    if args.fraction is None and args.n_users is None:
        raise SystemExit("Provide --fraction or --n-users")

    # Read source inter
    print(f"[1/4] reading {inter_src}")
    with inter_src.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = list(reader.fieldnames or [])
        rows   = [dict(r) for r in reader]

    # Find user_id column
    user_col = next((c for c in header if "user_id" in c), None)
    sess_col = next((c for c in header if "session_id" in c), None)
    item_col = next((c for c in header if "item_id" in c), None)
    if not user_col or not sess_col:
        raise SystemExit(f"Cannot find user_id / session_id columns in {header}")

    all_users = sorted({r[user_col] for r in rows})
    total_users = len(all_users)

    rng = random.Random(args.seed)
    if args.n_users is not None:
        n = min(args.n_users, total_users)
    else:
        n = max(1, round(total_users * args.fraction))

    sampled_users = set(rng.sample(all_users, n))
    print(f"[2/4] sampled {len(sampled_users):,} / {total_users:,} users  (seed={args.seed})")

    # Filter rows
    sampled_rows = [r for r in rows if r[user_col] in sampled_users]
    sampled_items = {r[item_col] for r in sampled_rows} if item_col else set()
    sampled_sessions = {r[sess_col] for r in sampled_rows}

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] writing inter: {inter_out}")
    with inter_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for r in sampled_rows:
            writer.writerow(r)

    # Filter item file
    print(f"[4/4] writing item: {item_out}")
    with item_src.open("r", encoding="utf-8", newline="") as fh:
        item_reader = csv.DictReader(fh, delimiter="\t")
        item_header = list(item_reader.fieldnames or [])
        item_col_key = next((c for c in item_header if "item_id" in c), None)
        item_rows = [dict(r) for r in item_reader]

    filtered_items = (
        [r for r in item_rows if r.get(item_col_key, "") in sampled_items]
        if item_col_key else item_rows
    )
    with item_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=item_header, delimiter="\t")
        writer.writeheader()
        for r in filtered_items:
            writer.writerow(r)

    summary = {
        "source_dataset": src_name,
        "output_dataset": out_name,
        "source_dir": str(src_dir.resolve()),
        "out_dir":    str(out_dir.resolve()),
        "params": {
            "fraction": args.fraction,
            "n_users":  args.n_users,
            "seed":     args.seed,
        },
        "stats": {
            "source_users":    total_users,
            "sampled_users":   len(sampled_users),
            "sampled_sessions": len(sampled_sessions),
            "sampled_rows":    len(sampled_rows),
            "sampled_items":   len(sampled_items),
        },
    }
    sp = out_dir / f"{out_name}.sample_summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    s = summary["stats"]
    print(f"done → {out_dir}")
    print(f"  users={s['sampled_users']:,}  sessions={s['sampled_sessions']:,}  items={s['sampled_items']:,}  rows={s['sampled_rows']:,}")


if __name__ == "__main__":
    main()
