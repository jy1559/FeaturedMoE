#!/usr/bin/env python3
"""Update baseline dataset/phase CSV summary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from common.phase_summary_csv import build_baseline_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Update baseline dataset/phase CSV summary.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. KuaiRecSmall0.1")
    parser.add_argument("--phase", required=True, help="Phase folder name, e.g. P0 or P0_SMOKE")
    args = parser.parse_args()

    out_path = build_baseline_summary(args.dataset, args.phase)
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
