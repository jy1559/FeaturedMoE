#!/usr/bin/env python3
"""Update fmoe_n2 phase CSV summaries from log files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from common.phase_summary_csv import build_fmoe_n2_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Update fmoe_n2 phase CSV summaries from log files.")
    parser.add_argument("--phase", required=True, help="Phase folder name, e.g. ARCH3")
    parser.add_argument("--axis", default="s00_router_feature_heavy_v1", help="Axis folder name.")
    args = parser.parse_args()

    out_paths = build_fmoe_n2_summaries(args.axis, args.phase)
    if not out_paths:
        phase_dir = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "logs"
            / "fmoe_n2"
            / args.axis
            / args.phase
        )
        print(f"[OK] no logs under {phase_dir}")
        return 0

    for out_path in out_paths:
        print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
