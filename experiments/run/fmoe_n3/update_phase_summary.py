#!/usr/bin/env python3
"""Update fmoe_n3 phase and axis CSV summaries from log files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from common.phase_summary_csv import build_fmoe_n3_axis_summary, build_fmoe_n3_summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Update fmoe_n3 phase/axis CSV summaries from log files.")
    parser.add_argument("--phase", default="CORE28", help="Phase folder name, e.g. CORE28")
    parser.add_argument("--axis", default="core_ablation_v2", help="Axis folder name.")
    args = parser.parse_args()

    written = []
    phase_out = build_fmoe_n3_summaries(args.axis, args.phase)
    written.extend(phase_out)
    axis_out = build_fmoe_n3_axis_summary(args.axis)
    written.append(axis_out)

    for out_path in written:
        print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
