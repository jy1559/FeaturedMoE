#!/usr/bin/env python3
"""Thin wrapper: reuse baseline_2 staged runner for fmoe_n4 track."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_RUNNER = THIS_DIR.parent / "baseline_2" / "run_staged_tuning.py"


def main() -> int:
    python_bin = os.environ.get("RUN_PYTHON_BIN", sys.executable)
    # 2-day scout preset: keep A->D flow but shrink search width/depth.
    defaults = [
        "--track",
        "fmoe_n4",
        "--models",
        "featured_moe_n3",
        "--axis",
        "ABCD_A12_hparam_v1",
        "--budget-profile",
        "fast",
        "--stage-a-struct-count",
        "8",
        "--stage-a-lr-grid",
        "1.6e-4,5e-4",
        "--promote-a-to-b",
        "4",
        "--promote-b-to-c",
        "2",
        "--promote-c-to-d",
        "1",
        "--stage-c-per-parent",
        "2",
        "--stage-d-per-parent",
        "2",
    ]
    cmd = [python_bin, str(BASE_RUNNER), *defaults, *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(THIS_DIR.parents[2])))


if __name__ == "__main__":
    raise SystemExit(main())

