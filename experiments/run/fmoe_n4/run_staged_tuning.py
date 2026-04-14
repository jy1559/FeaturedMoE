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
    defaults = [
        "--track",
        "fmoe_n4",
        "--models",
        "featured_moe_n3",
        "--axis",
        "ABCD_A12_hparam_v1",
        "--budget-profile",
        "deep",
        "--stage-a-struct-count",
        "36",
        "--stage-a-lr-grid",
        "8e-5,1.6e-4,3e-4,5e-4,8e-4,1.2e-3",
        "--promote-a-to-b",
        "16",
        "--promote-b-to-c",
        "6",
        "--promote-c-to-d",
        "3",
        "--stage-c-per-parent",
        "3",
        "--stage-d-per-parent",
        "3",
    ]
    cmd = [python_bin, str(BASE_RUNNER), *defaults, *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(THIS_DIR.parents[2])))


if __name__ == "__main__":
    raise SystemExit(main())

