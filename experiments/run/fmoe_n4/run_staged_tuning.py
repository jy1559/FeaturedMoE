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
        "ABCD_v1",
    ]
    cmd = [python_bin, str(BASE_RUNNER), *defaults, *sys.argv[1:]]
    return int(subprocess.call(cmd, cwd=str(THIS_DIR.parents[2])))


if __name__ == "__main__":
    raise SystemExit(main())

