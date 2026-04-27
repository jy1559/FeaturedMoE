#!/usr/bin/env python3
"""Deprecated wrapper for the legacy efficiency-first Q4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    legacy_script = Path(__file__).resolve().parent / "legacy" / "q4_efficiency.py"
    print(f"[deprecated] q4_efficiency.py moved to {legacy_script}", flush=True)
    cmd = [sys.executable, str(legacy_script), *sys.argv[1:]]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
