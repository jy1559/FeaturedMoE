#!/usr/bin/env python3
"""Compatibility wrapper for missing source.

Runs the compiled bytecode module using Python 3.12 so existing launch
commands keep working even when the source file is unavailable.
"""

from pathlib import Path
import os
import sys


def main() -> int:
	pyc_path = Path(__file__).with_name("__pycache__") / "cross_dataset_a12_portfolio.cpython-312.pyc"
	if not pyc_path.exists():
		print(f"error: missing bytecode file: {pyc_path}", file=sys.stderr)
		return 1

	repo_root = Path(__file__).resolve().parents[3]
	preferred_py312 = repo_root / ".venv" / "bin" / "python3.12"
	py312 = os.environ.get("RUN_PYTHON312_BIN", str(preferred_py312))

	if not Path(py312).exists():
		print(
			"error: python3.12 interpreter not found. "
			"Set RUN_PYTHON312_BIN to a valid python3.12 path.",
			file=sys.stderr,
		)
		return 1

	env = os.environ.copy()
	script_dir = str(Path(__file__).resolve().parent)
	existing = env.get("PYTHONPATH", "")
	env["PYTHONPATH"] = script_dir if not existing else f"{script_dir}:{existing}"

	os.execve(py312, [py312, str(pyc_path), *sys.argv[1:]], env)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
