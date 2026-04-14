#!/usr/bin/env python3
"""Build beauty feature_added_v4 from feature_added_v3.

This is a thin stage-3 wrapper around experiments/tools/build_feature_v4_from_v3.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "Datasets" / "processed" / "feature_added_v3"
DEFAULT_DST = REPO_ROOT / "Datasets" / "processed" / "feature_added_v4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="beauty")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SRC)
    p.add_argument("--target-root", type=Path, default=DEFAULT_DST)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_v4_module():
    script = Path(__file__).resolve().parent / "build_feature_v4_from_v3.py"
    spec = importlib.util.spec_from_file_location("build_feature_v4_from_v3", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load build_feature_v4_from_v3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    mod = load_v4_module()
    summary = mod.process_dataset(
        src_root=Path(args.source_root),
        dst_root=Path(args.target_root),
        dataset=str(args.dataset),
        overwrite=bool(args.overwrite),
    )
    print(
        "done "
        f"dataset={summary['dataset']} "
        f"valid_sessions={summary['output']['valid_sessions']} "
        f"test_sessions={summary['output']['test_sessions']}"
    )


if __name__ == "__main__":
    main()
