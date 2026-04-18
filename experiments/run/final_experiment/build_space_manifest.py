#!/usr/bin/env python3
"""Build the frozen search-space manifest for final_experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SERVER_SPLIT_PATH,
    DEFAULT_TUNING_SPACE_PATH,
    build_space_manifest,
    write_space_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final_experiment search-space manifest")
    parser.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--tuning-space-out", default=str(DEFAULT_TUNING_SPACE_PATH))
    parser.add_argument("--server-split-out", default=str(DEFAULT_SERVER_SPLIT_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_space_manifest()
    manifest_path = Path(str(args.manifest_out))
    tuning_space_path = Path(str(args.tuning_space_out))
    server_split_path = Path(str(args.server_split_out))
    write_space_outputs(manifest, manifest_path, tuning_space_path, server_split_path)

    server_splits = dict(manifest.get("server_splits") or {})
    runtime_values = {
        name: float((payload or {}).get("predicted_runtime_sec", 0.0) or 0.0)
        for name, payload in server_splits.items()
    }
    nonzero_runtime = [value for value in runtime_values.values() if value > 0.0]
    runtime_ratio = 0.0
    if len(nonzero_runtime) >= 2:
        runtime_ratio = (max(nonzero_runtime) - min(nonzero_runtime)) / max(nonzero_runtime)

    print(f"[final_experiment] manifest -> {manifest_path}")
    print(f"[final_experiment] tuning space -> {tuning_space_path}")
    print(f"[final_experiment] server split -> {server_split_path}")
    print(
        f"[final_experiment] pair_count={manifest.get('pair_count')} "
        f"baseline_pair_count={manifest.get('baseline_pair_count')} "
        f"route_bank_count={manifest.get('route_bank_count')}"
    )
    runtime_summary = " ".join(
        f"{name}={value:.1f}" for name, value in sorted(runtime_values.items())
    )
    print(f"[final_experiment] predicted_runtime_sec {runtime_summary} balance_gap={runtime_ratio * 100:.1f}%")
    if int(manifest.get("pair_count", 0) or 0) != 60:
        print("[WARN] pair_count is not 60")
    if int(manifest.get("route_bank_count", 0) or 0) != 6:
        print("[WARN] route_bank_count is not 6")
    if runtime_ratio > 0.10:
        print("[WARN] server runtime balance gap is above 10%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
