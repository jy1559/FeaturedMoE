#!/usr/bin/env python3
"""Stage2 dataset-aware refinement for A12 final tuning."""

from __future__ import annotations

import common


def main() -> int:
    args = common.parse_stage_args("stage2", "A12 final tuning stage2 dataset refinement")
    rows = common.build_stage2_rows(args)
    return int(common.launch_stage("stage2", args, rows))


if __name__ == "__main__":
    raise SystemExit(main())
