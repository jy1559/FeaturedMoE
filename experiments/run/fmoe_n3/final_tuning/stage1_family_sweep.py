#!/usr/bin/env python3
"""Stage1 family sweep for A12 final tuning."""

from __future__ import annotations

import common


def main() -> int:
    args = common.parse_stage_args("stage1", "A12 final tuning stage1 family sweep")
    rows = common.build_stage1_rows(args)
    return int(common.launch_stage("stage1", args, rows))


if __name__ == "__main__":
    raise SystemExit(main())
