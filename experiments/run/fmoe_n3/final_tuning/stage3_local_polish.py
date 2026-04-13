#!/usr/bin/env python3
"""Stage3 local polish for A12 final tuning."""

from __future__ import annotations

import common


def main() -> int:
    args = common.parse_stage_args("stage3", "A12 final tuning stage3 local polish")
    rows = common.build_stage3_rows(args)
    return int(common.launch_stage("stage3", args, rows))


if __name__ == "__main__":
    raise SystemExit(main())
