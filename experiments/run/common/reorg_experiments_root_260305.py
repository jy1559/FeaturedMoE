#!/usr/bin/env python3
"""Reorganize experiments root files by usage/type (정리_260305 root pass).

Policy:
  - Keep only core runtime files in experiments root.
  - Move docs/tools/tests into dedicated folders.
  - Quarantine legacy/low-use scripts for safe rollback.
  - Record move index for restore.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Row:
    original_path: str
    current_path: str
    action: str
    category: str
    reason: str
    moved_at_utc: str
    size_bytes: int
    sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    # .../experiments/run/common/reorg_experiments_root_260305.py
    return Path(__file__).resolve().parents[3]


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += int(p.stat().st_size)
    return total


def safe_move(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    final = dst
    if final.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        final = final.parent / f"{final.stem}__dup_{ts}{final.suffix}"
    shutil.move(str(src), str(final))
    return final


def write_index(index_dir: Path, rows: list[Row]) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest = index_dir / "root_reorg_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    moved_csv = index_dir / "root_reorg_moved.csv"
    with moved_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "original_path",
                "current_path",
                "action",
                "category",
                "reason",
                "moved_at_utc",
                "size_bytes",
                "sha256",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row.original_path,
                    row.current_path,
                    row.action,
                    row.category,
                    row.reason,
                    row.moved_at_utc,
                    row.size_bytes,
                    row.sha256,
                ]
            )

    restore_map = index_dir / "root_reorg_restore_map.tsv"
    with restore_map.open("w", encoding="utf-8") as f:
        for row in rows:
            if row.action == "move":
                f.write(f"{row.original_path}\t{row.current_path}\n")

    restore_sh = index_dir / "restore_root_reorg_260305.sh"
    restore_sh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/root_reorg_restore_map.tsv"
DRY_RUN="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[ ! -f "$MAP_FILE" ] && { echo "Missing map: $MAP_FILE"; exit 1; }

while IFS=$'\\t' read -r SRC DST; do
  [ -z "$SRC" ] && continue
  if [ ! -e "$DST" ]; then
    echo "[SKIP] missing current path: $DST"
    continue
  fi
  mkdir -p "$(dirname "$SRC")"
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] mv \\"$DST\\" \\"$SRC\\""
  else
    mv "$DST" "$SRC"
    echo "[RESTORE] $SRC"
  fi
done < "$MAP_FILE"
""",
        encoding="utf-8",
    )
    os.chmod(restore_sh, 0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorganize experiments root files.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = repo_root()
    exp = repo / "experiments"
    qbase = exp / "_quarantine" / "정리_260305"
    qlegacy = qbase / "75_root_레거시코드_테스트"
    docs_dir = exp / "docs"
    tools_dir = exp / "tools"
    tests_dir = exp / "tests"
    index_dir = qbase / "00_index"

    docs_map = {
        exp / "CHANGES.md": docs_dir / "CHANGES.md",
        exp / "CONFIG_GUIDE.md": docs_dir / "CONFIG_GUIDE.md",
        exp / "CONFIG_IMPLEMENTATION.md": docs_dir / "CONFIG_IMPLEMENTATION.md",
        exp / "MANUAL.md": docs_dir / "MANUAL.md",
    }
    tools_map = {
        exp / "cleanup_gpu.sh": tools_dir / "cleanup_gpu.sh",
        exp / "precompute_neg_samples.py": tools_dir / "precompute_neg_samples.py",
    }
    tests_map = {
        exp / "test_config_load.py": tests_dir / "test_config_load.py",
    }
    quarantine_map = {
        exp / "train.py": qlegacy / "experiments" / "train.py",
        exp / "utils.py": qlegacy / "experiments" / "utils.py",
        exp / "recbole_utils.py": qlegacy / "experiments" / "recbole_utils.py",
        exp / "sampled_eval_dataloader.py": qlegacy / "experiments" / "sampled_eval_dataloader.py",
        exp / "test_bsarec_gpu2.py": qlegacy / "experiments" / "test_bsarec_gpu2.py",
        exp / "test_caser_init.py": qlegacy / "experiments" / "test_caser_init.py",
        exp / "models.py": qlegacy / "experiments" / "models.py",
    }

    rows: list[Row] = []

    def record(src: Path, dst: Path, category: str, reason: str) -> None:
        if not src.exists():
            return
        size = path_size(src)
        digest = file_sha256(src) if src.is_file() else ""
        if args.dry_run:
            final = dst
            print(f"[DRY_RUN][{category}] {src} -> {final}")
        else:
            final = safe_move(src, dst)
            print(f"[MOVED][{category}] {src} -> {final}")
        rows.append(
            Row(
                original_path=str(src.resolve()),
                current_path=str((dst if args.dry_run else final).resolve()),
                action="move",
                category=category,
                reason=reason,
                moved_at_utc=utc_now(),
                size_bytes=size,
                sha256=digest,
            )
        )

    for src, dst in docs_map.items():
        record(src, dst, "root_docs_relocate", "문서 파일을 experiments/docs로 분리")

    for src, dst in tools_map.items():
        record(src, dst, "root_tools_relocate", "운영 유틸 스크립트를 experiments/tools로 분리")

    for src, dst in tests_map.items():
        record(src, dst, "root_tests_relocate", "유지할 검증 스크립트를 experiments/tests로 분리")

    for src, dst in quarantine_map.items():
        record(src, dst, "75_root_레거시코드_테스트", "루트 혼잡 완화를 위해 레거시/저사용 코드 격리")

    pycache_dir = exp / "__pycache__"
    if pycache_dir.exists():
        size = path_size(pycache_dir)
        if args.dry_run:
            print(f"[DRY_RUN][root_generated_cleanup] delete {pycache_dir}")
        else:
            shutil.rmtree(pycache_dir, ignore_errors=True)
            print(f"[DELETED][root_generated_cleanup] {pycache_dir}")
        rows.append(
            Row(
                original_path=str(pycache_dir.resolve()),
                current_path="",
                action="delete",
                category="root_generated_cleanup",
                reason="재생성 가능한 __pycache__ 정리",
                moved_at_utc=utc_now(),
                size_bytes=size,
                sha256="",
            )
        )

    if args.dry_run:
        print(f"[DRY_RUN] planned_actions={len(rows)}")
    else:
        write_index(index_dir, rows)
        print(f"[OK] actions={len(rows)}")
        print(f"[OK] index={index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
