#!/usr/bin/env python3
"""Reorganize legacy assets into experiments/_quarantine/정리_260305.

This script is destructive (moves files). It preserves a restore map/index:
  - 00_index/manifest.jsonl
  - 00_index/moved_files.csv
  - 00_index/restore_map.tsv
  - 00_index/restore_from_정리_260305.sh
  - 00_index/README.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MoveRow:
    original_path: str
    current_path: str
    category: str
    reason: str
    moved_at_utc: str
    size_bytes: int
    sha256: str
    purpose: str
    dataset: str
    model: str
    source_bucket: str


KNOWN_DATASETS = (
    "movielens1m",
    "retail_rocket",
    "amazon_beauty",
    "foursquare",
    "kuairec0.3",
    "kuairec",
    "lastfm0.3",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    # .../experiments/run/common/cleanup_to_quarantine_260305.py
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


def split_dataset_model(prefix: str) -> tuple[str, str]:
    raw = prefix.strip("_")
    if not raw:
        return "unknown_dataset", "unknown_model"

    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if raw == ds:
            return ds, "unknown_model"
        if raw.startswith(f"{ds}_"):
            model = raw[len(ds) + 1 :].strip("_")
            return ds, model or "unknown_model"

    if "_" in raw:
        left, right = raw.split("_", 1)
        return (left or "unknown_dataset"), (right or "unknown_model")

    return "unknown_dataset", raw


def parse_run_log_meta(path: Path, rel: Path) -> tuple[str, str, str, str]:
    parts = rel.parts
    track = parts[0] if len(parts) >= 1 else "unknown_track"
    purpose = parts[1] if len(parts) >= 2 else "unknown_axis"
    phase = parts[2] if len(parts) >= 3 else ""
    dataset = "unknown_dataset"
    model = "unknown_model"

    stem = path.stem
    prefix = ""

    markers: list[str] = []
    if purpose and purpose != "unknown_axis" and phase:
        markers.append(f"_{purpose}_{phase}_gpu")
    if purpose and purpose != "unknown_axis":
        markers.append(f"_{purpose}_")

    for marker in markers:
        if marker in stem:
            prefix = stem.split(marker, 1)[0]
            break

    if not prefix:
        m = re.match(r"^(?P<prefix>.+?)_(train|hparam|layout|schedule)_", stem)
        if m:
            prefix = m.group("prefix")

    if prefix:
        dataset, model = split_dataset_model(prefix)
    return track, purpose, dataset, model


def parse_legacy_log_meta(path: Path, rel: Path) -> tuple[str, str, str]:
    purpose = "train"
    low = path.name.lower()
    if "tune" in low or "hyperopt" in low:
        purpose = "hparam"

    model = rel.parts[0] if len(rel.parts) >= 2 else "unknown_model"
    dataset = "unknown_dataset"

    stem = path.stem
    parts_stem = stem.split("-")
    if len(parts_stem) >= 2:
        m0 = parts_stem[0].strip()
        d0 = parts_stem[1].strip()
        if m0:
            model = m0
        if d0:
            dataset = d0
    return purpose, dataset, model


def write_index(base: Path, rows: list[MoveRow]) -> None:
    idx = base / "00_index"
    idx.mkdir(parents=True, exist_ok=True)

    manifest = idx / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    csv_path = idx / "moved_files.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "original_path",
                "current_path",
                "category",
                "reason",
                "moved_at_utc",
                "size_bytes",
                "sha256",
                "source_bucket",
                "purpose",
                "dataset",
                "model",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.original_path,
                    row.current_path,
                    row.category,
                    row.reason,
                    row.moved_at_utc,
                    row.size_bytes,
                    row.sha256,
                    row.source_bucket,
                    row.purpose,
                    row.dataset,
                    row.model,
                ]
            )

    restore_map = idx / "restore_map.tsv"
    with restore_map.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row.original_path}\t{row.current_path}\n")

    restore_sh = idx / "restore_from_정리_260305.sh"
    restore_sh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="${SCRIPT_DIR}/restore_map.tsv"
DRY_RUN="false"
FILTER_RE=""

usage() {
  cat <<USAGE
Usage: $0 [--dry-run] [--filter <regex>]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN="true"; shift ;;
    --filter) FILTER_RE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ ! -f "$MAP_FILE" ] && { echo "Missing map: $MAP_FILE"; exit 1; }

while IFS=$'\\t' read -r SRC DST; do
  [ -z "$SRC" ] && continue

  if [ -n "$FILTER_RE" ]; then
    if [[ ! "$SRC" =~ $FILTER_RE ]] && [[ ! "$DST" =~ $FILTER_RE ]]; then
      continue
    fi
  fi

  if [ ! -e "$DST" ]; then
    echo "[SKIP] missing current path: $DST"
    continue
  fi

  mkdir -p "$(dirname "$SRC")"
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] mv \"$DST\" \"$SRC\""
  else
    mv "$DST" "$SRC"
    echo "[RESTORE] $SRC"
  fi
done < "$MAP_FILE"
""",
        encoding="utf-8",
    )
    os.chmod(restore_sh, 0o755)

    readme = idx / "README.md"
    readme.write_text(
        """# 정리_260305 Restore Index

## Files

- `manifest.jsonl`: 전체 이동 메타데이터
- `moved_files.csv`: 사람이 보기 쉬운 표
- `restore_map.tsv`: 원본 경로 <-> 현재 경로 매핑
- `restore_from_정리_260305.sh`: 복원 스크립트

## Restore Examples

```bash
# dry-run
bash restore_from_정리_260305.sh --dry-run

# path regex filter
bash restore_from_정리_260305.sh --filter 'run/fmoe'
```
""",
        encoding="utf-8",
    )


def write_log_inventory(repo: Path, rows: list[MoveRow]) -> None:
    inv_dir = repo / "experiments" / "run" / "artifacts" / "inventory"
    inv_dir.mkdir(parents=True, exist_ok=True)

    log_rows = [r for r in rows if r.current_path.endswith(".log")]
    grouped: dict[tuple[str, str, str, str], int] = {}
    for r in log_rows:
        key = (r.source_bucket, r.purpose, r.dataset, r.model)
        grouped[key] = grouped.get(key, 0) + 1

    csv_path = inv_dir / "log_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_bucket", "purpose", "dataset", "model", "count"])
        for (bucket, purpose, dataset, model), cnt in sorted(grouped.items()):
            writer.writerow([bucket, purpose, dataset, model, cnt])

    md_path = inv_dir / "log_summary.md"
    lines = [
        "# Log Inventory Summary",
        "",
        f"- generated_at_utc: {utc_now()}",
        f"- total_log_files: {len(log_rows)}",
        "",
        "| source_bucket | purpose | dataset | model | count |",
        "|---|---|---|---|---:|",
    ]
    for (bucket, purpose, dataset, model), cnt in sorted(grouped.items()):
        lines.append(f"| {bucket} | {purpose} | {dataset} | {model} | {cnt} |")
    if not grouped:
        lines.append("| - | - | - | - | 0 |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move legacy assets into 정리_260305 quarantine buckets.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = repo_root()
    exp = repo / "experiments"
    run = exp / "run"
    quar = exp / "_quarantine" / "정리_260305"
    categories = {
        "10": quar / "10_run_혼합스크립트",
        "20": quar / "20_run_기존로그",
        "21": quar / "21_run_기존결과",
        "30": quar / "30_legacy_log",
        "40": quar / "40_legacy_tensorboard",
        "50": quar / "50_legacy_wandb",
        "60": quar / "60_빈파일_깨진결과",
        "70": quar / "70_구문서_노트북",
    }
    for p in categories.values():
        p.mkdir(parents=True, exist_ok=True)

    moved: list[MoveRow] = []

    def record_move(
        src: Path,
        dst: Path,
        *,
        category: str,
        reason: str,
        source_bucket: str,
        purpose: str = "",
        dataset: str = "",
        model: str = "",
    ) -> None:
        if not src.exists():
            return
        original = str(src.resolve())
        size = path_size(src)
        digest = file_sha256(src) if src.is_file() else ""
        if args.dry_run:
            final = dst
            print(f"[DRY_RUN][{category}] {src} -> {final}")
        else:
            final = safe_move(src, dst)
            print(f"[MOVED][{category}] {src} -> {final}")
        moved.append(
            MoveRow(
                original_path=original,
                current_path=str((dst if args.dry_run else final).resolve()),
                category=category,
                reason=reason,
                moved_at_utc=utc_now(),
                size_bytes=size,
                sha256=digest,
                purpose=purpose,
                dataset=dataset,
                model=model,
                source_bucket=source_bucket,
            )
        )

    # 10) run mixed scripts
    for rel in [
        Path("experiments/run/fmoe/tune_hparam_hir.sh"),
        Path("experiments/run/fmoe/run_p2rep_and_hir_gpu7.sh"),
    ]:
        src = repo / rel
        dst = categories["10"] / rel
        record_move(
            src,
            dst,
            category="10_run_혼합스크립트",
            reason="fmoe_hir 경계 분리를 위해 run/fmoe 혼합 스크립트 비활성화",
            source_bucket="run_mixed_script",
        )

    # 20) run legacy logs (classify .log)
    run_log = run / "log"
    if run_log.exists():
        for p in sorted(run_log.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(run_log)
            if p.suffix.lower() == ".log":
                track, purpose, dataset, model = parse_run_log_meta(p, rel)
                dst = categories["20"] / track / purpose / dataset / model / p.name
                reason = "run/log legacy 로그를 artifacts 전환 전에 분류 보관"
                record_move(
                    p,
                    dst,
                    category="20_run_기존로그",
                    reason=reason,
                    source_bucket="run_log",
                    purpose=purpose,
                    dataset=dataset,
                    model=model,
                )
            else:
                dst = categories["20"] / "_misc" / rel
                record_move(
                    p,
                    dst,
                    category="20_run_기존로그",
                    reason="run/log 하위 non-log 파일 분리 보관",
                    source_bucket="run_log_misc",
                )
        if not args.dry_run:
            shutil.rmtree(run_log, ignore_errors=True)
            run_log.mkdir(parents=True, exist_ok=True)

    # 21 + 60) run legacy results
    run_results = run / "hyperopt_results"
    if run_results.exists():
        for p in sorted(run_results.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(run_results)
            if p.suffix.lower() == ".json":
                if p.stat().st_size == 0:
                    dst = categories["60"] / rel
                    record_move(
                        p,
                        dst,
                        category="60_빈파일_깨진결과",
                        reason="0-byte JSON 결과는 오염 결과로 별도 보관",
                        source_bucket="run_result_zero",
                    )
                    continue
                try:
                    with p.open("r", encoding="utf-8") as f:
                        json.load(f)
                except Exception:
                    dst = categories["60"] / rel
                    record_move(
                        p,
                        dst,
                        category="60_빈파일_깨진결과",
                        reason="JSON 파싱 실패 결과는 오염/깨진 결과로 별도 보관",
                        source_bucket="run_result_broken",
                    )
                    continue

            dst = categories["21"] / rel
            record_move(
                p,
                dst,
                category="21_run_기존결과",
                reason="run/hyperopt_results legacy 결과 보관",
                source_bucket="run_result",
            )
        if not args.dry_run:
            shutil.rmtree(run_results, ignore_errors=True)
            (run_results / "baseline").mkdir(parents=True, exist_ok=True)
            (run_results / "fmoe").mkdir(parents=True, exist_ok=True)
            (run_results / "fmoe_hir").mkdir(parents=True, exist_ok=True)

    # 30) legacy experiments/log
    legacy_log = exp / "log"
    if legacy_log.exists():
        for p in sorted(legacy_log.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(legacy_log)
            if p.suffix.lower() == ".log":
                purpose, dataset, model = parse_legacy_log_meta(p, rel)
                dst = categories["30"] / purpose / dataset / model / p.name
                record_move(
                    p,
                    dst,
                    category="30_legacy_log",
                    reason="legacy experiments/log 로그 분류 보관",
                    source_bucket="legacy_log",
                    purpose=purpose,
                    dataset=dataset,
                    model=model,
                )
            else:
                dst = categories["30"] / "_misc" / rel
                record_move(
                    p,
                    dst,
                    category="30_legacy_log",
                    reason="legacy log 하위 non-log 파일 보관",
                    source_bucket="legacy_log_misc",
                )
        if not args.dry_run:
            shutil.rmtree(legacy_log, ignore_errors=True)
            legacy_log.mkdir(parents=True, exist_ok=True)

    # 40) tensorboard
    tb = exp / "log_tensorboard"
    if tb.exists():
        dst = categories["40"] / "log_tensorboard"
        record_move(
            tb,
            dst,
            category="40_legacy_tensorboard",
            reason="legacy tensorboard 로그 보관",
            source_bucket="legacy_tensorboard",
        )
        if not args.dry_run:
            tb.mkdir(parents=True, exist_ok=True)

    # 50) wandb
    wandb = exp / "wandb"
    if wandb.exists():
        dst = categories["50"] / "wandb"
        record_move(
            wandb,
            dst,
            category="50_legacy_wandb",
            reason="legacy wandb 런 보관",
            source_bucket="legacy_wandb",
        )
        if not args.dry_run:
            wandb.mkdir(parents=True, exist_ok=True)

    # 70) notebook + empty docs
    for rel in [Path("experiments/run/dataset_model_score_visualization.ipynb")]:
        src = repo / rel
        dst = categories["70"] / rel
        record_move(
            src,
            dst,
            category="70_구문서_노트북",
            reason="run 루트 분석 노트북 분리 보관",
            source_bucket="legacy_notebook",
        )

    docs_dir = exp / "models" / "FeaturedMoE" / "docs"
    if docs_dir.exists():
        for p in sorted(docs_dir.glob("*.md")):
            if p.is_file() and p.stat().st_size == 0:
                rel = p.relative_to(repo)
                dst = categories["70"] / rel
                record_move(
                    p,
                    dst,
                    category="70_구문서_노트북",
                    reason="0-byte 구문서 보관",
                    source_bucket="legacy_empty_doc",
                )

    if not args.dry_run:
        write_index(quar, moved)
        write_log_inventory(repo, moved)
        print(f"[OK] moved_files={len(moved)}")
        print(f"[OK] index={quar / '00_index'}")
    else:
        print(f"[DRY_RUN] planned_moves={len(moved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
