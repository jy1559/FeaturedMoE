#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# run/common -> experiments/run -> experiments
EXP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/.." && pwd)"
QUAR_ROOT="${EXP_DIR}/_quarantine"

usage() {
  cat <<USAGE
Usage:
  $0 move --level <L90|L10> --bucket <name> --reason <text> [--manifest <path>] <path1> [path2 ...]
  $0 purge --days <N> [--level <L90|L10>]
USAGE
}

now_iso() {
  date -Iseconds
}

date_tag() {
  date +%Y%m%d
}

json_escape() {
  python3 - <<'PY' "$1"
import json,sys
print(json.dumps(sys.argv[1], ensure_ascii=True))
PY
}

write_manifest_line() {
  local manifest="$1"
  local level="$2"
  local src="$3"
  local dst="$4"
  local reason="$5"
  local size="$6"
  local moved_at
  moved_at="$(now_iso)"
  printf '{"level":%s,"path":%s,"dst":%s,"size":%s,"reason":%s,"moved_at":%s}\n' \
    "$(json_escape "$level")" \
    "$(json_escape "$src")" \
    "$(json_escape "$dst")" \
    "$(json_escape "$size")" \
    "$(json_escape "$reason")" \
    "$(json_escape "$moved_at")" >> "$manifest"
}

path_size_bytes() {
  local p="$1"
  if [ -f "$p" ]; then
    stat -c %s "$p"
  elif [ -d "$p" ]; then
    du -sb "$p" | awk '{print $1}'
  else
    echo 0
  fi
}

do_move() {
  local level=""
  local bucket=""
  local reason=""
  local manifest_override=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --level) level="$2"; shift 2 ;;
      --bucket) bucket="$2"; shift 2 ;;
      --reason) reason="$2"; shift 2 ;;
      --manifest) manifest_override="$2"; shift 2 ;;
      --) shift; break ;;
      -*) echo "Unknown option: $1"; usage; exit 1 ;;
      *) break ;;
    esac
  done

  [ -z "$level" ] && { echo "--level required"; exit 1; }
  [ -z "$bucket" ] && { echo "--bucket required"; exit 1; }
  [ -z "$reason" ] && { echo "--reason required"; exit 1; }
  [ "$#" -eq 0 ] && { echo "No paths to move"; exit 1; }

  local day target manifest
  day="$(date_tag)"
  target="${QUAR_ROOT}/${day}/${level}_${bucket}"
  mkdir -p "$target"

  if [ -n "$manifest_override" ]; then
    manifest="$manifest_override"
    mkdir -p "$(dirname "$manifest")"
  else
    manifest="${target}/manifest.jsonl"
  fi

  local src
  for src in "$@"; do
    if [ ! -e "$src" ]; then
      echo "[SKIP] missing: $src"
      continue
    fi

    local abs_src rel dst dst_parent sz
    abs_src="$(python3 - <<'PY' "$src"
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve())
PY
)"
    if [[ "$abs_src" == "$REPO_ROOT/"* ]]; then
      rel="${abs_src#${REPO_ROOT}/}"
    else
      rel="${abs_src#/}"
    fi
    dst="${target}/${rel}"
    dst_parent="$(dirname "$dst")"
    mkdir -p "$dst_parent"

    sz="$(path_size_bytes "$src")"
    mv "$src" "$dst"
    write_manifest_line "$manifest" "$level" "$abs_src" "$dst" "$reason" "$sz"
    echo "[MOVED][$level] $src -> $dst"
  done

  echo "Manifest: $manifest"
}

do_purge() {
  local days=""
  local level=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --days) days="$2"; shift 2 ;;
      --level) level="$2"; shift 2 ;;
      *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
  done

  [ -z "$days" ] && { echo "--days required"; exit 1; }
  [ ! -d "$QUAR_ROOT" ] && { echo "No quarantine root"; return 0; }

  local find_expr=("$QUAR_ROOT" -mindepth 2 -maxdepth 2 -type d -mtime "+$days")
  if [ -n "$level" ]; then
    find_expr+=( -name "${level}_*" )
  fi

  while IFS= read -r d; do
    echo "[PURGE] $d"
    rm -rf "$d"
  done < <(find "${find_expr[@]}")
}

main() {
  [ "$#" -lt 1 ] && { usage; exit 1; }
  local cmd="$1"; shift
  case "$cmd" in
    move) do_move "$@" ;;
    purge) do_purge "$@" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
