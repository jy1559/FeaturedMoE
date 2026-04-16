#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEXT_SCRIPT="${SCRIPT_DIR}/baseline2_addtuning2.sh"
LOG_ROOT="${SCRIPT_DIR}/../artifacts/logs/baseline_2/PAIR60_ADDTUNING2"
CURRENT_PATTERN='python .*baseline2_addtuning\.py( |$)'

mkdir -p "${LOG_ROOT}"

while pgrep -f "${CURRENT_PATTERN}" >/dev/null 2>&1; do
  echo "[queue_baseline2_addtuning2_after_current] waiting for current ADDTUNING to finish..."
  sleep 60
done

echo "[queue_baseline2_addtuning2_after_current] launching ADDTUNING2"
exec bash "${NEXT_SCRIPT}" "$@"