#!/usr/bin/env bash
set -euo pipefail

# Resume-safe wrapper for Q2~Q5.
#
# This preserves existing q2/q3/q4/q5 logs and lets the Python entrypoints skip
# already successful jobs via --resume-from-logs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CLEAN_FIRST="${CLEAN_FIRST:-0}"
export RESUME_FLAG="${RESUME_FLAG:---resume-from-logs}"

echo "[resume] CLEAN_FIRST=${CLEAN_FIRST} RESUME_FLAG=${RESUME_FLAG}"

exec bash "${SCRIPT_DIR}/run_q2_q5_tmux_clean.sh" "$@"