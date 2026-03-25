#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_phase10_13_canonical_rerun.py"
)

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

echo "[phase10_13_canonical_rerun] ${CMD[*]}"
"${CMD[@]}"
