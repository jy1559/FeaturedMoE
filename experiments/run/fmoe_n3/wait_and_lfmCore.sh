#!/usr/bin/env bash
# wait_and_lfmCore.sh
#
# Waits for a target PID to exit, then launches phase_core_28.sh for lastfm0.03
# on the same GPU set (default: 4,5,6,7).
#
# Usage:
#   bash wait_and_lfmCore.sh [PID]
#   bash wait_and_lfmCore.sh 385342
#
# Or run in the background so you can close the terminal:
#   nohup bash wait_and_lfmCore.sh 385342 > /tmp/lfmCore_wait.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Configuration ──────────────────────────────────────────────────────────────
WAIT_PID="${1:-}"          # PID to wait for; required
GPU_LIST="4,5,6,7"
DATASET="lastfm0.03"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="8300"
POLL_INTERVAL=60           # seconds between liveness checks
# ──────────────────────────────────────────────────────────────────────────────

if [ -z "${WAIT_PID}" ]; then
  echo "[wait_and_lfmCore] ERROR: no PID given." >&2
  echo "  Usage: bash $0 <PID>" >&2
  exit 1
fi

# Confirm the PID looks like a running process
if ! kill -0 "${WAIT_PID}" 2>/dev/null; then
  echo "[wait_and_lfmCore] WARNING: PID ${WAIT_PID} is already gone or not visible."
  echo "[wait_and_lfmCore] Proceeding immediately to launch phase_core_28."
else
  echo "[wait_and_lfmCore] $(date '+%F %T') Waiting for PID ${WAIT_PID} to finish..."
  echo "[wait_and_lfmCore] Will poll every ${POLL_INTERVAL}s."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_INTERVAL}"
  done
  echo "[wait_and_lfmCore] $(date '+%F %T') PID ${WAIT_PID} has exited."
fi

echo "[wait_and_lfmCore] $(date '+%F %T') Launching phase_core_28.sh for ${DATASET} on GPUs ${GPU_LIST}."

exec "${SCRIPT_DIR}/phase_core_28.sh" \
  --dataset      "${DATASET}"     \
  --gpus         "${GPU_LIST}"    \
  --max-evals    "${MAX_EVALS}"   \
  --tune-epochs  "${TUNE_EPOCHS}" \
  --tune-patience "${TUNE_PATIENCE}" \
  --seed-base    "${SEED_BASE}"
