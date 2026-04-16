#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOPUP_SCRIPT="${ROOT_DIR}/experiments/run/fmoe_n4/cross_dataset_a12_portfolio_beauty_lr_topup.sh"

POLL_SEC="${N4_WAIT_POLL_SEC:-20}"

portfolio_running() {
  pgrep -f '[r]un/fmoe_n4/cross_dataset_a12_portfolio.py' >/dev/null 2>&1
}

gpu_busy() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
  [[ -n "${out}" ]]
}

echo "[QUEUE] waiting for active CrossDataset portfolio processes to finish..."
while portfolio_running; do
  sleep "${POLL_SEC}"
done

echo "[QUEUE] portfolio process ended; waiting for GPUs to become idle..."
while gpu_busy; do
  sleep "${POLL_SEC}"
done

echo "[QUEUE] GPUs idle; launching beauty lr top-up"
exec bash "${TOPUP_SCRIPT}" "$@"