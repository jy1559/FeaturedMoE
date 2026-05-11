#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TOPUP_SCRIPT="${ROOT_DIR}/experiments/run/fmoe_n4/cross_dataset_a12_portfolio_topup.sh"

WAIT_PID="${N4_WAIT_PID:-}"
POLL_SEC="${N4_WAIT_POLL_SEC:-20}"
TOPUP_ARGS=()

usage() {
  cat <<USAGE
Usage: $0 [--wait-pid PID] [--poll-sec 20] [-- <topup args>]

Behavior:
- If --wait-pid is omitted, auto-detect running cross_dataset_a12_portfolio.py pid.
- Wait until that pid exits, then launch top-up runner.

Examples:
  bash experiments/run/fmoe_n4/cross_dataset_a12_portfolio_topup_queue.sh
  bash experiments/run/fmoe_n4/cross_dataset_a12_portfolio_topup_queue.sh --wait-pid 138250
  bash experiments/run/fmoe_n4/cross_dataset_a12_portfolio_topup_queue.sh -- --dry-run
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-pid)
      WAIT_PID="$2"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      TOPUP_ARGS+=("$@")
      break
      ;;
    *)
      TOPUP_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${WAIT_PID}" ]]; then
  WAIT_PID="$(pgrep -f 'run/fmoe_n4/cross_dataset_a12_portfolio.py' | head -n 1 || true)"
fi

if [[ -n "${WAIT_PID}" ]] && kill -0 "${WAIT_PID}" 2>/dev/null; then
  echo "[QUEUE] waiting for pid=${WAIT_PID} to exit..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${POLL_SEC}"
  done
  echo "[QUEUE] pid=${WAIT_PID} finished; launching top-up"
else
  if [[ -n "${WAIT_PID}" ]]; then
    echo "[QUEUE] pid=${WAIT_PID} not running; launching top-up now"
  else
    echo "[QUEUE] no running portfolio pid found; launching top-up now"
  fi
fi

exec bash "${TOPUP_SCRIPT}" "${TOPUP_ARGS[@]}"
