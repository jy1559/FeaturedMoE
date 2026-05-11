#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="${SCRIPT_DIR}/run_full_history_campaign.py"
SLACK_WRAPPER="${SCRIPT_DIR}/../baseline_2/run_with_slack_notify.sh"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/FMoE/bin/python}"
AXIS="${AXIS:-FULL_HISTORY_V4_C4}"
if [[ -z "${GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  else
    GPUS="0"
  fi
fi
SEEDS="${SEEDS:-1}"
MAX_EVALS="${MAX_EVALS:-10}"

CMD=(
  "${PYTHON_BIN}" "${RUNNER_PY}"
  --axis "${AXIS}"
  --gpus "${GPUS}"
  --seeds "${SEEDS}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs 100
  --tune-patience 10
  --resume-from-logs
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

if [[ "${SLACK_NOTIFY:-1}" == "1" && -x "${SLACK_WRAPPER}" ]]; then
  "${SLACK_WRAPPER}" --on --title "Baseline3 Full-History Campaign" --note "axis=${AXIS}" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi