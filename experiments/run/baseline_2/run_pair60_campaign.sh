#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="${SCRIPT_DIR}/run_pair60_campaign.py"
SLACK_WRAPPER="${SCRIPT_DIR}/run_with_slack_notify.sh"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/FMoE/bin/python}"
AXIS="${AXIS:-PAIR60_V4}"
GPUS="${GPUS:-0}"
SEEDS="${SEEDS:-1}"

CMD=(
  "${PYTHON_BIN}" "${RUNNER_PY}"
  --axis "${AXIS}"
  --gpus "${GPUS}"
  --seeds "${SEEDS}"
  --max-evals 6
  --tune-epochs 70
  --tune-patience 6
  --resume-from-logs
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

if [[ "${SLACK_NOTIFY:-1}" == "1" && -x "${SLACK_WRAPPER}" ]]; then
  "${SLACK_WRAPPER}" --on --title "Baseline2 Pair60 Campaign" --note "axis=${AXIS}" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi
