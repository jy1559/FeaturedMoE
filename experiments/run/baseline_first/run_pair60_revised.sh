#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="${SCRIPT_DIR}/run_pair60_revised.py"
SLACK_WRAPPER="${SCRIPT_DIR}/run_with_slack_notify.sh"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/FMoE/bin/python}"
AXIS="${AXIS:-PAIR60_V4_REVISED}"
GPUS="${GPUS:-0}"

CMD=(
  "${PYTHON_BIN}" "${RUNNER_PY}"
  --axis "${AXIS}"
  --gpus "${GPUS}"
  --search-algo random
  --max-evals 6
  --tune-epochs 70
  --tune-patience 6
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

if [[ "${SLACK_NOTIFY:-1}" == "1" && -x "${SLACK_WRAPPER}" ]]; then
  "${SLACK_WRAPPER}" --on --title "Baseline2 Pair60 Revised" --note "axis=${AXIS}" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi
