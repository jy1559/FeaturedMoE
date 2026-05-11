#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_PY="${SCRIPT_DIR}/run_beauty_3model_aggressive.py"
SLACK_WRAPPER="${SCRIPT_DIR}/../baseline_2/run_with_slack_notify.sh"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  source "${LOCAL_ENV_FILE}"
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/FMoE/bin/python}"
AXIS="${AXIS:-BEAUTY_3MODEL_AGGR16}"
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
MAX_EVALS="${MAX_EVALS:-16}"
TUNE_EPOCHS="${TUNE_EPOCHS:-100}"
TUNE_PATIENCE="${TUNE_PATIENCE:-10}"
RUNTIME_SEED_BASE="${RUNTIME_SEED_BASE:-3100}"

CMD=(
  "${PYTHON_BIN}" "${RUNNER_PY}"
  --axis "${AXIS}"
  --gpus "${GPUS}"
  --seeds "${SEEDS}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --runtime-seed-base "${RUNTIME_SEED_BASE}"
  --resume-from-logs
)

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

if [[ "${SLACK_NOTIFY:-1}" == "1" && -x "${SLACK_WRAPPER}" ]]; then
  "${SLACK_WRAPPER}" --on --title "Baseline3 Beauty 3Model Aggressive" --note "axis=${AXIS}" -- "${CMD[@]}"
else
  "${CMD[@]}"
fi