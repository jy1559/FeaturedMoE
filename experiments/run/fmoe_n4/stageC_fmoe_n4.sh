#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"

cd "${REPO_ROOT}"

"${PY_BIN}" "${SCRIPT_DIR}/run_staged_tuning.py" \
  --stage C \
  --budget-profile balanced \
  --datasets "KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket" \
  --gpus "${GPU_LIST:-0}" \
  --runtime-seed "${RUNTIME_SEED:-1}" \
  --resume-from-logs \
  "$@"

