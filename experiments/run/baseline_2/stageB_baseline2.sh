#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"

cd "${REPO_ROOT}"

"${PY_BIN}" "${SCRIPT_DIR}/run_staged_tuning.py" \
  --stage B \
  --track baseline_2 \
  --axis ABCD_v2_lean \
  --budget-profile lean \
  --stage-a-struct-count 18 \
  --stage-a-lr-grid "2e-4,6e-4,1.2e-3,3e-3" \
  --promote-a-to-b 6 \
  --promote-b-to-c 3 \
  --promote-c-to-d 2 \
  --stage-c-per-parent 2 \
  --stage-d-per-parent 2 \
  --models sasrec,tisasrec,gru4rec \
  --datasets "KuaiRecLargeStrictPosV2_0.2,beauty,lastfm0.03,foursquare,movielens1m,retail_rocket" \
  --gpus "${GPU_LIST:-0}" \
  --runtime-seed "${RUNTIME_SEED:-1}" \
  --resume-from-logs \
  "$@"

