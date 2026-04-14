#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"

cd "${REPO_ROOT}"

"${PY_BIN}" "${SCRIPT_DIR}/run_staged_tuning.py" \
  --stage D \
  --axis "${AXIS_NAME:-ABCD_A12_hparam_v1}" \
  --budget-profile "${BUDGET_PROFILE:-fast}" \
  --stage-a-struct-count "${STAGE_A_STRUCT_COUNT:-8}" \
  --stage-a-lr-grid "${STAGE_A_LR_GRID:-1.6e-4,5e-4}" \
  --promote-a-to-b "${PROMOTE_A_TO_B:-4}" \
  --promote-b-to-c "${PROMOTE_B_TO_C:-2}" \
  --promote-c-to-d "${PROMOTE_C_TO_D:-1}" \
  --promote-d-to-final "${PROMOTE_D_TO_FINAL:-1}" \
  --stage-c-per-parent "${STAGE_C_PER_PARENT:-2}" \
  --stage-d-per-parent "${STAGE_D_PER_PARENT:-2}" \
  --datasets "KuaiRecLargeStrictPosV2_0.2,lastfm0.03,beauty,foursquare,movielens1m,retail_rocket" \
  --gpus "${GPU_LIST:-0}" \
  --runtime-seed "${RUNTIME_SEED:-1}" \
  --final-seeds "${FINAL_SEEDS:-1}" \
  --resume-from-logs \
  "$@"

