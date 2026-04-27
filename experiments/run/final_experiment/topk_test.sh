#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DATASETS="${TARGET_DATASETS:-KuaiRecLargeStrictPosV2_0.2,beauty,foursquare,lastfm0.03,movielens1m,retail_rocket}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TRACK_NAME="${FINAL_EXP_TRACK:-final_experiment_topk_test}"
PARENT_TRACK="${TOPK_PARENT_TRACK:-final_experiment}"
MAX_EVALS="${MAX_EVALS:-4}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-1.5}"
OOM_RETRY_LIMIT="${OOM_RETRY_LIMIT:-4}"

export FINAL_EXP_TRACK="${TRACK_NAME}"
export FINAL_EXP_STAGE3_TUNE_EPOCHS="${FINAL_EXP_STAGE3_TUNE_EPOCHS:-24}"
export FINAL_EXP_STAGE3_TUNE_PATIENCE="${FINAL_EXP_STAGE3_TUNE_PATIENCE:-4}"

python3 "${SCRIPT_DIR}/topk_test.py" \
  --datasets "${TARGET_DATASETS}" \
  --gpus "${GPU_LIST}" \
  --parent-track "${PARENT_TRACK}" \
  --max-evals "${MAX_EVALS}" \
  --max-run-hours "${MAX_RUN_HOURS}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT}" \
  "$@"