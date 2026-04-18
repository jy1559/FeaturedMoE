#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-plan}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DATASETS="${DATASETS:-beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket}"
SEARCH_ALGO="${SEARCH_ALGO:-tpe}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-2}"
OOM_RETRY_LIMIT="${OOM_RETRY_LIMIT:-5}"

echo "[final_experiment_addtuning] action=${ACTION} datasets=${DATASETS} gpus=${GPU_LIST} search_algo=${SEARCH_ALGO} max_run_hours=${MAX_RUN_HOURS} oom_retry_limit=${OOM_RETRY_LIMIT}"

python3 "${SCRIPT_DIR}/addtuning.py" "${ACTION}" \
  --datasets "${DATASETS}" \
  --gpus "${GPU_LIST}" \
  --search-algo "${SEARCH_ALGO}" \
  --max-run-hours "${MAX_RUN_HOURS}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT}" \
  "${EXTRA_ARGS[@]}"