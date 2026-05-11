#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-stage1}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
DATASETS="${DATASETS:-beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-2}"
OOM_RETRY_LIMIT="${OOM_RETRY_LIMIT:-2}"
BASELINE_MODELS="fearec,duorec,fame,fdsa,gru4rec"

run_build() {
  echo "[final_experiment][server_b] build-space"
  python3 "${SCRIPT_DIR}/build_space_manifest.py"
}

run_build_with_args() {
  echo "[final_experiment][server_b] build-space extra_args=${EXTRA_ARGS[*]:-}"
  python3 "${SCRIPT_DIR}/build_space_manifest.py" "${EXTRA_ARGS[@]}"
}

run_stage1() {
  echo "[final_experiment][server_b] stage1 datasets=${DATASETS} models=${BASELINE_MODELS} gpus=${GPU_LIST} max_run_hours=${MAX_RUN_HOURS} oom_retry_limit=${OOM_RETRY_LIMIT}"
  python3 "${SCRIPT_DIR}/stage1_broad_search.py" \
    --manifest "${SCRIPT_DIR}/space_manifest.json" \
    --datasets "${DATASETS}" \
    --models "${BASELINE_MODELS}" \
    --gpus "${GPU_LIST}" \
    --search-algo tpe \
    --max-run-hours "${MAX_RUN_HOURS}" \
    --oom-retry-limit "${OOM_RETRY_LIMIT}" \
    "${EXTRA_ARGS[@]}"
}

run_stage2() {
  echo "[final_experiment][server_b] stage2 datasets=${DATASETS} models=${BASELINE_MODELS} gpus=${GPU_LIST} max_run_hours=${MAX_RUN_HOURS} oom_retry_limit=${OOM_RETRY_LIMIT}"
  python3 "${SCRIPT_DIR}/stage2_focus_search.py" \
    --manifest "${SCRIPT_DIR}/space_manifest.json" \
    --datasets "${DATASETS}" \
    --models "${BASELINE_MODELS}" \
    --gpus "${GPU_LIST}" \
    --search-algo tpe \
    --max-run-hours "${MAX_RUN_HOURS}" \
    --oom-retry-limit "${OOM_RETRY_LIMIT}" \
    "${EXTRA_ARGS[@]}"
}

run_stage3() {
  echo "[final_experiment][server_b] stage3 datasets=${DATASETS} models=${BASELINE_MODELS} gpus=${GPU_LIST} max_run_hours=${MAX_RUN_HOURS} oom_retry_limit=${OOM_RETRY_LIMIT}"
  python3 "${SCRIPT_DIR}/stage3_seed_confirm.py" \
    --manifest "${SCRIPT_DIR}/space_manifest.json" \
    --datasets "${DATASETS}" \
    --models "${BASELINE_MODELS}" \
    --gpus "${GPU_LIST}" \
    --search-algo tpe \
    --max-run-hours "${MAX_RUN_HOURS}" \
    --oom-retry-limit "${OOM_RETRY_LIMIT}" \
    "${EXTRA_ARGS[@]}"
}

case "${STAGE}" in
  build-space)
    run_build_with_args
    ;;
  stage1)
    run_build
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  stage3)
    run_stage3
    ;;
  all)
    run_build
    run_stage1
    run_stage2
    run_stage3
    ;;
  *)
    echo "usage: $0 [build-space|stage1|stage2|stage3|all]" >&2
    exit 1
    ;;
esac
