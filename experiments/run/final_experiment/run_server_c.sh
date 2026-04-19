#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-stage1}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
ALL_DATASETS="${DATASETS:-beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket}"
FAST_DATASETS="${FAST_DATASETS:-beauty,foursquare,KuaiRecLargeStrictPosV2_0.2}"
SLOW_DATASETS="${SLOW_DATASETS:-lastfm0.03,movielens1m,retail_rocket}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-2}"
OOM_RETRY_LIMIT="${OOM_RETRY_LIMIT:-5}"
SKIP_BUILD_SPACE="${SKIP_BUILD_SPACE:-0}"
ROUTE_MODEL="featured_moe_n3"

run_build() {
  if [[ "${SKIP_BUILD_SPACE}" == "1" ]]; then
    echo "[final_experiment][server_c] build-space skipped (SKIP_BUILD_SPACE=1)"
    return
  fi
  echo "[final_experiment][server_c] build-space"
  python3 "${SCRIPT_DIR}/build_space_manifest.py"
}

run_build_with_args() {
  if [[ "${SKIP_BUILD_SPACE}" == "1" ]]; then
    echo "[final_experiment][server_c] build-space skipped (SKIP_BUILD_SPACE=1)"
    return
  fi
  echo "[final_experiment][server_c] build-space extra_args=${EXTRA_ARGS[*]:-}"
  python3 "${SCRIPT_DIR}/build_space_manifest.py" "${EXTRA_ARGS[@]}"
}

run_stage() {
  local stage_name="$1"
  local datasets="$2"
  local note="$3"
  local launcher=""
  case "${stage_name}" in
    stage1) launcher="stage1_broad_search.py" ;;
    stage2) launcher="stage2_focus_search.py" ;;
    stage3) launcher="stage3_seed_confirm.py" ;;
    *) echo "unknown stage ${stage_name}" >&2; exit 1 ;;
  esac
  echo "[final_experiment][server_c] ${stage_name} datasets=${datasets} models=${ROUTE_MODEL} gpus=${GPU_LIST} max_run_hours=${MAX_RUN_HOURS} oom_retry_limit=${OOM_RETRY_LIMIT} note=${note}"
  python3 "${SCRIPT_DIR}/${launcher}" \
    --manifest "${SCRIPT_DIR}/space_manifest.json" \
    --datasets "${datasets}" \
    --models "${ROUTE_MODEL}" \
    --gpus "${GPU_LIST}" \
    --search-algo tpe \
    --max-run-hours "${MAX_RUN_HOURS}" \
    --oom-retry-limit "${OOM_RETRY_LIMIT}" \
    "${EXTRA_ARGS[@]}"
}

run_pipeline() {
  run_build
  run_stage stage1 "${FAST_DATASETS}" "history-biased broad screen on fast/core datasets"
  run_stage stage2 "${FAST_DATASETS}" "family-local refinement on fast/core datasets from stage1 winners"
  run_stage stage1 "${SLOW_DATASETS}" "history-biased broad screen on slow datasets"
  run_stage stage2 "${SLOW_DATASETS}" "family-local refinement on slow datasets from stage1 winners"
  run_stage stage2 "${ALL_DATASETS}" "safety pass to materialize all stage2 winners from accumulated stage1 results"
  run_stage stage3 "${ALL_DATASETS}" "seed confirm top route configs with optional history challenger when margin is small"
}

case "${STAGE}" in
  build-space)
    run_build_with_args
    ;;
  stage1)
    run_build
    run_stage stage1 "${ALL_DATASETS}" "broad FMoE family bank over all datasets"
    ;;
  stage1-fast)
    run_build
    run_stage stage1 "${FAST_DATASETS}" "broad FMoE family bank over fast/core datasets"
    ;;
  stage1-slow)
    run_build
    run_stage stage1 "${SLOW_DATASETS}" "broad FMoE family bank over slow datasets"
    ;;
  stage2)
    run_stage stage2 "${ALL_DATASETS}" "narrow around stage1 top-4 FMoE configs"
    ;;
  stage2-fast)
    run_stage stage2 "${FAST_DATASETS}" "narrow around stage1 top-4 FMoE configs on fast/core datasets"
    ;;
  stage2-slow)
    run_stage stage2 "${SLOW_DATASETS}" "narrow around stage1 top-4 FMoE configs on slow datasets"
    ;;
  stage3)
    run_stage stage3 "${ALL_DATASETS}" "seed-confirm top-2 FMoE configs per dataset"
    ;;
  stage3-fast)
    run_stage stage3 "${FAST_DATASETS}" "seed-confirm top-2 FMoE configs on fast/core datasets"
    ;;
  stage3-slow)
    run_stage stage3 "${SLOW_DATASETS}" "seed-confirm top-2 FMoE configs on slow datasets"
    ;;
  all)
    run_pipeline
    ;;
  pipeline)
    run_pipeline
    ;;
  *)
    echo "usage: $0 [build-space|stage1|stage1-fast|stage1-slow|stage2|stage2-fast|stage2-slow|stage3|stage3-fast|stage3-slow|all|pipeline]" >&2
    exit 1
    ;;
esac
