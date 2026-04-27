#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-pipeline}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")

TARGET_DATASETS="${TARGET_DATASETS:-foursquare,lastfm0.03,movielens1m}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
TRACK_NAME="${FINAL_EXP_TRACK:-final_experiment_target3_6h}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-1.5}"
OOM_RETRY_LIMIT="${OOM_RETRY_LIMIT:-4}"
SKIP_BUILD_SPACE="${SKIP_BUILD_SPACE:-0}"
ROUTE_MODEL="featured_moe_n3"

MANIFEST_PATH="${MANIFEST_PATH:-${SCRIPT_DIR}/space_manifest_target3_6h.json}"
TUNING_SPACE_PATH="${TUNING_SPACE_PATH:-${SCRIPT_DIR}/tuning_space_target3_6h.csv}"
SERVER_SPLIT_PATH="${SERVER_SPLIT_PATH:-${SCRIPT_DIR}/server_split_target3_6h.json}"

export FINAL_EXP_TRACK="${TRACK_NAME}"
export FINAL_EXP_EXTRA_HISTORY_TRACKS="${FINAL_EXP_EXTRA_HISTORY_TRACKS:-final_experiment}"
export FINAL_EXP_STAGE1_ROUTE_BANK_COUNTS="${FINAL_EXP_STAGE1_ROUTE_BANK_COUNTS:-foursquare=8,lastfm0.03=6,movielens1m=6}"
export FINAL_EXP_STAGE1_ROUTE_MAX_EVALS="${FINAL_EXP_STAGE1_ROUTE_MAX_EVALS:-foursquare=8,lastfm0.03=6,movielens1m=6}"
export FINAL_EXP_STAGE2_MAX_EVALS="${FINAL_EXP_STAGE2_MAX_EVALS:-foursquare=6,lastfm0.03=5,movielens1m=5}"
export FINAL_EXP_STAGE3_SEED_COUNTS="${FINAL_EXP_STAGE3_SEED_COUNTS:-foursquare=2,lastfm0.03=2,movielens1m=2}"
export FINAL_EXP_ROUTE_STAGE1_MIN_PRESET_ANCHORS="${FINAL_EXP_ROUTE_STAGE1_MIN_PRESET_ANCHORS:-foursquare=1,lastfm0.03=2,movielens1m=1}"
export FINAL_EXP_ROUTE_STAGE1_MIN_NEIGHBOR_PROBES="${FINAL_EXP_ROUTE_STAGE1_MIN_NEIGHBOR_PROBES:-foursquare=1,lastfm0.03=1,movielens1m=1}"
export FINAL_EXP_ROUTE_STAGE2_TARGET_FAMILIES="${FINAL_EXP_ROUTE_STAGE2_TARGET_FAMILIES:-foursquare=3,lastfm0.03=3,movielens1m=3}"
export FINAL_EXP_ROUTE_STAGE2_HISTORY_CHALLENGERS="${FINAL_EXP_ROUTE_STAGE2_HISTORY_CHALLENGERS:-foursquare=1,lastfm0.03=0,movielens1m=1}"
export FINAL_EXP_ROUTE_STAGE2_TRIALS_PER_FAMILY="${FINAL_EXP_ROUTE_STAGE2_TRIALS_PER_FAMILY:-foursquare=1,lastfm0.03=1,movielens1m=1}"
export FINAL_EXP_ROUTE_STAGE3_MAX_CONFIGS="${FINAL_EXP_ROUTE_STAGE3_MAX_CONFIGS:-foursquare=2,lastfm0.03=2,movielens1m=2}"
export FINAL_EXP_STAGE1_TUNE_EPOCHS="${FINAL_EXP_STAGE1_TUNE_EPOCHS:-24}"
export FINAL_EXP_STAGE2_TUNE_EPOCHS="${FINAL_EXP_STAGE2_TUNE_EPOCHS:-40}"
export FINAL_EXP_STAGE3_TUNE_EPOCHS="${FINAL_EXP_STAGE3_TUNE_EPOCHS:-60}"
export FINAL_EXP_STAGE1_TUNE_PATIENCE="${FINAL_EXP_STAGE1_TUNE_PATIENCE:-4}"
export FINAL_EXP_STAGE2_TUNE_PATIENCE="${FINAL_EXP_STAGE2_TUNE_PATIENCE:-5}"
export FINAL_EXP_STAGE3_TUNE_PATIENCE="${FINAL_EXP_STAGE3_TUNE_PATIENCE:-6}"

run_build() {
  if [[ "${SKIP_BUILD_SPACE}" == "1" ]]; then
    echo "[final_experiment][target3_6h] build-space skipped (SKIP_BUILD_SPACE=1)"
    return
  fi
  echo "[final_experiment][target3_6h] build-space track=${FINAL_EXP_TRACK} manifest=${MANIFEST_PATH}"
  python3 "${SCRIPT_DIR}/build_space_manifest.py" \
    --manifest-out "${MANIFEST_PATH}" \
    --tuning-space-out "${TUNING_SPACE_PATH}" \
    --server-split-out "${SERVER_SPLIT_PATH}"
}

run_stage() {
  local stage_name="$1"
  local launcher=""
  case "${stage_name}" in
    stage1) launcher="stage1_broad_search.py" ;;
    stage2) launcher="stage2_focus_search.py" ;;
    stage3) launcher="stage3_seed_confirm.py" ;;
    *) echo "unknown stage ${stage_name}" >&2; exit 1 ;;
  esac
  echo "[final_experiment][target3_6h] ${stage_name} datasets=${TARGET_DATASETS} gpus=${GPU_LIST} track=${FINAL_EXP_TRACK} max_run_hours=${MAX_RUN_HOURS}"
  python3 "${SCRIPT_DIR}/${launcher}" \
    --manifest "${MANIFEST_PATH}" \
    --datasets "${TARGET_DATASETS}" \
    --models "${ROUTE_MODEL}" \
    --gpus "${GPU_LIST}" \
    --search-algo tpe \
    --max-run-hours "${MAX_RUN_HOURS}" \
    --oom-retry-limit "${OOM_RETRY_LIMIT}" \
    "${EXTRA_ARGS[@]}"
}

run_pipeline() {
  run_build
  run_stage stage1
  run_stage stage2
  run_stage stage3
}

case "${STAGE}" in
  build-space)
    run_build
    ;;
  stage1)
    run_build
    run_stage stage1
    ;;
  stage2)
    run_stage stage2
    ;;
  stage3)
    run_stage stage3
    ;;
  all|pipeline)
    run_pipeline
    ;;
  *)
    echo "usage: $0 [build-space|stage1|stage2|stage3|all|pipeline]" >&2
    exit 1
    ;;
esac