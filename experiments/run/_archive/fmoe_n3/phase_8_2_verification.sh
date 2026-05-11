#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="4,5,6,7"
SEEDS="1,2,3,4"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="38000"
FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.1"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 4,5,6,7]
          [--seeds 1,2,3,4]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 38000]
          [--feature-group-bias-lambda 0.05] [--rule-bias-scale 0.1]
          [--dry-run] [--smoke-test]
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --max-evals)
      MAX_EVALS="$2"
      shift 2
      ;;
    --tune-epochs)
      TUNE_EPOCHS="$2"
      shift 2
      ;;
    --tune-patience)
      TUNE_PATIENCE="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --feature-group-bias-lambda)
      FEATURE_GROUP_BIAS_LAMBDA="$2"
      shift 2
      ;;
    --rule-bias-scale)
      RULE_BIAS_SCALE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --smoke-test)
      SMOKE_TEST="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"
IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

dataset_idx=0
for DATASET in "${DATASET_LIST[@]}"; do
  DATASET="$(echo "${DATASET}" | xargs)"
  if [ -z "${DATASET}" ]; then
    continue
  fi

  CUR_SEED_BASE=$((SEED_BASE + dataset_idx * 1000))

  CMD=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/run_phase8_2_verification.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --seeds "${SEEDS}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
    --feature-group-bias-lambda "${FEATURE_GROUP_BIAS_LAMBDA}"
    --rule-bias-scale "${RULE_BIAS_SCALE}"
  )

  if [ "${DRY_RUN}" = "true" ]; then
    CMD+=(--dry-run)
  fi
  if [ "${SMOKE_TEST}" = "true" ]; then
    CMD+=(--smoke-test)
  fi

  echo "[Dataset ${dataset_idx}] start: ${DATASET}"
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
  echo "[Dataset ${dataset_idx}] done: ${DATASET}"

  dataset_idx=$((dataset_idx + 1))
done

echo "[All Done] phase8_2 verification datasets completed in order: ${DATASETS}"
