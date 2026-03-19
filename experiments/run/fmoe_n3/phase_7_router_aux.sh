#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS="1,2,3,4"
GROUP="all"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="19000"
DRY_RUN="${DRY_RUN:-false}"
ONLY_SETTING=""
MANIFEST_OUT=""
RESUME_FROM_LOGS="true"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3,4,5,6,7]
          [--seeds 1,2,3,4] [--group all|router|aux]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 19000] [--only-setting R0_STD,AUX_R0_STD_BAL_A]
          [--resume-from-logs|--no-resume-from-logs]
          [--manifest-out path] [--dry-run]
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
    --group)
      GROUP="$2"
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
    --only-setting)
      ONLY_SETTING="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --resume-from-logs)
      RESUME_FROM_LOGS="true"
      shift
      ;;
    --no-resume-from-logs)
      RESUME_FROM_LOGS="false"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
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
    "${SCRIPT_DIR}/run_phase7_router_aux.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --seeds "${SEEDS}"
    --group "${GROUP}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
  )

  if [ -n "${ONLY_SETTING}" ]; then
    CMD+=(--only-setting "${ONLY_SETTING}")
  fi
  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    CMD+=(--dry-run)
  fi
  if [ "${RESUME_FROM_LOGS}" = "true" ]; then
    CMD+=(--resume-from-logs)
  else
    CMD+=(--no-resume-from-logs)
  fi

  echo "[Dataset ${dataset_idx}] start: ${DATASET}"
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
  echo "[Dataset ${dataset_idx}] done: ${DATASET}"

  dataset_idx=$((dataset_idx + 1))
done

echo "[All Done] phase7 datasets completed in order: ${DATASETS}"
