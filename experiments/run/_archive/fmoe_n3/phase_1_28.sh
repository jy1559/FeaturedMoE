#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2,lastfm0.03"
GPU_LIST="0,1,2,3"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="9800"
DRY_RUN="${DRY_RUN:-false}"
USE_RECOMMENDED_BUDGET="${USE_RECOMMENDED_BUDGET:-false}"
EVAL_LOGGING_TIMING="${EVAL_LOGGING_TIMING:-final_only}"
FEATURE_ABLATION_LOGGING="${FEATURE_ABLATION_LOGGING:-false}"
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3]
          [--max-evals 20] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 9800] [--manifest-out path] [--dry-run]
          [--use-recommended-budget]
          [--eval-logging-timing final_only|per_eval]
          [--feature-ablation-logging]

Runs datasets in order. Dataset A must finish before dataset B starts.
Default order: KuaiRecLargeStrictPosV2_0.2 -> lastfm0.03
`--use-recommended-budget` is opt-in. Without it, `--max-evals` is used as-is.
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
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --use-recommended-budget)
      USE_RECOMMENDED_BUDGET="true"
      shift
      ;;
    --eval-logging-timing)
      EVAL_LOGGING_TIMING="$2"
      shift 2
      ;;
    --feature-ablation-logging)
      FEATURE_ABLATION_LOGGING="true"
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
    "${SCRIPT_DIR}/run_phase1_28.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
    --eval-logging-timing "${EVAL_LOGGING_TIMING}"
  )

  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    CMD+=(--dry-run)
  fi
  if [ "${USE_RECOMMENDED_BUDGET}" = "true" ]; then
    CMD+=(--use-recommended-budget)
  fi
  if [ "${FEATURE_ABLATION_LOGGING}" = "true" ]; then
    CMD+=(--feature-ablation-logging)
  fi

  echo "[Dataset ${dataset_idx}] start: ${DATASET}"
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
  echo "[Dataset ${dataset_idx}] done: ${DATASET}"

  dataset_idx=$((dataset_idx + 1))
done

echo "[All Done] datasets completed in order: ${DATASETS}"
