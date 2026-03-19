#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3"
MAX_EVALS="7"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="12000"
DRY_RUN="${DRY_RUN:-false}"
EVAL_LOGGING_TIMING="${EVAL_LOGGING_TIMING:-final_only}"
FEATURE_ABLATION_LOGGING="${FEATURE_ABLATION_LOGGING:-false}"
MANIFEST_OUT=""
ONLY_RUN_PHASE=""
AXIS_FILTER=""
COMBO_FILTER=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 12000] [--manifest-out path] [--dry-run]
          [--only R_base_C1,K_12e_top6_C2] [--axis R|K] [--combo C1,C2]
          [--eval-logging-timing final_only|per_eval] [--feature-ablation-logging]

Phase4 defaults:
- max-evals=10
- tune-epochs=100
- tune-patience=10
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
    --only)
      ONLY_RUN_PHASE="$2"
      shift 2
      ;;
    --axis)
      AXIS_FILTER="$2"
      shift 2
      ;;
    --combo)
      COMBO_FILTER="$2"
      shift 2
      ;;
    --eval-logging-timing)
      EVAL_LOGGING_TIMING="$2"
      shift 2
      ;;
    --feature-ablation-logging)
      FEATURE_ABLATION_LOGGING="true"
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
    "${SCRIPT_DIR}/run_phase4_residual_topk.py"
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
  if [ -n "${ONLY_RUN_PHASE}" ]; then
    CMD+=(--only "${ONLY_RUN_PHASE}")
  fi
  if [ -n "${AXIS_FILTER}" ]; then
    CMD+=(--axis-filter "${AXIS_FILTER}")
  fi
  if [ -n "${COMBO_FILTER}" ]; then
    CMD+=(--combo "${COMBO_FILTER}")
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    CMD+=(--dry-run)
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
