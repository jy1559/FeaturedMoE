#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3"
SUITES="all"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="16000"
DRY_RUN="${DRY_RUN:-false}"
ONLY_RUN_PHASE=""
CATEGORY_FILTER=""
MANIFEST_OUT=""
FEATURE_ABLATION_LOGGING="false"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3]
          [--suites all|candidate,router,spec,feature,base]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 16000] [--manifest-out path] [--dry-run]
          [--only P6_CAND_A_S1,P6_SPEC_B_M2] [--category cand3x,spec_ablation]
          [--feature-ablation-logging|--no-feature-ablation-logging]

Phase6 defaults:
- max-evals=10 (candidate suite rows use 3x max-evals automatically)
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
    --suites)
      SUITES="$2"
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
    --category)
      CATEGORY_FILTER="$2"
      shift 2
      ;;
    --feature-ablation-logging)
      FEATURE_ABLATION_LOGGING="true"
      shift
      ;;
    --no-feature-ablation-logging)
      FEATURE_ABLATION_LOGGING="false"
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
    "${SCRIPT_DIR}/run_phase6_candidate_reinfor.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --suites "${SUITES}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
  )

  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
  fi
  if [ -n "${ONLY_RUN_PHASE}" ]; then
    CMD+=(--only "${ONLY_RUN_PHASE}")
  fi
  if [ -n "${CATEGORY_FILTER}" ]; then
    CMD+=(--category "${CATEGORY_FILTER}")
  fi
  if [ "${FEATURE_ABLATION_LOGGING}" = "true" ]; then
    CMD+=(--feature-ablation-logging)
  else
    CMD+=(--no-feature-ablation-logging)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    CMD+=(--dry-run)
  fi

  echo "[Dataset ${dataset_idx}] start: ${DATASET}"
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
  echo "[Dataset ${dataset_idx}] done: ${DATASET}"

  dataset_idx=$((dataset_idx + 1))
done

echo "[All Done] phase6 datasets completed in order: ${DATASETS}"
