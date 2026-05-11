#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3"
MAX_EVALS="30"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="10300"
DRY_RUN="${DRY_RUN:-false}"
USE_RECOMMENDED_BUDGET="${USE_RECOMMENDED_BUDGET:-false}"
EVAL_LOGGING_TIMING="${EVAL_LOGGING_TIMING:-final_only}"
FEATURE_ABLATION_LOGGING="${FEATURE_ABLATION_LOGGING:-false}"
ONLY_COMBOS=""
FAMILY_FILTER=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3]
          [--max-evals 0] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 10300] [--manifest-out path] [--dry-run]
          [--only P3S1_01,P3S2_01] [--family S1,S2]
          [--use-recommended-budget]
          [--eval-logging-timing final_only|per_eval]
          [--feature-ablation-logging]

P3 design:
- 4 structures x 5 combos = 20 total
- S1: standard + all gated_bias
- S2: factored + group_gated_bias
- S3: feature-source routing
- S4: deep layer-prefix surprise
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
      ONLY_COMBOS="$2"
      shift 2
      ;;
    --family)
      FAMILY_FILTER="$2"
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
    "${SCRIPT_DIR}/run_phase3_20.py"
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
  if [ -n "${ONLY_COMBOS}" ]; then
    CMD+=(--only "${ONLY_COMBOS}")
  fi
  if [ -n "${FAMILY_FILTER}" ]; then
    CMD+=(--family "${FAMILY_FILTER}")
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
