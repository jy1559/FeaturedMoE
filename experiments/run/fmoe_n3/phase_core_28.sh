#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="8300"
DRY_RUN="${DRY_RUN:-false}"
USE_RECOMMENDED_BUDGET="${USE_RECOMMENDED_BUDGET:-true}"
EVAL_LOGGING_TIMING="${EVAL_LOGGING_TIMING:-final_only}"
FEATURE_ABLATION_LOGGING="${FEATURE_ABLATION_LOGGING:-false}"
ONLY_COMBOS=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--dataset KuaiRecLargeStrictPosV2_0.2|lastfm0.03] [--gpus 0,1,2,3]
          [--max-evals 20] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 8300] [--manifest-out path] [--dry-run]
          [--only P00,D10,R30]
          [--use-recommended-budget]
          [--eval-logging-timing final_only|per_eval]
          [--feature-ablation-logging]

Default behavior:
  - USE_RECOMMENDED_BUDGET=true
  - recommended budget = every combo gets at least 3 evals
  - --max-evals is mainly used when recommended budget is disabled
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
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
CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_core_28.py"
  --dataset "${DATASET}"
  --gpus "${GPU_LIST}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --seed-base "${SEED_BASE}"
  --eval-logging-timing "${EVAL_LOGGING_TIMING}"
)

if [ -n "${MANIFEST_OUT}" ]; then
  CMD+=(--manifest-out "${MANIFEST_OUT}")
fi
if [ -n "${ONLY_COMBOS}" ]; then
  CMD+=(--only "${ONLY_COMBOS}")
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

run_echo_cmd "${CMD[@]}"
exec "${CMD[@]}"
