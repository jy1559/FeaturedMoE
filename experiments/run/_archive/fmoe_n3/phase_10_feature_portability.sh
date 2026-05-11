#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="4,5,6,7"
SEEDS="1"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="68000"

FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.1"
Z_LOSS_LAMBDA="1e-4"
BALANCE_LOSS_LAMBDA="0.0"
MACRO_HISTORY_WINDOW="5"

FAMILY_DROPOUT_PROB="0.10"
FEATURE_DROPOUT_PROB="0.15"
INCLUDE_EXTRA_24="true"
ONLY_SETTING=""
MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="2"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 4,5,6,7]
          [--seeds 1] [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 68000]
          [--feature-group-bias-lambda 0.05] [--rule-bias-scale 0.1]
          [--z-loss-lambda 1e-4] [--balance-loss-lambda 0.0] [--macro-history-window 5]
          [--family-dropout-prob 0.10] [--feature-dropout-prob 0.15]
          [--only-setting FULL,NO_TIMESTAMP]
          [--include-extra-24|--no-extra-24]
          [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--manifest-out path] [--dry-run] [--smoke-test] [--smoke-max-runs 2]
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
    --z-loss-lambda)
      Z_LOSS_LAMBDA="$2"
      shift 2
      ;;
    --balance-loss-lambda)
      BALANCE_LOSS_LAMBDA="$2"
      shift 2
      ;;
    --macro-history-window)
      MACRO_HISTORY_WINDOW="$2"
      shift 2
      ;;
    --family-dropout-prob)
      FAMILY_DROPOUT_PROB="$2"
      shift 2
      ;;
    --feature-dropout-prob)
      FEATURE_DROPOUT_PROB="$2"
      shift 2
      ;;
    --only-setting)
      ONLY_SETTING="$2"
      shift 2
      ;;
    --include-extra-24)
      INCLUDE_EXTRA_24="true"
      shift
      ;;
    --no-extra-24)
      INCLUDE_EXTRA_24="false"
      shift
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
    --verify-logging)
      VERIFY_LOGGING="true"
      shift
      ;;
    --no-verify-logging)
      VERIFY_LOGGING="false"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --smoke-test)
      SMOKE_TEST="true"
      shift
      ;;
    --smoke-max-runs)
      SMOKE_MAX_RUNS="$2"
      shift 2
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
    "${SCRIPT_DIR}/run_phase10_feature_portability.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --seeds "${SEEDS}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
    --feature-group-bias-lambda "${FEATURE_GROUP_BIAS_LAMBDA}"
    --rule-bias-scale "${RULE_BIAS_SCALE}"
    --z-loss-lambda "${Z_LOSS_LAMBDA}"
    --balance-loss-lambda "${BALANCE_LOSS_LAMBDA}"
    --macro-history-window "${MACRO_HISTORY_WINDOW}"
    --family-dropout-prob "${FAMILY_DROPOUT_PROB}"
    --feature-dropout-prob "${FEATURE_DROPOUT_PROB}"
    --smoke-max-runs "${SMOKE_MAX_RUNS}"
  )

  if [ -n "${ONLY_SETTING}" ]; then
    CMD+=(--only-setting "${ONLY_SETTING}")
  fi
  if [ "${INCLUDE_EXTRA_24}" = "true" ]; then
    CMD+=(--include-extra-24)
  else
    CMD+=(--no-extra-24)
  fi
  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
  fi
  if [ "${RESUME_FROM_LOGS}" = "true" ]; then
    CMD+=(--resume-from-logs)
  else
    CMD+=(--no-resume-from-logs)
  fi
  if [ "${VERIFY_LOGGING}" = "true" ]; then
    CMD+=(--verify-logging)
  else
    CMD+=(--no-verify-logging)
  fi
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

echo "[All Done] phase10 feature portability datasets completed in order: ${DATASETS}"
