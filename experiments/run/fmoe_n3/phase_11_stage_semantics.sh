#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="4,5,6,7"
HPARAMS="1"
SEEDS="1"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="71000"

FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.1"
Z_LOSS_LAMBDA="1e-4"
BALANCE_LOSS_LAMBDA="0.0"
MACRO_HISTORY_WINDOW="5"

ONLY_SETTING=""
MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="2"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2] [--gpus 4,5,6,7]
          [--hparams 1] [--seeds 1]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10] [--seed-base 71000]
          [--feature-group-bias-lambda 0.05] [--rule-bias-scale 0.1]
          [--z-loss-lambda 1e-4] [--balance-loss-lambda 0.0] [--macro-history-window 5]
          [--only-setting P11-00_MACRO_MID_MICRO,P11-23_LAYER2_MACRO_MID_MICRO]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 2]
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
    --hparams)
      HPARAMS="$2"
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
    "${SCRIPT_DIR}/run_phase11_stage_semantics.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --hparams "${HPARAMS}"
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
    --smoke-max-runs "${SMOKE_MAX_RUNS}"
  )

  if [ -n "${ONLY_SETTING}" ]; then
    CMD+=(--only-setting "${ONLY_SETTING}")
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

echo "[All Done] phase11 stage semantics datasets completed in order: ${DATASETS}"
