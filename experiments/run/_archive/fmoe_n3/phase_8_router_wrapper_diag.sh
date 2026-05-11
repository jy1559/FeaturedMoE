#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="4,5,6,7"
SCREENING_SEEDS="1"
CONFIRM_SEEDS="1,2,3"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="28000"
TOP_A="4"
TOP_B="4"
TOP_C="3"
TOP_D="3"
FIXED_WEIGHT_DECAY="1e-6"
FIXED_HIDDEN_DROPOUT_PROB="0.15"
FEATURE_GROUP_BIAS_LAMBDA="0.05"
RULE_BIAS_SCALE="0.1"
STOP_AFTER_STAGE="none"
ONLY_A_CANDIDATES=""
MANIFEST_OUT=""
RUN_SCREENING="true"
RUN_CONFIRM="true"
RESUME_FROM_LOGS="true"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 4,5,6,7]
          [--screening-seeds 1] [--confirm-seeds 1,2,3]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--top-a 4] [--top-b 4] [--top-c 3] [--top-d 3]
          [--fixed-weight-decay 1e-6] [--fixed-hidden-dropout-prob 0.15]
          [--feature-group-bias-lambda 0.05] [--rule-bias-scale 0.1]
          [--only-a-candidates all_w1,all_w4,mixed_2]
          [--stop-after-stage none|A|B|C|D|confirm]
          [--run-screening|--no-run-screening]
          [--run-confirm|--no-run-confirm]
          [--resume-from-logs|--no-resume-from-logs]
          [--manifest-out path] [--dry-run] [--smoke-test]
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
    --screening-seeds)
      SCREENING_SEEDS="$2"
      shift 2
      ;;
    --confirm-seeds)
      CONFIRM_SEEDS="$2"
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
    --top-a)
      TOP_A="$2"
      shift 2
      ;;
    --top-b)
      TOP_B="$2"
      shift 2
      ;;
    --top-c)
      TOP_C="$2"
      shift 2
      ;;
    --top-d)
      TOP_D="$2"
      shift 2
      ;;
    --fixed-weight-decay)
      FIXED_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --fixed-hidden-dropout-prob)
      FIXED_HIDDEN_DROPOUT_PROB="$2"
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
    --stop-after-stage)
      STOP_AFTER_STAGE="$2"
      shift 2
      ;;
    --only-a-candidates)
      ONLY_A_CANDIDATES="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --run-screening)
      RUN_SCREENING="true"
      shift
      ;;
    --no-run-screening)
      RUN_SCREENING="false"
      shift
      ;;
    --run-confirm)
      RUN_CONFIRM="true"
      shift
      ;;
    --no-run-confirm)
      RUN_CONFIRM="false"
      shift
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
    "${SCRIPT_DIR}/run_phase8_router_wrapper_diag.py"
    --dataset "${DATASET}"
    --gpus "${GPU_LIST}"
    --screening-seeds "${SCREENING_SEEDS}"
    --confirm-seeds "${CONFIRM_SEEDS}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed-base "${CUR_SEED_BASE}"
    --top-a "${TOP_A}"
    --top-b "${TOP_B}"
    --top-c "${TOP_C}"
    --top-d "${TOP_D}"
    --fixed-weight-decay "${FIXED_WEIGHT_DECAY}"
    --fixed-hidden-dropout-prob "${FIXED_HIDDEN_DROPOUT_PROB}"
    --feature-group-bias-lambda "${FEATURE_GROUP_BIAS_LAMBDA}"
    --rule-bias-scale "${RULE_BIAS_SCALE}"
    --stop-after-stage "${STOP_AFTER_STAGE}"
  )

  if [ -n "${ONLY_A_CANDIDATES}" ]; then
    CMD+=(--only-a-candidates "${ONLY_A_CANDIDATES}")
  fi
  if [ -n "${MANIFEST_OUT}" ]; then
    CMD+=(--manifest-out "${MANIFEST_OUT}_${DATASET}")
  fi

  if [ "${RUN_SCREENING}" = "true" ]; then
    CMD+=(--run-screening)
  else
    CMD+=(--no-run-screening)
  fi
  if [ "${RUN_CONFIRM}" = "true" ]; then
    CMD+=(--run-confirm)
  else
    CMD+=(--no-run-confirm)
  fi
  if [ "${RESUME_FROM_LOGS}" = "true" ]; then
    CMD+=(--resume-from-logs)
  else
    CMD+=(--no-resume-from-logs)
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

echo "[All Done] phase8 datasets completed in order: ${DATASETS}"
