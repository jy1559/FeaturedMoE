#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi
PYTHON_BIN="${RUN_PYTHON_BIN:-python}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SEEDS="${SEEDS:-1}"
ARCHITECTURES="${ARCHITECTURES:-A8,A10,A11,A12}"

# 8-way retail bank:
# - keep proven retail winners H1/H2/H3
# - widen to lighter/smaller and moderate-width variants without jumping to
#   the largest memory-heavy configs that previously OOMed in wrapper sweep
COMMON_HPARAMS="${COMMON_HPARAMS:-H2,H3,H6,H1,H4,H8,H9}"
DEFAULT_OUTLIER_HPARAM="${DEFAULT_OUTLIER_HPARAM:-H12}"

MAX_EVALS="${MAX_EVALS:-10}"
TUNE_EPOCHS="${TUNE_EPOCHS:-80}"
TUNE_PATIENCE="${TUNE_PATIENCE:-8}"

# Retail wrapper runs were OOM-prone at 8192/12288. Use a more conservative
# batch profile, roughly aligned with strong retail baseline scales.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-3072}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"

# Baseline retail winners concentrated near ~5e-4 to ~7e-4; keep that region
# well covered while still allowing some higher-LR exploration in 10 trials.
SEARCH_LR_MIN="${SEARCH_LR_MIN:-2.5e-4}"
SEARCH_LR_MAX="${SEARCH_LR_MAX:-2.8e-3}"
SEARCH_LR_SCHEDULER="${SEARCH_LR_SCHEDULER:-warmup_cosine}"

DRY_RUN="${DRY_RUN:-false}"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --architectures) ARCHITECTURES="$2"; shift 2 ;;
    --common-hparams) COMMON_HPARAMS="$2"; shift 2 ;;
    --default-outlier-hparam) DEFAULT_OUTLIER_HPARAM="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --search-lr-min) SEARCH_LR_MIN="$2"; shift 2 ;;
    --search-lr-max) SEARCH_LR_MAX="$2"; shift 2 ;;
    --search-lr-scheduler) SEARCH_LR_SCHEDULER="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

CSV_COUNT() {
  local raw="$1"
  local -a items=()
  IFS=',' read -r -a items <<< "${raw}"
  echo "${#items[@]}"
}

HPARAM_COUNT="$(CSV_COUNT "${COMMON_HPARAMS}")"
TOTAL_RUNS="$(( $(CSV_COUNT "${ARCHITECTURES}") * $(CSV_COUNT "${SEEDS}") * HPARAM_COUNT ))"

export SLACK_NOTIFY_TOTAL_RUNS="${SLACK_NOTIFY_TOTAL_RUNS:-${TOTAL_RUNS}}"
export SLACK_NOTIFY_SCOPE_LABEL="${SLACK_NOTIFY_SCOPE_LABEL:-retail-wrapper-recovery-8h}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_final_a8_a12_wrapper_sweep.py"
  --datasets retail_rocket
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --architectures "${ARCHITECTURES}"
  --common-hparams "${COMMON_HPARAMS}"
  --default-outlier-hparam "${DEFAULT_OUTLIER_HPARAM}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --dataset-batch-sizes "retail_rocket:${TRAIN_BATCH_SIZE}"
  --dataset-eval-batch-sizes "retail_rocket:${EVAL_BATCH_SIZE}"
  --search-lr-min "${SEARCH_LR_MIN}"
  --search-lr-max "${SEARCH_LR_MAX}"
  --search-lr-scheduler "${SEARCH_LR_SCHEDULER}"
  --no-resume-from-logs
  --no-migrate-existing-layout
)

if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[Retail Wrapper Recovery 8H] total_runs=${TOTAL_RUNS} gpus=${GPU_LIST} hparams=${COMMON_HPARAMS},${DEFAULT_OUTLIER_HPARAM} lr=[${SEARCH_LR_MIN},${SEARCH_LR_MAX}] batch=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
run_echo_cmd "${CMD[@]}"
"${CMD[@]}"
