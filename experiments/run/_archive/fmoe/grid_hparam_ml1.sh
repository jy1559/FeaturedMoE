#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="movielens1m"
GPU_LIST="0"
LAYOUTS="1,1,1,1,0;1,1,0,0,0;0,1,1,1,0;2,0,2,-1,0"
SCHEDULE_PRESETS="off,alpha_mild,temp_mild,topk_mild"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
SEARCH_PROFILE="narrow_ml1"
LOG_WANDB="false"
TRAIN_BATCH_SIZE="24576"
EVAL_BATCH_SIZE="24576"
PHASE_PREFIX="P1GRID"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1] [--layouts "v1;v2;v3"] [--schedule-presets off,alpha_mild,temp_mild,topk_mild]
          [--max-evals N] [--tune-epochs N] [--tune-patience N] [--seed N] [--search-profile narrow_ml1|wide]
          [--train-batch-size N] [--eval-batch-size N]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --layouts) LAYOUTS="$2"; shift 2 ;;
    --schedule-presets) SCHEDULE_PRESETS="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }
dispatch_parse_csv "$SCHEDULE_PRESETS" SCHED_ARR
[ "${#SCHED_ARR[@]}" -eq 0 ] && { echo "Empty schedule preset list"; exit 1; }
IFS=';' read -r -a LAYOUT_ARR <<< "$LAYOUTS"
[ "${#LAYOUT_ARR[@]}" -eq 0 ] && { echo "Empty layout list"; exit 1; }

TOTAL=$(( ${#LAYOUT_ARR[@]} * ${#SCHED_ARR[@]} ))
echo "[GRID] dataset=${DATASET} combos=${TOTAL} layouts=${#LAYOUT_ARR[@]} schedules=${#SCHED_ARR[@]} gpus=${GPU_LIST}"
echo "[GRID] max_evals=${MAX_EVALS} tune_epochs=${TUNE_EPOCHS} tune_patience=${TUNE_PATIENCE} search_profile=${SEARCH_PROFILE} train_batch_size=${TRAIN_BATCH_SIZE:-default(config)} eval_batch_size=${EVAL_BATCH_SIZE:-default(config)}"

INTERRUPTED="false"
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping all dispatched jobs..."
  dispatch_terminate_all GPUS
  exit 130
}
on_exit() {
  local rc=$?
  if [ "$INTERRUPTED" = "false" ] && [ "$rc" -ne 0 ]; then
    echo "[EXIT] non-zero exit (${rc}) -> cleaning up child jobs..."
    dispatch_terminate_all GPUS
  fi
}
trap on_interrupt INT TERM
trap on_exit EXIT

idx=0
for li in "${!LAYOUT_ARR[@]}"; do
  layout_vec="${LAYOUT_ARR[$li]}"
  for sched in "${SCHED_ARR[@]}"; do
    dispatch_wait_for_gpu GPUS
    idx=$((idx + 1))
    phase="${PHASE_PREFIX}_L$((li + 1))_${sched}"

    cmd=(
      bash "${SCRIPT_DIR}/tune_hparam.sh"
      --dataset "$DATASET"
      --layout-vector "$layout_vec"
      --schedule-preset "$sched"
      --gpu "$FREE_GPU"
      --max-evals "$MAX_EVALS"
      --tune-epochs "$TUNE_EPOCHS"
      --tune-patience "$TUNE_PATIENCE"
      --seed "$SEED"
      --search-profile "$SEARCH_PROFILE"
      --phase "$phase"
    )
    if [ -n "$TRAIN_BATCH_SIZE" ]; then
      cmd+=(--train-batch-size "$TRAIN_BATCH_SIZE")
    fi
    if [ -n "$EVAL_BATCH_SIZE" ]; then
      cmd+=(--eval-batch-size "$EVAL_BATCH_SIZE")
    fi
    if [ "$LOG_WANDB" = "true" ]; then
      cmd+=(--log-wandb)
    else
      cmd+=(--no-wandb)
    fi
    if [ "$DRY_RUN" = "true" ]; then
      cmd+=(--dry-run)
    fi

    echo "[${idx}/${TOTAL}] gpu=${FREE_GPU} phase=${phase} layout=[${layout_vec}] sched=${sched}"
    if [ "$DRY_RUN" = "true" ]; then
      echo "[DRY_RUN] $(run_cmd_str "${cmd[@]}")"
      continue
    fi

    setsid "${cmd[@]}" &
    dispatch_set_pid "$FREE_GPU" "$!"
  done
done

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi
dispatch_wait_all
trap - INT TERM EXIT
