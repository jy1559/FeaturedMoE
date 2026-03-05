#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

MODEL=""
DATASETS="movielens1m,amazon_beauty,foursquare,retail_rocket,kuairec0.3,lastfm0.3"
GPU_LIST="0"
MAX_EVALS=""
TUNE_EPOCHS=""
TUNE_PATIENCE=""
SEED="42"
LOG_WANDB="true"
PHASE="P1"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 --model <model> [--datasets d1,d2] [--gpus 0,1] [--max-evals N]
USAGE
}

dataset_to_config() {
  case "$1" in
    movielens1m) echo tune_ml ;;
    amazon_beauty) echo tune_ab ;;
    foursquare) echo tune_fs ;;
    retail_rocket) echo tune_rr ;;
    kuairec0.3|KuaiRec0.3) echo tune_kuai ;;
    lastfm0.3) echo tune_lfm ;;
    *) echo config ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$MODEL" ] && { echo "--model required"; exit 1; }

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

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
for ds in "${DATASET_ARR[@]}"; do
  dispatch_wait_for_gpu GPUS
  idx=$((idx + 1))

  cfg="$(dataset_to_config "$ds")"
  LOG_FILE_PATH="$(run_make_log_path baseline hparam "$ds" "$MODEL" "$FREE_GPU" "$PHASE")"
  cmd=(
    python hyperopt_tune.py
    --config-name "$cfg"
    --seed "$SEED"
    --run-group baseline
    --run-axis hparam
    --run-phase "$PHASE"
    "model=${MODEL}"
    "dataset=${ds}"
    "gpu_id=${FREE_GPU}"
    "+trial_epoch_log=false"
    "show_progress=false"
    "log_wandb=${LOG_WANDB}"
  )
  [ -n "$MAX_EVALS" ] && cmd+=(--max-evals "$MAX_EVALS")
  [ -n "$TUNE_EPOCHS" ] && cmd+=(--tune-epochs "$TUNE_EPOCHS")
  [ -n "$TUNE_PATIENCE" ] && cmd+=(--tune-patience "$TUNE_PATIENCE")
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi

  echo "[${idx}/${#DATASET_ARR[@]}] dataset=${ds} gpu=${FREE_GPU}"
  echo "[LOG] ${LOG_FILE_PATH}"
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] $(run_cmd_str "${cmd[@]}")"
    continue
  fi
  CMD_STR="$(run_cmd_str "${cmd[@]}")"
  RUN_ID="$(run_tracker_start \
    --track baseline \
    --axis hparam \
    --phase "$PHASE" \
    --dataset "$ds" \
    --model "$MODEL" \
    --cmd "$CMD_STR" \
    --log-file "$LOG_FILE_PATH")"
  (
    set +e
    LOG_FILE="${LOG_FILE_PATH}" PYTHONUNBUFFERED=1 "${cmd[@]}"
    RC=$?
    set -e
    if [ "$RC" -eq 0 ]; then
      STATUS="success"
    else
      STATUS="fail"
    fi
    run_tracker_end \
      --run-id "$RUN_ID" \
      --track baseline \
      --axis hparam \
      --phase "$PHASE" \
      --dataset "$ds" \
      --model "$MODEL" \
      --cmd "$CMD_STR" \
      --log-file "$LOG_FILE_PATH" \
      --status "$STATUS" \
      --exit-code "$RC"
    exit "$RC"
  ) &
  dispatch_set_pid "$FREE_GPU" "$!"
done

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi
dispatch_wait_all
trap - INT TERM EXIT
