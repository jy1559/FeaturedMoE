#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
MODEL=""
GPU_ID="0"
CONFIG_NAME=""
LOG_WANDB="false"
SPECIAL_LOGGING="false"
EPOCHS=""
PATIENCE=""
SEED="42"
PHASE="P0"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> --model <model> [--gpu <id>] [--config-name <cfg>] [--epochs N] [--patience N] [--special-logging]
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
    kuairecsmall0.1|KuaiRecSmall0.1) echo tune_kuai_small ;;
    lastfm0.03) echo tune_lfm_small ;;
    *) echo config ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --special-logging) SPECIAL_LOGGING="true"; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required"; exit 1; }
[ -z "$MODEL" ] && { echo "--model required"; exit 1; }
[ -z "$CONFIG_NAME" ] && CONFIG_NAME="$(dataset_to_config "$DATASET")"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LOG_FILE_PATH="$(run_make_log_path baseline train "$DATASET" "$MODEL" "$GPU_ID" "$PHASE")"

cmd=(
  python recbole_train.py
  --config-name "$CONFIG_NAME"
  "model=${MODEL}"
  "dataset=${DATASET}"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "++seed=${SEED}"
)

if [ -n "$EPOCHS" ]; then
  cmd+=("epochs=${EPOCHS}")
fi
if [ -n "$PATIENCE" ]; then
  cmd+=("stopping_step=${PATIENCE}")
fi
if [ "$SPECIAL_LOGGING" = "true" ]; then
  cmd+=("++special_logging=true")
fi

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi
CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track baseline \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH")"

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
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"
exit "$RC"
