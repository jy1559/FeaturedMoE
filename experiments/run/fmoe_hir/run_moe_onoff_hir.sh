#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_SCRIPT="${SCRIPT_DIR}/tune_hparam_hir.sh"

DATASET="movielens1m"
GPU_ID="6"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
SEARCH_PROFILE="wide"
TRAIN_BATCH_SIZE="16384"
EVAL_BATCH_SIZE="16384"
LOG_WANDB="true"
DRY_RUN="false"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpu 7] [--max-evals 20] [--tune-epochs 100]
          [--tune-patience 10] [--seed 42] [--search-profile wide|narrow_ml1]
          [--train-batch-size 16384] [--eval-batch-size 16384]
          [--log-wandb|--no-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

run_phase() {
  local phase="$1"
  local merge_mode="$2"
  local layout_vector="$3"

  echo ""
  echo "[RUN_PHASE] ${phase} | merge=${merge_mode} | schedule=off | layout=${layout_vector}"

  local cmd=(
    bash "${TUNE_SCRIPT}"
    --dataset "${DATASET}"
    --gpu "${GPU_ID}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${SEED}"
    --search-profile "${SEARCH_PROFILE}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --phase "${phase}"
    --stage-merge-mode "${merge_mode}"
    --schedule-preset off
    --layout-vector "${layout_vector}"
  )

  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  else
    cmd+=(--no-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi

  "${cmd[@]}"
}

echo "[PLAN] FeaturedMoE_HiR MoE on/off isolation run"
echo "[PLAN] total-attn fixed pairwise (serial L4, parallel L2)"
echo "[PLAN] defaults: search_profile=${SEARCH_PROFILE} max_evals=${MAX_EVALS} tune_epochs=${TUNE_EPOCHS} patience=${TUNE_PATIENCE}"
echo "[PLAN] batch train/eval=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE} | gpu=${GPU_ID} | wandb=${LOG_WANDB}"

run_phase "P4HIR_SER_MOE_ON_L4" "serial" "1,1,1,1,0"
run_phase "P4HIR_SER_MOE_OFF_L4" "serial" "4,-1,-1,-1,0"
run_phase "P4HIR_PAR_MOE_ON_L2" "parallel" "2,0,0,0,0"
run_phase "P4HIR_PAR_MOE_OFF_L2" "parallel" "2,-1,-1,-1,0"

echo ""
echo "[DONE] HiR MoE on/off 4-phase run finished"
