#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1,2,3"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="2470"
PHASE_PREFIX="V3D"
ROUTER_DESIGN="flat_clone_residual12"
DISTILL_MODE="clone_only"
DISTILL_ENABLE="true"
DISTILL_LAMBDA_GROUP="0.0"
DISTILL_LAMBDA_CLONE="0.003"
PARENT_RESULT=""
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m] [--gpus 0,1,2,3]
          [--router-design flat_clone_residual12]
          [--distill-mode none|group_only|clone_only|group_plus_clone]
          [--distill-enable true|false]
          [--distill-lambda-group 0.0] [--distill-lambda-clone 0.003]
          [--parent-result path]
          [--max-evals 10] [--tune-epochs 100] [--tune-patience 10]
          [--phase-prefix V3D] [--seed-base 2470]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --router-design) ROUTER_DESIGN="$2"; shift 2 ;;
    --distill-mode) DISTILL_MODE="$2"; shift 2 ;;
    --distill-enable) DISTILL_ENABLE="$2"; shift 2 ;;
    --distill-lambda-group) DISTILL_LAMBDA_GROUP="$2"; shift 2 ;;
    --distill-lambda-clone) DISTILL_LAMBDA_CLONE="$2"; shift 2 ;;
    --parent-result) PARENT_RESULT="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

SEM_ROWS=(
  $'C00\t0\t0.0015,0.0025,0.0040\tDENSE'
  $'C01\t2\t8e-4,0.0012,0.0018,0.0028\tTOP2'
  $'C02\t4\t8e-4,0.0012,0.0018,0.0028\tTOP4'
)

run_one_combo() {
  local dataset="$1"
  local gpu="$2"
  local combo_idx="$3"
  local row="${SEM_ROWS[$combo_idx]}"
  IFS=$'\t' read -r combo_id topk lr_space tag <<< "${row}"
  local seed=$(( SEED_BASE + combo_idx ))
  local phase="${PHASE_PREFIX}_${combo_id}_${tag}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${dataset}"
    --layout-id "7"
    --execution "serial"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "wide"
    --train-batch-size "3072"
    --eval-batch-size "6144"
    --embedding-size "128"
    --d-feat-emb "16"
    --d-expert-hidden "512"
    --d-router-hidden "64"
    --expert-scale "3"
    --lr-space "${lr_space}"
    --wd-space "0,1e-6,1e-4"
    --dropout-space "0.10"
    --balance-space "0.005"
    --exp-name "fmoe_v3_phase_d_routing_semantics"
    --exp-desc "ML1 dense vs sparse semantics screen for flat v3 router."
    --exp-focus "router_design,router_distill_mode,router_distill_enable,moe_top_k,learning_rate,weight_decay"
    --override "router_design=${ROUTER_DESIGN}"
    --override "++search.router_design=[${ROUTER_DESIGN}]"
    --override "router_impl=learned"
    --override "router_use_hidden=true"
    --override "router_use_feature=true"
    --override "router_group_bias_scale=0.5"
    --override "++search.router_group_bias_scale=[0.5]"
    --override "router_clone_residual_scale=0.5"
    --override "++search.router_clone_residual_scale=[0.5]"
    --override "router_distill_enable=${DISTILL_ENABLE}"
    --override "++search.router_distill_enable=[${DISTILL_ENABLE}]"
    --override "router_distill_mode=${DISTILL_MODE}"
    --override "++search.router_distill_mode=[${DISTILL_MODE}]"
    --override "router_distill_lambda_group=${DISTILL_LAMBDA_GROUP}"
    --override "++search.router_distill_lambda_group=[${DISTILL_LAMBDA_GROUP}]"
    --override "router_distill_lambda_clone=${DISTILL_LAMBDA_CLONE}"
    --override "++search.router_distill_lambda_clone=[${DISTILL_LAMBDA_CLONE}]"
    --override "router_distill_temperature=1.5"
    --override "++search.router_distill_temperature=[1.5]"
    --override "router_distill_until=0.2"
    --override "++search.router_distill_until=[0.2]"
    --override "++router_impl_by_stage={}"
    --override "++search.router_impl_by_stage=[{}]"
    --override "moe_top_k=${topk}"
    --override "++search.moe_top_k=[${topk}]"
  )
  if [ -n "${PARENT_RESULT}" ]; then
    cmd+=(--parent-result "${PARENT_RESULT}")
  fi
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[Phase D][%s][GPU %s] ' "${combo_id}" "${gpu}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

on_interrupt() {
  echo "[INTERRUPT] stopping phase D workers..."
  dispatch_terminate_all GPUS
  exit 130
}
trap on_interrupt INT TERM

for dataset in "${DATASET_ARR[@]}"; do
  echo "=== [${dataset}] Phase D routing-semantics (3 combos) ==="
  for combo_idx in 0 1 2; do
    dispatch_wait_for_gpu GPUS
    gpu="${FREE_GPU}"
    (
      set -euo pipefail
      run_one_combo "${dataset}" "${gpu}" "${combo_idx}"
    ) &
    dispatch_set_pid "${gpu}" "$!"
  done
  dispatch_wait_all
done
