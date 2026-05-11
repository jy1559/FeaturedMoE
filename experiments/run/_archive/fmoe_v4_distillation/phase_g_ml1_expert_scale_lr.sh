#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="movielens1m"
GPU_LIST="0,1,2,3"
SCALE_LIST="1,3,5,8"
MAX_EVALS="8"
TUNE_EPOCHS="40"
TUNE_PATIENCE="5"
SEED_BASE="2600"
PHASE_PREFIX="P2ESLR"
LR_SPACE="0.0028,0.0032,0.0036,0.0040,0.0048"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

# Current completed non-rule anchor from fmoe_v3 phase B.
ANCHOR_RESULT="/workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2router_c00_legacy_20260310_052922_110009_pid584683.json"
ANCHOR_MRR="0.0973"
ANCHOR_LR="0.003457647151765668"
ANCHOR_WD="7.057864066167489e-05"

LAYOUT_ID="7"
EXECUTION="serial"
TRAIN_BATCH_SIZE="3072"
EVAL_BATCH_SIZE="6144"
EMBEDDING_SIZE="128"
NUM_HEADS="8"
D_FEAT_EMB="16"
D_EXPERT_HIDDEN="512"
D_ROUTER_HIDDEN="64"
HIDDEN_DROPOUT="0.10"
BALANCE_LAMBDA="0.005"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpus 0,1,2,3] [--scales 1,3,5,8]
          [--lr-space csv] [--max-evals 8] [--tune-epochs 40] [--tune-patience 5]
          [--seed-base 2600] [--phase-prefix P2ESLR]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --scales) SCALE_LIST="$2"; shift 2 ;;
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
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
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }

dispatch_parse_csv "$SCALE_LIST" SCALES
[ "${#SCALES[@]}" -eq 0 ] && { echo "Empty scale list" >&2; exit 1; }

run_one_scale() {
  local scale="$1"
  local gpu="$2"
  local idx="$3"
  local seed=$(( SEED_BASE + idx ))
  local phase="${PHASE_PREFIX}_S${scale}"
  local cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "${DATASET}"
    --layout-id "${LAYOUT_ID}"
    --execution "${EXECUTION}"
    --gpu "${gpu}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${seed}"
    --phase "${phase}"
    --search-profile "wide"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --embedding-size "${EMBEDDING_SIZE}"
    --num-heads "${NUM_HEADS}"
    --d-feat-emb "${D_FEAT_EMB}"
    --d-expert-hidden "${D_EXPERT_HIDDEN}"
    --d-router-hidden "${D_ROUTER_HIDDEN}"
    --expert-scale "${scale}"
    --lr-space "${LR_SPACE}"
    --wd-space "${ANCHOR_WD}"
    --dropout-space "${HIDDEN_DROPOUT}"
    --balance-space "${BALANCE_LAMBDA}"
    --exp-name "fmoe_v3_phase_g_ml1_expert_scale_lr"
    --exp-desc "ML1 legacy flat-router anchor with narrow LR search while varying expert_scale only."
    --exp-focus "router_design,expert_scale,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda,fmoe_v2_layout_id,fmoe_stage_execution_mode"
    --override "learning_rate=${ANCHOR_LR}"
    --override "weight_decay=${ANCHOR_WD}"
    --override "hidden_dropout_prob=${HIDDEN_DROPOUT}"
    --override "balance_loss_lambda=${BALANCE_LAMBDA}"
    --override "router_design=flat_legacy"
    --override "++search.router_design=[flat_legacy]"
    --override "router_use_hidden=true"
    --override "router_use_feature=true"
    --override "router_impl=learned"
    --override "router_group_bias_scale=0.5"
    --override "++search.router_group_bias_scale=[0.5]"
    --override "router_clone_residual_scale=0.5"
    --override "++search.router_clone_residual_scale=[0.5]"
    --override "router_distill_enable=false"
    --override "++search.router_distill_enable=[false]"
    --override "router_distill_mode=none"
    --override "++search.router_distill_mode=[none]"
    --override "++router_impl_by_stage={}"
    --override "++search.router_impl_by_stage=[{}]"
    --override "moe_top_k=0"
    --override "++search.moe_top_k=[0]"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  printf '[Phase G][scale=%s][GPU %s] ' "${scale}" "${gpu}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
}

on_interrupt() {
  echo "[INTERRUPT] stopping phase G workers..."
  dispatch_terminate_all GPUS
  exit 130
}
trap on_interrupt INT TERM

echo "=== [${DATASET}] Phase G expert_scale/LR sweep ==="
echo "[Anchor] mrr@20=${ANCHOR_MRR} | lr=${ANCHOR_LR} | wd=${ANCHOR_WD} | result=${ANCHOR_RESULT}"
echo "[Anchor] layout=${LAYOUT_ID} mode=${EXECUTION} dims=${EMBEDDING_SIZE}/${D_FEAT_EMB}/${D_EXPERT_HIDDEN}/${D_ROUTER_HIDDEN} bs=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
echo "[Sweep] scales=${SCALE_LIST} | lr_space=${LR_SPACE} | max_evals=${MAX_EVALS}"

for idx in "${!SCALES[@]}"; do
  dispatch_wait_for_gpu GPUS
  gpu="${FREE_GPU}"
  (
    set -euo pipefail
    run_one_scale "${SCALES[$idx]}" "${gpu}" "${idx}"
  ) &
  dispatch_set_pid "${gpu}" "$!"
done

dispatch_wait_all
