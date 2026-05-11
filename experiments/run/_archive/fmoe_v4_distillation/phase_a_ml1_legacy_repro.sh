#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="movielens1m"
GPU_ID="0"
MAX_EVALS="12"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="2400"
PHASE_PREFIX="V3A"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpu 0]
          [--max-evals 12] [--tune-epochs 100] [--tune-patience 10]
          [--phase-prefix V3A] [--seed 2400] [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

cmd=(
  bash "${SCRIPT_DIR}/tune_hparam.sh"
  --dataset "${DATASET}"
  --layout-id "7"
  --execution "serial"
  --gpu "${GPU_ID}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --seed "${SEED}"
  --phase "${PHASE_PREFIX}_L7_LEGACY"
  --search-profile "wide"
  --train-batch-size "3072"
  --eval-batch-size "6144"
  --embedding-size "128"
  --d-feat-emb "16"
  --d-expert-hidden "512"
  --d-router-hidden "64"
  --expert-scale "3"
  --lr-space "0.0045,0.0060,0.0075,0.0095"
  --wd-space "0,1e-6,1e-4"
  --dropout-space "0.10"
  --balance-space "0.01"
  --exp-name "fmoe_v3_phase_a_legacy_repro"
  --exp-desc "ML1 exact non-rule legacy repro for v3 flat baseline."
  --exp-focus "router_design,router_impl,router_impl_by_stage,moe_top_k,fmoe_v2_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"
  --override "router_design=flat_legacy"
  --override "++search.router_design=[flat_legacy]"
  --override "router_use_hidden=true"
  --override "router_use_feature=true"
  --override "router_impl=learned"
  --override "++router_impl_by_stage={}"
  --override "++search.router_impl_by_stage=[{}]"
  --override "router_distill_enable=false"
  --override "++search.router_distill_enable=[false]"
  --override "router_distill_mode=none"
  --override "++search.router_distill_mode=[none]"
  --override "router_group_bias_scale=0.5"
  --override "++search.router_group_bias_scale=[0.5]"
  --override "router_clone_residual_scale=0.5"
  --override "++search.router_clone_residual_scale=[0.5]"
  --override "moe_top_k=0"
  --override "++search.moe_top_k=[0]"
  --override "fmoe_schedule_enable=false"
  --override "++search.fmoe_schedule_enable=[false]"
)

if [ "${LOG_WANDB}" = "true" ]; then
  cmd+=(--log-wandb)
fi
if [ "${DRY_RUN}" = "true" ]; then
  cmd+=(--dry-run)
fi

printf '[Phase A] '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
