#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
GPU_ID="0"
SEED="42"
PHASE="P0"
EPOCHS="40"
PATIENCE="8"
EVAL_EVERY="1"
LAYOUT_ID="15"
GROUP_TOP_K="0"
EXPERT_TOP_K="1"
TRAIN_BS=""
EVAL_BS=""
EMB=""
D_FEAT=""
D_EXP=""
D_ROUT=""
SCALE=""
LR=""
WD=""
DROP=""
BAL=""
GROUP_BAL="0.001"
INTRA_BAL="0.001"
SPEC_LAMBDA="1e-4"
INNER_RULE_MODE="distill"
INNER_RULE_LAMBDA="5e-3"
INNER_RULE_TEMPERATURE="1.5"
INNER_RULE_UNTIL="0.2"
INNER_RULE_BIAS_SCALE="1.0"
INNER_RULE_BIN_SHARPNESS="16.0"
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 --dataset <movielens1m|retail_rocket> [--gpu N] [--seed N]
          [--layout-id N] [--train-batch-size N] [--eval-batch-size N]
          [--embedding-size N] [--d-feat-emb N] [--d-expert-hidden N] [--d-router-hidden N]
          [--expert-scale N] [--inner-rule-mode off|distill|fused_bias|distill_and_fused_bias]
USAGE
}

load_defaults() {
  case "$DATASET" in
    movielens1m)
      : "${TRAIN_BS:=4096}"
      : "${EVAL_BS:=8192}"
      : "${EMB:=128}"
      : "${D_FEAT:=16}"
      : "${D_EXP:=160}"
      : "${D_ROUT:=64}"
      : "${SCALE:=4}"
      : "${LR:=0.0018}"
      : "${WD:=1e-5}"
      : "${DROP:=0.10}"
      : "${BAL:=0.0032}"
      ;;
    retail_rocket)
      : "${TRAIN_BS:=3072}"
      : "${EVAL_BS:=6144}"
      : "${EMB:=128}"
      : "${D_FEAT:=16}"
      : "${D_EXP:=160}"
      : "${D_ROUT:=64}"
      : "${SCALE:=4}"
      : "${LR:=8e-4}"
      : "${WD:=1e-5}"
      : "${DROP:=0.12}"
      : "${BAL:=0.0032}"
      ;;
    *)
      echo "Unsupported DATASET=$DATASET" >&2
      exit 1
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --eval-every) EVAL_EVERY="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --group-top-k) GROUP_TOP_K="$2"; shift 2 ;;
    --expert-top-k) EXPERT_TOP_K="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BS="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BS="$2"; shift 2 ;;
    --embedding-size) EMB="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT="$2"; shift 2 ;;
    --d-expert-hidden) D_EXP="$2"; shift 2 ;;
    --d-router-hidden) D_ROUT="$2"; shift 2 ;;
    --expert-scale) SCALE="$2"; shift 2 ;;
    --learning-rate) LR="$2"; shift 2 ;;
    --weight-decay) WD="$2"; shift 2 ;;
    --dropout) DROP="$2"; shift 2 ;;
    --balance-loss-lambda) BAL="$2"; shift 2 ;;
    --group-balance-lambda) GROUP_BAL="$2"; shift 2 ;;
    --intra-balance-lambda) INTRA_BAL="$2"; shift 2 ;;
    --group-feature-spec-aux-lambda) SPEC_LAMBDA="$2"; shift 2 ;;
    --inner-rule-mode) INNER_RULE_MODE="$2"; shift 2 ;;
    --inner-rule-lambda) INNER_RULE_LAMBDA="$2"; shift 2 ;;
    --inner-rule-temperature) INNER_RULE_TEMPERATURE="$2"; shift 2 ;;
    --inner-rule-until) INNER_RULE_UNTIL="$2"; shift 2 ;;
    --inner-rule-bias-scale) INNER_RULE_BIAS_SCALE="$2"; shift 2 ;;
    --inner-rule-bin-sharpness) INNER_RULE_BIN_SHARPNESS="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --exp-desc) EXP_DESC="$2"; shift 2 ;;
    --exp-focus) EXP_FOCUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required" >&2; exit 1; }
load_defaults

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

[ -z "$EXP_NAME" ] && EXP_NAME="fmoe_hgr_v3_${PHASE%%_*}_single"
[ -z "$EXP_DESC" ] && EXP_DESC="HGRv3 single run with hidden-only outer router and inner rule teacher."
[ -z "$EXP_FOCUS" ] && EXP_FOCUS="arch_layout_id,embedding_size,d_expert_hidden,d_router_hidden,expert_scale,inner_rule_mode,learning_rate,weight_decay,train_batch_size"

LOG_FILE_PATH="$(run_make_phase_log_path fmoe_hgr_v3 train "$DATASET" "FeaturedMoE_HGRv3" "$PHASE")"

cmd=(
  "$PY_BIN" recbole_train.py
  "model=featured_moe_hgr_v3"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "seed=${SEED}"
  "epochs=${EPOCHS}"
  "stopping_step=${PATIENCE}"
  "eval_every=${EVAL_EVERY}"
  "train_batch_size=${TRAIN_BS}"
  "eval_batch_size=${EVAL_BS}"
  "MAX_ITEM_LIST_LENGTH=10"
  "embedding_size=${EMB}"
  "hidden_size=${EMB}"
  "num_layers=-1"
  "arch_layout_id=${LAYOUT_ID}"
  "stage_merge_mode=serial"
  "group_router_mode=stage_wide"
  "group_top_k=${GROUP_TOP_K}"
  "expert_top_k=${EXPERT_TOP_K}"
  "d_feat_emb=${D_FEAT}"
  "d_expert_hidden=${D_EXP}"
  "d_router_hidden=${D_ROUT}"
  "expert_scale=${SCALE}"
  "router_design=group_factorized_interaction"
  "outer_router_use_hidden=true"
  "outer_router_use_feature=false"
  "inner_router_use_hidden=true"
  "inner_router_use_feature=true"
  "expert_use_feature=false"
  "group_balance_lambda=${GROUP_BAL}"
  "intra_balance_lambda=${INTRA_BAL}"
  "group_feature_spec_aux_enable=true"
  "group_feature_spec_aux_lambda=${SPEC_LAMBDA}"
  "group_feature_spec_stages=[mid]"
  "group_feature_spec_min_tokens=8"
  "inner_rule_enable=true"
  "inner_rule_mode=${INNER_RULE_MODE}"
  "inner_rule_lambda=${INNER_RULE_LAMBDA}"
  "inner_rule_temperature=${INNER_RULE_TEMPERATURE}"
  "inner_rule_until=${INNER_RULE_UNTIL}"
  "inner_rule_bias_scale=${INNER_RULE_BIAS_SCALE}"
  "inner_rule_bin_sharpness=${INNER_RULE_BIN_SHARPNESS}"
  "inner_rule_group_feature_pool=mean_ratio"
  "inner_rule_apply_stages=[macro,mid,micro]"
  "learning_rate=${LR}"
  "+weight_decay=${WD}"
  "hidden_dropout_prob=${DROP}"
  "balance_loss_lambda=${BAL}"
  "fmoe_schedule_enable=false"
  "wandb_project=FMoE_hgr_v3"
  "wandb_project_hyperopt=FMoE_hgr_v3"
)

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_hgr_v3 \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGRv3" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
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
  --track fmoe_hgr_v3 \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGRv3" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_hgr_v3 \
  FeaturedMoE_HGRv3 \
  "$(run_experiments_dir)/models/FeaturedMoE_HGRv3"
run_update_track_report fmoe_hgr_v3

exit "$RC"
