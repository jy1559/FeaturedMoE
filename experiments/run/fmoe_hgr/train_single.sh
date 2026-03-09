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
EPOCHS="100"
PATIENCE="10"
EVAL_EVERY="1"
LAYOUT_ID=""
STAGE_MERGE_MODE=""
GROUP_ROUTER_MODE=""
GROUP_TOP_K=""
MOE_TOP_K=""
PARENT_RESULT=""
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

TRAIN_BS=""
EVAL_BS=""
EMB=""
HEADS=""
D_FEAT=""
D_EXP=""
D_ROUT=""
SCALE=""
LR=""
WD=""
DROP=""
BAL=""
MID_TEMP=""
MICRO_TEMP=""
MID_FEAT_DROP=""
MICRO_FEAT_DROP=""
EXPERT_USE_FEATURE=""
MACRO_ROUTING_SCOPE=""
MACRO_SESSION_POOLING=""
PARALLEL_STAGE_GATE_TEMPERATURE=""

usage() {
  cat <<USAGE
Usage: $0 --dataset <movielens1m|retail_rocket> [--gpu N] [--seed N]
          [--layout-id N] [--stage-merge-mode serial|parallel]
          [--group-router-mode per_group|stage_wide|hybrid] [--group-top-k N]
          [--moe-top-k N]
          [--exp-name name] [--exp-desc text] [--exp-focus csv]
USAGE
}

load_defaults() {
  case "$DATASET" in
    movielens1m)
      : "${TRAIN_BS:=4096}"
      : "${EVAL_BS:=8192}"
      : "${EMB:=128}"
      : "${HEADS:=8}"
      : "${D_FEAT:=16}"
      : "${D_EXP:=160}"
      : "${D_ROUT:=64}"
      : "${SCALE:=3}"
      : "${LR:=0.0007}"
      : "${WD:=1e-5}"
      : "${DROP:=0.12}"
      : "${BAL:=0.01}"
      : "${LAYOUT_ID:=0}"
      : "${STAGE_MERGE_MODE:=serial}"
      : "${GROUP_ROUTER_MODE:=per_group}"
      : "${GROUP_TOP_K:=0}"
      : "${MOE_TOP_K:=0}"
      : "${MID_TEMP:=1.3}"
      : "${MICRO_TEMP:=1.3}"
      : "${MID_FEAT_DROP:=0.1}"
      : "${MICRO_FEAT_DROP:=0.1}"
      : "${EXPERT_USE_FEATURE:=false}"
      : "${MACRO_ROUTING_SCOPE:=session}"
      : "${MACRO_SESSION_POOLING:=query}"
      : "${PARALLEL_STAGE_GATE_TEMPERATURE:=1.0}"
      ;;
    retail_rocket)
      : "${TRAIN_BS:=3072}"
      : "${EVAL_BS:=6144}"
      : "${EMB:=128}"
      : "${HEADS:=8}"
      : "${D_FEAT:=16}"
      : "${D_EXP:=160}"
      : "${D_ROUT:=64}"
      : "${SCALE:=3}"
      : "${LR:=0.0004}"
      : "${WD:=1e-5}"
      : "${DROP:=0.15}"
      : "${BAL:=0.01}"
      : "${LAYOUT_ID:=0}"
      : "${STAGE_MERGE_MODE:=serial}"
      : "${GROUP_ROUTER_MODE:=per_group}"
      : "${GROUP_TOP_K:=0}"
      : "${MOE_TOP_K:=0}"
      : "${MID_TEMP:=1.3}"
      : "${MICRO_TEMP:=1.3}"
      : "${MID_FEAT_DROP:=0.1}"
      : "${MICRO_FEAT_DROP:=0.1}"
      : "${EXPERT_USE_FEATURE:=false}"
      : "${MACRO_ROUTING_SCOPE:=session}"
      : "${MACRO_SESSION_POOLING:=query}"
      : "${PARALLEL_STAGE_GATE_TEMPERATURE:=1.0}"
      ;;
    *)
      echo "Unsupported DATASET=$DATASET (use movielens1m|retail_rocket)" >&2
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
    --stage-merge-mode) STAGE_MERGE_MODE="$2"; shift 2 ;;
    --group-router-mode) GROUP_ROUTER_MODE="$2"; shift 2 ;;
    --group-top-k) GROUP_TOP_K="$2"; shift 2 ;;
    --moe-top-k) MOE_TOP_K="$2"; shift 2 ;;
    --parent-result) PARENT_RESULT="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BS="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BS="$2"; shift 2 ;;
    --embedding-size) EMB="$2"; shift 2 ;;
    --num-heads) HEADS="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT="$2"; shift 2 ;;
    --d-expert-hidden) D_EXP="$2"; shift 2 ;;
    --d-router-hidden) D_ROUT="$2"; shift 2 ;;
    --expert-scale) SCALE="$2"; shift 2 ;;
    --learning-rate) LR="$2"; shift 2 ;;
    --weight-decay) WD="$2"; shift 2 ;;
    --dropout) DROP="$2"; shift 2 ;;
    --balance-loss-lambda) BAL="$2"; shift 2 ;;
    --mid-router-temperature) MID_TEMP="$2"; shift 2 ;;
    --micro-router-temperature) MICRO_TEMP="$2"; shift 2 ;;
    --mid-router-feature-dropout) MID_FEAT_DROP="$2"; shift 2 ;;
    --micro-router-feature-dropout) MICRO_FEAT_DROP="$2"; shift 2 ;;
    --expert-use-feature) EXPERT_USE_FEATURE="$2"; shift 2 ;;
    --macro-routing-scope) MACRO_ROUTING_SCOPE="$2"; shift 2 ;;
    --macro-session-pooling) MACRO_SESSION_POOLING="$2"; shift 2 ;;
    --parallel-stage-gate-temperature) PARALLEL_STAGE_GATE_TEMPERATURE="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --exp-desc) EXP_DESC="$2"; shift 2 ;;
    --exp-focus) EXP_FOCUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required" >&2; exit 1; }
load_defaults

case "${STAGE_MERGE_MODE,,}" in
  serial|parallel) ;;
  *) echo "--stage-merge-mode must be serial|parallel" >&2; exit 1 ;;
esac
case "${GROUP_ROUTER_MODE,,}" in
  per_group|stage_wide|hybrid) ;;
  *) echo "--group-router-mode must be per_group|stage_wide|hybrid" >&2; exit 1 ;;
esac
STAGE_MERGE_MODE="${STAGE_MERGE_MODE,,}"
GROUP_ROUTER_MODE="${GROUP_ROUTER_MODE,,}"
EXPERT_USE_FEATURE="${EXPERT_USE_FEATURE,,}"
MACRO_ROUTING_SCOPE="${MACRO_ROUTING_SCOPE,,}"
MACRO_SESSION_POOLING="${MACRO_SESSION_POOLING,,}"

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_hgr_${PHASE%%_*}_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="Single-train HGR run for smoke/repro with fixed routing/layout defaults."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="stage_merge_mode,group_router_mode,arch_layout_id,group_top_k,moe_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,learning_rate,weight_decay,train_batch_size,eval_batch_size"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

LOG_FILE_PATH="$(run_make_log_path fmoe_hgr train "$DATASET" "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" recbole_train.py
  "model=featured_moe_hgr"
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
  "num_heads=${HEADS}"
  "num_layers=-1"
  "d_feat_emb=${D_FEAT}"
  "d_expert_hidden=${D_EXP}"
  "d_router_hidden=${D_ROUT}"
  "expert_scale=${SCALE}"
  "arch_layout_id=${LAYOUT_ID}"
  "stage_merge_mode=${STAGE_MERGE_MODE}"
  "group_router_mode=${GROUP_ROUTER_MODE}"
  "group_top_k=${GROUP_TOP_K}"
  "moe_top_k=${MOE_TOP_K}"
  "expert_use_feature=${EXPERT_USE_FEATURE}"
  "macro_routing_scope=${MACRO_ROUTING_SCOPE}"
  "macro_session_pooling=${MACRO_SESSION_POOLING}"
  "parallel_stage_gate_temperature=${PARALLEL_STAGE_GATE_TEMPERATURE}"
  "learning_rate=${LR}"
  "+weight_decay=${WD}"
  "hidden_dropout_prob=${DROP}"
  "balance_loss_lambda=${BAL}"
  "mid_router_temperature=${MID_TEMP}"
  "micro_router_temperature=${MICRO_TEMP}"
  "mid_router_feature_dropout=${MID_FEAT_DROP}"
  "micro_router_feature_dropout=${MICRO_FEAT_DROP}"
  "use_valid_ratio_gating=true"
  "fmoe_schedule_enable=false"
  "fmoe_debug_logging=false"
  "wandb_project=FMoE_hgr"
  "wandb_project_hyperopt=FMoE_hgr"
)

[ -n "$PARENT_RESULT" ] && cmd+=("parent_result=${PARENT_RESULT}")

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_hgr \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" \
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
  --track fmoe_hgr \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_hgr \
  FeaturedMoE_HGR \
  "$(run_experiments_dir)/models/FeaturedMoE_HGR"
run_update_track_report fmoe_hgr

exit "$RC"
