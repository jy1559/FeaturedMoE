#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
GPU_ID="0"
SEED="42"
PHASE="P0RULE"
EPOCHS="100"
PATIENCE="10"
EVAL_EVERY="1"
LAYOUT_ID="0"
EXECUTION="serial"
SCHEDULE="off"
ABLATION="B0"
RULE_N_BINS="5"
RULE_FEATURES_PER_EXPERT="99"
PARENT_RESULT=""
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

PAR_GATE_TOP_K="0"
PAR_GATE_TEMP="1.0"
MERGE_AUX_ENABLE="false"
MERGE_AUX_SCALE="1.0"

usage() {
  cat <<USAGE
Usage: $0 --dataset <movielens1m|retail_rocket> [--ablation B0|B1|R0|R1]
          [--gpu N] [--seed N] [--layout-id N] [--execution serial|parallel]
          [--rule-n-bins N] [--rule-feature-per-expert N]
USAGE
}

load_defaults() {
  case "$DATASET" in
    movielens1m)
      MAX_LEN="10"; EMB="128"; D_FEAT="16"; D_EXP="128"; D_ROUT="64"; SCALE="3"; HEADS="8"
      LR="0.0005507387"; WD="9.862213949e-05"; DROP="0.1804033874"; BAL="0.0073076253"
      TRAIN_BS="8192"; EVAL_BS="16384";;
    retail_rocket)
      MAX_LEN="10"; EMB="128"; D_FEAT="16"; D_EXP="128"; D_ROUT="64"; SCALE="3"; HEADS="8"
      LR="0.0001594032"; WD="9.862213949e-05"; DROP="0.1586694895"; BAL="0.0073076253"
      TRAIN_BS="4096"; EVAL_BS="8192";;
    *)
      echo "Unsupported DATASET=$DATASET (use movielens1m|retail_rocket)"; exit 1 ;;
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
    --execution) EXECUTION="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --ablation) ABLATION="$2"; shift 2 ;;
    --rule-n-bins) RULE_N_BINS="$2"; shift 2 ;;
    --rule-feature-per-expert) RULE_FEATURES_PER_EXPERT="$2"; shift 2 ;;
    --parallel-stage-gate-top-k) PAR_GATE_TOP_K="$2"; shift 2 ;;
    --parallel-stage-gate-temperature) PAR_GATE_TEMP="$2"; shift 2 ;;
    --stage-merge-aux-enable) MERGE_AUX_ENABLE="$2"; shift 2 ;;
    --stage-merge-aux-scale) MERGE_AUX_SCALE="$2"; shift 2 ;;
    --parent-result) PARENT_RESULT="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --exp-desc) EXP_DESC="$2"; shift 2 ;;
    --exp-focus) EXP_FOCUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required"; exit 1; }
case "${EXECUTION,,}" in
  serial|parallel) ;;
  *) echo "--execution must be serial|parallel"; exit 1 ;;
esac
EXECUTION="${EXECUTION,,}"

case "${ABLATION^^}" in
  B0|B1|R0|R1) ;;
  *) echo "--ablation must be one of B0|B1|R0|R1"; exit 1 ;;
esac
ABLATION="${ABLATION^^}"

load_defaults

ROUTER_IMPL="learned"
ROUTER_IMPL_BY_STAGE="{}"
ROUTER_USE_HIDDEN="true"
ROUTER_USE_FEATURE="true"

case "$ABLATION" in
  B0)
    ROUTER_IMPL="learned"
    ROUTER_USE_HIDDEN="true"
    ROUTER_USE_FEATURE="true"
    ROUTER_IMPL_BY_STAGE="{}"
    ;;
  B1)
    ROUTER_IMPL="learned"
    ROUTER_USE_HIDDEN="false"
    ROUTER_USE_FEATURE="true"
    ROUTER_IMPL_BY_STAGE="{}"
    ;;
  R0)
    ROUTER_IMPL="rule_soft"
    ROUTER_USE_HIDDEN="true"
    ROUTER_USE_FEATURE="true"
    ROUTER_IMPL_BY_STAGE="{}"
    ;;
  R1)
    ROUTER_IMPL="learned"
    ROUTER_USE_HIDDEN="true"
    ROUTER_USE_FEATURE="true"
    ROUTER_IMPL_BY_STAGE="{mid:rule_soft,micro:rule_soft}"
    ;;
esac

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_rule_${PHASE%%_*}_${EXECUTION}_${ABLATION}"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="Rule-based router ablation (${ABLATION}) with compute-matched training settings."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="fmoe_stage_execution_mode,fmoe_v2_layout_id,router_impl,router_impl_by_stage,router_use_hidden,router_use_feature,rule_router.n_bins,rule_router.feature_per_expert,learning_rate,weight_decay,train_batch_size,eval_batch_size"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

LOG_FILE_PATH="$(run_make_log_path fmoe_rule train "$DATASET" "FeaturedMoE_v2_${EXECUTION}_${ABLATION}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" recbole_train.py
  "model=featured_moe_v2"
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
  "MAX_ITEM_LIST_LENGTH=${MAX_LEN}"
  "embedding_size=${EMB}"
  "num_heads=${HEADS}"
  "d_feat_emb=${D_FEAT}"
  "d_expert_hidden=${D_EXP}"
  "d_router_hidden=${D_ROUT}"
  "expert_scale=${SCALE}"
  "fmoe_v2_layout_id=${LAYOUT_ID}"
  "fmoe_stage_execution_mode=${EXECUTION}"
  "learning_rate=${LR}"
  "+weight_decay=${WD}"
  "hidden_dropout_prob=${DROP}"
  "balance_loss_lambda=${BAL}"
  "fmoe_v2_parallel_stage_gate_top_k=${PAR_GATE_TOP_K}"
  "fmoe_v2_parallel_stage_gate_temperature=${PAR_GATE_TEMP}"
  "fmoe_v2_stage_merge_aux_enable=${MERGE_AUX_ENABLE}"
  "fmoe_v2_stage_merge_aux_lambda_scale=${MERGE_AUX_SCALE}"
  "router_impl=${ROUTER_IMPL}"
  "++router_impl_by_stage=${ROUTER_IMPL_BY_STAGE}"
  "router_use_hidden=${ROUTER_USE_HIDDEN}"
  "router_use_feature=${ROUTER_USE_FEATURE}"
  "rule_router.n_bins=${RULE_N_BINS}"
  "rule_router.feature_per_expert=${RULE_FEATURES_PER_EXPERT}"
  "moe_top_k=0"
  "fmoe_debug_logging=false"
)

if [ "$SCHEDULE" = "on" ]; then
  cmd+=(
    "fmoe_schedule_enable=true"
    "alpha_warmup_until=0.3"
    "alpha_warmup_start=0.1"
    "alpha_warmup_end=1.0"
    "temperature_warmup_until=0.3"
    "mid_router_temperature_start=1.3"
    "micro_router_temperature_start=1.3"
    "moe_top_k_start=0"
    "moe_top_k_warmup_until=0.3"
  )
else
  cmd+=(
    "fmoe_schedule_enable=false"
    "alpha_warmup_until=0"
    "temperature_warmup_until=0"
    "moe_top_k_warmup_until=0"
  )
fi

[ -n "$PARENT_RESULT" ] && cmd+=("parent_result=${PARENT_RESULT}")

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_rule \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v2_${EXECUTION}_${ABLATION}" \
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
  --track fmoe_rule \
  --axis train \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v2_${EXECUTION}_${ABLATION}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_rule \
  FeaturedMoE_v2 \
  "$(run_experiments_dir)/models/FeaturedMoE_v2"
run_update_track_report fmoe_rule

exit "$RC"
