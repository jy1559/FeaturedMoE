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
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LAYOUT_ID="0"
EXECUTION="serial"
SCHEDULE="off"
ABLATION="B0"
RULE_N_BINS="5"
RULE_FEATURES_PER_EXPERT="99"
SEARCH_PROFILE="p1_shallow"

TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EMBEDDING_SIZE=""
NUM_HEADS="8"
D_FEAT_EMB=""
D_EXPERT_HIDDEN=""
D_ROUTER_HIDDEN=""
EXPERT_SCALE="3"

LR_SPACE_OVERRIDE=""
WD_SPACE_OVERRIDE=""
HIDDEN_DROPOUT_OVERRIDE=""
BALANCE_LOSS_LAMBDA_OVERRIDE=""
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 --dataset <movielens1m|retail_rocket> [--ablation B0|B1|R0|R1]
          [--gpu N] [--seed N] [--layout-id N] [--execution serial|parallel]
          [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--search-profile p1_shallow|narrow_ml1|wide]
          [--lr-space csv] [--wd-space csv]
          [--hidden-dropout X] [--balance-loss-lambda X]
USAGE
}

csv_to_bracket_list() {
  local raw="${1:-}"
  raw="${raw//[[:space:]]/}"
  [ -z "$raw" ] && { echo ""; return 0; }
  IFS=',' read -r -a _arr <<< "$raw"
  local out="["
  local i
  for i in "${!_arr[@]}"; do
    local token="${_arr[$i]}"
    [ -z "$token" ] && continue
    if [ "$out" != "[" ]; then
      out+=","
    fi
    out+="$token"
  done
  out+="]"
  echo "$out"
}

load_defaults() {
  case "$DATASET" in
    movielens1m)
      MAX_LEN="10"; EMB="128"; D_FEAT="16"; D_EXP="128"; D_ROUT="64"; SCALE="3"; HEADS="8"
      LR="0.0005507387"; WD="9.862213949e-05"; DROP="0.1804033874"; BAL="0.0073076253"
      TRAIN_BS="8192"; EVAL_BS="16384" ;;
    retail_rocket)
      MAX_LEN="10"; EMB="128"; D_FEAT="16"; D_EXP="128"; D_ROUT="64"; SCALE="3"; HEADS="8"
      LR="0.0001594032"; WD="9.862213949e-05"; DROP="0.1586694895"; BAL="0.0073076253"
      TRAIN_BS="4096"; EVAL_BS="8192" ;;
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
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --execution) EXECUTION="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --ablation) ABLATION="$2"; shift 2 ;;
    --rule-n-bins) RULE_N_BINS="$2"; shift 2 ;;
    --rule-feature-per-expert) RULE_FEATURES_PER_EXPERT="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --lr-space) LR_SPACE_OVERRIDE="$2"; shift 2 ;;
    --wd-space) WD_SPACE_OVERRIDE="$2"; shift 2 ;;
    --hidden-dropout) HIDDEN_DROPOUT_OVERRIDE="$2"; shift 2 ;;
    --balance-loss-lambda) BALANCE_LOSS_LAMBDA_OVERRIDE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --embedding-size) EMBEDDING_SIZE="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT_EMB="$2"; shift 2 ;;
    --d-expert-hidden) D_EXPERT_HIDDEN="$2"; shift 2 ;;
    --d-router-hidden) D_ROUTER_HIDDEN="$2"; shift 2 ;;
    --expert-scale) EXPERT_SCALE="$2"; shift 2 ;;
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

EMB="${EMBEDDING_SIZE:-$EMB}"
D_FEAT="${D_FEAT_EMB:-$D_FEAT}"
D_EXP="${D_EXPERT_HIDDEN:-$D_EXP}"
D_ROUT="${D_ROUTER_HIDDEN:-$D_ROUT}"
HEADS="${NUM_HEADS:-$HEADS}"
SCALE="${EXPERT_SCALE:-$SCALE}"
TRAIN_BS="${TRAIN_BATCH_SIZE:-$TRAIN_BS}"
EVAL_BS="${EVAL_BATCH_SIZE:-$EVAL_BS}"
DROP="${HIDDEN_DROPOUT_OVERRIDE:-$DROP}"
BAL="${BALANCE_LOSS_LAMBDA_OVERRIDE:-$BAL}"

case "$SEARCH_PROFILE" in
  p1_shallow)
    LR_SPACE='[1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4,1e-3]'
    ;;
  narrow_ml1)
    LR_SPACE='[5e-4,2.5e-2]'
    WD_SPACE='[0.0,5e-5]'
    ;;
  wide)
    LR_SPACE='[1e-4,5e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4]'
    ;;
  *)
    echo "Unsupported --search-profile=${SEARCH_PROFILE}" >&2
    exit 1
    ;;
esac

if [ -n "$LR_SPACE_OVERRIDE" ]; then
  LR_SPACE="$(csv_to_bracket_list "$LR_SPACE_OVERRIDE")"
fi
if [ -n "$WD_SPACE_OVERRIDE" ]; then
  WD_SPACE="$(csv_to_bracket_list "$WD_SPACE_OVERRIDE")"
fi

ROUTER_IMPL="learned"
ROUTER_IMPL_BY_STAGE="{}"
ROUTER_USE_HIDDEN="true"
ROUTER_USE_FEATURE="true"
case "$ABLATION" in
  B0)
    ROUTER_IMPL="learned"; ROUTER_USE_HIDDEN="true"; ROUTER_USE_FEATURE="true"; ROUTER_IMPL_BY_STAGE="{}" ;;
  B1)
    ROUTER_IMPL="learned"; ROUTER_USE_HIDDEN="false"; ROUTER_USE_FEATURE="true"; ROUTER_IMPL_BY_STAGE="{}" ;;
  R0)
    ROUTER_IMPL="rule_soft"; ROUTER_USE_HIDDEN="true"; ROUTER_USE_FEATURE="true"; ROUTER_IMPL_BY_STAGE="{}" ;;
  R1)
    ROUTER_IMPL="learned"; ROUTER_USE_HIDDEN="true"; ROUTER_USE_FEATURE="true"; ROUTER_IMPL_BY_STAGE="{mid:rule_soft,micro:rule_soft}" ;;
esac

if [ "$SCHEDULE" = "on" ]; then
  SCH_ENABLE="true"; SCH_ALPHA_UNTIL="0.3"; SCH_ALPHA_START="0.1"; SCH_ALPHA_END="1.0"
  SCH_TEMP_UNTIL="0.3"; SCH_MID_TEMP_START="1.3"; SCH_MICRO_TEMP_START="1.3"
  SCH_TOPK_START="0"; SCH_TOPK_WARMUP="0.3"
else
  SCH_ENABLE="false"; SCH_ALPHA_UNTIL="0"; SCH_ALPHA_START="0.0"; SCH_ALPHA_END="1.0"
  SCH_TEMP_UNTIL="0"; SCH_MID_TEMP_START="1.3"; SCH_MICRO_TEMP_START="1.3"
  SCH_TOPK_START="0"; SCH_TOPK_WARMUP="0"
fi

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_rule_${PHASE%%_*}_hparam_${ABLATION}"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="Rule ablation(${ABLATION}) with fixed architecture combo; tune lr/wd in small budget."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="ablation,router_impl,router_impl_by_stage,rule_router.n_bins,rule_router.feature_per_expert,fmoe_stage_execution_mode,fmoe_v2_layout_id,learning_rate,weight_decay"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

LOG_FILE_PATH="$(run_make_log_path fmoe_rule hparam "$DATASET" "FeaturedMoE_v2_${EXECUTION}_${ABLATION}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_rule
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_v2_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=false"
  "enable_tf32=true"
  "fmoe_debug_logging=false"
  "train_batch_size=${TRAIN_BS}"
  "eval_batch_size=${EVAL_BS}"
  "MAX_ITEM_LIST_LENGTH=${MAX_LEN}"
  "++search.MAX_ITEM_LIST_LENGTH=[${MAX_LEN}]"
  "embedding_size=${EMB}"
  "++search.embedding_size=[${EMB}]"
  "num_heads=${HEADS}"
  "++search.num_heads=[${HEADS}]"
  "d_feat_emb=${D_FEAT}"
  "++search.d_feat_emb=[${D_FEAT}]"
  "d_expert_hidden=${D_EXP}"
  "++search.d_expert_hidden=[${D_EXP}]"
  "d_router_hidden=${D_ROUT}"
  "++search.d_router_hidden=[${D_ROUT}]"
  "expert_scale=${SCALE}"
  "++search.expert_scale=[${SCALE}]"
  "fmoe_v2_layout_id=${LAYOUT_ID}"
  "++search.fmoe_v2_layout_id=[${LAYOUT_ID}]"
  "fmoe_stage_execution_mode=${EXECUTION}"
  "++search.fmoe_stage_execution_mode=[${EXECUTION}]"
  "learning_rate=${LR}"
  "+weight_decay=${WD}"
  "hidden_dropout_prob=${DROP}"
  "balance_loss_lambda=${BAL}"
  "++search.learning_rate=${LR_SPACE}"
  "++search.weight_decay=${WD_SPACE}"
  "++search.hidden_dropout_prob=[${DROP}]"
  "++search.balance_loss_lambda=[${BAL}]"
  "router_impl=${ROUTER_IMPL}"
  "++search.router_impl=[${ROUTER_IMPL}]"
  "++router_impl_by_stage=${ROUTER_IMPL_BY_STAGE}"
  "++search.router_impl_by_stage=[${ROUTER_IMPL_BY_STAGE}]"
  "router_use_hidden=${ROUTER_USE_HIDDEN}"
  "++search.router_use_hidden=[${ROUTER_USE_HIDDEN}]"
  "router_use_feature=${ROUTER_USE_FEATURE}"
  "++search.router_use_feature=[${ROUTER_USE_FEATURE}]"
  "rule_router.n_bins=${RULE_N_BINS}"
  "rule_router.feature_per_expert=${RULE_FEATURES_PER_EXPERT}"
  "moe_top_k=0"
  "++search.moe_top_k=[0]"
  "fmoe_schedule_enable=${SCH_ENABLE}"
  "++search.fmoe_schedule_enable=[${SCH_ENABLE}]"
  "alpha_warmup_until=${SCH_ALPHA_UNTIL}"
  "++search.alpha_warmup_until=[${SCH_ALPHA_UNTIL}]"
  "alpha_warmup_start=${SCH_ALPHA_START}"
  "++search.alpha_warmup_start=[${SCH_ALPHA_START}]"
  "alpha_warmup_end=${SCH_ALPHA_END}"
  "++search.alpha_warmup_end=[${SCH_ALPHA_END}]"
  "temperature_warmup_until=${SCH_TEMP_UNTIL}"
  "++search.temperature_warmup_until=[${SCH_TEMP_UNTIL}]"
  "mid_router_temperature_start=${SCH_MID_TEMP_START}"
  "++search.mid_router_temperature_start=[${SCH_MID_TEMP_START}]"
  "micro_router_temperature_start=${SCH_MICRO_TEMP_START}"
  "++search.micro_router_temperature_start=[${SCH_MICRO_TEMP_START}]"
  "moe_top_k_start=${SCH_TOPK_START}"
  "++search.moe_top_k_start=[${SCH_TOPK_START}]"
  "moe_top_k_warmup_until=${SCH_TOPK_WARMUP}"
  "++search.moe_top_k_warmup_until=[${SCH_TOPK_WARMUP}]"
)

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_rule \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v2_${EXECUTION}_${ABLATION}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH")"

set +e
WANDB_DISABLED="true" LOG_FILE="${LOG_FILE_PATH}" PYTHONUNBUFFERED=1 "${cmd[@]}"
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
  --axis hparam \
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
