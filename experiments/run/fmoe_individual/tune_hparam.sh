#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
GPU_ID="0"
MAX_EVALS="8"
TUNE_EPOCHS="50"
TUNE_PATIENCE="10"
SEED="42"
PHASE="P1"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

LAYOUT_ID="0"
TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EMBEDDING_SIZE=""
D_FEAT_EMB=""
D_EXPERT_HIDDEN=""
D_ROUTER_HIDDEN=""
EXPERT_SCALE="4"
FEATURE_TOP_K="4"
INNER_EXPERT_TOP_K="0"
BASE_LR=""
BASE_WD=""
BASE_DROP="0.10"

LR_SPACE_OVERRIDE=""
WD_SPACE_OVERRIDE=""
DROP_SPACE_OVERRIDE=""
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> [--gpu N] [--max-evals N] [--layout-id N]
          [--feature-top-k N] [--inner-expert-top-k N]
          [--lr-space csv] [--wd-space csv] [--dropout-space csv]
USAGE
}

csv_to_bracket_list() {
  local raw="${1:-}"
  raw="${raw//[[:space:]]/}"
  [ -z "$raw" ] && { echo ""; return 0; }
  IFS=',' read -r -a _arr <<< "$raw"
  local out="["
  local token
  local first="true"
  for token in "${_arr[@]}"; do
    [ -z "$token" ] && continue
    if [ "$first" = "true" ]; then
      first="false"
    else
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
      : "${TRAIN_BATCH_SIZE:=4096}"
      : "${EVAL_BATCH_SIZE:=8192}"
      : "${EMBEDDING_SIZE:=128}"
      : "${D_FEAT_EMB:=16}"
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${BASE_LR:=1.8e-3}"
      : "${BASE_WD:=1e-6}"
      ;;
    retail_rocket)
      : "${TRAIN_BATCH_SIZE:=3072}"
      : "${EVAL_BATCH_SIZE:=6144}"
      : "${EMBEDDING_SIZE:=128}"
      : "${D_FEAT_EMB:=16}"
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${BASE_LR:=8e-4}"
      : "${BASE_WD:=1e-6}"
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
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --embedding-size) EMBEDDING_SIZE="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT_EMB="$2"; shift 2 ;;
    --d-expert-hidden) D_EXPERT_HIDDEN="$2"; shift 2 ;;
    --d-router-hidden) D_ROUTER_HIDDEN="$2"; shift 2 ;;
    --expert-scale) EXPERT_SCALE="$2"; shift 2 ;;
    --feature-top-k) FEATURE_TOP_K="$2"; shift 2 ;;
    --inner-expert-top-k) INNER_EXPERT_TOP_K="$2"; shift 2 ;;
    --learning-rate) BASE_LR="$2"; shift 2 ;;
    --weight-decay) BASE_WD="$2"; shift 2 ;;
    --dropout) BASE_DROP="$2"; shift 2 ;;
    --lr-space) LR_SPACE_OVERRIDE="$2"; shift 2 ;;
    --wd-space) WD_SPACE_OVERRIDE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE_OVERRIDE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
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

LR_SPACE="${LR_SPACE_OVERRIDE:-1e-4,2e-4,3.5e-4,6e-4,9e-4,1.3e-3,1.8e-3,2.6e-3,4.0e-3,8e-3}"
WD_SPACE="${WD_SPACE_OVERRIDE:-1e-6}"
DROP_SPACE="${DROP_SPACE_OVERRIDE:-0.09,0.10,0.11}"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"
PHASE_BUCKET="${PHASE%%_*}"
PHASE_BUCKET="${PHASE_BUCKET:-PNA}"

[ -z "$EXP_NAME" ] && EXP_NAME="fmoe_individual_${PHASE%%_*}_hparam"
[ -z "$EXP_DESC" ] && EXP_DESC="FeaturedMoE_Individual fixed layout/dim hyperopt run."
[ -z "$EXP_FOCUS" ] && EXP_FOCUS="arch_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,learning_rate,weight_decay,hidden_dropout_prob"

LOG_FILE_PATH="$(run_make_phase_log_path fmoe_individual hparam "$DATASET" "FeaturedMoE_Individual" "$PHASE")"
PHASE_SUMMARY_DIR="$(run_log_dir fmoe_individual)/hparam/${PHASE_BUCKET}"
PHASE_SUMMARY_TAG="$(run_dataset_tag "$DATASET")"
PHASE_SUMMARY_CSV="${PHASE_SUMMARY_DIR}/summary_${PHASE_SUMMARY_TAG}.csv"
PHASE_SUMMARY_MD="${PHASE_SUMMARY_DIR}/summary_${PHASE_SUMMARY_TAG}.md"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_individual
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_individual_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "enable_tf32=true"
  "wandb_project=FMoE_individual"
  "wandb_project_hyperopt=FMoE_individual"
  "train_batch_size=${TRAIN_BATCH_SIZE}"
  "eval_batch_size=${EVAL_BATCH_SIZE}"
  "MAX_ITEM_LIST_LENGTH=10"
  "num_layers=-1"
  "arch_layout_id=${LAYOUT_ID}"
  "++search.arch_layout_id=[${LAYOUT_ID}]"
  "embedding_size=${EMBEDDING_SIZE}"
  "hidden_size=${EMBEDDING_SIZE}"
  "++search.embedding_size=[${EMBEDDING_SIZE}]"
  "++search.hidden_size=[${EMBEDDING_SIZE}]"
  "d_feat_emb=${D_FEAT_EMB}"
  "++search.d_feat_emb=[${D_FEAT_EMB}]"
  "d_expert_hidden=${D_EXPERT_HIDDEN}"
  "++search.d_expert_hidden=[${D_EXPERT_HIDDEN}]"
  "d_router_hidden=${D_ROUTER_HIDDEN}"
  "++search.d_router_hidden=[${D_ROUTER_HIDDEN}]"
  "expert_scale=${EXPERT_SCALE}"
  "++search.expert_scale=[${EXPERT_SCALE}]"
  "stage_merge_mode=serial"
  "++search.stage_merge_mode=[serial]"
  "outer_router_use_hidden=true"
  "++search.outer_router_use_hidden=[true]"
  "outer_router_use_feature=true"
  "++search.outer_router_use_feature=[true]"
  "inner_router_use_hidden=true"
  "++search.inner_router_use_hidden=[true]"
  "inner_router_use_feature=true"
  "++search.inner_router_use_feature=[true]"
  "expert_use_feature=false"
  "++search.expert_use_feature=[false]"
  "feature_top_k=${FEATURE_TOP_K}"
  "++search.feature_top_k=[${FEATURE_TOP_K}]"
  "inner_expert_top_k=${INNER_EXPERT_TOP_K}"
  "++search.inner_expert_top_k=[${INNER_EXPERT_TOP_K}]"
  "use_aux_loss=false"
  "++search.use_aux_loss=[false]"
  "balance_loss_lambda=0.0"
  "++search.balance_loss_lambda=[0.0]"
  "group_balance_lambda=0.0"
  "++search.group_balance_lambda=[0.0]"
  "intra_balance_lambda=0.0"
  "++search.intra_balance_lambda=[0.0]"
  "search_space_type_overrides.learning_rate=loguniform"
  "search_space_type_overrides.weight_decay=choice"
  "learning_rate=${BASE_LR}"
  "+weight_decay=${BASE_WD}"
  "hidden_dropout_prob=${BASE_DROP}"
  "++search.learning_rate=$(csv_to_bracket_list "$LR_SPACE")"
  "++search.weight_decay=$(csv_to_bracket_list "$WD_SPACE")"
  "++search.hidden_dropout_prob=$(csv_to_bracket_list "$DROP_SPACE")"
  "fmoe_schedule_enable=false"
)

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

run_ensure_dir "$(dirname "$LOG_FILE_PATH")"
cat > "$LOG_FILE_PATH" <<EOF
===============================================================================
FeaturedMoE_Individual Hyperopt Run
phase: ${PHASE}
what: ${EXP_DESC}
focus: ${EXP_FOCUS}

core_setup:
  layout=${LAYOUT_ID} | dims=${EMBEDDING_SIZE}/${D_FEAT_EMB}/${D_EXPERT_HIDDEN}/${D_ROUTER_HIDDEN}
  batch(train/eval)=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}
  feature_top_k=${FEATURE_TOP_K} | inner_expert_top_k=${INNER_EXPERT_TOP_K} | expert_scale=${EXPERT_SCALE}
  outer_router=hidden+feature | inner_router=hidden+feature | expert_use_feature=false

search_space:
  lr=${LR_SPACE}
  wd=${WD_SPACE}
  dropout=${DROP_SPACE}
===============================================================================

EOF

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_individual \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_Individual" \
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
  --track fmoe_individual \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_Individual" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_individual \
  FeaturedMoE_Individual \
  "$(run_experiments_dir)/models/FeaturedMoE_Individual"
run_update_track_report fmoe_individual
if ! "$PY_BIN" "${RUN_DIR}/common/phase_experiment_summary.py" \
  --results-dir "$(run_results_dir fmoe_individual)" \
  --dataset "$DATASET" \
  --phase-bucket "$PHASE_BUCKET" \
  --output-csv "$PHASE_SUMMARY_CSV" \
  --output-md "$PHASE_SUMMARY_MD" \
  --title "FeaturedMoE_Individual ${PHASE_BUCKET} Summary" \
  --notes "feature-individual outer top-k=4 with dense inner router"; then
  echo "[WARN] phase experiment summary update failed: phase_bucket=${PHASE_BUCKET}, dataset=${DATASET}" >&2
fi

exit "$RC"
