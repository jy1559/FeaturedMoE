#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
GPU_ID="0"
MAX_EVALS="10"
TUNE_EPOCHS="40"
TUNE_PATIENCE="8"
SEED="42"
PHASE="R0"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

LAYOUT_ID="15"
TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EMBEDDING_SIZE=""
D_FEAT_EMB="16"
D_EXPERT_HIDDEN=""
D_ROUTER_HIDDEN=""
EXPERT_SCALE="4"
GROUP_TOP_K="0"
EXPERT_TOP_K="1"
GROUP_ROUTER_MODE="hybrid"
OUTER_ROUTER_USE_HIDDEN="true"
OUTER_ROUTER_USE_FEATURE="true"
OUTER_ROUTER_DESIGN="legacy_concat"
INNER_ROUTER_DESIGN="legacy_concat"
BASE_LR=""
BASE_WD="1e-5"
BASE_DROP="0.10"
BASE_BAL="0.0"
GROUP_BAL="0.0"
INTRA_BAL="0.0"
SPEC_LAMBDA="1e-4"
INNER_RULE_MODE="distill"
INNER_RULE_LAMBDA="5e-3"
INNER_RULE_TEMPERATURE="1.5"
INNER_RULE_UNTIL="0.2"
INNER_RULE_BIAS_SCALE="0.0"
INNER_RULE_BIN_SHARPNESS="16.0"
INNER_RULE_APPLY_STAGES="[macro,mid,micro]"

LR_SPACE_OVERRIDE=""
WD_SPACE_OVERRIDE=""
DROP_SPACE_OVERRIDE=""
BAL_SPACE_OVERRIDE=""
EXTRA_OVERRIDES=()
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> [--gpu N] [--max-evals N] [--layout-id N]
          [--inner-rule-mode off|distill|fused_bias|distill_and_fused_bias]
          [--lr-space csv] [--wd-space csv]
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
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${BASE_LR:=0.0017}"
      ;;
    retail_rocket)
      : "${TRAIN_BATCH_SIZE:=3072}"
      : "${EVAL_BATCH_SIZE:=6144}"
      : "${EMBEDDING_SIZE:=128}"
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${BASE_LR:=8e-4}"
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
    --group-top-k) GROUP_TOP_K="$2"; shift 2 ;;
    --expert-top-k) EXPERT_TOP_K="$2"; shift 2 ;;
    --group-router-mode) GROUP_ROUTER_MODE="$2"; shift 2 ;;
    --outer-router-use-hidden) OUTER_ROUTER_USE_HIDDEN="$2"; shift 2 ;;
    --outer-router-use-feature) OUTER_ROUTER_USE_FEATURE="$2"; shift 2 ;;
    --outer-router-design) OUTER_ROUTER_DESIGN="$2"; shift 2 ;;
    --inner-router-design) INNER_ROUTER_DESIGN="$2"; shift 2 ;;
    --learning-rate) BASE_LR="$2"; shift 2 ;;
    --weight-decay) BASE_WD="$2"; shift 2 ;;
    --dropout) BASE_DROP="$2"; shift 2 ;;
    --balance-loss-lambda) BASE_BAL="$2"; shift 2 ;;
    --group-balance-lambda) GROUP_BAL="$2"; shift 2 ;;
    --intra-balance-lambda) INTRA_BAL="$2"; shift 2 ;;
    --group-feature-spec-aux-lambda) SPEC_LAMBDA="$2"; shift 2 ;;
    --inner-rule-mode) INNER_RULE_MODE="$2"; shift 2 ;;
    --inner-rule-lambda) INNER_RULE_LAMBDA="$2"; shift 2 ;;
    --inner-rule-temperature) INNER_RULE_TEMPERATURE="$2"; shift 2 ;;
    --inner-rule-until) INNER_RULE_UNTIL="$2"; shift 2 ;;
    --inner-rule-bias-scale) INNER_RULE_BIAS_SCALE="$2"; shift 2 ;;
    --inner-rule-bin-sharpness) INNER_RULE_BIN_SHARPNESS="$2"; shift 2 ;;
    --inner-rule-apply-stages) INNER_RULE_APPLY_STAGES="$2"; shift 2 ;;
    --lr-space) LR_SPACE_OVERRIDE="$2"; shift 2 ;;
    --wd-space) WD_SPACE_OVERRIDE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE_OVERRIDE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE_OVERRIDE="$2"; shift 2 ;;
    --override) EXTRA_OVERRIDES+=("$2"); shift 2 ;;
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

LR_SPACE="${LR_SPACE_OVERRIDE:-1e-4,4e-4,7e-4,1.0e-3,1.3e-3,1.7e-3,2.2e-3,3.0e-3,4.5e-3,8e-3}"
WD_SPACE="${WD_SPACE_OVERRIDE:-1e-5}"
DROP_SPACE="${DROP_SPACE_OVERRIDE:-0.10}"
BAL_SPACE="${BAL_SPACE_OVERRIDE:-0.0}"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"
PHASE_BUCKET="${PHASE%%_*}"
[ -z "$PHASE_BUCKET" ] && PHASE_BUCKET="PNA"

[ -z "$EXP_NAME" ] && EXP_NAME="fmoe_hgr_v4_${PHASE%%_*}_hparam"
[ -z "$EXP_DESC" ] && EXP_DESC="HGRv4 fixed-anchor hyperopt: feature-aware outer restored, stat-soft inner teacher comparison."
[ -z "$EXP_FOCUS" ] && EXP_FOCUS="arch_layout_id,group_router_mode,embedding_size,d_expert_hidden,d_router_hidden,expert_scale,inner_rule_mode,inner_rule_lambda,inner_rule_temperature,inner_rule_until,learning_rate"

LOG_FILE_PATH="$(run_make_phase_log_path fmoe_hgr_v4 hparam "$DATASET" "FeaturedMoE_HGRv4" "$PHASE")"
PHASE_SUMMARY_DIR="$(run_log_dir fmoe_hgr_v4)/hparam/${PHASE_BUCKET}"
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
  --run-group fmoe_hgr_v4
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_hgr_v4_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "enable_tf32=true"
  "wandb_project=FMoE_hgr_v4"
  "wandb_project_hyperopt=FMoE_hgr_v4"
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
  "group_router_mode=${GROUP_ROUTER_MODE}"
  "++search.group_router_mode=[${GROUP_ROUTER_MODE}]"
  "group_top_k=${GROUP_TOP_K}"
  "++search.group_top_k=[${GROUP_TOP_K}]"
  "expert_top_k=${EXPERT_TOP_K}"
  "++search.expert_top_k=[${EXPERT_TOP_K}]"
  "router_design=${OUTER_ROUTER_DESIGN}"
  "++search.router_design=[${OUTER_ROUTER_DESIGN}]"
  "outer_router_use_hidden=${OUTER_ROUTER_USE_HIDDEN}"
  "++search.outer_router_use_hidden=[${OUTER_ROUTER_USE_HIDDEN}]"
  "outer_router_use_feature=${OUTER_ROUTER_USE_FEATURE}"
  "++search.outer_router_use_feature=[${OUTER_ROUTER_USE_FEATURE}]"
  "outer_router_design=${OUTER_ROUTER_DESIGN}"
  "++search.outer_router_design=[${OUTER_ROUTER_DESIGN}]"
  "inner_router_design=${INNER_ROUTER_DESIGN}"
  "++search.inner_router_design=[${INNER_ROUTER_DESIGN}]"
  "inner_router_use_hidden=true"
  "++search.inner_router_use_hidden=[true]"
  "inner_router_use_feature=true"
  "++search.inner_router_use_feature=[true]"
  "expert_use_feature=false"
  "++search.expert_use_feature=[false]"
  "group_balance_lambda=${GROUP_BAL}"
  "++search.group_balance_lambda=[${GROUP_BAL}]"
  "intra_balance_lambda=${INTRA_BAL}"
  "++search.intra_balance_lambda=[${INTRA_BAL}]"
  "group_feature_spec_aux_enable=true"
  "++search.group_feature_spec_aux_enable=[true]"
  "group_feature_spec_aux_lambda=${SPEC_LAMBDA}"
  "++search.group_feature_spec_aux_lambda=[${SPEC_LAMBDA}]"
  "group_feature_spec_stages=[mid]"
  "++search.group_feature_spec_stages=[[mid]]"
  "group_feature_spec_min_tokens=8"
  "++search.group_feature_spec_min_tokens=[8]"
  "inner_rule_enable=true"
  "++search.inner_rule_enable=[true]"
  "inner_rule_mode=${INNER_RULE_MODE}"
  "++search.inner_rule_mode=[${INNER_RULE_MODE}]"
  "inner_rule_lambda=${INNER_RULE_LAMBDA}"
  "++search.inner_rule_lambda=[${INNER_RULE_LAMBDA}]"
  "inner_rule_temperature=${INNER_RULE_TEMPERATURE}"
  "++search.inner_rule_temperature=[${INNER_RULE_TEMPERATURE}]"
  "inner_rule_until=${INNER_RULE_UNTIL}"
  "++search.inner_rule_until=[${INNER_RULE_UNTIL}]"
  "inner_rule_bias_scale=${INNER_RULE_BIAS_SCALE}"
  "++search.inner_rule_bias_scale=[${INNER_RULE_BIAS_SCALE}]"
  "inner_rule_bin_sharpness=${INNER_RULE_BIN_SHARPNESS}"
  "++search.inner_rule_bin_sharpness=[${INNER_RULE_BIN_SHARPNESS}]"
  "inner_rule_group_feature_pool=mean_ratio"
  "++search.inner_rule_group_feature_pool=[mean_ratio]"
  "inner_rule_apply_stages=${INNER_RULE_APPLY_STAGES}"
  "++search.inner_rule_apply_stages=[${INNER_RULE_APPLY_STAGES}]"
  "inner_rule_teacher_kind=group_stat_soft"
  "++search.inner_rule_teacher_kind=[group_stat_soft]"
  "search_space_type_overrides.learning_rate=choice"
  "search_space_type_overrides.weight_decay=choice"
  "learning_rate=${BASE_LR}"
  "+weight_decay=${BASE_WD}"
  "hidden_dropout_prob=${BASE_DROP}"
  "balance_loss_lambda=${BASE_BAL}"
  "++search.learning_rate=$(csv_to_bracket_list "$LR_SPACE")"
  "++search.weight_decay=$(csv_to_bracket_list "$WD_SPACE")"
  "++search.hidden_dropout_prob=$(csv_to_bracket_list "$DROP_SPACE")"
  "++search.balance_loss_lambda=$(csv_to_bracket_list "$BAL_SPACE")"
  "fmoe_schedule_enable=false"
)

if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  for override in "${EXTRA_OVERRIDES[@]}"; do
    cmd+=("$override")
  done
fi

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

run_ensure_dir "$(dirname "$LOG_FILE_PATH")"
cat > "$LOG_FILE_PATH" <<EOF
===============================================================================
HGRv4 Hyperopt Run
phase: ${PHASE}
what: ${EXP_DESC}
focus: ${EXP_FOCUS}

core_setup:
  layout=${LAYOUT_ID} | merge=serial | outer=${GROUP_ROUTER_MODE}:${OUTER_ROUTER_DESIGN}(hidden=${OUTER_ROUTER_USE_HIDDEN},feature=${OUTER_ROUTER_USE_FEATURE}) | inner=${INNER_ROUTER_DESIGN}
  dims=${EMBEDDING_SIZE}/${D_FEAT_EMB}/${D_EXPERT_HIDDEN}/${D_ROUTER_HIDDEN} | expert_scale=${EXPERT_SCALE}
  batch(train/eval)=${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}
  group_top_k=${GROUP_TOP_K} | expert_top_k=${EXPERT_TOP_K}

teacher_setup:
  teacher_kind=group_stat_soft
  inner_rule_mode=${INNER_RULE_MODE}
  lambda=${INNER_RULE_LAMBDA} | temperature=${INNER_RULE_TEMPERATURE} | until=${INNER_RULE_UNTIL}
  bias_scale=${INNER_RULE_BIAS_SCALE} | bin_sharpness=${INNER_RULE_BIN_SHARPNESS}

search_space:
  lr=${LR_SPACE}
  wd=${WD_SPACE}
  dropout=${DROP_SPACE}
  balance=${BAL_SPACE}
===============================================================================

EOF

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_hgr_v4 \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGRv4" \
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
  --track fmoe_hgr_v4 \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGRv4" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_hgr_v4 \
  FeaturedMoE_HGRv4 \
  "$(run_experiments_dir)/models/FeaturedMoE_HGRv4"
run_update_track_report fmoe_hgr_v4
if ! "$PY_BIN" "${RUN_DIR}/common/phase_experiment_summary.py" \
  --results-dir "$(run_results_dir fmoe_hgr_v4)" \
  --dataset "$DATASET" \
  --phase-bucket "$PHASE_BUCKET" \
  --output-csv "$PHASE_SUMMARY_CSV" \
  --output-md "$PHASE_SUMMARY_MD" \
  --title "HGRv4 ${PHASE_BUCKET} Summary" \
  --notes "feature-aware outer restored, group-stat inner teacher, 4-level distill comparison"; then
  echo "[WARN] phase experiment summary update failed: phase_bucket=${PHASE_BUCKET}, dataset=${DATASET}" >&2
fi

exit "$RC"
