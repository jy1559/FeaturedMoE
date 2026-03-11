#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
LAYOUT_ID="0"
EXECUTION="serial"
SCHEDULE_PRESET="off"
GPU_ID="0"
MAX_EVALS="40"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
PHASE="P1"
PARENT_RESULT=""
LOG_WANDB="false"
SEARCH_PROFILE="wide"
TRAIN_BATCH_SIZE="24576"
EVAL_BATCH_SIZE="24576"
EMBEDDING_SIZE="128"
NUM_HEADS="8"
D_FEAT_EMB="16"
D_EXPERT_HIDDEN="128"
D_ROUTER_HIDDEN="64"
EXPERT_SCALE="3"
DRY_RUN="${DRY_RUN:-false}"
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
Usage: $0 --dataset <ds> [--layout-id N] [--execution serial|parallel]
          [--schedule-preset off|alpha_mild|temp_mild|topk_mild|combined_legacy]
          [--search-profile wide|narrow_ml1|p1_shallow] [--gpu N]
          [--lr-space csv] [--wd-space csv]
          [--override 'hydra.key=value'] (repeatable)
          [--embedding-size N] [--num-heads N]
          [--d-feat-emb N] [--d-expert-hidden N] [--d-router-hidden N]
          [--expert-scale N]
          [--exp-name name] [--exp-desc text] [--exp-focus csv]
USAGE
}

csv_to_bracket_list() {
  local raw="${1:-}"
  raw="${raw//[[:space:]]/}"
  [ -z "$raw" ] && { echo ""; return 0; }
  IFS=',' read -r -a _arr <<< "$raw"
  if [ "${#_arr[@]}" -eq 0 ]; then
    echo ""
    return 0
  fi
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

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --execution) EXECUTION="$2"; shift 2 ;;
    --schedule-preset) SCHEDULE_PRESET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --parent-result|--parent_result) PARENT_RESULT="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --lr-space) LR_SPACE_OVERRIDE="$2"; shift 2 ;;
    --wd-space) WD_SPACE_OVERRIDE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE_OVERRIDE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE_OVERRIDE="$2"; shift 2 ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
      ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --exp-desc) EXP_DESC="$2"; shift 2 ;;
    --exp-focus) EXP_FOCUS="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --embedding-size) EMBEDDING_SIZE="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT_EMB="$2"; shift 2 ;;
    --d-expert-hidden) D_EXPERT_HIDDEN="$2"; shift 2 ;;
    --d-router-hidden) D_ROUTER_HIDDEN="$2"; shift 2 ;;
    --expert-scale) EXPERT_SCALE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
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
PY_BIN="$(run_python_bin)"

if [ -n "$TRAIN_BATCH_SIZE" ] && ! [[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "--train-batch-size must be a positive integer" >&2
  exit 1
fi
if [ -n "$EVAL_BATCH_SIZE" ] && ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "--eval-batch-size must be a positive integer" >&2
  exit 1
fi

case "$SEARCH_PROFILE" in
  wide)
    LR_SPACE='[1e-4,5e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4]'
    DROP_SPACE='[0.08,0.12]'
    BAL_SPACE='[0.001,0.003,0.01]'
    ;;
  narrow_ml1)
    LR_SPACE='[5e-4,2.5e-2]'
    WD_SPACE='[0.0,5e-5]'
    DROP_SPACE='[0.08,0.12]'
    BAL_SPACE='[0.001,0.003,0.01]'
    ;;
  p1_shallow)
    # Broad & shallow probe: discrete LR/WD only, others fixed.
    LR_SPACE='[1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4,1e-4]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4,1e-3]'
    DROP_SPACE='[0.1]'
    BAL_SPACE='[0.003]'
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
if [ -n "$DROP_SPACE_OVERRIDE" ]; then
  DROP_SPACE="$(csv_to_bracket_list "$DROP_SPACE_OVERRIDE")"
fi
if [ -n "$BAL_SPACE_OVERRIDE" ]; then
  BAL_SPACE="$(csv_to_bracket_list "$BAL_SPACE_OVERRIDE")"
fi

case "$SCHEDULE_PRESET" in
  off)
    SCH_ENABLE="false"
    SCH_ALPHA_UNTIL="0"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="1.3"
    SCH_MICRO_TEMP_START="1.3"
    SCH_TOPK="0"
    SCH_TOPK_POLICY="auto"
    SCH_TOPK_RATIO="0.5"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  alpha_mild)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.2"
    SCH_ALPHA_START="0.2"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="1.3"
    SCH_MICRO_TEMP_START="1.3"
    SCH_TOPK="0"
    SCH_TOPK_POLICY="auto"
    SCH_TOPK_RATIO="0.5"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  temp_mild)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0.2"
    SCH_MID_TEMP_START="1.8"
    SCH_MICRO_TEMP_START="1.8"
    SCH_TOPK="0"
    SCH_TOPK_POLICY="auto"
    SCH_TOPK_RATIO="0.5"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  topk_mild)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="1.3"
    SCH_MICRO_TEMP_START="1.3"
    SCH_TOPK="2"
    SCH_TOPK_POLICY="auto"
    SCH_TOPK_RATIO="0.34"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0.2"
    ;;
  combined_legacy)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.3"
    SCH_ALPHA_START="0.1"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0.3"
    SCH_MID_TEMP_START="1.3"
    SCH_MICRO_TEMP_START="1.3"
    SCH_TOPK="0"
    SCH_TOPK_POLICY="auto"
    SCH_TOPK_RATIO="0.5"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0.3"
    ;;
  *)
    echo "Unsupported --schedule-preset=${SCHEDULE_PRESET}" >&2
    exit 1
    ;;
esac

if [ -n "$PARENT_RESULT" ]; then
  [ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT"; exit 1; }
  read -r P_LR P_WD P_DROP P_BAL P_LAYOUT P_EXEC <<< "$("$PY_BIN" - <<'PY' "$PARENT_RESULT" "$LAYOUT_ID" "$EXECUTION"
import json,sys
p=sys.argv[1]
def_layout=sys.argv[2]
def_exec=sys.argv[3]
d=json.load(open(p,'r',encoding='utf-8'))
bp=d.get('best_params') or {}
trials=d.get('trials') or []
if not bp:
    ok=[t for t in trials if t.get('status') in (None,'ok') and isinstance(t.get('mrr@20'),(int,float))]
    if ok:
        ok.sort(key=lambda x:x.get('mrr@20',0), reverse=True)
        bp=ok[0].get('params') or {}
fixed=d.get('fixed_search') or {}
layout=bp.get('fmoe_v2_layout_id', fixed.get('fmoe_v2_layout_id', def_layout))
exec_mode=bp.get('fmoe_stage_execution_mode', fixed.get('fmoe_stage_execution_mode', def_exec))
print(
    bp.get('learning_rate', 5e-4),
    bp.get('weight_decay', 0.0),
    bp.get('hidden_dropout_prob', 0.15),
    bp.get('balance_loss_lambda', 0.003),
    layout,
    exec_mode,
)
PY
)"
  LAYOUT_ID="${P_LAYOUT}"
  EXECUTION="${P_EXEC}"
else
  P_LR="0.001"
  P_WD="0.0"
  P_DROP="0.1"
  P_BAL="0.003"
fi

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_v2_hir_${PHASE%%_*}_hparam"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="Hparam search (${SEARCH_PROFILE}) with factorized interaction router; optimize LR/WD + compact regularization."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="fmoe_stage_execution_mode,fmoe_v2_layout_id,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LOG_FILE_PATH="$(run_make_log_path fmoe_v2_hir hparam "$DATASET" "FeaturedMoE_v2_HiR_${EXECUTION}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_v2_hir
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_v2_hir_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "enable_tf32=true"
  "fmoe_debug_logging=false"
  "train_batch_size=${TRAIN_BATCH_SIZE}"
  "eval_batch_size=${EVAL_BATCH_SIZE}"
  "MAX_ITEM_LIST_LENGTH=10"
  "++search.MAX_ITEM_LIST_LENGTH=[10]"
  "embedding_size=${EMBEDDING_SIZE}"
  "++search.embedding_size=[${EMBEDDING_SIZE}]"
  "num_heads=${NUM_HEADS}"
  "++search.num_heads=[${NUM_HEADS}]"
  "d_feat_emb=${D_FEAT_EMB}"
  "++search.d_feat_emb=[${D_FEAT_EMB}]"
  "d_expert_hidden=${D_EXPERT_HIDDEN}"
  "++search.d_expert_hidden=[${D_EXPERT_HIDDEN}]"
  "d_router_hidden=${D_ROUTER_HIDDEN}"
  "++search.d_router_hidden=[${D_ROUTER_HIDDEN}]"
  "expert_scale=${EXPERT_SCALE}"
  "++search.expert_scale=[${EXPERT_SCALE}]"
  "router_design=group_factorized_interaction"
  "++search.router_design=[group_factorized_interaction]"
  "group_top_k=0"
  "++search.group_top_k=[0]"
  "expert_top_k=1"
  "++search.expert_top_k=[1]"
  "router_distill_enable=false"
  "++search.router_distill_enable=[false]"
  "router_distill_lambda=0.0"
  "++search.router_distill_lambda=[0.0]"
  "router_distill_temperature=1.5"
  "++search.router_distill_temperature=[1.5]"
  "router_distill_until=0.2"
  "++search.router_distill_until=[0.2]"
  "fmoe_v2_layout_id=${LAYOUT_ID}"
  "++search.fmoe_v2_layout_id=[${LAYOUT_ID}]"
  "fmoe_stage_execution_mode=${EXECUTION}"
  "++search.fmoe_stage_execution_mode=[${EXECUTION}]"
  "learning_rate=${P_LR}"
  "+weight_decay=${P_WD}"
  "hidden_dropout_prob=${P_DROP}"
  "balance_loss_lambda=${P_BAL}"
  "++search.learning_rate=${LR_SPACE}"
  "++search.weight_decay=${WD_SPACE}"
  "++search.hidden_dropout_prob=${DROP_SPACE}"
  "++search.balance_loss_lambda=${BAL_SPACE}"
  "fmoe_v2_feature_spec_aux_enable=true"
  "++search.fmoe_v2_feature_spec_aux_enable=[true]"
  "fmoe_v2_feature_spec_aux_lambda=3e-4"
  "++search.fmoe_v2_feature_spec_aux_lambda=[1e-4,3e-4,7e-4]"
  "fmoe_v2_feature_spec_stages=[mid]"
  "fmoe_v2_feature_spec_min_tokens=8"
  "moe_top_k=${SCH_TOPK}"
  "++search.moe_top_k=[${SCH_TOPK}]"
  "moe_top_k_policy=${SCH_TOPK_POLICY}"
  "++search.moe_top_k_policy=[${SCH_TOPK_POLICY}]"
  "moe_top_k_ratio=${SCH_TOPK_RATIO}"
  "++search.moe_top_k_ratio=[${SCH_TOPK_RATIO}]"
  "fmoe_v2_parallel_stage_gate_top_k=0"
  "++search.fmoe_v2_parallel_stage_gate_top_k=[0]"
  "fmoe_v2_parallel_stage_gate_temperature=1.0"
  "++search.fmoe_v2_parallel_stage_gate_temperature=[1.0]"
  "fmoe_v2_stage_merge_aux_enable=false"
  "++search.fmoe_v2_stage_merge_aux_enable=[false]"
  "fmoe_v2_stage_merge_aux_lambda_scale=1.0"
  "++search.fmoe_v2_stage_merge_aux_lambda_scale=[1.0]"
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

[ -n "$PARENT_RESULT" ] && cmd+=(--parent-result "$PARENT_RESULT")
if [ "$LOG_WANDB" = "true" ]; then
  cmd+=(--log-wandb)
fi
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_v2_hir \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v2_HiR_${EXECUTION}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH")"

set +e
if [ "$LOG_WANDB" = "true" ]; then
  WANDB_DISABLED="false" LOG_FILE="${LOG_FILE_PATH}" PYTHONUNBUFFERED=1 "${cmd[@]}"
else
  WANDB_DISABLED="true" LOG_FILE="${LOG_FILE_PATH}" PYTHONUNBUFFERED=1 "${cmd[@]}"
fi
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  STATUS="success"
else
  STATUS="fail"
fi
run_tracker_end \
  --run-id "$RUN_ID" \
  --track fmoe_v2_hir \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v2_HiR_${EXECUTION}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_v2_hir \
  FeaturedMoE_v2_HiR \
  "$(run_experiments_dir)/models/FeaturedMoE_v2_HiR"
run_update_track_report fmoe_v2_hir

exit "$RC"
