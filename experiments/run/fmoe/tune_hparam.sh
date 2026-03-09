#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
LAYOUT_ID="0"
LAYOUT_VECTOR=""
SCHEDULE="off"
SCHEDULE_PRESET=""
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
DRY_RUN="${DRY_RUN:-false}"
STAGE_REPEAT_MOE="off"

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> [--layout_id <id> | --layout-vector a,b,c,d,e] [--schedule <off|on>]
          [--schedule-preset off|alpha_mild|temp_mild|topk_mild|combined_legacy]
          [--stage-repeat-moe on|off]
          [--search-profile wide|narrow_ml1] [--gpu N]
          [--train-batch-size N] [--eval-batch-size N]
USAGE
}

phase_description_ko() {
  local phase="$1"
  local sched="$2"
  local rep="$3"
  local layout="$4"

  local sched_ko="스케줄 OFF(고정)"
  if [ "$sched" != "off" ]; then
    sched_ko="스케줄 ON(${sched})"
  fi

  local rep_ko="Stage-MoE 반복 OFF(기존: stage당 1회)"
  if [ "$rep" = "true" ]; then
    rep_ko="Stage-MoE 반복 ON(depth만큼 pre-attn→MoE 반복)"
  fi

  case "$phase" in
    *off_base*)
      echo "기준선 비교 실험: ${sched_ko}, Stage 반복 OFF, layout=${layout}"
      ;;
    *off_repeat*)
      echo "구조 반복 비교 실험: ${sched_ko}, Stage 반복 ON, layout=${layout}"
      ;;
    *temp_base*)
      echo "온도 스케줄 비교 실험: ${sched_ko}, Stage 반복 OFF, layout=${layout}"
      ;;
    *temp_repeat*)
      echo "온도+반복 결합 실험: ${sched_ko}, Stage 반복 ON, layout=${layout}"
      ;;
    *)
      echo "실험 설정: ${sched_ko}, ${rep_ko}, layout=${layout}"
      ;;
  esac
}

normalize_layout_vector() {
  local raw="$1"
  raw="${raw//[[:space:]]/}"
  IFS=',' read -r -a _arr <<< "$raw"
  if [ "${#_arr[@]}" -eq 4 ]; then
    _arr+=("0")
  fi
  if [ "${#_arr[@]}" -ne 5 ]; then
    echo "layout-vector must have 4 or 5 ints (got: $raw)" >&2
    exit 1
  fi
  local i
  for i in "${_arr[@]}"; do
    [[ "$i" =~ ^-?[0-9]+$ ]] || { echo "layout-vector element must be int: $i" >&2; exit 1; }
  done
  echo "[${_arr[0]},${_arr[1]},${_arr[2]},${_arr[3]},${_arr[4]}]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --layout_id|--layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --layout-vector) LAYOUT_VECTOR="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --schedule-preset) SCHEDULE_PRESET="$2"; shift 2 ;;
    --stage-repeat-moe) STAGE_REPEAT_MOE="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --parent-result) PARENT_RESULT="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required"; exit 1; }
if [ -n "$TRAIN_BATCH_SIZE" ] && ! [[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "--train-batch-size must be a positive integer" >&2
  exit 1
fi
if [ -n "$EVAL_BATCH_SIZE" ] && ! [[ "$EVAL_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
  echo "--eval-batch-size must be a positive integer" >&2
  exit 1
fi

if [ -n "$LAYOUT_VECTOR" ] && [ -n "$LAYOUT_ID" ] && [ "$LAYOUT_ID" != "0" ]; then
  echo "--layout_id and --layout-vector cannot be used together" >&2
  exit 1
fi

if [ -z "$SCHEDULE_PRESET" ]; then
  if [ "$SCHEDULE" = "on" ]; then
    SCHEDULE_PRESET="combined_legacy"
  else
    SCHEDULE_PRESET="off"
  fi
fi

case "$SEARCH_PROFILE" in
  wide)
    LR_SPACE='[1e-4,5e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4]'
    DROP_SPACE='[0.05,0.2]'
    BAL_SPACE='[0.0005,0.1]'
    ;;
  narrow_ml1)
    # Tight search around prior ML1 winners (~5e-4 lr, ~1e-4 wd, ~0.18 drop, ~0.007 bal).
    LR_SPACE='[5e-4,2.5e-2]'
    WD_SPACE='[0.0,5e-5]'
    DROP_SPACE='[0.05,0.15]'
    BAL_SPACE='[0.01,0.05]'
    ;;
  *)
    echo "Unsupported --search-profile=${SEARCH_PROFILE}" >&2
    exit 1
    ;;
esac

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

case "${STAGE_REPEAT_MOE,,}" in
  on|true|1)
    STAGE_REPEAT_BOOL="true"
    ;;
  off|false|0|"")
    STAGE_REPEAT_BOOL="false"
    ;;
  *)
    echo "Unsupported --stage-repeat-moe=${STAGE_REPEAT_MOE} (use on|off)" >&2
    exit 1
    ;;
esac

LAYOUT_CATALOG=""
if [ -n "$LAYOUT_VECTOR" ]; then
  LAYOUT_CATALOG="$(normalize_layout_vector "$LAYOUT_VECTOR")"
  LAYOUT_ID="0"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LOG_FILE_PATH="$(run_make_log_path fmoe hparam "$DATASET" FeaturedMoE "$GPU_ID" "$PHASE")"

cmd=(
  python hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "fmoe_debug_logging=false"
  "enable_tf32=true"
  "log_wandb=${LOG_WANDB}"
  "num_heads=8"
  "search.num_heads=[8]"
  "MAX_ITEM_LIST_LENGTH=10"
  "search.MAX_ITEM_LIST_LENGTH=[10]"
  "hidden_size=64"
  "search.hidden_size=[64]"
  "d_feat_emb=16"
  "search.d_feat_emb=[16]"
  "d_expert_hidden=256"
  "search.d_expert_hidden=[256]"
  "d_router_hidden=64"
  "search.d_router_hidden=[64]"
  "expert_scale=3"
  "search.expert_scale=[3]"
  "moe_top_k=${SCH_TOPK}"
  "search.moe_top_k=[${SCH_TOPK}]"
  "moe_top_k_policy=${SCH_TOPK_POLICY}"
  "search.moe_top_k_policy=[${SCH_TOPK_POLICY}]"
  "moe_top_k_ratio=${SCH_TOPK_RATIO}"
  "search.moe_top_k_ratio=[${SCH_TOPK_RATIO}]"
  "search.macro_routing_scope=[session]"
  "search.macro_session_pooling=[query]"
  "search.mid_router_temperature=[1.3]"
  "search.micro_router_temperature=[1.3]"
  "search.mid_router_feature_dropout=[0.1]"
  "search.micro_router_feature_dropout=[0.1]"
  "search.use_valid_ratio_gating=[true]"
  "stage_moe_repeat_after_pre_layer=${STAGE_REPEAT_BOOL}"
  "search.stage_moe_repeat_after_pre_layer=[${STAGE_REPEAT_BOOL}]"
  "search.learning_rate=${LR_SPACE}"
  "search.weight_decay=${WD_SPACE}"
  "search.hidden_dropout_prob=${DROP_SPACE}"
  "search.balance_loss_lambda=${BAL_SPACE}"
  "fmoe_schedule_enable=${SCH_ENABLE}"
  "search.fmoe_schedule_enable=[${SCH_ENABLE}]"
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

if [ -n "$TRAIN_BATCH_SIZE" ]; then
  cmd+=("train_batch_size=${TRAIN_BATCH_SIZE}")
fi
if [ -n "$EVAL_BATCH_SIZE" ]; then
  cmd+=("eval_batch_size=${EVAL_BATCH_SIZE}")
fi

if [ -n "$LAYOUT_CATALOG" ]; then
  cmd+=(
    "arch_layout_catalog=[${LAYOUT_CATALOG}]"
    "arch_layout_id=0"
    "search.arch_layout_id=[0]"
    "num_layers=-1"
    "search.num_layers=[-1]"
  )
else
  cmd+=(
    "arch_layout_id=${LAYOUT_ID}"
    "search.arch_layout_id=[${LAYOUT_ID}]"
  )
fi

[ -n "$PARENT_RESULT" ] && cmd+=(--parent-result "$PARENT_RESULT")

if [ "$LOG_WANDB" = "true" ]; then
  cmd+=(--log-wandb)
fi

echo "[INFO] schedule_preset=${SCHEDULE_PRESET} stage_repeat_moe=${STAGE_REPEAT_BOOL} search_profile=${SEARCH_PROFILE} layout_id=${LAYOUT_ID} layout_vector=${LAYOUT_VECTOR:-none} train_batch_size=${TRAIN_BATCH_SIZE:-default(config)} eval_batch_size=${EVAL_BATCH_SIZE:-default(config)}"
if [ -n "$LAYOUT_VECTOR" ]; then
  PHASE_LAYOUT_PRINT="[${LAYOUT_VECTOR}]"
else
  PHASE_LAYOUT_PRINT="layout_id=${LAYOUT_ID}"
fi
PHASE_DESC_KO="$(phase_description_ko "$PHASE" "$SCHEDULE_PRESET" "$STAGE_REPEAT_BOOL" "$PHASE_LAYOUT_PRINT")"
echo "[PHASE] ${PHASE}"
echo "[PHASE_KO] ${PHASE_DESC_KO}"
run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi
CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE" \
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
  --track fmoe \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe \
  FeaturedMoE \
  "$(run_experiments_dir)/models/FeaturedMoE"
run_update_track_report fmoe

exit "$RC"
