#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
LAYOUT_ID="0"
LAYOUT_VECTOR=""
LAYOUT_CATALOG_RAW=""
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
TRAIN_BATCH_SIZE="16384"
EVAL_BATCH_SIZE="16384"
DRY_RUN="${DRY_RUN:-false}"
STAGE_MERGE_MODE="serial"
BUNDLE_TOP_K="0"
PARALLEL_STAGE_GATE_TOP_K="0"

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> [--layout_id <id> | --layout-vector a,b,c,d,e | --layout-catalog v1;v2;...]
          [--schedule-preset off|alpha_mild|temp_mild|topk_mild|combined_legacy]
          [--stage-merge-mode serial|parallel] [--bundle-top-k N] [--parallel-stage-gate-top-k N]
          [--search-profile wide|narrow_ml1] [--gpu N]
          [--train-batch-size N] [--eval-batch-size N]
USAGE
}

phase_description_ko() {
  local phase="$1"
  local sched="$2"
  local merge_mode="$3"
  local layout="$4"

  local sched_ko="스케줄 OFF(고정)"
  if [ "$sched" != "off" ]; then
    sched_ko="스케줄 ON(${sched})"
  fi

  local merge_ko="Stage 결합 serial(Macro->Mid->Micro 순차)"
  if [ "$merge_mode" = "parallel" ]; then
    merge_ko="Stage 결합 parallel(stage-gate 가중합)"
  fi

  case "$phase" in
    *MOE_ON*)
      echo "HiR MoE ON/OFF 분리 실험(MoE ON): ${sched_ko}, ${merge_ko}, layout=${layout}"
      ;;
    *MOE_OFF*)
      echo "HiR MoE ON/OFF 분리 실험(MoE OFF): ${sched_ko}, ${merge_ko}, layout=${layout}"
      ;;
    *off_base*)
      echo "HiR 기준선 실험: ${sched_ko}, ${merge_ko}, layout=${layout}"
      ;;
    *temp*)
      echo "HiR 온도 스케줄 실험: ${sched_ko}, ${merge_ko}, layout=${layout}"
      ;;
    *)
      echo "HiR 실험 설정: ${sched_ko}, ${merge_ko}, layout=${layout}"
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

normalize_layout_catalog() {
  local raw="$1"
  raw="${raw//[[:space:]]/}"
  IFS=';' read -r -a _parts <<< "$raw"
  if [ "${#_parts[@]}" -lt 1 ]; then
    echo "layout-catalog must have at least one layout" >&2
    exit 1
  fi
  local out="["
  local ids="["
  local i
  for i in "${!_parts[@]}"; do
    [ -z "${_parts[$i]}" ] && continue
    local v
    v="$(normalize_layout_vector "${_parts[$i]}")"
    if [ "$out" != "[" ]; then
      out+=","
      ids+=","
    fi
    out+="${v}"
    ids+="${i}"
  done
  out+="]"
  ids+="]"
  if [ "$out" = "[]" ]; then
    echo "layout-catalog parsed empty: $raw" >&2
    exit 1
  fi
  echo "${out}|${ids}"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --layout_id|--layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --layout-vector) LAYOUT_VECTOR="$2"; shift 2 ;;
    --layout-catalog) LAYOUT_CATALOG_RAW="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --schedule-preset) SCHEDULE_PRESET="$2"; shift 2 ;;
    --stage-merge-mode) STAGE_MERGE_MODE="$2"; shift 2 ;;
    --bundle-top-k) BUNDLE_TOP_K="$2"; shift 2 ;;
    --parallel-stage-gate-top-k) PARALLEL_STAGE_GATE_TOP_K="$2"; shift 2 ;;
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
if ! [[ "$BUNDLE_TOP_K" =~ ^[0-9]+$ ]]; then
  echo "--bundle-top-k must be a non-negative integer" >&2
  exit 1
fi
if ! [[ "$PARALLEL_STAGE_GATE_TOP_K" =~ ^[0-9]+$ ]]; then
  echo "--parallel-stage-gate-top-k must be a non-negative integer" >&2
  exit 1
fi

if [ -n "$LAYOUT_VECTOR" ] && [ -n "$LAYOUT_CATALOG_RAW" ]; then
  echo "--layout-vector and --layout-catalog cannot be used together" >&2
  exit 1
fi
if [ -n "$LAYOUT_CATALOG_RAW" ] && [ -n "$LAYOUT_ID" ] && [ "$LAYOUT_ID" != "0" ]; then
  echo "--layout_id and --layout-catalog cannot be used together" >&2
  exit 1
fi
if [ -n "$LAYOUT_VECTOR" ] && [ -n "$LAYOUT_ID" ] && [ "$LAYOUT_ID" != "0" ]; then
  echo "--layout_id and --layout-vector cannot be used together" >&2
  exit 1
fi

case "${STAGE_MERGE_MODE,,}" in
  serial|parallel) ;;
  *)
    echo "Unsupported --stage-merge-mode=${STAGE_MERGE_MODE} (use serial|parallel)" >&2
    exit 1
    ;;
esac
STAGE_MERGE_MODE="${STAGE_MERGE_MODE,,}"

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

LAYOUT_CATALOG_JSON=""
LAYOUT_SEARCH_IDS="[0]"
if [ -n "$LAYOUT_CATALOG_RAW" ]; then
  _parsed="$(normalize_layout_catalog "$LAYOUT_CATALOG_RAW")"
  LAYOUT_CATALOG_JSON="${_parsed%%|*}"
  LAYOUT_SEARCH_IDS="${_parsed##*|}"
  LAYOUT_ID="0"
elif [ -n "$LAYOUT_VECTOR" ]; then
  _v="$(normalize_layout_vector "$LAYOUT_VECTOR")"
  LAYOUT_CATALOG_JSON="[${_v}]"
  LAYOUT_SEARCH_IDS="[0]"
  LAYOUT_ID="0"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LOG_FILE_PATH="$(run_make_log_path fmoe_hir hparam "$DATASET" FeaturedMoE_HiR "$GPU_ID" "$PHASE")"

cmd=(
  python hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_hir
  --run-axis hparam
  --run-phase "$PHASE"
  "model=featured_moe_hir_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "fmoe_debug_logging=false"
  "enable_tf32=true"
  "log_wandb=${LOG_WANDB}"
  "wandb_project=FMoE_hir"
  "wandb_project_hyperopt=FMoE_hir"
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
  "expert_scale=4"
  "search.expert_scale=[4]"
  "stage_merge_mode=${STAGE_MERGE_MODE}"
  "search.stage_merge_mode=[${STAGE_MERGE_MODE}]"
  "bundle_top_k=${BUNDLE_TOP_K}"
  "search.bundle_top_k=[${BUNDLE_TOP_K}]"
  "parallel_stage_gate_top_k=${PARALLEL_STAGE_GATE_TOP_K}"
  "search.parallel_stage_gate_top_k=[${PARALLEL_STAGE_GATE_TOP_K}]"
  "hir_use_bundle_aux_loss=true"
  "search.hir_use_bundle_aux_loss=[true]"
  "hir_bundle_aux_lambda_scale=1.0"
  "search.hir_bundle_aux_lambda_scale=[1.0]"
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

if [ -n "$LAYOUT_CATALOG_JSON" ]; then
  cmd+=(
    "arch_layout_catalog=${LAYOUT_CATALOG_JSON}"
    "arch_layout_id=0"
    "search.arch_layout_id=${LAYOUT_SEARCH_IDS}"
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

echo "[INFO] schedule_preset=${SCHEDULE_PRESET} stage_merge_mode=${STAGE_MERGE_MODE} bundle_top_k=${BUNDLE_TOP_K} parallel_stage_gate_top_k=${PARALLEL_STAGE_GATE_TOP_K} search_profile=${SEARCH_PROFILE} layout_id=${LAYOUT_ID} layout_vector=${LAYOUT_VECTOR:-none} layout_catalog=${LAYOUT_CATALOG_RAW:-none} train_batch_size=${TRAIN_BATCH_SIZE:-default(config)} eval_batch_size=${EVAL_BATCH_SIZE:-default(config)}"
if [ -n "$LAYOUT_CATALOG_RAW" ]; then
  PHASE_LAYOUT_PRINT="catalog=${LAYOUT_CATALOG_RAW}"
elif [ -n "$LAYOUT_VECTOR" ]; then
  PHASE_LAYOUT_PRINT="[${LAYOUT_VECTOR}]"
else
  PHASE_LAYOUT_PRINT="layout_id=${LAYOUT_ID}"
fi
PHASE_DESC_KO="$(phase_description_ko "$PHASE" "$SCHEDULE_PRESET" "$STAGE_MERGE_MODE" "$PHASE_LAYOUT_PRINT")"
echo "[PHASE] ${PHASE}"
echo "[PHASE_KO] ${PHASE_DESC_KO}"
run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"

if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_hir \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HiR" \
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
  --track fmoe_hir \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HiR" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"
exit "$RC"
