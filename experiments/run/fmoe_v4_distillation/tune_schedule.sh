#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
PARENT_RESULT=""
MODE="combined"
GPU_ID="0"
MAX_EVALS=""
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
PHASE="P4"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> --parent-result <json> [--mode alpha|temp|topk|combined]
         [--exp-name name] [--exp-desc text] [--exp-focus csv]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --parent-result|--parent_result) PARENT_RESULT="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --exp-desc) EXP_DESC="$2"; shift 2 ;;
    --exp-focus) EXP_FOCUS="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required"; exit 1; }
[ -z "$PARENT_RESULT" ] && { echo "--parent-result required"; exit 1; }
[ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT"; exit 1; }
PY_BIN="$(run_python_bin)"

if [ -z "$MAX_EVALS" ]; then
  case "$MODE" in
    alpha|temp|topk) MAX_EVALS="12" ;;
    combined) MAX_EVALS="20" ;;
    *) echo "Unsupported --mode=$MODE"; exit 1 ;;
  esac
fi

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_v3_${PHASE%%_*}_schedule_${MODE}"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="Schedule-axis tuning (${MODE}) on top of fixed best layout/execution."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="alpha_warmup_until,temperature_warmup_until,moe_top_k,moe_top_k_ratio,fmoe_v2_parallel_stage_gate_top_k,fmoe_v2_parallel_stage_gate_temperature"
fi

read -r P_LR P_WD P_DROP P_BAL P_LAYOUT P_EXEC P_TOPK <<< "$("$PY_BIN" - <<'PY' "$PARENT_RESULT"
import json,sys
p=sys.argv[1]
d=json.load(open(p,'r',encoding='utf-8'))
bp=d.get('best_params') or {}
trials=d.get('trials') or []
if not bp:
    ok=[t for t in trials if t.get('status') in (None,'ok') and isinstance(t.get('mrr@20'),(int,float))]
    if ok:
        ok.sort(key=lambda x:x.get('mrr@20',0), reverse=True)
        bp=ok[0].get('params') or {}
fixed=d.get('fixed_search') or {}
print(
    bp.get('learning_rate', fixed.get('learning_rate', 5e-4)),
    bp.get('weight_decay', fixed.get('weight_decay', 0.0)),
    bp.get('hidden_dropout_prob', fixed.get('hidden_dropout_prob', 0.15)),
    bp.get('balance_loss_lambda', fixed.get('balance_loss_lambda', 0.01)),
    bp.get('fmoe_v2_layout_id', fixed.get('fmoe_v2_layout_id', 0)),
    bp.get('fmoe_stage_execution_mode', fixed.get('fmoe_stage_execution_mode', 'serial')),
    bp.get('moe_top_k', fixed.get('moe_top_k', 0)),
)
PY
)"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LOG_FILE_PATH="$(run_make_log_path fmoe_v3 schedule "$DATASET" "FeaturedMoE_v3_${MODE}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_v3
  --run-axis schedule
  --run-phase "$PHASE"
  --parent-result "$PARENT_RESULT"
  "model=featured_moe_v3_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "enable_tf32=true"
  "fmoe_debug_logging=false"
  "embedding_size=128"
  "++search.embedding_size=[128]"
  "num_heads=8"
  "++search.num_heads=[8]"
  "MAX_ITEM_LIST_LENGTH=10"
  "++search.MAX_ITEM_LIST_LENGTH=[10]"
  "d_feat_emb=16"
  "++search.d_feat_emb=[16]"
  "d_expert_hidden=128"
  "++search.d_expert_hidden=[128]"
  "d_router_hidden=64"
  "++search.d_router_hidden=[64]"
  "expert_scale=3"
  "++search.expert_scale=[3]"
  "fmoe_v2_layout_id=${P_LAYOUT}"
  "++search.fmoe_v2_layout_id=[${P_LAYOUT}]"
  "fmoe_stage_execution_mode=${P_EXEC}"
  "++search.fmoe_stage_execution_mode=[${P_EXEC}]"
  "learning_rate=${P_LR}"
  "+weight_decay=${P_WD}"
  "hidden_dropout_prob=${P_DROP}"
  "balance_loss_lambda=${P_BAL}"
  "++search.learning_rate=[${P_LR}]"
  "++search.weight_decay=[${P_WD}]"
  "++search.hidden_dropout_prob=[${P_DROP}]"
  "++search.balance_loss_lambda=[${P_BAL}]"
  "fmoe_schedule_enable=true"
  "++search.fmoe_schedule_enable=[true]"
  "log_wandb=${LOG_WANDB}"
)

case "$MODE" in
  alpha)
    cmd+=(
      "moe_top_k=${P_TOPK}"
      "++search.moe_top_k=[${P_TOPK}]"
      "++search.moe_top_k_policy=[auto]"
      "++search.moe_top_k_ratio=[0.5]"
      "++search.alpha_warmup_until=[0.1,0.2,0.3,0.4]"
      "++search.alpha_warmup_start=[0.0,0.1,0.2]"
      "++search.alpha_warmup_end=[0.8,1.0]"
      "++search.temperature_warmup_until=[0.3]"
      "++search.mid_router_temperature_start=[1.3]"
      "++search.micro_router_temperature_start=[1.3]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.3]"
    )
    ;;
  temp)
    cmd+=(
      "moe_top_k=${P_TOPK}"
      "++search.moe_top_k=[${P_TOPK}]"
      "++search.moe_top_k_policy=[auto]"
      "++search.moe_top_k_ratio=[0.5]"
      "++search.alpha_warmup_until=[0.3]"
      "++search.alpha_warmup_start=[0.1]"
      "++search.alpha_warmup_end=[1.0]"
      "++search.temperature_warmup_until=[0.1,0.2,0.3,0.4]"
      "++search.mid_router_temperature_start=[1.3,1.8,2.2]"
      "++search.micro_router_temperature_start=[1.3,1.8,2.2]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.3]"
    )
    ;;
  topk)
    cmd+=(
      "moe_top_k=${P_TOPK}"
      "++search.alpha_warmup_until=[0.3]"
      "++search.alpha_warmup_start=[0.1]"
      "++search.alpha_warmup_end=[1.0]"
      "++search.temperature_warmup_until=[0.3]"
      "++search.mid_router_temperature_start=[1.3]"
      "++search.micro_router_temperature_start=[1.3]"
      "++search.moe_top_k=[0,2,3]"
      "++search.moe_top_k_policy=[dense,auto]"
      "++search.moe_top_k_ratio=[0.34,0.5,0.67]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.1,0.2,0.3,0.4]"
    )
    ;;
  combined)
    cmd+=(
      "moe_top_k=${P_TOPK}"
      "++search.alpha_warmup_until=[0.1,0.2,0.3,0.4]"
      "++search.alpha_warmup_start=[0.0,0.1,0.2]"
      "++search.alpha_warmup_end=[0.8,1.0]"
      "++search.temperature_warmup_until=[0.1,0.2,0.3,0.4]"
      "++search.mid_router_temperature_start=[1.3,1.8,2.2]"
      "++search.micro_router_temperature_start=[1.3,1.8,2.2]"
      "++search.moe_top_k=[0,2,3]"
      "++search.moe_top_k_policy=[dense,auto]"
      "++search.moe_top_k_ratio=[0.34,0.5,0.67]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.1,0.2,0.3,0.4]"
    )
    ;;
  *)
    echo "Unsupported --mode=$MODE"; exit 1 ;;
esac

if [ "$P_EXEC" = "parallel" ]; then
  cmd+=(
    "fmoe_v2_parallel_stage_gate_top_k=0"
    "++search.fmoe_v2_parallel_stage_gate_top_k=[0,1,2]"
    "fmoe_v2_parallel_stage_gate_temperature=1.0"
    "++search.fmoe_v2_parallel_stage_gate_temperature=[0.8,1.0,1.2]"
    "fmoe_v2_stage_merge_aux_enable=false"
    "++search.fmoe_v2_stage_merge_aux_enable=[false,true]"
    "fmoe_v2_stage_merge_aux_lambda_scale=1.0"
    "++search.fmoe_v2_stage_merge_aux_lambda_scale=[0.5,1.0]"
  )
else
  cmd+=(
    "fmoe_v2_parallel_stage_gate_top_k=0"
    "++search.fmoe_v2_parallel_stage_gate_top_k=[0]"
    "fmoe_v2_parallel_stage_gate_temperature=1.0"
    "++search.fmoe_v2_parallel_stage_gate_temperature=[1.0]"
    "fmoe_v2_stage_merge_aux_enable=false"
    "++search.fmoe_v2_stage_merge_aux_enable=[false]"
    "fmoe_v2_stage_merge_aux_lambda_scale=1.0"
    "++search.fmoe_v2_stage_merge_aux_lambda_scale=[1.0]"
  )
fi

if [ "$LOG_WANDB" = "true" ]; then
  cmd+=(--log-wandb)
fi

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_v3 \
  --axis schedule \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v3_${MODE}" \
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
  --track fmoe_v3 \
  --axis schedule \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_v3_${MODE}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_v3 \
  FeaturedMoE_v3 \
  "$(run_experiments_dir)/models/FeaturedMoE_v3"
run_update_track_report fmoe_v3

exit "$RC"
