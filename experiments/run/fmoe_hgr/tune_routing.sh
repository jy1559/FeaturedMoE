#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
PARENT_RESULT=""
MODE="routing"
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
Usage: $0 --dataset <ds> --parent-result <json> [--mode routing|schedule|combined]
         [--gpu N] [--max-evals N]
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
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required" >&2; exit 1; }
[ -z "$PARENT_RESULT" ] && { echo "--parent-result required" >&2; exit 1; }
[ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT" >&2; exit 1; }

case "$MODE" in
  routing)
    : "${MAX_EVALS:=16}"
    ;;
  schedule)
    : "${MAX_EVALS:=16}"
    ;;
  combined)
    : "${MAX_EVALS:=24}"
    ;;
  *)
    echo "Unsupported --mode=${MODE}" >&2
    exit 1
    ;;
esac

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_hgr_${PHASE%%_*}_routing_${MODE}"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="HGR routing stabilization search (${MODE}) on top of fixed best structure."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="group_top_k,moe_top_k,parallel_stage_gate_top_k,parallel_stage_gate_temperature,expert_use_feature,macro_routing_scope,mid_router_temperature,micro_router_temperature,mid_router_feature_dropout,micro_router_feature_dropout,alpha_warmup_until,temperature_warmup_until"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

read -r P_LR P_WD P_DROP P_BAL P_LAYOUT P_MERGE P_GROUP_MODE P_GROUP_TOPK P_MOE_TOPK \
  P_EMB P_HEADS P_DFEAT P_DEXP P_DROUT P_SCALE P_TRAIN_BS P_EVAL_BS \
  P_MID_TEMP P_MICRO_TEMP P_MID_FDROP P_MICRO_FDROP P_EXPERT_FEAT \
  P_MACRO_SCOPE P_MACRO_POOL P_PAR_TEMP <<< "$("$PY_BIN" - <<'PY' "$PARENT_RESULT"
import json
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
bp = d.get("best_params") or {}
trials = d.get("trials") or []
if not bp:
    ok = [t for t in trials if t.get("status") in (None, "ok") and isinstance(t.get("mrr@20"), (int, float))]
    if ok:
        ok.sort(key=lambda x: x.get("mrr@20", 0), reverse=True)
        bp = ok[0].get("params") or {}
fixed = d.get("fixed_search") or {}

def pick(key, default):
    if key in bp:
        return bp[key]
    if key in fixed:
        return fixed[key]
    return default

print(
    pick("learning_rate", 7e-4),
    pick("weight_decay", 1e-5),
    pick("hidden_dropout_prob", 0.12),
    pick("balance_loss_lambda", 0.01),
    pick("arch_layout_id", 0),
    pick("stage_merge_mode", "serial"),
    pick("group_router_mode", "per_group"),
    pick("group_top_k", 0),
    pick("moe_top_k", 0),
    pick("embedding_size", 128),
    pick("num_heads", 8),
    pick("d_feat_emb", 16),
    pick("d_expert_hidden", 160),
    pick("d_router_hidden", 64),
    pick("expert_scale", 3),
    pick("train_batch_size", 4096),
    pick("eval_batch_size", 8192),
    pick("mid_router_temperature", 1.3),
    pick("micro_router_temperature", 1.3),
    pick("mid_router_feature_dropout", 0.1),
    pick("micro_router_feature_dropout", 0.1),
    pick("expert_use_feature", False),
    pick("macro_routing_scope", "session"),
    pick("macro_session_pooling", "query"),
    pick("parallel_stage_gate_temperature", 1.0),
)
PY
)"

LOG_FILE_PATH="$(run_make_log_path fmoe_hgr routing "$DATASET" "FeaturedMoE_HGR_${MODE}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_hgr
  --run-axis routing
  --run-phase "$PHASE"
  --parent-result "$PARENT_RESULT"
  "model=featured_moe_hgr_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "enable_tf32=true"
  "fmoe_debug_logging=false"
  "wandb_project=FMoE_hgr"
  "wandb_project_hyperopt=FMoE_hgr"
  "embedding_size=${P_EMB}"
  "hidden_size=${P_EMB}"
  "++search.embedding_size=[${P_EMB}]"
  "++search.hidden_size=[${P_EMB}]"
  "num_heads=${P_HEADS}"
  "++search.num_heads=[${P_HEADS}]"
  "MAX_ITEM_LIST_LENGTH=10"
  "++search.MAX_ITEM_LIST_LENGTH=[10]"
  "num_layers=-1"
  "++search.num_layers=[-1]"
  "d_feat_emb=${P_DFEAT}"
  "++search.d_feat_emb=[${P_DFEAT}]"
  "d_expert_hidden=${P_DEXP}"
  "++search.d_expert_hidden=[${P_DEXP}]"
  "d_router_hidden=${P_DROUT}"
  "++search.d_router_hidden=[${P_DROUT}]"
  "expert_scale=${P_SCALE}"
  "++search.expert_scale=[${P_SCALE}]"
  "train_batch_size=${P_TRAIN_BS}"
  "eval_batch_size=${P_EVAL_BS}"
  "arch_layout_id=${P_LAYOUT}"
  "++search.arch_layout_id=[${P_LAYOUT}]"
  "stage_merge_mode=${P_MERGE}"
  "++search.stage_merge_mode=[${P_MERGE}]"
  "group_router_mode=${P_GROUP_MODE}"
  "++search.group_router_mode=[${P_GROUP_MODE}]"
  "expert_use_feature=${P_EXPERT_FEAT}"
  "++search.expert_use_feature=[${P_EXPERT_FEAT}]"
  "macro_routing_scope=${P_MACRO_SCOPE}"
  "++search.macro_routing_scope=[${P_MACRO_SCOPE}]"
  "macro_session_pooling=${P_MACRO_POOL}"
  "++search.macro_session_pooling=[${P_MACRO_POOL}]"
  "learning_rate=${P_LR}"
  "+weight_decay=${P_WD}"
  "hidden_dropout_prob=${P_DROP}"
  "balance_loss_lambda=${P_BAL}"
  "++search.learning_rate=[${P_LR}]"
  "++search.weight_decay=[${P_WD}]"
  "++search.hidden_dropout_prob=[${P_DROP}]"
  "++search.balance_loss_lambda=[${P_BAL}]"
  "use_valid_ratio_gating=true"
  "++search.use_valid_ratio_gating=[true]"
)

if [ "$P_SCALE" -ge 3 ]; then
  ROUTE_MOE_TOPK_SPACE="[0,2]"
else
  ROUTE_MOE_TOPK_SPACE="[0]"
fi

case "$MODE" in
  routing)
    cmd+=(
      "group_top_k=${P_GROUP_TOPK}"
      "++search.group_top_k=[0,2]"
      "moe_top_k=${P_MOE_TOPK}"
      "++search.moe_top_k=${ROUTE_MOE_TOPK_SPACE}"
      "mid_router_temperature=${P_MID_TEMP}"
      "++search.mid_router_temperature=[1.0,1.15,1.3,1.5]"
      "micro_router_temperature=${P_MICRO_TEMP}"
      "++search.micro_router_temperature=[1.0,1.15,1.3,1.5]"
      "mid_router_feature_dropout=${P_MID_FDROP}"
      "++search.mid_router_feature_dropout=[0.05,0.1,0.15]"
      "micro_router_feature_dropout=${P_MICRO_FDROP}"
      "++search.micro_router_feature_dropout=[0.05,0.1,0.15]"
      "fmoe_schedule_enable=false"
      "++search.fmoe_schedule_enable=[false]"
    )
    ;;
  schedule)
    cmd+=(
      "group_top_k=${P_GROUP_TOPK}"
      "++search.group_top_k=[${P_GROUP_TOPK}]"
      "moe_top_k=${P_MOE_TOPK}"
      "++search.moe_top_k=[${P_MOE_TOPK}]"
      "mid_router_temperature=${P_MID_TEMP}"
      "++search.mid_router_temperature=[${P_MID_TEMP}]"
      "micro_router_temperature=${P_MICRO_TEMP}"
      "++search.micro_router_temperature=[${P_MICRO_TEMP}]"
      "mid_router_feature_dropout=${P_MID_FDROP}"
      "++search.mid_router_feature_dropout=[${P_MID_FDROP}]"
      "micro_router_feature_dropout=${P_MICRO_FDROP}"
      "++search.micro_router_feature_dropout=[${P_MICRO_FDROP}]"
      "fmoe_schedule_enable=true"
      "++search.fmoe_schedule_enable=[true]"
      "++search.alpha_warmup_until=[0.1,0.2,0.3]"
      "++search.alpha_warmup_start=[0.0,0.1,0.2]"
      "++search.alpha_warmup_end=[0.8,1.0]"
      "++search.temperature_warmup_until=[0.1,0.2,0.3]"
      "++search.mid_router_temperature_start=[1.3,1.6,1.9]"
      "++search.micro_router_temperature_start=[1.3,1.6,1.9]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.1,0.2,0.3]"
    )
    ;;
  combined)
    cmd+=(
      "group_top_k=${P_GROUP_TOPK}"
      "++search.group_top_k=[0,2]"
      "moe_top_k=${P_MOE_TOPK}"
      "++search.moe_top_k=${ROUTE_MOE_TOPK_SPACE}"
      "mid_router_temperature=${P_MID_TEMP}"
      "++search.mid_router_temperature=[1.0,1.15,1.3,1.5]"
      "micro_router_temperature=${P_MICRO_TEMP}"
      "++search.micro_router_temperature=[1.0,1.15,1.3,1.5]"
      "mid_router_feature_dropout=${P_MID_FDROP}"
      "++search.mid_router_feature_dropout=[0.05,0.1,0.15]"
      "micro_router_feature_dropout=${P_MICRO_FDROP}"
      "++search.micro_router_feature_dropout=[0.05,0.1,0.15]"
      "fmoe_schedule_enable=true"
      "++search.fmoe_schedule_enable=[false,true]"
      "++search.alpha_warmup_until=[0.1,0.2,0.3]"
      "++search.alpha_warmup_start=[0.0,0.1,0.2]"
      "++search.alpha_warmup_end=[0.8,1.0]"
      "++search.temperature_warmup_until=[0.1,0.2,0.3]"
      "++search.mid_router_temperature_start=[1.3,1.6,1.9]"
      "++search.micro_router_temperature_start=[1.3,1.6,1.9]"
      "++search.moe_top_k_start=[0]"
      "++search.moe_top_k_warmup_until=[0.1,0.2,0.3]"
    )
    ;;
esac

if [ "$P_MERGE" = "parallel" ]; then
  cmd+=(
    "parallel_stage_gate_top_k=0"
    "++search.parallel_stage_gate_top_k=[0,2]"
    "parallel_stage_gate_temperature=${P_PAR_TEMP}"
    "++search.parallel_stage_gate_temperature=[0.8,1.0,1.2]"
  )
else
  cmd+=(
    "parallel_stage_gate_top_k=0"
    "++search.parallel_stage_gate_top_k=[0]"
    "parallel_stage_gate_temperature=${P_PAR_TEMP}"
    "++search.parallel_stage_gate_temperature=[${P_PAR_TEMP}]"
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
  --track fmoe_hgr \
  --axis routing \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${MODE}" \
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
  --track fmoe_hgr \
  --axis routing \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${MODE}" \
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
