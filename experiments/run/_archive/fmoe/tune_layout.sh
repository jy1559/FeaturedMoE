#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
PARENT_RESULT=""
LAYOUT_CANDIDATES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
SCHEDULE="off"
GPU_ID="0"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
PHASE="P2"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> --parent_result <json> [--layout-candidates csv]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --parent_result|--parent-result) PARENT_RESULT="$2"; shift 2 ;;
    --layout-candidates) LAYOUT_CANDIDATES="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required"; exit 1; }
[ -z "$PARENT_RESULT" ] && { echo "--parent_result required"; exit 1; }
[ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT"; exit 1; }

read -r P_LR P_WD P_DROP P_BAL P_LAYOUT <<< "$(python - <<'PY' "$PARENT_RESULT"
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
layout=bp.get('arch_layout_id')
if layout is None:
    fs=d.get('fixed_search') or {}
    layout=fs.get('arch_layout_id',0)
print(bp.get('learning_rate',5e-4), bp.get('weight_decay',0.0), bp.get('hidden_dropout_prob',0.15), bp.get('balance_loss_lambda',0.01), layout)
PY
)"

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

LAYOUT_LIST="[${LAYOUT_CANDIDATES}]"
LOG_FILE_PATH="$(run_make_log_path fmoe layout "$DATASET" FeaturedMoE "$GPU_ID" "$PHASE")"

cmd=(
  python hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe
  --run-axis layout
  --run-phase "$PHASE"
  --parent-result "$PARENT_RESULT"
  "model=featured_moe_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "fmoe_debug_logging=false"
  "enable_tf32=true"
  "learning_rate=${P_LR}"
  "+weight_decay=${P_WD}"
  "hidden_dropout_prob=${P_DROP}"
  "balance_loss_lambda=${P_BAL}"
  "search.learning_rate=[${P_LR}]"
  "search.weight_decay=[${P_WD}]"
  "search.hidden_dropout_prob=[${P_DROP}]"
  "search.balance_loss_lambda=[${P_BAL}]"
  "search.arch_layout_id=${LAYOUT_LIST}"
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
  "moe_top_k=0"
  "search.moe_top_k=[0]"
  "search.moe_top_k_policy=[auto]"
  "search.moe_top_k_ratio=[0.5]"
  "log_wandb=${LOG_WANDB}"
)

if [ "$SCHEDULE" = "on" ]; then
  cmd+=(
    "fmoe_schedule_enable=true"
    "search.fmoe_schedule_enable=[true]"
  )
else
  cmd+=(
    "fmoe_schedule_enable=false"
    "search.fmoe_schedule_enable=[false]"
    "++search.alpha_warmup_until=[0]"
    "++search.temperature_warmup_until=[0]"
    "++search.moe_top_k_warmup_until=[0]"
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
  --track fmoe \
  --axis layout \
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
  --axis layout \
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
