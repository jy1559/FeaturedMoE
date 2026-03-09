#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
LAYOUT_ID=""
STAGE_MERGE_MODE=""
GROUP_ROUTER_MODE=""
GROUP_TOP_K=""
MOE_TOP_K=""
GPU_ID="0"
MAX_EVALS="24"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
PHASE="P1"
PARENT_RESULT=""
LOG_WANDB="false"
SEARCH_PROFILE="p1_shallow"
SCHEDULE_PRESET="off"
DRY_RUN="${DRY_RUN:-false}"
LR_SPACE_OVERRIDE=""
WD_SPACE_OVERRIDE=""
DROP_SPACE_OVERRIDE=""
BAL_SPACE_OVERRIDE=""
EXTRA_OVERRIDES=()
EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""

TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EMBEDDING_SIZE=""
NUM_HEADS=""
D_FEAT_EMB=""
D_EXPERT_HIDDEN=""
D_ROUTER_HIDDEN=""
EXPERT_SCALE=""
BASE_LR=""
BASE_WD=""
BASE_DROP=""
BASE_BAL=""
MID_ROUTER_TEMPERATURE=""
MICRO_ROUTER_TEMPERATURE=""
MID_ROUTER_FEATURE_DROPOUT=""
MICRO_ROUTER_FEATURE_DROPOUT=""
EXPERT_USE_FEATURE=""
MACRO_ROUTING_SCOPE=""
MACRO_SESSION_POOLING=""
PARALLEL_STAGE_GATE_TEMPERATURE=""

usage() {
  cat <<USAGE
Usage: $0 --dataset <ds> [--layout-id N] [--stage-merge-mode serial|parallel]
          [--group-router-mode per_group|stage_wide|hybrid] [--group-top-k N]
          [--search-profile wide|p1_shallow|confirm_narrow|structure_refine]
          [--schedule-preset off|alpha_mild|alpha_cold|temp_mild|alpha_temp_cold|topk_mild|combined_legacy]
          [--lr-space csv] [--wd-space csv] [--dropout-space csv] [--balance-space csv]
          [--override 'hydra.key=value'] (repeatable)
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
      : "${NUM_HEADS:=8}"
      : "${D_FEAT_EMB:=16}"
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${EXPERT_SCALE:=3}"
      : "${BASE_LR:=0.0007}"
      : "${BASE_WD:=1e-5}"
      : "${BASE_DROP:=0.12}"
      : "${BASE_BAL:=0.01}"
      : "${LAYOUT_ID:=0}"
      : "${STAGE_MERGE_MODE:=serial}"
      : "${GROUP_ROUTER_MODE:=per_group}"
      : "${GROUP_TOP_K:=0}"
      : "${MOE_TOP_K:=0}"
      : "${MID_ROUTER_TEMPERATURE:=1.3}"
      : "${MICRO_ROUTER_TEMPERATURE:=1.3}"
      : "${MID_ROUTER_FEATURE_DROPOUT:=0.1}"
      : "${MICRO_ROUTER_FEATURE_DROPOUT:=0.1}"
      : "${EXPERT_USE_FEATURE:=false}"
      : "${MACRO_ROUTING_SCOPE:=session}"
      : "${MACRO_SESSION_POOLING:=query}"
      : "${PARALLEL_STAGE_GATE_TEMPERATURE:=1.0}"
      ;;
    retail_rocket)
      : "${TRAIN_BATCH_SIZE:=3072}"
      : "${EVAL_BATCH_SIZE:=6144}"
      : "${EMBEDDING_SIZE:=128}"
      : "${NUM_HEADS:=8}"
      : "${D_FEAT_EMB:=16}"
      : "${D_EXPERT_HIDDEN:=160}"
      : "${D_ROUTER_HIDDEN:=64}"
      : "${EXPERT_SCALE:=3}"
      : "${BASE_LR:=0.0004}"
      : "${BASE_WD:=1e-5}"
      : "${BASE_DROP:=0.15}"
      : "${BASE_BAL:=0.01}"
      : "${LAYOUT_ID:=0}"
      : "${STAGE_MERGE_MODE:=serial}"
      : "${GROUP_ROUTER_MODE:=per_group}"
      : "${GROUP_TOP_K:=0}"
      : "${MOE_TOP_K:=0}"
      : "${MID_ROUTER_TEMPERATURE:=1.3}"
      : "${MICRO_ROUTER_TEMPERATURE:=1.3}"
      : "${MID_ROUTER_FEATURE_DROPOUT:=0.1}"
      : "${MICRO_ROUTER_FEATURE_DROPOUT:=0.1}"
      : "${EXPERT_USE_FEATURE:=false}"
      : "${MACRO_ROUTING_SCOPE:=session}"
      : "${MACRO_SESSION_POOLING:=query}"
      : "${PARALLEL_STAGE_GATE_TEMPERATURE:=1.0}"
      ;;
    *)
      echo "Unsupported DATASET=$DATASET (use movielens1m|retail_rocket)" >&2
      exit 1
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --stage-merge-mode) STAGE_MERGE_MODE="$2"; shift 2 ;;
    --group-router-mode) GROUP_ROUTER_MODE="$2"; shift 2 ;;
    --group-top-k) GROUP_TOP_K="$2"; shift 2 ;;
    --moe-top-k) MOE_TOP_K="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --parent-result|--parent_result) PARENT_RESULT="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --schedule-preset) SCHEDULE_PRESET="$2"; shift 2 ;;
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
    --learning-rate) BASE_LR="$2"; shift 2 ;;
    --weight-decay) BASE_WD="$2"; shift 2 ;;
    --dropout) BASE_DROP="$2"; shift 2 ;;
    --balance-loss-lambda) BASE_BAL="$2"; shift 2 ;;
    --mid-router-temperature) MID_ROUTER_TEMPERATURE="$2"; shift 2 ;;
    --micro-router-temperature) MICRO_ROUTER_TEMPERATURE="$2"; shift 2 ;;
    --mid-router-feature-dropout) MID_ROUTER_FEATURE_DROPOUT="$2"; shift 2 ;;
    --micro-router-feature-dropout) MICRO_ROUTER_FEATURE_DROPOUT="$2"; shift 2 ;;
    --expert-use-feature) EXPERT_USE_FEATURE="$2"; shift 2 ;;
    --macro-routing-scope) MACRO_ROUTING_SCOPE="$2"; shift 2 ;;
    --macro-session-pooling) MACRO_SESSION_POOLING="$2"; shift 2 ;;
    --parallel-stage-gate-temperature) PARALLEL_STAGE_GATE_TEMPERATURE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ -z "$DATASET" ] && { echo "--dataset required" >&2; exit 1; }
load_defaults

case "${STAGE_MERGE_MODE,,}" in
  serial|parallel) ;;
  *) echo "--stage-merge-mode must be serial|parallel" >&2; exit 1 ;;
esac
case "${GROUP_ROUTER_MODE,,}" in
  per_group|stage_wide|hybrid) ;;
  *) echo "--group-router-mode must be per_group|stage_wide|hybrid" >&2; exit 1 ;;
esac
STAGE_MERGE_MODE="${STAGE_MERGE_MODE,,}"
GROUP_ROUTER_MODE="${GROUP_ROUTER_MODE,,}"
EXPERT_USE_FEATURE="${EXPERT_USE_FEATURE,,}"
MACRO_ROUTING_SCOPE="${MACRO_ROUTING_SCOPE,,}"
MACRO_SESSION_POOLING="${MACRO_SESSION_POOLING,,}"

case "$EXPERT_USE_FEATURE" in
  true|false) ;;
  *) echo "--expert-use-feature must be true|false" >&2; exit 1 ;;
esac
case "$MACRO_ROUTING_SCOPE" in
  token|session) ;;
  *) echo "--macro-routing-scope must be token|session" >&2; exit 1 ;;
esac
case "$MACRO_SESSION_POOLING" in
  query|mean|last) ;;
  *) echo "--macro-session-pooling must be query|mean|last" >&2; exit 1 ;;
esac

case "$SEARCH_PROFILE" in
  wide)
    LR_SPACE='[5e-5,3e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4,1e-3]'
    DROP_SPACE='[0.05,0.2]'
    BAL_SPACE='[0.001,0.05]'
    ;;
  p1_shallow)
    LR_SPACE='[7.5e-5,2.5e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4,5e-4]'
    DROP_SPACE='[0.08,0.18]'
    BAL_SPACE='[0.001,0.05]'
    ;;
  confirm_narrow)
    LR_SPACE='[1.5e-4,3e-3]'
    WD_SPACE='[0.0,1e-6,1e-5,5e-5,1e-4]'
    DROP_SPACE='[0.08,0.12,0.16]'
    BAL_SPACE='[0.001,0.003,0.01,0.03]'
    ;;
  structure_refine)
    LR_SPACE='[1e-4,1e-2]'
    WD_SPACE='[0.0,1e-6,1e-5,1e-4,1e-3]'
    DROP_SPACE='[0.08,0.12,0.16]'
    BAL_SPACE='[0.001,0.003,0.01,0.03]'
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
    SCH_MID_TEMP_START="${MID_ROUTER_TEMPERATURE}"
    SCH_MICRO_TEMP_START="${MICRO_ROUTER_TEMPERATURE}"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  alpha_mild)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.2"
    SCH_ALPHA_START="0.2"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="${MID_ROUTER_TEMPERATURE}"
    SCH_MICRO_TEMP_START="${MICRO_ROUTER_TEMPERATURE}"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  alpha_cold)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.3"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="${MID_ROUTER_TEMPERATURE}"
    SCH_MICRO_TEMP_START="${MICRO_ROUTER_TEMPERATURE}"
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
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  alpha_temp_cold)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.3"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0.25"
    SCH_MID_TEMP_START="1.8"
    SCH_MICRO_TEMP_START="1.8"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0"
    ;;
  topk_mild)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0"
    SCH_ALPHA_START="0.0"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0"
    SCH_MID_TEMP_START="${MID_ROUTER_TEMPERATURE}"
    SCH_MICRO_TEMP_START="${MICRO_ROUTER_TEMPERATURE}"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0.2"
    ;;
  combined_legacy)
    SCH_ENABLE="true"
    SCH_ALPHA_UNTIL="0.3"
    SCH_ALPHA_START="0.1"
    SCH_ALPHA_END="1.0"
    SCH_TEMP_UNTIL="0.3"
    SCH_MID_TEMP_START="1.6"
    SCH_MICRO_TEMP_START="1.6"
    SCH_TOPK_START="0"
    SCH_TOPK_WARMUP="0.3"
    ;;
  *)
    echo "Unsupported --schedule-preset=${SCHEDULE_PRESET}" >&2
    exit 1
    ;;
esac

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

if [ -n "$PARENT_RESULT" ]; then
  [ ! -f "$PARENT_RESULT" ] && { echo "parent result not found: $PARENT_RESULT" >&2; exit 1; }
  read -r P_LR P_WD P_DROP P_BAL P_LAYOUT P_MERGE P_GROUP_MODE P_GROUP_TOPK \
    P_EMB P_DFEAT P_DEXP P_DROUT P_SCALE P_TRAIN_BS P_EVAL_BS P_MOE_TOPK \
    P_MID_TEMP P_MICRO_TEMP P_MID_FDROP P_MICRO_FDROP P_EXPERT_FEAT \
    P_MACRO_SCOPE P_MACRO_POOL P_PAR_TEMP <<< "$("$PY_BIN" - \
      "$PARENT_RESULT" "$LAYOUT_ID" "$STAGE_MERGE_MODE" "$GROUP_ROUTER_MODE" "$GROUP_TOP_K" \
      "$EMBEDDING_SIZE" "$D_FEAT_EMB" "$D_EXPERT_HIDDEN" "$D_ROUTER_HIDDEN" "$EXPERT_SCALE" \
      "$TRAIN_BATCH_SIZE" "$EVAL_BATCH_SIZE" "$MOE_TOP_K" "$MID_ROUTER_TEMPERATURE" \
      "$MICRO_ROUTER_TEMPERATURE" "$MID_ROUTER_FEATURE_DROPOUT" "$MICRO_ROUTER_FEATURE_DROPOUT" \
      "$EXPERT_USE_FEATURE" "$MACRO_ROUTING_SCOPE" "$MACRO_SESSION_POOLING" \
      "$PARALLEL_STAGE_GATE_TEMPERATURE" <<'PY'
import json
import sys

path = sys.argv[1]
defaults = {
    "arch_layout_id": sys.argv[2],
    "stage_merge_mode": sys.argv[3],
    "group_router_mode": sys.argv[4],
    "group_top_k": sys.argv[5],
    "embedding_size": sys.argv[6],
    "d_feat_emb": sys.argv[7],
    "d_expert_hidden": sys.argv[8],
    "d_router_hidden": sys.argv[9],
    "expert_scale": sys.argv[10],
    "train_batch_size": sys.argv[11],
    "eval_batch_size": sys.argv[12],
    "moe_top_k": sys.argv[13],
    "mid_router_temperature": sys.argv[14],
    "micro_router_temperature": sys.argv[15],
    "mid_router_feature_dropout": sys.argv[16],
    "micro_router_feature_dropout": sys.argv[17],
    "expert_use_feature": sys.argv[18],
    "macro_routing_scope": sys.argv[19],
    "macro_session_pooling": sys.argv[20],
    "parallel_stage_gate_temperature": sys.argv[21],
}

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
    pick("arch_layout_id", defaults["arch_layout_id"]),
    pick("stage_merge_mode", defaults["stage_merge_mode"]),
    pick("group_router_mode", defaults["group_router_mode"]),
    pick("group_top_k", defaults["group_top_k"]),
    pick("embedding_size", defaults["embedding_size"]),
    pick("d_feat_emb", defaults["d_feat_emb"]),
    pick("d_expert_hidden", defaults["d_expert_hidden"]),
    pick("d_router_hidden", defaults["d_router_hidden"]),
    pick("expert_scale", defaults["expert_scale"]),
    pick("train_batch_size", defaults["train_batch_size"]),
    pick("eval_batch_size", defaults["eval_batch_size"]),
    pick("moe_top_k", defaults["moe_top_k"]),
    pick("mid_router_temperature", defaults["mid_router_temperature"]),
    pick("micro_router_temperature", defaults["micro_router_temperature"]),
    pick("mid_router_feature_dropout", defaults["mid_router_feature_dropout"]),
    pick("micro_router_feature_dropout", defaults["micro_router_feature_dropout"]),
    pick("expert_use_feature", defaults["expert_use_feature"]),
    pick("macro_routing_scope", defaults["macro_routing_scope"]),
    pick("macro_session_pooling", defaults["macro_session_pooling"]),
    pick("parallel_stage_gate_temperature", defaults["parallel_stage_gate_temperature"]),
)
PY
)"
  BASE_LR="${P_LR}"
  BASE_WD="${P_WD}"
  BASE_DROP="${P_DROP}"
  BASE_BAL="${P_BAL}"
  LAYOUT_ID="${P_LAYOUT}"
  STAGE_MERGE_MODE="${P_MERGE}"
  GROUP_ROUTER_MODE="${P_GROUP_MODE}"
  GROUP_TOP_K="${P_GROUP_TOPK}"
  EMBEDDING_SIZE="${P_EMB}"
  D_FEAT_EMB="${P_DFEAT}"
  D_EXPERT_HIDDEN="${P_DEXP}"
  D_ROUTER_HIDDEN="${P_DROUT}"
  EXPERT_SCALE="${P_SCALE}"
  TRAIN_BATCH_SIZE="${P_TRAIN_BS}"
  EVAL_BATCH_SIZE="${P_EVAL_BS}"
  MOE_TOP_K="${P_MOE_TOPK}"
  MID_ROUTER_TEMPERATURE="${P_MID_TEMP}"
  MICRO_ROUTER_TEMPERATURE="${P_MICRO_TEMP}"
  MID_ROUTER_FEATURE_DROPOUT="${P_MID_FDROP}"
  MICRO_ROUTER_FEATURE_DROPOUT="${P_MICRO_FDROP}"
  EXPERT_USE_FEATURE="${P_EXPERT_FEAT}"
  MACRO_ROUTING_SCOPE="${P_MACRO_SCOPE}"
  MACRO_SESSION_POOLING="${P_MACRO_POOL}"
  PARALLEL_STAGE_GATE_TEMPERATURE="${P_PAR_TEMP}"
fi

EXPERT_USE_FEATURE="${EXPERT_USE_FEATURE,,}"
MACRO_ROUTING_SCOPE="${MACRO_ROUTING_SCOPE,,}"
MACRO_SESSION_POOLING="${MACRO_SESSION_POOLING,,}"

if [ -z "$EXP_NAME" ]; then
  EXP_NAME="fmoe_hgr_${PHASE%%_*}_hparam"
fi
if [ -z "$EXP_DESC" ]; then
  EXP_DESC="HGR hparam search (${SEARCH_PROFILE}) with fixed routing/layout combo; optimize LR/WD (+optional dropout/balance)."
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="stage_merge_mode,group_router_mode,arch_layout_id,group_top_k,moe_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"
fi

LOG_FILE_PATH="$(run_make_log_path fmoe_hgr hparam "$DATASET" "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" "$GPU_ID" "$PHASE")"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_hgr
  --run-axis hparam
  --run-phase "$PHASE"
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
  "train_batch_size=${TRAIN_BATCH_SIZE}"
  "eval_batch_size=${EVAL_BATCH_SIZE}"
  "MAX_ITEM_LIST_LENGTH=10"
  "++search.MAX_ITEM_LIST_LENGTH=[10]"
  "num_layers=-1"
  "++search.num_layers=[-1]"
  "embedding_size=${EMBEDDING_SIZE}"
  "hidden_size=${EMBEDDING_SIZE}"
  "++search.embedding_size=[${EMBEDDING_SIZE}]"
  "++search.hidden_size=[${EMBEDDING_SIZE}]"
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
  "arch_layout_id=${LAYOUT_ID}"
  "++search.arch_layout_id=[${LAYOUT_ID}]"
  "stage_merge_mode=${STAGE_MERGE_MODE}"
  "++search.stage_merge_mode=[${STAGE_MERGE_MODE}]"
  "group_router_mode=${GROUP_ROUTER_MODE}"
  "++search.group_router_mode=[${GROUP_ROUTER_MODE}]"
  "group_top_k=${GROUP_TOP_K}"
  "++search.group_top_k=[${GROUP_TOP_K}]"
  "moe_top_k=${MOE_TOP_K}"
  "++search.moe_top_k=[${MOE_TOP_K}]"
  "expert_use_feature=${EXPERT_USE_FEATURE}"
  "++search.expert_use_feature=[${EXPERT_USE_FEATURE}]"
  "macro_routing_scope=${MACRO_ROUTING_SCOPE}"
  "++search.macro_routing_scope=[${MACRO_ROUTING_SCOPE}]"
  "macro_session_pooling=${MACRO_SESSION_POOLING}"
  "++search.macro_session_pooling=[${MACRO_SESSION_POOLING}]"
  "parallel_stage_gate_temperature=${PARALLEL_STAGE_GATE_TEMPERATURE}"
  "++search.parallel_stage_gate_temperature=[${PARALLEL_STAGE_GATE_TEMPERATURE}]"
  "mid_router_temperature=${MID_ROUTER_TEMPERATURE}"
  "++search.mid_router_temperature=[${MID_ROUTER_TEMPERATURE}]"
  "micro_router_temperature=${MICRO_ROUTER_TEMPERATURE}"
  "++search.micro_router_temperature=[${MICRO_ROUTER_TEMPERATURE}]"
  "mid_router_feature_dropout=${MID_ROUTER_FEATURE_DROPOUT}"
  "++search.mid_router_feature_dropout=[${MID_ROUTER_FEATURE_DROPOUT}]"
  "micro_router_feature_dropout=${MICRO_ROUTER_FEATURE_DROPOUT}"
  "++search.micro_router_feature_dropout=[${MICRO_ROUTER_FEATURE_DROPOUT}]"
  "use_valid_ratio_gating=true"
  "++search.use_valid_ratio_gating=[true]"
  "learning_rate=${BASE_LR}"
  "+weight_decay=${BASE_WD}"
  "hidden_dropout_prob=${BASE_DROP}"
  "balance_loss_lambda=${BASE_BAL}"
  "++search.learning_rate=${LR_SPACE}"
  "++search.weight_decay=${WD_SPACE}"
  "++search.hidden_dropout_prob=${DROP_SPACE}"
  "++search.balance_loss_lambda=${BAL_SPACE}"
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
  --track fmoe_hgr \
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" \
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
  --axis hparam \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "FeaturedMoE_HGR_${STAGE_MERGE_MODE}_${GROUP_ROUTER_MODE}" \
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
