#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET=""
GPU_ID="0"
SEED="42"
PHASE="P0"
RUN_AXIS="hparam"
MAX_EVALS="8"
TUNE_EPOCHS="18"
TUNE_PATIENCE="3"
LAYOUT_ID="7"
EXECUTION="serial"
PARENT_RESULT=""
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
RESULT_PATH_FILE=""
LOG_PATH_FILE=""

MAX_ITEM_LIST_LENGTH="10"
TRAIN_BATCH_SIZE=""
EVAL_BATCH_SIZE=""
EMBEDDING_SIZE="128"
NUM_HEADS="8"
D_FEAT_EMB=""
D_EXPERT_HIDDEN=""
D_ROUTER_HIDDEN="64"
EXPERT_SCALE=""

LEARNING_RATE=""
WEIGHT_DECAY=""
HIDDEN_DROPOUT=""
BALANCE_LOSS_LAMBDA=""
HAS_LR_OVERRIDE="false"
HAS_WD_OVERRIDE="false"
HAS_DROPOUT_OVERRIDE="false"
HAS_BALANCE_OVERRIDE="false"

LR_SPACE_OVERRIDE=""
WD_SPACE_OVERRIDE=""
DROPOUT_SPACE_OVERRIDE=""
BALANCE_SPACE_OVERRIDE=""

ROUTER_FAMILY="plain"
ROUTER_IMPL="learned"
ROUTER_IMPL_BY_STAGE=""
RULE_BIAS_SCALE=""
FEATURE_ENCODER_MODE="linear"
FEATURE_ENCODER_SIN_N_FREQS="4"
MOE_BLOCK_VARIANT="moe"
ROUTER_GROUP_FEATURE_MODE="none"
Z_LOSS_LAMBDA="0.0"
GATE_ENTROPY_LAMBDA="0.0"
GATE_ENTROPY_UNTIL="0.0"
MOE_TOP_K="0"
MOE_TOP_K_POLICY="auto"
MOE_TOP_K_RATIO="0.5"

RULE_VARIANT="ratio_bins"
RULE_N_BINS="5"
RULE_FEATURE_PER_EXPERT="4"

MID_ROUTER_TEMPERATURE="1.2"
MICRO_ROUTER_TEMPERATURE="1.2"
FMOE_SCHEDULE_ENABLE="false"
STAGE_INTER_LAYER_STYLE="attn"
STATE_TAG=""
COMBO_DESC=""

EXP_NAME=""
EXP_DESC=""
EXP_FOCUS=""
EXTRA_OVERRIDES=()

usage() {
  cat <<USAGE
Usage: $0 --dataset <dataset> [--gpu N] [--seed N] [--phase P0_Q01]
          [--run-axis hparam|arch_s01_base] [--state-tag S01_base]
          [--layout-id N] [--execution serial|parallel]
          [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--router-family plain|hybrid|bias|rule]
          [--router-impl learned|rule_soft]
          [--stage-inter-layer-style attn|identity|nonlinear|ffn]
          [--moe-block-variant moe|dense_ffn|nonlinear|identity]
          [--router-group-feature-mode none|mean|mean_std]
          [--feature-encoder-mode linear|sinusoidal_selected]
          [--expert-scale N] [--moe-top-k N] [--moe-top-k-policy auto|fixed]
          [--z-loss-lambda X] [--gate-entropy-lambda X] [--gate-entropy-until X]
          [--lr-space lo,hi] [--wd-space csv] [--dropout-space csv] [--balance-space csv]
          [--parent-result path] [--override hydra.key=value]
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

phase_slug() {
  local phase="$1"
  python3 - <<'PY' "$phase"
import re
import sys
phase = str(sys.argv[1] or "").strip().lower()
phase = re.sub(r"[^a-z0-9._-]+", "_", phase).strip("._-")
print(phase[:80])
PY
}

write_optional_path_file() {
  local path_file="$1"
  local value="$2"
  [ -z "$path_file" ] && return 0
  mkdir -p "$(dirname "$path_file")"
  printf '%s\n' "$value" >"$path_file"
}

load_defaults() {
  case "$DATASET" in
    KuaiRecSmall0.1)
      DEF_TRAIN_BS="6144"
      DEF_EVAL_BS="12288"
      DEF_D_FEAT="16"
      DEF_D_EXP="128"
      DEF_SCALE="3"
      DEF_LR="0.001"
      DEF_WD="5e-5"
      DEF_DROP="0.10"
      DEF_BAL="0.002"
      ;;
    lastfm0.03)
      DEF_TRAIN_BS="4096"
      DEF_EVAL_BS="4096"
      DEF_D_FEAT="16"
      DEF_D_EXP="128"
      DEF_SCALE="3"
      DEF_LR="8e-4"
      DEF_WD="5e-5"
      DEF_DROP="0.10"
      DEF_BAL="0.002"
      ;;
    *)
      echo "Unsupported dataset for fmoe_n runner: ${DATASET}" >&2
      exit 1
      ;;
  esac
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --run-axis) RUN_AXIS="$2"; shift 2 ;;
    --state-tag) STATE_TAG="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --execution) EXECUTION="$2"; shift 2 ;;
    --parent-result|--parent_result) PARENT_RESULT="$2"; shift 2 ;;
    --router-family) ROUTER_FAMILY="$2"; shift 2 ;;
    --router-impl) ROUTER_IMPL="$2"; shift 2 ;;
    --router-impl-by-stage) ROUTER_IMPL_BY_STAGE="$2"; shift 2 ;;
    --rule-bias-scale) RULE_BIAS_SCALE="$2"; shift 2 ;;
    --feature-encoder-mode) FEATURE_ENCODER_MODE="$2"; shift 2 ;;
    --feature-encoder-sinusoidal-n-freqs) FEATURE_ENCODER_SIN_N_FREQS="$2"; shift 2 ;;
    --moe-block-variant) MOE_BLOCK_VARIANT="$2"; shift 2 ;;
    --router-group-feature-mode) ROUTER_GROUP_FEATURE_MODE="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --embedding-size) EMBEDDING_SIZE="$2"; shift 2 ;;
    --num-heads) NUM_HEADS="$2"; shift 2 ;;
    --d-feat-emb) D_FEAT_EMB="$2"; shift 2 ;;
    --d-expert-hidden) D_EXPERT_HIDDEN="$2"; shift 2 ;;
    --d-router-hidden) D_ROUTER_HIDDEN="$2"; shift 2 ;;
    --expert-scale) EXPERT_SCALE="$2"; shift 2 ;;
    --learning-rate) LEARNING_RATE="$2"; HAS_LR_OVERRIDE="true"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; HAS_WD_OVERRIDE="true"; shift 2 ;;
    --hidden-dropout) HIDDEN_DROPOUT="$2"; HAS_DROPOUT_OVERRIDE="true"; shift 2 ;;
    --balance-loss-lambda) BALANCE_LOSS_LAMBDA="$2"; HAS_BALANCE_OVERRIDE="true"; shift 2 ;;
    --z-loss-lambda) Z_LOSS_LAMBDA="$2"; shift 2 ;;
    --gate-entropy-lambda) GATE_ENTROPY_LAMBDA="$2"; shift 2 ;;
    --gate-entropy-until) GATE_ENTROPY_UNTIL="$2"; shift 2 ;;
    --lr-space) LR_SPACE_OVERRIDE="$2"; shift 2 ;;
    --wd-space) WD_SPACE_OVERRIDE="$2"; shift 2 ;;
    --dropout-space) DROPOUT_SPACE_OVERRIDE="$2"; shift 2 ;;
    --balance-space) BALANCE_SPACE_OVERRIDE="$2"; shift 2 ;;
    --moe-top-k) MOE_TOP_K="$2"; shift 2 ;;
    --moe-top-k-policy) MOE_TOP_K_POLICY="$2"; shift 2 ;;
    --moe-top-k-ratio) MOE_TOP_K_RATIO="$2"; shift 2 ;;
    --rule-variant) RULE_VARIANT="$2"; shift 2 ;;
    --rule-n-bins) RULE_N_BINS="$2"; shift 2 ;;
    --rule-feature-per-expert) RULE_FEATURE_PER_EXPERT="$2"; shift 2 ;;
    --mid-router-temperature) MID_ROUTER_TEMPERATURE="$2"; shift 2 ;;
    --micro-router-temperature) MICRO_ROUTER_TEMPERATURE="$2"; shift 2 ;;
    --fmoe-schedule-enable) FMOE_SCHEDULE_ENABLE="$2"; shift 2 ;;
    --stage-inter-layer-style) STAGE_INTER_LAYER_STYLE="$2"; shift 2 ;;
    --result-path-file) RESULT_PATH_FILE="$2"; shift 2 ;;
    --log-path-file) LOG_PATH_FILE="$2"; shift 2 ;;
    --combo-desc) COMBO_DESC="$2"; shift 2 ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      shift 2
      ;;
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
case "${EXECUTION,,}" in
  serial|parallel) ;;
  *) echo "--execution must be serial|parallel" >&2; exit 1 ;;
esac
EXECUTION="${EXECUTION,,}"
RUN_AXIS="$(run_sanitize "${RUN_AXIS}")"
[ -n "$RUN_AXIS" ] || RUN_AXIS="hparam"
RUN_AXIS="$(printf '%s' "$RUN_AXIS" | tr '[:upper:]' '[:lower:]')"

if [ -n "$STATE_TAG" ]; then
  STATE_TAG="$(run_sanitize "${STATE_TAG}")"
  [ -n "$STATE_TAG" ] || { echo "--state-tag must not sanitize to empty" >&2; exit 1; }
  RUN_AXIS="$(printf '%s' "$STATE_TAG" | tr '[:upper:]' '[:lower:]')"
fi

case "${ROUTER_FAMILY,,}" in
  plain|hybrid|bias|rule) ;;
  *) echo "--router-family must be plain|hybrid|bias|rule" >&2; exit 1 ;;
esac
ROUTER_FAMILY="${ROUTER_FAMILY,,}"

case "${ROUTER_IMPL,,}" in
  learned|rule_soft) ;;
  *) echo "--router-impl must be learned|rule_soft" >&2; exit 1 ;;
esac
ROUTER_IMPL="${ROUTER_IMPL,,}"

case "${STAGE_INTER_LAYER_STYLE,,}" in
  attn|identity|nonlinear|ffn) ;;
  *) echo "--stage-inter-layer-style must be attn|identity|nonlinear|ffn" >&2; exit 1 ;;
esac
STAGE_INTER_LAYER_STYLE="${STAGE_INTER_LAYER_STYLE,,}"

case "${MOE_BLOCK_VARIANT,,}" in
  moe|dense_ffn|nonlinear|identity) ;;
  *) echo "--moe-block-variant must be moe|dense_ffn|nonlinear|identity" >&2; exit 1 ;;
esac
MOE_BLOCK_VARIANT="${MOE_BLOCK_VARIANT,,}"

case "${ROUTER_GROUP_FEATURE_MODE,,}" in
  none|mean|mean_std) ;;
  *) echo "--router-group-feature-mode must be none|mean|mean_std" >&2; exit 1 ;;
esac
ROUTER_GROUP_FEATURE_MODE="${ROUTER_GROUP_FEATURE_MODE,,}"

load_defaults

if [ -n "$PARENT_RESULT" ]; then
  [ -f "$PARENT_RESULT" ] || { echo "parent result not found: $PARENT_RESULT" >&2; exit 1; }
  read -r P_LR P_WD P_DROP P_BAL P_LAYOUT P_EXEC <<< "$("$(run_python_bin)" - <<'PY' "$PARENT_RESULT" "$LAYOUT_ID" "$EXECUTION"
import json
import sys

path = sys.argv[1]
def_layout = sys.argv[2]
def_exec = sys.argv[3]
data = json.load(open(path, "r", encoding="utf-8"))
best = data.get("best_params") or {}
trials = data.get("trials") or []
if not best:
    ok = [t for t in trials if t.get("status") in (None, "ok") and isinstance(t.get("mrr@20"), (int, float))]
    if ok:
        ok.sort(key=lambda x: x.get("mrr@20", 0), reverse=True)
        best = ok[0].get("params") or {}
fixed = data.get("fixed_search") or {}
layout = best.get("fmoe_v2_layout_id", fixed.get("fmoe_v2_layout_id", def_layout))
execution = best.get("fmoe_stage_execution_mode", fixed.get("fmoe_stage_execution_mode", def_exec))
print(
    best.get("learning_rate", fixed.get("learning_rate", 0.001)),
    best.get("weight_decay", fixed.get("weight_decay", 5e-5)),
    best.get("hidden_dropout_prob", fixed.get("hidden_dropout_prob", 0.1)),
    best.get("balance_loss_lambda", fixed.get("balance_loss_lambda", 0.002)),
    layout,
    execution,
)
PY
)"
  if [ "$HAS_LR_OVERRIDE" != "true" ]; then
    LEARNING_RATE="$P_LR"
  fi
  if [ "$HAS_WD_OVERRIDE" != "true" ]; then
    WEIGHT_DECAY="$P_WD"
  fi
  if [ "$HAS_DROPOUT_OVERRIDE" != "true" ]; then
    HIDDEN_DROPOUT="$P_DROP"
  fi
  if [ "$HAS_BALANCE_OVERRIDE" != "true" ]; then
    BALANCE_LOSS_LAMBDA="$P_BAL"
  fi
  LAYOUT_ID="${P_LAYOUT}"
  EXECUTION="${P_EXEC}"
fi

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEF_TRAIN_BS}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-$DEF_EVAL_BS}"
D_FEAT_EMB="${D_FEAT_EMB:-$DEF_D_FEAT}"
D_EXPERT_HIDDEN="${D_EXPERT_HIDDEN:-$DEF_D_EXP}"
EXPERT_SCALE="${EXPERT_SCALE:-$DEF_SCALE}"
LEARNING_RATE="${LEARNING_RATE:-$DEF_LR}"
WEIGHT_DECAY="${WEIGHT_DECAY:-$DEF_WD}"
HIDDEN_DROPOUT="${HIDDEN_DROPOUT:-$DEF_DROP}"
BALANCE_LOSS_LAMBDA="${BALANCE_LOSS_LAMBDA:-$DEF_BAL}"

case "$ROUTER_FAMILY" in
  plain)
    FAMILY_ROUTER_KIND="learned"
    FAMILY_ROUTER_IMPL="{}"
    FAMILY_RULE_BIAS="0.0"
    ;;
  hybrid)
    FAMILY_ROUTER_KIND="learned"
    FAMILY_ROUTER_IMPL="{mid:rule_soft,micro:rule_soft}"
    FAMILY_RULE_BIAS="0.0"
    ;;
  bias)
    FAMILY_ROUTER_KIND="learned"
    FAMILY_ROUTER_IMPL="{}"
    FAMILY_RULE_BIAS="0.15"
    ;;
  rule)
    FAMILY_ROUTER_KIND="rule_soft"
    FAMILY_ROUTER_IMPL="{}"
    FAMILY_RULE_BIAS="0.0"
    ;;
esac

if [ "$ROUTER_IMPL" = "learned" ] && [ "$ROUTER_FAMILY" = "rule" ]; then
  ROUTER_IMPL="$FAMILY_ROUTER_KIND"
fi
ROUTER_IMPL_BY_STAGE="${ROUTER_IMPL_BY_STAGE:-$FAMILY_ROUTER_IMPL}"
RULE_BIAS_SCALE="${RULE_BIAS_SCALE:-$FAMILY_RULE_BIAS}"

if [ -n "$LR_SPACE_OVERRIDE" ]; then
  LR_SPACE="$(csv_to_bracket_list "$LR_SPACE_OVERRIDE")"
else
  LR_SPACE="[${LEARNING_RATE}]"
fi
if [ -n "$WD_SPACE_OVERRIDE" ]; then
  WD_SPACE="$(csv_to_bracket_list "$WD_SPACE_OVERRIDE")"
else
  WD_SPACE="[${WEIGHT_DECAY}]"
fi
if [ -n "$DROPOUT_SPACE_OVERRIDE" ]; then
  DROPOUT_SPACE="$(csv_to_bracket_list "$DROPOUT_SPACE_OVERRIDE")"
else
  DROPOUT_SPACE="[${HIDDEN_DROPOUT}]"
fi
if [ -n "$BALANCE_SPACE_OVERRIDE" ]; then
  BALANCE_SPACE="$(csv_to_bracket_list "$BALANCE_SPACE_OVERRIDE")"
else
  BALANCE_SPACE="[${BALANCE_LOSS_LAMBDA}]"
fi

if [ -z "$EXP_NAME" ]; then
  if [ -n "$STATE_TAG" ]; then
    EXP_NAME="fmoe_n_${STATE_TAG}_${PHASE%%_*}"
  else
    EXP_NAME="fmoe_n_${PHASE%%_*}_${RUN_AXIS}"
  fi
fi
if [ -z "$EXP_DESC" ]; then
  if [ -n "$STATE_TAG" ]; then
    EXP_DESC="FeaturedMoE_N state=${STATE_TAG} hyperopt run with fixed combo and LR-first search."
  else
    EXP_DESC="FeaturedMoE_N hyperopt run with fixed combo and LR-first search."
  fi
fi
if [ -z "$EXP_FOCUS" ]; then
  EXP_FOCUS="fmoe_v2_layout_id,fmoe_stage_execution_mode,router_family,router_impl,stage_inter_layer_style,moe_block_variant,router_group_feature_mode,feature_encoder_mode,expert_scale,moe_top_k,learning_rate,weight_decay,balance_loss_lambda,z_loss_lambda,gate_entropy_lambda"
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"
PHASE_BUCKET="${PHASE%%_*}"
[ -n "$PHASE_BUCKET" ] || PHASE_BUCKET="$PHASE"
SUMMARY_SCRIPT="${SCRIPT_DIR}/update_phase_summary.py"

MODEL_TRACK_NAME="FeaturedMoE_N_${EXECUTION}_${ROUTER_FAMILY}"
if [ -n "$STATE_TAG" ]; then
  MODEL_TRACK_NAME="${MODEL_TRACK_NAME}_${STATE_TAG}"
fi

AXIS_DIR_NAME="$(run_sanitize "${RUN_AXIS}")"
PHASE_DIR_NAME="$(run_sanitize "${PHASE_BUCKET}")"
DATASET_DIR_NAME="$(run_sanitize "${DATASET}")"
MODEL_DIR_NAME="$(run_model_tag "${MODEL_TRACK_NAME}")"
LOG_DIR_PATH="$(run_log_dir fmoe_n)/${AXIS_DIR_NAME}/${PHASE_DIR_NAME}/${DATASET_DIR_NAME}/${MODEL_DIR_NAME}"
run_ensure_dir "$LOG_DIR_PATH"
LOG_FILE_PATH="${LOG_DIR_PATH}/$(run_timestamp)_$(run_sanitize "${RUN_AXIS}")_$(run_sanitize "${PHASE}").log"
write_optional_path_file "$LOG_PATH_FILE" "$LOG_FILE_PATH"

cmd=(
  "$PY_BIN" hyperopt_tune.py
  --config-name config
  --max-evals "$MAX_EVALS"
  --tune-epochs "$TUNE_EPOCHS"
  --tune-patience "$TUNE_PATIENCE"
  --seed "$SEED"
  --run-group fmoe_n
  --run-axis "$RUN_AXIS"
  --run-phase "$PHASE"
  "model=featured_moe_n_tune"
  "dataset=${DATASET}"
  "eval_mode=session"
  "feature_mode=full_v2"
  "gpu_id=${GPU_ID}"
  "log_wandb=${LOG_WANDB}"
  "enable_tf32=true"
  "fmoe_debug_logging=false"
  "MAX_ITEM_LIST_LENGTH=${MAX_ITEM_LIST_LENGTH}"
  "++search.MAX_ITEM_LIST_LENGTH=[${MAX_ITEM_LIST_LENGTH}]"
  "train_batch_size=${TRAIN_BATCH_SIZE}"
  "++search.train_batch_size=[${TRAIN_BATCH_SIZE}]"
  "eval_batch_size=${EVAL_BATCH_SIZE}"
  "++search.eval_batch_size=[${EVAL_BATCH_SIZE}]"
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
  "fmoe_v2_layout_id=${LAYOUT_ID}"
  "++search.fmoe_v2_layout_id=[${LAYOUT_ID}]"
  "fmoe_stage_execution_mode=${EXECUTION}"
  "++search.fmoe_stage_execution_mode=[${EXECUTION}]"
  "router_design=simple_flat"
  "++search.router_design=[simple_flat]"
  "router_impl=${ROUTER_IMPL}"
  "++search.router_impl=[${ROUTER_IMPL}]"
  "router_use_hidden=true"
  "++search.router_use_hidden=[true]"
  "router_use_feature=true"
  "++search.router_use_feature=[true]"
  "expert_use_hidden=true"
  "++search.expert_use_hidden=[true]"
  "expert_use_feature=true"
  "++search.expert_use_feature=[true]"
  "++router_impl_by_stage=${ROUTER_IMPL_BY_STAGE}"
  "++search.router_impl_by_stage=[${ROUTER_IMPL_BY_STAGE}]"
  "macro_routing_scope=session"
  "++search.macro_routing_scope=[session]"
  "macro_session_pooling=mean"
  "++search.macro_session_pooling=[mean]"
  "feature_encoder_mode=${FEATURE_ENCODER_MODE}"
  "++search.feature_encoder_mode=[${FEATURE_ENCODER_MODE}]"
  "feature_encoder_sinusoidal_n_freqs=${FEATURE_ENCODER_SIN_N_FREQS}"
  "++search.feature_encoder_sinusoidal_n_freqs=[${FEATURE_ENCODER_SIN_N_FREQS}]"
  "moe_block_variant=${MOE_BLOCK_VARIANT}"
  "++search.moe_block_variant=[${MOE_BLOCK_VARIANT}]"
  "router_group_feature_mode=${ROUTER_GROUP_FEATURE_MODE}"
  "++search.router_group_feature_mode=[${ROUTER_GROUP_FEATURE_MODE}]"
  "rule_bias_scale=${RULE_BIAS_SCALE}"
  "++search.rule_bias_scale=[${RULE_BIAS_SCALE}]"
  "rule_router.variant=${RULE_VARIANT}"
  "rule_router.n_bins=${RULE_N_BINS}"
  "rule_router.feature_per_expert=${RULE_FEATURE_PER_EXPERT}"
  "++search={rule_router.variant:[${RULE_VARIANT}],rule_router.n_bins:[${RULE_N_BINS}],rule_router.feature_per_expert:[${RULE_FEATURE_PER_EXPERT}]}"
  "mid_router_temperature=${MID_ROUTER_TEMPERATURE}"
  "++search.mid_router_temperature=[${MID_ROUTER_TEMPERATURE}]"
  "micro_router_temperature=${MICRO_ROUTER_TEMPERATURE}"
  "++search.micro_router_temperature=[${MICRO_ROUTER_TEMPERATURE}]"
  "mid_router_feature_dropout=0.0"
  "++search.mid_router_feature_dropout=[0.0]"
  "micro_router_feature_dropout=0.0"
  "++search.micro_router_feature_dropout=[0.0]"
  "use_valid_ratio_gating=true"
  "++search.use_valid_ratio_gating=[true]"
  "learning_rate=${LEARNING_RATE}"
  "+weight_decay=${WEIGHT_DECAY}"
  "hidden_dropout_prob=${HIDDEN_DROPOUT}"
  "balance_loss_lambda=${BALANCE_LOSS_LAMBDA}"
  "z_loss_lambda=${Z_LOSS_LAMBDA}"
  "++search.z_loss_lambda=[${Z_LOSS_LAMBDA}]"
  "gate_entropy_lambda=${GATE_ENTROPY_LAMBDA}"
  "++search.gate_entropy_lambda=[${GATE_ENTROPY_LAMBDA}]"
  "gate_entropy_until=${GATE_ENTROPY_UNTIL}"
  "++search.gate_entropy_until=[${GATE_ENTROPY_UNTIL}]"
  "++search.learning_rate=${LR_SPACE}"
  "++search.weight_decay=${WD_SPACE}"
  "++search.hidden_dropout_prob=${DROPOUT_SPACE}"
  "++search.balance_loss_lambda=${BALANCE_SPACE}"
  "++search_space_type_overrides.learning_rate=loguniform"
  "moe_top_k=${MOE_TOP_K}"
  "++search.moe_top_k=[${MOE_TOP_K}]"
  "moe_top_k_policy=${MOE_TOP_K_POLICY}"
  "++search.moe_top_k_policy=[${MOE_TOP_K_POLICY}]"
  "moe_top_k_ratio=${MOE_TOP_K_RATIO}"
  "++search.moe_top_k_ratio=[${MOE_TOP_K_RATIO}]"
  "fmoe_schedule_enable=${FMOE_SCHEDULE_ENABLE}"
  "++search.fmoe_schedule_enable=[${FMOE_SCHEDULE_ENABLE}]"
  "stage_inter_layer_style=${STAGE_INTER_LAYER_STYLE}"
  "++search.stage_inter_layer_style=[${STAGE_INTER_LAYER_STYLE}]"
  "alpha_warmup_until=0"
  "++search.alpha_warmup_until=[0]"
  "alpha_warmup_start=0.0"
  "++search.alpha_warmup_start=[0.0]"
  "alpha_warmup_end=1.0"
  "++search.alpha_warmup_end=[1.0]"
  "temperature_warmup_until=0"
  "++search.temperature_warmup_until=[0]"
  "mid_router_temperature_start=${MID_ROUTER_TEMPERATURE}"
  "++search.mid_router_temperature_start=[${MID_ROUTER_TEMPERATURE}]"
  "micro_router_temperature_start=${MICRO_ROUTER_TEMPERATURE}"
  "++search.micro_router_temperature_start=[${MICRO_ROUTER_TEMPERATURE}]"
  "moe_top_k_start=${MOE_TOP_K}"
  "++search.moe_top_k_start=[${MOE_TOP_K}]"
  "moe_top_k_warmup_until=0"
  "++search.moe_top_k_warmup_until=[0]"
  "fmoe_special_logging=true"
  "++search.fmoe_special_logging=[true]"
)

if [ -n "$STATE_TAG" ]; then
  cmd+=("++arch_state_tag=${STATE_TAG}")
fi
if [ -n "$COMBO_DESC" ]; then
  cmd+=("++combo_desc=${COMBO_DESC}")
fi

[ -n "$PARENT_RESULT" ] && cmd+=(--parent-result "$PARENT_RESULT")
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi
if [ "$LOG_WANDB" = "true" ]; then
  cmd+=(--log-wandb)
fi

run_echo_cmd "${cmd[@]}"
echo "[LOG] ${LOG_FILE_PATH}"
if [ "$DRY_RUN" = "true" ]; then
  write_optional_path_file "$RESULT_PATH_FILE" ""
  exit 0
fi

CMD_STR="$(run_cmd_str "${cmd[@]}")"
RUN_ID="$(run_tracker_start \
  --track fmoe_n \
  --axis "$RUN_AXIS" \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "${MODEL_TRACK_NAME}" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH")"

set +e
if [ "$LOG_WANDB" = "true" ]; then
  WANDB_DISABLED="false" \
  LOG_FILE="${LOG_FILE_PATH}" \
  PYTHONUNBUFFERED=1 \
  FMOE_N_PHASE_SUMMARY="true" \
  FMOE_N_SUMMARY_PHASE="${PHASE_BUCKET}" \
  FMOE_N_SUMMARY_AXIS="${RUN_AXIS}" \
  "${cmd[@]}"
else
  WANDB_DISABLED="true" \
  LOG_FILE="${LOG_FILE_PATH}" \
  PYTHONUNBUFFERED=1 \
  FMOE_N_PHASE_SUMMARY="true" \
  FMOE_N_SUMMARY_PHASE="${PHASE_BUCKET}" \
  FMOE_N_SUMMARY_AXIS="${RUN_AXIS}" \
  "${cmd[@]}"
fi
RC=$?
set -e

if [ "$RC" -eq 0 ]; then
  STATUS="success"
else
  STATUS="fail"
fi

RESULT_DIR="$(run_results_dir fmoe_n)"
PHASE_SLUG="$(phase_slug "$PHASE")"
RESULT_MIRROR_DIR="${RESULT_DIR}/normal/${RUN_AXIS}/${PHASE_BUCKET}/${DATASET_DIR_NAME}/$(run_model_tag "$MODEL_TRACK_NAME")"
RESULT_PATH="$("$PY_BIN" - <<'PY' "$RESULT_DIR" "$RESULT_MIRROR_DIR" "$DATASET" "$PHASE_SLUG"
from pathlib import Path
import sys

result_dir = Path(sys.argv[1])
mirror_dir = Path(sys.argv[2])
dataset = sys.argv[3]
phase_slug = sys.argv[4]
matches = []
if mirror_dir.is_dir():
    matches = sorted(mirror_dir.glob(f"{dataset}_FeaturedMoE_N_{phase_slug}_*.json"))
if not matches:
    matches = sorted(result_dir.glob(f"{dataset}_FeaturedMoE_N_{phase_slug}_*.json"))
if matches:
    print(matches[-1])
PY
)"
write_optional_path_file "$RESULT_PATH_FILE" "$RESULT_PATH"
if [ -n "$RESULT_PATH" ]; then
  echo "[RESULT_JSON] ${RESULT_PATH}"
fi

run_tracker_end \
  --run-id "$RUN_ID" \
  --track fmoe_n \
  --axis "$RUN_AXIS" \
  --phase "$PHASE" \
  --dataset "$DATASET" \
  --model "$MODEL_TRACK_NAME" \
  --exp-name "$EXP_NAME" \
  --exp-desc "$EXP_DESC" \
  --exp-focus "$EXP_FOCUS" \
  --cmd "$CMD_STR" \
  --log-file "$LOG_FILE_PATH" \
  --status "$STATUS" \
  --exit-code "$RC"

run_update_model_report \
  fmoe_n \
  FeaturedMoE_N \
  "$(run_experiments_dir)/models/FeaturedMoE_N"
run_update_track_report fmoe_n

if [ -f "$SUMMARY_SCRIPT" ]; then
  FMOE_N_PHASE_SUMMARY="true" \
  FMOE_N_SUMMARY_PHASE="${PHASE_BUCKET}" \
  FMOE_N_SUMMARY_AXIS="${RUN_AXIS}" \
  "$PY_BIN" "$SUMMARY_SCRIPT" --phase "$PHASE_BUCKET" --axis "$RUN_AXIS" >/dev/null 2>&1 || true
fi

exit "$RC"
