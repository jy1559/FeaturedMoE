#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m,retail_rocket"
GPU_LIST="4,5,6,7"
SEED_BASE="1620"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
COMBOS_PER_GPU="3"
LR_RANGE="1e-4,2e-2"
WD_VALUES="0.0,1e-6,1e-5,1e-4,1e-3"
DROPOUT_VALUES="0.05,0.1,0.15,0.2,0.25"
BALANCE_VALUES="0.003,0.01,0.03,0.05,0.1"

# Major architecture combos for ML1 first pass.
# Axis-1: merge_mode x allocator_top_k (6 combos)
HIR2_COMBO_MODES=("serial_weighted" "serial_weighted" "serial_weighted" "parallel_weighted" "parallel_weighted" "parallel_weighted")
HIR2_COMBO_TOPKS=("0" "1" "2" "0" "1" "2")
# Axis-2: attention-depth profile (6 combos)
#   L0: no extra transformer layers
#   L1: shallow stage-aware depth
#   L2: deeper global post and micro pre
#   L3: deepest profile for stress-test
#   L4/L5: deep-post emphasis variants
HIR2_LAYER_PROFILE_TAGS=("L0" "L1" "L2" "L3" "L4" "L5")
HIR2_LAYER_GLOBAL_PRE=("0" "1" "1" "2" "2" "1")
HIR2_LAYER_GLOBAL_POST=("0" "1" "2" "2" "3" "3")
HIR2_LAYER_MACRO_PRE=("0" "0" "1" "1" "1" "0")
HIR2_LAYER_MID_PRE=("0" "1" "1" "2" "2" "2")
HIR2_LAYER_MICRO_PRE=("0" "1" "2" "2" "3" "3")

# Axis-3: capacity profile (top FMoE-v2 inspired dimensions)
HIR2_ARCH_PROFILE_TAGS=("A0" "A1" "A2" "A3")
HIR2_ARCH_EMBEDDING=("128" "160" "160" "128")
HIR2_ARCH_D_FEAT=("16" "16" "24" "16")
HIR2_ARCH_D_EXPERT=("128" "160" "192" "512")
HIR2_ARCH_D_ROUTER=("64" "80" "96" "64")
HIR2_ARCH_EXPERT_SCALE=("3" "3" "3" "3")
HIR2_ARCH_TRAIN_BS=("8192" "6144" "6144" "3072")
HIR2_ARCH_EVAL_BS=("16384" "12288" "12288" "6144")

# Axis-4: stage allocator/stability profile
HIR2_STAGE_CTRL_TAGS=("S0" "S1" "S2")
HIR2_STAGE_ALLOC_TEMP=("1.0" "0.9" "0.8")
HIR2_STAGE_ALLOC_POOLING=("query" "query" "mean")
HIR2_STAGE_DELTA_SCALE=("1.0" "1.5" "2.0")
HIR2_STAGE_WEIGHT_FLOOR=("0.0" "0.05" "0.1")
HIR2_STAGE_ENTROPY_AUX=("0.0" "5e-4" "1e-3")

# Budget: ML1 first pass = (#gpus * combos_per_gpu) independent searches
ML1_SEARCH_EVALS="30"
# Budget: RR transfer
RR_TRANSFER_EVALS="30"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 4,5,6,7]
          [--seed-base 1620] [--combos-per-gpu 3]
          [--tune-epochs 100] [--tune-patience 10]
          [--ml1-search-evals 30] [--rr-transfer-evals 30]
          [--lr-range 1e-4,2e-2] [--wd-values 0.0,1e-6,1e-5,1e-4,1e-3]
          [--dropout-values 0.05,0.1,0.15,0.2,0.25]
          [--balance-values 0.003,0.01,0.03,0.05,0.1]
          [--log-wandb|--no-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus|--group-b-gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --ml1-search-evals) ML1_SEARCH_EVALS="$2"; shift 2 ;;
    # Backward-compatible aliases
    --ml1-struct-evals) ML1_SEARCH_EVALS="$2"; shift 2 ;;
    --ml1-refine-evals) ML1_SEARCH_EVALS="$2"; shift 2 ;;
    --rr-transfer-evals) RR_TRANSFER_EVALS="$2"; shift 2 ;;
    --lr-range) LR_RANGE="$2"; shift 2 ;;
    --wd-values) WD_VALUES="$2"; shift 2 ;;
    --dropout-values) DROPOUT_VALUES="$2"; shift 2 ;;
    --balance-values) BALANCE_VALUES="$2"; shift 2 ;;
    # Backward-compatible aliases
    --dropout-range) DROPOUT_VALUES="$2"; shift 2 ;;
    --balance-range) BALANCE_VALUES="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DS_ARR
[ "${#DS_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }
if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi
if [[ "${LR_RANGE}" != *,* ]]; then
  echo "--lr-range must be min,max" >&2
  exit 1
fi
if [[ "${WD_VALUES}" != *,* ]]; then
  echo "--wd-values must contain comma-separated values" >&2
  exit 1
fi
if [[ "${DROPOUT_VALUES}" != *,* ]]; then
  echo "--dropout-values must contain comma-separated values" >&2
  exit 1
fi
if [[ "${BALANCE_VALUES}" != *,* ]]; then
  echo "--balance-values must contain comma-separated values" >&2
  exit 1
fi
if [ "${#HIR2_COMBO_MODES[@]}" -ne "${#HIR2_COMBO_TOPKS[@]}" ]; then
  echo "HIR2 combo spec size mismatch: modes=${#HIR2_COMBO_MODES[@]} topks=${#HIR2_COMBO_TOPKS[@]}" >&2
  exit 1
fi
layer_profile_count="${#HIR2_LAYER_PROFILE_TAGS[@]}"
if [ "$layer_profile_count" -le 0 ]; then
  echo "HIR2 layer profile list cannot be empty." >&2
  exit 1
fi
if [ "$layer_profile_count" -ne "${#HIR2_LAYER_GLOBAL_PRE[@]}" ] \
  || [ "$layer_profile_count" -ne "${#HIR2_LAYER_GLOBAL_POST[@]}" ] \
  || [ "$layer_profile_count" -ne "${#HIR2_LAYER_MACRO_PRE[@]}" ] \
  || [ "$layer_profile_count" -ne "${#HIR2_LAYER_MID_PRE[@]}" ] \
  || [ "$layer_profile_count" -ne "${#HIR2_LAYER_MICRO_PRE[@]}" ]; then
  echo "HIR2 layer profile spec size mismatch." >&2
  exit 1
fi
arch_profile_count="${#HIR2_ARCH_PROFILE_TAGS[@]}"
if [ "$arch_profile_count" -le 0 ]; then
  echo "HIR2 arch profile list cannot be empty." >&2
  exit 1
fi
if [ "$arch_profile_count" -ne "${#HIR2_ARCH_EMBEDDING[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_D_FEAT[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_D_EXPERT[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_D_ROUTER[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_EXPERT_SCALE[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_TRAIN_BS[@]}" ] \
  || [ "$arch_profile_count" -ne "${#HIR2_ARCH_EVAL_BS[@]}" ]; then
  echo "HIR2 arch profile spec size mismatch." >&2
  exit 1
fi
stage_ctrl_count="${#HIR2_STAGE_CTRL_TAGS[@]}"
if [ "$stage_ctrl_count" -le 0 ]; then
  echo "HIR2 stage control profile list cannot be empty." >&2
  exit 1
fi
if [ "$stage_ctrl_count" -ne "${#HIR2_STAGE_ALLOC_TEMP[@]}" ] \
  || [ "$stage_ctrl_count" -ne "${#HIR2_STAGE_ALLOC_POOLING[@]}" ] \
  || [ "$stage_ctrl_count" -ne "${#HIR2_STAGE_DELTA_SCALE[@]}" ] \
  || [ "$stage_ctrl_count" -ne "${#HIR2_STAGE_WEIGHT_FLOOR[@]}" ] \
  || [ "$stage_ctrl_count" -ne "${#HIR2_STAGE_ENTROPY_AUX[@]}" ]; then
  echo "HIR2 stage control profile spec size mismatch." >&2
  exit 1
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

dataset_has() {
  local target="$1"
  local ds
  for ds in "${DS_ARR[@]}"; do
    if [ "$ds" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

batch_sizes() {
  local ds="$1"
  case "$ds" in
    movielens1m) echo "8192 16384" ;;
    retail_rocket) echo "4096 8192" ;;
    *) echo "4096 8192" ;;
  esac
}

gcd_int() {
  local a="$1"
  local b="$2"
  while [ "$b" -ne 0 ]; do
    local t=$((a % b))
    a="$b"
    b="$t"
  done
  echo "$a"
}

phase_top_results() {
  local dataset="$1"
  local phase_prefix="$2"
  local topn="$3"
  "$PY_BIN" - <<'PY' "$dataset" "$phase_prefix" "$topn"
import json
import os
import sys
from pathlib import Path

dataset = sys.argv[1]
phase_prefix = sys.argv[2]
topn = int(sys.argv[3])
root = Path(os.environ.get("HYPEROPT_RESULTS_DIR", "run/artifacts/results")) / "fmoe_hir2"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob(f"{dataset}_FeaturedMoE_HiR2_*.json"):
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        continue
    phase = str(d.get("run_phase", ""))
    if not phase.startswith(phase_prefix):
        continue
    score = d.get("best_mrr@20")
    if not isinstance(score, (int, float)):
        bvr = d.get("best_valid_result", {})
        score = bvr.get("mrr@20", float("-inf")) if isinstance(bvr, dict) else float("-inf")
    rows.append((float(score), p.stat().st_mtime, p))

rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
for _, _, path in rows[:topn]:
    print(path)
PY
}

extract_fixed_overrides_from_result() {
  local result_json="$1"
  "$PY_BIN" - <<'PY' "$result_json"
import json
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
bp = d.get("best_params") or {}
fs = d.get("fixed_search") or {}

def pick(key, default):
    if key in bp:
        return bp[key]
    if key in fs:
        return fs[key]
    return default

def emit(key, value):
    if isinstance(value, bool):
        v = "true" if value else "false"
    else:
        v = value
    print(f"{key}={v}")
    print(f"++search.{key}=[{v}]")

emit("hir2_stage_merge_mode", pick("hir2_stage_merge_mode", "serial_weighted"))
emit("hir2_stage_allocator_top_k", int(pick("hir2_stage_allocator_top_k", 0)))
emit("hir2_stage_allocator_temperature", float(pick("hir2_stage_allocator_temperature", 1.0)))
emit("hir2_stage_allocator_pooling", pick("hir2_stage_allocator_pooling", "query"))
emit("hir2_global_pre_layers", int(pick("hir2_global_pre_layers", 0)))
emit("hir2_global_post_layers", int(pick("hir2_global_post_layers", 0)))
emit("hir2_macro_pre_layers", int(pick("hir2_macro_pre_layers", 0)))
emit("hir2_mid_pre_layers", int(pick("hir2_mid_pre_layers", 0)))
emit("hir2_micro_pre_layers", int(pick("hir2_micro_pre_layers", 0)))
emit("hir2_stage_entropy_aux_lambda", float(pick("hir2_stage_entropy_aux_lambda", 0.0)))
emit("hir2_stage_weight_floor", float(pick("hir2_stage_weight_floor", 0.0)))
emit("hir2_stage_delta_scale", float(pick("hir2_stage_delta_scale", 1.0)))
emit("embedding_size", int(pick("embedding_size", 128)))
emit("num_heads", int(pick("num_heads", 8)))
emit("d_feat_emb", int(pick("d_feat_emb", 16)))
emit("d_expert_hidden", int(pick("d_expert_hidden", 128)))
emit("d_router_hidden", int(pick("d_router_hidden", 64)))
emit("expert_scale", int(pick("expert_scale", 3)))
emit("moe_top_k", int(pick("moe_top_k", 0)))
emit("train_batch_size", int(pick("train_batch_size", 4096)))
emit("eval_batch_size", int(pick("eval_batch_size", 8192)))
PY
}

run_hir2_search() {
  local dataset="$1"
  local gpu="$2"
  local phase="$3"
  local max_evals="$4"
  local seed="$5"
  local parent_result="$6"
  shift 6
  local -a overrides=("$@")

  read -r train_bs eval_bs <<< "$(batch_sizes "$dataset")"
  local log_file
  log_file="$(run_make_log_path fmoe_hir2 hparam "$dataset" FeaturedMoE_HiR2 "$gpu" "$phase")"

  local -a cmd=(
    "$PY_BIN" hyperopt_tune.py
    --config-name config
    --max-evals "$max_evals"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --seed "$seed"
    --run-group fmoe_hir2
    --run-axis hparam
    --run-phase "$phase"
    "model=featured_moe_hir2_tune"
    "+search_space_type_overrides={weight_decay:choice,hidden_dropout_prob:choice,balance_loss_lambda:choice,hir2_stage_entropy_aux_lambda:choice,hir2_stage_weight_floor:choice,hir2_stage_delta_scale:choice,hir2_stage_allocator_temperature:choice,hir2_stage_allocator_pooling:choice}"
    "dataset=${dataset}"
    "eval_mode=session"
    "feature_mode=full_v2"
    "gpu_id=${gpu}"
    "log_wandb=${LOG_WANDB}"
    "enable_tf32=true"
    "fmoe_debug_logging=false"
    "MAX_ITEM_LIST_LENGTH=10"
    "++search.MAX_ITEM_LIST_LENGTH=[10]"
    "train_batch_size=${train_bs}"
    "++search.train_batch_size=[${train_bs}]"
    "eval_batch_size=${eval_bs}"
    "++search.eval_batch_size=[${eval_bs}]"
    "embedding_size=128"
    "++search.embedding_size=[128]"
    "num_heads=8"
    "++search.num_heads=[8]"
    "d_feat_emb=16"
    "++search.d_feat_emb=[16]"
    "d_expert_hidden=128"
    "++search.d_expert_hidden=[128]"
    "d_router_hidden=64"
    "++search.d_router_hidden=[64]"
    "expert_scale=3"
    "++search.expert_scale=[3]"
    "moe_top_k=0"
    "++search.moe_top_k=[0]"
    "hir2_stage_allocator_pooling=query"
    "++search.hir2_stage_allocator_pooling=[query]"
    "hir2_stage_allocator_temperature=1.0"
    "++search.hir2_stage_allocator_temperature=[1.0]"
    "hir2_stage_allocator_use_hidden=true"
    "++search.hir2_stage_allocator_use_hidden=[true]"
    "hir2_stage_allocator_use_feature=true"
    "++search.hir2_stage_allocator_use_feature=[true]"
    "hir2_stage_entropy_aux_lambda=0.0"
    "++search.hir2_stage_entropy_aux_lambda=[0.0]"
    "hir2_stage_weight_floor=0.0"
    "++search.hir2_stage_weight_floor=[0.0]"
    "hir2_stage_delta_scale=1.0"
    "++search.hir2_stage_delta_scale=[1.0]"
    "macro_router_temperature=1.0"
    "++search.macro_router_temperature=[1.0]"
    "mid_router_temperature=1.3"
    "++search.mid_router_temperature=[1.3]"
    "micro_router_temperature=1.3"
    "++search.micro_router_temperature=[1.3]"
    "mid_router_feature_dropout=0.1"
    "++search.mid_router_feature_dropout=[0.1]"
    "micro_router_feature_dropout=0.1"
    "++search.micro_router_feature_dropout=[0.1]"
    "use_valid_ratio_gating=true"
    "++search.use_valid_ratio_gating=[true]"
    "learning_rate=0.001"
    "balance_loss_lambda=0.01"
    "++search.learning_rate=[${LR_RANGE}]"
    "++search.weight_decay=[${WD_VALUES}]"
    "++search.hidden_dropout_prob=[${DROPOUT_VALUES}]"
    "++search.balance_loss_lambda=[${BALANCE_VALUES}]"
  )
  if [ -n "$parent_result" ]; then
    cmd+=(--parent-result "$parent_result")
  fi
  local ov
  for ov in "${overrides[@]}"; do
    cmd+=("$ov")
  done

  run_echo_cmd "${cmd[@]}"
  echo "[LOG] ${log_file}"
  if [ "$DRY_RUN" = "true" ]; then
    return 0
  fi

  local cmd_str run_id rc status
  cmd_str="$(run_cmd_str "${cmd[@]}")"
  run_id="$(run_tracker_start \
    --track fmoe_hir2 \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_HiR2" \
    --exp-name "fmoe_hir2_first_pass" \
    --exp-desc "HiR2 first pass: distributed ML1 structure search + transfer." \
    --exp-focus "hir2_stage_merge_mode,hir2_stage_allocator_top_k,hir2_stage_allocator_pooling,hir2_stage_allocator_temperature,hir2_global_pre_layers,hir2_global_post_layers,hir2_macro_pre_layers,hir2_mid_pre_layers,hir2_micro_pre_layers,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,hir2_stage_delta_scale,hir2_stage_weight_floor,hir2_stage_entropy_aux_lambda,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda" \
    --cmd "$cmd_str" \
    --log-file "$log_file")"

  set +e
  LOG_FILE="${log_file}" PYTHONUNBUFFERED=1 "${cmd[@]}"
  rc=$?
  set -e
  status="success"
  if [ "$rc" -ne 0 ]; then
    status="fail"
  fi

  run_tracker_end \
    --run-id "$run_id" \
    --track fmoe_hir2 \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_HiR2" \
    --exp-name "fmoe_hir2_first_pass" \
    --exp-desc "HiR2 first pass: distributed ML1 structure search + transfer." \
    --exp-focus "hir2_stage_merge_mode,hir2_stage_allocator_top_k,hir2_stage_allocator_pooling,hir2_stage_allocator_temperature,hir2_global_pre_layers,hir2_global_post_layers,hir2_macro_pre_layers,hir2_mid_pre_layers,hir2_micro_pre_layers,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,hir2_stage_delta_scale,hir2_stage_weight_floor,hir2_stage_entropy_aux_lambda,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda" \
    --cmd "$cmd_str" \
    --log-file "$log_file" \
    --status "$status" \
    --exit-code "$rc"

  return "$rc"
}

ml1_best_parent=""
if dataset_has "movielens1m"; then
  mode_combo_count="${#HIR2_COMBO_MODES[@]}"
  total_pattern_count=$(( mode_combo_count * layer_profile_count * arch_profile_count * stage_ctrl_count ))
  layer_stride=5
  while [ "$(gcd_int "$layer_stride" "$layer_profile_count")" -ne 1 ]; do
    layer_stride=$((layer_stride + 2))
  done
  arch_stride=3
  while [ "$(gcd_int "$arch_stride" "$arch_profile_count")" -ne 1 ]; do
    arch_stride=$((arch_stride + 2))
  done
  stage_ctrl_stride=2
  while [ "$(gcd_int "$stage_ctrl_stride" "$stage_ctrl_count")" -ne 1 ]; do
    stage_ctrl_stride=$((stage_ctrl_stride + 2))
  done
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "[HIR2] ml1_total_jobs=${total_jobs} pattern_count=${total_pattern_count} strides(layer=${layer_stride},arch=${arch_stride},ctrl=${stage_ctrl_stride}) (mode_topk=${mode_combo_count} x layer_profile=${layer_profile_count} x arch_profile=${arch_profile_count} x stage_ctrl=${stage_ctrl_count})"

  pids=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        job_idx=$(( gidx * COMBOS_PER_GPU + slot ))
        mode_idx=$(( job_idx % mode_combo_count ))
        layer_idx=$(( (job_idx * layer_stride) % layer_profile_count ))
        arch_idx=$(( (job_idx * arch_stride) % arch_profile_count ))
        stage_ctrl_idx=$(( (job_idx * stage_ctrl_stride) % stage_ctrl_count ))
        repeat_idx=$(( job_idx / total_pattern_count + 1 ))
        merge_mode="${HIR2_COMBO_MODES[$mode_idx]}"
        alloc_top_k="${HIR2_COMBO_TOPKS[$mode_idx]}"
        layer_tag="${HIR2_LAYER_PROFILE_TAGS[$layer_idx]}"
        layer_global_pre="${HIR2_LAYER_GLOBAL_PRE[$layer_idx]}"
        layer_global_post="${HIR2_LAYER_GLOBAL_POST[$layer_idx]}"
        layer_macro_pre="${HIR2_LAYER_MACRO_PRE[$layer_idx]}"
        layer_mid_pre="${HIR2_LAYER_MID_PRE[$layer_idx]}"
        layer_micro_pre="${HIR2_LAYER_MICRO_PRE[$layer_idx]}"
        arch_tag="${HIR2_ARCH_PROFILE_TAGS[$arch_idx]}"
        emb="${HIR2_ARCH_EMBEDDING[$arch_idx]}"
        dfeat="${HIR2_ARCH_D_FEAT[$arch_idx]}"
        dexp="${HIR2_ARCH_D_EXPERT[$arch_idx]}"
        drouter="${HIR2_ARCH_D_ROUTER[$arch_idx]}"
        escale="${HIR2_ARCH_EXPERT_SCALE[$arch_idx]}"
        train_bs="${HIR2_ARCH_TRAIN_BS[$arch_idx]}"
        eval_bs="${HIR2_ARCH_EVAL_BS[$arch_idx]}"
        ctrl_tag="${HIR2_STAGE_CTRL_TAGS[$stage_ctrl_idx]}"
        alloc_temp="${HIR2_STAGE_ALLOC_TEMP[$stage_ctrl_idx]}"
        alloc_pool="${HIR2_STAGE_ALLOC_POOLING[$stage_ctrl_idx]}"
        delta_scale="${HIR2_STAGE_DELTA_SCALE[$stage_ctrl_idx]}"
        weight_floor="${HIR2_STAGE_WEIGHT_FLOOR[$stage_ctrl_idx]}"
        entropy_aux="${HIR2_STAGE_ENTROPY_AUX[$stage_ctrl_idx]}"
        merge_tag="SER"
        if [ "$merge_mode" = "parallel_weighted" ]; then
          merge_tag="PAR"
        fi
        phase="HIR2_ML1_G${gpu}_C$((slot + 1))_${merge_tag}_TK${alloc_top_k}_${layer_tag}_${arch_tag}_${ctrl_tag}_R${repeat_idx}"
        seed=$((SEED_BASE + 100 + job_idx))
        echo "[HIR2][ML1] gpu=${gpu} slot=$((slot + 1)) job=${job_idx}/${total_jobs} mode=${merge_mode} topk=${alloc_top_k} layer=${layer_tag} arch=${arch_tag}(emb=${emb},dfeat=${dfeat},dexp=${dexp},drouter=${drouter},bs=${train_bs}/${eval_bs}) ctrl=${ctrl_tag}(temp=${alloc_temp},pool=${alloc_pool},delta=${delta_scale},floor=${weight_floor},ent=${entropy_aux})"
        combo_overrides=(
          "hir2_stage_merge_mode=${merge_mode}"
          "++search.hir2_stage_merge_mode=[${merge_mode}]"
          "hir2_stage_allocator_top_k=${alloc_top_k}"
          "++search.hir2_stage_allocator_top_k=[${alloc_top_k}]"
          "hir2_stage_allocator_temperature=${alloc_temp}"
          "++search.hir2_stage_allocator_temperature=[${alloc_temp}]"
          "hir2_stage_allocator_pooling=${alloc_pool}"
          "++search.hir2_stage_allocator_pooling=[${alloc_pool}]"
          "hir2_global_pre_layers=${layer_global_pre}"
          "++search.hir2_global_pre_layers=[${layer_global_pre}]"
          "hir2_global_post_layers=${layer_global_post}"
          "++search.hir2_global_post_layers=[${layer_global_post}]"
          "hir2_macro_pre_layers=${layer_macro_pre}"
          "++search.hir2_macro_pre_layers=[${layer_macro_pre}]"
          "hir2_mid_pre_layers=${layer_mid_pre}"
          "++search.hir2_mid_pre_layers=[${layer_mid_pre}]"
          "hir2_micro_pre_layers=${layer_micro_pre}"
          "++search.hir2_micro_pre_layers=[${layer_micro_pre}]"
          "hir2_stage_delta_scale=${delta_scale}"
          "++search.hir2_stage_delta_scale=[${delta_scale}]"
          "hir2_stage_weight_floor=${weight_floor}"
          "++search.hir2_stage_weight_floor=[${weight_floor}]"
          "hir2_stage_entropy_aux_lambda=${entropy_aux}"
          "++search.hir2_stage_entropy_aux_lambda=[${entropy_aux}]"
          "embedding_size=${emb}"
          "++search.embedding_size=[${emb}]"
          "num_heads=8"
          "++search.num_heads=[8]"
          "d_feat_emb=${dfeat}"
          "++search.d_feat_emb=[${dfeat}]"
          "d_expert_hidden=${dexp}"
          "++search.d_expert_hidden=[${dexp}]"
          "d_router_hidden=${drouter}"
          "++search.d_router_hidden=[${drouter}]"
          "expert_scale=${escale}"
          "++search.expert_scale=[${escale}]"
          "train_batch_size=${train_bs}"
          "++search.train_batch_size=[${train_bs}]"
          "eval_batch_size=${eval_bs}"
          "++search.eval_batch_size=[${eval_bs}]"
        )
        run_hir2_search movielens1m "$gpu" "$phase" "${ML1_SEARCH_EVALS}" "$seed" "" "${combo_overrides[@]}"
      done
    ) &
    pids+=("$!")
  done

  rc=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  if [ "$rc" -ne 0 ]; then
    echo "[HIR2] ML1 first-pass phase failed." >&2
    exit 1
  fi

  if [ "$DRY_RUN" != "true" ]; then
    mapfile -t ml1_best < <(phase_top_results movielens1m HIR2_ML1_ 1)
    if [ "${#ml1_best[@]}" -gt 0 ]; then
      ml1_best_parent="${ml1_best[0]}"
    fi
  fi
fi

if dataset_has "retail_rocket"; then
  rr_overrides=()
  if [ -n "$ml1_best_parent" ] && [ "$DRY_RUN" != "true" ]; then
    mapfile -t rr_overrides < <(extract_fixed_overrides_from_result "$ml1_best_parent")
    run_hir2_search retail_rocket "${GPUS[0]}" HIR2_RR_TRANSFER "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 31))" "$ml1_best_parent" "${rr_overrides[@]}"
  else
    rr_overrides=(
      "hir2_stage_merge_mode=serial_weighted"
      "++search.hir2_stage_merge_mode=[serial_weighted]"
      "hir2_stage_allocator_top_k=0"
      "++search.hir2_stage_allocator_top_k=[0]"
      "hir2_stage_allocator_temperature=0.9"
      "++search.hir2_stage_allocator_temperature=[0.9]"
      "hir2_stage_allocator_pooling=query"
      "++search.hir2_stage_allocator_pooling=[query]"
      "hir2_global_pre_layers=2"
      "++search.hir2_global_pre_layers=[2]"
      "hir2_global_post_layers=2"
      "++search.hir2_global_post_layers=[2]"
      "hir2_macro_pre_layers=1"
      "++search.hir2_macro_pre_layers=[1]"
      "hir2_mid_pre_layers=2"
      "++search.hir2_mid_pre_layers=[2]"
      "hir2_micro_pre_layers=2"
      "++search.hir2_micro_pre_layers=[2]"
      "hir2_stage_delta_scale=1.5"
      "++search.hir2_stage_delta_scale=[1.5]"
      "hir2_stage_weight_floor=0.05"
      "++search.hir2_stage_weight_floor=[0.05]"
      "hir2_stage_entropy_aux_lambda=5e-4"
      "++search.hir2_stage_entropy_aux_lambda=[5e-4]"
      "embedding_size=160"
      "++search.embedding_size=[160]"
      "num_heads=8"
      "++search.num_heads=[8]"
      "d_feat_emb=24"
      "++search.d_feat_emb=[24]"
      "d_expert_hidden=192"
      "++search.d_expert_hidden=[192]"
      "d_router_hidden=96"
      "++search.d_router_hidden=[96]"
      "expert_scale=3"
      "++search.expert_scale=[3]"
      "train_batch_size=4096"
      "++search.train_batch_size=[4096]"
      "eval_batch_size=8192"
      "++search.eval_batch_size=[8192]"
    )
    run_hir2_search retail_rocket "${GPUS[0]}" HIR2_RR_TRANSFER "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 31))" "" "${rr_overrides[@]}"
  fi
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_model_report \
    fmoe_hir2 \
    FeaturedMoE_HiR2 \
    "$(run_experiments_dir)/models/FeaturedMoE_HiR2"
  run_update_track_report fmoe_hir2
fi
