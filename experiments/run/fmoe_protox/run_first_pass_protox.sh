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
SEED_BASE="2620"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
COMBOS_PER_GPU="3"

ML1_SEARCH_EVALS="40"
RR_TRANSFER_EVALS="40"

LR_RANGE="1e-4,2e-2"
WD_VALUES="0.0,1e-6,1e-5,1e-4,1e-3"
DROPOUT_VALUES="0.05,0.1,0.15,0.2,0.25"
BALANCE_VALUES="0.003,0.01,0.03,0.05,0.1"
PROTO_USAGE_VALUES="0.0,1e-4,3e-4,1e-3,3e-3"
PROTO_ENTROPY_VALUES="0.0,1e-4,3e-4,1e-3,3e-3"

# Capacity profiles (C0~C3)
PROTOX_CAP_TAGS=("C0" "C1" "C2" "C3")
PROTOX_CAP_EMBEDDING=("128" "160" "128" "160")
PROTOX_CAP_D_FEAT=("16" "24" "16" "16")
PROTOX_CAP_D_EXPERT=("160" "192" "512" "160")
PROTOX_CAP_D_ROUTER=("80" "96" "64" "80")
PROTOX_CAP_EXPERT_SCALE=("3" "3" "3" "3")
PROTOX_CAP_TRAIN_BS=("8192" "6144" "3072" "6144")
PROTOX_CAP_EVAL_BS=("16384" "12288" "6144" "12288")
PROTOX_CAP_GLOBAL_PRE=("0" "0" "0" "0")
PROTOX_CAP_GLOBAL_POST=("0" "0" "0" "2")
PROTOX_CAP_MACRO_PRE=("0" "0" "0" "0")
PROTOX_CAP_MID_PRE=("0" "0" "0" "0")
PROTOX_CAP_MICRO_PRE=("0" "0" "0" "0")

# Routing/stability profiles (R0~R2)
PROTOX_ROUTE_TAGS=("R0" "R1" "R2")
PROTOX_ROUTE_TOPK=("0" "2" "2")
PROTOX_ROUTE_TEMP_START=("1.2" "1.1" "1.0")
PROTOX_ROUTE_TEMP_END=("1.0" "0.9" "0.8")
PROTOX_ROUTE_STAGE_FLOOR=("0.0" "0.05" "0.1")
PROTOX_ROUTE_DELTA_SCALE=("1.0" "1.5" "2.0")

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 4,5,6,7]
          [--seed-base 2620] [--combos-per-gpu 3]
          [--tune-epochs 100] [--tune-patience 10]
          [--ml1-search-evals 40] [--rr-transfer-evals 40]
          [--lr-range 1e-4,2e-2] [--wd-values 0.0,1e-6,1e-5,1e-4,1e-3]
          [--dropout-values 0.05,0.1,0.15,0.2,0.25]
          [--balance-values 0.003,0.01,0.03,0.05,0.1]
          [--proto-usage-values 0.0,1e-4,3e-4,1e-3,3e-3]
          [--proto-entropy-values 0.0,1e-4,3e-4,1e-3,3e-3]
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
    --rr-transfer-evals) RR_TRANSFER_EVALS="$2"; shift 2 ;;
    --lr-range) LR_RANGE="$2"; shift 2 ;;
    --wd-values) WD_VALUES="$2"; shift 2 ;;
    --dropout-values) DROPOUT_VALUES="$2"; shift 2 ;;
    --balance-values) BALANCE_VALUES="$2"; shift 2 ;;
    --proto-usage-values) PROTO_USAGE_VALUES="$2"; shift 2 ;;
    --proto-entropy-values) PROTO_ENTROPY_VALUES="$2"; shift 2 ;;
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
for v in "$LR_RANGE" "$WD_VALUES" "$DROPOUT_VALUES" "$BALANCE_VALUES" "$PROTO_USAGE_VALUES" "$PROTO_ENTROPY_VALUES"; do
  if [[ "$v" != *,* ]]; then
    echo "Value list must be comma-separated: ${v}" >&2
    exit 1
  fi
done

cap_count="${#PROTOX_CAP_TAGS[@]}"
route_count="${#PROTOX_ROUTE_TAGS[@]}"
if [ "$cap_count" -le 0 ] || [ "$route_count" -le 0 ]; then
  echo "ProtoX profile arrays cannot be empty" >&2
  exit 1
fi

if [ "$cap_count" -ne "${#PROTOX_CAP_EMBEDDING[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_D_FEAT[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_D_EXPERT[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_D_ROUTER[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_EXPERT_SCALE[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_TRAIN_BS[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_EVAL_BS[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_GLOBAL_PRE[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_GLOBAL_POST[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_MACRO_PRE[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_MID_PRE[@]}" ] \
  || [ "$cap_count" -ne "${#PROTOX_CAP_MICRO_PRE[@]}" ]; then
  echo "ProtoX capacity profile size mismatch." >&2
  exit 1
fi

if [ "$route_count" -ne "${#PROTOX_ROUTE_TOPK[@]}" ] \
  || [ "$route_count" -ne "${#PROTOX_ROUTE_TEMP_START[@]}" ] \
  || [ "$route_count" -ne "${#PROTOX_ROUTE_TEMP_END[@]}" ] \
  || [ "$route_count" -ne "${#PROTOX_ROUTE_STAGE_FLOOR[@]}" ] \
  || [ "$route_count" -ne "${#PROTOX_ROUTE_DELTA_SCALE[@]}" ]; then
  echo "ProtoX routing profile size mismatch." >&2
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
root = Path(os.environ.get("HYPEROPT_RESULTS_DIR", "run/artifacts/results")) / "fmoe_protox"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob(f"{dataset}_FeaturedMoE_ProtoX_*.json"):
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

emit("protox_stage_merge_mode", pick("protox_stage_merge_mode", "serial_weighted"))
emit("proto_num", int(pick("proto_num", 8)))
emit("proto_dim", int(pick("proto_dim", 48)))
emit("proto_top_k", int(pick("proto_top_k", 2)))
emit("proto_temperature_start", float(pick("proto_temperature_start", 1.1)))
emit("proto_temperature_end", float(pick("proto_temperature_end", 0.9)))
emit("proto_pooling", pick("proto_pooling", "query"))
emit("stage_weight_floor", float(pick("stage_weight_floor", 0.05)))
emit("stage_delta_scale", float(pick("stage_delta_scale", 1.5)))
emit("protox_global_pre_layers", int(pick("protox_global_pre_layers", 0)))
emit("protox_global_post_layers", int(pick("protox_global_post_layers", 0)))
emit("protox_macro_pre_layers", int(pick("protox_macro_pre_layers", 0)))
emit("protox_mid_pre_layers", int(pick("protox_mid_pre_layers", 0)))
emit("protox_micro_pre_layers", int(pick("protox_micro_pre_layers", 0)))
emit("embedding_size", int(pick("embedding_size", 128)))
emit("num_heads", int(pick("num_heads", 8)))
emit("d_feat_emb", int(pick("d_feat_emb", 16)))
emit("d_expert_hidden", int(pick("d_expert_hidden", 160)))
emit("d_router_hidden", int(pick("d_router_hidden", 80)))
emit("expert_scale", int(pick("expert_scale", 3)))
emit("train_batch_size", int(pick("train_batch_size", 4096)))
emit("eval_batch_size", int(pick("eval_batch_size", 8192)))
PY
}

run_protox_search() {
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
  log_file="$(run_make_log_path fmoe_protox hparam "$dataset" FeaturedMoE_ProtoX "$gpu" "$phase")"

  local -a cmd=(
    "$PY_BIN" hyperopt_tune.py
    --config-name config
    --max-evals "$max_evals"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --seed "$seed"
    --run-group fmoe_protox
    --run-axis hparam
    --run-phase "$phase"
    "model=featured_moe_protox_tune"
    "+search_space_type_overrides={weight_decay:choice,hidden_dropout_prob:choice,balance_loss_lambda:choice,proto_usage_lambda:choice,proto_entropy_lambda:choice,stage_weight_floor:choice,stage_delta_scale:choice,proto_top_k:choice,proto_temperature_start:choice,proto_temperature_end:choice,protox_stage_merge_mode:choice}"
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
    "protox_stage_merge_mode=serial_weighted"
    "++search.protox_stage_merge_mode=[serial_weighted]"
    "proto_num=8"
    "++search.proto_num=[8]"
    "proto_dim=48"
    "++search.proto_dim=[48]"
    "proto_top_k=2"
    "++search.proto_top_k=[2]"
    "proto_temperature_start=1.1"
    "++search.proto_temperature_start=[1.1]"
    "proto_temperature_end=0.9"
    "++search.proto_temperature_end=[0.9]"
    "proto_pooling=query"
    "++search.proto_pooling=[query]"
    "proto_router_use_hidden=true"
    "++search.proto_router_use_hidden=[true]"
    "proto_router_use_feature=true"
    "++search.proto_router_use_feature=[true]"
    "embedding_size=128"
    "++search.embedding_size=[128]"
    "num_heads=8"
    "++search.num_heads=[8]"
    "d_feat_emb=16"
    "++search.d_feat_emb=[16]"
    "d_expert_hidden=160"
    "++search.d_expert_hidden=[160]"
    "d_router_hidden=80"
    "++search.d_router_hidden=[80]"
    "expert_scale=3"
    "++search.expert_scale=[3]"
    "protox_global_pre_layers=0"
    "++search.protox_global_pre_layers=[0]"
    "protox_global_post_layers=0"
    "++search.protox_global_post_layers=[0]"
    "protox_macro_pre_layers=0"
    "++search.protox_macro_pre_layers=[0]"
    "protox_mid_pre_layers=0"
    "++search.protox_mid_pre_layers=[0]"
    "protox_micro_pre_layers=0"
    "++search.protox_micro_pre_layers=[0]"
    "stage_weight_floor=0.05"
    "++search.stage_weight_floor=[0.05]"
    "stage_delta_scale=1.5"
    "++search.stage_delta_scale=[1.5]"
    "moe_top_k=0"
    "++search.moe_top_k=[0]"
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
    "proto_usage_lambda=0.0"
    "proto_entropy_lambda=0.0"
    "++search.learning_rate=[${LR_RANGE}]"
    "++search.weight_decay=[${WD_VALUES}]"
    "++search.hidden_dropout_prob=[${DROPOUT_VALUES}]"
    "++search.balance_loss_lambda=[${BALANCE_VALUES}]"
    "++search.proto_usage_lambda=[${PROTO_USAGE_VALUES}]"
    "++search.proto_entropy_lambda=[${PROTO_ENTROPY_VALUES}]"
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
    --track fmoe_protox \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_ProtoX" \
    --exp-name "fmoe_protox_first_pass" \
    --exp-desc "ProtoX first pass: prototype-first architecture search + transfer." \
    --exp-focus "protox_stage_merge_mode,proto_num,proto_dim,proto_top_k,proto_temperature_start,proto_temperature_end,proto_pooling,stage_weight_floor,stage_delta_scale,protox_global_post_layers,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda,proto_usage_lambda,proto_entropy_lambda" \
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
    --track fmoe_protox \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_ProtoX" \
    --exp-name "fmoe_protox_first_pass" \
    --exp-desc "ProtoX first pass: prototype-first architecture search + transfer." \
    --exp-focus "protox_stage_merge_mode,proto_num,proto_dim,proto_top_k,proto_temperature_start,proto_temperature_end,proto_pooling,stage_weight_floor,stage_delta_scale,protox_global_post_layers,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda,proto_usage_lambda,proto_entropy_lambda" \
    --cmd "$cmd_str" \
    --log-file "$log_file" \
    --status "$status" \
    --exit-code "$rc"

  return "$rc"
}

ml1_best_parent=""
if dataset_has "movielens1m"; then
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  total_pattern_count=$(( cap_count * route_count ))
  echo "[PROTOX] ml1_total_jobs=${total_jobs} pattern_count=${total_pattern_count} (capacity=${cap_count} x routing=${route_count})"

  pids=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        job_idx=$(( gidx * COMBOS_PER_GPU + slot ))
        cap_idx=$(( job_idx % cap_count ))
        route_idx=$(( (job_idx / cap_count) % route_count ))
        repeat_idx=$(( job_idx / total_pattern_count + 1 ))

        cap_tag="${PROTOX_CAP_TAGS[$cap_idx]}"
        emb="${PROTOX_CAP_EMBEDDING[$cap_idx]}"
        dfeat="${PROTOX_CAP_D_FEAT[$cap_idx]}"
        dexp="${PROTOX_CAP_D_EXPERT[$cap_idx]}"
        drouter="${PROTOX_CAP_D_ROUTER[$cap_idx]}"
        escale="${PROTOX_CAP_EXPERT_SCALE[$cap_idx]}"
        train_bs="${PROTOX_CAP_TRAIN_BS[$cap_idx]}"
        eval_bs="${PROTOX_CAP_EVAL_BS[$cap_idx]}"
        gpre="${PROTOX_CAP_GLOBAL_PRE[$cap_idx]}"
        gpost="${PROTOX_CAP_GLOBAL_POST[$cap_idx]}"
        mpre="${PROTOX_CAP_MACRO_PRE[$cap_idx]}"
        midpre="${PROTOX_CAP_MID_PRE[$cap_idx]}"
        micpre="${PROTOX_CAP_MICRO_PRE[$cap_idx]}"

        route_tag="${PROTOX_ROUTE_TAGS[$route_idx]}"
        topk="${PROTOX_ROUTE_TOPK[$route_idx]}"
        tstart="${PROTOX_ROUTE_TEMP_START[$route_idx]}"
        tend="${PROTOX_ROUTE_TEMP_END[$route_idx]}"
        sfloor="${PROTOX_ROUTE_STAGE_FLOOR[$route_idx]}"
        dscale="${PROTOX_ROUTE_DELTA_SCALE[$route_idx]}"

        phase="PROTOX_ML1_G${gpu}_C$((slot + 1))_${cap_tag}_${route_tag}_R${repeat_idx}"
        seed=$((SEED_BASE + 100 + job_idx))

        echo "[PROTOX][ML1] gpu=${gpu} slot=$((slot + 1)) job=${job_idx}/${total_jobs} cap=${cap_tag}(emb=${emb},dfeat=${dfeat},dexp=${dexp},drouter=${drouter},bs=${train_bs}/${eval_bs},post=${gpost}) route=${route_tag}(topk=${topk},temp=${tstart}->${tend},floor=${sfloor},delta=${dscale})"

        combo_overrides=(
          "protox_stage_merge_mode=serial_weighted"
          "++search.protox_stage_merge_mode=[serial_weighted]"
          "proto_top_k=${topk}"
          "++search.proto_top_k=[${topk}]"
          "proto_temperature_start=${tstart}"
          "++search.proto_temperature_start=[${tstart}]"
          "proto_temperature_end=${tend}"
          "++search.proto_temperature_end=[${tend}]"
          "stage_weight_floor=${sfloor}"
          "++search.stage_weight_floor=[${sfloor}]"
          "stage_delta_scale=${dscale}"
          "++search.stage_delta_scale=[${dscale}]"
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
          "protox_global_pre_layers=${gpre}"
          "++search.protox_global_pre_layers=[${gpre}]"
          "protox_global_post_layers=${gpost}"
          "++search.protox_global_post_layers=[${gpost}]"
          "protox_macro_pre_layers=${mpre}"
          "++search.protox_macro_pre_layers=[${mpre}]"
          "protox_mid_pre_layers=${midpre}"
          "++search.protox_mid_pre_layers=[${midpre}]"
          "protox_micro_pre_layers=${micpre}"
          "++search.protox_micro_pre_layers=[${micpre}]"
          "train_batch_size=${train_bs}"
          "++search.train_batch_size=[${train_bs}]"
          "eval_batch_size=${eval_bs}"
          "++search.eval_batch_size=[${eval_bs}]"
        )
        run_protox_search movielens1m "$gpu" "$phase" "${ML1_SEARCH_EVALS}" "$seed" "" "${combo_overrides[@]}"
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
    echo "[PROTOX] ML1 first-pass phase failed." >&2
    exit 1
  fi

  if [ "$DRY_RUN" != "true" ]; then
    mapfile -t ml1_best < <(phase_top_results movielens1m PROTOX_ML1_ 1)
    if [ "${#ml1_best[@]}" -gt 0 ]; then
      ml1_best_parent="${ml1_best[0]}"
    fi
  fi
fi

if dataset_has "retail_rocket"; then
  rr_overrides=()
  if [ -n "$ml1_best_parent" ] && [ "$DRY_RUN" != "true" ]; then
    mapfile -t rr_overrides < <(extract_fixed_overrides_from_result "$ml1_best_parent")
    run_protox_search retail_rocket "${GPUS[0]}" PROTOX_RR_TRANSFER "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 31))" "$ml1_best_parent" "${rr_overrides[@]}"
  else
    rr_overrides=(
      "protox_stage_merge_mode=serial_weighted"
      "++search.protox_stage_merge_mode=[serial_weighted]"
      "proto_top_k=2"
      "++search.proto_top_k=[2]"
      "proto_temperature_start=1.1"
      "++search.proto_temperature_start=[1.1]"
      "proto_temperature_end=0.9"
      "++search.proto_temperature_end=[0.9]"
      "stage_weight_floor=0.05"
      "++search.stage_weight_floor=[0.05]"
      "stage_delta_scale=1.5"
      "++search.stage_delta_scale=[1.5]"
      "protox_global_post_layers=2"
      "++search.protox_global_post_layers=[2]"
      "embedding_size=160"
      "++search.embedding_size=[160]"
      "d_feat_emb=24"
      "++search.d_feat_emb=[24]"
      "d_expert_hidden=192"
      "++search.d_expert_hidden=[192]"
      "d_router_hidden=96"
      "++search.d_router_hidden=[96]"
      "train_batch_size=4096"
      "++search.train_batch_size=[4096]"
      "eval_batch_size=8192"
      "++search.eval_batch_size=[8192]"
    )
    run_protox_search retail_rocket "${GPUS[0]}" PROTOX_RR_TRANSFER "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 31))" "" "${rr_overrides[@]}"
  fi
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_model_report \
    fmoe_protox \
    FeaturedMoE_ProtoX \
    "$(run_experiments_dir)/models/FeaturedMoE_ProtoX"
  run_update_track_report fmoe_protox
fi
