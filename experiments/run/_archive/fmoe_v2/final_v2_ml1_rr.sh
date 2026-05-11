#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m,retail_rocket"
GPU_LIST="0,1,2,3"
SEED_BASE="620"
LAYOUT_ID="7"
EXECUTION="serial"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
COMBOS_PER_GPU="3"

# Default: broader final search per arm
ML1_R1_CTRL_EVALS="30"
ML1_R1_SPEC_EVALS="30"
ML1_B0_SPEC_EVALS="30"
RR_TRANSFER_EVALS="30"

# Evidence-backed ML1 architecture profiles (from top P2DB runs).
# tag, emb, d_feat, d_expert, d_router, train_bs, eval_bs
ARCH_PROFILE_TAGS=("A128_B8192" "A512_B3072" "A160_B6144" "A192_B8192")
ARCH_PROFILE_EMBEDDING=("128" "128" "160" "160")
ARCH_PROFILE_D_FEAT=("16" "16" "16" "16")
ARCH_PROFILE_D_EXPERT=("128" "512" "160" "192")
ARCH_PROFILE_D_ROUTER=("64" "64" "80" "80")
ARCH_PROFILE_TRAIN_BS=("8192" "3072" "6144" "8192")
ARCH_PROFILE_EVAL_BS=("16384" "6144" "12288" "16384")

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1,2,3]
          [--seed-base 620] [--layout-id 7] [--execution serial|parallel]
          [--combos-per-gpu 3]
          [--max-evals 30]
          [--tune-epochs 100] [--tune-patience 10]
          [--ml1-r1-ctrl-evals 30] [--ml1-r1-spec-evals 30]
          [--ml1-b0-spec-evals 30] [--rr-transfer-evals 30]
          [--log-wandb|--no-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus|--group-a-gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --execution) EXECUTION="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals)
      ML1_R1_CTRL_EVALS="$2"
      ML1_R1_SPEC_EVALS="$2"
      ML1_B0_SPEC_EVALS="$2"
      RR_TRANSFER_EVALS="$2"
      shift 2
      ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --ml1-r1-ctrl-evals) ML1_R1_CTRL_EVALS="$2"; shift 2 ;;
    --ml1-r1-spec-evals) ML1_R1_SPEC_EVALS="$2"; shift 2 ;;
    --ml1-b0-spec-evals) ML1_B0_SPEC_EVALS="$2"; shift 2 ;;
    --rr-transfer-evals) RR_TRANSFER_EVALS="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

case "${EXECUTION,,}" in
  serial|parallel) ;;
  *) echo "--execution must be serial|parallel" >&2; exit 1 ;;
esac
EXECUTION="${EXECUTION,,}"
if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DS_ARR
[ "${#DS_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

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
root = Path(os.environ.get("HYPEROPT_RESULTS_DIR", "run/artifacts/results")) / "fmoe_v2"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob(f"{dataset}_FeaturedMoE_v2_*.json"):
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

extract_overrides_from_result() {
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

router_impl = str(pick("router_impl", "learned"))
router_map = pick("router_impl_by_stage", {})
if not isinstance(router_map, dict):
    router_map = {}

aux_enable = bool(pick("fmoe_v2_feature_spec_aux_enable", False))
aux_lambda = float(pick("fmoe_v2_feature_spec_aux_lambda", 0.0))
aux_stages = pick("fmoe_v2_feature_spec_stages", ["mid", "micro"])
if isinstance(aux_stages, str):
    txt = aux_stages.strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
    aux_stages = [tok.strip() for tok in txt.split(",") if tok.strip()]
if not isinstance(aux_stages, list) or len(aux_stages) == 0:
    aux_stages = ["mid", "micro"]
stage_str = "[" + ",".join(str(x) for x in aux_stages) + "]"
min_tokens = float(pick("fmoe_v2_feature_spec_min_tokens", 16))

if router_map:
    map_str = "{" + ",".join(f"{k}:{router_map[k]}" for k in sorted(router_map.keys())) + "}"
else:
    map_str = "{}"

print(f"router_impl={router_impl}")
print(f"++search.router_impl=[{router_impl}]")
print(f"++router_impl_by_stage={map_str}")
print(f"++search.router_impl_by_stage=[{map_str}]")
print(f"fmoe_v2_feature_spec_aux_enable={'true' if aux_enable else 'false'}")
print(f"++search.fmoe_v2_feature_spec_aux_enable=[{'true' if aux_enable else 'false'}]")
print(f"fmoe_v2_feature_spec_aux_lambda={aux_lambda}")
print(f"++search.fmoe_v2_feature_spec_aux_lambda=[{aux_lambda}]")
print(f"fmoe_v2_feature_spec_stages={stage_str}")
print(f"fmoe_v2_feature_spec_min_tokens={min_tokens}")
print(f"embedding_size={int(pick('embedding_size', 128))}")
print(f"++search.embedding_size=[{int(pick('embedding_size', 128))}]")
print(f"num_heads={int(pick('num_heads', 8))}")
print(f"++search.num_heads=[{int(pick('num_heads', 8))}]")
print(f"d_feat_emb={int(pick('d_feat_emb', 16))}")
print(f"++search.d_feat_emb=[{int(pick('d_feat_emb', 16))}]")
print(f"d_expert_hidden={int(pick('d_expert_hidden', 128))}")
print(f"++search.d_expert_hidden=[{int(pick('d_expert_hidden', 128))}]")
print(f"d_router_hidden={int(pick('d_router_hidden', 64))}")
print(f"++search.d_router_hidden=[{int(pick('d_router_hidden', 64))}]")
print(f"expert_scale={int(pick('expert_scale', 3))}")
print(f"++search.expert_scale=[{int(pick('expert_scale', 3))}]")
PY
}

run_v2_tune() {
  local dataset="$1"
  local gpu="$2"
  local phase="$3"
  local max_evals="$4"
  local seed="$5"
  local parent_result="$6"
  shift 6
  local -a overrides=("$@")

  read -r train_bs eval_bs <<< "$(batch_sizes "$dataset")"

  local -a cmd=(
    bash "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "$dataset"
    --layout-id "$LAYOUT_ID"
    --execution "$EXECUTION"
    --schedule-preset off
    --gpu "$gpu"
    --max-evals "$max_evals"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --seed "$seed"
    --phase "$phase"
    --search-profile narrow_ml1
    --train-batch-size "$train_bs"
    --eval-batch-size "$eval_bs"
    --exp-name "fmoe_v2_final"
    --exp-desc "FMoEv2 final track: rule-control/spec-aux + transfer."
    --exp-focus "embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,train_batch_size,router_impl,router_impl_by_stage,fmoe_v2_feature_spec_aux_enable,fmoe_v2_feature_spec_aux_lambda,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda"
  )

  if [ -n "$parent_result" ]; then
    cmd+=(--parent-result "$parent_result")
  fi
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  else
    cmd+=(--no-wandb)
  fi
  local ov
  for ov in "${overrides[@]}"; do
    cmd+=(--override "$ov")
  done
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  echo "[FINAL_V2] dataset=${dataset} phase=${phase} gpu=${gpu} evals=${max_evals}"
  "${cmd[@]}"
}

if dataset_has "movielens1m"; then
  profile_count="${#ARCH_PROFILE_TAGS[@]}"
  if [ "$profile_count" -le 0 ]; then
    echo "ARCH profile list cannot be empty." >&2
    exit 1
  fi
  if [ "$profile_count" -ne "${#ARCH_PROFILE_EMBEDDING[@]}" ] \
    || [ "$profile_count" -ne "${#ARCH_PROFILE_D_FEAT[@]}" ] \
    || [ "$profile_count" -ne "${#ARCH_PROFILE_D_EXPERT[@]}" ] \
    || [ "$profile_count" -ne "${#ARCH_PROFILE_D_ROUTER[@]}" ] \
    || [ "$profile_count" -ne "${#ARCH_PROFILE_TRAIN_BS[@]}" ] \
    || [ "$profile_count" -ne "${#ARCH_PROFILE_EVAL_BS[@]}" ]; then
    echo "ARCH profile spec size mismatch." >&2
    exit 1
  fi

  r1_ctrl_overrides=(
    "router_impl=learned"
    "++search.router_impl=[learned]"
    "++router_impl_by_stage={mid:rule_soft,micro:rule_soft}"
    "++search.router_impl_by_stage=[{mid:rule_soft,micro:rule_soft}]"
    "fmoe_v2_feature_spec_aux_enable=false"
    "++search.fmoe_v2_feature_spec_aux_enable=[false]"
    "fmoe_v2_feature_spec_aux_lambda=0.0"
    "++search.fmoe_v2_feature_spec_aux_lambda=[0.0]"
  )
  r1_spec_overrides=(
    "router_impl=learned"
    "++search.router_impl=[learned]"
    "++router_impl_by_stage={mid:rule_soft,micro:rule_soft}"
    "++search.router_impl_by_stage=[{mid:rule_soft,micro:rule_soft}]"
    "fmoe_v2_feature_spec_aux_enable=true"
    "++search.fmoe_v2_feature_spec_aux_enable=[true]"
    "fmoe_v2_feature_spec_aux_lambda=0.0"
    "++search.fmoe_v2_feature_spec_aux_lambda=[0.0,1e-4,3e-4,1e-3]"
    "fmoe_v2_feature_spec_stages=[mid,micro]"
    "fmoe_v2_feature_spec_min_tokens=16"
  )
  b0_spec_overrides=(
    "router_impl=learned"
    "++search.router_impl=[learned]"
    "++router_impl_by_stage={}"
    "++search.router_impl_by_stage=[{}]"
    "fmoe_v2_feature_spec_aux_enable=true"
    "++search.fmoe_v2_feature_spec_aux_enable=[true]"
    "fmoe_v2_feature_spec_aux_lambda=0.0"
    "++search.fmoe_v2_feature_spec_aux_lambda=[0.0,1e-4,3e-4,1e-3]"
    "fmoe_v2_feature_spec_stages=[mid,micro]"
    "fmoe_v2_feature_spec_min_tokens=16"
  )

  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "[FINAL_V2] ml1_total_jobs=${total_jobs} arm_count=3 arch_profiles=${profile_count}"

  pids=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        job_idx=$(( gidx * COMBOS_PER_GPU + slot ))
        arm=$(( job_idx % 3 ))
        profile_idx=$(( (job_idx / 3) % profile_count ))
        repeat_idx=$(( (job_idx / (3 * profile_count)) + 1 ))
        profile_tag="${ARCH_PROFILE_TAGS[$profile_idx]}"
        emb="${ARCH_PROFILE_EMBEDDING[$profile_idx]}"
        dfeat="${ARCH_PROFILE_D_FEAT[$profile_idx]}"
        dexp="${ARCH_PROFILE_D_EXPERT[$profile_idx]}"
        drouter="${ARCH_PROFILE_D_ROUTER[$profile_idx]}"
        train_bs="${ARCH_PROFILE_TRAIN_BS[$profile_idx]}"
        eval_bs="${ARCH_PROFILE_EVAL_BS[$profile_idx]}"
        arch_overrides=(
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
          "expert_scale=3"
          "++search.expert_scale=[3]"
          "train_batch_size=${train_bs}"
          "++search.train_batch_size=[${train_bs}]"
          "eval_batch_size=${eval_bs}"
          "++search.eval_batch_size=[${eval_bs}]"
        )
        case "$arm" in
          0)
            arm_tag="R1_CTRL"
            max_evals="$ML1_R1_CTRL_EVALS"
            overrides=("${r1_ctrl_overrides[@]}")
            ;;
          1)
            arm_tag="R1_SPEC"
            max_evals="$ML1_R1_SPEC_EVALS"
            overrides=("${r1_spec_overrides[@]}")
            ;;
          *)
            arm_tag="B0_SPEC"
            max_evals="$ML1_B0_SPEC_EVALS"
            overrides=("${b0_spec_overrides[@]}")
            ;;
        esac
        phase="FINALV2_ML1_G${gpu}_C$((slot + 1))_${arm_tag}_${profile_tag}_R${repeat_idx}"
        seed=$((SEED_BASE + 100 + job_idx))
        combo_overrides=("${overrides[@]}" "${arch_overrides[@]}")
        run_v2_tune movielens1m "$gpu" "$phase" "$max_evals" "$seed" "" "${combo_overrides[@]}"
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
    echo "[FINAL_V2] ML1 phase failed." >&2
    exit 1
  fi
fi

if ! dataset_has "retail_rocket"; then
  echo "[FINAL_V2] retail_rocket not requested; stopping after ML1." >&2
  exit 0
fi

declare -a transfer_parents=()
if [ "$DRY_RUN" != "true" ]; then
  mapfile -t transfer_parents < <(phase_top_results movielens1m FINALV2_ML1_ 2)
fi

if [ "${#transfer_parents[@]}" -eq 0 ]; then
  echo "[FINAL_V2] parent result not found; using fallback transfer templates."
  transfer_overrides_1=(
    "router_impl=learned"
    "++search.router_impl=[learned]"
    "++router_impl_by_stage={mid:rule_soft,micro:rule_soft}"
    "++search.router_impl_by_stage=[{mid:rule_soft,micro:rule_soft}]"
    "fmoe_v2_feature_spec_aux_enable=true"
    "++search.fmoe_v2_feature_spec_aux_enable=[true]"
    "fmoe_v2_feature_spec_aux_lambda=3e-4"
    "++search.fmoe_v2_feature_spec_aux_lambda=[3e-4]"
    "fmoe_v2_feature_spec_stages=[mid,micro]"
    "fmoe_v2_feature_spec_min_tokens=16"
    "embedding_size=128"
    "++search.embedding_size=[128]"
    "num_heads=8"
    "++search.num_heads=[8]"
    "d_feat_emb=16"
    "++search.d_feat_emb=[16]"
    "d_expert_hidden=512"
    "++search.d_expert_hidden=[512]"
    "d_router_hidden=64"
    "++search.d_router_hidden=[64]"
    "expert_scale=3"
    "++search.expert_scale=[3]"
  )
  transfer_overrides_2=(
    "router_impl=learned"
    "++search.router_impl=[learned]"
    "++router_impl_by_stage={}"
    "++search.router_impl_by_stage=[{}]"
    "fmoe_v2_feature_spec_aux_enable=true"
    "++search.fmoe_v2_feature_spec_aux_enable=[true]"
    "fmoe_v2_feature_spec_aux_lambda=3e-4"
    "++search.fmoe_v2_feature_spec_aux_lambda=[3e-4]"
    "fmoe_v2_feature_spec_stages=[mid,micro]"
    "fmoe_v2_feature_spec_min_tokens=16"
    "embedding_size=160"
    "++search.embedding_size=[160]"
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
  )

  run_v2_tune retail_rocket "${GPUS[0]}" FINALV2_RR_T1 "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 21))" "" "${transfer_overrides_1[@]}"
  run_v2_tune retail_rocket "${GPUS[$((1 % ${#GPUS[@]}))]}" FINALV2_RR_T2 "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 22))" "" "${transfer_overrides_2[@]}"
else
  idx=0
  for parent in "${transfer_parents[@]}"; do
    idx=$((idx + 1))
    gpu="${GPUS[$(((idx - 1) % ${#GPUS[@]}))]}"
    phase="FINALV2_RR_T${idx}"
    mapfile -t ovs < <(extract_overrides_from_result "$parent")
    run_v2_tune retail_rocket "$gpu" "$phase" "${RR_TRANSFER_EVALS}" "$((SEED_BASE + 20 + idx))" "$parent" "${ovs[@]}"
  done
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_track_report fmoe_v2
fi
