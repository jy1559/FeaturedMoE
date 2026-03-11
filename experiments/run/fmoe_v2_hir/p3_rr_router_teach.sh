#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="retail_rocket"
GPU_LIST="0,1,2,3"
COMBOS_PER_GPU="3"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="15"
SEED_BASE="1180"
PHASE_PREFIX="P3RRT"
SEED_PROFILE="l16_f24"
CATALOG_PROFILE="teach12"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
OOM_RETRY_MIN_TRAIN_BS="1024"
PARENT_PHASE_PREFIXES="P2RGI,P2RGI2,P1RGI2,P1RFI"
DROP_SPACE="0.10"
BAL_SPACE="0.001,0.002,0.003"
SCHEDULE_PRESET="off"

EXP_FOCUS="fmoe_v2_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_top_k,router_distill_enable,router_distill_lambda,router_distill_temperature,router_distill_until,fmoe_v2_feature_spec_aux_lambda,fmoe_v2_feature_spec_stages,learning_rate,weight_decay,balance_loss_lambda"

usage() {
  cat <<USAGE
Usage: $0 [--datasets retail_rocket] [--gpus 0,1,2,3]
          [--combos-per-gpu 3] [--max-evals 10] [--tune-epochs 100] [--tune-patience 15]
          [--phase-prefix P3RRT] [--seed-profile l16_f24|l16_base|l15_base]
          [--catalog-profile teach12|semantics12]
          [--parent-phase-prefixes csv] [--oom-retry-min-train-bs N]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --seed-profile) SEED_PROFILE="$2"; shift 2 ;;
    --catalog-profile) CATALOG_PROFILE="$2"; shift 2 ;;
    --parent-phase-prefixes) PARENT_PHASE_PREFIXES="$2"; shift 2 ;;
    --oom-retry-min-train-bs) OOM_RETRY_MIN_TRAIN_BS="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$MAX_EVALS" -le 0 ]; then
  echo "--max-evals must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TUNE_EPOCHS" =~ ^[0-9]+$ ]] || [ "$TUNE_EPOCHS" -le 0 ]; then
  echo "--tune-epochs must be a positive integer" >&2
  exit 1
fi
if ! [[ "$TUNE_PATIENCE" =~ ^[0-9]+$ ]] || [ "$TUNE_PATIENCE" -le 0 ]; then
  echo "--tune-patience must be a positive integer" >&2
  exit 1
fi

seed_profile_spec() {
  local seed_profile="$1"
  case "$seed_profile" in
    l16_f24) echo "L16F24 16 serial 128 24 160 64 3 4096 8192" ;;
    l16_base) echo "L16BASE 16 serial 128 16 128 64 3 4096 8192" ;;
    l15_base) echo "L15BASE 15 serial 128 16 128 64 3 4096 8192" ;;
    *)
      echo "Unsupported --seed-profile=${seed_profile}" >&2
      exit 1
      ;;
  esac
}

read -r SEED_TAG LAYOUT_ID EXECUTION EMB DFEAT DEXP DROUTER EXPERT_SCALE TRAIN_BS EVAL_BS <<< "$(seed_profile_spec "$SEED_PROFILE")"

generate_combo_rows() {
  python3 - "$CATALOG_PROFILE" <<'PY'
import sys

profile = sys.argv[1]
if profile == "teach12":
    rows = [
        ("C00", "1", "false", "0.0",  "1.5", "0.2",  "3e-4", "[mid]",        "2.8e-4,3.4e-4,4.0e-4,4.8e-4,5.6e-4", "2.5e-5,5e-5,7.5e-5",        "K1", "D0", "SMID", "U20"),
        ("C01", "1", "false", "0.0",  "1.5", "0.2",  "7e-4", "[mid]",        "2.8e-4,3.4e-4,4.0e-4,4.8e-4,5.6e-4", "2.5e-5,5e-5,7.5e-5",        "K1", "D0", "SMID", "U20"),
        ("C02", "2", "false", "0.0",  "1.5", "0.2",  "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K2", "D0", "SMID", "U20"),
        ("C03", "1", "true",  "2e-3", "1.5", "0.2",  "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D2", "SMID", "U20"),
        ("C04", "1", "true",  "5e-3", "1.5", "0.2",  "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D5", "SMID", "U20"),
        ("C05", "1", "true",  "5e-3", "1.5", "0.35", "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D5", "SMID", "U35"),
        ("C06", "1", "true",  "2e-3", "1.5", "0.2",  "3e-4", "[mid,micro]",  "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D2", "SMM",  "U20"),
        ("C07", "1", "true",  "5e-3", "1.5", "0.2",  "3e-4", "[mid,micro]",  "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D5", "SMM",  "U20"),
        ("C08", "1", "true",  "5e-3", "1.5", "0.2",  "7e-4", "[mid,micro]",  "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K1", "D5", "SMM",  "U20"),
        ("C09", "2", "true",  "2e-3", "1.5", "0.2",  "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K2", "D2", "SMID", "U20"),
        ("C10", "2", "true",  "5e-3", "1.5", "0.2",  "3e-4", "[mid]",        "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K2", "D5", "SMID", "U20"),
        ("C11", "2", "true",  "5e-3", "1.5", "0.2",  "3e-4", "[mid,micro]",  "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5",   "K2", "D5", "SMM",  "U20"),
    ]
elif profile == "semantics12":
    rows = [
        ("C00", "0", "false", "0.0", "1.5", "0.0", "1e-4", "[mid]",       "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.6e-4", "2.5e-5,5e-5,7.5e-5",      "K0", "D0", "SMID", "U00"),
        ("C01", "0", "false", "0.0", "1.5", "0.0", "3e-4", "[mid]",       "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.6e-4", "2.5e-5,5e-5,7.5e-5",      "K0", "D0", "SMID", "U00"),
        ("C02", "0", "false", "0.0", "1.5", "0.0", "3e-4", "[mid,micro]", "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.6e-4", "2.5e-5,5e-5,7.5e-5",      "K0", "D0", "SMM",  "U00"),
        ("C03", "0", "false", "0.0", "1.5", "0.0", "1e-4", "[mid,micro]", "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.6e-4", "2.5e-5,5e-5,7.5e-5",      "K0", "D0", "SMM",  "U00"),
        ("C04", "2", "false", "0.0", "1.5", "0.0", "1e-4", "[mid]",       "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5", "K2", "D0", "SMID", "U00"),
        ("C05", "2", "false", "0.0", "1.5", "0.0", "3e-4", "[mid]",       "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5", "K2", "D0", "SMID", "U00"),
        ("C06", "2", "false", "0.0", "1.5", "0.0", "3e-4", "[mid,micro]", "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5", "K2", "D0", "SMM",  "U00"),
        ("C07", "2", "false", "0.0", "1.5", "0.0", "1e-4", "[mid,micro]", "2.2e-4,2.8e-4,3.4e-4,4.0e-4,4.8e-4", "1e-5,2.5e-5,5e-5,7.5e-5", "K2", "D0", "SMM",  "U00"),
        ("C08", "1", "false", "0.0", "1.5", "0.0", "1e-4", "[mid]",       "2.6e-4,3.2e-4,3.8e-4,4.6e-4,5.4e-4", "2.5e-5,5e-5,7.5e-5",      "K1", "D0", "SMID", "U00"),
        ("C09", "1", "false", "0.0", "1.5", "0.0", "3e-4", "[mid]",       "2.6e-4,3.2e-4,3.8e-4,4.6e-4,5.4e-4", "2.5e-5,5e-5,7.5e-5",      "K1", "D0", "SMID", "U00"),
        ("C10", "1", "false", "0.0", "1.5", "0.0", "3e-4", "[mid,micro]", "2.6e-4,3.2e-4,3.8e-4,4.6e-4,5.4e-4", "2.5e-5,5e-5,7.5e-5",      "K1", "D0", "SMM",  "U00"),
        ("C11", "1", "false", "0.0", "1.5", "0.0", "1e-4", "[mid,micro]", "2.6e-4,3.2e-4,3.8e-4,4.6e-4,5.4e-4", "2.5e-5,5e-5,7.5e-5",      "K1", "D0", "SMM",  "U00"),
    ]
else:
    raise SystemExit(f"Unsupported --catalog-profile={profile}")

for row in rows:
    print("\t".join(row))
PY
}

read_combo() {
  local idx="$1"
  generate_combo_rows | sed -n "$((idx + 1))p"
}

combo_count() {
  generate_combo_rows | wc -l | tr -d ' '
}

compute_covered_combos() {
  local combo_n="$1"
  local gpu_n="$2"
  local combos_per_gpu="$3"
  local total_slots=$(( gpu_n * combos_per_gpu ))
  if [ "$total_slots" -gt "$combo_n" ]; then
    echo "$combo_n"
  else
    echo "$total_slots"
  fi
}

planned_combo_count_for_gpu() {
  local gpu_idx="$1"
  local covered_combo_n="$2"
  local gpu_n="$3"
  local combos_per_gpu="$4"
  local count=0
  local slot combo_idx

  for ((slot=0; slot<combos_per_gpu; slot++)); do
    combo_idx=$(( slot * gpu_n + gpu_idx ))
    if [ "$combo_idx" -lt "$covered_combo_n" ]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

latest_phase_log() {
  local dataset="$1"
  local phase="$2"
  local dataset_tag model_tag phase_bucket dir
  dataset_tag="$(run_dataset_tag "$dataset")"
  model_tag="$(run_model_tag "FeaturedMoE_v2_HiR")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  dir="$(run_log_dir fmoe_v2_hir)/hparam/${phase_bucket}/${dataset_tag}/${model_tag}"
  [ -d "$dir" ] || { echo ""; return 0; }
  ls -1t "$dir"/*"_hparam_${phase}.log" 2>/dev/null | head -n1
}

is_oom_log() {
  local log_file="$1"
  [ -n "$log_file" ] || return 1
  [ -f "$log_file" ] || return 1
  grep -qi "CUDA out of memory" "$log_file"
}

find_parent_result() {
  local dataset="$1"
  local layout_id="$2"
  local emb="$3"
  local dfeat="$4"
  local dexp="$5"
  local drouter="$6"
  local train_bs="$7"
  local phase_prefixes="$8"
  python3 - <<'PY' "$dataset" "$layout_id" "$emb" "$dfeat" "$dexp" "$drouter" "$train_bs" "$phase_prefixes" "$(run_results_dir fmoe_v2_hir)"
import json
import re
import sys
from pathlib import Path

dataset = sys.argv[1]
layout_id = int(sys.argv[2])
emb = int(sys.argv[3])
dfeat = int(sys.argv[4])
dexp = int(sys.argv[5])
drouter = int(sys.argv[6])
train_bs = int(sys.argv[7])
phase_prefixes = [x.strip().lower() for x in sys.argv[8].split(",") if x.strip()]
root = Path(sys.argv[9])

def best_for_prefix(prefix):
    best = None
    for path in root.glob(f"{dataset}_FeaturedMoE_v2_HiR_*.json"):
        name = path.name.lower()
        if prefix and prefix not in name:
            continue
        m = re.search(r"[Ll](\d+)_[Ee](\d+)_[Ff](\d+)_[Hh](\d+)_[Rr](\d+)_[Bb](\d+)", path.stem)
        if not m:
            continue
        row = tuple(map(int, m.groups()))
        if row != (layout_id, emb, dfeat, dexp, drouter, train_bs):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        trials = [t for t in data.get("trials", []) if isinstance(t.get("mrr@20"), (int, float))]
        if not trials:
            continue
        score = max(t["mrr@20"] for t in trials)
        if best is None or score > best[0]:
            best = (score, str(path))
    return best

for prefix in phase_prefixes:
    best = best_for_prefix(prefix)
    if best is not None:
        print(best[1])
        raise SystemExit(0)

best = best_for_prefix("")
print("" if best is None else best[1])
PY
}

COMBO_N="$(combo_count)"
[ "$COMBO_N" -le 0 ] && { echo "P3 combo catalog is empty"; exit 1; }
COVERED_COMBOS="$(compute_covered_combos "$COMBO_N" "${#GPUS[@]}" "$COMBOS_PER_GPU")"

catalog_profile_desc() {
  case "$1" in
    teach12)
      echo "compare top-k/distill/spec profiles under long-horizon tuning"
      ;;
    semantics12)
      echo "compare expert_top_k=0/1/2 under no-distill, light feature-spec, and long-horizon tuning"
      ;;
    *)
      echo "profile=${1}"
      ;;
  esac
}

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping RR router-teach P3 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

for ds in "${DATASET_ARR[@]}"; do
  parent_result="$(find_parent_result "$ds" "$LAYOUT_ID" "$EMB" "$DFEAT" "$DEXP" "$DROUTER" "$TRAIN_BS" "$PARENT_PHASE_PREFIXES")"
  if [ -n "$parent_result" ]; then
    echo "[${PHASE_PREFIX}] parent_result=${parent_result}"
  else
    echo "[${PHASE_PREFIX}] warning: no exact parent result found for seed=${SEED_PROFILE}" >&2
  fi

  echo "=== [${ds}] RR router-teach P3 combos=${COVERED_COMBOS}/${COMBO_N} seed=${SEED_PROFILE} profile=${CATALOG_PROFILE} gpus=${#GPUS[@]} cpg=${COMBOS_PER_GPU} ==="
  gpu_plan=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    planned_count="$(planned_combo_count_for_gpu "$gidx" "$COVERED_COMBOS" "${#GPUS[@]}" "$COMBOS_PER_GPU")"
    gpu_plan+=("G${gpu}=${planned_count}")
  done
  echo "[${PHASE_PREFIX}] gpu_combo_plan=$(IFS=,; echo "${gpu_plan[*]}")"

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for ((slot=0; slot<COMBOS_PER_GPU; slot++)); do
        combo_idx=$(( slot * ${#GPUS[@]} + gidx ))
        if [ "$combo_idx" -ge "$COVERED_COMBOS" ]; then
          continue
        fi

        combo_row="$(read_combo "$combo_idx")"
        [ -n "$combo_row" ] || { echo "[${PHASE_PREFIX}] missing combo row idx=${combo_idx}" >&2; exit 1; }

        IFS=$'\t' read -r combo_id expert_top_k distill_enable distill_lambda distill_temp distill_until spec_lambda spec_stages lr_space wd_space topk_tag distill_tag spec_tag until_tag <<< "$combo_row"
        seed=$(( SEED_BASE + combo_idx ))
        phase="${PHASE_PREFIX}_G${gpu}_${combo_id}_${SEED_TAG}_L${LAYOUT_ID}_E${EMB}_F${DFEAT}_H${DEXP}_R${DROUTER}_B${TRAIN_BS}_${topk_tag}_${distill_tag}_${spec_tag}_${until_tag}"
        exp_name="${PHASE_PREFIX}_${ds}_${SEED_PROFILE}_router_teach"
        exp_desc="${PHASE_PREFIX}: RR router-teach P3 around ${SEED_PROFILE}. Fixed seed L${LAYOUT_ID}/${EMB}/${DFEAT}/${DEXP}/${DROUTER}/bs${TRAIN_BS}; $(catalog_profile_desc "$CATALOG_PROFILE")."

        echo "[${PHASE_PREFIX}][${ds}] gpu=${gpu} combo=${combo_id} seed=${SEED_PROFILE} topk=${expert_top_k} distill=${distill_lambda}@${distill_until} spec=${spec_stages}:${spec_lambda}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "$ds"
          --layout-id "$LAYOUT_ID"
          --execution "$EXECUTION"
          --schedule-preset "$SCHEDULE_PRESET"
          --gpu "$gpu"
          --max-evals "$MAX_EVALS"
          --tune-epochs "$TUNE_EPOCHS"
          --tune-patience "$TUNE_PATIENCE"
          --seed "$seed"
          --phase "$phase"
          --search-profile p1_shallow
          --train-batch-size "$TRAIN_BS"
          --eval-batch-size "$EVAL_BS"
          --embedding-size "$EMB"
          --num-heads 8
          --d-feat-emb "$DFEAT"
          --d-expert-hidden "$DEXP"
          --d-router-hidden "$DROUTER"
          --expert-scale "$EXPERT_SCALE"
          --lr-space "$lr_space"
          --wd-space "$wd_space"
          --dropout-space "$DROP_SPACE"
          --balance-space "$BAL_SPACE"
          --exp-name "$exp_name"
          --exp-desc "$exp_desc"
          --exp-focus "$EXP_FOCUS"
          --override "hidden_dropout_prob=0.10"
          --override "group_top_k=0"
          --override "++search.group_top_k=[0]"
          --override "expert_top_k=${expert_top_k}"
          --override "++search.expert_top_k=[${expert_top_k}]"
          --override "fmoe_v2_feature_spec_aux_lambda=${spec_lambda}"
          --override "++search.fmoe_v2_feature_spec_aux_lambda=[${spec_lambda}]"
          --override "fmoe_v2_feature_spec_stages=${spec_stages}"
          --override "++search.fmoe_v2_feature_spec_stages=[${spec_stages}]"
          --override "router_distill_enable=${distill_enable}"
          --override "++search.router_distill_enable=[${distill_enable}]"
          --override "router_distill_lambda=${distill_lambda}"
          --override "++search.router_distill_lambda=[${distill_lambda}]"
          --override "router_distill_temperature=${distill_temp}"
          --override "++search.router_distill_temperature=[${distill_temp}]"
          --override "router_distill_until=${distill_until}"
          --override "++search.router_distill_until=[${distill_until}]"
        )
        if [ -n "$parent_result" ]; then
          cmd+=(--parent-result "$parent_result")
        fi
        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        else
          cmd+=(--no-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi

        set +e
        "${cmd[@]}"
        rc=$?
        set -e

        if [ "$rc" -ne 0 ]; then
          log_path="$(latest_phase_log "$ds" "$phase")"
          if is_oom_log "$log_path" && [ "$TRAIN_BS" -ge $((OOM_RETRY_MIN_TRAIN_BS * 2)) ]; then
            retry_train_bs=$(( TRAIN_BS / 2 ))
            retry_eval_bs=$(( EVAL_BS / 2 ))
            retry_phase="${phase}_RBS${retry_train_bs}"
            echo "[${PHASE_PREFIX}][OOM-RETRY] ${ds} combo=${combo_id} phase=${phase} -> ${retry_phase} bs ${TRAIN_BS}/${EVAL_BS} -> ${retry_train_bs}/${retry_eval_bs}"
            retry_cmd=(
              bash "${SCRIPT_DIR}/tune_hparam.sh"
              --dataset "$ds"
              --layout-id "$LAYOUT_ID"
              --execution "$EXECUTION"
              --schedule-preset "$SCHEDULE_PRESET"
              --gpu "$gpu"
              --max-evals "$MAX_EVALS"
              --tune-epochs "$TUNE_EPOCHS"
              --tune-patience "$TUNE_PATIENCE"
              --seed "$seed"
              --phase "$retry_phase"
              --search-profile p1_shallow
              --train-batch-size "$retry_train_bs"
              --eval-batch-size "$retry_eval_bs"
              --embedding-size "$EMB"
              --num-heads 8
              --d-feat-emb "$DFEAT"
              --d-expert-hidden "$DEXP"
              --d-router-hidden "$DROUTER"
              --expert-scale "$EXPERT_SCALE"
              --lr-space "$lr_space"
              --wd-space "$wd_space"
              --dropout-space "$DROP_SPACE"
              --balance-space "$BAL_SPACE"
              --exp-name "$exp_name"
              --exp-desc "${exp_desc} OOM-retry(train/eval=${retry_train_bs}/${retry_eval_bs})."
              --exp-focus "$EXP_FOCUS"
              --override "hidden_dropout_prob=0.10"
              --override "group_top_k=0"
              --override "++search.group_top_k=[0]"
              --override "expert_top_k=${expert_top_k}"
              --override "++search.expert_top_k=[${expert_top_k}]"
              --override "fmoe_v2_feature_spec_aux_lambda=${spec_lambda}"
              --override "++search.fmoe_v2_feature_spec_aux_lambda=[${spec_lambda}]"
              --override "fmoe_v2_feature_spec_stages=${spec_stages}"
              --override "++search.fmoe_v2_feature_spec_stages=[${spec_stages}]"
              --override "router_distill_enable=${distill_enable}"
              --override "++search.router_distill_enable=[${distill_enable}]"
              --override "router_distill_lambda=${distill_lambda}"
              --override "++search.router_distill_lambda=[${distill_lambda}]"
              --override "router_distill_temperature=${distill_temp}"
              --override "++search.router_distill_temperature=[${distill_temp}]"
              --override "router_distill_until=${distill_until}"
              --override "++search.router_distill_until=[${distill_until}]"
            )
            if [ -n "$parent_result" ]; then
              retry_cmd+=(--parent-result "$parent_result")
            fi
            if [ "$LOG_WANDB" = "true" ]; then
              retry_cmd+=(--log-wandb)
            else
              retry_cmd+=(--no-wandb)
            fi
            if [ "$DRY_RUN" = "true" ]; then
              retry_cmd+=(--dry-run)
            fi
            "${retry_cmd[@]}"
          else
            exit "$rc"
          fi
        fi
      done
    ) &
    WORKER_PIDS+=("$!")
  done

  FAIL=0
  for p in "${WORKER_PIDS[@]}"; do
    if ! wait "$p"; then
      FAIL=1
    fi
  done

  if [ "$FAIL" -ne 0 ]; then
    echo "[ERROR] RR router-teach P3 failed for dataset=${ds}" >&2
    exit 1
  fi

  echo "=== [${ds}] RR router-teach P3 done ==="
done

trap - INT TERM

if [ "$INTERRUPTED" = "true" ]; then
  exit 130
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_track_report fmoe_v2_hir
fi
