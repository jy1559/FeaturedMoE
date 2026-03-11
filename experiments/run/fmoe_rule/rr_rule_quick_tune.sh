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
CATALOG_PROFILE="rr_rule8"
COMBO_CATALOG=""
COMBOS_PER_GPU=""
MAX_EVALS="8"
TUNE_EPOCHS="60"
TUNE_PATIENCE="8"
SEED_BASE="1400"
PHASE_PREFIX="RRRULE"
RULE_N_BINS="5"
RULE_FEATURES_PER_EXPERT="4"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--datasets retail_rocket] [--gpus 0,1,2,3]
          [--catalog-profile rr_rule8|rr_rule12] [--combo-catalog spec]
          [--combos-per-gpu N] [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--phase-prefix RRRULE] [--seed-base N]
          [--rule-n-bins N] [--rule-feature-per-expert N]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --catalog-profile) CATALOG_PROFILE="$2"; shift 2 ;;
    --combo-catalog) COMBO_CATALOG="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --rule-n-bins) RULE_N_BINS="$2"; shift 2 ;;
    --rule-feature-per-expert) RULE_FEATURES_PER_EXPERT="$2"; shift 2 ;;
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

if [ -n "$COMBOS_PER_GPU" ] && { ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; }; then
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

build_rr_rule8_combo_catalog() {
  python3 - <<'PY'
rows = [
    ("R1", 16, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L16F24"),
    ("R1", 16, "serial", 128, 16, 128,  64, 3, 4096, 8192, "L16BASE"),
    ("R1", 16, "serial", 160, 24, 192,  96, 3, 3072, 6144, "L16BIG"),
    ("R1", 15, "serial", 160, 16, 160,  80, 3, 4096, 8192, "L15MED"),
    ("R1", 18, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L18F24"),
    ("R1", 15, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L15F24"),
    ("R0", 16, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L16F24"),
    ("R0", 16, "serial", 128, 16, 128,  64, 3, 4096, 8192, "L16BASE"),
]
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_rr_rule12_combo_catalog() {
  python3 - <<'PY'
rows = [
    ("R1", 16, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L16F24"),
    ("R1", 16, "serial", 128, 16, 128,  64, 3, 4096, 8192, "L16BASE"),
    ("R1", 16, "serial", 160, 24, 192,  96, 3, 3072, 6144, "L16BIG"),
    ("R1", 16, "serial", 160, 16, 160,  80, 3, 4096, 8192, "L16MED"),
    ("R1", 15, "serial", 160, 16, 160,  80, 3, 4096, 8192, "L15MED"),
    ("R1", 15, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L15F24"),
    ("R1", 18, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L18F24"),
    ("R1", 18, "serial", 128, 16, 128,  64, 3, 4096, 8192, "L18BASE"),
    ("R0", 16, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L16F24"),
    ("R0", 16, "serial", 128, 16, 128,  64, 3, 4096, 8192, "L16BASE"),
    ("R0", 15, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L15F24"),
    ("R0", 18, "serial", 128, 24, 160,  64, 3, 4096, 8192, "L18F24"),
]
print(";".join(",".join(str(x) for x in row) for row in rows))
PY
}

build_combo_catalog() {
  case "$CATALOG_PROFILE" in
    rr_rule8) build_rr_rule8_combo_catalog ;;
    rr_rule12) build_rr_rule12_combo_catalog ;;
    *)
      echo "Unsupported --catalog-profile=${CATALOG_PROFILE}" >&2
      exit 1
      ;;
  esac
}

read_combo() {
  local combo_catalog="$1"
  local idx="$2"
  python3 - <<'PY' "$combo_catalog" "$idx"
import sys

raw = sys.argv[1]
idx = int(sys.argv[2])
rows = [x.strip() for x in raw.split(';') if x.strip()]
if not rows:
    raise SystemExit("empty combo catalog")
row = rows[idx % len(rows)]
parts = [x.strip() for x in row.split(',')]
if len(parts) != 11:
    raise SystemExit(f"invalid combo row: {row}")
print(" ".join(parts))
PY
}

combo_count() {
  local combo_catalog="$1"
  python3 - <<'PY' "$combo_catalog"
import sys
rows = [x.strip() for x in sys.argv[1].split(';') if x.strip()]
print(len(rows))
PY
}

compute_auto_combos_per_gpu() {
  local combo_n="$1"
  local gpu_n="$2"
  echo $(( (combo_n + gpu_n - 1) / gpu_n ))
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

lr_space_for_combo() {
  local arm="$1"
  local dexp="$2"
  local drouter="$3"
  local train_bs="$4"

  if [ "$arm" = "R0" ]; then
    if [ "$train_bs" -le 3072 ] || [ "$drouter" -ge 96 ] || [ "$dexp" -ge 192 ]; then
      echo "8e-4,1.5e-3,3e-3,5e-3,8e-3"
    else
      echo "1e-3,2e-3,3.5e-3,5e-3,8e-3,1e-2"
    fi
  else
    if [ "$train_bs" -le 3072 ] || [ "$drouter" -ge 96 ] || [ "$dexp" -ge 192 ]; then
      echo "3e-4,4.5e-4,6e-4,8e-4,1e-3"
    else
      echo "3e-4,4e-4,5e-4,6.5e-4,8e-4,1e-3,1.2e-3"
    fi
  fi
}

wd_space_for_combo() {
  local arm="$1"

  if [ "$arm" = "R0" ]; then
    echo "0.0,1e-5,2.5e-5,5e-5"
  else
    echo "0.0,2.5e-5,5e-5,1e-4,1.5e-4"
  fi
}

dropout_for_combo() {
  local arm="$1"
  if [ "$arm" = "R0" ]; then
    echo "0.12"
  else
    echo "0.10"
  fi
}

balance_for_combo() {
  local arm="$1"
  if [ "$arm" = "R0" ]; then
    echo "0.007"
  else
    echo "0.010"
  fi
}

catalog_profile_desc() {
  case "$CATALOG_PROFILE" in
    rr_rule8)
      echo "RetailRocket quick rule probe: ML1 hybrid-rule winner + RR v2 winning layouts/dims, mostly R1 with R0 sentinels."
      ;;
    rr_rule12)
      echo "RetailRocket broader rule probe: expanded R1/R0 layout-dim matrix centered on L16/L15/L18."
      ;;
  esac
}

run_one_combo() {
  local dataset="$1"
  local gpu="$2"
  local combo_idx="$3"

  local arm layout execution emb dfeat dexp drouter scale train_bs eval_bs tag
  read -r arm layout execution emb dfeat dexp drouter scale train_bs eval_bs tag <<< "$(read_combo "$COMBO_CATALOG" "$combo_idx")"

  local lr_space wd_space hidden_drop balance phase seed exp_name exp_desc exp_focus
  lr_space="$(lr_space_for_combo "$arm" "$dexp" "$drouter" "$train_bs")"
  wd_space="$(wd_space_for_combo "$arm")"
  hidden_drop="$(dropout_for_combo "$arm")"
  balance="$(balance_for_combo "$arm")"
  phase="${PHASE_PREFIX}_${arm}_G${gpu}_C$(printf '%02d' "$combo_idx")_${tag}"
  seed=$(( SEED_BASE + combo_idx ))
  exp_name="rr_rule_quick_${arm,,}_${tag,,}"
  exp_desc="$(catalog_profile_desc)"
  exp_focus="ablation,router_impl,router_impl_by_stage,rule_router.n_bins,rule_router.feature_per_expert,fmoe_stage_execution_mode,fmoe_v2_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,train_batch_size,eval_batch_size,hidden_dropout_prob,balance_loss_lambda,learning_rate,weight_decay"

  local cmd=(
    "${SCRIPT_DIR}/tune_hparam.sh"
    --dataset "$dataset"
    --gpu "$gpu"
    --seed "$seed"
    --phase "$phase"
    --max-evals "$MAX_EVALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --layout-id "$layout"
    --execution "$execution"
    --schedule off
    --ablation "$arm"
    --rule-n-bins "$RULE_N_BINS"
    --rule-feature-per-expert "$RULE_FEATURES_PER_EXPERT"
    --search-profile p1_shallow
    --lr-space "$lr_space"
    --wd-space "$wd_space"
    --hidden-dropout "$hidden_drop"
    --balance-loss-lambda "$balance"
    --train-batch-size "$train_bs"
    --eval-batch-size "$eval_bs"
    --embedding-size "$emb"
    --d-feat-emb "$dfeat"
    --d-expert-hidden "$dexp"
    --d-router-hidden "$drouter"
    --expert-scale "$scale"
    --exp-name "$exp_name"
    --exp-desc "$exp_desc"
    --exp-focus "$exp_focus"
  )
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi
  if [ "$LOG_WANDB" = "true" ]; then
    :
  fi

  echo "[RR_RULE] dataset=${dataset} gpu=${gpu} combo=${combo_idx} arm=${arm} layout=L${layout} tag=${tag} lr=[${lr_space}] wd=[${wd_space}]"
  "${cmd[@]}"
}

if [ -z "$COMBO_CATALOG" ]; then
  COMBO_CATALOG="$(build_combo_catalog)"
fi
TOTAL_COMBOS="$(combo_count "$COMBO_CATALOG")"

if [ -z "$COMBOS_PER_GPU" ]; then
  COMBOS_PER_GPU="$(compute_auto_combos_per_gpu "$TOTAL_COMBOS" "${#GPUS[@]}")"
fi

COVERED_COMBOS="$(compute_covered_combos "$TOTAL_COMBOS" "${#GPUS[@]}" "$COMBOS_PER_GPU")"

echo "[RR_RULE] profile=${CATALOG_PROFILE} total_combos=${TOTAL_COMBOS} covered=${COVERED_COMBOS} combos_per_gpu=${COMBOS_PER_GPU}"
echo "[RR_RULE] rationale: ML1 rule-hybrid best was R1(L7, lr~3e-4~1.2e-3, wd~0~1.4e-4) while RR v2 best was L16/L15/L18 with moderate dims and lr~3e-4~8e-4."

trap 'dispatch_terminate_all GPUS' INT TERM

for ds in "${DATASET_ARR[@]}"; do
  echo "=== [${ds}] RR quick rule probe start ==="

  for gidx in "${!GPUS[@]}"; do
    planned="$(planned_combo_count_for_gpu "$gidx" "$COVERED_COMBOS" "${#GPUS[@]}" "$COMBOS_PER_GPU")"
    echo "[RR_RULE] gpu=${GPUS[$gidx]} planned_jobs=${planned}"
  done

  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      for ((slot=0; slot<COMBOS_PER_GPU; slot++)); do
        combo_idx=$(( slot * ${#GPUS[@]} + gidx ))
        if [ "$combo_idx" -lt "$COVERED_COMBOS" ]; then
          run_one_combo "$ds" "$gpu" "$combo_idx"
        fi
      done
    ) &
    dispatch_set_pid "$gpu" "$!"
    sleep 1
  done

  dispatch_wait_all
  echo "=== [${ds}] RR quick rule probe done ==="
done
