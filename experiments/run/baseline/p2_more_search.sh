#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="KuaiRecLargeStrictPosV2_0.2"
CONFIG_NAME="tune_kuai_strict_small"
GPU_LIST="0,1,2,3"
MAX_EVALS="5"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="5200"
PHASE="P2_MORE_SEARCH"
LOG_WANDB="false"
SPECIAL_LOGGING="false"
SKIP_COMPLETED_LOGS="${SKIP_COMPLETED_LOGS:-false}"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN=""

# Requested fixed order
MODELS=("bsarec" "fenrec" "patt" "fame" "sigma")
COMBOS=("C01" "C02" "C03" "C04" "C05" "C06" "C07" "C08" "C09" "C10" "C11" "C12" "C13" "C14" "C15" "C16")

declare -ag WORKER_PIDS=()

auto_trim() {
  echo "$1" | xargs
}

usage() {
  cat <<USAGE
Usage: $0 [--gpus 0,1,2,3] [--max-evals 5] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 5200] [--phase P2_MORE_SEARCH]
          [--models bsarec,fenrec,patt,fame,sigma]
          [--skip-completed-logs | --no-skip-completed-logs]
          [--dry-run] [--log-wandb] [--special-logging]

Policy:
- Dataset fixed: ${DATASET}
- Model order fixed: BSARec -> FENRec -> PAtt -> FAME -> SIGMA
- 16 combos per model, round-robin to 4 GPUs
- Skip criteria uses logs only (not result json) when --skip-completed-logs is enabled
- A task is considered completed only if log tail contains:
  [RUN_STATUS] END status=normal
USAGE
}

parse_models_filter() {
  local raw="$1"
  local token m found
  local -a parsed=()

  IFS=',' read -r -a tokens <<< "$raw"
  for token in "${tokens[@]}"; do
    m="$(auto_trim "$token")"
    [ -z "$m" ] && continue
    found=0
    for allowed in "${MODELS[@]}"; do
      if [ "$m" = "$allowed" ]; then
        found=1
        break
      fi
    done
    if [ "$found" -ne 1 ]; then
      echo "Unknown model in --models: ${m}" >&2
      return 1
    fi
    parsed+=("$m")
  done

  if [ "${#parsed[@]}" -eq 0 ]; then
    echo "--models produced empty model list" >&2
    return 1
  fi

  MODELS=("${parsed[@]}")
}

make_log_path() {
  local model="$1"
  local combo="$2"
  local phase="$3"
  local gpu="$4"
  local dir ts rid
  ts="$(date +%Y%m%d_%H%M%S)"
  rid="${RANDOM}"
  dir="${RUN_DIR}/artifacts/logs/baseline/${DATASET}/${phase}"
  mkdir -p "$dir"
  echo "${dir}/${model}_${combo}_g${gpu}_${ts}_${rid}.log"
}

# Skip uses log evidence only. No json-based skip.
find_completed_log_for_task() {
  local model="$1"
  local combo="$2"
  local gpu="$3"
  local log_dir file

  log_dir="${RUN_DIR}/artifacts/logs/baseline/${DATASET}/${PHASE}"
  if [ ! -d "$log_dir" ]; then
    return 1
  fi

  shopt -s nullglob
  for file in "$log_dir"/${model}_${combo}_g${gpu}_*.log; do
    if tail -n 80 "$file" | grep -Eq '\[RUN_STATUS\] END status=normal'; then
      echo "$file"
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob

  return 1
}

lr_range_for_combo() {
  local model="$1"
  local combo="$2"

  case "$model" in
    bsarec|fenrec|patt|fame|sigma) ;;
    *)
      echo "Unsupported model for lr range: model=${model}" >&2
      return 1
      ;;
  esac

  case "$combo" in
    C01|C02|C03|C04|C05|C06|C07|C08|C09|C10|C11|C12|C13|C14|C15|C16) ;;
    *)
      echo "Unsupported combo for lr range: combo=${combo}" >&2
      return 1
      ;;
  esac

  # Hyperopt tunes learning_rate only, globally unified range.
  echo "2e-4|6e-3"
}

combo_fixed_reg_for_combo() {
  local model="$1"
  local combo="$2"

  case "${model}:${combo}" in
    bsarec:C01|bsarec:C02|bsarec:C03|bsarec:C04) echo "0.10|1e-4" ;;
    bsarec:C05|bsarec:C06|bsarec:C07|bsarec:C08) echo "0.15|1e-4" ;;
    bsarec:C09|bsarec:C10|bsarec:C11|bsarec:C12) echo "0.20|5e-5" ;;
    bsarec:C13|bsarec:C14|bsarec:C15|bsarec:C16) echo "0.10|5e-5" ;;

    fenrec:C01|fenrec:C02|fenrec:C03|fenrec:C04) echo "0.10|1e-4" ;;
    fenrec:C05|fenrec:C06|fenrec:C07|fenrec:C08) echo "0.12|1e-4" ;;
    fenrec:C09|fenrec:C10|fenrec:C11|fenrec:C12) echo "0.15|5e-5" ;;
    fenrec:C13|fenrec:C14|fenrec:C15|fenrec:C16) echo "0.10|5e-5" ;;

    patt:C01|patt:C02|patt:C03|patt:C04) echo "0.10|1e-4" ;;
    patt:C05|patt:C06|patt:C07|patt:C08) echo "0.12|1e-4" ;;
    patt:C09|patt:C10|patt:C11|patt:C12) echo "0.15|5e-5" ;;
    patt:C13|patt:C14|patt:C15|patt:C16) echo "0.10|5e-5" ;;

    fame:C01|fame:C02|fame:C03|fame:C04) echo "0.10|1e-4" ;;
    fame:C05|fame:C06|fame:C07|fame:C08) echo "0.15|1e-4" ;;
    fame:C09|fame:C10|fame:C11|fame:C12) echo "0.18|5e-5" ;;
    fame:C13|fame:C14|fame:C15|fame:C16) echo "0.12|5e-5" ;;

    sigma:C01|sigma:C02|sigma:C03|sigma:C04) echo "0.18|1e-4" ;;
    sigma:C05|sigma:C06|sigma:C07|sigma:C08) echo "0.20|1e-4" ;;
    sigma:C09|sigma:C10|sigma:C11|sigma:C12) echo "0.25|5e-5" ;;
    sigma:C13|sigma:C14|sigma:C15|sigma:C16) echo "0.18|5e-5" ;;
    *)
      echo "Unsupported combo fixed-reg target: model=${model}, combo=${combo}" >&2
      return 1
      ;;
  esac
}

combo_struct_overrides() {
  local model="$1"
  local combo="$2"

  case "${model}:${combo}" in
    bsarec:C01) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C02) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C03) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C04) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C05) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C06) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C07) printf '%s\n' "hidden_size=224" "num_layers=2" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C08) printf '%s\n' "hidden_size=224" "num_layers=3" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C09) printf '%s\n' "hidden_size=256" "num_layers=2" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C10) printf '%s\n' "hidden_size=256" "num_layers=3" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C11) printf '%s\n' "hidden_size=160" "num_layers=4" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C12) printf '%s\n' "hidden_size=192" "num_layers=4" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C13) printf '%s\n' "hidden_size=224" "num_layers=4" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C14) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C15) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C16) printf '%s\n' "hidden_size=192" "num_layers=1" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=20" ;;

    fenrec:C01) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "cl_weight=0.10" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C02) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "cl_weight=0.12" "cl_temperature=0.12" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C03) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=4" "cl_weight=0.15" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C04) printf '%s\n' "hidden_size=224" "num_layers=2" "num_heads=8" "cl_weight=0.18" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C05) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "cl_weight=0.20" "cl_temperature=0.15" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C06) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "cl_weight=0.22" "cl_temperature=0.12" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C07) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "cl_weight=0.24" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C08) printf '%s\n' "hidden_size=224" "num_layers=3" "num_heads=8" "cl_weight=0.26" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C09) printf '%s\n' "hidden_size=256" "num_layers=2" "num_heads=8" "cl_weight=0.18" "cl_temperature=0.15" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C10) printf '%s\n' "hidden_size=256" "num_layers=3" "num_heads=8" "cl_weight=0.22" "cl_temperature=0.12" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C11) printf '%s\n' "hidden_size=160" "num_layers=4" "num_heads=4" "cl_weight=0.28" "cl_temperature=0.08" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C12) printf '%s\n' "hidden_size=192" "num_layers=4" "num_heads=8" "cl_weight=0.30" "cl_temperature=0.08" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C13) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "cl_weight=0.08" "cl_temperature=0.20" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C14) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=4" "cl_weight=0.10" "cl_temperature=0.18" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C15) printf '%s\n' "hidden_size=192" "num_layers=1" "num_heads=8" "cl_weight=0.12" "cl_temperature=0.15" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C16) printf '%s\n' "hidden_size=224" "num_layers=1" "num_heads=8" "cl_weight=0.14" "cl_temperature=0.15" "++MAX_ITEM_LIST_LENGTH=20" ;;

    patt:C01) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "diversity_gamma=0.08" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C02) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "diversity_gamma=0.10" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C03) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=4" "diversity_gamma=0.12" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C04) printf '%s\n' "hidden_size=224" "num_layers=2" "num_heads=8" "diversity_gamma=0.14" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C05) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "diversity_gamma=0.16" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C06) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "diversity_gamma=0.18" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C07) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "diversity_gamma=0.20" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C08) printf '%s\n' "hidden_size=224" "num_layers=3" "num_heads=8" "diversity_gamma=0.22" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C09) printf '%s\n' "hidden_size=256" "num_layers=2" "num_heads=8" "diversity_gamma=0.16" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C10) printf '%s\n' "hidden_size=256" "num_layers=3" "num_heads=8" "diversity_gamma=0.18" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C11) printf '%s\n' "hidden_size=160" "num_layers=4" "num_heads=4" "diversity_gamma=0.24" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C12) printf '%s\n' "hidden_size=192" "num_layers=4" "num_heads=8" "diversity_gamma=0.26" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C13) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "diversity_gamma=0.06" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C14) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=4" "diversity_gamma=0.08" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C15) printf '%s\n' "hidden_size=192" "num_layers=1" "num_heads=8" "diversity_gamma=0.10" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C16) printf '%s\n' "hidden_size=224" "num_layers=1" "num_heads=8" "diversity_gamma=0.12" "++MAX_ITEM_LIST_LENGTH=20" ;;

    fame:C01) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C02) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C03) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=4" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C04) printf '%s\n' "hidden_size=224" "num_layers=2" "num_heads=8" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C05) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C06) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C07) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C08) printf '%s\n' "hidden_size=224" "num_layers=3" "num_heads=8" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C09) printf '%s\n' "hidden_size=256" "num_layers=2" "num_heads=8" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C10) printf '%s\n' "hidden_size=256" "num_layers=3" "num_heads=8" "num_experts=10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C11) printf '%s\n' "hidden_size=160" "num_layers=4" "num_heads=4" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C12) printf '%s\n' "hidden_size=192" "num_layers=4" "num_heads=8" "num_experts=10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C13) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C14) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C15) printf '%s\n' "hidden_size=192" "num_layers=1" "num_heads=8" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C16) printf '%s\n' "hidden_size=224" "num_layers=1" "num_heads=8" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;

    sigma:C01) printf '%s\n' "hidden_size=128" "num_layers=2" "state_size=16" "conv_kernel=4" "remaining_ratio=0.60" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C02) printf '%s\n' "hidden_size=160" "num_layers=2" "state_size=16" "conv_kernel=4" "remaining_ratio=0.65" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C03) printf '%s\n' "hidden_size=192" "num_layers=2" "state_size=16" "conv_kernel=8" "remaining_ratio=0.70" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C04) printf '%s\n' "hidden_size=224" "num_layers=2" "state_size=32" "conv_kernel=8" "remaining_ratio=0.70" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C05) printf '%s\n' "hidden_size=128" "num_layers=3" "state_size=16" "conv_kernel=4" "remaining_ratio=0.70" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C06) printf '%s\n' "hidden_size=160" "num_layers=3" "state_size=16" "conv_kernel=4" "remaining_ratio=0.75" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C07) printf '%s\n' "hidden_size=192" "num_layers=3" "state_size=32" "conv_kernel=8" "remaining_ratio=0.75" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C08) printf '%s\n' "hidden_size=224" "num_layers=3" "state_size=32" "conv_kernel=8" "remaining_ratio=0.80" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C09) printf '%s\n' "hidden_size=256" "num_layers=2" "state_size=32" "conv_kernel=8" "remaining_ratio=0.75" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C10) printf '%s\n' "hidden_size=256" "num_layers=3" "state_size=32" "conv_kernel=8" "remaining_ratio=0.80" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C11) printf '%s\n' "hidden_size=160" "num_layers=4" "state_size=16" "conv_kernel=4" "remaining_ratio=0.80" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C12) printf '%s\n' "hidden_size=192" "num_layers=4" "state_size=32" "conv_kernel=8" "remaining_ratio=0.85" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C13) printf '%s\n' "hidden_size=128" "num_layers=1" "state_size=16" "conv_kernel=4" "remaining_ratio=0.55" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C14) printf '%s\n' "hidden_size=160" "num_layers=1" "state_size=16" "conv_kernel=4" "remaining_ratio=0.60" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C15) printf '%s\n' "hidden_size=192" "num_layers=1" "state_size=32" "conv_kernel=8" "remaining_ratio=0.65" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C16) printf '%s\n' "hidden_size=224" "num_layers=1" "state_size=32" "conv_kernel=8" "remaining_ratio=0.65" "++MAX_ITEM_LIST_LENGTH=20" ;;

    *)
      echo "Unsupported combo structure target: model=${model}, combo=${combo}" >&2
      return 1
      ;;
  esac
}

run_combo_job() {
  local model="$1"
  local combo="$2"
  local gpu="$3"
  local evals="$4"
  local seed="$5"
  local lr_lo="$6"
  local lr_hi="$7"
  local dropout="$8"
  local wd="$9"
  local run_phase="${10}"

  local log_path cmd_str
  local -a cmd=()
  local -a struct_overrides=()
  local -a search_singleton_overrides=()
  local hidden_size_val=""
  local num_heads_val=""
  local num_layers_val=""
  local key=""
  local value=""
  local kv

  log_path="$(make_log_path "$model" "$combo" "$PHASE" "$gpu")"
  mapfile -t struct_overrides < <(combo_struct_overrides "$model" "$combo")

  for kv in "${struct_overrides[@]}"; do
    case "$kv" in
      hidden_size=*) hidden_size_val="${kv#hidden_size=}" ;;
      num_heads=*) num_heads_val="${kv#num_heads=}" ;;
      num_layers=*) num_layers_val="${kv#num_layers=}" ;;
    esac

    if [[ "$kv" == *=* && "$kv" != ++* ]]; then
      key="${kv%%=*}"
      value="${kv#*=}"
      search_singleton_overrides+=("++search.${key}=[${value}]")
    fi
  done

  if [ -n "$hidden_size_val" ] && [ -n "$num_heads_val" ]; then
    if [ $((hidden_size_val % num_heads_val)) -ne 0 ]; then
      echo "[CONFIG_ERROR] hidden_size(${hidden_size_val}) must be divisible by num_heads(${num_heads_val}) model=${model} combo=${combo}" >&2
      return 1
    fi
  fi

  cmd=(
    "$PYTHON_BIN" hyperopt_tune.py
    --config-name "$CONFIG_NAME"
    --seed "$seed"
    --max-evals "$evals"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --run-group baseline
    --run-axis hparam
    --run-phase "$run_phase"
    "model=${model}"
    "dataset=${DATASET}"
    "gpu_id=${gpu}"
    "++seed=${seed}"
    "+combo_id=${combo}"
    "+trial_epoch_log=false"
    "show_progress=false"
    "log_wandb=${LOG_WANDB}"
    "++search.learning_rate=[${lr_lo},${lr_hi}]"
    "++search.weight_decay=[${wd}]"
    "++search.dropout_ratio=[${dropout}]"
    "++search_space_type_overrides.learning_rate=loguniform"
    "++search_space_type_overrides.weight_decay=choice"
    "++search_space_type_overrides.dropout_ratio=choice"
    "++weight_decay=${wd}"
    "++dropout_ratio=${dropout}"
  )

  # FENRec matmul-shape safety.
  if [ "$model" = "fenrec" ] && [ -n "$hidden_size_val" ]; then
    cmd+=("embedding_size=${hidden_size_val}")
  fi

  if [ "${#search_singleton_overrides[@]}" -gt 0 ]; then
    cmd+=("${search_singleton_overrides[@]}")
  fi

  cmd+=("${struct_overrides[@]}")

  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$SPECIAL_LOGGING" = "true" ]; then
    cmd+=("++special_logging=true")
  fi

  echo "[JOB] model=${model} combo=${combo} gpu=${gpu} evals=${evals} lr=[${lr_lo},${lr_hi}] drop=${dropout} wd=${wd}"
  echo "[LOG] ${log_path}"

  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] $(run_cmd_str "${cmd[@]}")"
    return 0
  fi

  cmd_str="$(run_cmd_str "${cmd[@]}")"
  local run_id
  run_id="$(run_tracker_start \
    --track baseline \
    --axis hparam \
    --phase "$run_phase" \
    --dataset "$DATASET" \
    --model "$model" \
    --cmd "$cmd_str" \
    --log-file "$log_path")"

  set +e
  LOG_FILE="$log_path" PYTHONUNBUFFERED=1 "${cmd[@]}"
  local rc=$?
  set -e

  local status
  if [ "$rc" -eq 0 ]; then
    status="success"
  else
    status="fail"
  fi

  run_tracker_end \
    --run-id "$run_id" \
    --track baseline \
    --axis hparam \
    --phase "$run_phase" \
    --dataset "$DATASET" \
    --model "$model" \
    --cmd "$cmd_str" \
    --log-file "$log_path" \
    --status "$status" \
    --exit-code "$rc"

  if ! "$PYTHON_BIN" "${SCRIPT_DIR}/update_phase_summary.py" --dataset "$DATASET" --phase "$PHASE" >/dev/null 2>&1; then
    echo "[WARN] summary update failed: dataset=${DATASET} phase=${PHASE}" >&2
  fi

  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi

  return 0
}

run_gpu_queue() {
  local gpu="$1"
  local queue_payload="$2"
  local line model combo evals seed lr_lo lr_hi drop wd run_phase

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    IFS='|' read -r model combo evals seed lr_lo lr_hi drop wd run_phase <<< "$line"
    if ! run_combo_job "$model" "$combo" "$gpu" "$evals" "$seed" "$lr_lo" "$lr_hi" "$drop" "$wd" "$run_phase"; then
      echo "[GPU ${gpu}] failed task: ${line}" >&2
      return 1
    fi
  done <<< "$queue_payload"

  return 0
}

on_interrupt() {
  echo "[INTERRUPT] stopping gpu workers..." >&2
  for pid in "${WORKER_PIDS[@]:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --models) parse_models_filter "$2"; shift 2 ;;
    --skip-completed-logs) SKIP_COMPLETED_LOGS="true"; shift ;;
    --no-skip-completed-logs) SKIP_COMPLETED_LOGS="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --special-logging) SPECIAL_LOGGING="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi
PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"

if ! "$PYTHON_BIN" - <<'PY'
import importlib
must_have = ["torch", "recbole", "lightgbm", "hyperopt", "yaml"]
missing = []
for name in must_have:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("missing_python_packages=" + ",".join(missing))

import recbole_patch  # noqa: F401
from recbole.utils import utils as rbu
for model_name in ["BSARec", "FENRec", "PAtt", "FAME", "SIGMA", "FeaturedMoE_N3"]:
  try:
    _ = rbu.get_model(model_name)
  except Exception as e:
    raise SystemExit(f"missing_or_broken_model={model_name}: {e}")
print("[ENV_CHECK] python and core packages OK")
PY
then
  echo "[ERROR] runtime environment check failed for PYTHON_BIN=${PYTHON_BIN}" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -lt 4 ]; then
  echo "--gpus requires at least 4 GPU ids" >&2
  exit 1
fi
ACTIVE_GPUS=("$(auto_trim "${GPUS[0]}")" "$(auto_trim "${GPUS[1]}")" "$(auto_trim "${GPUS[2]}")" "$(auto_trim "${GPUS[3]}")")

QUEUES=("" "" "" "")

echo "[PLAN] dataset=${DATASET}"
echo "[PLAN] python_bin=${PYTHON_BIN}"
echo "[PLAN] model order=${MODELS[*]}"
echo "[PLAN] combos=${COMBOS[*]}"
echo "[PLAN] max_evals=${MAX_EVALS} tune_epochs=${TUNE_EPOCHS} tune_patience=${TUNE_PATIENCE}"
echo "[PLAN] round-robin mapping: C01->${ACTIVE_GPUS[0]}, C02->${ACTIVE_GPUS[1]}, C03->${ACTIVE_GPUS[2]}, C04->${ACTIVE_GPUS[3]}, ..."
echo "[PLAN] skip_completed_logs=${SKIP_COMPLETED_LOGS}"

for model_idx in "${!MODELS[@]}"; do
  model="${MODELS[$model_idx]}"

  for combo_idx in "${!COMBOS[@]}"; do
    combo="${COMBOS[$combo_idx]}"
    slot_idx=$((combo_idx % 4))
    gpu="${ACTIVE_GPUS[$slot_idx]}"

    IFS='|' read -r drop wd <<< "$(combo_fixed_reg_for_combo "$model" "$combo")"
    IFS='|' read -r lr_lo lr_hi <<< "$(lr_range_for_combo "$model" "$combo")"
    seed=$((SEED_BASE + model_idx * 1000 + combo_idx))
    run_phase="${PHASE}_M$(printf '%02d' "$model_idx")_${model^^}_${combo}"

    if [ "$SKIP_COMPLETED_LOGS" = "true" ]; then
      if done_log="$(find_completed_log_for_task "$model" "$combo" "$gpu")"; then
        echo "[SKIP_LOG_DONE] model=${model} combo=${combo} gpu=${gpu} log=${done_log##*/}"
        continue
      fi
    fi

    line="${model}|${combo}|${MAX_EVALS}|${seed}|${lr_lo}|${lr_hi}|${drop}|${wd}|${run_phase}"
    QUEUES[$slot_idx]="${QUEUES[$slot_idx]}${line}"$'\n'
    echo "[MAP] model=${model} combo=${combo} -> gpu=${gpu}"
  done
done

for i in 0 1 2 3; do
  cnt="$(printf '%s' "${QUEUES[$i]}" | awk 'NF>0{n+=1} END{print n+0}')"
  echo "[QUEUE] gpu=${ACTIVE_GPUS[$i]} tasks=${cnt}"
done

trap on_interrupt INT TERM
WORKER_PIDS=()
for i in 0 1 2 3; do
  gpu="${ACTIVE_GPUS[$i]}"
  payload="${QUEUES[$i]}"
  (
    run_gpu_queue "$gpu" "$payload"
  ) &
  WORKER_PIDS+=("$!")
done

failed=0
for p in "${WORKER_PIDS[@]}"; do
  if ! wait "$p"; then
    failed=1
  fi
done

trap - INT TERM
if [ "$failed" -ne 0 ]; then
  echo "[ERROR] one or more gpu queues failed" >&2
  exit 1
fi

if [ "$DRY_RUN" = "false" ]; then
  if ! "$PYTHON_BIN" "${SCRIPT_DIR}/update_phase_summary.py" --dataset "$DATASET" --phase "$PHASE" >/dev/null 2>&1; then
    echo "[WARN] summary update failed: dataset=${DATASET} phase=${PHASE}" >&2
  fi
fi

echo "[ALL DONE] completed all queued tasks for dataset: ${DATASET}"
