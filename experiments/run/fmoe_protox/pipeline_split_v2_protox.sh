#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GROUP_A_GPUS="0,1,2,3"
GROUP_B_GPUS="4,5,6,7"
DATASETS="movielens1m,retail_rocket"
SWAP_GROUPS="false"
DRY_RUN="${DRY_RUN:-false}"
LOG_WANDB="false"
COMBOS_PER_GPU="3"
MAX_EVALS="40"

PROTOX_LR_RANGE=""
PROTOX_WD_VALUES=""
PROTOX_DROPOUT_VALUES=""
PROTOX_BALANCE_VALUES=""
PROTOX_USAGE_VALUES=""
PROTOX_ENTROPY_VALUES=""

usage() {
  cat <<USAGE
Usage: $0 [--group-a-gpus 0,1,2,3] [--group-b-gpus 4,5,6,7]
          [--datasets movielens1m,retail_rocket] [--swap-groups]
          [--combos-per-gpu 3] [--max-evals 40]
          [--protox-lr-range 1e-4,2e-2] [--protox-wd-values 0.0,1e-6,1e-5,1e-4,1e-3]
          [--protox-dropout-values 0.05,0.1,0.15,0.2,0.25]
          [--protox-balance-values 0.003,0.01,0.03,0.05,0.1]
          [--protox-usage-values 0.0,1e-4,3e-4,1e-3,3e-3]
          [--protox-entropy-values 0.0,1e-4,3e-4,1e-3,3e-3]
          [--log-wandb|--no-wandb] [--dry-run]

Default role:
  group-A -> fmoe_v2_final
  group-B -> fmoe_protox_first
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --group-a-gpus) GROUP_A_GPUS="$2"; shift 2 ;;
    --group-b-gpus) GROUP_B_GPUS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --swap-groups) SWAP_GROUPS="true"; shift ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --protox-lr-range) PROTOX_LR_RANGE="$2"; shift 2 ;;
    --protox-wd-values) PROTOX_WD_VALUES="$2"; shift 2 ;;
    --protox-dropout-values) PROTOX_DROPOUT_VALUES="$2"; shift 2 ;;
    --protox-balance-values) PROTOX_BALANCE_VALUES="$2"; shift 2 ;;
    --protox-usage-values) PROTOX_USAGE_VALUES="$2"; shift 2 ;;
    --protox-entropy-values) PROTOX_ENTROPY_VALUES="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be a positive integer" >&2
  exit 1
fi

ROLE_A="fmoe_v2_final"
ROLE_B="fmoe_protox_first"
if [ "$SWAP_GROUPS" = "true" ]; then
  ROLE_A="fmoe_protox_first"
  ROLE_B="fmoe_v2_final"
fi

run_role() {
  local role="$1"
  local gpus="$2"
  local -a cmd

  case "$role" in
    fmoe_v2_final)
      cmd=(
        bash "${RUN_DIR}/fmoe_v2/final_v2_ml1_rr.sh"
        --datasets "$DATASETS"
        --gpus "$gpus"
        --combos-per-gpu "$COMBOS_PER_GPU"
        --ml1-r1-ctrl-evals "$MAX_EVALS"
        --ml1-r1-spec-evals "$MAX_EVALS"
        --ml1-b0-spec-evals "$MAX_EVALS"
        --rr-transfer-evals "$MAX_EVALS"
      )
      ;;
    fmoe_protox_first)
      cmd=(
        bash "${SCRIPT_DIR}/run_first_pass_protox.sh"
        --datasets "$DATASETS"
        --gpus "$gpus"
        --combos-per-gpu "$COMBOS_PER_GPU"
        --ml1-search-evals "$MAX_EVALS"
        --rr-transfer-evals "$MAX_EVALS"
      )
      if [ -n "$PROTOX_LR_RANGE" ]; then
        cmd+=(--lr-range "$PROTOX_LR_RANGE")
      fi
      if [ -n "$PROTOX_WD_VALUES" ]; then
        cmd+=(--wd-values "$PROTOX_WD_VALUES")
      fi
      if [ -n "$PROTOX_DROPOUT_VALUES" ]; then
        cmd+=(--dropout-values "$PROTOX_DROPOUT_VALUES")
      fi
      if [ -n "$PROTOX_BALANCE_VALUES" ]; then
        cmd+=(--balance-values "$PROTOX_BALANCE_VALUES")
      fi
      if [ -n "$PROTOX_USAGE_VALUES" ]; then
        cmd+=(--proto-usage-values "$PROTOX_USAGE_VALUES")
      fi
      if [ -n "$PROTOX_ENTROPY_VALUES" ]; then
        cmd+=(--proto-entropy-values "$PROTOX_ENTROPY_VALUES")
      fi
      ;;
    *)
      echo "Unsupported role: $role" >&2
      return 1
      ;;
  esac

  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  else
    cmd+=(--no-wandb)
  fi
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  echo "[SPLIT] role=${role} gpus=${gpus}"
  "${cmd[@]}"
}

echo "[SPLIT] group-A=${GROUP_A_GPUS} role=${ROLE_A}"
echo "[SPLIT] group-B=${GROUP_B_GPUS} role=${ROLE_B}"
echo "[SPLIT] datasets=${DATASETS} swap=${SWAP_GROUPS}"
echo "[SPLIT] combos_per_gpu=${COMBOS_PER_GPU} max_evals=${MAX_EVALS}"
if [ -n "$PROTOX_LR_RANGE" ] || [ -n "$PROTOX_WD_VALUES" ] || [ -n "$PROTOX_DROPOUT_VALUES" ] \
  || [ -n "$PROTOX_BALANCE_VALUES" ] || [ -n "$PROTOX_USAGE_VALUES" ] || [ -n "$PROTOX_ENTROPY_VALUES" ]; then
  echo "[SPLIT] protox_ranges lr=${PROTOX_LR_RANGE:-default} wd=${PROTOX_WD_VALUES:-default} dropout=${PROTOX_DROPOUT_VALUES:-default} balance=${PROTOX_BALANCE_VALUES:-default} usage=${PROTOX_USAGE_VALUES:-default} entropy=${PROTOX_ENTROPY_VALUES:-default}"
fi

run_role "$ROLE_A" "$GROUP_A_GPUS" &
PID_A=$!
run_role "$ROLE_B" "$GROUP_B_GPUS" &
PID_B=$!

RC=0
if ! wait "$PID_A"; then
  RC=1
fi
if ! wait "$PID_B"; then
  RC=1
fi
exit "$RC"
