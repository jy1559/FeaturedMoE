#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="movielens1m"
B_GPUS="0,1"
C_GPUS="2,3"
MAX_EVALS="6"
TUNE_EPOCHS="60"
TUNE_PATIENCE="6"
B_PHASE_PREFIX="V3B"
C_PHASE_PREFIX="V3C"
BASE_ROUTER_DESIGN="flat_hidden_group_clone12"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m]
          [--b-gpus 0,1] [--c-gpus 2,3]
          [--max-evals 6] [--tune-epochs 60] [--tune-patience 6]
          [--b-phase-prefix V3B] [--c-phase-prefix V3C]
          [--base-router-design flat_hidden_group_clone12]
          [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --b-gpus) B_GPUS="$2"; shift 2 ;;
    --c-gpus) C_GPUS="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --b-phase-prefix) B_PHASE_PREFIX="$2"; shift 2 ;;
    --c-phase-prefix) C_PHASE_PREFIX="$2"; shift 2 ;;
    --base-router-design) BASE_ROUTER_DESIGN="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

PIDS=()
on_interrupt() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

cmd_b=(
  bash "${SCRIPT_DIR}/phase_b_ml1_router_structure.sh"
  --datasets "${DATASET}"
  --gpus "${B_GPUS}"
  --catalog-profile "quick6"
  --combos-per-gpu "3"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --phase-prefix "${B_PHASE_PREFIX}"
)
cmd_c=(
  bash "${SCRIPT_DIR}/phase_c_ml1_distill_modes.sh"
  --datasets "${DATASET}"
  --gpus "${C_GPUS}"
  --catalog-profile "quick6"
  --combos-per-gpu "3"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --phase-prefix "${C_PHASE_PREFIX}"
  --base-router-design "${BASE_ROUTER_DESIGN}"
)

if [ "${LOG_WANDB}" = "true" ]; then
  cmd_b+=(--log-wandb)
  cmd_c+=(--log-wandb)
fi
if [ "${DRY_RUN}" = "true" ]; then
  cmd_b+=(--dry-run)
  cmd_c+=(--dry-run)
fi

printf '[BC Quick][B] '
printf '%q ' "${cmd_b[@]}"
printf '\n'
"${cmd_b[@]}" &
PIDS+=("$!")

printf '[BC Quick][C] '
printf '%q ' "${cmd_c[@]}"
printf '\n'
"${cmd_c[@]}" &
PIDS+=("$!")

wait "${PIDS[@]}"
