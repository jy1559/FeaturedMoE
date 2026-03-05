#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNE_SCRIPT="${SCRIPT_DIR}/tune_hparam_hir.sh"

DATASET="movielens1m"
GPU_ID="7"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
SEARCH_PROFILE="narrow_ml1"
LOG_WANDB="false"
DRY_RUN="false"

SERIAL_LAYOUT_CATALOG="0,1,1,0,0;1,1,1,1,0"
PARALLEL_LAYOUT_CATALOG="2,0,0,0,0;4,0,0,0,0"

usage() {
  cat <<USAGE
Usage: $0 [--dataset movielens1m] [--gpu 7] [--max-evals 10] [--tune-epochs 100]
          [--tune-patience 10] [--seed 42] [--search-profile narrow_ml1|wide]
          [--log-wandb|--no-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --gpu|--gpu-id) GPU_ID="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --search-profile) SEARCH_PROFILE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

run_phase() {
  local phase="$1"
  local merge_mode="$2"
  local sched="$3"
  local layouts="$4"

  echo ""
  echo "[RUN_PHASE] ${phase} | merge=${merge_mode} | schedule=${sched} | layouts=${layouts}"

  local cmd=(
    bash "${TUNE_SCRIPT}"
    --dataset "${DATASET}"
    --gpu "${GPU_ID}"
    --max-evals "${MAX_EVALS}"
    --tune-epochs "${TUNE_EPOCHS}"
    --tune-patience "${TUNE_PATIENCE}"
    --seed "${SEED}"
    --search-profile "${SEARCH_PROFILE}"
    --phase "${phase}"
    --stage-merge-mode "${merge_mode}"
    --schedule-preset "${sched}"
    --layout-catalog "${layouts}"
  )
  if [ "${LOG_WANDB}" = "true" ]; then
    cmd+=(--log-wandb)
  else
    cmd+=(--no-wandb)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  "${cmd[@]}"
}

echo "[PLAN] FeaturedMoE_HiR 4-phase run"
echo "[PLAN] expert_scale=4 (fixed), wandb_project=FMoE_hir"
echo "[PLAN] serial layouts=${SERIAL_LAYOUT_CATALOG}"
echo "[PLAN] parallel layouts=${PARALLEL_LAYOUT_CATALOG}"

run_phase "P3HIR_SER_off" "serial" "off" "${SERIAL_LAYOUT_CATALOG}"
run_phase "P3HIR_SER_temp" "serial" "temp_mild" "${SERIAL_LAYOUT_CATALOG}"
run_phase "P3HIR_PAR_off" "parallel" "off" "${PARALLEL_LAYOUT_CATALOG}"
run_phase "P3HIR_PAR_temp" "parallel" "temp_mild" "${PARALLEL_LAYOUT_CATALOG}"

echo ""
echo "[DONE] 4-phase HiR run finished"
