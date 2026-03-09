#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

GROUP_GPUS="0,1,2,3"
DATASETS="movielens1m,retail_rocket"
COMBOS_PER_GPU="3"
MAX_EVALS="40"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--gpus 0,1,2,3] [--datasets movielens1m,retail_rocket]
          [--combos-per-gpu 3] [--max-evals 40]
          [--log-wandb|--no-wandb] [--dry-run]

Runs only FMoEv2 final track.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus|--group-a-gpus) GROUP_GPUS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
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

cmd=(
  bash "${RUN_DIR}/fmoe_v2/final_v2_ml1_rr.sh"
  --datasets "$DATASETS"
  --gpus "$GROUP_GPUS"
  --combos-per-gpu "$COMBOS_PER_GPU"
  --ml1-r1-ctrl-evals "$MAX_EVALS"
  --ml1-r1-spec-evals "$MAX_EVALS"
  --ml1-b0-spec-evals "$MAX_EVALS"
  --rr-transfer-evals "$MAX_EVALS"
)

if [ "$LOG_WANDB" = "true" ]; then
  cmd+=(--log-wandb)
else
  cmd+=(--no-wandb)
fi
if [ "$DRY_RUN" = "true" ]; then
  cmd+=(--dry-run)
fi

echo "[FINAL_ONLY] gpus=${GROUP_GPUS} datasets=${DATASETS} combos_per_gpu=${COMBOS_PER_GPU} max_evals=${MAX_EVALS}"
"${cmd[@]}"
