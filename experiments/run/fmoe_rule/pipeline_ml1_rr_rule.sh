#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="movielens1m,retail_rocket"
GPU_LIST="0,1"
SEED_BASE="42"
LAYOUT_ID="0"
EXECUTION="serial"
SCHEDULE="off"
ARMS="B0,B1,R0,R1"
EPOCHS="100"
PATIENCE="10"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1]
          [--arms B0,B1,R0,R1] [--layout-id N] [--execution serial|parallel]
          [--schedule off|on] [--seed-base N] [--epochs N] [--patience N]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --execution) EXECUTION="$2"; shift 2 ;;
    --schedule) SCHEDULE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
IFS=',' read -r -a ARM_ARR <<< "$ARMS"

if [ "${#DATASET_ARR[@]}" -eq 0 ] || [ "${#GPU_ARR[@]}" -eq 0 ] || [ "${#ARM_ARR[@]}" -eq 0 ]; then
  echo "datasets/gpus/arms must be non-empty" >&2
  exit 1
fi

idx=0
for ds in "${DATASET_ARR[@]}"; do
  for arm in "${ARM_ARR[@]}"; do
    gidx=$(( idx % ${#GPU_ARR[@]} ))
    gpu="${GPU_ARR[$gidx]}"
    seed=$(( SEED_BASE + idx ))
    phase="RULE_${arm}_${ds}"

    cmd=(
      "${SCRIPT_DIR}/train_single.sh"
      --dataset "$ds"
      --gpu "$gpu"
      --seed "$seed"
      --phase "$phase"
      --epochs "$EPOCHS"
      --patience "$PATIENCE"
      --layout-id "$LAYOUT_ID"
      --execution "$EXECUTION"
      --schedule "$SCHEDULE"
      --ablation "$arm"
    )

    echo "[PIPE] ds=${ds} arm=${arm} gpu=${gpu} seed=${seed}"
    if [ "$DRY_RUN" = "true" ]; then
      cmd+=(--dry-run)
    fi
    "${cmd[@]}"

    idx=$((idx + 1))
  done
done
