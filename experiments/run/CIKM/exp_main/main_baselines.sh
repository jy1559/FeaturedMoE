#!/usr/bin/env bash
# CIKM 2026 – launch baseline experiments (P0 main table)
#
# Usage:
#   bash main_baselines.sh [GPU_IDS...]
#
# Examples:
#   bash main_baselines.sh          # single GPU 0
#   bash main_baselines.sh 0 1      # two GPUs, parallel
#   bash main_baselines.sh 0 1 2 3  # four GPUs

set -euo pipefail

GPUS="${@:-0}"
GPU_ARRAY=($GPUS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/venv/FMoE/bin/python

echo "[main_baselines.sh] GPUs: ${GPU_ARRAY[*]}"
echo "[main_baselines.sh] Starting at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

cd "$SCRIPT_DIR"
exec "$PYTHON" main_baselines.py --gpus "${GPU_ARRAY[@]}"
