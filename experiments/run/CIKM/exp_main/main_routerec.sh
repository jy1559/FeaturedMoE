#!/usr/bin/env bash
# CIKM 2026 – launch RouteRec (FeaturedMoE_N3) experiments (P0 main table)
#
# Usage:
#   bash main_routerec.sh [GPU_IDS...]
#
# Examples:
#   bash main_routerec.sh        # GPU 0
#   bash main_routerec.sh 0 1    # two GPUs (KuaiRec on 0, lastfm on 1)

set -euo pipefail

GPUS="${@:-0}"
GPU_ARRAY=($GPUS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/venv/FMoE/bin/python

echo "[main_routerec.sh] GPUs: ${GPU_ARRAY[*]}"
echo "[main_routerec.sh] Starting at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

cd "$SCRIPT_DIR"
exec "$PYTHON" main_routerec.py --gpus "${GPU_ARRAY[@]}"
