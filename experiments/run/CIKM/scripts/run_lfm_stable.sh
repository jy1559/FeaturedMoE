#!/usr/bin/env bash
# Stable full LastFM run for CIKM experiments.
#
# Usage:
#   bash run_lfm_stable.sh 0
#   bash run_lfm_stable.sh 0 --models sasrec featured_moe_n3 --max-evals 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/venv/FMoE/bin/python}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

GPU="${1:-0}"
if [ "$#" -gt 0 ]; then
    shift
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTER_LOG="$LOG_DIR/run_lfm_stable.outer.log"

echo "================================================================="
echo "  CIKM LastFM stable full run"
echo "  Started : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  GPU     : $GPU"
echo "  Log     : $OUTER_LOG"
echo "================================================================="

exec "$PYTHON" "$SCRIPT_DIR/exp_main/main_lfm_stable.py" --gpu "$GPU" "$@" \
    2>&1 | tee -a "$OUTER_LOG"
