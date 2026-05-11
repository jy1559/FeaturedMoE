#!/usr/bin/env bash
# Run the CIKM main-table experiments on both full datasets:
#   KuaiRec + lastfm
#
# Usage:
#   bash run_full_kuai_lfm.sh 0 1 2 3
#   bash run_full_kuai_lfm.sh 0
#
# GPU allocation:
#   1 GPU : baselines first, then RouteRec on the same GPU
#   2+ GPUs: last GPU for RouteRec, remaining GPUs for baselines

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/venv/FMoE/bin/python}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <gpu_id> [gpu_id ...]"
    echo "Example: bash $0 0 1 2 3"
    exit 1
fi

ALL_GPUS=("$@")
N=${#ALL_GPUS[@]}

if [ "$N" -eq 1 ]; then
    BASELINE_GPUS=("${ALL_GPUS[0]}")
    ROUTE_GPUS=("${ALL_GPUS[0]}")
    PARALLEL=0
else
    BASELINE_GPUS=("${ALL_GPUS[@]:0:$((N-1))}")
    ROUTE_GPUS=("${ALL_GPUS[$((N-1))]}")
    PARALLEL=1
fi

BASELINE_LOG="$LOG_DIR/run_full_kuai_lfm_baselines.log"
ROUTE_LOG="$LOG_DIR/run_full_kuai_lfm_routerec.log"

echo "================================================================="
echo "  CIKM full run: KuaiRec + lastfm"
echo "  Started    : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Datasets   : KuaiRec lastfm"
echo "  Baselines  : ${BASELINE_GPUS[*]}"
echo "  RouteRec   : ${ROUTE_GPUS[*]}"
echo "  Logs       : $LOG_DIR"
echo "================================================================="

run_baselines() {
    "$PYTHON" "$SCRIPT_DIR/exp_main/main_baselines.py" \
        --datasets KuaiRec lastfm \
        --gpus "${BASELINE_GPUS[@]}" \
        > "$BASELINE_LOG" 2>&1
}

run_routerec() {
    "$PYTHON" "$SCRIPT_DIR/exp_main/main_routerec.py" \
        --datasets KuaiRec lastfm \
        --gpus "${ROUTE_GPUS[@]}" \
        > "$ROUTE_LOG" 2>&1
}

if [ "$PARALLEL" -eq 0 ]; then
    echo "[run_full_kuai_lfm] 1 GPU detected; running sequentially."
    echo "[run_full_kuai_lfm] Baselines log: $BASELINE_LOG"
    run_baselines
    echo "[run_full_kuai_lfm] RouteRec log : $ROUTE_LOG"
    run_routerec
else
    echo "[run_full_kuai_lfm] Launching baselines -> $BASELINE_LOG"
    run_baselines &
    BASELINE_PID=$!

    echo "[run_full_kuai_lfm] Launching RouteRec  -> $ROUTE_LOG"
    run_routerec &
    ROUTE_PID=$!

    BASELINE_RC=0
    ROUTE_RC=0
    wait "$BASELINE_PID" || BASELINE_RC=$?
    wait "$ROUTE_PID" || ROUTE_RC=$?

    if [ "$BASELINE_RC" -ne 0 ] || [ "$ROUTE_RC" -ne 0 ]; then
        echo "[run_full_kuai_lfm] failed: baselines=$BASELINE_RC routerec=$ROUTE_RC"
        exit 1
    fi
fi

echo "================================================================="
echo "  DONE       : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Summaries  :"
echo "    $SCRIPT_DIR/results/main_baselines_summary.csv"
echo "    $SCRIPT_DIR/results/main_routerec_summary.csv"
echo "================================================================="
