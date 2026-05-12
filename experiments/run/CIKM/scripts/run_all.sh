#!/usr/bin/env bash
# CIKM 2026 – run ALL main experiments (baselines + RouteRec) on N GPUs.
#
# Usage:
#   bash run_all.sh 0 1 2 3          # 4 GPUs (recommended)
#   bash run_all.sh 0 1 2            # 3 GPUs
#   bash run_all.sh 0                # 1 GPU (sequential)
#
# GPU allocation:
#   Last GPU   → RouteRec (FMoE_N3) on KuaiRec then lastfm
#   Remaining  → 9 baselines × 2 datasets (18 jobs, round-robin)
#
# Logs:
#   logs/run_all_baselines.log
#   logs/run_all_routerec.log
#
# Results:
#   results/main_baselines_summary.csv
#   results/main_routerec_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/venv/FMoE/bin/python
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── parse GPU args ─────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "[run_all] No GPUs specified. Usage: bash run_all.sh 0 1 2 3"
    exit 1
fi

ALL_GPUS=("$@")
N=${#ALL_GPUS[@]}

if [ "$N" -eq 1 ]; then
    # Only 1 GPU: run baselines then routerec sequentially on same GPU
    BASELINE_GPUS=("${ALL_GPUS[0]}")
    ROUTE_GPUS=("${ALL_GPUS[0]}")
else
    # Last GPU for RouteRec; rest for baselines
    ROUTE_GPUS=("${ALL_GPUS[$((N-1))]}")
    BASELINE_GPUS=("${ALL_GPUS[@]:0:$((N-1))}")
fi

echo "================================================================="
echo "  CIKM 2026 – full experiment run"
echo "  Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Total GPUs : ${ALL_GPUS[*]}"
echo "  Baselines  : GPU(s) ${BASELINE_GPUS[*]}  (9 models × KuaiRec + lastfm)"
echo "  RouteRec   : GPU(s) ${ROUTE_GPUS[*]}     (FMoE_N3 × KuaiRec + lastfm)"
echo "================================================================="

BASELINE_LOG="$LOG_DIR/run_all_baselines.log"
ROUTE_LOG="$LOG_DIR/run_all_routerec.log"

# ── launch baselines ───────────────────────────────────────────────────────────
echo ""
echo "[run_all] Launching baselines → $BASELINE_LOG"
"$PYTHON" "$SCRIPT_DIR/exp_main/main_baselines.py" \
    --gpus "${BASELINE_GPUS[@]}" \
    > "$BASELINE_LOG" 2>&1 &
BASELINE_PID=$!
echo "[run_all] Baselines PID=$BASELINE_PID"

# ── launch RouteRec ────────────────────────────────────────────────────────────
echo "[run_all] Launching RouteRec  → $ROUTE_LOG"
"$PYTHON" "$SCRIPT_DIR/exp_main/main_routerec.py" \
    --gpus "${ROUTE_GPUS[@]}" \
    > "$ROUTE_LOG" 2>&1 &
ROUTE_PID=$!
echo "[run_all] RouteRec  PID=$ROUTE_PID"

echo ""
echo "Both runners started. Waiting for completion..."
echo "  tail -f $BASELINE_LOG"
echo "  tail -f $ROUTE_LOG"
echo ""

# ── wait and report ────────────────────────────────────────────────────────────
BASELINE_RC=0
ROUTE_RC=0

wait $BASELINE_PID || BASELINE_RC=$?
echo "[run_all] Baselines finished  rc=$BASELINE_RC  $(date -u '+%H:%M:%S UTC')"

wait $ROUTE_PID    || ROUTE_RC=$?
echo "[run_all] RouteRec  finished  rc=$ROUTE_RC   $(date -u '+%H:%M:%S UTC')"

echo ""
echo "================================================================="
echo "  DONE  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Baselines rc=$BASELINE_RC  |  RouteRec rc=$ROUTE_RC"
echo "  Results:"
echo "    $SCRIPT_DIR/results/main_baselines_summary.csv"
echo "    $SCRIPT_DIR/results/main_routerec_summary.csv"
echo "================================================================="

if [ "$BASELINE_RC" -ne 0 ] || [ "$ROUTE_RC" -ne 0 ]; then
    exit 1
fi
