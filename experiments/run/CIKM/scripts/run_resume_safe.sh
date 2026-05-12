#!/usr/bin/env bash
# Resume CIKM KuaiRec + LastFM jobs.
# Skips clean completed logs and runs only pending/error jobs.
#
# Usage from this CIKM directory:
#   bash run_resume_safe.sh 0 1 2 3
#
# Optional knobs:
#   MAX_EVALS=1 LFM_PARALLEL=4 LFM_TRAIN_BATCH_SIZE=1024 LFM_EVAL_BATCH_SIZE=256 bash run_resume_safe.sh 0 1 2 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/venv/FMoE/bin/python}"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_resume_safe.sh <gpu_id> [gpu_id ...]"
    echo "Example: bash run_resume_safe.sh 0 1 2 3"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_TAG="$(date -u '+%Y%m%d_%H%M%S')"
OUTER_LOG="$LOG_DIR/run_resume_safe_${RUN_TAG}.outer.log"
PREWARM_LOG="$LOG_DIR/prewarm_lfm_cache_${RUN_TAG}.log"
ln -sfn "$(basename "$OUTER_LOG")" "$LOG_DIR/run_resume_safe.latest.log"
ln -sfn "$(basename "$PREWARM_LOG")" "$LOG_DIR/prewarm_lfm_cache.latest.log"

if [ "${PREWARM_LFM_CACHE:-1}" != "0" ]; then
    echo "[PREWARM] LastFM light no-item baseline cache (${LFM_CACHE_BASE_MODEL:-sasrec})" | tee -a "$PREWARM_LOG"
    "$PYTHON" "$SCRIPT_DIR/exp_main/prewarm_lfm_cache.py" \
        --model "${LFM_CACHE_BASE_MODEL:-sasrec}" \
        --light-data \
        --drop-item-features \
        --max-len "${LFM_MAX_LEN:-10}" \
        --train-batch-size "${LFM_TRAIN_BATCH_SIZE:-1024}" \
        --eval-batch-size "${LFM_EVAL_BATCH_SIZE:-256}" \
        --eval-sample-num "${LFM_EVAL_SAMPLE_NUM:-1000}" \
        2>&1 | tee -a "$PREWARM_LOG"

    echo "[PREWARM] LastFM light side-info/category cache (${LFM_CACHE_BASE_MODEL:-sasrec})" | tee -a "$PREWARM_LOG"
    "$PYTHON" "$SCRIPT_DIR/exp_main/prewarm_lfm_cache.py" \
        --model "${LFM_CACHE_BASE_MODEL:-sasrec}" \
        --light-data \
        --max-len "${LFM_MAX_LEN:-10}" \
        --train-batch-size "${LFM_TRAIN_BATCH_SIZE:-1024}" \
        --eval-batch-size "${LFM_EVAL_BATCH_SIZE:-256}" \
        --eval-sample-num "${LFM_EVAL_SAMPLE_NUM:-1000}" \
        2>&1 | tee -a "$PREWARM_LOG"

    if [ "${PREWARM_LFM_FEATURE_CACHE:-1}" != "0" ]; then
        echo "[PREWARM] LastFM RouteRec feature cache (${LFM_CACHE_FEATURE_MODEL:-featured_moe_n3_tune})" | tee -a "$PREWARM_LOG"
        "$PYTHON" "$SCRIPT_DIR/exp_main/prewarm_lfm_cache.py" \
            --model "${LFM_CACHE_FEATURE_MODEL:-featured_moe_n3_tune}" \
            --max-len "${LFM_MAX_LEN:-10}" \
            --train-batch-size "${LFM_TRAIN_BATCH_SIZE:-1024}" \
            --eval-batch-size "${LFM_EVAL_BATCH_SIZE:-256}" \
            --eval-sample-num "${LFM_EVAL_SAMPLE_NUM:-1000}" \
            2>&1 | tee -a "$PREWARM_LOG"
    fi
fi

"$PYTHON" "$SCRIPT_DIR/exp_main/main_resume_safe.py" "$@" \
   --max-evals "${MAX_EVALS:-1}" \
   --epochs "${EPOCHS:-100}" \
   --patience "${PATIENCE:-10}" \
   --lfm-train-batch-size "${LFM_TRAIN_BATCH_SIZE:-1024}" \
   --lfm-eval-batch-size "${LFM_EVAL_BATCH_SIZE:-256}" \
   --lfm-parallel "${LFM_PARALLEL:-4}" \
   --lfm-max-len "${LFM_MAX_LEN:-10}" \
   --lfm-eval-sample-num "${LFM_EVAL_SAMPLE_NUM:-1000}" \
   --oom-retry-limit "${OOM_RETRY_LIMIT:-5}" \
   2>&1 | tee -a "$OUTER_LOG"
