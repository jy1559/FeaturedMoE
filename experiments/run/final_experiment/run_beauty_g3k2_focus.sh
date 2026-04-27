#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST="${GPU_LIST:-0,1,2,3}"

export FINAL_EXP_TRACK="${FINAL_EXP_TRACK:-final_topk_beauty_g3k2}"
export RUN_PYTHON_BIN="${RUN_PYTHON_BIN:-/venv/FMoE-server-c/bin/python}"

# Keep the sweep short and comparable to the prior top-k quick pipeline.
export FINAL_EXP_STAGE1_TUNE_EPOCHS="${FINAL_EXP_STAGE1_TUNE_EPOCHS:-18}"
export FINAL_EXP_STAGE1_TUNE_PATIENCE="${FINAL_EXP_STAGE1_TUNE_PATIENCE:-3}"

echo "[beauty-g3k2-focus] track=${FINAL_EXP_TRACK} gpus=${GPU_LIST}"

python3 "${SCRIPT_DIR}/beauty_g3k2_focus.py" \
  --gpus "${GPU_LIST}" \
  --max-evals "${MAX_EVALS:-4}" \
  --max-run-hours "${MAX_RUN_HOURS:-0.45}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT:-4}" \
  "$@"

echo "[beauty-g3k2-focus] done"
