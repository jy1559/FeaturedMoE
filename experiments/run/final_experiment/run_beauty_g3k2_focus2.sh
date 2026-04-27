#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_LIST="${GPU_LIST:-0,1,2,3}"

export FINAL_EXP_TRACK="${FINAL_EXP_TRACK:-final_topk_beauty_g3k2_v2}"
export RUN_PYTHON_BIN="${RUN_PYTHON_BIN:-/venv/FMoE-server-c/bin/python}"

export FINAL_EXP_STAGE1_TUNE_EPOCHS="${FINAL_EXP_STAGE1_TUNE_EPOCHS:-22}"
export FINAL_EXP_STAGE1_TUNE_PATIENCE="${FINAL_EXP_STAGE1_TUNE_PATIENCE:-4}"

echo "[beauty-g3k2-focus2] track=${FINAL_EXP_TRACK} gpus=${GPU_LIST}"

python3 "${SCRIPT_DIR}/beauty_g3k2_focus2.py" \
  --gpus "${GPU_LIST}" \
  --max-evals "${MAX_EVALS:-8}" \
  --max-run-hours "${MAX_RUN_HOURS:-0.65}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT:-4}" \
  "$@"

echo "[beauty-g3k2-focus2] done"
