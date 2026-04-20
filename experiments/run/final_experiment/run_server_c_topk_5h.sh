#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DATASETS="${TARGET_DATASETS:-beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

export FINAL_EXP_TRACK="${FINAL_EXP_TRACK:-final_topk}"
export RUN_PYTHON_BIN="${RUN_PYTHON_BIN:-/venv/FMoE-server-c/bin/python}"

# Keep the whole 3-stage run short (target <= 5h total under normal multi-GPU conditions).
export FINAL_EXP_STAGE1_TUNE_EPOCHS="${FINAL_EXP_STAGE1_TUNE_EPOCHS:-16}"
export FINAL_EXP_STAGE1_TUNE_PATIENCE="${FINAL_EXP_STAGE1_TUNE_PATIENCE:-3}"
export FINAL_EXP_STAGE2_TUNE_EPOCHS="${FINAL_EXP_STAGE2_TUNE_EPOCHS:-22}"
export FINAL_EXP_STAGE2_TUNE_PATIENCE="${FINAL_EXP_STAGE2_TUNE_PATIENCE:-4}"
export FINAL_EXP_STAGE3_TUNE_EPOCHS="${FINAL_EXP_STAGE3_TUNE_EPOCHS:-18}"
export FINAL_EXP_STAGE3_TUNE_PATIENCE="${FINAL_EXP_STAGE3_TUNE_PATIENCE:-3}"

echo "[topk-5h] track=${FINAL_EXP_TRACK} datasets=${TARGET_DATASETS} gpus=${GPU_LIST}"

python3 "${SCRIPT_DIR}/topk_stage1.py" \
  --datasets "${TARGET_DATASETS}" \
  --gpus "${GPU_LIST}" \
  --max-evals "${STAGE1_MAX_EVALS:-2}" \
  --max-run-hours "${STAGE1_MAX_RUN_HOURS:-0.65}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT:-4}" \
  "$@"

python3 "${SCRIPT_DIR}/topk_stage2.py" \
  --datasets "${TARGET_DATASETS}" \
  --gpus "${GPU_LIST}" \
  --max-evals "${STAGE2_MAX_EVALS:-3}" \
  --max-run-hours "${STAGE2_MAX_RUN_HOURS:-0.85}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT:-4}" \
  "$@"

python3 "${SCRIPT_DIR}/topk_stage3.py" \
  --datasets "${TARGET_DATASETS}" \
  --gpus "${GPU_LIST}" \
  --seeds "${STAGE3_SEEDS:-3}" \
  --max-run-hours "${STAGE3_MAX_RUN_HOURS:-0.55}" \
  --oom-retry-limit "${OOM_RETRY_LIMIT:-4}" \
  "$@"

echo "[topk-5h] done"