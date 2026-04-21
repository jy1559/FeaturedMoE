#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"

/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q2_q3_rank_interleave.py \
  --suite "${Q23_SUITE:-q3}" \
  --gpus "${Q23_GPUS:-0}" \
  --datasets "${Q23_DATASETS:-KuaiRecLargeStrictPosV2_0.2,foursquare}" \
  --top-k-configs "${Q23_TOPK:-4}" \
  --rank-order "${Q23_RANKS:-1,2,3,4}" \
  --seeds "${Q23_SEEDS:-1,2,3,4,5}" \
  --max-evals "${Q23_MAX_EVALS:-3}" \
  --tune-epochs "${Q23_TUNE_EPOCHS:-100}" \
  --tune-patience "${Q23_TUNE_PATIENCE:-10}" \
  --max-run-hours "${Q23_MAX_RUN_HOURS:-1.0}" \
  "$@"
