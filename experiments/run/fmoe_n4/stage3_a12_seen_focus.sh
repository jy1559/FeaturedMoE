#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/stage3_a12_seen_focus.py \
  --datasets "${N4_DATASETS:-KuaiRecLargeStrictPosV2_0.2}" \
  --template-count "${N4_TEMPLATE_COUNT:-16}" \
  --max-evals "${N4_MAX_EVALS:-12}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-100}" \
  --tune-patience "${N4_TUNE_PATIENCE:-10}" \
  --batch-size "${N4_BATCH_SIZE:-4096}" \
  --eval-batch-size "${N4_EVAL_BATCH_SIZE:-4096}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"