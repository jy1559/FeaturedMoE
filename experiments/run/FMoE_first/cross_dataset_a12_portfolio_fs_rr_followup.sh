#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/cross_dataset_a12_portfolio.py \
  --datasets foursquare \
  --dataset-template-counts "${N4_FOURSQUARE_TEMPLATE_COUNTS:-foursquare:8}" \
  --template-start-index "${N4_FOURSQUARE_TEMPLATE_START_INDEX:-12}" \
  --max-evals "${N4_FOURSQUARE_MAX_EVALS:-8}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-100}" \
  --tune-patience "${N4_TUNE_PATIENCE:-10}" \
  --batch-size "${N4_FOURSQUARE_BATCH_SIZE:-3072}" \
  --eval-batch-size "${N4_FOURSQUARE_EVAL_BATCH_SIZE:-4096}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"

"${PY_BIN}" run/fmoe_n4/cross_dataset_a12_portfolio.py \
  --datasets retail_rocket \
  --dataset-template-counts "${N4_RETAIL_TEMPLATE_COUNTS:-retail_rocket:8}" \
  --template-start-index "${N4_RETAIL_TEMPLATE_START_INDEX:-6}" \
  --max-evals "${N4_RETAIL_MAX_EVALS:-6}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-100}" \
  --tune-patience "${N4_TUNE_PATIENCE:-10}" \
  --batch-size "${N4_RETAIL_BATCH_SIZE:-3072}" \
  --eval-batch-size "${N4_RETAIL_EVAL_BATCH_SIZE:-4096}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"