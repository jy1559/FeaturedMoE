#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/cross_dataset_a12_portfolio_fs_rr_ml_followup.py \
  --foursquare-template-count "${N4_FOURSQUARE_TEMPLATE_COUNT:-8}" \
  --foursquare-template-start-index "${N4_FOURSQUARE_TEMPLATE_START_INDEX:-20}" \
  --foursquare-max-evals "${N4_FOURSQUARE_MAX_EVALS:-8}" \
  --foursquare-batch-size "${N4_FOURSQUARE_BATCH_SIZE:-3072}" \
  --foursquare-eval-batch-size "${N4_FOURSQUARE_EVAL_BATCH_SIZE:-4096}" \
  --retail-template-count "${N4_RETAIL_TEMPLATE_COUNT:-4}" \
  --retail-template-start-index "${N4_RETAIL_TEMPLATE_START_INDEX:-14}" \
  --retail-max-evals "${N4_RETAIL_MAX_EVALS:-6}" \
  --retail-batch-size "${N4_RETAIL_BATCH_SIZE:-3072}" \
  --retail-eval-batch-size "${N4_RETAIL_EVAL_BATCH_SIZE:-4096}" \
  --movielens-template-count "${N4_MOVIELENS_TEMPLATE_COUNT:-4}" \
  --movielens-template-start-index "${N4_MOVIELENS_TEMPLATE_START_INDEX:-6}" \
  --movielens-max-evals "${N4_MOVIELENS_MAX_EVALS:-6}" \
  --movielens-batch-size "${N4_MOVIELENS_BATCH_SIZE:-4096}" \
  --movielens-eval-batch-size "${N4_MOVIELENS_EVAL_BATCH_SIZE:-6144}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-100}" \
  --tune-patience "${N4_TUNE_PATIENCE:-10}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"