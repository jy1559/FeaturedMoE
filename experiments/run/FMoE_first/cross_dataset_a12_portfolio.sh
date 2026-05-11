#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/cross_dataset_a12_portfolio.py \
  --datasets "${N4_DATASETS:-beauty,foursquare,movielens1m,retail_rocket,lastfm0.03}" \
  --dataset-template-counts "${N4_TEMPLATE_COUNTS:-beauty:24,foursquare:12,movielens1m:6,retail_rocket:6,lastfm0.03:4}" \
  --max-evals "${N4_MAX_EVALS:-20}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-100}" \
  --tune-patience "${N4_TUNE_PATIENCE:-0}" \
  --batch-size "${N4_BATCH_SIZE:-4096}" \
  --eval-batch-size "${N4_EVAL_BATCH_SIZE:-6144}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"