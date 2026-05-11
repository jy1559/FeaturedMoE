#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/cross_dataset_a12_portfolio_topup.py \
  --gpus "${N4_GPUS:-0,1,2,3,4,5,6,7}" \
  --seeds "${N4_SEEDS:-1}" \
  --search-algo "${N4_SEARCH_ALGO:-tpe}" \
  "$@"
