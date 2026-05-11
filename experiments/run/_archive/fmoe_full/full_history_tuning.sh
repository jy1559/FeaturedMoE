#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_full/full_history_tuning.py \
  --datasets "${FMOE_FULL_DATASETS:-beauty,KuaiRecLargeStrictPosV2_0.2,foursquare,retail_rocket,movielens1m,lastfm0.03}" \
  --gpus "${FMOE_FULL_GPUS:-0,1,2,3}" \
  "$@"