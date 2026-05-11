#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="python"
fi

cd "${ROOT_DIR}/experiments"

"${PY_BIN}" run/fmoe_n4/stage1_a12_broad_templates.py \
  --datasets "${N4_DATASETS:-KuaiRecLargeStrictPosV2_0.2}" \
  --template-count "${N4_TEMPLATE_COUNT:-8}" \
  --max-evals "${N4_MAX_EVALS:-8}" \
  --tune-epochs "${N4_TUNE_EPOCHS:-25}" \
  --tune-patience "${N4_TUNE_PATIENCE:-3}" \
  --gpus "${N4_GPUS:-0,1,2,3}" \
  --seeds "${N4_SEEDS:-1}" \
  "$@"
