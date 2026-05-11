#!/usr/bin/env bash
set -euo pipefail

# AI502 transfer learning 실험 실행 wrapper.
# 예:
#   ./run_ai502_transfer.sh native "0,1,2,3"
#   ./run_ai502_transfer.sh init "0,1,2,3,4,5,6,7"
#   ./run_ai502_transfer.sh freeze "0,1,2,3"
#   ./run_ai502_transfer.sh multihop "0,1,2,3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-native}"
GPUS="${2:-0}"
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true

cd "${SCRIPT_DIR}"
PY_BIN="${RUN_PYTHON_BIN:-${PYTHON_BIN:-/venv/FMoE/bin/python}}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi
export RUN_PYTHON_BIN="${PY_BIN}"
export PYTHONPATH="/workspace/FeaturedMoE/experiments:/workspace/FeaturedMoE${PYTHONPATH:+:${PYTHONPATH}}"
echo "[AI502] python=${PY_BIN}"

"${PY_BIN}" run_ai502_transfer.py \
  --phase "${PHASE}" \
  --profile fast \
  --gpus "${GPUS}" \
  --lr-mode fixed1 \
  --datasets "beauty,foursquare,KuaiRecLargeStrictPosV2_0.2,lastfm0.03,movielens1m,retail_rocket" \
  --hparams "shared_3,shared_4,shared_5,shared_6" \
  --seeds "1,2,3" \
  "$@"
