#!/usr/bin/env bash
set -euo pipefail

# AI502 transfer follow-up 실행 스크립트.
# 기존 artifacts/checkpoints/native는 재사용하고, 부족한 seed/hparam native만 top-up한다.
# transfer 결과는 artifacts_real 아래에 별도로 저장한다.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="0,1,2,3"
SEEDS="1,2,3,4,5"
POLICIES="std,loaded_lr_0.35,global_lr_0.5"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --policies)
      POLICIES="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "${SCRIPT_DIR}"
PY_BIN="${RUN_PYTHON_BIN:-${PYTHON_BIN:-/venv/FMoE/bin/python}}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3)"
fi
export RUN_PYTHON_BIN="${PY_BIN}"
export PYTHONPATH="/workspace/FeaturedMoE/experiments:/workspace/FeaturedMoE${PYTHONPATH:+:${PYTHONPATH}}"

echo "[AI502_REAL] python=${PY_BIN}"
echo "[AI502_REAL] gpus=${GPUS} seeds=${SEEDS} policies=${POLICIES}"

"${PY_BIN}" run_ai502_real.py \
  --phase all \
  --gpus "${GPUS}" \
  --seeds "${SEEDS}" \
  --policies "${POLICIES}" \
  --resume \
  "${EXTRA_ARGS[@]}"
