#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${RUN_PYTHON_BIN:-${PYTHON_BIN:-/venv/FMoE/bin/python}}"
GPUS="0,1,2,3"
SEEDS="1,2,3,4,5"
PHASE="all"
POLICIES="std,loaded_lr_0.35,loaded_lr_0.05"
MODES="feature_encoder_init,group_router_init,feature_encoder_group_router_init,feature_encoder_a12_router_init,full_except_feature_router_init,full_model_init"
PAIRS=""
EPOCHS="100"
PATIENCE="10"
DRY_RUN=0
CLEAN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --policies) POLICIES="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --pairs) PAIRS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --patience) PATIENCE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --clean) CLEAN=1; shift ;;
    --no-resume) EXTRA_ARGS+=(--no-resume); shift ;;
    --keep-going) EXTRA_ARGS+=(--keep-going); shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$ROOT_DIR/artifacts_group1_best"
  rm -f "$ROOT_DIR/result_group1_best.md"
fi

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/run_ai502_group1_best.py"
  --phase "$PHASE"
  --gpus "$GPUS"
  --seeds "$SEEDS"
  --policies "$POLICIES"
  --modes "$MODES"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
)
if [[ -n "$PAIRS" ]]; then
  CMD+=(--pairs "$PAIRS")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=(--dry-run)
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[AI502 group1 best] ${CMD[*]}"
exec "${CMD[@]}"
