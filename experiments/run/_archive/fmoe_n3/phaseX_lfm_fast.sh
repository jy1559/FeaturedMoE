#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="lastfm0.03"
GPU_LIST="0,1,2,3"
MAX_EVALS="5"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="19000"
PHASE_NAME="LFMFAST12"
FIXED_DROPOUT="0.15"
FIXED_WEIGHT_DECAY="1e-6"
DRY_RUN="${DRY_RUN:-false}"
ONLY_PHASES=""
CATEGORY_FILTER=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--dataset lastfm0.03] [--gpus 0,1,2,3]
          [--max-evals 5] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 19000] [--phase-name LFMFAST12]
          [--fixed-dropout 0.15] [--fixed-weight-decay 1e-6]
          [--manifest-out path] [--dry-run]
          [--only LFMFAST_C0,LFMFAST_C4] [--category control,transfer_topk]
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --max-evals)
      MAX_EVALS="$2"
      shift 2
      ;;
    --tune-epochs)
      TUNE_EPOCHS="$2"
      shift 2
      ;;
    --tune-patience)
      TUNE_PATIENCE="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --phase-name)
      PHASE_NAME="$2"
      shift 2
      ;;
    --fixed-dropout)
      FIXED_DROPOUT="$2"
      shift 2
      ;;
    --fixed-weight-decay)
      FIXED_WEIGHT_DECAY="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --only)
      ONLY_PHASES="$2"
      shift 2
      ;;
    --category)
      CATEGORY_FILTER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"
CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_phaseX_lfm_fast.py"
  --dataset "${DATASET}"
  --gpus "${GPU_LIST}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --seed-base "${SEED_BASE}"
  --phase-name "${PHASE_NAME}"
  --fixed-dropout "${FIXED_DROPOUT}"
  --fixed-weight-decay "${FIXED_WEIGHT_DECAY}"
)

if [ -n "${MANIFEST_OUT}" ]; then
  CMD+=(--manifest-out "${MANIFEST_OUT}")
fi
if [ -n "${ONLY_PHASES}" ]; then
  CMD+=(--only "${ONLY_PHASES}")
fi
if [ -n "${CATEGORY_FILTER}" ]; then
  CMD+=(--category "${CATEGORY_FILTER}")
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi

run_echo_cmd "${CMD[@]}"
exec "${CMD[@]}"
