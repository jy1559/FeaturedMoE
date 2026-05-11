#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="sasrec,gru4rec,duorec,difsr,fame"
SEARCH_MODE="lr"
PROFILES="AUTO4"
LR_SPACES="AUTO4"
GPU_LIST="0,1,2,3"
SEEDS="1"
SEED_BASE="150000"

MAX_EVALS="10"
TUNE_EPOCHS="80"
TUNE_PATIENCE="12"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="4"

usage() {
  cat <<USAGE
Usage: $0 [--datasets lastfm0.03,amazon_beauty] [--models sasrec,gru4rec,duorec,difsr,fame]
          [--mode hparam|lr]
          [--profiles AUTO4|AUTO16|C1D1,C1D2,...,C4D4]
          [--lr-spaces AUTO4|2e-4:6e-4,6e-4:2e-3,2e-3:6e-3,3e-3:1e-2]
          [--gpus 0,1,2,3] [--seeds 1]
          [--max-evals 10] [--tune-epochs 80] [--tune-patience 12]
          [--seed-base 150000]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 4]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --mode|--search-mode) SEARCH_MODE="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --lr-spaces) LR_SPACES="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
    --resume-from-logs) RESUME_FROM_LOGS="true"; shift ;;
    --no-resume-from-logs) RESUME_FROM_LOGS="false"; shift ;;
    --verify-logging) VERIFY_LOGGING="true"; shift ;;
    --no-verify-logging) VERIFY_LOGGING="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --smoke-test) SMOKE_TEST="true"; shift ;;
    --smoke-max-runs) SMOKE_MAX_RUNS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"
CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_widesearch_anchor_core.py"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --search-mode "${SEARCH_MODE}"
  --profiles "${PROFILES}"
  --lr-spaces "${LR_SPACES}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --seed-base "${SEED_BASE}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ -n "${MANIFEST_OUT}" ]; then
  CMD+=(--manifest-out "${MANIFEST_OUT}")
fi
if [ "${RESUME_FROM_LOGS}" = "true" ]; then
  CMD+=(--resume-from-logs)
else
  CMD+=(--no-resume-from-logs)
fi
if [ "${VERIFY_LOGGING}" = "true" ]; then
  CMD+=(--verify-logging)
else
  CMD+=(--no-verify-logging)
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ "${SMOKE_TEST}" = "true" ]; then
  CMD+=(--smoke-test)
fi

run_echo_cmd "${CMD[@]}"
"${CMD[@]}"

echo "[All Done] baseline WideSearch(anchor2/core5) completed: ${DATASETS}"
