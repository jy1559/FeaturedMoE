#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="amazon_beauty,KuaiRecLargeStrictPosV2_0.2,movielens1m"
MODELS="bsarec,gru4rec,fame,tisasrec,duorec,sasrec,fearec"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS="1,2,3"
SEED_BASE="230000"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="8"
FAST_SCREEN="false"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus 0,1,2,3,4,5,6,7]
          [--seeds 1,2,3] [--seed-base 230000]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 8] [--fast-screen]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
    --resume-from-logs) RESUME_FROM_LOGS="true"; shift ;;
    --no-resume-from-logs) RESUME_FROM_LOGS="false"; shift ;;
    --verify-logging) VERIFY_LOGGING="true"; shift ;;
    --no-verify-logging) VERIFY_LOGGING="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --smoke-test) SMOKE_TEST="true"; shift ;;
    --smoke-max-runs) SMOKE_MAX_RUNS="$2"; shift 2 ;;
    --fast-screen) FAST_SCREEN="true"; shift ;;
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
  "${SCRIPT_DIR}/run_stageH_targeted_recovery.py"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --seed-base "${SEED_BASE}"
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
if [ "${FAST_SCREEN}" = "true" ]; then
  CMD+=(--fast-screen)
fi

run_echo_cmd "${CMD[@]}"
"${CMD[@]}"

echo "[All Done] baseline StageH targeted recovery completed: ${DATASETS}"
