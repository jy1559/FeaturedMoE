#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="sasrec,gru4rec,duorec,difsr,fame"
LR_BANDS="AUTO6"
GPU_LIST="0,1,2,3"
SEEDS="1"
SEED_BASE="160000"

MAX_EVALS_DEFAULT="6"
MAX_EVALS_DUOREC="4"
MAX_EVALS_FAME="5"

TUNE_EPOCHS_DEFAULT="36"
TUNE_EPOCHS_DUOREC="24"
TUNE_EPOCHS_FAME="32"

TUNE_PATIENCE_DEFAULT="5"
TUNE_PATIENCE_DUOREC="3"
TUNE_PATIENCE_FAME="4"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="4"

usage() {
  cat <<USAGE
Usage: $0 [--datasets lastfm0.03,amazon_beauty] [--models sasrec,gru4rec,duorec,difsr,fame]
          [--lr-bands AUTO6|2e-4:7e-4,4e-4:1.2e-3,...]
          [--gpus 0,1,2,3] [--seeds 1] [--seed-base 160000]
          [--max-evals-default 6] [--max-evals-duorec 4] [--max-evals-fame 5]
          [--tune-epochs-default 36] [--tune-epochs-duorec 24] [--tune-epochs-fame 32]
          [--tune-patience-default 5] [--tune-patience-duorec 3] [--tune-patience-fame 4]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 4]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --lr-bands) LR_BANDS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;

    --max-evals-default) MAX_EVALS_DEFAULT="$2"; shift 2 ;;
    --max-evals-duorec) MAX_EVALS_DUOREC="$2"; shift 2 ;;
    --max-evals-fame) MAX_EVALS_FAME="$2"; shift 2 ;;

    --tune-epochs-default) TUNE_EPOCHS_DEFAULT="$2"; shift 2 ;;
    --tune-epochs-duorec) TUNE_EPOCHS_DUOREC="$2"; shift 2 ;;
    --tune-epochs-fame) TUNE_EPOCHS_FAME="$2"; shift 2 ;;

    --tune-patience-default) TUNE_PATIENCE_DEFAULT="$2"; shift 2 ;;
    --tune-patience-duorec) TUNE_PATIENCE_DUOREC="$2"; shift 2 ;;
    --tune-patience-fame) TUNE_PATIENCE_FAME="$2"; shift 2 ;;

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
  "${SCRIPT_DIR}/run_stageA_lr.py"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --lr-bands "${LR_BANDS}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --seed-base "${SEED_BASE}"
  --max-evals-default "${MAX_EVALS_DEFAULT}"
  --max-evals-duorec "${MAX_EVALS_DUOREC}"
  --max-evals-fame "${MAX_EVALS_FAME}"
  --tune-epochs-default "${TUNE_EPOCHS_DEFAULT}"
  --tune-epochs-duorec "${TUNE_EPOCHS_DUOREC}"
  --tune-epochs-fame "${TUNE_EPOCHS_FAME}"
  --tune-patience-default "${TUNE_PATIENCE_DEFAULT}"
  --tune-patience-duorec "${TUNE_PATIENCE_DUOREC}"
  --tune-patience-fame "${TUNE_PATIENCE_FAME}"
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

echo "[All Done] baseline StageA LR completed: ${DATASETS}"
