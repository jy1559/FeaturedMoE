#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="gru4rec,duorec,difsr,fame"
PROFILES="E1,E2,E3,E4,E5"
D_TOPK="2"
GPU_LIST="0,1,2,3"
SEEDS="1,2"
SEED_BASE="200000"

MAX_EVALS_DEFAULT="8"
MAX_EVALS_DUOREC="8"
MAX_EVALS_FAME="8"

TUNE_EPOCHS_DEFAULT="68"
TUNE_EPOCHS_DUOREC="50"
TUNE_EPOCHS_FAME="56"

TUNE_PATIENCE_DEFAULT="10"
TUNE_PATIENCE_DUOREC="8"
TUNE_PATIENCE_FAME="8"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

DRY_RUN="false"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="6"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--profiles E1..E5] [--d-topk 2]
          [--gpus ...] [--seeds 1,2] [--seed-base ...]
          [--max-evals-default 8] [--max-evals-duorec 8] [--max-evals-fame 8]
          [--tune-epochs-default 68] [--tune-epochs-duorec 50] [--tune-epochs-fame 56]
          [--tune-patience-default 10] [--tune-patience-duorec 8] [--tune-patience-fame 8]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 6]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --d-topk) D_TOPK="$2"; shift 2 ;;
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
  "${SCRIPT_DIR}/run_stageE_relr_seed.py"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${PROFILES}"
  --b-topk "${D_TOPK}"
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

echo "[All Done] baseline StageE relr-seed completed: ${DATASETS}"
