#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="sasrec,gru4rec,duorec,difsr,fame"
GPU_LIST="0,1,2,3"
SEEDS="1"

B_PROFILES="B1,B2,B3,B4,B5,B6"
C_PROFILES="C1,C2,C3,C4,C5,C6"
C_B_TOPK="2"

RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="4"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus ...] [--seeds ...]
          [--b-profiles B1,B2,B3,B4,B5,B6] [--c-profiles C1,C2,C3,C4,C5,C6] [--c-b-topk 2]
          [--resume-from-logs|--no-resume-from-logs] [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 4]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --b-profiles) B_PROFILES="$2"; shift 2 ;;
    --c-profiles) C_PROFILES="$2"; shift 2 ;;
    --c-b-topk) C_B_TOPK="$2"; shift 2 ;;
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

B_CMD=(
  "${SCRIPT_DIR}/stageB_structure.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${B_PROFILES}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

C_CMD=(
  "${SCRIPT_DIR}/stageC_focus.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${C_PROFILES}"
  --b-topk "${C_B_TOPK}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ "${RESUME_FROM_LOGS}" = "true" ]; then
  B_CMD+=(--resume-from-logs)
  C_CMD+=(--resume-from-logs)
else
  B_CMD+=(--no-resume-from-logs)
  C_CMD+=(--no-resume-from-logs)
fi
if [ "${VERIFY_LOGGING}" = "true" ]; then
  B_CMD+=(--verify-logging)
  C_CMD+=(--verify-logging)
else
  B_CMD+=(--no-verify-logging)
  C_CMD+=(--no-verify-logging)
fi
if [ "${DRY_RUN}" = "true" ]; then
  B_CMD+=(--dry-run)
  C_CMD+=(--dry-run)
fi
if [ "${SMOKE_TEST}" = "true" ]; then
  B_CMD+=(--smoke-test)
  C_CMD+=(--smoke-test)
fi

echo "[stageBC] Running Stage B"
"${B_CMD[@]}"

echo "[stageBC] Running Stage C"
"${C_CMD[@]}"

echo "[All Done] baseline StageB+StageC completed: ${DATASETS}"
