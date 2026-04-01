#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="lastfm0.03,amazon_beauty"
# Low-performing models first (SASRec excluded on purpose)
MODELS="gru4rec,duorec,difsr,fame"
GPU_LIST="0,1,2,3"
SEEDS="1"

# Additive default: run only new diversity profiles on top of completed history
B_PROFILES="B7,B8"
C_PROFILES="C7,C8"
C_B_TOPK="3"
FULL_PROFILE_POOL="false"

RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="false"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="6"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus ...] [--seeds ...]
          [--b-profiles ...] [--c-profiles ...] [--c-b-topk 3]
          [--full-profile-pool]
          [--resume-from-logs|--no-resume-from-logs] [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 6]
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
    --full-profile-pool) FULL_PROFILE_POOL="true"; shift ;;
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

if [ "${FULL_PROFILE_POOL}" = "true" ]; then
  B_PROFILES="B1,B2,B3,B4,B5,B6,B7,B8"
  C_PROFILES="C1,C2,C3,C4,C5,C6,C7,C8"
fi

B_CMD=(
  "${SCRIPT_DIR}/stageB_structure.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${B_PROFILES}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS}"
  --max-evals-default 10
  --max-evals-duorec 10
  --max-evals-fame 10
  --tune-epochs-default 56
  --tune-epochs-duorec 44
  --tune-epochs-fame 48
  --tune-patience-default 8
  --tune-patience-duorec 7
  --tune-patience-fame 7
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
  --max-evals-default 10
  --max-evals-duorec 10
  --max-evals-fame 10
  --tune-epochs-default 60
  --tune-epochs-duorec 46
  --tune-epochs-fame 52
  --tune-patience-default 9
  --tune-patience-duorec 7
  --tune-patience-fame 8
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

echo "[stageBC_2nd] Running Stage B (aggressive low-model mode)"
"${B_CMD[@]}"

echo "[stageBC_2nd] Running Stage C (aggressive low-model mode)"
"${C_CMD[@]}"

echo "[All Done] baseline StageB+StageC 2nd completed: ${DATASETS} / ${MODELS}"
