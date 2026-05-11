#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="gru4rec,duorec,difsr,fame"
GPU_LIST="0,1,2,3"
SEEDS_D="1"
SEEDS_E="1,2"

D_PROFILES="D1,D2,D3,D4,D5,D6,D7,D8"
E_PROFILES="E1,E2,E3,E4,E5"
D_C_TOPK="3"
E_D_TOPK="2"

RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="false"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="6"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus ...]
          [--seeds-d 1] [--seeds-e 1,2]
          [--d-profiles ...] [--e-profiles ...]
          [--d-c-topk 3] [--e-d-topk 2]
          [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 6]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seeds-d) SEEDS_D="$2"; shift 2 ;;
    --seeds-e) SEEDS_E="$2"; shift 2 ;;
    --d-profiles) D_PROFILES="$2"; shift 2 ;;
    --e-profiles) E_PROFILES="$2"; shift 2 ;;
    --d-c-topk) D_C_TOPK="$2"; shift 2 ;;
    --e-d-topk) E_D_TOPK="$2"; shift 2 ;;
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

D_CMD=(
  "${SCRIPT_DIR}/stageD_micro.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${D_PROFILES}"
  --c-topk "${D_C_TOPK}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS_D}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

E_CMD=(
  "${SCRIPT_DIR}/stageE_relr_seed.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --profiles "${E_PROFILES}"
  --d-topk "${E_D_TOPK}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS_E}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ "${RESUME_FROM_LOGS}" = "true" ]; then
  D_CMD+=(--resume-from-logs)
  E_CMD+=(--resume-from-logs)
else
  D_CMD+=(--no-resume-from-logs)
  E_CMD+=(--no-resume-from-logs)
fi
if [ "${VERIFY_LOGGING}" = "true" ]; then
  D_CMD+=(--verify-logging)
  E_CMD+=(--verify-logging)
else
  D_CMD+=(--no-verify-logging)
  E_CMD+=(--no-verify-logging)
fi
if [ "${DRY_RUN}" = "true" ]; then
  D_CMD+=(--dry-run)
  E_CMD+=(--dry-run)
fi
if [ "${SMOKE_TEST}" = "true" ]; then
  D_CMD+=(--smoke-test)
  E_CMD+=(--smoke-test)
fi

echo "[stageDE_2nd] Running Stage D"
"${D_CMD[@]}"

echo "[stageDE_2nd] Running Stage E"
"${E_CMD[@]}"

echo "[All Done] baseline StageD+StageE 2nd completed: ${DATASETS} / ${MODELS}"
