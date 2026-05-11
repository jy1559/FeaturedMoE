#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS="lastfm0.03,amazon_beauty"
MODELS="gru4rec,duorec,difsr,fame"
GPU_LIST="0,1,2,3"
SEEDS_BC="1"
SEEDS_D="1"
SEEDS_E="1,2"

RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="false"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="6"
BC_FULL_PROFILE_POOL="false"

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus ...]
          [--seeds-bc 1] [--seeds-d 1] [--seeds-e 1,2]
          [--bc-full-profile-pool]
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
    --seeds-bc) SEEDS_BC="$2"; shift 2 ;;
    --seeds-d) SEEDS_D="$2"; shift 2 ;;
    --seeds-e) SEEDS_E="$2"; shift 2 ;;
    --bc-full-profile-pool) BC_FULL_PROFILE_POOL="true"; shift ;;
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

BC_CMD=(
  "${SCRIPT_DIR}/stageBC_2nd.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --gpus "${GPU_LIST}"
  --seeds "${SEEDS_BC}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ "${BC_FULL_PROFILE_POOL}" = "true" ]; then
  BC_CMD+=(--full-profile-pool)
fi

DE_CMD=(
  "${SCRIPT_DIR}/stageDE_2nd.sh"
  --datasets "${DATASETS}"
  --models "${MODELS}"
  --gpus "${GPU_LIST}"
  --seeds-d "${SEEDS_D}"
  --seeds-e "${SEEDS_E}"
  --smoke-max-runs "${SMOKE_MAX_RUNS}"
)

if [ "${RESUME_FROM_LOGS}" = "true" ]; then
  BC_CMD+=(--resume-from-logs)
  DE_CMD+=(--resume-from-logs)
else
  BC_CMD+=(--no-resume-from-logs)
  DE_CMD+=(--no-resume-from-logs)
fi
if [ "${VERIFY_LOGGING}" = "true" ]; then
  BC_CMD+=(--verify-logging)
  DE_CMD+=(--verify-logging)
else
  BC_CMD+=(--no-verify-logging)
  DE_CMD+=(--no-verify-logging)
fi
if [ "${DRY_RUN}" = "true" ]; then
  BC_CMD+=(--dry-run)
  DE_CMD+=(--dry-run)
fi
if [ "${SMOKE_TEST}" = "true" ]; then
  BC_CMD+=(--smoke-test)
  DE_CMD+=(--smoke-test)
fi

echo "[stageBCDE_2nd] Running Stage BC 2nd"
"${BC_CMD[@]}"

echo "[stageBCDE_2nd] Running Stage DE 2nd"
"${DE_CMD[@]}"

echo "[All Done] baseline StageBCDE 2nd completed: ${DATASETS} / ${MODELS}"
