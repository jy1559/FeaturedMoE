#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULE_DIR="$(cd "${SCRIPT_DIR}/../fmoe_rule" && pwd)"

DATASETS="retail_rocket"
GPU_LIST="0,1,2,3"
CATALOG_PROFILE="rr_rule8"
MAX_EVALS="8"
TUNE_EPOCHS="60"
TUNE_PATIENCE="8"
PHASE_PREFIX="RRRULEV3"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--datasets retail_rocket] [--gpus 0,1,2,3]
          [--catalog-profile rr_rule8|rr_rule12]
          [--max-evals 8] [--tune-epochs 60] [--tune-patience 8]
          [--phase-prefix RRRULEV3] [--log-wandb] [--dry-run]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --catalog-profile) CATALOG_PROFILE="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

cmd=(
  bash "${RULE_DIR}/rr_rule_quick_tune.sh"
  --datasets "${DATASETS}"
  --gpus "${GPU_LIST}"
  --catalog-profile "${CATALOG_PROFILE}"
  --max-evals "${MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --phase-prefix "${PHASE_PREFIX}"
)
if [ "${LOG_WANDB}" = "true" ]; then
  cmd+=(--log-wandb)
fi
if [ "${DRY_RUN}" = "true" ]; then
  cmd+=(--dry-run)
fi

printf '[RR Rule Sanity] '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
