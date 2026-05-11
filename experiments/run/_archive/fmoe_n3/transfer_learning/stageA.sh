#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/../common/run_metadata.sh"

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi
PYTHON_BIN="${RUN_PYTHON_BIN:-python}"

GPU_LIST="${GPU_LIST:-0,1,2,3}"
ARCHITECTURES="${ARCHITECTURES:-A12}"
PAIRS="${PAIRS:-kuairec_to_lastfm,lastfm_to_kuairec,amazon_to_retail,foursquare_to_movielens}"
SEEDS="${SEEDS:-1}"

SOURCE_ONLY="${SOURCE_ONLY:-false}"
TARGET_ONLY="${TARGET_ONLY:-false}"
DRY_RUN="${DRY_RUN:-false}"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --architectures) ARCHITECTURES="$2"; shift 2 ;;
    --pairs) PAIRS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --source-only) SOURCE_ONLY="true"; shift ;;
    --target-only) TARGET_ONLY="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

csv_count() {
  local raw="$1"
  local -a items=()
  IFS=',' read -r -a items <<< "${raw}"
  echo "${#items[@]}"
}

ARCH_COUNT="$(csv_count "${ARCHITECTURES}")"
PAIR_COUNT="$(csv_count "${PAIRS}")"

PER_PAIR_RUNS=26
if [ "${SOURCE_ONLY}" = "true" ]; then
  PER_PAIR_RUNS=2
elif [ "${TARGET_ONLY}" = "true" ]; then
  PER_PAIR_RUNS=24
fi

TOTAL_RUNS="$(( ARCH_COUNT * PAIR_COUNT * PER_PAIR_RUNS ))"

export SLACK_NOTIFY_TOTAL_RUNS="${SLACK_NOTIFY_TOTAL_RUNS:-${TOTAL_RUNS}}"
export SLACK_NOTIFY_SCOPE_LABEL="${SLACK_NOTIFY_SCOPE_LABEL:-transfer-learning-stagea}"
export SLACK_NOTIFY_NOTE="${SLACK_NOTIFY_NOTE:-StageA transfer sweep (${ARCHITECTURES}; ${PAIRS})}"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/stageA.py"
  --gpus "${GPU_LIST}"
  --architectures "${ARCHITECTURES}"
  --pairs "${PAIRS}"
  --seeds "${SEEDS}"
)

if [ "${SOURCE_ONLY}" = "true" ]; then
  CMD+=(--source-only)
fi
if [ "${TARGET_ONLY}" = "true" ]; then
  CMD+=(--target-only)
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[Transfer Learning StageA] total_runs=${TOTAL_RUNS} gpus=${GPU_LIST} architectures=${ARCHITECTURES} pairs=${PAIRS}"
run_echo_cmd "${CMD[@]}"
"${CMD[@]}"
