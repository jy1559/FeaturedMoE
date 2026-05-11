#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHASE11_SCRIPT="${SCRIPT_DIR}/phase_11_stage_semantics.sh"
PHASE12_SCRIPT="${SCRIPT_DIR}/phase_12_layout_composition.sh"
PHASE13_SCRIPT="${SCRIPT_DIR}/phase_13_feature_sanity.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3,4,5,6,7"
MAX_EVALS="20"
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"

EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2] [--gpus 0,1,2,3,4,5,6,7] [--max-evals 20]
          [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [extra args passed through to phase_11/12/13]

Example:
  SLACK_NOTIFY=1 bash run_with_slack_notify.sh -- \\
    bash phase_11_12_13_sequential.sh --gpus 0,1,2,3,4,5,6,7 --max-evals 20
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --max-evals)
      MAX_EVALS="$2"
      shift 2
      ;;
    --resume-from-logs)
      RESUME_FROM_LOGS="true"
      shift
      ;;
    --no-resume-from-logs)
      RESUME_FROM_LOGS="false"
      shift
      ;;
    --verify-logging)
      VERIFY_LOGGING="true"
      shift
      ;;
    --no-verify-logging)
      VERIFY_LOGGING="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

run_phase() {
  local phase_name="$1"
  local phase_script="$2"
  shift 2

  local cmd=(
    bash "${phase_script}"
    --datasets "${DATASETS}"
    --gpus "${GPU_LIST}"
    --max-evals "${MAX_EVALS}"
  )

  if [ "${RESUME_FROM_LOGS}" = "true" ]; then
    cmd+=(--resume-from-logs)
  else
    cmd+=(--no-resume-from-logs)
  fi

  if [ "${VERIFY_LOGGING}" = "true" ]; then
    cmd+=(--verify-logging)
  else
    cmd+=(--no-verify-logging)
  fi

  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  echo "[Start] ${phase_name}"
  echo "[Cmd] ${cmd[*]}"
  "${cmd[@]}"
  echo "[Done] ${phase_name}"
}

run_phase "phase11_stage_semantics" "${PHASE11_SCRIPT}"
run_phase "phase12_layout_composition" "${PHASE12_SCRIPT}"
run_phase "phase13_feature_sanity" "${PHASE13_SCRIPT}"

echo "[All Done] phase11 -> phase12 -> phase13 completed"
