#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket"
MODELS="sasrec,gru4rec,tisasrec,duorec,sigma,bsarec,fearec,fame"
GPU_LIST="0,1,2,3,4,5,6,7"
SEEDS="1"
SEED_BASE="230000"

MANIFEST_OUT=""
RESUME_FROM_LOGS="true"
VERIFY_LOGGING="true"
DRY_RUN="${DRY_RUN:-false}"
SMOKE_TEST="false"
SMOKE_MAX_RUNS="8"
FAST_SCREEN="false"
FOLLOWUP_RESCUE="false"
FOLLOWUP_FULL="false"
OVERNIGHT_AUTO="false"
PROMOTE_FROM_LATEST_RESCUE="false"
PROMOTE_TOPK="4"
PROMOTE_MIN_RATIO="0.95"
UNDERPERFORM_SCREEN="false"
LOWPERFORM2_SCREEN="false"
WEAK_RATIO="0.90"
STRONG_RATIO="0.75"
LOWPERFORM2_RATIO="0.80"
LOWPERFORM2_ULTRA_RATIO="0.50"
LOWPERFORM2_REGULAR_MAX_EVALS="15"
LOWPERFORM2_WIDE_MAX_EVALS="50"
CANDIDATE_TAG=""

usage() {
  cat <<USAGE
Usage: $0 [--datasets ...] [--models ...] [--gpus 0,1,2,3,4,5,6,7]
          [--seeds 1,2,3] [--seed-base 230000]
          [--manifest-out path] [--resume-from-logs|--no-resume-from-logs]
          [--verify-logging|--no-verify-logging]
          [--dry-run] [--smoke-test] [--smoke-max-runs 8] [--fast-screen]
          [--followup-rescue] [--followup-full] [--overnight-auto]
          [--underperform-screen] [--weak-ratio 0.90] [--strong-ratio 0.75]
          [--lowperform2-screen] [--lowperform2-ratio 0.80] [--lowperform2-ultra-ratio 0.50]
          [--lowperform2-regular-max-evals 15] [--lowperform2-wide-max-evals 50]
          [--candidate-tag TAG]
          [--promote-from-latest-rescue] [--promote-topk 4] [--promote-min-ratio 0.95]
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
    --followup-rescue) FOLLOWUP_RESCUE="true"; shift ;;
    --followup-full) FOLLOWUP_FULL="true"; shift ;;
    --overnight-auto) OVERNIGHT_AUTO="true"; shift ;;
    --underperform-screen) UNDERPERFORM_SCREEN="true"; shift ;;
    --lowperform2-screen) LOWPERFORM2_SCREEN="true"; shift ;;
    --weak-ratio) WEAK_RATIO="$2"; shift 2 ;;
    --strong-ratio) STRONG_RATIO="$2"; shift 2 ;;
    --lowperform2-ratio) LOWPERFORM2_RATIO="$2"; shift 2 ;;
    --lowperform2-ultra-ratio) LOWPERFORM2_ULTRA_RATIO="$2"; shift 2 ;;
    --lowperform2-regular-max-evals) LOWPERFORM2_REGULAR_MAX_EVALS="$2"; shift 2 ;;
    --lowperform2-wide-max-evals) LOWPERFORM2_WIDE_MAX_EVALS="$2"; shift 2 ;;
    --candidate-tag) CANDIDATE_TAG="$2"; shift 2 ;;
    --promote-from-latest-rescue) PROMOTE_FROM_LATEST_RESCUE="true"; shift ;;
    --promote-topk) PROMOTE_TOPK="$2"; shift 2 ;;
    --promote-min-ratio) PROMOTE_MIN_RATIO="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"

build_cmd() {
  local mode="$1"
  local -a cmd=(
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
    cmd+=(--manifest-out "${MANIFEST_OUT}")
  fi
  if [ -n "${CANDIDATE_TAG}" ]; then
    cmd+=(--candidate-tag "${CANDIDATE_TAG}")
  fi
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
  if [ "${DRY_RUN}" = "true" ]; then
    cmd+=(--dry-run)
  fi
  if [ "${SMOKE_TEST}" = "true" ]; then
    cmd+=(--smoke-test)
  fi
  case "${mode}" in
    fast) cmd+=(--fast-screen) ;;
    rescue) cmd+=(--followup-rescue) ;;
    full) cmd+=(--followup-full) ;;
    full_auto) cmd+=(--followup-full --promote-from-latest-rescue --promote-topk "${PROMOTE_TOPK}" --promote-min-ratio "${PROMOTE_MIN_RATIO}") ;;
    under) cmd+=(--underperform-screen --weak-ratio "${WEAK_RATIO}" --strong-ratio "${STRONG_RATIO}") ;;
    low2) cmd+=(--lowperform2-screen --lowperform2-ratio "${LOWPERFORM2_RATIO}" --lowperform2-ultra-ratio "${LOWPERFORM2_ULTRA_RATIO}" --lowperform2-regular-max-evals "${LOWPERFORM2_REGULAR_MAX_EVALS}" --lowperform2-wide-max-evals "${LOWPERFORM2_WIDE_MAX_EVALS}") ;;
  esac
  printf '%s\n' "${cmd[@]}"
}

run_mode() {
  local mode="$1"
  mapfile -t CMD < <(build_cmd "${mode}")
  run_echo_cmd "${CMD[@]}"
  "${CMD[@]}"
}

if [ "${OVERNIGHT_AUTO}" = "true" ]; then
  run_mode "rescue"
  run_mode "full_auto"
else
  mode="base"
  if [ "${FAST_SCREEN}" = "true" ]; then
    mode="fast"
  fi
  if [ "${UNDERPERFORM_SCREEN}" = "true" ]; then
    mode="under"
  fi
  if [ "${LOWPERFORM2_SCREEN}" = "true" ]; then
    mode="low2"
  fi
  if [ "${FOLLOWUP_RESCUE}" = "true" ]; then
    mode="rescue"
  fi
  if [ "${FOLLOWUP_FULL}" = "true" ] && [ "${PROMOTE_FROM_LATEST_RESCUE}" = "true" ]; then
    mode="full_auto"
  elif [ "${FOLLOWUP_FULL}" = "true" ]; then
    mode="full"
  fi
  run_mode "${mode}"
fi

echo "[All Done] baseline StageH targeted recovery completed: ${DATASETS}"
