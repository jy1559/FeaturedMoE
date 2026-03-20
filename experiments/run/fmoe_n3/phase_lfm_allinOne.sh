#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASET="lastfm0.03"
GPU_LIST="0,1,2,3"
FROM_PHASE="0"
TO_PHASE="2"
P0_MAX_EVALS="10"
P1_MAX_EVALS="10"
P2_MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="22000"
ANCHORS="AN_S,AN_M,AN_L"
SUITES="layout_suite,router_suite,feature_embed_suite,feature_family_mask_suite,topk_suite,expert_scale_suite,seq_len_suite,aux_balance_suite,aux_spec_suite,residual_suite"
CATEGORY_FILTER=""
ONLY_RUN_PHASE=""
MANIFEST_OUT=""
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<USAGE
Usage: $0 [--dataset lastfm0.03] [--gpus 0,1,2,3]
          [--from-phase 0] [--to-phase 2]
          [--p0-max-evals 10] [--p1-max-evals 10] [--p2-max-evals 10]
          [--tune-epochs 100] [--tune-patience 10] [--seed-base 22000]
          [--anchors AN_S,AN_M,AN_L]
          [--suites router_suite,feature_embed_suite,topk_suite]
          [--category router_suite,topk_suite] [--only P0_AN_S_BASE,P1_AN_M_BASE]
          [--manifest-out path] [--dry-run]
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --from-phase)
      FROM_PHASE="$2"
      shift 2
      ;;
    --to-phase)
      TO_PHASE="$2"
      shift 2
      ;;
    --p0-max-evals)
      P0_MAX_EVALS="$2"
      shift 2
      ;;
    --p1-max-evals)
      P1_MAX_EVALS="$2"
      shift 2
      ;;
    --p2-max-evals)
      P2_MAX_EVALS="$2"
      shift 2
      ;;
    --tune-epochs)
      TUNE_EPOCHS="$2"
      shift 2
      ;;
    --tune-patience)
      TUNE_PATIENCE="$2"
      shift 2
      ;;
    --seed-base)
      SEED_BASE="$2"
      shift 2
      ;;
    --anchors)
      ANCHORS="$2"
      shift 2
      ;;
    --suites)
      SUITES="$2"
      shift 2
      ;;
    --category)
      CATEGORY_FILTER="$2"
      shift 2
      ;;
    --only)
      ONLY_RUN_PHASE="$2"
      shift 2
      ;;
    --manifest-out)
      MANIFEST_OUT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi

PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"
CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/run_phase_lfm_allinone.py"
  --dataset "${DATASET}"
  --gpus "${GPU_LIST}"
  --from-phase "${FROM_PHASE}"
  --to-phase "${TO_PHASE}"
  --p0-max-evals "${P0_MAX_EVALS}"
  --p1-max-evals "${P1_MAX_EVALS}"
  --p2-max-evals "${P2_MAX_EVALS}"
  --tune-epochs "${TUNE_EPOCHS}"
  --tune-patience "${TUNE_PATIENCE}"
  --seed-base "${SEED_BASE}"
  --anchors "${ANCHORS}"
)

if [ -n "${SUITES}" ]; then
  CMD+=(--suites "${SUITES}")
fi
if [ -n "${CATEGORY_FILTER}" ]; then
  CMD+=(--category "${CATEGORY_FILTER}")
fi
if [ -n "${ONLY_RUN_PHASE}" ]; then
  CMD+=(--only "${ONLY_RUN_PHASE}")
fi
if [ -n "${MANIFEST_OUT}" ]; then
  CMD+=(--manifest-out "${MANIFEST_OUT}")
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi

run_echo_cmd "${CMD[@]}"
exec "${CMD[@]}"
