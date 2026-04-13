#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FMOE_RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAP="${FMOE_RUN_DIR}/run_with_slack_notify.sh"

COMMON_ARGS=("$@")

export SLACK_NOTIFY_SCOPE_LABEL="${SLACK_NOTIFY_SCOPE_LABEL:-A12 Final Tuning}"

SLACK_NOTIFY_TOTAL_RUNS=33 "${WRAP}" --on --title "A12 Final Tuning Stage1" --note "family sweep (33 runs, broad families, discrete search)" -- \
  bash "${SCRIPT_DIR}/stage1_family_sweep.sh" "${COMMON_ARGS[@]}"

SLACK_NOTIFY_TOTAL_RUNS=20 "${WRAP}" --on --title "A12 Final Tuning Stage2" --note "dataset refinement (20 runs, winner-family local search)" -- \
  bash "${SCRIPT_DIR}/stage2_dataset_refine.sh" "${COMMON_ARGS[@]}"

SLACK_NOTIFY_TOTAL_RUNS=10 "${WRAP}" --on --title "A12 Final Tuning Stage3" --note "local polish (10 runs, narrow local search)" -- \
  bash "${SCRIPT_DIR}/stage3_local_polish.sh" "${COMMON_ARGS[@]}"
