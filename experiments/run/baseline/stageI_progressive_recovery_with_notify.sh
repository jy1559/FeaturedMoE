#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_NAME="step1"
ARGS=("$@")
idx=0
while [ $idx -lt $# ]; do
  if [ "${ARGS[$idx]}" = "--step" ] && [ $((idx + 1)) -lt $# ]; then
    STEP_NAME="${ARGS[$((idx + 1))]}"
  fi
  idx=$((idx + 1))
done

TITLE="${SLACK_NOTIFY_TITLE:-Baseline StageI ${STEP_NAME}}"
NOTE="${SLACK_NOTIFY_NOTE:-progressive recovery ${STEP_NAME}}"

export SLACK_NOTIFY=1
export SLACK_NOTIFY_PROGRESS_STEP="${SLACK_NOTIFY_PROGRESS_STEP:-10}"
export SLACK_NOTIFY_SCOPE_LABEL="${SLACK_NOTIFY_SCOPE_LABEL:-baseline-stagei}"

exec "${SCRIPT_DIR}/run_with_slack_notify.sh" \
  --on \
  --title "${TITLE}" \
  --note "${NOTE}" \
  -- bash "${SCRIPT_DIR}/stageI_progressive_recovery.sh" "${ARGS[@]}"
