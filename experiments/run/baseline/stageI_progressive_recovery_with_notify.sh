#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITLE="${SLACK_NOTIFY_TITLE:-Baseline StageI}"
NOTE="${SLACK_NOTIFY_NOTE:-progressive recovery pass1->2->3}"

export SLACK_NOTIFY=1
export SLACK_NOTIFY_PROGRESS_STEP="${SLACK_NOTIFY_PROGRESS_STEP:-10}"
export SLACK_NOTIFY_SCOPE_LABEL="${SLACK_NOTIFY_SCOPE_LABEL:-baseline-stagei}"

exec "${SCRIPT_DIR}/run_with_slack_notify.sh" \
  --on \
  --title "${TITLE}" \
  --note "${NOTE}" \
  -- bash "${SCRIPT_DIR}/stageI_progressive_recovery.sh" "$@"
