#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
  echo "SLACK_WEBHOOK_URL is not set."
  echo "Set it in shell env or ${LOCAL_ENV_FILE}."
  exit 1
fi

curl -fsS -X POST \
  -H "Content-type: application/json" \
  --data '{"text":"실험 알림 테스트: 서버에서 Slack 연결 성공"}' \
  "${SLACK_WEBHOOK_URL}" >/dev/null

echo "Slack webhook test sent."
