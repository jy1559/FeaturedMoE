#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_with_slack_notify.sh [--on|--off] [--title TITLE] [--note TEXT] -- <command> [args...]
  run_with_slack_notify.sh [--on|--off] [--title TITLE] [--note TEXT] <command> [args...]

Examples:
  ./run_with_slack_notify.sh --on bash experiments/run/baseline_2/run_all_stages.sh
  SLACK_NOTIFY=1 ./run_with_slack_notify.sh python experiments/run/baseline_2/run_staged_tuning.py --stage A

Notes:
  - Notification is OFF by default.
  - Enable with --on or SLACK_NOTIFY=1.
  - Set SLACK_WEBHOOK_URL in env or experiments/run/baseline_2/.env.slack.
USAGE
}

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/}"
  printf "%s" "${s}"
}

send_slack() {
  local text="$1"
  local payload
  payload="{\"text\":\"$(json_escape "${text}")\"}"
  if ! curl -fsS -X POST -H "Content-type: application/json" --data "${payload}" "${SLACK_WEBHOOK_URL}" >/dev/null; then
    echo "Slack notify failed to send."
  fi
}

send_end_notification() {
  local exit_code="$1"
  local end_kst="$2"
  local elapsed="$3"
  local interrupted_by="${4:-}"

  if [[ "${notify}" != "1" || -z "${SLACK_WEBHOOK_URL:-}" ]]; then
    return
  fi

  local status_text="FAILED"
  local icon=":x:"
  if [[ "${exit_code}" -eq 0 ]]; then
    status_text="SUCCESS"
    icon=":white_check_mark:"
  fi
  if [[ -n "${interrupted_by}" ]]; then
    status_text="TERMINATED"
    icon=":warning:"
  fi

  local text="${icon} [${title}] ${status_text}
elapsed=${elapsed}s
started_kst=${start_kst}
finished_kst=${end_kst}"

  if [[ -n "${note}" ]]; then
    text="${text}
note=${note}"
  fi
  send_slack "${text}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

notify="${SLACK_NOTIFY:-0}"
title="${SLACK_NOTIFY_TITLE:-Baseline2 Run}"
note="${SLACK_NOTIFY_NOTE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --on)
      notify=1
      shift
      ;;
    --off)
      notify=0
      shift
      ;;
    --title)
      if [[ $# -lt 2 ]]; then
        echo "--title requires a value."
        exit 2
      fi
      title="$2"
      shift 2
      ;;
    --note)
      if [[ $# -lt 2 ]]; then
        echo "--note requires a value."
        exit 2
      fi
      note="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

cmd=("$@")
start_epoch="$(date +%s)"
start_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
child_pid=""

on_interrupt() {
  local sig="$1"
  trap - INT TERM
  local exit_code=130
  if [[ "${sig}" == "TERM" ]]; then
    exit_code=143
  fi

  if [[ -n "${child_pid}" ]]; then
    kill -s "${sig}" "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi

  local end_epoch end_kst elapsed
  end_epoch="$(date +%s)"
  end_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
  elapsed="$((end_epoch - start_epoch))"
  send_end_notification "${exit_code}" "${end_kst}" "${elapsed}" "${sig}"
  exit "${exit_code}"
}

trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM

if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  send_slack ":rocket: [${title}] STARTED
started_kst=${start_kst}
cmd=$(printf "%q " "${cmd[@]}")"
fi

"${cmd[@]}" &
child_pid=$!
wait "${child_pid}"
exit_code=$?

end_epoch="$(date +%s)"
end_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
elapsed="$((end_epoch - start_epoch))"
send_end_notification "${exit_code}" "${end_kst}" "${elapsed}"
exit "${exit_code}"

