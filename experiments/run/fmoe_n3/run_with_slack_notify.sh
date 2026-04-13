#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_with_slack_notify.sh [--on|--off] [--title TITLE] [--note TEXT] -- <command> [args...]
  run_with_slack_notify.sh [--on|--off] [--title TITLE] [--note TEXT] <command> [args...]

Examples:
  ./run_with_slack_notify.sh --on python experiments/run/fmoe_n3/run_phase8_router_wrapper_diag.py
  ./run_with_slack_notify.sh --on sh experiments/run/fmoe_n3/phase_8_router_wrapper_diag.sh
  SLACK_NOTIFY=1 ./run_with_slack_notify.sh python your_script.py

Notes:
  - Notification is OFF by default.
  - Enable with --on or SLACK_NOTIFY=1.
  - Set SLACK_WEBHOOK_URL in env or experiments/run/fmoe_n3/.env.slack.
  - Add a short custom message with --note or SLACK_NOTIFY_NOTE.
USAGE
}

infer_slack_defaults() {
  local -n _cmd_ref=$1
  local current_title="$2"
  local current_note="$3"
  local inferred_title="${current_title}"
  local inferred_note="${current_note}"
  local inferred_total_runs="${SLACK_NOTIFY_TOTAL_RUNS:-}"

  local script_path=""
  if [[ ${#_cmd_ref[@]} -ge 2 && ( "${_cmd_ref[0]}" == "bash" || "${_cmd_ref[0]}" == "sh" ) ]]; then
    script_path="${_cmd_ref[1]}"
  elif [[ ${#_cmd_ref[@]} -ge 1 ]]; then
    script_path="${_cmd_ref[0]}"
  fi

  local script_name=""
  if [[ -n "${script_path}" ]]; then
    script_name="$(basename "${script_path}")"
  fi

  if [[ "${current_title}" == "FMoE Run" && -n "${script_name}" ]]; then
    case "${script_name}" in
      phase_14_retail_wrapper_recovery_8h.sh)
        inferred_title="Retail Wrapper Recovery 8H"
        ;;
      stageA.sh)
        inferred_title="Transfer Learning StageA"
        ;;
      *.sh)
        inferred_title="${script_name%.sh}"
        ;;
    esac
  fi

  if [[ -z "${current_note}" && "${script_name}" == "phase_14_retail_wrapper_recovery_8h.sh" ]]; then
    inferred_note="retail_rocket A8/A10/A11/A12 x 8 hparams"
  fi
  if [[ -z "${current_note}" && "${script_name}" == "stageA.sh" ]]; then
    inferred_note="4 dataset pairs with source/target transfer sweeps"
  fi

  if [[ -z "${inferred_total_runs}" && "${script_name}" == "phase_14_retail_wrapper_recovery_8h.sh" ]]; then
    inferred_total_runs="28"
  fi
  if [[ -z "${inferred_total_runs}" && "${script_name}" == "stageA.sh" ]]; then
    inferred_total_runs="72"
  fi

  title="${inferred_title}"
  note="${inferred_note}"
  export SLACK_NOTIFY_TOTAL_RUNS="${inferred_total_runs}"
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

  if [[ -n "${interrupted_by}" ]]; then
    text="${text}
interrupted_by=${interrupted_by}"
  fi

  if [[ "${verbose}" == "1" ]]; then
    text="${text}
user_host=${user_host}
cwd=${cwd}
cmd=${command_str}"
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
title="${SLACK_NOTIFY_TITLE:-FMoE Run}"
note="${SLACK_NOTIFY_NOTE:-}"
verbose="${SLACK_NOTIFY_VERBOSE:-0}"

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
command_str="$(printf "%q " "${cmd[@]}")"
infer_slack_defaults cmd "${title}" "${note}"

start_epoch="$(date +%s)"
start_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
user_host="${USER:-unknown}@$(hostname)"
cwd="$(pwd)"
child_pid=""
child_pgid=""

on_interrupt() {
  local sig="$1"
  trap - INT TERM

  local exit_code=130
  if [[ "${sig}" == "TERM" ]]; then
    exit_code=143
  fi

  if [[ -n "${child_pgid}" ]]; then
    # Stop the entire command process-group so no queued follow-up jobs launch.
    kill -TERM -- "-${child_pgid}" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-${child_pgid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  elif [[ -n "${child_pid}" ]]; then
    kill -s "${sig}" "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi

  local end_epoch
  local end_kst
  local elapsed
  end_epoch="$(date +%s)"
  end_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
  elapsed="$((end_epoch - start_epoch))"
  send_end_notification "${exit_code}" "${end_kst}" "${elapsed}" "${sig}"
  exit "${exit_code}"
}

trap 'on_interrupt INT' INT
trap 'on_interrupt TERM' TERM

export SLACK_NOTIFY="${notify}"
export SLACK_NOTIFY_TITLE="${title}"
export SLACK_NOTIFY_NOTE="${note}"
export SLACK_NOTIFY_VERBOSE="${verbose}"

if [[ "${notify}" == "1" ]]; then
  if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
    echo "Slack notify skipped: SLACK_WEBHOOK_URL is not set."
  else
    start_text=":rocket: [${title}] STARTED
started_kst=${start_kst}"
    if [[ -n "${SLACK_NOTIFY_TOTAL_RUNS:-}" ]]; then
      start_text="${start_text}
total_runs=${SLACK_NOTIFY_TOTAL_RUNS}"
    fi
    if [[ -n "${note}" ]]; then
      start_text="${start_text}
note=${note}"
    fi
    if [[ "${verbose}" == "1" ]]; then
      start_text="${start_text}
user_host=${user_host}
cwd=${cwd}
cmd=${command_str}"
    fi
    send_slack "${start_text}"
  fi
fi

# Run in a dedicated session/process-group so Ctrl+C can terminate all descendants.
setsid "${cmd[@]}" &
child_pid=$!
child_pgid="${child_pid}"
wait "${child_pid}"
exit_code=$?

end_epoch="$(date +%s)"
end_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"
elapsed="$((end_epoch - start_epoch))"
send_end_notification "${exit_code}" "${end_kst}" "${elapsed}"

exit "${exit_code}"
