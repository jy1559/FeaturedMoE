#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_all_stages.sh [--slack-on|--slack-off] [--title TITLE] [--note TEXT] [--] [extra args for run_staged_tuning.py]

Examples:
  bash experiments/run/baseline_2/run_all_stages.sh
  GPU_LIST=0,1 bash experiments/run/baseline_2/run_all_stages.sh -- --datasets amazon_beauty,lastfm0.03 --models sasrec,tisasrec
  SLACK_NOTIFY=1 bash experiments/run/baseline_2/run_all_stages.sh --slack-on --title "Baseline2 ABCD"

Notes:
  - 기본은 3개 모델(sasrec,tisasrec,gru4rec) x 6개 데이터셋 실행.
  - GPU는 GPU_LIST 환경변수(예: GPU_LIST=0,1,2) 또는 --gpus override 사용.
  - Slack은 기본 OFF. --slack-on 또는 SLACK_NOTIFY=1로 활성화.
  - .env.slack는 experiments/run/baseline_2/.env.slack 를 자동 로드.
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
  curl -fsS -X POST -H "Content-type: application/json" --data "${payload}" "${SLACK_WEBHOOK_URL}" >/dev/null || true
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY_BIN="${RUN_PYTHON_BIN:-/venv/FMoE/bin/python}"
LOCAL_ENV_FILE="${SCRIPT_DIR}/.env.slack"

if [[ -f "${LOCAL_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${LOCAL_ENV_FILE}"
fi

notify="${SLACK_NOTIFY:-0}"
title="${SLACK_NOTIFY_TITLE:-Baseline2 ABCD}"
note="${SLACK_NOTIFY_NOTE:-}"
verbose="${SLACK_NOTIFY_VERBOSE:-0}"

extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --slack-on)
      notify=1
      shift
      ;;
    --slack-off)
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
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

cd "${REPO_ROOT}"

datasets_default="KuaiRecLargeStrictPosV2_0.2,lastfm0.03,amazon_beauty,foursquare,movielens1m,retail_rocket"
models_default="sasrec,tisasrec,gru4rec"
gpus_default="${GPU_LIST:-0}"
runtime_seed_default="${RUNTIME_SEED:-1}"
final_seeds_default="${FINAL_SEEDS:-1,2,3}"

export SLACK_NOTIFY="${notify}"
export SLACK_NOTIFY_TITLE="${title}"
export SLACK_NOTIFY_NOTE="${note}"
export SLACK_NOTIFY_VERBOSE="${verbose}"

start_epoch="$(date +%s)"
start_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"

if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  start_text=":rocket: [${title}] STARTED
started_kst=${start_kst}
datasets_default=${datasets_default}
models_default=${models_default}
gpus=${gpus_default}
runtime_seed=${runtime_seed_default}
final_seeds=${final_seeds_default}"
  if [[ -n "${note}" ]]; then
    start_text="${start_text}
note=${note}"
  fi
  if [[ "${verbose}" == "1" ]]; then
    start_text="${start_text}
cwd=$(pwd)
extra_args=$(printf '%q ' "${extra_args[@]}")"
  fi
  send_slack "${start_text}"
fi

run_stage() {
  local stage="$1"
  local stage_start stage_end stage_elapsed
  stage_start="$(date +%s)"
  "${PY_BIN}" "${SCRIPT_DIR}/run_staged_tuning.py" \
    --stage "${stage}" \
    --track baseline_2 \
    --axis ABCD_v1 \
    --budget-profile balanced \
    --models "${models_default}" \
    --datasets "${datasets_default}" \
    --gpus "${gpus_default}" \
    --runtime-seed "${runtime_seed_default}" \
    --final-seeds "${final_seeds_default}" \
    --resume-from-logs \
    "${extra_args[@]}"
  stage_end="$(date +%s)"
  stage_elapsed="$((stage_end - stage_start))"
  if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then
    send_slack ":white_check_mark: [${title}] stage ${stage} done (elapsed=${stage_elapsed}s)"
  fi
}

current_stage="N/A"
trap 'if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then send_slack ":warning: [${title}] interrupted (stage=${current_stage})"; fi' INT TERM
trap 'if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then send_slack ":x: [${title}] failed at stage=${current_stage}"; fi' ERR

current_stage="A"
run_stage "A"
current_stage="B"
run_stage "B"
current_stage="C"
run_stage "C"
current_stage="D"
run_stage "D"

end_epoch="$(date +%s)"
elapsed="$((end_epoch - start_epoch))"
end_kst="$(TZ=Asia/Seoul date +"%Y-%m-%d %H:%M:%S KST")"

if [[ "${notify}" == "1" && -n "${SLACK_WEBHOOK_URL:-}" ]]; then
  done_text=":white_check_mark: [${title}] ALL DONE
started_kst=${start_kst}
finished_kst=${end_kst}
elapsed=${elapsed}s"
  if [[ -n "${note}" ]]; then
    done_text="${done_text}
note=${note}"
  fi
  send_slack "${done_text}"
fi
