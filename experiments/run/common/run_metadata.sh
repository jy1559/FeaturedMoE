#!/usr/bin/env bash
set -euo pipefail

run_root_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # run/common -> experiments/run
  cd "${script_dir}/.." && pwd
}

run_experiments_dir() {
  local run_dir
  run_dir="$(run_root_dir)"
  cd "${run_dir}/.." && pwd
}

run_repo_root() {
  local exp_dir
  exp_dir="$(run_experiments_dir)"
  cd "${exp_dir}/.." && pwd
}

run_timestamp() {
  date +%Y%m%d_%H%M%S_%3N
}

run_today() {
  date +%Y%m%d
}

run_sanitize() {
  local s="${1:-}"
  s="${s// /_}"
  s="${s//\//_}"
  s="${s//:/_}"
  s="${s//,/__}"
  s="${s//[^a-zA-Z0-9._-]/_}"
  printf '%s' "$s"
}

run_ensure_dir() {
  mkdir -p "$1"
}

run_cmd_str() {
  local q
  printf -v q '%q ' "$@"
  printf '%s' "${q% }"
}

run_echo_cmd() {
  echo "[RUN] $(run_cmd_str "$@")"
}

run_artifacts_dir() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/artifacts"
}

run_logs_root() {
  local art_dir
  art_dir="$(run_artifacts_dir)"
  echo "${art_dir}/logs"
}

run_log_dir() {
  local group="${1:?group required}"
  local logs_root
  logs_root="$(run_logs_root)"
  echo "${logs_root}/${group}"
}

run_legacy_log_root() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/log"
}

run_results_root() {
  local art_dir
  art_dir="$(run_artifacts_dir)"
  echo "${art_dir}/results"
}

run_results_dir() {
  local group="${1:?group required}"
  local root
  root="$(run_results_root)"
  echo "${root}/${group}"
}

run_legacy_results_root() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/hyperopt_results"
}

run_timeline_dir() {
  local art_dir
  art_dir="$(run_artifacts_dir)"
  echo "${art_dir}/timeline"
}

run_inventory_dir() {
  local art_dir
  art_dir="$(run_artifacts_dir)"
  echo "${art_dir}/inventory"
}

run_tracker_script() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/common/experiment_tracker.py"
}

run_python_bin() {
  if [ -n "${RUN_PYTHON_BIN:-}" ]; then
    echo "${RUN_PYTHON_BIN}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  echo "python3"  # fallback; command-not-found will show a clear error.
}

run_make_log_path() {
  local group="${1:?group}"
  local axis="${2:?axis}"
  local dataset="${3:?dataset}"
  local model="${4:?model}"
  local gpu="${5:-na}"
  local phase="${6:-PNA}"

  local d tag axis_tag phase_bucket
  axis_tag="$(run_sanitize "${axis}")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  [ -z "$phase_bucket" ] && phase_bucket="PNA"

  d="$(run_log_dir "$group")/${axis_tag}/${phase_bucket}/$(run_sanitize "${dataset}")/$(run_sanitize "${model}")"
  run_ensure_dir "$d"

  tag="$(run_sanitize "${dataset}")_$(run_sanitize "${model}")_$(run_sanitize "${axis}")_${phase}_gpu${gpu}_$(run_timestamp)"
  echo "${d}/${tag}.log"
}

run_tracker_start() {
  "$(run_python_bin)" "$(run_tracker_script)" start "$@"
}

run_tracker_end() {
  "$(run_python_bin)" "$(run_tracker_script)" end "$@"
}

run_export_runtime_env() {
  local exp_dir
  local run_dir
  local art_dir
  local results_root
  local timeline_dir
  local inventory_dir

  exp_dir="$(run_experiments_dir)"
  run_dir="$(run_root_dir)"
  art_dir="$(run_artifacts_dir)"
  results_root="$(run_results_root)"
  timeline_dir="$(run_timeline_dir)"
  inventory_dir="$(run_inventory_dir)"

  export PYTHONPATH="${exp_dir}"
  export RUN_ARTIFACTS_DIR="${RUN_ARTIFACTS_DIR:-${art_dir}}"
  export RUN_LOGS_DIR="${RUN_LOGS_DIR:-${art_dir}/logs}"
  export HYPEROPT_RESULTS_DIR="${HYPEROPT_RESULTS_DIR:-${results_root}}"
  export RUN_TIMELINE_DIR="${RUN_TIMELINE_DIR:-${timeline_dir}}"
  export RUN_INVENTORY_DIR="${RUN_INVENTORY_DIR:-${inventory_dir}}"

  run_ensure_dir "${RUN_ARTIFACTS_DIR}"
  run_ensure_dir "${RUN_LOGS_DIR}"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/baseline"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hir"
  run_ensure_dir "${RUN_TIMELINE_DIR}"
  run_ensure_dir "${RUN_INVENTORY_DIR}"

  # Keep legacy folders for read fallback compatibility.
  run_ensure_dir "${run_dir}/log"
  run_ensure_dir "${run_dir}/hyperopt_results"
}
