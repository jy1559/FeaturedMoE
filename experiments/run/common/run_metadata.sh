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

run_dataset_tag() {
  local raw="${1:-}"
  local key
  key="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$key" in
    movielens1m) echo "ML1" ;;
    retail_rocket|retailrocket) echo "ReR" ;;
    foursquare) echo "FSQ" ;;
    lastfm0.3) echo "LF3" ;;
    lastfm0.03) echo "LF03" ;;
    kuairec0.3) echo "KU3" ;;
    kuairecsmall0.1) echo "KU01" ;;
    kuaireclargestrictposv2) echo "KUL" ;;
    kuaireclargestrictposv2_0.2) echo "KUL02" ;;
    amazon_beauty|amazonbeauty) echo "AMA" ;;
    *)
      key="$(printf '%s' "$raw" | tr -cd '[:alnum:]' | tr '[:lower:]' '[:upper:]')"
      if [ "${#key}" -ge 3 ]; then
        echo "${key:0:3}"
      elif [ -n "$key" ]; then
        printf '%-3s' "$key" | tr ' ' 'X'
      else
        echo "UNK"
      fi
      ;;
  esac
}

run_model_tag() {
  local raw="${1:-}"
  local key
  key="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$key" in
    *featuredmoe_protox*) echo "FMoEProtoX" ;;
    *featured_moe_individual*|*featuredmoe_individual*) echo "FMoEIndividual" ;;
    *featuredmoe_hir2*) echo "FMoEHiR2" ;;
    *featured_moe_v2_hir*|*featuredmoe_v2_hir*) echo "FMoEv2HiR" ;;
    *featured_moe_v3*|*featuredmoe_v3*) echo "FMoEv3" ;;
    *featured_moe_v4_distillation*|*featuredmoe_v4_distillation*) echo "FMoEv4D" ;;
    *featured_moe_hgr_v4*|*featuredmoe_hgr_v4*|*featuredmoe_hgrv4*|*featured_moe_hgrv4*) echo "FMoEHGRv4" ;;
    *featured_moe_hgr_v3*|*featuredmoe_hgr_v3*|*featuredmoe_hgrv3*|*featured_moe_hgrv3*) echo "FMoEHGRv3" ;;
    *featured_moe_hgr*|*featuredmoe_hgr*) echo "FMoEHGR" ;;
    *featuredmoe_hir*) echo "FMoEHiR" ;;
    *featuredmoe_v2*) echo "FMoEv2" ;;
    *featured_moe_n3*|*featuredmoe_n3*) echo "FMoEN3" ;;
    *featured_moe_n2*|*featuredmoe_n2*) echo "FMoEN2" ;;
    *featuredmoe*) echo "FMoE" ;;
    *)
      echo "$(run_sanitize "$raw")"
      ;;
  esac
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

run_model_report_script() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/common/model_experiment_report.py"
}

run_track_report_script() {
  local run_dir
  run_dir="$(run_root_dir)"
  echo "${run_dir}/common/track_experiment_report.py"
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

  local d tag axis_tag phase_bucket dataset_tag model_tag
  axis_tag="$(run_sanitize "${axis}")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  [ -z "$phase_bucket" ] && phase_bucket="PNA"
  dataset_tag="$(run_dataset_tag "${dataset}")"
  model_tag="$(run_model_tag "${model}")"

  d="$(run_log_dir "$group")/${axis_tag}/${phase_bucket}/${dataset_tag}/${model_tag}"
  run_ensure_dir "$d"

  tag="$(run_timestamp)_$(run_sanitize "${axis}")_$(run_sanitize "${phase}")"
  echo "${d}/${tag}.log"
}

run_make_phase_log_path() {
  local group="${1:?group}"
  local axis="${2:?axis}"
  local dataset="${3:?dataset}"
  local model="${4:?model}"
  local phase="${5:-PNA}"

  local d axis_tag phase_bucket dataset_tag model_tag short_phase base path n combo_prefix
  axis_tag="$(run_sanitize "${axis}")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  [ -z "$phase_bucket" ] && phase_bucket="PNA"
  dataset_tag="$(run_dataset_tag "${dataset}")"
  model_tag="$(run_model_tag "${model}")"

  d="$(run_log_dir "$group")/${axis_tag}/${phase_bucket}/${dataset_tag}/${model_tag}"
  run_ensure_dir "$d"

  short_phase="${phase#${phase%%_*}_}"
  [ -z "$short_phase" ] && short_phase="${phase}"
  base="$(run_sanitize "$short_phase")"
  [ -z "$base" ] && base="run"
  combo_prefix=""
  if [[ "$short_phase" =~ (^|_)C([0-9]+)($|_) ]]; then
    printf -v combo_prefix "%03d_" "$((10#${BASH_REMATCH[2]}))"
  fi
  base="${combo_prefix}${base}"

  path="${d}/${base}.log"
  if [ ! -e "$path" ]; then
    echo "$path"
    return 0
  fi

  n=2
  while :; do
    path="${d}/${base}_r${n}.log"
    if [ ! -e "$path" ]; then
      echo "$path"
      return 0
    fi
    n=$((n + 1))
  done
}

run_make_sequential_phase_log_path() {
  local group="${1:?group}"
  local axis="${2:?axis}"
  local dataset="${3:?dataset}"
  local model="${4:?model}"
  local phase="${5:-PNA}"

  local d axis_tag phase_bucket dataset_tag model_tag short_phase base lock_dir counter_file
  local next_id max_id path n prefix
  axis_tag="$(run_sanitize "${axis}")"
  phase_bucket="$(run_sanitize "${phase%%_*}")"
  [ -z "$phase_bucket" ] && phase_bucket="PNA"
  dataset_tag="$(run_dataset_tag "${dataset}")"
  model_tag="$(run_model_tag "${model}")"

  d="$(run_log_dir "$group")/${axis_tag}/${phase_bucket}/${dataset_tag}/${model_tag}"
  run_ensure_dir "$d"

  short_phase="${phase#${phase%%_*}_}"
  [ -z "$short_phase" ] && short_phase="${phase}"
  base="$(run_sanitize "$short_phase")"
  [ -z "$base" ] && base="run"

  lock_dir="${d}/.log_index_lock"
  counter_file="${d}/.log_index_counter"

  while ! mkdir "$lock_dir" 2>/dev/null; do
    sleep 0.02
  done

  if [ -f "$counter_file" ] && read -r next_id <"$counter_file" && [[ "$next_id" =~ ^[0-9]+$ ]]; then
    :
  else
    max_id=-1
    while IFS= read -r name; do
      if [[ "$name" =~ ^([0-9]{3})_.*\.log$ ]]; then
        n=$((10#${BASH_REMATCH[1]}))
        if [ "$n" -gt "$max_id" ]; then
          max_id="$n"
        fi
      fi
    done < <(find "$d" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort)
    next_id=$((max_id + 1))
  fi
  echo $((next_id + 1)) >"$counter_file"
  rmdir "$lock_dir"

  printf -v prefix "%03d" "$next_id"
  path="${d}/${prefix}_${base}.log"
  if [ ! -e "$path" ]; then
    echo "$path"
    return 0
  fi

  n=2
  while :; do
    path="${d}/${prefix}_${base}_r${n}.log"
    if [ ! -e "$path" ]; then
      echo "$path"
      return 0
    fi
    n=$((n + 1))
  done
}

run_tracker_start() {
  "$(run_python_bin)" "$(run_tracker_script)" start "$@"
}

run_tracker_end() {
  "$(run_python_bin)" "$(run_tracker_script)" end "$@"
}

run_update_model_report() {
  local track="${1:?track required}"
  local model="${2:?model required}"
  local model_dir="${3:?model_dir required}"
  local py_bin
  py_bin="$(run_python_bin)"

  if ! "$py_bin" "$(run_model_report_script)" \
    --track "$track" \
    --model "$model" \
    --model-dir "$model_dir"; then
    echo "[WARN] model experiment report update failed: track=${track}, model=${model}" >&2
  fi
}

run_update_track_report() {
  local track="${1:?track required}"
  local py_bin
  local logs_root
  local out_dir
  py_bin="$(run_python_bin)"
  logs_root="$(run_logs_root)"
  out_dir="${logs_root}/${track}"
  run_ensure_dir "${out_dir}"

  if ! "$py_bin" "$(run_track_report_script)" \
    --track "$track" \
    --output-csv "${out_dir}/experiment_overview.csv" \
    --output-md "${out_dir}/experiment_overview.md"; then
    echo "[WARN] track experiment report update failed: track=${track}" >&2
  fi
}

run_export_runtime_env() {
  local exp_dir
  local run_dir
  local art_dir
  local results_root
  local timeline_dir
  local inventory_dir
  local wandb_dir

  exp_dir="$(run_experiments_dir)"
  run_dir="$(run_root_dir)"
  art_dir="$(run_artifacts_dir)"
  results_root="$(run_results_root)"
  timeline_dir="$(run_timeline_dir)"
  inventory_dir="$(run_inventory_dir)"
  wandb_dir="${RUN_WANDB_DIR:-${art_dir}/wandb}"

  export PYTHONPATH="${exp_dir}"
  export RUN_ARTIFACTS_DIR="${RUN_ARTIFACTS_DIR:-${art_dir}}"
  export RUN_LOGS_DIR="${RUN_LOGS_DIR:-${art_dir}/logs}"
  export HYPEROPT_RESULTS_DIR="${HYPEROPT_RESULTS_DIR:-${results_root}}"
  export RUN_TIMELINE_DIR="${RUN_TIMELINE_DIR:-${timeline_dir}}"
  export RUN_INVENTORY_DIR="${RUN_INVENTORY_DIR:-${inventory_dir}}"
  export RUN_WANDB_DIR="${wandb_dir}"
  export WANDB_DIR="${WANDB_DIR:-${wandb_dir}}"

  run_ensure_dir "${RUN_ARTIFACTS_DIR}"
  run_ensure_dir "${RUN_LOGS_DIR}"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/baseline"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hir"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hgr"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hgr_v3"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hgr_v4"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_hir2"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_protox"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_v2"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_v4_distillation"
  run_ensure_dir "${HYPEROPT_RESULTS_DIR}/fmoe_rule"
  run_ensure_dir "${RUN_TIMELINE_DIR}"
  run_ensure_dir "${RUN_INVENTORY_DIR}"
  run_ensure_dir "${WANDB_DIR}"

  # Keep legacy folders for read fallback compatibility.
  run_ensure_dir "${run_dir}/log"
  run_ensure_dir "${run_dir}/hyperopt_results"
}
