#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

MODE=""
GPU_LIST="0,1,2,3"
PHASE="P0"
PHASE_SET="false"
MAX_EVALS="10"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED="42"
LOG_WANDB="false"
SPECIAL_LOGGING="false"
PAIR_LIST=""
SMOKE="false"
DRY_RUN="${DRY_RUN:-false}"

WEIGHT_DECAY_CHOICES='[0.0,1e-7,1e-6,1e-5,5e-5,1e-4,5e-4]'
CUSTOM_LOG_ROOT="${RUN_DIR}/artifacts/logs/baseline"

declare -ag ACTIVE_GPUS=()
declare -ag WAVE_PIDS=()

update_phase_summary() {
  local dataset="$1"
  local phase="$2"
  local py_bin
  py_bin="$(run_python_bin)"
  if ! "$py_bin" "${RUN_DIR}/baseline/update_phase_summary.py" --dataset "$dataset" --phase "$phase" >/dev/null 2>&1; then
    echo "[WARN] baseline summary update failed: dataset=${dataset} phase=${phase}" >&2
  fi
}

usage() {
  cat <<'USAGE'
Usage: tune_small_ab.sh --mode a|b|all [--gpus 0,1,2,3] [--phase P0]
                        [--max-evals 10] [--tune-epochs 40] [--tune-patience 10]
                        [--seed 42] [--pairs 1,2] [--smoke] [--dry-run] [--log-wandb]
                        [--special-logging]
USAGE
}

dataset_to_config() {
  case "$1" in
    KuaiRecSmall0.1|kuairecsmall0.1) echo "tune_kuai_small" ;;
    lastfm0.03) echo "tune_lfm_small" ;;
    *)
      echo "Unsupported dataset: $1" >&2
      return 1
      ;;
  esac
}

combo_desc() {
  case "$1" in
    C1) echo "a_c1_hi_bs_short" ;;
    C2) echo "a_c2_std_bs_mid" ;;
    C3) echo "b_c3_long" ;;
    C4) echo "b_c4_wide" ;;
    *)
      echo "unknown_combo" >&2
      return 1
      ;;
  esac
}

combo_dropout_choices() {
  case "$1" in
    C1) echo "[0.0,0.05,0.10,0.15]" ;;
    C2) echo "[0.05,0.10,0.15,0.20]" ;;
    C3) echo "[0.05,0.10,0.15,0.20,0.25]" ;;
    C4) echo "[0.10,0.15,0.20,0.25]" ;;
    *)
      echo "Unsupported combo: $1" >&2
      return 1
      ;;
  esac
}

model_tier() {
  case "$1" in
    sasrec|gru4rec) echo "Light" ;;
    bsarec|fenrec|patt|srgnn) echo "Medium" ;;
    fame|sigma) echo "Heavy" ;;
    *)
      echo "Unknown model tier for: $1" >&2
      return 1
      ;;
  esac
}

combo_struct_overrides() {
  case "$1:$2" in
    sasrec:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sasrec:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sasrec:C3) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sasrec:C4) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;

    gru4rec:C1) printf '%s\n' "hidden_size=128" "num_layers=1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    gru4rec:C2) printf '%s\n' "hidden_size=160" "num_layers=1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    gru4rec:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    gru4rec:C4) printf '%s\n' "hidden_size=160" "num_layers=2" "++MAX_ITEM_LIST_LENGTH=20" ;;

    bsarec:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    bsarec:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    bsarec:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=10" ;;
    bsarec:C4) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;

    fame:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "num_experts=2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C4) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;

    fenrec:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=2" "cl_weight=0.1" "cl_temperature=0.2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "cl_weight=0.05" "cl_temperature=0.2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "cl_weight=0.1" "cl_temperature=0.1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C4) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "cl_weight=0.2" "cl_temperature=0.2" "++MAX_ITEM_LIST_LENGTH=20" ;;

    patt:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=2" "diversity_gamma=0.1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "diversity_gamma=0.05" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "diversity_gamma=0.1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C4) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "diversity_gamma=0.2" "++MAX_ITEM_LIST_LENGTH=20" ;;

    sigma:C1) printf '%s\n' "hidden_size=128" "num_layers=2" "state_size=16" "conv_kernel=4" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C2) printf '%s\n' "hidden_size=128" "num_layers=1" "state_size=16" "conv_kernel=8" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "state_size=32" "conv_kernel=4" "remaining_ratio=0.7" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C4) printf '%s\n' "hidden_size=160" "num_layers=2" "state_size=32" "conv_kernel=8" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=20" ;;

    srgnn:C1) printf '%s\n' "hidden_size=128" "step=1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    srgnn:C2) printf '%s\n' "hidden_size=128" "step=2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    srgnn:C3) printf '%s\n' "hidden_size=160" "step=1" "++MAX_ITEM_LIST_LENGTH=10" ;;
    srgnn:C4) printf '%s\n' "hidden_size=160" "step=3" "++MAX_ITEM_LIST_LENGTH=20" ;;
    *)
      echo "Unsupported combo structure: $1 $2" >&2
      return 1
      ;;
  esac
}

combo_search_fix_overrides() {
  case "$1:$2" in
    sasrec:C1) printf '%s\n' "++search.num_layers=[2]" "++search.attn_dropout_prob=[0.1]" ;;
    sasrec:C2) printf '%s\n' "++search.num_layers=[1]" "++search.attn_dropout_prob=[0.1]" ;;
    sasrec:C3) printf '%s\n' "++search.num_layers=[3]" "++search.attn_dropout_prob=[0.15]" ;;
    sasrec:C4) printf '%s\n' "++search.num_layers=[2]" "++search.attn_dropout_prob=[0.1]" ;;

    gru4rec:C1) printf '%s\n' "++search.num_layers=[1]" ;;
    gru4rec:C2) printf '%s\n' "++search.num_layers=[1]" ;;
    gru4rec:C3) printf '%s\n' "++search.num_layers=[2]" ;;
    gru4rec:C4) printf '%s\n' "++search.num_layers=[2]" ;;

    bsarec:C1) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" ;;
    bsarec:C2) printf '%s\n' "++search.num_layers=[1]" "++search.num_heads=[4]" ;;
    bsarec:C3) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[8]" ;;
    bsarec:C4) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" ;;

    fame:C1) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" "++search.num_experts=[4]" ;;
    fame:C2) printf '%s\n' "++search.num_layers=[1]" "++search.num_heads=[2]" "++search.num_experts=[2]" ;;
    fame:C3) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" "++search.num_experts=[8]" ;;
    fame:C4) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" "++search.num_experts=[4]" ;;

    fenrec:C1) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.1]" "++search.cl_temperature=[0.2]" ;;
    fenrec:C2) printf '%s\n' "++search.num_layers=[1]" "++search.cl_weight=[0.05]" "++search.cl_temperature=[0.2]" ;;
    fenrec:C3) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.1]" "++search.cl_temperature=[0.1]" ;;
    fenrec:C4) printf '%s\n' "++search.num_layers=[3]" "++search.cl_weight=[0.2]" "++search.cl_temperature=[0.2]" ;;

    patt:C1) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.1]" ;;
    patt:C2) printf '%s\n' "++search.num_layers=[1]" "++search.diversity_gamma=[0.05]" ;;
    patt:C3) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.1]" ;;
    patt:C4) printf '%s\n' "++search.num_layers=[3]" "++search.diversity_gamma=[0.2]" ;;

    sigma:C1) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[16]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[4]" ;;
    sigma:C2) printf '%s\n' "++search.num_layers=[1]" "++search.state_size=[16]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[8]" ;;
    sigma:C3) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[32]" "++search.remaining_ratio=[0.7]" "++search.conv_kernel=[4]" ;;
    sigma:C4) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[32]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[8]" ;;

    srgnn:C1) printf '%s\n' "++search.step=[1]" ;;
    srgnn:C2) printf '%s\n' "++search.step=[2]" ;;
    srgnn:C3) printf '%s\n' "++search.step=[1]" ;;
    srgnn:C4) printf '%s\n' "++search.step=[3]" ;;
    *)
      echo "Unsupported combo search fix: $1 $2" >&2
      return 1
      ;;
  esac
}

batch_and_lr() {
  local dataset="$1"
  local model="$2"
  local combo="$3"
  local tier
  tier="$(model_tier "$model")"

  case "${dataset}:${tier}:${combo}" in
    KuaiRecSmall0.1:Light:C1) echo "8192|16384|1e-4|8e-3" ;;
    KuaiRecSmall0.1:Light:C2) echo "4096|8192|7e-5|5e-3" ;;
    KuaiRecSmall0.1:Light:C3) echo "4096|8192|5e-5|3e-3" ;;
    KuaiRecSmall0.1:Light:C4) echo "2048|4096|3e-5|2e-3" ;;

    KuaiRecSmall0.1:Medium:C1) echo "4096|8192|8e-5|5e-3" ;;
    KuaiRecSmall0.1:Medium:C2) echo "4096|8192|5e-5|3e-3" ;;
    KuaiRecSmall0.1:Medium:C3) echo "2048|4096|3e-5|2e-3" ;;
    KuaiRecSmall0.1:Medium:C4) echo "2048|4096|2e-5|1.2e-3" ;;

    KuaiRecSmall0.1:Heavy:C1) echo "4096|8192|5e-5|3e-3" ;;
    KuaiRecSmall0.1:Heavy:C2) echo "2048|4096|3e-5|2e-3" ;;
    KuaiRecSmall0.1:Heavy:C3) echo "2048|4096|2e-5|1.2e-3" ;;
    KuaiRecSmall0.1:Heavy:C4) echo "1024|2048|1e-5|8e-4" ;;

    lastfm0.03:Light:C1) echo "4096|8192|7e-5|5e-3" ;;
    lastfm0.03:Light:C2) echo "2048|4096|5e-5|3e-3" ;;
    lastfm0.03:Light:C3) echo "2048|4096|3e-5|2e-3" ;;
    lastfm0.03:Light:C4) echo "1024|2048|2e-5|1.2e-3" ;;

    lastfm0.03:Medium:C1) echo "2048|4096|5e-5|3e-3" ;;
    lastfm0.03:Medium:C2) echo "2048|4096|3e-5|2e-3" ;;
    lastfm0.03:Medium:C3) echo "1024|2048|2e-5|1.2e-3" ;;
    lastfm0.03:Medium:C4) echo "1024|2048|1e-5|8e-4" ;;

    lastfm0.03:Heavy:C1) echo "2048|4096|3e-5|2e-3" ;;
    lastfm0.03:Heavy:C2) echo "1024|2048|2e-5|1.2e-3" ;;
    lastfm0.03:Heavy:C3) echo "1024|2048|1e-5|8e-4" ;;
    lastfm0.03:Heavy:C4) echo "512|1024|1e-5|5e-4" ;;
    *)
      echo "Unsupported batch/lr combo: ${dataset} ${model} ${combo}" >&2
      return 1
      ;;
  esac
}

sanitize_component() {
  local raw="${1:-}"
  raw="${raw// /_}"
  raw="${raw//\//_}"
  raw="${raw//:/_}"
  raw="${raw//[^a-zA-Z0-9._-]/_}"
  printf '%s' "$raw"
}

make_log_path() {
  local dataset="$1"
  local phase="$2"
  local model="$3"
  local desc="$4"
  local reserve="${5:-true}"
  local dir lock_dir next_file next_id max_id name n prefix path model_slug desc_slug

  dir="${CUSTOM_LOG_ROOT}/${dataset}/${phase}"
  mkdir -p "$dir"
  lock_dir="${dir}/.index.lock"
  next_file="${dir}/.next_index"

  if [ "$reserve" = "true" ]; then
    while ! mkdir "$lock_dir" 2>/dev/null; do
      sleep 0.02
    done
  fi

  if [ -f "$next_file" ] && read -r next_id <"$next_file" && [[ "$next_id" =~ ^[0-9]+$ ]]; then
    :
  else
    max_id=-1
    while IFS= read -r name; do
      if [[ "$name" =~ ^([0-9]{3})_.*\.log$ ]]; then
        n=$((10#${BASH_REMATCH[1]}))
      elif [[ "$name" =~ ^[^_]+_([0-9]{3})_.*\.log$ ]]; then
        n=$((10#${BASH_REMATCH[1]}))
      else
        continue
      fi
      if [ "$n" -gt "$max_id" ]; then
        max_id="$n"
      fi
    done < <(find "$dir" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort)
    next_id=$((max_id + 1))
  fi

  if [ "$reserve" = "true" ]; then
    echo $((next_id + 1)) >"$next_file"
    rmdir "$lock_dir"
  fi

  printf -v prefix "%03d" "$next_id"
  model_slug="$(sanitize_component "$model")"
  desc_slug="$(sanitize_component "$desc")"
  path="${dir}/${model_slug}_${prefix}_${desc_slug}.log"
  if [ ! -e "$path" ]; then
    echo "$path"
    return 0
  fi

  n=2
  while :; do
    path="${dir}/${model_slug}_${prefix}_${desc_slug}_r${n}.log"
    if [ ! -e "$path" ]; then
      echo "$path"
      return 0
    fi
    n=$((n + 1))
  done
}

selected_pair_indexes() {
  local -n out_ref=$1
  local raw_pairs=()
  local token
  out_ref=()

  if [ -z "$PAIR_LIST" ]; then
    out_ref=(1 2 3 4)
    return 0
  fi

  dispatch_parse_csv "$PAIR_LIST" raw_pairs
  for token in "${raw_pairs[@]}"; do
    case "$token" in
      1|2|3|4) out_ref+=("$token") ;;
      *)
        echo "Invalid --pairs entry: $token (allowed: 1,2,3,4)" >&2
        return 1
        ;;
    esac
  done
}

pair_models() {
  case "$1" in
    1) echo "sasrec|gru4rec" ;;
    2) echo "bsarec|fame" ;;
    3) echo "fenrec|patt" ;;
    4) echo "sigma|srgnn" ;;
    *)
      echo "Unknown pair index: $1" >&2
      return 1
      ;;
  esac
}

run_small_job() {
  local gpu="$1"
  local dataset="$2"
  local model="$3"
  local wave="$4"
  local combo="$5"
  local pair_idx="$6"
  local cfg combo_slug log_path reserve run_phase pair_tag batch_spec train_bs eval_bs lr_lo lr_hi
  local search_override type_override cmd_str run_id status rc
  local py_bin
  local -a cmd=()
  local -a struct_overrides=()
  local -a search_fix_overrides=()

  cfg="$(dataset_to_config "$dataset")"
  combo_slug="$(combo_desc "$combo")"
  pair_tag="$(printf 'pair%02d' "$pair_idx")"
  run_phase="${PHASE}_$(printf '%s' "$wave" | tr '[:lower:]' '[:upper:]')_${pair_tag}_${combo}"
  reserve="true"
  if [ "$DRY_RUN" = "true" ]; then
    reserve="false"
  fi
  log_path="$(make_log_path "$dataset" "$PHASE" "$model" "$combo_slug" "$reserve")"
  batch_spec="$(batch_and_lr "$dataset" "$model" "$combo")"
  IFS='|' read -r train_bs eval_bs lr_lo lr_hi <<< "$batch_spec"
  mapfile -t struct_overrides < <(combo_struct_overrides "$model" "$combo")
  type_override="search_space_type_overrides={learning_rate:loguniform,weight_decay:choice,dropout_ratio:choice}"
  py_bin="$(run_python_bin)"
  mapfile -t search_fix_overrides < <(combo_search_fix_overrides "$model" "$combo")

  cmd=(
    "$py_bin" hyperopt_tune.py
    --config-name "$cfg"
    --seed "$SEED"
    --max-evals "$MAX_EVALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --run-group baseline
    --run-axis hparam
    --run-phase "$run_phase"
    "model=${model}"
    "dataset=${dataset}"
    "gpu_id=${gpu}"
    "++seed=${SEED}"
    "train_batch_size=${train_bs}"
    "eval_batch_size=${eval_bs}"
    "+wave=$(printf '%s' "$wave" | tr '[:lower:]' '[:upper:]')"
    "+pair_id=${pair_tag}"
    "+combo_id=${combo}"
    "+combo_desc=${combo_slug}"
    "+trial_epoch_log=false"
    "show_progress=false"
    "log_wandb=${LOG_WANDB}"
    "++search.learning_rate=[${lr_lo},${lr_hi}]"
    "++search.weight_decay=${WEIGHT_DECAY_CHOICES}"
    "++search.dropout_ratio=$(combo_dropout_choices "$combo")"
    "$type_override"
  )
  cmd+=("${struct_overrides[@]}")
  cmd+=("${search_fix_overrides[@]}")
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$SPECIAL_LOGGING" = "true" ]; then
    cmd+=("++special_logging=true")
  fi

  echo "[JOB] gpu=${gpu} dataset=${dataset} model=${model} combo=${combo} phase=${run_phase}"
  echo "[LOG] ${log_path}"
  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] $(run_cmd_str "${cmd[@]}")"
    return 0
  fi

  cmd_str="$(run_cmd_str "${cmd[@]}")"
  run_id="$(run_tracker_start \
    --track baseline \
    --axis hparam \
    --phase "$run_phase" \
    --dataset "$dataset" \
    --model "$model" \
    --cmd "$cmd_str" \
    --log-file "$log_path")"
  update_phase_summary "$dataset" "$PHASE"

  set +e
  BASELINE_PHASE_SUMMARY=1 \
  BASELINE_SUMMARY_PHASE="$PHASE" \
  RUN_ID="$run_id" \
  LOG_FILE="${log_path}" \
  PYTHONUNBUFFERED=1 \
  "${cmd[@]}"
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    status="success"
  else
    status="fail"
  fi
  run_tracker_end \
    --run-id "$run_id" \
    --track baseline \
    --axis hparam \
    --phase "$run_phase" \
    --dataset "$dataset" \
    --model "$model" \
    --cmd "$cmd_str" \
    --log-file "$log_path" \
    --status "$status" \
    --exit-code "$rc"
  update_phase_summary "$dataset" "$PHASE"
  return "$rc"
}

run_queue_worker() {
  local gpu="$1"
  local wave="$2"
  local pair_idx="$3"
  shift 3
  local spec model dataset combo
  local -a queue_specs=("$@")

  if [ "$SMOKE" = "true" ] && [ "${#queue_specs[@]}" -gt 1 ]; then
    queue_specs=("${queue_specs[0]}")
  fi

  for spec in "${queue_specs[@]}"; do
    IFS='|' read -r model dataset combo <<< "$spec"
    run_small_job "$gpu" "$dataset" "$model" "$wave" "$combo" "$pair_idx"
  done
}

run_pair_wave() {
  local wave="$1"
  shift
  local -a wave_gpus=("$@")
  local pair_idx pair_spec m1 m2 combo_a combo_b
  local -a selected_pairs=()
  local -a q0=()
  local -a q1=()
  local -a q2=()
  local -a q3=()
  local -a worker_pids=()
  local gpu
  local failed=0

  selected_pair_indexes selected_pairs
  ACTIVE_GPUS=("${wave_gpus[@]}")

  case "$wave" in
    a|A) combo_a="C1"; combo_b="C2" ;;
    b|B) combo_a="C3"; combo_b="C4" ;;
    *)
      echo "Unsupported wave: $wave" >&2
      return 1
      ;;
  esac

  for pair_idx in "${selected_pairs[@]}"; do
    pair_spec="$(pair_models "$pair_idx")"
    IFS='|' read -r m1 m2 <<< "$pair_spec"
    q0=("${m1}|KuaiRecSmall0.1|${combo_a}" "${m2}|KuaiRecSmall0.1|${combo_b}")
    q1=("${m1}|lastfm0.03|${combo_a}" "${m2}|lastfm0.03|${combo_b}")
    q2=("${m2}|KuaiRecSmall0.1|${combo_a}" "${m1}|KuaiRecSmall0.1|${combo_b}")
    q3=("${m2}|lastfm0.03|${combo_a}" "${m1}|lastfm0.03|${combo_b}")

    echo "[PAIR] wave=$(printf '%s' "$wave" | tr '[:lower:]' '[:upper:]') pair=${pair_idx} models=${m1},${m2}"
    echo "  gpu${wave_gpus[0]}: ${q0[*]}"
    echo "  gpu${wave_gpus[1]}: ${q1[*]}"
    echo "  gpu${wave_gpus[2]}: ${q2[*]}"
    echo "  gpu${wave_gpus[3]}: ${q3[*]}"

    worker_pids=()
    (
      run_queue_worker "${wave_gpus[0]}" "$wave" "$pair_idx" "${q0[@]}"
    ) &
    worker_pids+=("$!")
    dispatch_set_pid "${wave_gpus[0]}" "$!"

    (
      run_queue_worker "${wave_gpus[1]}" "$wave" "$pair_idx" "${q1[@]}"
    ) &
    worker_pids+=("$!")
    dispatch_set_pid "${wave_gpus[1]}" "$!"

    (
      run_queue_worker "${wave_gpus[2]}" "$wave" "$pair_idx" "${q2[@]}"
    ) &
    worker_pids+=("$!")
    dispatch_set_pid "${wave_gpus[2]}" "$!"

    (
      run_queue_worker "${wave_gpus[3]}" "$wave" "$pair_idx" "${q3[@]}"
    ) &
    worker_pids+=("$!")
    dispatch_set_pid "${wave_gpus[3]}" "$!"

    failed=0
    for gpu in "${worker_pids[@]}"; do
      if ! wait "$gpu"; then
        failed=1
      fi
    done
    for gpu in "${wave_gpus[@]}"; do
      DISPATCH_GPU_PID["$gpu"]=""
    done
    if [ "$failed" -ne 0 ]; then
      echo "[ERROR] wave=${wave} pair=${pair_idx} failed" >&2
      return 1
    fi
  done
}

run_mode_single_wave() {
  local wave="$1"
  shift
  local -a gpus=("$@")
  if [ "${#gpus[@]}" -lt 4 ]; then
    echo "--mode ${wave} requires at least 4 GPUs" >&2
    return 1
  fi
  if [ "${#gpus[@]}" -gt 4 ]; then
    echo "[INFO] mode=${wave}: using first 4 GPUs only -> ${gpus[*]:0:4}"
  fi
  run_pair_wave "$wave" "${gpus[@]:0:4}"
}

on_interrupt() {
  local pid
  echo "[INTERRUPT] stopping dispatched jobs..." >&2
  if [ "${#ACTIVE_GPUS[@]}" -gt 0 ]; then
    dispatch_terminate_all ACTIVE_GPUS
  fi
  for pid in "${WAVE_PIDS[@]}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      dispatch_signal_group_if_session_leader TERM "$pid"
      dispatch_signal_tree TERM "$pid"
    fi
  done
  exit 130
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --phase) PHASE="$2"; PHASE_SET="true"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --pairs) PAIR_LIST="$2"; shift 2 ;;
    --smoke) SMOKE="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --special-logging) SPECIAL_LOGGING="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[ -z "$MODE" ] && { echo "--mode required" >&2; usage; exit 1; }

if [ "$SMOKE" = "true" ]; then
  MAX_EVALS="1"
  TUNE_EPOCHS="1"
  TUNE_PATIENCE="1"
  LOG_WANDB="false"
  if [ "$PHASE_SET" = "false" ]; then
    PHASE="P0_SMOKE"
  fi
fi

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
mkdir -p "$CUSTOM_LOG_ROOT"

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -lt 4 ] && { echo "--gpus requires at least 4 entries" >&2; exit 1; }

trap on_interrupt INT TERM

case "${MODE,,}" in
  a|b)
    run_mode_single_wave "${MODE,,}" "${GPUS[@]}"
    ;;
  all)
    if [ "${#GPUS[@]}" -ge 8 ]; then
      echo "[MODE] all -> wave A on GPUs ${GPUS[*]:0:4}, wave B on GPUs ${GPUS[*]:4:4}"
      (
        run_pair_wave "a" "${GPUS[@]:0:4}"
      ) &
      WAVE_PIDS+=("$!")
      (
        run_pair_wave "b" "${GPUS[@]:4:4}"
      ) &
      WAVE_PIDS+=("$!")

      set +e
      wait "${WAVE_PIDS[0]}"
      rc_a=$?
      wait "${WAVE_PIDS[1]}"
      rc_b=$?
      set -e
      if [ "$rc_a" -ne 0 ] || [ "$rc_b" -ne 0 ]; then
        exit 1
      fi
    else
      echo "[MODE] all -> using first 4 GPUs for A then B sequentially"
      run_pair_wave "a" "${GPUS[@]:0:4}"
      run_pair_wave "b" "${GPUS[@]:0:4}"
    fi
    ;;
  *)
    echo "Unsupported --mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

trap - INT TERM
