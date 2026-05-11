#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"

DATASETS="KuaiRecLargeStrictPosV2_0.2,lastfm0.03"
GPU_LIST="0,1,2,3"
BASE_MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="4200"
PHASE="P1_FIXREG"
LOG_WANDB="false"
SPECIAL_LOGGING="false"
DRY_RUN="${DRY_RUN:-false}"
PYTHON_BIN=""

MODELS=("bsarec" "patt" "fame" "fenrec" "sigma")
WAVE1_COMBOS=("C1" "C2" "C3" "C4")
WAVE2_COMBOS=("C5" "C6" "C7" "C8")

declare -ag WORKER_PIDS=()

phase_token_lc() {
  printf '%s' "$PHASE" | tr '[:upper:]' '[:lower:]'
}

auto_trim() {
  echo "$1" | xargs
}

usage() {
  cat <<USAGE
Usage: $0 [--datasets KuaiRecLargeStrictPosV2_0.2,lastfm0.03] [--gpus 0,1,2,3]
          [--base-max-evals 20] [--tune-epochs 100] [--tune-patience 10]
          [--seed-base 4200] [--phase P1_FIXREG] [--dry-run] [--log-wandb]
          [--special-logging]

Execution policy:
- Target models only: BSARec -> PAtt -> FAME -> FENRec -> SIGMA
- 8 combos total: wave1(C1..C4) then wave2(C5..C8)
- Dataset order per wave is preserved: dataset1 -> dataset2 -> ...
- GPU queue is independent: each GPU processes its own queue continuously without global barrier
- Phase summary CSV is updated after each task (on normal run)
USAGE
}

update_phase_summary() {
  local dataset="$1"
  local phase="$2"
  local py_bin
  py_bin="$(run_python_bin)"
  if ! "$py_bin" "${SCRIPT_DIR}/update_phase_summary.py" --dataset "$dataset" --phase "$phase" >/dev/null 2>&1; then
    echo "[WARN] summary update failed: dataset=${dataset} phase=${phase}" >&2
  fi
}

dataset_to_config() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    kuaireclargestrictposv2_0.2) echo "tune_kuai_strict_small" ;;
    lastfm0.03) echo "tune_lfm_small" ;;
    *)
      echo "Unsupported dataset: $1" >&2
      return 1
      ;;
  esac
}

dataset_weight() {
  local dataset="$1"
  local dataset_count="$2"
  local ds_key
  ds_key="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]')"

  if [ "$ds_key" = "lastfm0.03" ] && [ "$dataset_count" -gt 1 ]; then
    echo "0.75"
  else
    echo "1.00"
  fi
}

model_weight() {
  case "$1" in
    bsarec) echo "1.00" ;;
    patt) echo "0.90" ;;
    fame) echo "0.75" ;;
    fenrec) echo "0.65" ;;
    sigma) echo "0.50" ;;
    *)
      echo "Unknown model weight target: $1" >&2
      return 1
      ;;
  esac
}

calc_model_evals() {
  local base="$1"
  local dsw="$2"
  local mw="$3"
  local val
  val="$(awk -v b="$base" -v d="$dsw" -v m="$mw" 'BEGIN{print int(b*d*m)}')"
  if [ -z "$val" ] || [ "$val" -lt 3 ]; then
    val=3
  fi
  echo "$val"
}

lr_range() {
  local dataset="$1"
  local model="$2"
  local ds_key
  ds_key="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]')"

  case "${ds_key}:${model}" in
    kuaireclargestrictposv2_0.2:bsarec) echo "3e-4|1e-2" ;;
    kuaireclargestrictposv2_0.2:patt)   echo "5e-4|1e-2" ;;
    kuaireclargestrictposv2_0.2:fame)   echo "2e-4|6e-3" ;;
    kuaireclargestrictposv2_0.2:fenrec) echo "3e-4|8e-3" ;;
    kuaireclargestrictposv2_0.2:sigma)  echo "1e-4|4e-3" ;;

    lastfm0.03:bsarec) echo "8e-5|1.5e-3" ;;
    lastfm0.03:patt)   echo "1e-4|1.2e-3" ;;
    lastfm0.03:fame)   echo "6e-5|8e-4" ;;
    lastfm0.03:fenrec) echo "8e-5|1.0e-3" ;;
    lastfm0.03:sigma)  echo "5e-5|8e-4" ;;
    *)
      echo "Unsupported lr range target: dataset=${dataset}, model=${model}" >&2
      return 1
      ;;
  esac
}

fixed_reg_params() {
  local dataset="$1"
  local model="$2"
  local ds_key
  ds_key="$(printf '%s' "$dataset" | tr '[:upper:]' '[:lower:]')"

  case "${ds_key}:${model}" in
    kuaireclargestrictposv2_0.2:bsarec) echo "0.15|1e-4" ;;
    kuaireclargestrictposv2_0.2:patt)   echo "0.10|1e-4" ;;
    kuaireclargestrictposv2_0.2:fame)   echo "0.10|1e-4" ;;
    kuaireclargestrictposv2_0.2:fenrec) echo "0.10|1e-4" ;;
    kuaireclargestrictposv2_0.2:sigma)  echo "0.20|1e-4" ;;

    lastfm0.03:bsarec) echo "0.10|5e-4" ;;
    lastfm0.03:patt)   echo "0.15|0.0" ;;
    lastfm0.03:fame)   echo "0.10|1e-4" ;;
    lastfm0.03:fenrec) echo "0.20|0.0" ;;
    lastfm0.03:sigma)  echo "0.05|1e-4" ;;
    *)
      echo "Unsupported fixed-reg target: dataset=${dataset}, model=${model}" >&2
      return 1
      ;;
  esac
}

combo_struct_overrides() {
  local model="$1"
  local combo="$2"

  case "${model}:${combo}" in
    bsarec:C1) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=2" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C2) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    bsarec:C3) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C4) printf '%s\n' "hidden_size=256" "num_layers=2" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C5) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C6) printf '%s\n' "hidden_size=160" "num_layers=1" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=20" ;;
    bsarec:C7) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "++MAX_ITEM_LIST_LENGTH=30" ;;
    bsarec:C8) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "++MAX_ITEM_LIST_LENGTH=30" ;;

    patt:C1) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "diversity_gamma=0.05" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C2) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=2" "diversity_gamma=0.10" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "diversity_gamma=0.10" "++MAX_ITEM_LIST_LENGTH=10" ;;
    patt:C4) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=8" "diversity_gamma=0.25" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C5) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "diversity_gamma=0.20" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C6) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "diversity_gamma=0.15" "++MAX_ITEM_LIST_LENGTH=20" ;;
    patt:C7) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "diversity_gamma=0.25" "++MAX_ITEM_LIST_LENGTH=30" ;;
    patt:C8) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "diversity_gamma=0.30" "++MAX_ITEM_LIST_LENGTH=30" ;;

    fame:C1) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "num_experts=2" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C2) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fame:C4) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=8" "num_experts=8" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C5) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "num_experts=4" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C6) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "num_experts=6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fame:C7) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "num_experts=10" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fame:C8) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "num_experts=12" "++MAX_ITEM_LIST_LENGTH=30" ;;

    fenrec:C1) printf '%s\n' "hidden_size=128" "num_layers=1" "num_heads=2" "cl_weight=0.05" "cl_temperature=0.20" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C2) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=2" "cl_weight=0.10" "cl_temperature=0.20" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "num_heads=4" "cl_weight=0.10" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=10" ;;
    fenrec:C4) printf '%s\n' "hidden_size=192" "num_layers=2" "num_heads=8" "cl_weight=0.25" "cl_temperature=0.10" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C5) printf '%s\n' "hidden_size=160" "num_layers=3" "num_heads=4" "cl_weight=0.20" "cl_temperature=0.20" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C6) printf '%s\n' "hidden_size=160" "num_layers=2" "num_heads=4" "cl_weight=0.15" "cl_temperature=0.15" "++MAX_ITEM_LIST_LENGTH=20" ;;
    fenrec:C7) printf '%s\n' "hidden_size=128" "num_layers=3" "num_heads=4" "cl_weight=0.30" "cl_temperature=0.25" "++MAX_ITEM_LIST_LENGTH=30" ;;
    fenrec:C8) printf '%s\n' "hidden_size=192" "num_layers=3" "num_heads=8" "cl_weight=0.35" "cl_temperature=0.30" "++MAX_ITEM_LIST_LENGTH=30" ;;

    sigma:C1) printf '%s\n' "hidden_size=128" "num_layers=1" "state_size=16" "conv_kernel=8" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C2) printf '%s\n' "hidden_size=128" "num_layers=2" "state_size=16" "conv_kernel=4" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C3) printf '%s\n' "hidden_size=128" "num_layers=2" "state_size=32" "conv_kernel=4" "remaining_ratio=0.7" "++MAX_ITEM_LIST_LENGTH=10" ;;
    sigma:C4) printf '%s\n' "hidden_size=160" "num_layers=3" "state_size=32" "conv_kernel=8" "remaining_ratio=0.8" "++MAX_ITEM_LIST_LENGTH=30" ;;
    sigma:C5) printf '%s\n' "hidden_size=160" "num_layers=2" "state_size=32" "conv_kernel=8" "remaining_ratio=0.5" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C6) printf '%s\n' "hidden_size=160" "num_layers=1" "state_size=32" "conv_kernel=4" "remaining_ratio=0.6" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C7) printf '%s\n' "hidden_size=192" "num_layers=2" "state_size=32" "conv_kernel=8" "remaining_ratio=0.7" "++MAX_ITEM_LIST_LENGTH=20" ;;
    sigma:C8) printf '%s\n' "hidden_size=192" "num_layers=3" "state_size=32" "conv_kernel=8" "remaining_ratio=0.9" "++MAX_ITEM_LIST_LENGTH=30" ;;

    *)
      echo "Unsupported combo structure target: model=${model}, combo=${combo}" >&2
      return 1
      ;;
  esac
}

combo_search_fix_overrides() {
  local model="$1"
  local combo="$2"

  case "${model}:${combo}" in
    bsarec:C1) printf '%s\n' "++search.num_layers=[1]" "++search.num_heads=[2]" ;;
    bsarec:C2) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" ;;
    bsarec:C3) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" ;;
    bsarec:C4) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[8]" ;;
    bsarec:C5) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" ;;
    bsarec:C6) printf '%s\n' "++search.num_layers=[1]" "++search.num_heads=[8]" ;;
    bsarec:C7) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" ;;
    bsarec:C8) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[8]" ;;

    patt:C1) printf '%s\n' "++search.num_layers=[1]" "++search.diversity_gamma=[0.05]" ;;
    patt:C2) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.1]" ;;
    patt:C3) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.1]" ;;
    patt:C4) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.25]" ;;
    patt:C5) printf '%s\n' "++search.num_layers=[3]" "++search.diversity_gamma=[0.2]" ;;
    patt:C6) printf '%s\n' "++search.num_layers=[2]" "++search.diversity_gamma=[0.15]" ;;
    patt:C7) printf '%s\n' "++search.num_layers=[3]" "++search.diversity_gamma=[0.25]" ;;
    patt:C8) printf '%s\n' "++search.num_layers=[3]" "++search.diversity_gamma=[0.3]" ;;

    fame:C1) printf '%s\n' "++search.num_layers=[1]" "++search.num_heads=[2]" "++search.num_experts=[2]" ;;
    fame:C2) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" "++search.num_experts=[4]" ;;
    fame:C3) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" "++search.num_experts=[8]" ;;
    fame:C4) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[8]" "++search.num_experts=[8]" ;;
    fame:C5) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" "++search.num_experts=[4]" ;;
    fame:C6) printf '%s\n' "++search.num_layers=[2]" "++search.num_heads=[4]" "++search.num_experts=[6]" ;;
    fame:C7) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[4]" "++search.num_experts=[10]" ;;
    fame:C8) printf '%s\n' "++search.num_layers=[3]" "++search.num_heads=[8]" "++search.num_experts=[12]" ;;

    fenrec:C1) printf '%s\n' "++search.num_layers=[1]" "++search.cl_weight=[0.05]" "++search.cl_temperature=[0.2]" ;;
    fenrec:C2) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.1]" "++search.cl_temperature=[0.2]" ;;
    fenrec:C3) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.1]" "++search.cl_temperature=[0.1]" ;;
    fenrec:C4) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.25]" "++search.cl_temperature=[0.1]" ;;
    fenrec:C5) printf '%s\n' "++search.num_layers=[3]" "++search.cl_weight=[0.2]" "++search.cl_temperature=[0.2]" ;;
    fenrec:C6) printf '%s\n' "++search.num_layers=[2]" "++search.cl_weight=[0.15]" "++search.cl_temperature=[0.15]" ;;
    fenrec:C7) printf '%s\n' "++search.num_layers=[3]" "++search.cl_weight=[0.3]" "++search.cl_temperature=[0.25]" ;;
    fenrec:C8) printf '%s\n' "++search.num_layers=[3]" "++search.cl_weight=[0.35]" "++search.cl_temperature=[0.3]" ;;

    sigma:C1) printf '%s\n' "++search.num_layers=[1]" "++search.state_size=[16]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[8]" ;;
    sigma:C2) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[16]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[4]" ;;
    sigma:C3) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[32]" "++search.remaining_ratio=[0.7]" "++search.conv_kernel=[4]" ;;
    sigma:C4) printf '%s\n' "++search.num_layers=[3]" "++search.state_size=[32]" "++search.remaining_ratio=[0.8]" "++search.conv_kernel=[8]" ;;
    sigma:C5) printf '%s\n' "++search.num_layers=[2]" "++search.state_size=[32]" "++search.remaining_ratio=[0.5]" "++search.conv_kernel=[8]" ;;
    sigma:C6) printf '%s\n' "++search.num_layers=[1]" "++search.state_size=[32]" "++search.remaining_ratio=[0.6]" "++search.conv_kernel=[4]" ;;
    sigma:C7) printf '%s\n' "++search.num_layers=[3]" "++search.state_size=[32]" "++search.remaining_ratio=[0.8]" "++search.conv_kernel=[8]" ;;
    sigma:C8) printf '%s\n' "++search.num_layers=[3]" "++search.state_size=[32]" "++search.remaining_ratio=[0.9]" "++search.conv_kernel=[8]" ;;

    *)
      echo "Unsupported combo search-fix target: model=${model}, combo=${combo}" >&2
      return 1
      ;;
  esac
}

make_log_path() {
  local dataset="$1"
  local model="$2"
  local combo="$3"
  local phase="$4"
  local gpu="$5"
  local dir ts rid
  ts="$(date +%Y%m%d_%H%M%S)"
  rid="${RANDOM}"
  dir="${RUN_DIR}/artifacts/logs/baseline/${dataset}/${phase}"
  mkdir -p "$dir"
  echo "${dir}/${model}_${combo}_g${gpu}_${ts}_${rid}.log"
}

cleanup_incomplete_artifacts_for_dataset() {
  local dataset="$1"
  local results_root logs_dir phase_lc
  results_root="${RUN_DIR}/artifacts/results/baseline"
  logs_dir="${RUN_DIR}/artifacts/logs/baseline/${dataset}/${PHASE}"
  phase_lc="$(phase_token_lc)"

  "$PYTHON_BIN" - "$results_root" "$logs_dir" "$dataset" "$phase_lc" <<'PY'
import glob
import json
import os
import sys

results_root, logs_dir, dataset, phase_lc = sys.argv[1:5]

deleted_results = []
deleted_refs = []
deleted_logs = []

pattern = os.path.join(results_root, f"{dataset}_*_{phase_lc}_*.json")
for path in sorted(glob.glob(pattern)):
  data = None
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception:
    data = None

  should_delete = False
  refs = []
  if data is None:
    should_delete = True
  else:
    n_completed = int(data.get("n_completed", -1) or -1)
    max_evals = int(data.get("max_evals", 0) or 0)
    interrupted = bool(data.get("interrupted", False))
    should_delete = interrupted or max_evals <= 0 or n_completed < max_evals
    refs = [
      data.get("normal_result_mirror_file", ""),
      data.get("special_result_file", ""),
      data.get("special_log_file", ""),
    ]

  if not should_delete:
    continue

  if os.path.exists(path):
    os.remove(path)
    deleted_results.append(path)

  for ref in refs:
    if ref and os.path.exists(ref):
      os.remove(ref)
      deleted_refs.append(ref)

if os.path.isdir(logs_dir):
  for log_path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
    try:
      with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    except Exception:
      txt = ""
    if "[RUN_STATUS] END status=terminated" in txt or "[RUN_STATUS] TERMINATED signal=" in txt:
      os.remove(log_path)
      deleted_logs.append(log_path)

print(
  f"[CLEANUP] dataset={dataset} phase={phase_lc} "
  f"deleted_results={len(deleted_results)} deleted_refs={len(deleted_refs)} deleted_logs={len(deleted_logs)}"
)
PY
}

find_completed_result_for_task() {
  local dataset="$1"
  local model="$2"
  local combo="$3"
  local run_phase="$4"
  local required_evals="$5"
  local results_root phase_lc
  results_root="${RUN_DIR}/artifacts/results/baseline"
  phase_lc="$(phase_token_lc)"

  "$PYTHON_BIN" - "$results_root" "$dataset" "$phase_lc" "$model" "$combo" "$run_phase" "$required_evals" <<'PY'
import glob
import json
import os
import sys

results_root, dataset, phase_lc, model, combo, run_phase, required_evals = sys.argv[1:8]
required_evals = int(required_evals)
combo_lc = combo.lower()
best_path = ""
best_mtime = -1.0

pattern = os.path.join(results_root, f"{dataset}_*_{phase_lc}_*_{model}_{combo_lc}_*.json")
for path in glob.glob(pattern):
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except Exception:
    continue

  if str(data.get("run_phase", "")).upper() != run_phase.upper():
    continue

  interrupted = bool(data.get("interrupted", False))
  n_completed = int(data.get("n_completed", 0) or 0)
  max_evals = int(data.get("max_evals", 0) or 0)
  if interrupted:
    continue
  if n_completed < required_evals:
    continue
  if max_evals < required_evals:
    continue

  mtime = os.path.getmtime(path)
  if mtime > best_mtime:
    best_mtime = mtime
    best_path = path

if best_path:
  print(best_path)
  sys.exit(0)
sys.exit(1)
PY
}

run_combo_job() {
  local dataset="$1"
  local cfg="$2"
  local model="$3"
  local combo="$4"
  local gpu="$5"
  local evals="$6"
  local seed="$7"
  local lr_lo="$8"
  local lr_hi="$9"
  local dropout="${10}"
  local wd="${11}"
  local run_phase="${12}"

  local log_path cmd_str
  local -a cmd=()
  local -a struct_overrides=()
  local -a search_fix_overrides=()

  log_path="$(make_log_path "$dataset" "$model" "$combo" "$PHASE" "$gpu")"
  mapfile -t struct_overrides < <(combo_struct_overrides "$model" "$combo")
  mapfile -t search_fix_overrides < <(combo_search_fix_overrides "$model" "$combo")

  cmd=(
    "$PYTHON_BIN" hyperopt_tune.py
    --config-name "$cfg"
    --seed "$seed"
    --max-evals "$evals"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --run-group baseline
    --run-axis hparam
    --run-phase "$run_phase"
    "model=${model}"
    "dataset=${dataset}"
    "gpu_id=${gpu}"
    "++seed=${seed}"
    "+combo_id=${combo}"
    "+trial_epoch_log=false"
    "show_progress=false"
    "log_wandb=${LOG_WANDB}"
    "++search.learning_rate=[${lr_lo},${lr_hi}]"
    "++search.weight_decay=[${wd}]"
    "++search.dropout_ratio=[${dropout}]"
    "++search_space_type_overrides.learning_rate=loguniform"
    "++search_space_type_overrides.weight_decay=choice"
    "++search_space_type_overrides.dropout_ratio=choice"
  )
  cmd+=("${struct_overrides[@]}")
  cmd+=("${search_fix_overrides[@]}")
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$SPECIAL_LOGGING" = "true" ]; then
    cmd+=("++special_logging=true")
  fi

  echo "[JOB] dataset=${dataset} model=${model} combo=${combo} gpu=${gpu} evals=${evals} lr=[${lr_lo},${lr_hi}] drop=${dropout} wd=${wd}"
  echo "[LOG] ${log_path}"

  if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY_RUN] $(run_cmd_str "${cmd[@]}")"
    return 0
  fi

  cmd_str="$(run_cmd_str "${cmd[@]}")"
  local run_id
  run_id="$(run_tracker_start \
    --track baseline \
    --axis hparam \
    --phase "$run_phase" \
    --dataset "$dataset" \
    --model "$model" \
    --cmd "$cmd_str" \
    --log-file "$log_path")"

  set +e
  LOG_FILE="$log_path" PYTHONUNBUFFERED=1 "${cmd[@]}"
  local rc=$?
  set -e

  local status
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

  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi

  return 0
}

run_gpu_queue() {
  local gpu="$1"
  local queue_payload="$2"
  local line wave dataset cfg model combo evals seed lr_lo lr_hi drop wd run_phase

  while IFS= read -r line; do
    [ -z "$line" ] && continue
    IFS='|' read -r wave dataset cfg model combo evals seed lr_lo lr_hi drop wd run_phase <<< "$line"
    if ! run_combo_job "$dataset" "$cfg" "$model" "$combo" "$gpu" "$evals" "$seed" "$lr_lo" "$lr_hi" "$drop" "$wd" "$run_phase"; then
      echo "[GPU ${gpu}] failed task: ${line}" >&2
      return 1
    fi
  done <<< "$queue_payload"

  return 0
}

on_interrupt() {
  echo "[INTERRUPT] stopping gpu workers..." >&2
  for pid in "${WORKER_PIDS[@]:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --base-max-evals) BASE_MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --special-logging) SPECIAL_LOGGING="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

if [ -z "${RUN_PYTHON_BIN:-}" ] && [ -x "/venv/FMoE/bin/python" ]; then
  export RUN_PYTHON_BIN="/venv/FMoE/bin/python"
fi
PYTHON_BIN="${RUN_PYTHON_BIN:-$(run_python_bin)}"

# Fail fast if critical deps are missing in the actual runtime interpreter.
if ! "$PYTHON_BIN" - <<'PY'
import importlib
must_have = ["torch", "recbole", "lightgbm", "hyperopt", "yaml"]
missing = []
for name in must_have:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit("missing_python_packages=" + ",".join(missing))

# Verify custom model registration end-to-end.
import recbole_patch  # noqa: F401
from recbole.utils import utils as rbu
for model_name in ["BSARec", "PAtt", "FAME", "FENRec", "SIGMA", "FeaturedMoE_N3"]:
  try:
    _ = rbu.get_model(model_name)
  except Exception as e:
    raise SystemExit(f"missing_or_broken_model={model_name}: {e}")

print("[ENV_CHECK] python and core packages OK")
PY
then
  echo "[ERROR] runtime environment check failed for PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "[HINT] install into this interpreter: ${PYTHON_BIN} -m pip install lightgbm" >&2
  exit 1
fi

IFS=',' read -r -a DATASET_ARR <<< "$DATASETS"
if [ "${#DATASET_ARR[@]}" -eq 0 ]; then
  echo "Empty datasets" >&2
  exit 1
fi

if [ "$DRY_RUN" = "false" ]; then
  for dataset_raw in "${DATASET_ARR[@]}"; do
    dataset="$(auto_trim "$dataset_raw")"
    [ -z "$dataset" ] && continue
    cleanup_incomplete_artifacts_for_dataset "$dataset"
  done
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -lt 4 ]; then
  echo "--gpus requires at least 4 GPU ids" >&2
  exit 1
fi
ACTIVE_GPUS=("$(auto_trim "${GPUS[0]}")" "$(auto_trim "${GPUS[1]}")" "$(auto_trim "${GPUS[2]}")" "$(auto_trim "${GPUS[3]}")")

QUEUES=("" "" "" "")

echo "[PLAN] datasets=${DATASETS}"
echo "[PLAN] python_bin=${PYTHON_BIN}"
echo "[PLAN] model order=${MODELS[*]}"
echo "[PLAN] wave1=${WAVE1_COMBOS[*]} wave2=${WAVE2_COMBOS[*]}"
echo "[PLAN] combo->gpu wave1: C1->${ACTIVE_GPUS[0]}, C2->${ACTIVE_GPUS[1]}, C3->${ACTIVE_GPUS[2]}, C4->${ACTIVE_GPUS[3]}"
echo "[PLAN] combo->gpu wave2: C5->${ACTIVE_GPUS[0]}, C6->${ACTIVE_GPUS[1]}, C7->${ACTIVE_GPUS[2]}, C8->${ACTIVE_GPUS[3]}"
echo "[PLAN] fixed regularization mode (wd/drop fixed), lr-only search"
echo "[PLAN] special_logging=${SPECIAL_LOGGING}"

dataset_count="${#DATASET_ARR[@]}"
for wave_idx in 0 1; do
  if [ "$wave_idx" -eq 0 ]; then
    CUR_COMBOS=("${WAVE1_COMBOS[@]}")
    wave_name="W1"
  else
    CUR_COMBOS=("${WAVE2_COMBOS[@]}")
    wave_name="W2"
  fi

  for dataset_idx in "${!DATASET_ARR[@]}"; do
    dataset="$(auto_trim "${DATASET_ARR[$dataset_idx]}")"
    [ -z "$dataset" ] && continue

    cfg="$(dataset_to_config "$dataset")"
    dsw="$(dataset_weight "$dataset" "$dataset_count")"

    for model_idx in "${!MODELS[@]}"; do
      model="${MODELS[$model_idx]}"
      mw="$(model_weight "$model")"
      evals="$(calc_model_evals "$BASE_MAX_EVALS" "$dsw" "$mw")"
      IFS='|' read -r lr_lo lr_hi <<< "$(lr_range "$dataset" "$model")"
      IFS='|' read -r drop wd <<< "$(fixed_reg_params "$dataset" "$model")"

      for slot_idx in 0 1 2 3; do
        combo="${CUR_COMBOS[$slot_idx]}"
        gpu="${ACTIVE_GPUS[$slot_idx]}"
        seed=$((SEED_BASE + wave_idx * 100000 + dataset_idx * 10000 + model_idx * 100 + slot_idx))
        run_phase="${PHASE}_${wave_name}_D$(printf '%02d' "$dataset_idx")_${model^^}_${combo}"

        if completed_path="$(find_completed_result_for_task "$dataset" "$model" "$combo" "$run_phase" "$evals")"; then
          echo "[SKIP] dataset=${dataset} model=${model} combo=${combo} phase=${run_phase} (completed: ${completed_path##*/})"
          continue
        fi

        line="${wave_name}|${dataset}|${cfg}|${model}|${combo}|${evals}|${seed}|${lr_lo}|${lr_hi}|${drop}|${wd}|${run_phase}"
        QUEUES[$slot_idx]="${QUEUES[$slot_idx]}${line}"$'\n'
      done
    done
  done
done

for i in 0 1 2 3; do
  cnt="$(printf '%s' "${QUEUES[$i]}" | awk 'NF>0{n+=1} END{print n+0}')"
  echo "[QUEUE] gpu=${ACTIVE_GPUS[$i]} tasks=${cnt}"
done

trap on_interrupt INT TERM

WORKER_PIDS=()
for i in 0 1 2 3; do
  gpu="${ACTIVE_GPUS[$i]}"
  payload="${QUEUES[$i]}"
  (
    run_gpu_queue "$gpu" "$payload"
  ) &
  WORKER_PIDS+=("$!")
done

failed=0
for p in "${WORKER_PIDS[@]}"; do
  if ! wait "$p"; then
    failed=1
  fi
done

trap - INT TERM

if [ "$failed" -ne 0 ]; then
  echo "[ERROR] one or more gpu queues failed" >&2
  exit 1
fi

if [ "$DRY_RUN" = "false" ]; then
  for dataset_raw in "${DATASET_ARR[@]}"; do
    dataset="$(auto_trim "$dataset_raw")"
    [ -z "$dataset" ] && continue
    update_phase_summary "$dataset" "$PHASE"
  done
fi

echo "[ALL DONE] completed all queued tasks for datasets: ${DATASETS}"
