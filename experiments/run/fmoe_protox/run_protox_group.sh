#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

GROUP_GPUS="4,5,6,7"
DATASETS="movielens1m"
SEED_BASE="3400"
PHASE_PREFIX="P2_ml1_focus"

ROUND1_COMBOS_PER_GPU="8"
ROUND1_MAX_EVALS="15"
ROUND1_TUNE_EPOCHS="40"
ROUND1_TUNE_PATIENCE="15"
ROUND2_TOP_RATIO="0.5"
ROUND2_SCALE="1.25"

LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"

LR_RANGE="2e-4,2e-2"
WD_VALUES="0.0,1e-7,1e-6,1e-5,1e-4"
DROPOUT_VALUES="0.0,0.05,0.1,0.15,0.2,0.25"
BALANCE_VALUES="0.01,0.03,0.05,0.1,0.2"
PROTO_USAGE_VALUES="0.0,1e-5,1e-4,3e-4,1e-3"
PROTO_ENTROPY_VALUES="0.0,1e-4,3e-4,1e-3,3e-3,1e-2"

# Fixed defaults for P1 screening
FIXED_EMBEDDING_SIZE="160"
FIXED_NUM_HEADS="8"
FIXED_D_FEAT="16"
FIXED_EXPERT_SCALE="3"
FIXED_PROTO_NUM="8"
FIXED_PROTO_POOLING="query"
FIXED_MOE_TOP_K="0"
FIXED_USE_VALID_RATIO="true"
FIXED_MACRO_TEMP="1.0"
FIXED_MID_TEMP="1.3"
FIXED_MICRO_TEMP="1.3"
FIXED_MID_FEAT_DROPOUT="0.1"
FIXED_MICRO_FEAT_DROPOUT="0.1"

# Catalog arrays
COMBO_TAGS=()
COMBO_MERGE=()
COMBO_GPRE=()
COMBO_GPOST=()
COMBO_MACRO=()
COMBO_MID=()
COMBO_MICRO=()
COMBO_DEXP=()
COMBO_DROUTER=()
COMBO_PDIM=()
COMBO_PTOPK=()
COMBO_TSTART=()
COMBO_TEND=()
COMBO_FLOOR=()
COMBO_DSCALE=()

usage() {
  cat <<USAGE
Usage: $0 [--gpus 4,5,6,7] [--datasets movielens1m]
          [--seed-base 3400] [--phase-prefix P2_ml1_focus]
          [--round1-combos-per-gpu 8] [--round1-max-evals 15]
          [--round1-tune-epochs 40] [--round1-tune-patience 15]
          [--round2-top-ratio 0.5] [--round2-scale 1.25]
          [--protox-lr-range 2e-4,2e-2]
          [--protox-wd-values 0.0,1e-7,1e-6,1e-5,1e-4]
          [--protox-dropout-values 0.0,0.05,0.1,0.15,0.2,0.25]
          [--protox-balance-values 0.01,0.03,0.05,0.1,0.2]
          [--protox-usage-values 0.0,1e-5,1e-4,3e-4,1e-3]
          [--protox-entropy-values 0.0,1e-4,3e-4,1e-3,3e-3,1e-2]
          [--log-wandb|--no-wandb] [--dry-run]

Legacy aliases:
  --combos-per-gpu => --round1-combos-per-gpu
  --max-evals      => --round1-max-evals
  --tune-epochs    => --round1-tune-epochs
  --tune-patience  => --round1-tune-patience

Execution order:
  ML1 Round1 -> RR Round1 -> ML1 Round2 -> RR Round2
USAGE
}

add_combo() {
  local tag="$1" merge="$2" gpre="$3" gpost="$4" macro="$5" mid="$6" micro="$7"
  local dexp="$8" drouter="$9" pdim="${10}" ptopk="${11}" tstart="${12}" tend="${13}" floor="${14}" dscale="${15}"

  COMBO_TAGS+=("$tag")
  COMBO_MERGE+=("$merge")
  COMBO_GPRE+=("$gpre")
  COMBO_GPOST+=("$gpost")
  COMBO_MACRO+=("$macro")
  COMBO_MID+=("$mid")
  COMBO_MICRO+=("$micro")
  COMBO_DEXP+=("$dexp")
  COMBO_DROUTER+=("$drouter")
  COMBO_PDIM+=("$pdim")
  COMBO_PTOPK+=("$ptopk")
  COMBO_TSTART+=("$tstart")
  COMBO_TEND+=("$tend")
  COMBO_FLOOR+=("$floor")
  COMBO_DSCALE+=("$dscale")
}

build_combo_catalog() {
  COMBO_TAGS=()
  COMBO_MERGE=()
  COMBO_GPRE=()
  COMBO_GPOST=()
  COMBO_MACRO=()
  COMBO_MID=()
  COMBO_MICRO=()
  COMBO_DEXP=()
  COMBO_DROUTER=()
  COMBO_PDIM=()
  COMBO_PTOPK=()
  COMBO_TSTART=()
  COMBO_TEND=()
  COMBO_FLOOR=()
  COMBO_DSCALE=()

  # Focused second tuning around the better ML1 families.
  # Keep most combos near the good region, but reserve one aggressive point per family.
  add_combo "S01" "serial_weighted"   "0" "1" "0" "1" "1" "160" "80"  "48" "2" "1.1" "0.9"  "0.05" "1.5"
  add_combo "S02" "serial_weighted"   "0" "1" "0" "1" "1" "192" "96"  "64" "2" "1.0" "0.8"  "0.05" "1.5"
  add_combo "S03" "serial_weighted"   "0" "1" "0" "1" "1" "512" "128" "64" "2" "0.9" "0.7"  "0.10" "2.0"
  add_combo "S04" "serial_weighted"   "0" "2" "0" "1" "1" "160" "80"  "48" "2" "1.1" "0.9"  "0.05" "1.5"
  add_combo "S05" "serial_weighted"   "0" "2" "0" "1" "1" "320" "112" "64" "2" "1.0" "0.8"  "0.10" "1.5"
  add_combo "P01" "parallel_weighted" "1" "0" "1" "1" "0" "192" "96"  "64" "2" "0.9" "0.7"  "0.10" "2.0"
  add_combo "P02" "parallel_weighted" "1" "0" "1" "1" "0" "160" "80"  "48" "2" "1.0" "0.8"  "0.05" "1.5"
  add_combo "P03" "parallel_weighted" "1" "0" "1" "1" "0" "512" "128" "64" "2" "0.9" "0.7"  "0.10" "2.0"
}

combo_count() {
  echo "${#COMBO_TAGS[@]}"
}

combo_index_from_tag() {
  local tag="$1"
  local i
  for i in "${!COMBO_TAGS[@]}"; do
    if [ "${COMBO_TAGS[$i]}" = "$tag" ]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

dataset_alias() {
  local ds="$1"
  case "$ds" in
    movielens1m) echo "ML1" ;;
    retail_rocket) echo "RR" ;;
    *) echo "$(run_sanitize "$ds" | tr '[:lower:]' '[:upper:]')" ;;
  esac
}

dataset_has() {
  local target="$1"
  local ds
  for ds in "${DS_ARR[@]}"; do
    if [ "$ds" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

batch_sizes() {
  local ds="$1"
  case "$ds" in
    movielens1m) echo "6144 12288" ;;
    retail_rocket) echo "4096 8192" ;;
    *) echo "4096 8192" ;;
  esac
}

round_to_5() {
  local v="$1"
  "$PY_BIN" - <<'PY' "$v"
import sys
v = float(sys.argv[1])
out = int(5 * round(v / 5.0))
if out < 5:
    out = 5
print(out)
PY
}

ceil_int() {
  local v="$1"
  "$PY_BIN" - <<'PY' "$v"
import math
import sys
print(int(math.ceil(float(sys.argv[1]))))
PY
}

validate_float_positive() {
  local v="$1" name="$2"
  "$PY_BIN" - <<'PY' "$v" "$name"
import sys
v = float(sys.argv[1])
name = sys.argv[2]
if v <= 0:
    raise SystemExit(f"{name} must be > 0")
PY
}

validate_float_in_unit() {
  local v="$1" name="$2"
  "$PY_BIN" - <<'PY' "$v" "$name"
import sys
v = float(sys.argv[1])
name = sys.argv[2]
if not (0.0 < v <= 1.0):
    raise SystemExit(f"{name} must be in (0,1]")
PY
}

protox_summary_dir() {
  echo "$(run_log_dir fmoe_protox)/hparam/${PHASE_PREFIX}"
}

protox_dataset_log_dir() {
  local dataset="$1"
  local ds_tag
  ds_tag="$(run_dataset_tag "$dataset")"
  echo "$(protox_summary_dir)/${ds_tag}/FMoEProtoX"
}

protox_make_log_path() {
  local dataset="$1"
  local phase="$2"
  local d
  d="$(protox_dataset_log_dir "$dataset")"
  run_ensure_dir "$d"
  echo "${d}/$(run_timestamp)_hparam_$(run_sanitize "$phase").log"
}

build_combo_overrides() {
  local combo_idx="$1"
  local variant_kind="$2"
  local -n out_overrides_ref="$3"
  local -n out_tag_ref="$4"

  local merge gpre gpost mpre midpre micpre dexp drouter pdim ptopk tstart tend floor dscale
  merge="${COMBO_MERGE[$combo_idx]}"
  gpre="${COMBO_GPRE[$combo_idx]}"
  gpost="${COMBO_GPOST[$combo_idx]}"
  mpre="${COMBO_MACRO[$combo_idx]}"
  midpre="${COMBO_MID[$combo_idx]}"
  micpre="${COMBO_MICRO[$combo_idx]}"
  dexp="${COMBO_DEXP[$combo_idx]}"
  drouter="${COMBO_DROUTER[$combo_idx]}"
  pdim="${COMBO_PDIM[$combo_idx]}"
  ptopk="${COMBO_PTOPK[$combo_idx]}"
  tstart="${COMBO_TSTART[$combo_idx]}"
  tend="${COMBO_TEND[$combo_idx]}"
  floor="${COMBO_FLOOR[$combo_idx]}"
  dscale="${COMBO_DSCALE[$combo_idx]}"
  out_tag_ref="${COMBO_TAGS[$combo_idx]}"

  if [ "$variant_kind" -eq 1 ]; then
    pdim="64"
    out_tag_ref="${out_tag_ref}V1"
  elif [ "$variant_kind" -eq 2 ]; then
    drouter="96"
    out_tag_ref="${out_tag_ref}V2"
  elif [ "$variant_kind" -eq 3 ]; then
    ptopk="2"
    tstart="1.0"
    tend="0.8"
    out_tag_ref="${out_tag_ref}V3"
  elif [ "$variant_kind" -eq 4 ]; then
    floor="0.1"
    dscale="2.0"
    out_tag_ref="${out_tag_ref}V4"
  fi

  out_overrides_ref=(
    "protox_stage_merge_mode=${merge}"
    "++search.protox_stage_merge_mode=[${merge}]"
    "protox_global_pre_layers=${gpre}"
    "++search.protox_global_pre_layers=[${gpre}]"
    "protox_global_post_layers=${gpost}"
    "++search.protox_global_post_layers=[${gpost}]"
    "protox_macro_pre_layers=${mpre}"
    "++search.protox_macro_pre_layers=[${mpre}]"
    "protox_mid_pre_layers=${midpre}"
    "++search.protox_mid_pre_layers=[${midpre}]"
    "protox_micro_pre_layers=${micpre}"
    "++search.protox_micro_pre_layers=[${micpre}]"
    "d_expert_hidden=${dexp}"
    "++search.d_expert_hidden=[${dexp}]"
    "d_router_hidden=${drouter}"
    "++search.d_router_hidden=[${drouter}]"
    "proto_dim=${pdim}"
    "++search.proto_dim=[${pdim}]"
    "proto_top_k=${ptopk}"
    "++search.proto_top_k=[${ptopk}]"
    "proto_temperature_start=${tstart}"
    "++search.proto_temperature_start=[${tstart}]"
    "proto_temperature_end=${tend}"
    "++search.proto_temperature_end=[${tend}]"
    "stage_weight_floor=${floor}"
    "++search.stage_weight_floor=[${floor}]"
    "stage_delta_scale=${dscale}"
    "++search.stage_delta_scale=[${dscale}]"
  )
}

ml1_top_combo_tags() {
  local topn="$1"
  "$PY_BIN" - <<'PY' "$topn" "$PHASE_PREFIX" "$RUN_DIR"
import json
import re
import sys
from pathlib import Path

topn = int(sys.argv[1])
prefix = sys.argv[2]
run_dir = Path(sys.argv[3])
root = run_dir / "artifacts" / "results" / "fmoe_protox"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob("movielens1m_FeaturedMoE_ProtoX_*.json"):
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        continue
    phase = str(d.get("run_phase", ""))
    if not phase.startswith(f"{prefix}_ML1_R1_"):
        continue
    m = d.get("best_mrr@20")
    if not isinstance(m, (int, float)):
        bvr = d.get("best_valid_result", {})
        m = bvr.get("mrr@20") if isinstance(bvr, dict) else None
    if not isinstance(m, (int, float)):
        continue
    n_completed = int(d.get("n_completed") or 0)
    mm = re.search(r"_C\d+_([A-Za-z0-9.-]+)$", phase)
    if not mm:
        continue
    combo = mm.group(1)
    rows.append((float(m), n_completed, phase, combo))

rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
seen = set()
out = []
for _, _, _, combo in rows:
    if combo in seen:
        continue
    seen.add(combo)
    out.append(combo)
    if len(out) >= topn:
        break

for c in out:
    print(c)
PY
}

round1_top_combo_result_files() {
  local dataset="$1"
  local alias="$2"
  local topn="$3"
  "$PY_BIN" - <<'PY' "$dataset" "$alias" "$topn" "$PHASE_PREFIX" "$RUN_DIR"
import json
import re
import sys
from pathlib import Path

dataset = sys.argv[1]
alias = sys.argv[2]
topn = int(sys.argv[3])
prefix = sys.argv[4]
run_dir = Path(sys.argv[5])
root = run_dir / "artifacts" / "results" / "fmoe_protox"
if not root.exists():
    raise SystemExit(0)

rows = []
for p in root.glob(f"{dataset}_FeaturedMoE_ProtoX_*.json"):
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        continue
    phase = str(d.get("run_phase", ""))
    if not phase.startswith(f"{prefix}_{alias}_R1_"):
        continue

    m_combo = re.search(r"_C\d+_([A-Za-z0-9.-]+)$", phase)
    if not m_combo:
        continue
    combo_tag = m_combo.group(1)

    score = d.get("best_mrr@20")
    if not isinstance(score, (int, float)):
        bvr = d.get("best_valid_result", {})
        score = bvr.get("mrr@20") if isinstance(bvr, dict) else None
    if not isinstance(score, (int, float)):
        continue

    n_completed = int(d.get("n_completed") or 0)
    rows.append((combo_tag, float(score), n_completed, phase, p))

# best run per combo
best_by_combo = {}
for combo, score, n_completed, phase, path in rows:
    prev = best_by_combo.get(combo)
    cand = (score, n_completed, phase, path)
    if prev is None:
        best_by_combo[combo] = cand
        continue
    if (score, n_completed, -1) > (prev[0], prev[1], -1):
        best_by_combo[combo] = cand
    elif score == prev[0] and n_completed == prev[1] and phase < prev[2]:
        best_by_combo[combo] = cand

combo_rows = [(combo, *vals) for combo, vals in best_by_combo.items()]
combo_rows.sort(key=lambda x: (-x[1], -x[2], x[3]))
for combo, score, n_completed, phase, path in combo_rows[:topn]:
    print(str(path))
PY
}

extract_result_overrides() {
  local result_json="$1"
  "$PY_BIN" - <<'PY' "$result_json"
import json
import re
import sys

path = sys.argv[1]
d = json.load(open(path, "r", encoding="utf-8"))
fs = d.get("fixed_search") or {}
phase = str(d.get("run_phase", ""))
m = re.search(r"_C\d+_([A-Za-z0-9.-]+)$", phase)
tag = m.group(1) if m else "SEL"

print(f"__TAG={tag}")

def pick(k, default):
    v = fs.get(k, default)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)

keys = {
    "protox_stage_merge_mode": "serial_weighted",
    "protox_global_pre_layers": 0,
    "protox_global_post_layers": 0,
    "protox_macro_pre_layers": 0,
    "protox_mid_pre_layers": 0,
    "protox_micro_pre_layers": 0,
    "d_expert_hidden": 160,
    "d_router_hidden": 80,
    "proto_dim": 48,
    "proto_top_k": 0,
    "proto_temperature_start": 1.2,
    "proto_temperature_end": 1.0,
    "stage_weight_floor": 0.0,
    "stage_delta_scale": 1.0,
}

for k, default in keys.items():
    v = pick(k, default)
    print(f"{k}={v}")
    print(f"++search.{k}=[{v}]")
PY
}

run_protox_search() {
  local dataset="$1"
  local gpu="$2"
  local phase="$3"
  local max_evals="$4"
  local tune_epochs="$5"
  local tune_patience="$6"
  local seed="$7"
  shift 7
  local -a overrides=("$@")

  read -r train_bs eval_bs <<< "$(batch_sizes "$dataset")"
  local log_file
  log_file="$(protox_make_log_path "$dataset" "$phase")"

  local -a cmd=(
    "$PY_BIN" hyperopt_tune.py
    --config-name config
    --max-evals "$max_evals"
    --tune-epochs "$tune_epochs"
    --tune-patience "$tune_patience"
    --seed "$seed"
    --run-group fmoe_protox
    --run-axis hparam
    --run-phase "$phase"
    "model=featured_moe_protox_tune"
    "+search_space_type_overrides={weight_decay:choice,hidden_dropout_prob:choice,balance_loss_lambda:choice,proto_usage_lambda:choice,proto_entropy_lambda:choice}"
    "dataset=${dataset}"
    "eval_mode=session"
    "feature_mode=full_v2"
    "gpu_id=${gpu}"
    "log_wandb=${LOG_WANDB}"
    "enable_tf32=true"
    "fmoe_debug_logging=false"
    "MAX_ITEM_LIST_LENGTH=10"
    "++search.MAX_ITEM_LIST_LENGTH=[10]"
    "train_batch_size=${train_bs}"
    "++search.train_batch_size=[${train_bs}]"
    "eval_batch_size=${eval_bs}"
    "++search.eval_batch_size=[${eval_bs}]"

    "proto_num=${FIXED_PROTO_NUM}"
    "++search.proto_num=[${FIXED_PROTO_NUM}]"
    "proto_pooling=${FIXED_PROTO_POOLING}"
    "++search.proto_pooling=[${FIXED_PROTO_POOLING}]"
    "proto_router_use_hidden=true"
    "++search.proto_router_use_hidden=[true]"
    "proto_router_use_feature=true"
    "++search.proto_router_use_feature=[true]"
    "+protox_stage_token_correction=true"
    "++search.protox_stage_token_correction=[true]"
    "+protox_stage_token_correction_scale=0.5"
    "++search.protox_stage_token_correction_scale=[0.5]"

    "embedding_size=${FIXED_EMBEDDING_SIZE}"
    "++search.embedding_size=[${FIXED_EMBEDDING_SIZE}]"
    "num_heads=${FIXED_NUM_HEADS}"
    "++search.num_heads=[${FIXED_NUM_HEADS}]"
    "d_feat_emb=${FIXED_D_FEAT}"
    "++search.d_feat_emb=[${FIXED_D_FEAT}]"
    "expert_scale=${FIXED_EXPERT_SCALE}"
    "++search.expert_scale=[${FIXED_EXPERT_SCALE}]"

    "moe_top_k=${FIXED_MOE_TOP_K}"
    "++search.moe_top_k=[${FIXED_MOE_TOP_K}]"
    "macro_router_temperature=${FIXED_MACRO_TEMP}"
    "++search.macro_router_temperature=[${FIXED_MACRO_TEMP}]"
    "mid_router_temperature=${FIXED_MID_TEMP}"
    "++search.mid_router_temperature=[${FIXED_MID_TEMP}]"
    "micro_router_temperature=${FIXED_MICRO_TEMP}"
    "++search.micro_router_temperature=[${FIXED_MICRO_TEMP}]"
    "mid_router_feature_dropout=${FIXED_MID_FEAT_DROPOUT}"
    "++search.mid_router_feature_dropout=[${FIXED_MID_FEAT_DROPOUT}]"
    "micro_router_feature_dropout=${FIXED_MICRO_FEAT_DROPOUT}"
    "++search.micro_router_feature_dropout=[${FIXED_MICRO_FEAT_DROPOUT}]"
    "use_valid_ratio_gating=${FIXED_USE_VALID_RATIO}"
    "++search.use_valid_ratio_gating=[${FIXED_USE_VALID_RATIO}]"

    "learning_rate=0.001"
    "hidden_dropout_prob=0.1"
    "balance_loss_lambda=0.01"
    "proto_usage_lambda=0.0"
    "proto_entropy_lambda=0.0"

    "++search.learning_rate=[${LR_RANGE}]"
    "++search.weight_decay=[${WD_VALUES}]"
    "++search.hidden_dropout_prob=[${DROPOUT_VALUES}]"
    "++search.balance_loss_lambda=[${BALANCE_VALUES}]"
    "++search.proto_usage_lambda=[${PROTO_USAGE_VALUES}]"
    "++search.proto_entropy_lambda=[${PROTO_ENTROPY_VALUES}]"
  )

  local ov
  for ov in "${overrides[@]}"; do
    cmd+=("$ov")
  done

  run_echo_cmd "${cmd[@]}"
  echo "[LOG] ${log_file}"

  if [ "$DRY_RUN" = "true" ]; then
    return 0
  fi

  local cmd_str run_id rc status
  cmd_str="$(run_cmd_str "${cmd[@]}")"
  run_id="$(run_tracker_start \
    --track fmoe_protox \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_ProtoX" \
    --exp-name "protox_focus_tune" \
    --exp-desc "ProtoX focused ML1-centered second tuning." \
    --exp-focus "layout,layers,d_expert_hidden,d_router_hidden,proto_dim,proto_top_k,proto_temperature_start,proto_temperature_end,stage_weight_floor,stage_delta_scale,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda,proto_usage_lambda,proto_entropy_lambda" \
    --cmd "$cmd_str" \
    --log-file "$log_file")"

  set +e
  LOG_FILE="${log_file}" PYTHONUNBUFFERED=1 "${cmd[@]}"
  rc=$?
  set -e

  status="success"
  if [ "$rc" -ne 0 ]; then
    status="fail"
  fi

  run_tracker_end \
    --run-id "$run_id" \
    --track fmoe_protox \
    --axis hparam \
    --phase "$phase" \
    --dataset "$dataset" \
    --model "FeaturedMoE_ProtoX" \
    --exp-name "protox_focus_tune" \
    --exp-desc "ProtoX focused ML1-centered second tuning." \
    --exp-focus "layout,layers,d_expert_hidden,d_router_hidden,proto_dim,proto_top_k,proto_temperature_start,proto_temperature_end,stage_weight_floor,stage_delta_scale,learning_rate,weight_decay,hidden_dropout_prob,balance_loss_lambda,proto_usage_lambda,proto_entropy_lambda" \
    --cmd "$cmd_str" \
    --log-file "$log_file" \
    --status "$status" \
    --exit-code "$rc"

  return "$rc"
}

run_round1_ml1() {
  local dataset="movielens1m"
  local alias="ML1"
  local total_jobs=$(( ${#GPUS[@]} * ROUND1_COMBOS_PER_GPU ))
  local n_combo
  n_combo="$(combo_count)"

  echo "[P1][${alias}][R1] total_jobs=${total_jobs} combos_per_gpu=${ROUND1_COMBOS_PER_GPU} combo_catalog=${n_combo}"

  local -a pids=()
  local gidx gpu
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      local slot combo_idx seed phase combo_tag
      local -a overrides=()
      for slot in $(seq 0 $((ROUND1_COMBOS_PER_GPU - 1))); do
        combo_idx=$(( slot % n_combo ))
        build_combo_overrides "$combo_idx" 0 overrides combo_tag
        seed=$((SEED_BASE + 10000 + gidx * 1000 + slot))
        phase="${PHASE_PREFIX}_${alias}_R1_G${gpu}_C$(printf '%02d' $((slot + 1)))_${combo_tag}"
        echo "[P1][${alias}][R1] gpu=${gpu} slot=$((slot + 1)) combo=${combo_tag} seed=${seed}"
        run_protox_search "$dataset" "$gpu" "$phase" "$ROUND1_MAX_EVALS" "$ROUND1_TUNE_EPOCHS" "$ROUND1_TUNE_PATIENCE" "$seed" "${overrides[@]}"
      done
    ) &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  return "$rc"
}

run_round1_rr() {
  local dataset="retail_rocket"
  local alias="RR"
  local total_jobs=$(( ${#GPUS[@]} * ROUND1_COMBOS_PER_GPU ))
  local n_combo
  n_combo="$(combo_count)"
  local rr_slots derived_count derived_start

  local -a rr_src_idx=()
  local -a rr_variant=()
  local i
  rr_slots="$ROUND1_COMBOS_PER_GPU"
  derived_count=$(( rr_slots / 5 ))
  if [ "$derived_count" -lt 1 ]; then
    derived_count=1
  fi
  derived_start=$(( rr_slots - derived_count ))
  for i in $(seq 0 $((rr_slots - 1))); do
    rr_src_idx+=("$(( i % n_combo ))")
    rr_variant+=("0")
  done

  if [ "$DRY_RUN" != "true" ] && dataset_has movielens1m; then
    local -a top_tags=()
    mapfile -t top_tags < <(ml1_top_combo_tags "$derived_count")
    for i in "${!top_tags[@]}"; do
      if [ "$i" -ge "$derived_count" ]; then
        break
      fi
      local top_tag idx
      top_tag="${top_tags[$i]}"
      if idx="$(combo_index_from_tag "$top_tag")"; then
        rr_src_idx[$((derived_start + i))]="$idx"
        rr_variant[$((derived_start + i))]="$((i + 1))"
      fi
    done
  fi

  echo "[P1][${alias}][R1] total_jobs=${total_jobs} combos_per_gpu=${ROUND1_COMBOS_PER_GPU} rr_mix=80%independent+20%ml1-derived"

  local -a pids=()
  local gidx gpu
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      local slot combo_slot combo_idx variant seed phase combo_tag
      local -a overrides=()
      for slot in $(seq 0 $((ROUND1_COMBOS_PER_GPU - 1))); do
        combo_slot=$(( slot % rr_slots ))
        combo_idx="${rr_src_idx[$combo_slot]}"
        variant="${rr_variant[$combo_slot]}"
        build_combo_overrides "$combo_idx" "$variant" overrides combo_tag
        seed=$((SEED_BASE + 20000 + gidx * 1000 + slot))
        phase="${PHASE_PREFIX}_${alias}_R1_G${gpu}_C$(printf '%02d' $((slot + 1)))_${combo_tag}"
        echo "[P1][${alias}][R1] gpu=${gpu} slot=$((slot + 1)) combo=${combo_tag} seed=${seed}"
        run_protox_search "$dataset" "$gpu" "$phase" "$ROUND1_MAX_EVALS" "$ROUND1_TUNE_EPOCHS" "$ROUND1_TUNE_PATIENCE" "$seed" "${overrides[@]}"
      done
    ) &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  return "$rc"
}

run_round2_dataset() {
  local dataset="$1"
  local alias="$2"
  local round2_combos_per_gpu="$3"
  local round2_max_evals="$4"
  local round2_epochs="$5"

  local round2_total=$(( round2_combos_per_gpu * ${#GPUS[@]} ))
  echo "[P1][${alias}][R2] total_jobs=${round2_total} combos_per_gpu=${round2_combos_per_gpu} (top unique combos from R1) max_evals=${round2_max_evals} epochs=${round2_epochs}"

  local -a selected_results=()
  if [ "$DRY_RUN" != "true" ]; then
    mapfile -t selected_results < <(round1_top_combo_result_files "$dataset" "$alias" "$round2_combos_per_gpu")
  fi

  local -a pids=()
  local gidx gpu
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      local slot job_idx seed phase combo_tag
      local -a overrides=()
      local -a lines=()

      for slot in $(seq 0 $((round2_combos_per_gpu - 1))); do
        job_idx=$(( gidx * round2_combos_per_gpu + slot ))
        combo_tag="SEL$(printf '%02d' $((job_idx + 1)))"
        overrides=()

        if [ "$DRY_RUN" = "true" ] || [ "${#selected_results[@]}" -eq 0 ]; then
          local fallback_idx
          fallback_idx=$(( slot % $(combo_count) ))
          build_combo_overrides "$fallback_idx" 0 overrides combo_tag
        else
          local src_json
          src_json="${selected_results[$(( slot % ${#selected_results[@]} ))]}"
          mapfile -t lines < <(extract_result_overrides "$src_json")
          local line
          for line in "${lines[@]}"; do
            if [[ "$line" == __TAG=* ]]; then
              combo_tag="${line#__TAG=}"
            else
              overrides+=("$line")
            fi
          done
        fi

        local round2_alias_offset=0
        if [ "$alias" = "RR" ]; then
          round2_alias_offset=10000
        fi
        seed=$((SEED_BASE + 30000 + round2_alias_offset + gidx * 1000 + slot))
        phase="${PHASE_PREFIX}_${alias}_R2_G${gpu}_C$(printf '%02d' $((slot + 1)))_${combo_tag}"
        echo "[P1][${alias}][R2] gpu=${gpu} slot=$((slot + 1)) combo=${combo_tag} seed=${seed}"
        run_protox_search "$dataset" "$gpu" "$phase" "$round2_max_evals" "$round2_epochs" "$ROUND1_TUNE_PATIENCE" "$seed" "${overrides[@]}"
      done
    ) &
    pids+=("$!")
  done

  local rc=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      rc=1
    fi
  done
  return "$rc"
}

generate_dataset_summary() {
  local dataset="$1"
  local summary_root
  summary_root="$(protox_summary_dir)"
  run_ensure_dir "$summary_root"

  local csv_path md_path
  csv_path="${summary_root}/summary_${dataset}.csv"
  md_path="${summary_root}/summary_${dataset}.md"

  "$PY_BIN" - <<'PY' "$dataset" "$PHASE_PREFIX" "$csv_path" "$md_path" "$summary_root" "$RUN_DIR"
import csv
import json
import re
import sys
from pathlib import Path

dataset = sys.argv[1]
prefix = sys.argv[2]
out_csv = Path(sys.argv[3])
out_md = Path(sys.argv[4])
summary_root = Path(sys.argv[5])
run_dir = Path(sys.argv[6])
root = run_dir / "artifacts" / "results" / "fmoe_protox"

if not root.exists():
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset","round","phase","combo_tag","best_mrr@20","n_completed","max_evals",
            "fixed_core","best_tuned_params","log_file","result_file"
        ])
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(f"# {dataset} {prefix} Summary\n\nNo results found.\n", encoding="utf-8")
    raise SystemExit(0)


def best_mrr(d):
    x = d.get("best_mrr@20")
    if isinstance(x, (int, float)):
        return float(x)
    bvr = d.get("best_valid_result")
    if isinstance(bvr, dict):
        y = bvr.get("mrr@20")
        if isinstance(y, (int, float)):
            return float(y)
    return None


def best_params_from_trials(d):
    trials = d.get("trials")
    if not isinstance(trials, list):
        return {}
    best = None
    for t in trials:
        if not isinstance(t, dict):
            continue
        m = t.get("mrr@20")
        if not isinstance(m, (int, float)):
            continue
        if best is None or float(m) > best[0]:
            best = (float(m), t)
    if best is None:
        return {}
    params = best[1].get("params")
    return params if isinstance(params, dict) else {}


def sanitize(s):
    s = s.replace(" ", "_").replace("/", "_").replace(":", "_").replace(",", "__")
    return re.sub(r"[^a-zA-Z0-9._-]", "_", s)

rows = []
for p in root.glob(f"{dataset}_FeaturedMoE_ProtoX_*.json"):
    try:
        d = json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        continue

    phase = str(d.get("run_phase", ""))
    if not phase.startswith(f"{prefix}_"):
        continue

    mrr = best_mrr(d)
    n_completed = int(d.get("n_completed") or 0)
    max_evals = int(d.get("max_evals") or 0)

    if "_R1_" in phase:
        round_id = "R1"
    elif "_R2_" in phase:
        round_id = "R2"
    else:
        round_id = "-"

    mm = re.search(r"_C\d+_([A-Za-z0-9.-]+)$", phase)
    combo_tag = mm.group(1) if mm else "-"

    fs = d.get("fixed_search") if isinstance(d.get("fixed_search"), dict) else {}
    fixed_core = {
        "merge": fs.get("protox_stage_merge_mode"),
        "gpre": fs.get("protox_global_pre_layers"),
        "gpost": fs.get("protox_global_post_layers"),
        "macro": fs.get("protox_macro_pre_layers"),
        "mid": fs.get("protox_mid_pre_layers"),
        "micro": fs.get("protox_micro_pre_layers"),
        "d_exp": fs.get("d_expert_hidden"),
        "d_router": fs.get("d_router_hidden"),
        "proto_dim": fs.get("proto_dim"),
        "proto_top_k": fs.get("proto_top_k"),
        "temp": [fs.get("proto_temperature_start"), fs.get("proto_temperature_end")],
        "floor": fs.get("stage_weight_floor"),
        "delta": fs.get("stage_delta_scale"),
    }

    best_tuned = best_params_from_trials(d)

    phase_key = sanitize(phase)
    log_candidates = sorted(summary_root.rglob(f"*{phase_key}*.log"))
    log_file = str(log_candidates[-1]) if log_candidates else ""

    rows.append({
        "dataset": dataset,
        "round": round_id,
        "phase": phase,
        "combo_tag": combo_tag,
        "best_mrr@20": mrr,
        "n_completed": n_completed,
        "max_evals": max_evals,
        "fixed_core": json.dumps(fixed_core, ensure_ascii=False, sort_keys=True),
        "best_tuned_params": json.dumps(best_tuned, ensure_ascii=False, sort_keys=True),
        "log_file": log_file,
        "result_file": str(p),
    })

rows.sort(key=lambda r: (
    -(r["best_mrr@20"] if isinstance(r["best_mrr@20"], (int, float)) else -1.0),
    -r["n_completed"],
    r["phase"],
))

out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", encoding="utf-8", newline="") as f:
    fields = [
        "dataset","round","phase","combo_tag","best_mrr@20","n_completed","max_evals",
        "fixed_core","best_tuned_params","log_file","result_file"
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        row = dict(r)
        if isinstance(row["best_mrr@20"], (int, float)):
            row["best_mrr@20"] = f"{row['best_mrr@20']:.6f}"
        else:
            row["best_mrr@20"] = ""
        w.writerow(row)

lines = []
lines.append(f"# {dataset} {prefix} Summary")
lines.append("")
lines.append(f"- run_phase_prefix: {prefix}_")
lines.append(f"- runs: {len(rows)}")

best = rows[0] if rows else None
if best and isinstance(best.get("best_mrr@20"), (int, float)):
    lines.append(f"- best_mrr@20: {best['best_mrr@20']:.6f} ({best['phase']})")
lines.append("")
lines.append("| rank | round | phase | combo | mrr@20 | completed | max_evals | log |")
lines.append("|---:|---|---|---|---:|---:|---:|---|")
for i, r in enumerate(rows[:30], 1):
    mrr = f"{r['best_mrr@20']:.6f}" if isinstance(r["best_mrr@20"], (int, float)) else "-"
    log_s = r["log_file"] if r["log_file"] else "-"
    lines.append(
        f"| {i} | {r['round']} | {r['phase']} | {r['combo_tag']} | {mrr} | {r['n_completed']} | {r['max_evals']} | {log_s} |"
    )

out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

write_readme() {
  local summary_root
  summary_root="$(protox_summary_dir)"
  run_ensure_dir "$summary_root"

  local readme_path
  readme_path="${summary_root}/README.md"

  {
    echo "# ProtoX ${PHASE_PREFIX}"
    echo
    echo "## Overview"
    echo "- Entry script: experiments/run/fmoe_protox/run_protox_group.sh"
    echo "- Order: ML1 R1 -> RR R1 -> ML1 R2 -> RR R2 (datasets not requested are skipped)"
    echo "- Round1 budget: combos_per_gpu=${ROUND1_COMBOS_PER_GPU}, max_eval=${ROUND1_MAX_EVALS}, epochs=${ROUND1_TUNE_EPOCHS}, patience=${ROUND1_TUNE_PATIENCE}"
    echo "- Round2 rule: top_ratio=${ROUND2_TOP_RATIO}, scale=${ROUND2_SCALE}, eval/epochs rounded to multiples of 5"
    echo
    echo "## Fixed Params"
    echo "- embedding_size=${FIXED_EMBEDDING_SIZE}, num_heads=${FIXED_NUM_HEADS}, d_feat_emb=${FIXED_D_FEAT}, expert_scale=${FIXED_EXPERT_SCALE}"
    echo "- proto_num=${FIXED_PROTO_NUM}, proto_pooling=${FIXED_PROTO_POOLING}, moe_top_k=${FIXED_MOE_TOP_K}"
    echo "- temperatures: macro=${FIXED_MACRO_TEMP}, mid=${FIXED_MID_TEMP}, micro=${FIXED_MICRO_TEMP}"
    echo "- stage token correction: enabled (scale=0.5)"
    echo
    echo "## Search Ranges"
    echo "- learning_rate=[${LR_RANGE}] (loguniform)"
    echo "- weight_decay=[${WD_VALUES}] (choice)"
    echo "- hidden_dropout_prob=[${DROPOUT_VALUES}] (choice)"
    echo "- balance_loss_lambda=[${BALANCE_VALUES}] (choice)"
    echo "- proto_usage_lambda=[${PROTO_USAGE_VALUES}] (choice)"
    echo "- proto_entropy_lambda=[${PROTO_ENTROPY_VALUES}] (choice)"
    echo
    echo "## Round1 Combo Catalog"
    echo "| idx | tag | group | merge | gpre | gpost | macro | mid | micro | d_exp | d_router | proto_dim | top_k | temp_start | temp_end | floor | delta |"
    echo "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    local i group
    for i in "${!COMBO_TAGS[@]}"; do
      if [[ "${COMBO_TAGS[$i]}" == S* ]]; then group="serial"; else group="parallel"; fi
      echo "| $((i + 1)) | ${COMBO_TAGS[$i]} | ${group} | ${COMBO_MERGE[$i]} | ${COMBO_GPRE[$i]} | ${COMBO_GPOST[$i]} | ${COMBO_MACRO[$i]} | ${COMBO_MID[$i]} | ${COMBO_MICRO[$i]} | ${COMBO_DEXP[$i]} | ${COMBO_DROUTER[$i]} | ${COMBO_PDIM[$i]} | ${COMBO_PTOPK[$i]} | ${COMBO_TSTART[$i]} | ${COMBO_TEND[$i]} | ${COMBO_FLOOR[$i]} | ${COMBO_DSCALE[$i]} |"
    done
    echo
    echo "## RR Mix Rule"
    echo "- RR Round1: 80% independent combos + 20% ML1 top-derived variants"
    echo "- If ML1 top extraction fails, RR Round1 uses 100% independent catalog"
    echo
    echo "## Summary Files"
    echo "- summary_movielens1m.csv / .md"
    echo "- summary_retail_rocket.csv / .md"
  } > "$readme_path"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus|--group-b-gpus) GROUP_GPUS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;

    --round1-combos-per-gpu|--combos-per-gpu) ROUND1_COMBOS_PER_GPU="$2"; shift 2 ;;
    --round1-max-evals|--max-evals) ROUND1_MAX_EVALS="$2"; shift 2 ;;
    --round1-tune-epochs|--tune-epochs) ROUND1_TUNE_EPOCHS="$2"; shift 2 ;;
    --round1-tune-patience|--tune-patience) ROUND1_TUNE_PATIENCE="$2"; shift 2 ;;
    --round2-top-ratio) ROUND2_TOP_RATIO="$2"; shift 2 ;;
    --round2-scale) ROUND2_SCALE="$2"; shift 2 ;;

    --protox-lr-range|--lr-range) LR_RANGE="$2"; shift 2 ;;
    --protox-wd-values|--wd-values) WD_VALUES="$2"; shift 2 ;;
    --protox-dropout-values|--dropout-values) DROPOUT_VALUES="$2"; shift 2 ;;
    --protox-balance-values|--balance-values) BALANCE_VALUES="$2"; shift 2 ;;
    --protox-usage-values|--proto-usage-values) PROTO_USAGE_VALUES="$2"; shift 2 ;;
    --protox-entropy-values|--proto-entropy-values) PROTO_ENTROPY_VALUES="$2"; shift 2 ;;

    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GROUP_GPUS" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DS_ARR
[ "${#DS_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

if ! [[ "$ROUND1_COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$ROUND1_COMBOS_PER_GPU" -le 0 ]; then
  echo "--round1-combos-per-gpu must be a positive integer" >&2
  exit 1
fi
if ! [[ "$ROUND1_MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$ROUND1_MAX_EVALS" -le 0 ]; then
  echo "--round1-max-evals must be a positive integer" >&2
  exit 1
fi
if ! [[ "$ROUND1_TUNE_EPOCHS" =~ ^[0-9]+$ ]] || [ "$ROUND1_TUNE_EPOCHS" -le 0 ]; then
  echo "--round1-tune-epochs must be a positive integer" >&2
  exit 1
fi
if ! [[ "$ROUND1_TUNE_PATIENCE" =~ ^[0-9]+$ ]] || [ "$ROUND1_TUNE_PATIENCE" -le 0 ]; then
  echo "--round1-tune-patience must be a positive integer" >&2
  exit 1
fi
if ! [[ "$SEED_BASE" =~ ^[0-9]+$ ]]; then
  echo "--seed-base must be an integer" >&2
  exit 1
fi

for v in "$LR_RANGE" "$WD_VALUES" "$DROPOUT_VALUES" "$BALANCE_VALUES" "$PROTO_USAGE_VALUES" "$PROTO_ENTROPY_VALUES"; do
  if [[ "$v" != *,* ]]; then
    echo "Value list must be comma-separated: ${v}" >&2
    exit 1
  fi
done

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env
PY_BIN="$(run_python_bin)"

validate_float_in_unit "$ROUND2_TOP_RATIO" "--round2-top-ratio"
validate_float_positive "$ROUND2_SCALE" "--round2-scale"

build_combo_catalog

round2_combos_per_gpu="$("$PY_BIN" - <<'PY' "$ROUND1_COMBOS_PER_GPU" "$ROUND2_TOP_RATIO"
import math
import sys
print(int(math.ceil(float(sys.argv[1]) * float(sys.argv[2]))))
PY
)"
round2_max_evals="$(round_to_5 "$("$PY_BIN" - <<'PY' "$ROUND1_MAX_EVALS" "$ROUND2_SCALE"
import sys
print(float(sys.argv[1]) * float(sys.argv[2]))
PY
)")"
round2_tune_epochs="$(round_to_5 "$("$PY_BIN" - <<'PY' "$ROUND1_TUNE_EPOCHS" "$ROUND2_SCALE"
import sys
print(float(sys.argv[1]) * float(sys.argv[2]))
PY
)")"

echo "[PROTOX_P1] gpus=${GROUP_GPUS} datasets=${DATASETS} dry_run=${DRY_RUN}"
echo "[PROTOX_P1] Round1 combos_per_gpu=${ROUND1_COMBOS_PER_GPU} max_evals=${ROUND1_MAX_EVALS} epochs=${ROUND1_TUNE_EPOCHS} patience=${ROUND1_TUNE_PATIENCE}"
echo "[PROTOX_P1] Round2 combos_per_gpu=${round2_combos_per_gpu} max_evals=${round2_max_evals} epochs=${round2_tune_epochs} patience=${ROUND1_TUNE_PATIENCE}"

echo "[PROTOX_P1] phase_prefix=${PHASE_PREFIX}"

write_readme

# Ordered execution:
# 1) ML1 R1 -> 2) RR R1 -> 3) ML1 R2 -> 4) RR R2

if dataset_has movielens1m; then
  if ! run_round1_ml1; then
    echo "[PROTOX_P1] ML1 Round1 failed." >&2
    exit 1
  fi
  generate_dataset_summary movielens1m
fi

if dataset_has retail_rocket; then
  if ! run_round1_rr; then
    echo "[PROTOX_P1] RR Round1 failed." >&2
    exit 1
  fi
  generate_dataset_summary retail_rocket
fi

if dataset_has movielens1m; then
  if ! run_round2_dataset movielens1m ML1 "$round2_combos_per_gpu" "$round2_max_evals" "$round2_tune_epochs"; then
    echo "[PROTOX_P1] ML1 Round2 failed." >&2
    exit 1
  fi
  generate_dataset_summary movielens1m
fi

if dataset_has retail_rocket; then
  if ! run_round2_dataset retail_rocket RR "$round2_combos_per_gpu" "$round2_max_evals" "$round2_tune_epochs"; then
    echo "[PROTOX_P1] RR Round2 failed." >&2
    exit 1
  fi
  generate_dataset_summary retail_rocket
fi

if [ "$DRY_RUN" != "true" ]; then
  run_update_model_report \
    fmoe_protox \
    FeaturedMoE_ProtoX \
    "$(run_experiments_dir)/models/FeaturedMoE_ProtoX"
  run_update_track_report fmoe_protox
fi

echo "[PROTOX_P1] completed. summary_dir=$(protox_summary_dir)"
