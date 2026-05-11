#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1"
COMBOS_PER_GPU="3"
MAX_EVALS="20"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
SEED_BASE="240"
PHASE_PREFIX="P2DB"
SCHEDULE_PRESET="off"
SERIAL_LAYOUT_ID="7"
PARALLEL_LAYOUT_ID="13"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
OOM_RETRY_MIN_TRAIN_BS="1024"
MODES="both"
SPLIT_AFTER_SERIAL="false"
SKIP_SERIAL_ON_SPLIT="false"
PARALLEL_GPU_LIST="0,1,2,3"
RULE_GPU_LIST="4,5,6,7"
RULE_ARMS="R0,R1"
RULE_LAYOUT_ID="7"
RULE_EXECUTION="serial"
RULE_SCHEDULE="off"
RULE_EPOCHS="100"
RULE_PATIENCE="10"
RULE_MAX_EVALS="10"
RULE_SEED_BASE="1042"
RULE_N_BINS="5"
RULE_FEATURES_PER_EXPERT="99"
RULE_SEARCH_PROFILE="p1_shallow"
DROP_SPACE="0.05,0.1,0.15"
BAL_SPACE="0.003,0.01,0.03,0.05"

# emb,d_feat,d_expert,d_router,train_bs,eval_bs
RULE_COMBO_CATALOG="\
128,16,128,64,4096,8192;\
128,16,128,64,6144,12288;\
160,16,160,80,4096,8192;\
160,24,192,96,6144,12288;\
192,24,192,96,4096,8192;\
192,24,224,96,6144,12288;\
160,16,192,80,8192,16384;\
128,24,160,64,8192,16384"

# emb,d_feat,d_expert,d_router,expert_scale,train_bs,eval_bs
COMBO_CATALOG="\
128,16,128,64,3,3072,6144;\
128,16,128,64,3,4096,8192;\
128,16,160,64,3,4096,8192;\
128,24,160,64,3,4096,8192;\
128,16,192,80,3,8192,16384;\
128,16,256,80,3,6144,12288;\
128,16,512,64,3,4096,8192;\
128,16,512,96,3,3072,6144;\
160,16,160,80,3,4096,8192;\
160,16,192,80,3,6144,12288;\
160,16,256,96,3,4096,8192;\
160,24,192,96,3,6144,12288;\
160,24,256,96,3,6144,12288;\
192,16,192,96,3,4096,8192;\
192,16,320,112,3,4096,8192;\
192,24,224,96,3,6144,12288;\
192,24,320,112,3,6144,12288;\
192,24,512,128,3,4096,8192;\
224,16,256,96,3,6144,12288;\
224,16,384,128,3,4096,8192;\
224,24,384,128,3,6144,12288;\
256,16,256,112,3,4096,8192;\
256,16,512,128,3,4096,8192;\
96,32,192,192,2,4096,8192"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1]
          [--combos-per-gpu N] [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--serial-layout-id N] [--parallel-layout-id N] [--combo-catalog spec]
          [--phase-prefix P2DB] [--seed-base N]
          [--oom-retry-min-train-bs N]
          [--drop-space csv] [--balance-space csv]
          [--modes serial|parallel|both] [--split-after-serial] [--skip-serial]
          [--parallel-gpus 0,1,2,3] [--rule-gpus 4,5,6,7] [--rule-arms R0,R1]
          [--rule-max-evals 10] [--rule-combo-catalog spec]
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --combos-per-gpu) COMBOS_PER_GPU="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --serial-layout-id) SERIAL_LAYOUT_ID="$2"; shift 2 ;;
    --parallel-layout-id) PARALLEL_LAYOUT_ID="$2"; shift 2 ;;
    --combo-catalog) COMBO_CATALOG="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --oom-retry-min-train-bs) OOM_RETRY_MIN_TRAIN_BS="$2"; shift 2 ;;
    --drop-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --modes) MODES="$2"; shift 2 ;;
    --split-after-serial) SPLIT_AFTER_SERIAL="true"; shift ;;
    --no-split-after-serial) SPLIT_AFTER_SERIAL="false"; shift ;;
    --skip-serial) SKIP_SERIAL_ON_SPLIT="true"; shift ;;
    --no-skip-serial) SKIP_SERIAL_ON_SPLIT="false"; shift ;;
    --parallel-gpus) PARALLEL_GPU_LIST="$2"; shift 2 ;;
    --rule-gpus) RULE_GPU_LIST="$2"; shift 2 ;;
    --rule-arms) RULE_ARMS="$2"; shift 2 ;;
    --rule-layout-id) RULE_LAYOUT_ID="$2"; shift 2 ;;
    --rule-execution) RULE_EXECUTION="$2"; shift 2 ;;
    --rule-schedule) RULE_SCHEDULE="$2"; shift 2 ;;
    --rule-epochs) RULE_EPOCHS="$2"; shift 2 ;;
    --rule-patience) RULE_PATIENCE="$2"; shift 2 ;;
    --rule-max-evals) RULE_MAX_EVALS="$2"; shift 2 ;;
    --rule-seed-base) RULE_SEED_BASE="$2"; shift 2 ;;
    --rule-n-bins) RULE_N_BINS="$2"; shift 2 ;;
    --rule-feature-per-expert) RULE_FEATURES_PER_EXPERT="$2"; shift 2 ;;
    --rule-search-profile) RULE_SEARCH_PROFILE="$2"; shift 2 ;;
    --rule-combo-catalog) RULE_COMBO_CATALOG="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be positive integer" >&2
  exit 1
fi
case "${MODES,,}" in
  serial|parallel|both) ;;
  *) echo "--modes must be one of serial|parallel|both"; exit 1 ;;
esac
MODES="${MODES,,}"

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list"; exit 1; }

dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list"; exit 1; }

read_combo() {
  local idx="$1"
  python3 - <<'PY' "$COMBO_CATALOG" "$idx"
import sys
raw=sys.argv[1]
idx=int(sys.argv[2])
rows=[x.strip() for x in raw.split(';') if x.strip()]
if not rows:
    raise SystemExit('empty combo catalog')
row=rows[idx % len(rows)]
parts=[x.strip() for x in row.split(',')]
if len(parts) not in (6, 7):
    raise SystemExit(f'invalid combo row: {row}')
if len(parts) == 6:
    emb,dfeat,dexp,drouter,train_bs,eval_bs = parts
    parts = [emb,dfeat,dexp,drouter,'3',train_bs,eval_bs]
print(' '.join(parts))
PY
}

combo_count() {
  python3 - <<'PY' "$COMBO_CATALOG"
import sys
rows=[x.strip() for x in sys.argv[1].split(';') if x.strip()]
print(len(rows))
PY
}

read_rule_combo() {
  local idx="$1"
  python3 - <<'PY' "$RULE_COMBO_CATALOG" "$idx"
import sys
raw=sys.argv[1]
idx=int(sys.argv[2])
rows=[x.strip() for x in raw.split(';') if x.strip()]
if not rows:
    raise SystemExit('empty rule combo catalog')
row=rows[idx % len(rows)]
parts=[x.strip() for x in row.split(',')]
if len(parts)!=6:
    raise SystemExit(f'invalid rule combo row: {row}')
print(' '.join(parts))
PY
}

rule_combo_count() {
  python3 - <<'PY' "$RULE_COMBO_CATALOG"
import sys
rows=[x.strip() for x in sys.argv[1].split(';') if x.strip()]
print(len(rows))
PY
}

lr_space_for_bs_profile() {
  local train_bs="$1"
  local profile="$2"
  if [ "$train_bs" -le 4096 ]; then
    case "$profile" in
      0) echo "2e-4,3e-4,5e-4,7.5e-4,1e-3,1.5e-3,2e-3" ;;
      1) echo "3e-4,5e-4,1e-3,2e-3,3e-3,5e-3,7.5e-3" ;;
      *) echo "5e-4,1e-3,2e-3,3e-3,5e-3,7.5e-3,1e-2" ;;
    esac
  elif [ "$train_bs" -le 8192 ]; then
    case "$profile" in
      0) echo "3e-4,5e-4,7.5e-4,1e-3,1.5e-3,2e-3,3e-3" ;;
      1) echo "5e-4,1e-3,2e-3,3e-3,5e-3,7.5e-3,1e-2" ;;
      *) echo "7.5e-4,1.5e-3,2.5e-3,4e-3,6e-3,9e-3,1.2e-2" ;;
    esac
  else
    case "$profile" in
      0) echo "5e-4,1e-3,1.5e-3,2.5e-3,4e-3,6e-3,8e-3" ;;
      1) echo "7.5e-4,1.5e-3,3e-3,5e-3,7.5e-3,1e-2,1.25e-2" ;;
      *) echo "1e-3,2e-3,3.5e-3,5e-3,7.5e-3,1.1e-2,1.5e-2" ;;
    esac
  fi
}

apply_oom_safety_cap() {
  local emb="$1"
  local dfeat="$2"
  local dexp="$3"
  local drouter="$4"
  local expert_scale="$5"
  local train_bs="$6"
  local eval_bs="$7"

  local capped_train="$train_bs"
  local capped_eval="$eval_bs"

  if { [ "$dexp" -ge 512 ] || [ "$drouter" -ge 128 ]; } && [ "$capped_train" -gt 4096 ]; then
    capped_train=4096
    capped_eval=8192
  fi
  if [ "$emb" -ge 224 ] && { [ "$dexp" -ge 384 ] || [ "$drouter" -ge 112 ]; } && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if [ "$expert_scale" -ge 3 ] && [ "$dexp" -ge 384 ] && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi

  echo "${capped_train} ${capped_eval}"
}

wd_space_for_profile() {
  local profile="$1"
  case "$profile" in
    0) echo "0,1e-7,1e-6,1e-5,5e-5,1e-4" ;;
    1) echo "0,1e-6,1e-5,1e-4,1e-3,5e-3" ;;
    *) echo "0,1e-6,1e-5,1e-4,5e-4,1e-3" ;;
  esac
}

latest_phase_log() {
  local dataset="$1"
  local phase="$2"
  local dataset_tag model_tag dir
  dataset_tag="$(run_dataset_tag "$dataset")"
  model_tag="$(run_model_tag "FeaturedMoE_v3")"
  dir="$(run_log_dir fmoe_v3)/hparam/${PHASE_PREFIX}/${dataset_tag}/${model_tag}"
  [ -d "$dir" ] || { echo ""; return 0; }
  ls -1t "$dir"/*"_hparam_${phase}.log" 2>/dev/null | head -n1
}

is_oom_log() {
  local log_file="$1"
  [ -n "$log_file" ] || return 1
  [ -f "$log_file" ] || return 1
  grep -qi "CUDA out of memory" "$log_file"
}
COMBO_N="$(combo_count)"
RULE_COMBO_N="$(rule_combo_count)"

run_mode_jobs_for_dataset() {
  local ds="$1"
  shift
  local -a modes=("$@")

  local mode_names
  mode_names="$(IFS=','; echo "${modes[*]}")"
  echo "=== [${ds}] P2 dim/batch combo search (modes=${mode_names}; serial L${SERIAL_LAYOUT_ID}, parallel L${PARALLEL_LAYOUT_ID}) ==="
  echo "[P2DB] combo_catalog_size=${COMBO_N}, gpus=${#GPUS[@]}, combos_per_gpu=${COMBOS_PER_GPU}"

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for mode in "${modes[@]}"; do
        if [ "$mode" = "serial" ]; then
          layout_id="$SERIAL_LAYOUT_ID"
        else
          layout_id="$PARALLEL_LAYOUT_ID"
        fi

        for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
          combo_idx=$(( gidx * COMBOS_PER_GPU + slot ))
          read -r emb dfeat dexp drouter expert_scale train_bs eval_bs <<< "$(read_combo "$combo_idx")"
          read -r train_bs eval_bs <<< "$(apply_oom_safety_cap "$emb" "$dfeat" "$dexp" "$drouter" "$expert_scale" "$train_bs" "$eval_bs")"
          profile_id=$(( combo_idx % 3 ))
          lr_space="$(lr_space_for_bs_profile "$train_bs" "$profile_id")"
          wd_space="$(wd_space_for_profile "$profile_id")"
          if [ "$mode" = "parallel" ]; then
            mode_seed_offset=1000
          else
            mode_seed_offset=0
          fi
          seed=$((SEED_BASE + combo_idx + mode_seed_offset))
          phase="${PHASE_PREFIX}_G${gpu}_C$((slot+1))_${mode}_L${layout_id}_E${emb}_R${drouter}_B${train_bs}"

          exp_name="${PHASE_PREFIX}_${ds}_${mode}_dimbatch"
          exp_desc="${PHASE_PREFIX}: fixed ${mode}/L${layout_id}, combo(complexity: emb/feat/expert/router/scale/batch) sweep + LR/WD/dropout/balance(profile=${profile_id}) search."
          exp_focus="fmoe_stage_execution_mode,fmoe_v2_layout_id,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,hidden_dropout_prob,balance_loss_lambda,learning_rate,weight_decay"

          echo "[P2DB][${ds}] gpu=${gpu} mode=${mode} layout=${layout_id} combo=${combo_idx}/${COMBO_N} profile=${profile_id} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} escale=${expert_scale} bs=${train_bs}/${eval_bs}"

          cmd=(
            bash "${SCRIPT_DIR}/tune_hparam.sh"
            --dataset "$ds"
            --layout-id "$layout_id"
            --execution "$mode"
            --schedule-preset "$SCHEDULE_PRESET"
            --gpu "$gpu"
            --max-evals "$MAX_EVALS"
            --tune-epochs "$TUNE_EPOCHS"
            --tune-patience "$TUNE_PATIENCE"
            --seed "$seed"
            --phase "$phase"
            --search-profile p1_shallow
            --train-batch-size "$train_bs"
            --eval-batch-size "$eval_bs"
            --embedding-size "$emb"
            --num-heads 8
            --d-feat-emb "$dfeat"
            --d-expert-hidden "$dexp"
            --d-router-hidden "$drouter"
            --expert-scale "$expert_scale"
            --lr-space "$lr_space"
            --wd-space "$wd_space"
            --dropout-space "$DROP_SPACE"
            --balance-space "$BAL_SPACE"
            --exp-name "$exp_name"
            --exp-desc "$exp_desc"
            --exp-focus "$exp_focus"
          )

          if [ "$LOG_WANDB" = "true" ]; then
            cmd+=(--log-wandb)
          else
            cmd+=(--no-wandb)
          fi
          if [ "$DRY_RUN" = "true" ]; then
            cmd+=(--dry-run)
          fi

          set +e
          "${cmd[@]}"
          rc=$?
          set -e

          if [ "$rc" -ne 0 ]; then
            log_path="$(latest_phase_log "$ds" "$phase")"
            if is_oom_log "$log_path" && [ "$train_bs" -ge $((OOM_RETRY_MIN_TRAIN_BS * 2)) ]; then
              retry_train_bs=$(( train_bs / 2 ))
              retry_eval_bs=$(( eval_bs / 2 ))
              retry_lr_space="$(lr_space_for_bs_profile "$retry_train_bs" "$profile_id")"
              retry_phase="${phase}_RBS${retry_train_bs}"
              retry_desc="${exp_desc} OOM-retry(train/eval=${retry_train_bs}/${retry_eval_bs})."
              echo "[P2DB][OOM-RETRY] ${ds} ${mode} layout=${layout_id} phase=${phase} -> ${retry_phase} bs ${train_bs}/${eval_bs} -> ${retry_train_bs}/${retry_eval_bs}"
              retry_cmd=(
                bash "${SCRIPT_DIR}/tune_hparam.sh"
                --dataset "$ds"
                --layout-id "$layout_id"
                --execution "$mode"
                --schedule-preset "$SCHEDULE_PRESET"
                --gpu "$gpu"
                --max-evals "$MAX_EVALS"
                --tune-epochs "$TUNE_EPOCHS"
                --tune-patience "$TUNE_PATIENCE"
                --seed "$seed"
                --phase "$retry_phase"
                --search-profile p1_shallow
                --train-batch-size "$retry_train_bs"
                --eval-batch-size "$retry_eval_bs"
                --embedding-size "$emb"
                --num-heads 8
                --d-feat-emb "$dfeat"
                --d-expert-hidden "$dexp"
                --d-router-hidden "$drouter"
                --expert-scale "$expert_scale"
                --lr-space "$retry_lr_space"
                --wd-space "$wd_space"
                --dropout-space "$DROP_SPACE"
                --balance-space "$BAL_SPACE"
                --exp-name "$exp_name"
                --exp-desc "$retry_desc"
                --exp-focus "$exp_focus"
              )
              if [ "$LOG_WANDB" = "true" ]; then
                retry_cmd+=(--log-wandb)
              else
                retry_cmd+=(--no-wandb)
              fi
              if [ "$DRY_RUN" = "true" ]; then
                retry_cmd+=(--dry-run)
              fi
              "${retry_cmd[@]}"
            else
              exit "$rc"
            fi
          fi
        done
      done
    ) &
    WORKER_PIDS+=("$!")
  done

  FAIL=0
  for p in "${WORKER_PIDS[@]}"; do
    if ! wait "$p"; then
      FAIL=1
    fi
  done

  if [ "$FAIL" -ne 0 ]; then
    echo "[ERROR] P2 dim/batch combo failed for dataset=${ds}" >&2
    return 1
  fi
  return 0
}

run_rule_ablation_for_dataset() {
  local ds="$1"
  local -a rule_gpus rule_arms pids
  dispatch_parse_csv "$RULE_GPU_LIST" rule_gpus
  dispatch_parse_csv "$RULE_ARMS" rule_arms
  [ "${#rule_gpus[@]}" -eq 0 ] && { echo "[ERROR] Empty rule GPU list"; return 1; }
  [ "${#rule_arms[@]}" -eq 0 ] && { echo "[ERROR] Empty rule arm list"; return 1; }

  echo "=== [${ds}] fmoe_rule split-run (gpus=${RULE_GPU_LIST}, arms=${RULE_ARMS}, combos_per_gpu=${COMBOS_PER_GPU}, max_evals=${RULE_MAX_EVALS}) ==="
  pids=()
  local gidx
  for gidx in "${!rule_gpus[@]}"; do
    (
      set -euo pipefail
      local gpu arm slot combo_idx seed profile_id phase lr_space wd_space
      local emb dfeat dexp drouter train_bs eval_bs
      gpu="${rule_gpus[$gidx]}"
      arm="${rule_arms[$((gidx % ${#rule_arms[@]}))]}"
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        combo_idx=$(( gidx * COMBOS_PER_GPU + slot ))
        read -r emb dfeat dexp drouter train_bs eval_bs <<< "$(read_rule_combo "$combo_idx")"
        profile_id=$(( combo_idx % 3 ))
        lr_space="$(lr_space_for_bs_profile "$train_bs" "$profile_id")"
        wd_space="$(wd_space_for_profile "$profile_id")"
        seed=$((RULE_SEED_BASE + combo_idx))
        phase="RULE_${arm}_${PHASE_PREFIX}_G${gpu}_C$((slot+1))_${ds}_E${emb}_R${drouter}_B${train_bs}"
        echo "[RULE][${ds}] arm=${arm} gpu=${gpu} combo=${combo_idx}/${RULE_COMBO_N} profile=${profile_id} emb=${emb} dfeat=${dfeat} dexp=${dexp} drouter=${drouter} bs=${train_bs}/${eval_bs}"
        cmd=(
          bash "${RUN_DIR}/fmoe_rule/tune_hparam.sh"
          --dataset "$ds"
          --gpu "$gpu"
          --seed "$seed"
          --phase "$phase"
          --max-evals "$RULE_MAX_EVALS"
          --tune-epochs "$RULE_EPOCHS"
          --tune-patience "$RULE_PATIENCE"
          --layout-id "$RULE_LAYOUT_ID"
          --execution "$RULE_EXECUTION"
          --schedule "$RULE_SCHEDULE"
          --ablation "$arm"
          --rule-n-bins "$RULE_N_BINS"
          --rule-feature-per-expert "$RULE_FEATURES_PER_EXPERT"
          --search-profile "$RULE_SEARCH_PROFILE"
          --train-batch-size "$train_bs"
          --eval-batch-size "$eval_bs"
          --embedding-size "$emb"
          --num-heads 8
          --d-feat-emb "$dfeat"
          --d-expert-hidden "$dexp"
          --d-router-hidden "$drouter"
          --expert-scale 3
          --lr-space "$lr_space"
          --wd-space "$wd_space"
          --exp-name "rule_split_${PHASE_PREFIX}_${arm}_${ds}"
          --exp-desc "Split run: rule=${arm}, fixed combo(dim/router/batch), tune lr/wd(max_evals=${RULE_MAX_EVALS})."
          --exp-focus "ablation,router_impl,router_impl_by_stage,rule_router.n_bins,rule_router.feature_per_expert,fmoe_v2_layout_id,fmoe_stage_execution_mode,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,train_batch_size,eval_batch_size,learning_rate,weight_decay"
        )
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi
        "${cmd[@]}"
      done
    ) &
    pids+=("$!")
  done
  local fail=0 p
  for p in "${pids[@]}"; do
    if ! wait "$p"; then
      fail=1
    fi
  done
  [ "$fail" -eq 0 ]
}

EXP_DIR="$(run_experiments_dir)"
cd "$EXP_DIR"
run_export_runtime_env

for ds in "${DATASET_ARR[@]}"; do
  if [ "$SPLIT_AFTER_SERIAL" = "true" ] && [ "$MODES" = "both" ]; then
    if [ "$SKIP_SERIAL_ON_SPLIT" = "true" ]; then
      echo "=== [${ds}] split mode: skip serial stage and start parallel+rule directly ==="
    else
      if ! run_mode_jobs_for_dataset "$ds" "serial"; then
        echo "[ERROR] P2 serial stage failed for dataset=${ds}" >&2
        exit 1
      fi
    fi
    if [ "$SKIP_SERIAL_ON_SPLIT" = "true" ]; then
      echo "=== [${ds}] start split run: parallel(${PARALLEL_GPU_LIST}) + rule(${RULE_GPU_LIST}) ==="
    else
      echo "=== [${ds}] serial stage done; start split run: parallel(${PARALLEL_GPU_LIST}) + rule(${RULE_GPU_LIST}) ==="
    fi

    parallel_cmd=(
      bash "${SCRIPT_DIR}/p2_dim_batch_combo.sh"
      --datasets "$ds"
      --gpus "$PARALLEL_GPU_LIST"
      --combos-per-gpu "$COMBOS_PER_GPU"
      --max-evals "$MAX_EVALS"
      --tune-epochs "$TUNE_EPOCHS"
      --tune-patience "$TUNE_PATIENCE"
      --serial-layout-id "$SERIAL_LAYOUT_ID"
      --parallel-layout-id "$PARALLEL_LAYOUT_ID"
      --combo-catalog "$COMBO_CATALOG"
      --phase-prefix "$PHASE_PREFIX"
      --seed-base "$SEED_BASE"
      --oom-retry-min-train-bs "$OOM_RETRY_MIN_TRAIN_BS"
      --modes parallel
      --no-split-after-serial
    )
    if [ "$LOG_WANDB" = "true" ]; then
      parallel_cmd+=(--log-wandb)
    else
      parallel_cmd+=(--no-wandb)
    fi
    if [ "$DRY_RUN" = "true" ]; then
      parallel_cmd+=(--dry-run)
    fi

    "${parallel_cmd[@]}" &
    P_PAR="$!"
    run_rule_ablation_for_dataset "$ds" &
    P_RULE="$!"
    wait "$P_PAR" || { echo "[ERROR] split parallel stage failed for dataset=${ds}" >&2; exit 1; }
    wait "$P_RULE" || { echo "[ERROR] split rule stage failed for dataset=${ds}" >&2; exit 1; }
  else
    case "$MODES" in
      serial) if ! run_mode_jobs_for_dataset "$ds" "serial"; then exit 1; fi ;;
      parallel) if ! run_mode_jobs_for_dataset "$ds" "parallel"; then exit 1; fi ;;
      both) if ! run_mode_jobs_for_dataset "$ds" "serial" "parallel"; then exit 1; fi ;;
    esac
  fi
  echo "=== [${ds}] P2 dim/batch combo done ==="
done

run_update_track_report fmoe_v3
