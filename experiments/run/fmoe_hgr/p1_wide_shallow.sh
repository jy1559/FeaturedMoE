#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASETS="movielens1m"
GPU_LIST="0,1,2,3"
COMBOS_PER_GPU="20"
MAX_EVALS="10"
TUNE_EPOCHS="20"
TUNE_PATIENCE="5"
SEED_BASE="420"
PHASE_PREFIX="P1HGR_widewide"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
LR_SPACE="5e-5,1e-1"
WD_SPACE="0,1e-6,1e-5,1e-4,1e-3"
DROP_SPACE="0.08,0.18"
BAL_SPACE="0.001,0.05"
EXP_NAME_BASE="P1_hgr_wide_shallow"
EXP_DESC_BASE="Wide-shallow HGR screen for combo pruning. Half of the budget stays on the base extended layout [1,0,1,0,1,0,1,0] to isolate routing/capacity effects; the other half stresses layout diversity, with almost all layouts in the 2-6 total-attention range plus two heavy outliers at total-attn 7 and 8."
EXP_FOCUS="stage_merge_mode,group_router_mode,arch_layout_id,group_top_k,moe_top_k,expert_use_feature,macro_routing_scope,parallel_stage_gate_temperature,embedding_size,d_feat_emb,d_expert_hidden,d_router_hidden,expert_scale,train_batch_size,eval_batch_size,learning_rate,weight_decay"

BASE_LAYOUT_ID="0"
DIVERSE_LAYOUT_IDS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
HEAVY_LAYOUT_IDS="21,22"

usage() {
  cat <<USAGE
Usage: $0 [--datasets movielens1m,retail_rocket] [--gpus 0,1]
          [--combos-per-gpu N] [--max-evals N] [--tune-epochs N] [--tune-patience N]
          [--base-layout-id N] [--diverse-layout-ids csv] [--heavy-layout-ids csv]
          [--lr-space csv] [--wd-space csv]
          [--dropout-space csv] [--balance-space csv]
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
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --base-layout-id) BASE_LAYOUT_ID="$2"; shift 2 ;;
    --diverse-layout-ids) DIVERSE_LAYOUT_IDS="$2"; shift 2 ;;
    --heavy-layout-ids) HEAVY_LAYOUT_IDS="$2"; shift 2 ;;
    --lr-space) LR_SPACE="$2"; shift 2 ;;
    --wd-space) WD_SPACE="$2"; shift 2 ;;
    --dropout-space) DROP_SPACE="$2"; shift 2 ;;
    --balance-space) BAL_SPACE="$2"; shift 2 ;;
    --exp-name-base) EXP_NAME_BASE="$2"; shift 2 ;;
    --exp-desc-base) EXP_DESC_BASE="$2"; shift 2 ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if ! [[ "$COMBOS_PER_GPU" =~ ^[0-9]+$ ]] || [ "$COMBOS_PER_GPU" -le 0 ]; then
  echo "--combos-per-gpu must be positive integer" >&2
  exit 1
fi
if ! [[ "$MAX_EVALS" =~ ^[0-9]+$ ]] || [ "$MAX_EVALS" -le 0 ]; then
  echo "--max-evals must be positive integer" >&2
  exit 1
fi

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 0 ] && { echo "Empty GPU list" >&2; exit 1; }
dispatch_parse_csv "$DATASETS" DATASET_ARR
[ "${#DATASET_ARR[@]}" -eq 0 ] && { echo "Empty dataset list" >&2; exit 1; }

generate_combo_rows() {
  python3 - <<'PY' "$BASE_LAYOUT_ID" "$DIVERSE_LAYOUT_IDS" "$HEAVY_LAYOUT_IDS"
import sys

base_layout_id = int(sys.argv[1])
diverse_layout_ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
heavy_layout_ids = [int(x) for x in sys.argv[3].split(",") if x.strip()]

base_route_profiles = [
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "off"),
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "alpha_cold"),
    ("serial", "hybrid", 0, 0, "false", "session", "query", "1.0", "off"),
    ("serial", "stage_wide", 0, 0, "false", "token", "query", "1.0", "off"),
    ("parallel", "per_group", 0, 0, "false", "session", "query", "1.0", "off"),
    ("parallel", "hybrid", 2, 0, "false", "session", "query", "0.8", "combined_legacy"),
    ("serial", "per_group", 0, 0, "true", "session", "query", "1.0", "off"),
    ("parallel", "hybrid", 2, 0, "true", "session", "query", "1.2", "alpha_temp_cold"),
]

base_capacity_profiles = [
    (128, 16, 160, 64, 3, 4096, 8192),   # anchor
    (128, 16, 192, 80, 3, 3072, 6144),   # moderate router/ffn increase
    (160, 16, 224, 96, 3, 3072, 6144),   # wider hidden/out router
    (96, 16, 128, 48, 2, 6144, 12288),   # under-capacity outlier
    (128, 24, 256, 128, 4, 2048, 4096),  # heavy outlier
]

diverse_route_profiles = [
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "alpha_cold"),
    ("parallel", "hybrid", 2, 0, "false", "session", "query", "1.0", "off"),
]

diverse_capacity = (128, 16, 160, 64, 3, 4096, 8192)
heavy_specs = [
    ("serial", "hybrid", 0, 0, "false", "session", "query", "1.0", "alpha_cold", 160, 16, 224, 96, 3, 3072, 6144),
    ("parallel", "hybrid", 2, 0, "true", "session", "query", "0.9", "alpha_temp_cold", 128, 24, 256, 128, 4, 2048, 4096),
]

anchor_rows = []
for merge_mode, group_mode, group_top_k, moe_top_k, expert_feat, macro_scope, macro_pool, par_temp, sched in base_route_profiles:
    for emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs in base_capacity_profiles:
        anchor_rows.append(
            (
                base_layout_id,
                merge_mode,
                group_mode,
                group_top_k,
                moe_top_k,
                emb,
                d_feat,
                d_exp,
                d_router,
                expert_scale,
                train_bs,
                eval_bs,
                expert_feat,
                macro_scope,
                macro_pool,
                par_temp,
                sched,
            )
        )

regular_diverse_rows = []
for layout_id in diverse_layout_ids:
    for merge_mode, group_mode, group_top_k, moe_top_k, expert_feat, macro_scope, macro_pool, par_temp, sched in diverse_route_profiles:
        emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs = diverse_capacity
        regular_diverse_rows.append(
            (
                layout_id,
                merge_mode,
                group_mode,
                group_top_k,
                moe_top_k,
                emb,
                d_feat,
                d_exp,
                d_router,
                expert_scale,
                train_bs,
                eval_bs,
                expert_feat,
                macro_scope,
                macro_pool,
                par_temp,
                sched,
            )
        )

heavy_rows = []
for layout_id, spec in zip(heavy_layout_ids, heavy_specs):
    (
        merge_mode,
        group_mode,
        group_top_k,
        moe_top_k,
        expert_feat,
        macro_scope,
        macro_pool,
        par_temp,
        sched,
        emb,
        d_feat,
        d_exp,
        d_router,
        expert_scale,
        train_bs,
        eval_bs,
    ) = spec
    heavy_rows.append(
        (
            layout_id,
            merge_mode,
            group_mode,
            group_top_k,
            moe_top_k,
            emb,
            d_feat,
            d_exp,
            d_router,
            expert_scale,
            train_bs,
            eval_bs,
            expert_feat,
            macro_scope,
            macro_pool,
            par_temp,
            sched,
        )
    )

diverse_rows = heavy_rows + regular_diverse_rows

if len(anchor_rows) != 40:
    raise SystemExit(f"expected 40 anchor rows, got {len(anchor_rows)}")
if len(regular_diverse_rows) != 38:
    raise SystemExit(f"expected 38 regular diverse rows, got {len(regular_diverse_rows)}")
if len(heavy_rows) != len(heavy_layout_ids):
    raise SystemExit(f"expected heavy rows to match heavy layouts, got {len(heavy_rows)} vs {len(heavy_layout_ids)}")
if len(diverse_rows) != 40:
    raise SystemExit(f"expected 40 diverse rows, got {len(diverse_rows)}")

rows = []
for idx in range(max(len(anchor_rows), len(diverse_rows))):
    if idx < len(anchor_rows):
        rows.append(anchor_rows[idx])
    if idx < len(diverse_rows):
        rows.append(diverse_rows[idx])

for row in rows:
    print(",".join(str(x) for x in row))
PY
}

read_combo() {
  local idx="$1"
  python3 - <<'PY' "$idx" "$BASE_LAYOUT_ID" "$DIVERSE_LAYOUT_IDS" "$HEAVY_LAYOUT_IDS"
import sys

target_idx = int(sys.argv[1])
base_layout_id = int(sys.argv[2])
diverse_layout_ids = [int(x) for x in sys.argv[3].split(",") if x.strip()]
heavy_layout_ids = [int(x) for x in sys.argv[4].split(",") if x.strip()]

base_route_profiles = [
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "off"),
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "alpha_cold"),
    ("serial", "hybrid", 0, 0, "false", "session", "query", "1.0", "off"),
    ("serial", "stage_wide", 0, 0, "false", "token", "query", "1.0", "off"),
    ("parallel", "per_group", 0, 0, "false", "session", "query", "1.0", "off"),
    ("parallel", "hybrid", 2, 0, "false", "session", "query", "0.8", "combined_legacy"),
    ("serial", "per_group", 0, 0, "true", "session", "query", "1.0", "off"),
    ("parallel", "hybrid", 2, 0, "true", "session", "query", "1.2", "alpha_temp_cold"),
]
base_capacity_profiles = [
    (128, 16, 160, 64, 3, 4096, 8192),
    (128, 16, 192, 80, 3, 3072, 6144),
    (160, 16, 224, 96, 3, 3072, 6144),
    (96, 16, 128, 48, 2, 6144, 12288),
    (128, 24, 256, 128, 4, 2048, 4096),
]
diverse_route_profiles = [
    ("serial", "per_group", 0, 0, "false", "session", "query", "1.0", "alpha_cold"),
    ("parallel", "hybrid", 2, 0, "false", "session", "query", "1.0", "off"),
]
diverse_capacity = (128, 16, 160, 64, 3, 4096, 8192)
heavy_specs = [
    ("serial", "hybrid", 0, 0, "false", "session", "query", "1.0", "alpha_cold", 160, 16, 224, 96, 3, 3072, 6144),
    ("parallel", "hybrid", 2, 0, "true", "session", "query", "0.9", "alpha_temp_cold", 128, 24, 256, 128, 4, 2048, 4096),
]

anchor_rows = []
for merge_mode, group_mode, group_top_k, moe_top_k, expert_feat, macro_scope, macro_pool, par_temp, sched in base_route_profiles:
    for emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs in base_capacity_profiles:
        anchor_rows.append((base_layout_id, merge_mode, group_mode, group_top_k, moe_top_k, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs, expert_feat, macro_scope, macro_pool, par_temp, sched))
regular_diverse_rows = []
for layout_id in diverse_layout_ids:
    for merge_mode, group_mode, group_top_k, moe_top_k, expert_feat, macro_scope, macro_pool, par_temp, sched in diverse_route_profiles:
        emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs = diverse_capacity
        regular_diverse_rows.append((layout_id, merge_mode, group_mode, group_top_k, moe_top_k, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs, expert_feat, macro_scope, macro_pool, par_temp, sched))
heavy_rows = []
for layout_id, spec in zip(heavy_layout_ids, heavy_specs):
    merge_mode, group_mode, group_top_k, moe_top_k, expert_feat, macro_scope, macro_pool, par_temp, sched, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs = spec
    heavy_rows.append((layout_id, merge_mode, group_mode, group_top_k, moe_top_k, emb, d_feat, d_exp, d_router, expert_scale, train_bs, eval_bs, expert_feat, macro_scope, macro_pool, par_temp, sched))
diverse_rows = heavy_rows + regular_diverse_rows

rows = []
for idx in range(max(len(anchor_rows), len(diverse_rows))):
    if idx < len(anchor_rows):
        rows.append(anchor_rows[idx])
    if idx < len(diverse_rows):
        rows.append(diverse_rows[idx])

row = rows[target_idx % len(rows)]
print(" ".join(str(x) for x in row))
PY
}

combo_count() {
  python3 - <<'PY' "$BASE_LAYOUT_ID" "$DIVERSE_LAYOUT_IDS" "$HEAVY_LAYOUT_IDS"
import sys
base_layout_id = int(sys.argv[1])
diverse_layout_ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
heavy_layout_ids = [int(x) for x in sys.argv[3].split(",") if x.strip()]
base_route_profiles = 8
base_capacity_profiles = 5
diverse_route_profiles = 2
print(base_route_profiles * base_capacity_profiles + len(diverse_layout_ids) * diverse_route_profiles + len(heavy_layout_ids))
PY
}

apply_oom_safety_cap() {
  local dataset="$1"
  local emb="$2"
  local d_exp="$3"
  local d_router="$4"
  local expert_scale="$5"
  local train_bs="$6"
  local eval_bs="$7"

  local capped_train="$train_bs"
  local capped_eval="$eval_bs"

  if [ "$dataset" = "retail_rocket" ] && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if { [ "$d_exp" -ge 224 ] || [ "$d_router" -ge 96 ]; } && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi
  if [ "$expert_scale" -ge 4 ] && [ "$capped_train" -gt 2048 ]; then
    capped_train=2048
    capped_eval=4096
  fi
  if [ "$emb" -ge 160 ] && [ "$capped_train" -gt 3072 ]; then
    capped_train=3072
    capped_eval=6144
  fi

  echo "${capped_train} ${capped_eval}"
}

INTERRUPTED="false"
WORKER_PIDS=()
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping HGR P1 workers..."
  local p
  for p in "${WORKER_PIDS[@]:-}"; do
    kill -TERM "$p" 2>/dev/null || true
  done
  wait || true
  exit 130
}
trap on_interrupt INT TERM

combo_total="$(combo_count)"
anchor_total=40
diverse_total=$(( combo_total - anchor_total ))
echo "[P1-HGR] combo_count=${combo_total} anchor_layout_rows=${anchor_total} diverse_layout_rows=${diverse_total} heavy_layout_ids=${HEAVY_LAYOUT_IDS} combos_per_gpu=${COMBOS_PER_GPU} gpus=${GPU_LIST}"

for ds in "${DATASET_ARR[@]}"; do
  total_jobs=$(( ${#GPUS[@]} * COMBOS_PER_GPU ))
  echo "=== [${ds}] HGR P1 wide-shallow (${total_jobs} runs = ${#GPUS[@]} gpus x ${COMBOS_PER_GPU}) ==="
  if [ "$total_jobs" -lt "$combo_total" ]; then
    echo "[P1-HGR] warning: total_jobs=${total_jobs} < combo_total=${combo_total}; only the first interleaved subset will run."
  elif [ "$total_jobs" -gt "$combo_total" ]; then
    echo "[P1-HGR] warning: total_jobs=${total_jobs} > combo_total=${combo_total}; some combos will repeat."
  fi

  WORKER_PIDS=()
  for gidx in "${!GPUS[@]}"; do
    gpu="${GPUS[$gidx]}"
    (
      set -euo pipefail
      for slot in $(seq 0 $((COMBOS_PER_GPU - 1))); do
        idx=$(( gidx * COMBOS_PER_GPU + slot ))
        seed=$(( SEED_BASE + idx ))
        combo_id=$(( idx % combo_total ))
        read -r layout_id merge_mode group_mode group_topk moe_topk emb dfeat dexp drouter scale train_bs eval_bs expert_feat macro_scope macro_pool par_temp schedule_preset <<< "$(read_combo "$combo_id")"
        read -r safe_train_bs safe_eval_bs <<< "$(apply_oom_safety_cap "$ds" "$emb" "$dexp" "$drouter" "$scale" "$train_bs" "$eval_bs")"

        phase="${PHASE_PREFIX}_C$(printf '%02d' "$combo_id")_${merge_mode}_${group_mode}"
        exp_name="${EXP_NAME_BASE}"
        exp_desc="${EXP_DESC_BASE} combo=C${combo_id} layout=${layout_id} merge=${merge_mode} group=${group_mode} group_topk=${group_topk} moe_topk=${moe_topk} expert_feat=${expert_feat} macro_scope=${macro_scope} gate_temp=${par_temp} sched=${schedule_preset}"

        cmd=(
          bash "${SCRIPT_DIR}/tune_hparam.sh"
          --dataset "${ds}"
          --gpu "${gpu}"
          --max-evals "${MAX_EVALS}"
          --tune-epochs "${TUNE_EPOCHS}"
          --tune-patience "${TUNE_PATIENCE}"
          --seed "${seed}"
          --phase "${phase}"
          --search-profile wide
          --schedule-preset "${schedule_preset}"
          --layout-id "${layout_id}"
          --stage-merge-mode "${merge_mode}"
          --group-router-mode "${group_mode}"
          --group-top-k "${group_topk}"
          --moe-top-k "${moe_topk}"
          --expert-use-feature "${expert_feat}"
          --macro-routing-scope "${macro_scope}"
          --macro-session-pooling "${macro_pool}"
          --parallel-stage-gate-temperature "${par_temp}"
          --train-batch-size "${safe_train_bs}"
          --eval-batch-size "${safe_eval_bs}"
          --embedding-size "${emb}"
          --d-feat-emb "${dfeat}"
          --d-expert-hidden "${dexp}"
          --d-router-hidden "${drouter}"
          --expert-scale "${scale}"
          --lr-space "${LR_SPACE}"
          --wd-space "${WD_SPACE}"
          --dropout-space "${DROP_SPACE}"
          --balance-space "${BAL_SPACE}"
          --exp-name "${exp_name}"
          --exp-desc "${exp_desc}"
          --exp-focus "${EXP_FOCUS}"
        )

        if [ "$LOG_WANDB" = "true" ]; then
          cmd+=(--log-wandb)
        else
          cmd+=(--no-wandb)
        fi
        if [ "$DRY_RUN" = "true" ]; then
          cmd+=(--dry-run)
        fi

        echo "[P1-HGR] gpu=${gpu} slot=${slot} combo=C${combo_id} layout=${layout_id} merge=${merge_mode} group=${group_mode} gtopk=${group_topk} etopk=${moe_topk} expert_feat=${expert_feat} macro=${macro_scope} gate_temp=${par_temp} sched=${schedule_preset} bs=${safe_train_bs}/${safe_eval_bs}"
        "${cmd[@]}"
      done
    ) &
    WORKER_PIDS+=("$!")
  done

  wait
  if [ "$INTERRUPTED" = "true" ]; then
    exit 130
  fi
done

run_update_track_report fmoe_hgr
