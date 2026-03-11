#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

GPU_LIST="0,1,2,3"
SEED_BASE="6200"
PHASE_PREFIX="P05"
MAX_EVALS="4"
TUNE_EPOCHS="45"
TUNE_PATIENCE="6"
AUTO_PRUNE_RELATIVE="true"
SIGMA_THRESHOLD="1.0"
MAX_DROP_KUAI="3"
MAX_DROP_LFM="1"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
MANIFEST_PATH=""
MANIFEST_OUT=""
PLAN_ONLY="false"

usage() {
  cat <<USAGE
Usage: $0 [--manifest path] [--gpus 0,1,2,3] [--seed-base 6200]
          [--sigma-threshold 1.0] [--max-drop-kuai 3] [--max-drop-lfm 1]
          [--auto-prune-relative|--no-auto-prune-relative]
          [--plan-only] [--dry-run]
USAGE
}

write_status_json() {
  local status_path="$1"
  local combo_id="$2"
  local dataset="$3"
  local wave="$4"
  local gpu_slot="$5"
  local gpu_id="$6"
  local phase="$7"
  local status="$8"
  local return_code="$9"
  local result_path="${10}"
  local log_path="${11}"
  python3 - <<'PY' "$status_path" "$combo_id" "$dataset" "$wave" "$gpu_slot" "$gpu_id" "$phase" "$status" "$return_code" "$result_path" "$log_path"
import json
import sys

status_path, combo_id, dataset, wave, gpu_slot, gpu_id, phase, status, return_code, result_path, log_path = sys.argv[1:]
payload = {
    "combo_id": combo_id,
    "dataset": dataset,
    "wave": int(wave),
    "gpu_slot": int(gpu_slot),
    "gpu_id": gpu_id,
    "phase": phase,
    "status": status,
    "return_code": int(return_code),
    "result_path": result_path or "",
    "log_path": log_path or "",
}
with open(status_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

merge_manifest() {
  local plan_path="$1"
  local status_dir="$2"
  local manifest_path="$3"
  python3 - <<'PY' "$plan_path" "$status_dir" "$manifest_path"
import json
from pathlib import Path
import sys

plan_path = Path(sys.argv[1])
status_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])

plan = json.loads(plan_path.read_text(encoding="utf-8"))
status_map = {}
for path in sorted(status_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    status_map[data["combo_id"]] = data

for combo in plan.get("kept_combos", []):
    status = status_map.get(combo["combo_id"])
    if status:
        combo.update(status)
        result_path = Path(status.get("result_path") or "")
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                combo["best_mrr@20"] = result.get("best_mrr@20")
            except Exception as exc:
                combo["result_read_error"] = str(exc)

manifest_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
PY
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest) MANIFEST_PATH="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --auto-prune-relative) AUTO_PRUNE_RELATIVE="true"; shift ;;
    --no-auto-prune-relative) AUTO_PRUNE_RELATIVE="false"; shift ;;
    --sigma-threshold) SIGMA_THRESHOLD="$2"; shift 2 ;;
    --max-drop-kuai) MAX_DROP_KUAI="$2"; shift 2 ;;
    --max-drop-lfm) MAX_DROP_LFM="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
    --plan-only) PLAN_ONLY="true"; shift ;;
    --log-wandb) LOG_WANDB="true"; shift ;;
    --no-wandb) LOG_WANDB="false"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 4 ] || { echo "--gpus must contain exactly 4 gpu ids" >&2; exit 1; }

INV_DIR="$(run_inventory_dir)/fmoe_n"
run_ensure_dir "$INV_DIR"

if [ -z "$MANIFEST_PATH" ]; then
  MANIFEST_PATH="${INV_DIR}/p0_manifest_latest.json"
fi
[ -f "$MANIFEST_PATH" ] || { echo "manifest not found: ${MANIFEST_PATH}" >&2; exit 1; }

RUN_TAG="$(run_timestamp)"
WORK_DIR="${INV_DIR}/p05_${RUN_TAG}"
STATUS_DIR="${WORK_DIR}/status"
run_ensure_dir "$WORK_DIR"
run_ensure_dir "$STATUS_DIR"

PLAN_PATH="${WORK_DIR}/p05_plan.json"
if [ -z "$MANIFEST_OUT" ]; then
  MANIFEST_OUT="${INV_DIR}/p05_manifest_${RUN_TAG}.json"
fi
LATEST_MANIFEST="${INV_DIR}/p05_manifest_latest.json"

python3 - <<'PY' "$MANIFEST_PATH" "$PLAN_PATH" "$GPU_LIST" "$AUTO_PRUNE_RELATIVE" "$SIGMA_THRESHOLD" "$MAX_DROP_KUAI" "$MAX_DROP_LFM"
import json
from collections import Counter
from pathlib import Path
import math
import statistics
import sys

manifest_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
gpu_csv = sys.argv[3]
auto_prune_relative = sys.argv[4].lower() == "true"
sigma_threshold = float(sys.argv[5])
max_drop_kuai = int(sys.argv[6])
max_drop_lfm = int(sys.argv[7])
gpus = [x.strip() for x in gpu_csv.split(",") if x.strip()]

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
combos = manifest.get("combos", [])

def load_result(row):
    path = row.get("result_path") or ""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

eligible = []
excluded = []
for row in combos:
    result = load_result(row)
    if row.get("status") != "success" or result is None:
        reason = "missing_successful_parent"
        excluded.append({"combo_id": row.get("combo_id"), "dataset": row.get("dataset"), "reason": reason})
        continue
    best_params = result.get("best_params") or {}
    fixed_search = result.get("fixed_search") or {}
    best_lr = best_params.get("learning_rate", fixed_search.get("learning_rate"))
    best_wd = best_params.get("weight_decay", fixed_search.get("weight_decay", 5e-5))
    best_drop = best_params.get("hidden_dropout_prob", fixed_search.get("hidden_dropout_prob", 0.1))
    best_bal = best_params.get("balance_loss_lambda", fixed_search.get("balance_loss_lambda", row.get("balance_loss_lambda", 0.002)))
    score = result.get("best_mrr@20")
    if best_lr is None or score is None:
        excluded.append({"combo_id": row.get("combo_id"), "dataset": row.get("dataset"), "reason": "missing_best_lr_or_score"})
        continue
    enriched = dict(row)
    enriched["parent_result"] = str(Path(row["result_path"]).resolve())
    enriched["score"] = float(score)
    enriched["best_lr"] = float(best_lr)
    enriched["best_weight_decay"] = float(best_wd)
    enriched["best_hidden_dropout_prob"] = float(best_drop)
    enriched["best_balance_loss_lambda"] = float(best_bal)
    enriched["effective_train_batch_size"] = int(row.get("effective_train_batch_size") or row.get("train_batch_size"))
    enriched["effective_eval_batch_size"] = int(row.get("effective_eval_batch_size") or row.get("eval_batch_size"))
    eligible.append(enriched)

dataset_layout_floor = {
    "KuaiRecSmall0.1": {"L7", "L15", "L16", "L19"},
    "lastfm0.03": {"L7", "L16"},
}
strict_family_floor = {"KuaiRecSmall0.1": True, "lastfm0.03": False}

for row in eligible:
    row["layout_name"] = f"L{int(row['layout_id'])}"

stats = {}
for dataset in sorted({r["dataset"] for r in eligible}):
    ds_rows = [r for r in eligible if r["dataset"] == dataset]
    scores = [r["score"] for r in ds_rows]
    mean = statistics.mean(scores) if scores else 0.0
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    stats[dataset] = {"mean": mean, "std": std}

dropped_ids = set()
drop_reasons = {}

def violates_floor(dataset, kept_rows):
    required_layouts = dataset_layout_floor[dataset]
    layout_counts = Counter(r["layout_name"] for r in kept_rows)
    if any(layout_counts.get(x, 0) <= 0 for x in required_layouts):
        return True
    if strict_family_floor[dataset]:
        fam_counts = Counter(r["family"] for r in kept_rows)
        if any(fam_counts.get(x, 0) <= 0 for x in ("plain", "hybrid", "bias")):
            return True
    return False

if auto_prune_relative:
    for dataset in ("KuaiRecSmall0.1", "lastfm0.03"):
        ds_rows = [r for r in eligible if r["dataset"] == dataset]
        if not ds_rows:
            continue
        mean = stats[dataset]["mean"]
        std = stats[dataset]["std"]
        threshold = mean - sigma_threshold * std
        candidates = [
            r for r in ds_rows
            if std > 0 and r["score"] < threshold
        ]
        candidates.sort(key=lambda r: (r["score"], 1 if r["combo_id"] == "Q16" else 0, r["combo_id"]))
        cap = max_drop_kuai if dataset == "KuaiRecSmall0.1" else max_drop_lfm
        for cand in candidates:
            if sum(1 for x in dropped_ids if x.startswith("Q")) >= cap and dataset == "KuaiRecSmall0.1":
                break
            if sum(1 for x in dropped_ids if x.startswith("F")) >= cap and dataset == "lastfm0.03":
                break
            proposal = [r for r in ds_rows if r["combo_id"] not in dropped_ids and r["combo_id"] != cand["combo_id"]]
            if violates_floor(dataset, proposal):
                continue
            dropped_ids.add(cand["combo_id"])
            drop_reasons[cand["combo_id"]] = {
                "reason": "relative_sigma_drop",
                "score": cand["score"],
                "threshold": threshold,
                "mean": mean,
                "std": std,
            }

kept = []
dropped = []
for row in eligible:
    low = max(float(row["lr_min"]), float(row["best_lr"]) * 0.6)
    high = min(float(row["lr_max"]), float(row["best_lr"]) * 1.5)
    row["narrow_lr_min"] = low
    row["narrow_lr_max"] = high
    row["assigned_gpu"] = gpus[int(row["assigned_gpu_slot"])]
    if row["combo_id"] in dropped_ids:
        payload = dict(row)
        payload.update(drop_reasons[row["combo_id"]])
        dropped.append(payload)
    else:
        kept.append(row)

plan = {
    "track": "fmoe_n",
    "phase": "P0.5",
    "source_manifest": str(manifest_path.resolve()),
    "draft_note": "P0.5 is intentionally draft. Revisit sigma/drop policy after first real P0 results.",
    "auto_prune_relative": auto_prune_relative,
    "sigma_threshold": sigma_threshold,
    "max_drop_kuai": max_drop_kuai,
    "max_drop_lfm": max_drop_lfm,
    "dataset_stats": stats,
    "excluded": excluded,
    "dropped_combos": dropped,
    "kept_combos": kept,
}
plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
PY

echo "[P0.5] source manifest=${MANIFEST_PATH}"
echo "[P0.5] plan=${PLAN_PATH}"
echo "[P0.5] note=draft; revise after first real P0 results"

python3 - <<'PY' "$PLAN_PATH"
import json
import sys

plan = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(f"[P0.5] kept={len(plan.get('kept_combos', []))} dropped={len(plan.get('dropped_combos', []))} excluded={len(plan.get('excluded', []))}")
for row in plan.get("dropped_combos", []):
    print(f"[P0.5][drop] {row['combo_id']} dataset={row['dataset']} score={row['score']:.6f} threshold={row['threshold']:.6f}")
PY

if [ "$PLAN_ONLY" = "true" ]; then
  exit 0
fi

if [ "$DRY_RUN" = "true" ]; then
  echo "[P0.5] dry-run only; commands will not execute"
fi

FAIL_COUNT=0
INTERRUPTED="false"
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping P0.5 workers..."
  dispatch_terminate_all GPUS
  exit 130
}
trap on_interrupt INT TERM

for wave in 1 2 3 4 5 6; do
  echo "=== [P0.5] wave ${wave} ==="
  mapfile -t wave_rows < <(python3 - <<'PY' "$PLAN_PATH" "$wave"
import json
import sys

plan = json.load(open(sys.argv[1], "r", encoding="utf-8"))
wave = int(sys.argv[2])
for row in plan.get("kept_combos", []):
    if int(row["wave"]) == wave:
        print("|".join([
            row["combo_id"],
            row["dataset"],
            str(row["wave"]),
            row["family"],
            str(row["layout_id"]),
            row["execution"],
            str(row["effective_train_batch_size"]),
            str(row["effective_eval_batch_size"]),
            str(row["narrow_lr_min"]),
            str(row["narrow_lr_max"]),
            str(row["d_feat_emb"]),
            str(row["d_expert_hidden"]),
            str(row["expert_scale"]),
            str(row["moe_top_k"]),
            row["moe_top_k_policy"],
            row["feature_encoder_mode"],
            str(row["best_balance_loss_lambda"]),
            row["router_impl_by_stage"],
            str(row["rule_bias_scale"]),
            row["parent_result"],
            str(row["assigned_gpu_slot"]),
        ]))
PY
)
  [ "${#wave_rows[@]}" -gt 0 ] || continue

  wave_pids=()
  for row in "${wave_rows[@]}"; do
    IFS='|' read -r combo_id dataset wave_num family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl_by_stage rule_bias_scale parent_result gpu_slot <<< "$row"
    gpu_id="${GPUS[$gpu_slot]}"
    order_num="${combo_id#?}"
    seed=$(( SEED_BASE + 10#${order_num} ))
    (
      phase="${PHASE_PREFIX}_${combo_id}"
      result_path_file="${STATUS_DIR}/${combo_id}.result.txt"
      log_path_file="${STATUS_DIR}/${combo_id}.log.txt"
      status_path="${STATUS_DIR}/${combo_id}.json"

      cmd=(
        bash "${SCRIPT_DIR}/tune_hparam.sh"
        --dataset "$dataset"
        --gpu "$gpu_id"
        --seed "$seed"
        --phase "$phase"
        --max-evals "$MAX_EVALS"
        --tune-epochs "$TUNE_EPOCHS"
        --tune-patience "$TUNE_PATIENCE"
        --layout-id "$layout_id"
        --execution "$execution"
        --router-family "$family"
        --router-impl-by-stage "$router_impl_by_stage"
        --rule-bias-scale "$rule_bias_scale"
        --feature-encoder-mode "$feature_encoder_mode"
        --train-batch-size "$train_bs"
        --eval-batch-size "$eval_bs"
        --embedding-size "128"
        --num-heads "8"
        --d-feat-emb "$d_feat_emb"
        --d-expert-hidden "$d_expert_hidden"
        --d-router-hidden "64"
        --expert-scale "$expert_scale"
        --weight-decay "5e-5"
        --hidden-dropout "0.10"
        --balance-loss-lambda "$balance_loss_lambda"
        --mid-router-temperature "1.2"
        --micro-router-temperature "1.2"
        --fmoe-schedule-enable "false"
        --moe-top-k "$moe_top_k"
        --moe-top-k-policy "$moe_top_k_policy"
        --lr-space "${lr_min},${lr_max}"
        --parent-result "$parent_result"
        --result-path-file "$result_path_file"
        --log-path-file "$log_path_file"
        --exp-name "fmoe_n_p05_narrow"
        --exp-desc "FeaturedMoE_N P0.5 narrow LR follow-up. Draft script; revise after first P0."
        --exp-focus "combo_id,parent_result,learning_rate,fmoe_v2_layout_id,feature_encoder_mode,expert_scale,moe_top_k"
      )
      if [ "$LOG_WANDB" = "true" ]; then
        cmd+=(--log-wandb)
      fi
      if [ "$DRY_RUN" = "true" ]; then
        cmd+=(--dry-run)
      fi

      printf '[P0.5][%s][GPU %s] ' "$combo_id" "$gpu_id"
      printf '%q ' "${cmd[@]}"
      printf '\n'

      set +e
      "${cmd[@]}"
      rc=$?
      set -e
      result_path=""
      log_path=""
      [ -f "$result_path_file" ] && read -r result_path <"$result_path_file" || true
      [ -f "$log_path_file" ] && read -r log_path <"$log_path_file" || true
      status="success"
      if [ "$DRY_RUN" = "true" ]; then
        status="dry_run"
        rc=0
      elif [ "$rc" -ne 0 ]; then
        status="fail"
      fi
      write_status_json "$status_path" "$combo_id" "$dataset" "$wave_num" "$gpu_slot" "$gpu_id" "$phase" "$status" "$rc" "$result_path" "$log_path"
      exit "$rc"
    ) &
    pid=$!
    wave_pids+=("$pid")
    dispatch_set_pid "$gpu_id" "$pid"
  done

  for pid in "${wave_pids[@]}"; do
    if ! wait "$pid"; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
  done
done

merge_manifest "$PLAN_PATH" "$STATUS_DIR" "$MANIFEST_OUT"
cp "$MANIFEST_OUT" "$LATEST_MANIFEST"

echo "[P0.5] manifest written: $MANIFEST_OUT"
echo "[P0.5] latest manifest: ${LATEST_MANIFEST}"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "[P0.5] completed with failures: $FAIL_COUNT" >&2
  exit 1
fi
exit 0
