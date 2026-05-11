#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FMOE_N_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="$(cd "${FMOE_N_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

GPU_LIST="0,1,2,3"
SEED_BASE="5200"
STATE_TAG="S00_base_v1"
PHASE_PREFIX="P0"
MAX_EVALS="8"
TUNE_EPOCHS="18"
TUNE_PATIENCE="3"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
WAVES_CSV=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--gpus 0,1,2,3] [--seed-base 5200] [--waves 1,2,3]
          [--state-tag S00_base_v1]
          [--max-evals 8] [--tune-epochs 18] [--tune-patience 3]
          [--manifest-out path] [--dry-run]
USAGE
}

combo_table() {
  cat <<'EOF'
Q01|KuaiRecSmall0.1|1|plain|7|serial|6144|12288|3.2e-4|1.0e-3|16|128|3|0|auto|linear|0.002|{}|0.0|false|rerun_keep_l7_plain_lowlr
Q03|KuaiRecSmall0.1|1|bias|7|serial|6144|12288|3.8e-4|1.2e-3|16|128|3|0|auto|linear|0.002|{}|0.15|false|rerun_keep_l7_bias_lowlr
Q04|KuaiRecSmall0.1|1|plain|16|serial|6144|12288|4.0e-3|1.0e-2|16|128|3|0|auto|linear|0.002|{}|0.0|false|rerun_keep_l16_plain_highlr
Q06|KuaiRecSmall0.1|2|bias|16|serial|6144|12288|3.2e-4|1.0e-3|16|128|3|0|auto|linear|0.002|{}|0.15|false|rerun_keep_l16_bias_lowlr
Q07|KuaiRecSmall0.1|2|plain|19|serial|6144|12288|3.2e-4|1.0e-3|16|128|3|0|auto|linear|0.002|{}|0.0|false|rerun_keep_l19_plain_lowlr
Q09|KuaiRecSmall0.1|2|bias|19|serial|6144|12288|3.8e-4|1.2e-3|16|128|3|0|auto|linear|0.002|{}|0.15|false|rerun_keep_l19_bias_lowlr
Q10|KuaiRecSmall0.1|3|plain|15|serial|6144|12288|3.2e-4|1.0e-3|16|128|3|0|auto|linear|0.002|{}|0.0|false|rerun_new_l15_plain_lowlr
Q11|KuaiRecSmall0.1|3|hybrid|15|serial|6144|12288|4.0e-4|1.3e-3|16|128|3|0|auto|linear|0.002|{mid:rule_soft,micro:rule_soft}|0.0|false|rerun_single_hybrid_sentry
Q12|KuaiRecSmall0.1|3|bias|15|serial|6144|12288|3.8e-4|1.2e-3|16|128|3|0|auto|linear|0.002|{}|0.15|false|rerun_new_l15_bias_lowlr
Q13|KuaiRecSmall0.1|4|plain|7|serial|6144|12288|3.0e-4|9.0e-4|16|128|1|0|auto|linear|0.002|{}|0.0|false|expert_scale_1_lowlr
Q14|KuaiRecSmall0.1|4|plain|7|serial|6144|12288|3.0e-4|1.2e-3|16|128|5|0|auto|linear|0.002|{}|0.0|false|expert_scale_5_lowlr
Q15|KuaiRecSmall0.1|4|plain|7|serial|6144|12288|3.0e-4|1.0e-3|16|128|3|2|fixed|linear|0.002|{}|0.0|false|moe_topk_2_lowlr
Q16|KuaiRecSmall0.1|5|plain|7|serial|6144|12288|2.5e-4|1.2e-3|64|128|3|0|auto|linear|0.002|{}|0.0|true|d_feat_64_sentry_lowlr
Q17|KuaiRecSmall0.1|5|plain|7|serial|6144|12288|3.0e-4|1.0e-3|16|128|3|0|auto|sinusoidal_selected|0.002|{}|0.0|false|sinusoidal_selected_lowlr
Q18|KuaiRecSmall0.1|5|plain|7|serial|6144|12288|3.0e-4|1.0e-3|16|128|3|0|auto|linear|0.0|{}|0.0|false|balance_zero_lowlr
F01|lastfm0.03|1|plain|7|serial|4096|4096|1.8e-4|3.5e-4|16|128|3|0|auto|linear|0.002|{}|0.0|false|lfm_keep_l7_plain_tight
F03|lastfm0.03|2|bias|7|serial|4096|4096|1.8e-4|3.5e-4|16|128|3|0|auto|linear|0.002|{}|0.15|false|lfm_keep_l7_bias_tight
F02|lastfm0.03|3|hybrid|7|serial|4096|4096|2.0e-4|4.5e-4|16|128|3|0|auto|linear|0.002|{mid:rule_soft,micro:rule_soft}|0.0|false|lfm_single_hybrid_sentry
F04|lastfm0.03|4|plain|16|serial|4096|4096|1.8e-4|3.5e-4|16|128|3|0|auto|linear|0.002|{}|0.0|false|lfm_new_l16_plain_tight
F06|lastfm0.03|5|bias|16|serial|4096|4096|1.8e-4|3.5e-4|16|128|3|0|auto|linear|0.002|{}|0.15|false|lfm_new_l16_bias_tight
EOF
}

selected_wave() {
  local wave="$1"
  if [ -z "$WAVES_CSV" ]; then
    return 0
  fi
  local token
  IFS=',' read -r -a _waves <<< "$WAVES_CSV"
  for token in "${_waves[@]}"; do
    token="${token//[[:space:]]/}"
    [ "$token" = "$wave" ] && return 0
  done
  return 1
}

write_plan_json() {
  local out_path="$1"
  local gpu_csv="$2"
  local combo_text
  combo_text="$(combo_table)"
  COMBO_TABLE_TEXT="$combo_text" python3 - <<'PY' "$out_path" "$gpu_csv"
import json
import os
import sys

out_path = sys.argv[1]
gpu_csv = sys.argv[2]
gpus = [x.strip() for x in gpu_csv.split(",") if x.strip()]
lfm_slot_cycle = [0, 1, 2, 3, 0, 1]
state_tag = os.environ.get("STATE_TAG", "").strip().lower() or "hparam"
rows = []

for raw in os.environ.get("COMBO_TABLE_TEXT", "").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    parts = raw.split("|")
    if len(parts) != 21:
        raise SystemExit(f"invalid combo row: {raw}")
    (
        combo_id, dataset, wave, family, layout_id, execution,
        train_bs, eval_bs, lr_min, lr_max, d_feat_emb, d_expert_hidden,
        expert_scale, moe_top_k, moe_top_k_policy, feature_encoder_mode,
        balance_loss_lambda, router_impl_by_stage, rule_bias_scale,
        allow_fallback, notes,
    ) = parts
    wave_i = int(wave)
    row = {
        "combo_id": combo_id,
        "dataset": dataset,
        "wave": wave_i,
        "family": family,
        "layout_id": int(layout_id),
        "execution": execution,
        "train_batch_size": int(train_bs),
        "eval_batch_size": int(eval_bs),
        "lr_min": float(lr_min),
        "lr_max": float(lr_max),
        "d_feat_emb": int(d_feat_emb),
        "d_expert_hidden": int(d_expert_hidden),
        "expert_scale": int(expert_scale),
        "moe_top_k": int(moe_top_k),
        "moe_top_k_policy": moe_top_k_policy,
        "feature_encoder_mode": feature_encoder_mode,
        "balance_loss_lambda": float(balance_loss_lambda),
        "router_impl_by_stage": router_impl_by_stage,
        "rule_bias_scale": float(rule_bias_scale),
        "allow_fallback": allow_fallback.lower() == "true",
        "notes": notes,
        "assigned_gpu_slot": None,
        "assigned_gpu": None,
    }
    rows.append(row)

waves = sorted({r["wave"] for r in rows})
lfm_slots = lfm_slot_cycle[: len(waves)]

for wave_index, wave_i in enumerate(waves):
    wave_rows = [r for r in rows if r["wave"] == wave_i]
    lfm_slot = lfm_slots[wave_index]
    kuai_slots = [i for i in range(4) if i != lfm_slot]
    kuai_rows = [r for r in wave_rows if r["dataset"] != "lastfm0.03"]
    lfm_rows = [r for r in wave_rows if r["dataset"] == "lastfm0.03"]
    for idx, row in enumerate(kuai_rows):
        row["assigned_gpu_slot"] = kuai_slots[idx]
        row["assigned_gpu"] = gpus[kuai_slots[idx]]
    for row in lfm_rows:
        row["assigned_gpu_slot"] = lfm_slot
        row["assigned_gpu"] = gpus[lfm_slot]

payload = {
    "track": "fmoe_n",
    "axis": state_tag,
    "phase": "P0",
    "lfm_gpu_slot_rotation": lfm_slots,
    "waves": waves,
    "gpus": gpus,
    "combos": rows,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
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
  local fallback_used="${12}"
  local oom_detected="${13}"
  local effective_train_bs="${14}"
  local effective_eval_bs="${15}"
  python3 - <<'PY' "$status_path" "$combo_id" "$dataset" "$wave" "$gpu_slot" "$gpu_id" "$phase" "$status" "$return_code" "$result_path" "$log_path" "$fallback_used" "$oom_detected" "$effective_train_bs" "$effective_eval_bs"
import json
import sys

(
    status_path, combo_id, dataset, wave, gpu_slot, gpu_id, phase, status,
    return_code, result_path, log_path, fallback_used, oom_detected,
    effective_train_bs, effective_eval_bs,
) = sys.argv[1:]

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
    "fallback_used": fallback_used.lower() == "true",
    "oom_detected": oom_detected.lower() == "true",
    "effective_train_batch_size": int(effective_train_bs),
    "effective_eval_batch_size": int(effective_eval_bs),
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

enriched = []
for combo in plan["combos"]:
    row = dict(combo)
    status = status_map.get(combo["combo_id"])
    if status:
      row.update(status)
      result_path = Path(status.get("result_path") or "")
      if result_path.is_file():
          try:
              result = json.loads(result_path.read_text(encoding="utf-8"))
              row["best_mrr@20"] = result.get("best_mrr@20")
              best = result.get("best_params") or {}
              row["best_lr"] = best.get("learning_rate")
              row["best_weight_decay"] = best.get("weight_decay")
              row["best_hidden_dropout_prob"] = best.get("hidden_dropout_prob")
              row["best_balance_loss_lambda"] = best.get("balance_loss_lambda")
          except Exception as exc:
              row["result_read_error"] = str(exc)
    else:
      row["status"] = "not_run"
    enriched.append(row)

payload = dict(plan)
payload["combos"] = enriched
payload["n_success"] = sum(1 for c in enriched if c.get("status") == "success")
payload["n_fail"] = sum(1 for c in enriched if c.get("status") == "fail")
payload["n_dry_run"] = sum(1 for c in enriched if c.get("status") == "dry_run")
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_combo() {
  local row="$1"
  local gpu_id="$2"
  local seed="$3"
  local status_dir="$4"
  local manifest_plan="$5"

  IFS='|' read -r combo_id dataset wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl_by_stage rule_bias_scale allow_fallback notes <<< "$row"

  local phase="${PHASE_PREFIX}_${combo_id}"
  local result_path_file="${status_dir}/${combo_id}.result.txt"
  local log_path_file="${status_dir}/${combo_id}.log.txt"
  local status_path="${status_dir}/${combo_id}.json"
  local rc=0
  local status="success"
  local result_path=""
  local log_path=""
  local fallback_used="false"
  local oom_detected="false"
  local effective_train_bs="$train_bs"
  local effective_eval_bs="$eval_bs"

  local cmd=(
    bash "${FMOE_N_DIR}/tune_hparam.sh"
    --dataset "$dataset"
    --gpu "$gpu_id"
    --seed "$seed"
    --phase "$phase"
    --state-tag "$STATE_TAG"
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
    --hidden-dropout "0.10"
    --weight-decay "5e-5"
    --balance-loss-lambda "$balance_loss_lambda"
    --mid-router-temperature "1.2"
    --micro-router-temperature "1.2"
    --fmoe-schedule-enable "false"
    --moe-top-k "$moe_top_k"
    --moe-top-k-policy "$moe_top_k_policy"
    --lr-space "${lr_min},${lr_max}"
    --combo-desc "$notes"
    --result-path-file "$result_path_file"
    --log-path-file "$log_path_file"
    --exp-name "fmoe_n_p0_anchor"
    --exp-desc "FeaturedMoE_N P0 anchor wave sweep."
    --exp-focus "combo_id,dataset,family,fmoe_v2_layout_id,expert_scale,feature_encoder_mode,moe_top_k,learning_rate"
  )
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  printf '[P0][%s][GPU %s] ' "$combo_id" "$gpu_id"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  if [ -f "$result_path_file" ]; then
    read -r result_path <"$result_path_file" || true
  fi
  if [ -f "$log_path_file" ]; then
    read -r log_path <"$log_path_file" || true
  fi

  if [ "$DRY_RUN" = "true" ]; then
    status="dry_run"
    rc=0
  elif [ "$rc" -ne 0 ] && [ "$allow_fallback" = "true" ] && [ -n "$log_path" ] && grep -Eqi 'out of memory|cuda out of memory|oom' "$log_path"; then
    oom_detected="true"
    fallback_used="true"
    effective_train_bs="4096"
    effective_eval_bs="8192"
    phase="${PHASE_PREFIX}_${combo_id}_FB1"
    printf '[P0][%s] OOM detected, retry with fallback batch %s/%s\n' "$combo_id" "$effective_train_bs" "$effective_eval_bs"
    local fb_result_path_file="${status_dir}/${combo_id}.fb.result.txt"
    local fb_log_path_file="${status_dir}/${combo_id}.fb.log.txt"
    local fb_cmd=(
      bash "${FMOE_N_DIR}/tune_hparam.sh"
      --dataset "$dataset"
      --gpu "$gpu_id"
      --seed "$seed"
      --phase "$phase"
      --state-tag "$STATE_TAG"
      --max-evals "$MAX_EVALS"
      --tune-epochs "$TUNE_EPOCHS"
      --tune-patience "$TUNE_PATIENCE"
      --layout-id "$layout_id"
      --execution "$execution"
      --router-family "$family"
      --router-impl-by-stage "$router_impl_by_stage"
      --rule-bias-scale "$rule_bias_scale"
      --feature-encoder-mode "$feature_encoder_mode"
      --train-batch-size "$effective_train_bs"
      --eval-batch-size "$effective_eval_bs"
      --embedding-size "128"
      --num-heads "8"
      --d-feat-emb "$d_feat_emb"
      --d-expert-hidden "$d_expert_hidden"
      --d-router-hidden "64"
      --expert-scale "$expert_scale"
      --hidden-dropout "0.10"
      --weight-decay "5e-5"
      --balance-loss-lambda "$balance_loss_lambda"
      --mid-router-temperature "1.2"
      --micro-router-temperature "1.2"
      --fmoe-schedule-enable "false"
      --moe-top-k "$moe_top_k"
      --moe-top-k-policy "$moe_top_k_policy"
      --lr-space "${lr_min},${lr_max}"
      --combo-desc "$notes"
      --result-path-file "$fb_result_path_file"
      --log-path-file "$fb_log_path_file"
      --exp-name "fmoe_n_p0_anchor"
      --exp-desc "FeaturedMoE_N P0 anchor wave sweep fallback for Q16."
      --exp-focus "combo_id,dataset,family,fmoe_v2_layout_id,expert_scale,feature_encoder_mode,moe_top_k,learning_rate"
    )
    if [ "$LOG_WANDB" = "true" ]; then
      fb_cmd+=(--log-wandb)
    fi
    set +e
    "${fb_cmd[@]}"
    rc=$?
    set -e
    if [ -f "$fb_result_path_file" ]; then
      read -r result_path <"$fb_result_path_file" || true
    fi
    if [ -f "$fb_log_path_file" ]; then
      read -r log_path <"$fb_log_path_file" || true
    fi
  fi

  if [ "$rc" -ne 0 ] && [ "$status" != "dry_run" ]; then
    status="fail"
  fi

  local gpu_slot
  gpu_slot="$(python3 - <<'PY' "$manifest_plan" "$combo_id"
import json
import sys

plan = json.load(open(sys.argv[1], "r", encoding="utf-8"))
combo_id = sys.argv[2]
for row in plan["combos"]:
    if row["combo_id"] == combo_id:
        print(row["assigned_gpu_slot"])
        break
PY
)"

  write_status_json "$status_path" "$combo_id" "$dataset" "$wave" "$gpu_slot" "$gpu_id" "$phase" "$status" "$rc" "$result_path" "$log_path" "$fallback_used" "$oom_detected" "$effective_train_bs" "$effective_eval_bs"
  return "$rc"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --seed-base) SEED_BASE="$2"; shift 2 ;;
    --state-tag) STATE_TAG="$2"; shift 2 ;;
    --phase-prefix) PHASE_PREFIX="$2"; shift 2 ;;
    --max-evals) MAX_EVALS="$2"; shift 2 ;;
    --tune-epochs) TUNE_EPOCHS="$2"; shift 2 ;;
    --tune-patience) TUNE_PATIENCE="$2"; shift 2 ;;
    --waves) WAVES_CSV="$2"; shift 2 ;;
    --manifest-out) MANIFEST_OUT="$2"; shift 2 ;;
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

RUN_TAG="$(run_timestamp)"
INV_DIR="$(run_inventory_dir)/fmoe_n"
run_ensure_dir "$INV_DIR"
STATE_TAG="$(run_sanitize "$STATE_TAG")"
[ -n "$STATE_TAG" ] || { echo "--state-tag must not sanitize to empty" >&2; exit 1; }
WORK_DIR="${INV_DIR}/p0_${STATE_TAG}_${RUN_TAG}"
STATUS_DIR="${WORK_DIR}/status"
run_ensure_dir "$WORK_DIR"
run_ensure_dir "$STATUS_DIR"

PLAN_PATH="${WORK_DIR}/combo_plan.json"
STATE_TAG="$STATE_TAG" write_plan_json "$PLAN_PATH" "$GPU_LIST"

if [ -z "$MANIFEST_OUT" ]; then
  MANIFEST_OUT="${INV_DIR}/p0_manifest_${STATE_TAG}_${RUN_TAG}.json"
fi
LATEST_MANIFEST="${INV_DIR}/p0_manifest_${STATE_TAG}_latest.json"

echo "[P0] plan=${PLAN_PATH}"
echo "[P0] manifest=${MANIFEST_OUT}"
echo "[P0] state=${STATE_TAG}"
mapfile -t ALL_WAVES < <(combo_table | awk -F'|' '{print $3}' | sort -n | uniq)
LFM_GPU_SLOTS=(0 1 2 3 0 1)
ACTIVE_LFM_SLOTS=("${LFM_GPU_SLOTS[@]:0:${#ALL_WAVES[@]}}")
echo "[P0] lfm gpu rotation slots=$(IFS=,; echo "${ACTIVE_LFM_SLOTS[*]}")"

INTERRUPTED="false"
on_interrupt() {
  INTERRUPTED="true"
  echo "[INTERRUPT] stopping P0 workers..."
  dispatch_terminate_all GPUS
  exit 130
}
trap on_interrupt INT TERM

FAIL_COUNT=0

for wave in "${ALL_WAVES[@]}"; do
  selected_wave "$wave" || continue

  echo "=== [P0] wave ${wave} ==="
  mapfile -t wave_rows < <(combo_table | awk -F'|' -v want="$wave" '$3 == want { print }')
  [ "${#wave_rows[@]}" -gt 0 ] || continue

  wave_pids=()
  wave_rcs=()
  for row in "${wave_rows[@]}"; do
    IFS='|' read -r combo_id dataset _wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl_by_stage rule_bias_scale allow_fallback notes <<< "$row"
    gpu_slot="$(python3 - <<'PY' "$PLAN_PATH" "$combo_id"
import json
import sys

plan = json.load(open(sys.argv[1], "r", encoding="utf-8"))
combo_id = sys.argv[2]
for row in plan["combos"]:
    if row["combo_id"] == combo_id:
        print(row["assigned_gpu_slot"])
        break
PY
)"
    gpu_id="${GPUS[$gpu_slot]}"
    order_num="${combo_id#?}"
    seed=$(( SEED_BASE + 10#${order_num} ))
    (
      set +e
      run_combo "$row" "$gpu_id" "$seed" "$STATUS_DIR" "$PLAN_PATH"
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

echo "[P0] manifest written: $MANIFEST_OUT"
echo "[P0] latest manifest: ${LATEST_MANIFEST}"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "[P0] completed with failures: $FAIL_COUNT" >&2
  exit 1
fi
exit 0
