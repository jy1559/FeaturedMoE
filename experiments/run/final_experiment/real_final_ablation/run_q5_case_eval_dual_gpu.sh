#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/workspace/FeaturedMoE"
PYTHON_BIN="${PYTHON_BIN:-/venv/FMoE/bin/python}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/experiments/run/artifacts/logs/real_final_ablation/q5/case_eval_rerun}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"

KUAIREC_RESULT="${ROOT_DIR}/experiments/run/artifacts/results/real_final_ablation/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_q5_kuaireclargestrictposv2_0_2_behavior_guided_r01_s1_20260420_075638_796778_pid1231413.json"
KUAIREC_CKPT="${ROOT_DIR}/experiments/run/artifacts/results/real_final_ablation/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_q5_kuaireclargestrictposv2_0_2_behavior_guided_r01_s1_20260420_075638_796778_pid1231413_best_model_state.pth"
FOURSQUARE_RESULT="${ROOT_DIR}/experiments/run/artifacts/results/real_final_ablation/foursquare_FeaturedMoE_N3_q5_foursquare_behavior_guided_r01_s1_20260420_081735_232866_pid1235892.json"
FOURSQUARE_CKPT="${ROOT_DIR}/experiments/run/artifacts/results/real_final_ablation/foursquare_FeaturedMoE_N3_q5_foursquare_behavior_guided_r01_s1_20260420_081735_232866_pid1235892_best_model_state.pth"

KUAIREC_BUNDLE="Q5_KUAIRECLARGESTRICTPOSV2_0_2_BEHAVIOR_GUIDED_R01_S1_${RUN_TAG}"
FOURSQUARE_BUNDLE="Q5_FOURSQUARE_BEHAVIOR_GUIDED_R01_S1_${RUN_TAG}"

PIDS=()

cleanup() {
  local exit_code=$?
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

trap cleanup INT TERM

mkdir -p "${OUTPUT_ROOT}"
cd "${ROOT_DIR}"

run_case_eval() {
  local dataset_label="$1"
  local result_json="$2"
  local checkpoint_file="$3"
  local bundle_name="$4"
  local gpu_id="$5"

  echo "[launch] dataset=${dataset_label} gpu=${gpu_id} bundle=${bundle_name}"
  "${PYTHON_BIN}" experiments/run/fmoe_n4/eval_checkpoint_case_subsets.py \
    --source-result-json "${result_json}" \
    --checkpoint-file "${checkpoint_file}" \
    --bundle-name "${bundle_name}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu-id "${gpu_id}" \
    "$@"
}

export_tables() {
  local bundle_name="$1"
  local manifest_path="${OUTPUT_ROOT}/${bundle_name}/case_eval_manifest.csv"
  local table_dir="${OUTPUT_ROOT}/${bundle_name}/tables"

  echo "[export] bundle=${bundle_name}"
  "${PYTHON_BIN}" experiments/run/fmoe_n4/export_case_eval_tables.py \
    --manifest "${manifest_path}" \
    --output-dir "${table_dir}"
}

EXTRA_ARGS=("$@")

(
  "${PYTHON_BIN}" experiments/run/fmoe_n4/eval_checkpoint_case_subsets.py \
    --source-result-json "${KUAIREC_RESULT}" \
    --checkpoint-file "${KUAIREC_CKPT}" \
    --bundle-name "${KUAIREC_BUNDLE}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu-id "${GPU_A}" \
    "${EXTRA_ARGS[@]}"
) &
PIDS+=("$!")

(
  "${PYTHON_BIN}" experiments/run/fmoe_n4/eval_checkpoint_case_subsets.py \
    --source-result-json "${FOURSQUARE_RESULT}" \
    --checkpoint-file "${FOURSQUARE_CKPT}" \
    --bundle-name "${FOURSQUARE_BUNDLE}" \
    --output-root "${OUTPUT_ROOT}" \
    --gpu-id "${GPU_B}" \
    "${EXTRA_ARGS[@]}"
) &
PIDS+=("$!")

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

PIDS=()

export_tables "${KUAIREC_BUNDLE}"
export_tables "${FOURSQUARE_BUNDLE}"

echo "[done] output_root=${OUTPUT_ROOT}"
echo "[done] kuairec_bundle=${OUTPUT_ROOT}/${KUAIREC_BUNDLE}"
echo "[done] foursquare_bundle=${OUTPUT_ROOT}/${FOURSQUARE_BUNDLE}"