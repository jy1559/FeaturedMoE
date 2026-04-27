#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FIRST_BASE_SETTING_JSONS="/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/beauty_FeaturedMoE_N3_p4xd_xd_beauty_b25_lr_h8_seen_anchor_s1_20260416_030201_046233_pid244350.json,/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_p4s3_s3_kuaireclargestrictposv2_0_2_s02_h14_seen_hi_s1_20260415_081315_824627_pid124072.json"
SECOND_BASE_SETTING_JSONS="/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/beauty_FeaturedMoE_N3_p4xd_xd_beauty_b17_hyp_midaux_lowaux_h3_s1_20260416_022118_916556_pid232570.json,/workspace/FeaturedMoE/experiments/run/artifacts/results/fmoe_n4/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N3_p4s3_s3_kuaireclargestrictposv2_0_2_s07_h10_len25_f24_s1_20260415_081315_832081_pid124077.json"

SETTING_SLOTS="first,second"
WITH_SECONDARY_FOLLOWUP="true"
RUN_TRANSFER="true"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setting-slots)
            SETTING_SLOTS="$2"
            shift 2
            ;;
        --base-slot)
            if [[ "$2" == "primary" ]]; then
                SETTING_SLOTS="first"
            elif [[ "$2" == "secondary" ]]; then
                SETTING_SLOTS="second"
            else
                SETTING_SLOTS="$2"
            fi
            shift 2
            ;;
        --with-secondary-followup)
            WITH_SECONDARY_FOLLOWUP="true"
            shift
            ;;
        --no-secondary-followup)
            WITH_SECONDARY_FOLLOWUP="false"
            shift
            ;;
        --skip-transfer)
            RUN_TRANSFER="false"
            shift
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

select_base_jsons() {
    case "$1" in
        first)
            printf '%s' "${FIRST_BASE_SETTING_JSONS}"
            ;;
        second)
            printf '%s' "${SECOND_BASE_SETTING_JSONS}"
            ;;
        *)
            echo "Unknown setting slot: $1" >&2
            return 1
            ;;
    esac
}

select_transfer_preset() {
    case "$1" in
        first)
            printf '%s' "shared_a"
            ;;
        second)
            printf '%s' "shared_b"
            ;;
        *)
            echo "Unknown setting slot for transfer preset: $1" >&2
            return 1
            ;;
    esac
}

run_core_suite() {
    local base_jsons="$1"
    local setting_tier="$2"
    local axis_suffix="$3"
    local include_baseline_always="$4"
    local -a cmd=(
        bash "${SCRIPT_DIR}/ablation_core_global_queue.sh"
        --base-result-jsons "${base_jsons}"
        --setting-tier "${setting_tier}"
        --setting-scope all
        --axis-suffix "${axis_suffix}"
        --seeds 1,2,3,4
        --max-evals 10
        --tune-epochs 100
        --tune-patience 10
        --lr-mode band9
    )
    if [[ "${include_baseline_always}" == "true" ]]; then
        cmd+=(--include-baseline-always)
    fi
    cmd+=("${FORWARD_ARGS[@]}")
    "${cmd[@]}"
}

run_setting_bundle() {
    local setting_slot="$1"
    local base_jsons
    base_jsons="$(select_base_jsons "${setting_slot}")"

    run_core_suite "${base_jsons}" essential "${setting_slot}_primary" false

    if [[ "${WITH_SECONDARY_FOLLOWUP}" == "true" ]]; then
        run_core_suite "${base_jsons}" extended "${setting_slot}_secondary" true
    fi
}

run_transfer_suite() {
    local setting_slot="$1"
    local transfer_preset
    transfer_preset="$(select_transfer_preset "${setting_slot}")"
    bash "${SCRIPT_DIR}/ablation_portability_followup.sh" \
        --pairs beauty_to_kuairec \
        --hparam-presets "${transfer_preset}" \
        --setting-tier essential \
        --setting-scope all \
        --seeds 1,2,3,4 \
        --max-evals 10 \
        --tune-epochs 100 \
        --tune-patience 10 \
        --lr-mode band9 \
        "${FORWARD_ARGS[@]}"
}

IFS=',' read -r -a RAW_SLOTS <<< "${SETTING_SLOTS}"
SLOTS=()
for token in "${RAW_SLOTS[@]}"; do
    token="${token// /}"
    [[ -z "${token}" ]] && continue
    case "${token}" in
        first|1)
            SLOTS+=("first")
            ;;
        second|2)
            SLOTS+=("second")
            ;;
        *)
            echo "Unknown setting slot: ${token}" >&2
            exit 1
            ;;
    esac
done

if [[ "${#SLOTS[@]}" -eq 0 ]]; then
    echo "No setting slots selected" >&2
    exit 1
fi

for idx in "${!SLOTS[@]}"; do
    slot="${SLOTS[$idx]}"
    run_setting_bundle "${slot}"
    if [[ "${RUN_TRANSFER}" == "true" && "${idx}" -eq 0 ]]; then
        run_transfer_suite "${slot}"
    fi
done