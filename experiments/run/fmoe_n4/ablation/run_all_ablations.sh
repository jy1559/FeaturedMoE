#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ABLATIONS="0,1,2,3,4,5"
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ablations)
            ABLATIONS="$2"
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

run_one() {
    local token="$1"
    case "$token" in
        1|routing)
            bash "${SCRIPT_DIR}/ablation_routing_control.sh" "${FORWARD_ARGS[@]}"
            ;;
        0|baseline)
            bash "${SCRIPT_DIR}/ablation_0_baseline_pool.sh" "${FORWARD_ARGS[@]}"
            ;;
        2|stage)
            bash "${SCRIPT_DIR}/ablation_stage_structure.sh" "${FORWARD_ARGS[@]}"
            ;;
        3|cue)
            bash "${SCRIPT_DIR}/ablation_cue_family.sh" "${FORWARD_ARGS[@]}"
            ;;
        4|objective)
            bash "${SCRIPT_DIR}/ablation_objective_variants.sh" "${FORWARD_ARGS[@]}"
            ;;
        5|portability|transfer)
            bash "${SCRIPT_DIR}/ablation_portability_followup.sh" "${FORWARD_ARGS[@]}"
            ;;
        *)
            echo "Unknown ablation token: ${token}" >&2
            return 1
            ;;
    esac
}

IFS=',' read -r -a TOKENS <<< "${ABLATIONS}"
for token in "${TOKENS[@]}"; do
    run_one "${token}"
done