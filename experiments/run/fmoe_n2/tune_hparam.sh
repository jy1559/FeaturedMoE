#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TRACK_GROUP="fmoe_n2"
export TRACK_MODEL_CONFIG="featured_moe_n2_tune"
export TRACK_MODEL_CLASS="FeaturedMoE_N2"
export TRACK_MODEL_PREFIX="FeaturedMoE_N2"
export TRACK_MODEL_DIR="${RUN_DIR}/../models/FeaturedMoE_N2"
export TRACK_SUMMARY_SCRIPT="${SCRIPT_DIR}/update_phase_summary.py"

exec "${RUN_DIR}/fmoe_n/tune_hparam.sh" "$@"
