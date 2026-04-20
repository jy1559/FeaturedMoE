#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FeaturedMoE
/venv/FMoE/bin/python experiments/run/final_experiment/real_final_ablation/q4_efficiency.py "$@"
