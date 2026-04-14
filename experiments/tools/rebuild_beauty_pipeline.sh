#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET="beauty"

python experiments/tools/build_beauty_basic_from_raw.py \
  --dataset "$DATASET" \
  --session-gap-days 14 \
  --min-session-len 5 \
  --min-item-freq 3 \
  --overwrite

python experiments/tools/build_beauty_feature_v3.py \
  --dataset "$DATASET" \
  --fit-session-ratio 0.7 \
  --micro-window 5 \
  --split-ratios 0.7,0.15,0.15 \
  --split-strategy tail_stratified \
  --overwrite

python experiments/tools/build_beauty_feature_v4.py \
  --dataset "$DATASET" \
  --overwrite

echo "beauty pipeline complete"
