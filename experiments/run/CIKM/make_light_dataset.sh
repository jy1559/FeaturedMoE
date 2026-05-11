#!/usr/bin/env bash
# Build lightweight non-RouteRec dataset views from final_dataset.
#
# Keeps:
#   *.inter: session_id, item_id, timestamp, user_id
#   *.item : item_id, category
#
# RouteRec should continue to use final_dataset because it needs macro/mid/micro features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SRC_ROOT="${SRC_ROOT:-$REPO_ROOT/Datasets/processed/final_dataset}"
DST_ROOT="${DST_ROOT:-$REPO_ROOT/Datasets/processed/final_dataset_light}"

mkdir -p "$DST_ROOT"
export LC_ALL=C

for src_dir in "$SRC_ROOT"/*; do
    [ -d "$src_dir" ] || continue
    dataset="$(basename "$src_dir")"
    dst_dir="$DST_ROOT/$dataset"
    mkdir -p "$dst_dir"
    echo "[LIGHT] $dataset -> $dst_dir"

    # Metadata/summaries are tiny and useful for provenance.
    find "$src_dir" -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' \) -exec cp -a {} "$dst_dir/" \;

    for split in "" ".train" ".valid" ".test"; do
        src_file="$src_dir/$dataset$split.inter"
        dst_file="$dst_dir/$dataset$split.inter"
        [ -f "$src_file" ] || continue
        tmp_file="$dst_file.tmp.$$"
        cut -f 1-4 "$src_file" > "$tmp_file"
        mv "$tmp_file" "$dst_file"
    done

    src_item="$src_dir/$dataset.item"
    if [ -f "$src_item" ]; then
        tmp_item="$dst_dir/$dataset.item.tmp.$$"
        cut -f 1-2 "$src_item" > "$tmp_item"
        mv "$tmp_item" "$dst_dir/$dataset.item"
    fi
done

echo "[LIGHT] done: $DST_ROOT"
