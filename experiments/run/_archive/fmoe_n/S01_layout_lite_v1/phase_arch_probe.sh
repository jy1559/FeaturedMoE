#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FMOE_N_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="$(cd "${FMOE_N_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="KuaiRecSmall0.1"
GPU_LIST="0,1,2,3"
SEED_BASE="6100"
STATE_TAG="S01_layout_lite_v1"
PHASE_PREFIX="ARCH3"
MAX_EVALS="4"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
WAVES_CSV=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--dataset KuaiRecSmall0.1|KuaiRecLargeStrictPosV2_0.2|lastfm0.03] [--gpus 0,1,2,3] [--state-tag S01_layout_lite_v1]
          [--phase-prefix ARCH3] [--max-evals 4] [--tune-epochs 100] [--tune-patience 10]
          [--waves 1,2,3,4,5,6,7] [--manifest-out path] [--dry-run]
USAGE
}

emit_kuai_combo_table() {
  local dataset="$1"
  local train_bs="$2"
  local eval_bs="$3"
  cat <<'EOF' | sed -e "s#__DATASET__#${dataset}#g" -e "s#__TRAIN_BS__#${train_bs}#g" -e "s#__EVAL_BS__#${eval_bs}#g"
A01|__DATASET__|1|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l8_attn_moe
A02|__DATASET__|1|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l30_attn_moe
A03|__DATASET__|1|plain|7|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|5.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l7_attn_moe
A04|__DATASET__|1|plain|16|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|5.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l16_attn_moe
A05|__DATASET__|2|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|4.5e-4|1.2e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l8_attn_dense_ffn
A06|__DATASET__|2|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|4.5e-4|1.0e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l30_attn_dense_ffn
A07|__DATASET__|2|plain|7|serial|__TRAIN_BS__|__EVAL_BS__|4.0e-4|1.0e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l7_attn_dense_ffn
A08|__DATASET__|2|plain|16|serial|__TRAIN_BS__|__EVAL_BS__|4.0e-4|9.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l16_attn_dense_ffn
A09|__DATASET__|3|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|6.0e-4|1.5e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l8
A10|__DATASET__|3|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|6.0e-4|1.3e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l30
A11|__DATASET__|3|plain|7|serial|__TRAIN_BS__|__EVAL_BS__|5.0e-4|1.2e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l7
A12|__DATASET__|3|plain|16|serial|__TRAIN_BS__|__EVAL_BS__|5.0e-4|1.1e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l16
A13|__DATASET__|4|rule|8|serial|__TRAIN_BS__|__EVAL_BS__|3.0e-4|7.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l8
A14|__DATASET__|4|rule|30|serial|__TRAIN_BS__|__EVAL_BS__|3.0e-4|7.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l30
A15|__DATASET__|4|rule|7|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l7
A16|__DATASET__|4|rule|16|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l16
A17|__DATASET__|5|hybrid|8|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l8_attn_moe
A18|__DATASET__|5|hybrid|30|serial|__TRAIN_BS__|__EVAL_BS__|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l30_attn_moe
A19|__DATASET__|5|plain|19|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|5.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_plain_l19_balance0
A20|__DATASET__|5|hybrid|19|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|5.0e-3|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_hybrid_l19_balance4e3
A21|__DATASET__|6|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l8
A22|__DATASET__|6|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l30
A23|__DATASET__|6|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l8
A24|__DATASET__|6|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l30
A25|__DATASET__|7|plain|8|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l8
A26|__DATASET__|7|plain|30|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l30
A27|__DATASET__|7|bias|8|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.15|moe|mean_std|0.0|4.0e-4|0.25|attn|groupmeanstd_entropy_bias_l8
A28|__DATASET__|7|hybrid|16|serial|__TRAIN_BS__|__EVAL_BS__|2.0e-4|3.5e-3|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|mean_std|1.0e-4|4.0e-4|0.25|attn|groupmeanstd_hybrid_l16_fullreg
EOF
}

combo_table() {
  case "$DATASET" in
    KuaiRecSmall0.1)
      cat <<'EOF'
A01|KuaiRecSmall0.1|1|plain|8|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l8_attn_moe
A02|KuaiRecSmall0.1|1|plain|30|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l30_attn_moe
A03|KuaiRecSmall0.1|1|plain|7|serial|6144|12288|2.0e-4|5.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l7_attn_moe
A04|KuaiRecSmall0.1|1|plain|16|serial|6144|12288|2.0e-4|5.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l16_attn_moe
A05|KuaiRecSmall0.1|2|plain|8|serial|6144|12288|4.5e-4|1.2e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l8_attn_dense_ffn
A06|KuaiRecSmall0.1|2|plain|30|serial|6144|12288|4.5e-4|1.0e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l30_attn_dense_ffn
A07|KuaiRecSmall0.1|2|plain|7|serial|6144|12288|4.0e-4|1.0e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l7_attn_dense_ffn
A08|KuaiRecSmall0.1|2|plain|16|serial|6144|12288|4.0e-4|9.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l16_attn_dense_ffn
A09|KuaiRecSmall0.1|3|plain|8|serial|6144|12288|6.0e-4|1.5e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l8
A10|KuaiRecSmall0.1|3|plain|30|serial|6144|12288|6.0e-4|1.3e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l30
A11|KuaiRecSmall0.1|3|plain|7|serial|6144|12288|5.0e-4|1.2e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l7
A12|KuaiRecSmall0.1|3|plain|16|serial|6144|12288|5.0e-4|1.1e-2|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l16
A13|KuaiRecSmall0.1|4|rule|8|serial|6144|12288|3.0e-4|7.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l8
A14|KuaiRecSmall0.1|4|rule|30|serial|6144|12288|3.0e-4|7.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l30
A15|KuaiRecSmall0.1|4|rule|7|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l7
A16|KuaiRecSmall0.1|4|rule|16|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l16
A17|KuaiRecSmall0.1|5|hybrid|8|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l8_attn_moe
A18|KuaiRecSmall0.1|5|hybrid|30|serial|6144|12288|2.5e-4|6.0e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l30_attn_moe
A19|KuaiRecSmall0.1|5|plain|19|serial|6144|12288|2.0e-4|5.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_plain_l19_balance0
A20|KuaiRecSmall0.1|5|hybrid|19|serial|6144|12288|2.0e-4|5.0e-3|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_hybrid_l19_balance4e3
A21|KuaiRecSmall0.1|6|plain|8|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l8
A22|KuaiRecSmall0.1|6|plain|30|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l30
A23|KuaiRecSmall0.1|6|plain|8|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l8
A24|KuaiRecSmall0.1|6|plain|30|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l30
A25|KuaiRecSmall0.1|7|plain|8|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l8
A26|KuaiRecSmall0.1|7|plain|30|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l30
A27|KuaiRecSmall0.1|7|bias|8|serial|6144|12288|2.0e-4|4.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.15|moe|mean_std|0.0|4.0e-4|0.25|attn|groupmeanstd_entropy_bias_l8
A28|KuaiRecSmall0.1|7|hybrid|16|serial|6144|12288|2.0e-4|3.5e-3|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|mean_std|1.0e-4|4.0e-4|0.25|attn|groupmeanstd_hybrid_l16_fullreg
EOF
      ;;
    lastfm0.03)
      cat <<'EOF'
A01|lastfm0.03|1|plain|8|serial|4096|4096|1.2e-4|1.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l8_attn_moe
A02|lastfm0.03|1|plain|30|serial|4096|4096|1.2e-4|1.5e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l30_attn_moe
A03|lastfm0.03|1|plain|7|serial|4096|4096|1.0e-4|1.4e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l7_attn_moe
A04|lastfm0.03|1|plain|16|serial|4096|4096|1.0e-4|1.4e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|anchor_plain_l16_attn_moe
A05|lastfm0.03|2|plain|8|serial|4096|4096|2.5e-4|2.2e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l8_attn_dense_ffn
A06|lastfm0.03|2|plain|30|serial|4096|4096|2.5e-4|2.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l30_attn_dense_ffn
A07|lastfm0.03|2|plain|7|serial|4096|4096|2.2e-4|2.0e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l7_attn_dense_ffn
A08|lastfm0.03|2|plain|16|serial|4096|4096|2.2e-4|1.8e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|dense_ffn|none|0.0|0.0|0.0|attn|moe_off_l16_attn_dense_ffn
A09|lastfm0.03|3|plain|8|serial|4096|4096|3.0e-4|2.8e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l8
A10|lastfm0.03|3|plain|30|serial|4096|4096|3.0e-4|2.4e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l30
A11|lastfm0.03|3|plain|7|serial|4096|4096|2.8e-4|2.3e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l7
A12|lastfm0.03|3|plain|16|serial|4096|4096|2.8e-4|2.1e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|identity|none|0.0|0.0|0.0|attn|pure_attention_l16
A13|lastfm0.03|4|rule|8|serial|4096|4096|1.4e-4|1.8e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l8
A14|lastfm0.03|4|rule|30|serial|4096|4096|1.4e-4|1.8e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l30
A15|lastfm0.03|4|rule|7|serial|4096|4096|1.2e-4|1.6e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l7
A16|lastfm0.03|4|rule|16|serial|4096|4096|1.2e-4|1.6e-3|16|128|3|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|0.0|0.0|0.0|attn|full_rule_l16
A17|lastfm0.03|5|hybrid|8|serial|4096|4096|1.2e-4|1.6e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l8_attn_moe
A18|lastfm0.03|5|hybrid|30|serial|4096|4096|1.2e-4|1.6e-3|16|128|3|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|hybrid_l30_attn_moe
A19|lastfm0.03|5|plain|19|serial|4096|4096|1.0e-4|1.3e-3|16|128|3|0|auto|linear|0.0|learned|{}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_plain_l19_balance0
A20|lastfm0.03|5|hybrid|19|serial|4096|4096|1.0e-4|1.3e-3|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|0.0|0.0|0.0|attn|heavy_hybrid_l19_balance4e3
A21|lastfm0.03|6|plain|8|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l8
A22|lastfm0.03|6|plain|30|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|1.0e-4|0.0|0.0|attn|zloss_plain_l30
A23|lastfm0.03|6|plain|8|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l8
A24|lastfm0.03|6|plain|30|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|none|0.0|4.0e-4|0.25|attn|entropy_plain_l30
A25|lastfm0.03|7|plain|8|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l8
A26|lastfm0.03|7|plain|30|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean|1.0e-4|0.0|0.0|attn|groupmean_zloss_plain_l30
A27|lastfm0.03|7|bias|8|serial|4096|4096|1.0e-4|1.0e-3|16|128|3|0|auto|linear|0.002|learned|{}|0.15|moe|mean_std|0.0|4.0e-4|0.25|attn|groupmeanstd_entropy_bias_l8
A28|lastfm0.03|7|hybrid|16|serial|4096|4096|1.0e-4|9.0e-4|16|128|3|0|auto|linear|0.004|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|mean_std|1.0e-4|4.0e-4|0.25|attn|groupmeanstd_hybrid_l16_fullreg
EOF
      ;;
    KuaiRecLargeStrictPosV2_0.2|kuaireclargestrictposv2_0.2)
      emit_kuai_combo_table "KuaiRecLargeStrictPosV2_0.2" "4096" "8192"
      ;;
    *)
      echo "Unsupported dataset for phase_arch_probe.sh: $DATASET" >&2
      return 1
      ;;
  esac
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
  COMBO_TABLE_TEXT="$combo_text" python3 - <<'PY' "$out_path" "$gpu_csv" "$DATASET" "$STATE_TAG"
import json
import os
import sys

out_path = sys.argv[1]
gpu_csv = sys.argv[2]
dataset_scope = sys.argv[3]
state_tag = sys.argv[4]
gpus = [x.strip() for x in gpu_csv.split(",") if x.strip()]
rows = []

for raw in os.environ.get("COMBO_TABLE_TEXT", "").splitlines():
    raw = raw.strip()
    if not raw:
        continue
    parts = raw.split("|")
    if len(parts) != 27:
        raise SystemExit(f"invalid combo row: {raw}")
    (
        combo_id, dataset, wave, family, layout_id, execution, train_bs, eval_bs,
        lr_min, lr_max, d_feat_emb, d_expert_hidden, expert_scale, moe_top_k,
        moe_top_k_policy, feature_encoder_mode, balance_loss_lambda, router_impl,
        router_impl_by_stage, rule_bias_scale, moe_block_variant, router_group_feature_mode,
        z_loss_lambda, gate_entropy_lambda, gate_entropy_until, inter_style, combo_desc,
    ) = parts
    row = {
        "combo_id": combo_id,
        "dataset": dataset,
        "state_tag": state_tag,
        "wave": int(wave),
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
        "router_impl": router_impl,
        "router_impl_by_stage": router_impl_by_stage,
        "rule_bias_scale": float(rule_bias_scale),
        "moe_block_variant": moe_block_variant,
        "router_group_feature_mode": router_group_feature_mode,
        "z_loss_lambda": float(z_loss_lambda),
        "gate_entropy_lambda": float(gate_entropy_lambda),
        "gate_entropy_until": float(gate_entropy_until),
        "stage_inter_layer_style": inter_style,
        "combo_desc": combo_desc,
        "assigned_gpu_slot": None,
        "assigned_gpu": None,
    }
    rows.append(row)

waves = sorted({r["wave"] for r in rows})
for wave_i in waves:
    wave_rows = [r for r in rows if r["wave"] == wave_i]
    for idx, row in enumerate(wave_rows):
        row["assigned_gpu_slot"] = idx
        row["assigned_gpu"] = gpus[idx]

payload = {
    "track": "fmoe_n",
    "axis": state_tag.lower(),
    "phase": "ARCH",
    "dataset_scope": dataset_scope,
    "datasets": sorted({r["dataset"] for r in rows}),
    "gpus": gpus,
    "waves": waves,
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
  python3 - <<'PY' "$status_path" "$combo_id" "$dataset" "$wave" "$gpu_slot" "$gpu_id" "$phase" "$status" "$return_code" "$result_path" "$log_path"
import json
import sys

(
    status_path, combo_id, dataset, wave, gpu_slot, gpu_id, phase, status,
    return_code, result_path, log_path,
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

  IFS='|' read -r combo_id dataset wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl router_impl_by_stage rule_bias_scale moe_block_variant router_group_feature_mode z_loss_lambda gate_entropy_lambda gate_entropy_until inter_style combo_desc <<< "$row"

  local phase="${PHASE_PREFIX}_${combo_id}"
  local result_path_file="${status_dir}/${combo_id}.result.txt"
  local log_path_file="${status_dir}/${combo_id}.log.txt"
  local status_path="${status_dir}/${combo_id}.json"
  local rc=0
  local status="success"
  local result_path=""
  local log_path=""

  local cmd=(
    bash "${FMOE_N_DIR}/tune_hparam.sh"
    --dataset "$dataset"
    --gpu "$gpu_id"
    --seed "$seed"
    --phase "$phase"
    --run-axis "arch_probe"
    --state-tag "$STATE_TAG"
    --max-evals "$MAX_EVALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --layout-id "$layout_id"
    --execution "$execution"
    --router-family "$family"
    --router-impl "$router_impl"
    --router-impl-by-stage "$router_impl_by_stage"
    --rule-bias-scale "$rule_bias_scale"
    --feature-encoder-mode "$feature_encoder_mode"
    --stage-inter-layer-style "$inter_style"
    --moe-block-variant "$moe_block_variant"
    --router-group-feature-mode "$router_group_feature_mode"
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
    --z-loss-lambda "$z_loss_lambda"
    --gate-entropy-lambda "$gate_entropy_lambda"
    --gate-entropy-until "$gate_entropy_until"
    --mid-router-temperature "1.2"
    --micro-router-temperature "1.2"
    --fmoe-schedule-enable "false"
    --moe-top-k "$moe_top_k"
    --moe-top-k-policy "$moe_top_k_policy"
    --lr-space "${lr_min},${lr_max}"
    --combo-desc "$combo_desc"
    --result-path-file "$result_path_file"
    --log-path-file "$log_path_file"
    --exp-name "fmoe_n_${STATE_TAG}_arch_probe"
    --exp-desc "FeaturedMoE_N architecture probe for ${STATE_TAG}."
    --exp-focus "combo_id,fmoe_v2_layout_id,router_family,router_impl,stage_inter_layer_style,moe_block_variant,router_group_feature_mode,feature_encoder_mode,learning_rate"
  )
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  printf '[ARCH][%s][GPU %s] ' "$combo_id" "$gpu_id"
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
  elif [ "$rc" -ne 0 ]; then
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

  write_status_json "$status_path" "$combo_id" "$dataset" "$wave" "$gpu_slot" "$gpu_id" "$phase" "$status" "$rc" "$result_path" "$log_path"
  return "$rc"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
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

STATE_TAG="$(run_sanitize "$STATE_TAG")"
[ -n "$STATE_TAG" ] || { echo "--state-tag must not sanitize to empty" >&2; exit 1; }

dispatch_parse_csv "$GPU_LIST" GPUS
[ "${#GPUS[@]}" -eq 4 ] || { echo "--gpus must contain exactly 4 gpu ids" >&2; exit 1; }

RUN_TAG="$(run_timestamp)"
INV_DIR="$(run_inventory_dir)/fmoe_n"
run_ensure_dir "$INV_DIR"
WORK_DIR="${INV_DIR}/arch_${STATE_TAG}_${RUN_TAG}"
STATUS_DIR="${WORK_DIR}/status"
run_ensure_dir "$WORK_DIR"
run_ensure_dir "$STATUS_DIR"

PLAN_PATH="${WORK_DIR}/combo_plan.json"
write_plan_json "$PLAN_PATH" "$GPU_LIST"

if [ -z "$MANIFEST_OUT" ]; then
  MANIFEST_OUT="${INV_DIR}/arch_manifest_${STATE_TAG}_${RUN_TAG}.json"
fi
LATEST_MANIFEST="${INV_DIR}/arch_manifest_${STATE_TAG}_latest.json"

echo "[ARCH] dataset=${DATASET}"
echo "[ARCH] state=${STATE_TAG}"
echo "[ARCH] plan=${PLAN_PATH}"
echo "[ARCH] manifest=${MANIFEST_OUT}"

FAIL_COUNT=0
mapfile -t ALL_WAVES < <(combo_table | awk -F'|' '{print $3}' | sort -n | uniq)

for wave in "${ALL_WAVES[@]}"; do
  selected_wave "$wave" || continue

  echo "=== [ARCH] wave ${wave} ==="
  mapfile -t wave_rows < <(combo_table | awk -F'|' -v want="$wave" '$3 == want { print }')
  [ "${#wave_rows[@]}" -gt 0 ] || continue

  wave_pids=()
  for row in "${wave_rows[@]}"; do
    IFS='|' read -r combo_id dataset _wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl router_impl_by_stage rule_bias_scale moe_block_variant router_group_feature_mode z_loss_lambda gate_entropy_lambda gate_entropy_until inter_style combo_desc <<< "$row"
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

echo "[ARCH] manifest written: $MANIFEST_OUT"
echo "[ARCH] latest manifest: ${LATEST_MANIFEST}"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "[ARCH] completed with failures: $FAIL_COUNT" >&2
  exit 1
fi
exit 0
