#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_DIR="$(cd "${TRACK_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/run_metadata.sh"
# shellcheck source=/dev/null
source "${RUN_DIR}/common/dispatch_gpu_queue.sh"

DATASET="KuaiRecLargeStrictPosV2_0.2"
GPU_LIST="0,1,2,3"
SEED_BASE="7300"
STATE_TAG="S00_router_feature_heavy_v1"
PHASE_PREFIX="ARCH3"
MAX_EVALS="3"
TUNE_EPOCHS="100"
TUNE_PATIENCE="10"
LOG_WANDB="false"
DRY_RUN="${DRY_RUN:-false}"
WAVES_CSV=""
MANIFEST_OUT=""

usage() {
  cat <<USAGE
Usage: $0 [--dataset KuaiRecLargeStrictPosV2_0.2|lastfm0.03] [--gpus 0,1,2,3]
          [--state-tag S00_router_feature_heavy_v1] [--phase-prefix ARCH3]
          [--max-evals 3] [--tune-epochs 100] [--tune-patience 10]
          [--waves 1,2,3,4,5,6,7,8] [--manifest-out path] [--dry-run]
USAGE
}

emit_kul02_combo_table() {
  cat <<'EOF'
A01|KuaiRecLargeStrictPosV2_0.2|1|plain|8|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l8_plain_attn_es1
A02|KuaiRecLargeStrictPosV2_0.2|1|plain|30|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|anchor_l30_plain_ffn_es1
A03|KuaiRecLargeStrictPosV2_0.2|1|plain|16|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l16_plain_attn_es1
A04|KuaiRecLargeStrictPosV2_0.2|1|plain|19|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.0|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l19_plain_attn_es1_balance0
A05|KuaiRecLargeStrictPosV2_0.2|2|plain|34|serial|8192|16384|1.5e-4|3e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|dense_ffn|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l34_moe_off_dense_ffn
A06|KuaiRecLargeStrictPosV2_0.2|2|plain|35|serial|8192|16384|1.5e-4|3e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|dense_ffn|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l35_moe_off_dense_ffn
A07|KuaiRecLargeStrictPosV2_0.2|2|plain|34|serial|8192|16384|2.5e-4|6e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|identity|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l34_pure_attention
A08|KuaiRecLargeStrictPosV2_0.2|2|plain|35|serial|8192|16384|2.5e-4|6e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|identity|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l35_pure_attention
A09|KuaiRecLargeStrictPosV2_0.2|3|rule|8|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|rule_l8_full_rule_attn
A10|KuaiRecLargeStrictPosV2_0.2|3|rule|30|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|rule_l30_full_rule_ffn
A11|KuaiRecLargeStrictPosV2_0.2|3|hybrid|8|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|hybrid_l8_mid_micro_attn
A12|KuaiRecLargeStrictPosV2_0.2|3|bias|30|serial|8192|16384|5e-5|1.5e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.15|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|bias_l30_rule_bias_ffn
A13|KuaiRecLargeStrictPosV2_0.2|4|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|false|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_only_l8_proj128_attn
A14|KuaiRecLargeStrictPosV2_0.2|4|plain|30|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|false|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_only_l30_proj128_ffn
A15|KuaiRecLargeStrictPosV2_0.2|4|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|false|true|256|1|1.5|1.0|1.5|1.0e-4|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_only_l8_proj256_groupmeanstd_zloss
A16|KuaiRecLargeStrictPosV2_0.2|4|hybrid|30|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|mean_std|false|true|128|1|1.5|1.0|1.5|1.0e-4|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_only_hybrid_l30_groupmeanstd_zloss
A17|KuaiRecLargeStrictPosV2_0.2|5|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_l8_proj128_attn
A18|KuaiRecLargeStrictPosV2_0.2|5|plain|30|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_heavy_l30_proj128_ffn
A19|KuaiRecLargeStrictPosV2_0.2|5|bias|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.15|moe|mean_std|true|true|256|1|2.5|0.5|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_bias_l8_proj256_groupmeanstd
A20|KuaiRecLargeStrictPosV2_0.2|5|hybrid|16|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_hybrid_l16_attn
A21|KuaiRecLargeStrictPosV2_0.2|6|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|1.5|1.0|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|ablation_l8_es1_proj128_groupmeanstd
A22|KuaiRecLargeStrictPosV2_0.2|6|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|1.5|1.0|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|ablation_l8_es3_proj128_groupmeanstd
A23|KuaiRecLargeStrictPosV2_0.2|6|plain|30|serial|8192|16384|3e-5|8e-4|64|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|capacity_l30_dfeat64_proj128
A24|KuaiRecLargeStrictPosV2_0.2|6|plain|16|serial|8192|16384|3e-5|6e-4|16|128|128|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|capacity_l16_drouter128_proj128
A25|KuaiRecLargeStrictPosV2_0.2|7|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|1.0e-4|4.0e-4|0.25|0.0|0.0|attn|none|0.0|0.1|0.5|3|aux_l8_zloss_entropy
A26|KuaiRecLargeStrictPosV2_0.2|7|plain|30|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|1.0e-4|4.0e-4|0.25|0.0|0.0|ffn|none|0.0|0.1|0.5|3|aux_l30_zloss_entropy
A27|KuaiRecLargeStrictPosV2_0.2|7|plain|8|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.01|0.002|attn|none|0.0|0.1|0.5|3|aux_l8_rule_agreement_group_coverage
A28|KuaiRecLargeStrictPosV2_0.2|7|plain|30|serial|8192|16384|3e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.01|0.002|ffn|none|0.0|0.1|0.5|3|aux_l30_rule_agreement_group_coverage
A29|KuaiRecLargeStrictPosV2_0.2|8|plain|8|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|scheduler_none_base
A30|KuaiRecLargeStrictPosV2_0.2|8|plain|8|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|cosine|0.0|0.1|0.5|3|scheduler_cosine_base
A31|KuaiRecLargeStrictPosV2_0.2|8|plain|8|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|warmup_cosine|0.10|0.05|0.5|3|scheduler_warmup_cosine_base
A32|KuaiRecLargeStrictPosV2_0.2|8|plain|8|serial|8192|16384|3e-5|6e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|plateau|0.0|0.1|0.5|3|scheduler_plateau_base
EOF
}

emit_lastfm_combo_table() {
  cat <<'EOF'
A01|lastfm0.03|1|plain|8|serial|4096|4096|1e-4|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l8_plain_attn_es1
A02|lastfm0.03|1|plain|30|serial|4096|4096|1e-4|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|anchor_l30_plain_ffn_es1
A03|lastfm0.03|1|plain|16|serial|4096|4096|8e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l16_plain_attn_es1
A04|lastfm0.03|1|plain|19|serial|4096|4096|8e-5|5e-4|16|128|64|1|0|auto|linear|0.0|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|anchor_l19_plain_attn_es1_balance0
A05|lastfm0.03|2|plain|34|serial|4096|4096|2e-4|2e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|dense_ffn|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l34_moe_off_dense_ffn
A06|lastfm0.03|2|plain|35|serial|4096|4096|2e-4|2e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|dense_ffn|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l35_moe_off_dense_ffn
A07|lastfm0.03|2|plain|34|serial|4096|4096|2e-4|3e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|identity|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l34_pure_attention
A08|lastfm0.03|2|plain|35|serial|4096|4096|2e-4|3e-3|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|identity|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|control_l35_pure_attention
A09|lastfm0.03|3|rule|8|serial|4096|4096|8e-5|8e-4|16|128|64|1|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|rule_l8_full_rule_attn
A10|lastfm0.03|3|rule|30|serial|4096|4096|8e-5|8e-4|16|128|64|1|0|auto|linear|0.002|rule_soft|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|rule_l30_full_rule_ffn
A11|lastfm0.03|3|hybrid|8|serial|4096|4096|8e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|hybrid_l8_mid_micro_attn
A12|lastfm0.03|3|bias|30|serial|4096|4096|8e-5|8e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.15|moe|none|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|bias_l30_rule_bias_ffn
A13|lastfm0.03|4|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|false|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_only_l8_proj128_attn
A14|lastfm0.03|4|plain|30|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|false|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_only_l30_proj128_ffn
A15|lastfm0.03|4|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|false|true|256|1|1.5|1.0|1.5|1.0e-4|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_only_l8_proj256_groupmeanstd_zloss
A16|lastfm0.03|4|hybrid|30|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|mean_std|false|true|128|1|1.5|1.0|1.5|1.0e-4|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_only_hybrid_l30_groupmeanstd_zloss
A17|lastfm0.03|5|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_l8_proj128_attn
A18|lastfm0.03|5|plain|30|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|feature_heavy_l30_proj128_ffn
A19|lastfm0.03|5|bias|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.15|moe|mean_std|true|true|256|1|2.5|0.5|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_bias_l8_proj256_groupmeanstd
A20|lastfm0.03|5|hybrid|16|serial|4096|4096|5e-5|4e-4|16|128|64|1|0|auto|linear|0.002|learned|{mid:rule_soft,micro:rule_soft}|0.0|moe|none|true|true|128|1|2.0|0.75|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|feature_heavy_hybrid_l16_attn
A21|lastfm0.03|6|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|1.5|1.0|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|ablation_l8_es1_proj128_groupmeanstd
A22|lastfm0.03|6|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|3|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|1.5|1.0|1.5|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|ablation_l8_es3_proj128_groupmeanstd
A23|lastfm0.03|6|plain|30|serial|4096|4096|5e-5|5e-4|64|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|ffn|none|0.0|0.1|0.5|3|capacity_l30_dfeat64_proj128
A24|lastfm0.03|6|plain|16|serial|4096|4096|5e-5|4e-4|16|128|128|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|128|1|1.5|1.0|1.0|0.0|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|capacity_l16_drouter128_proj128
A25|lastfm0.03|7|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|1.0e-4|4.0e-4|0.25|0.0|0.0|attn|none|0.0|0.1|0.5|3|aux_l8_zloss_entropy
A26|lastfm0.03|7|plain|30|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|none|true|true|0|1|1.0|1.0|1.0|1.0e-4|4.0e-4|0.25|0.0|0.0|ffn|none|0.0|0.1|0.5|3|aux_l30_zloss_entropy
A27|lastfm0.03|7|plain|8|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.01|0.002|attn|none|0.0|0.1|0.5|3|aux_l8_rule_agreement_group_coverage
A28|lastfm0.03|7|plain|30|serial|4096|4096|5e-5|5e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|0|1|1.0|1.0|1.0|0.0|0.0|0.0|0.01|0.002|ffn|none|0.0|0.1|0.5|3|aux_l30_rule_agreement_group_coverage
A29|lastfm0.03|8|plain|8|serial|4096|4096|5e-5|4e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|none|0.0|0.1|0.5|3|scheduler_none_base
A30|lastfm0.03|8|plain|8|serial|4096|4096|5e-5|4e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|cosine|0.0|0.1|0.5|3|scheduler_cosine_base
A31|lastfm0.03|8|plain|8|serial|4096|4096|5e-5|4e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|warmup_cosine|0.10|0.05|0.5|3|scheduler_warmup_cosine_base
A32|lastfm0.03|8|plain|8|serial|4096|4096|5e-5|4e-4|16|128|64|1|0|auto|linear|0.002|learned|{}|0.0|moe|mean_std|true|true|128|1|2.0|0.75|1.0|1.0e-4|0.0|0.0|0.0|0.0|attn|plateau|0.0|0.1|0.5|3|scheduler_plateau_base
EOF
}

combo_table() {
  case "$DATASET" in
    KuaiRecLargeStrictPosV2_0.2|kuaireclargestrictposv2_0.2)
      emit_kul02_combo_table
      ;;
    lastfm0.03)
      emit_lastfm_combo_table
      ;;
    *)
      echo "Unsupported dataset for phase_arch3.sh: $DATASET" >&2
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
    if len(parts) != 42:
        raise SystemExit(f"invalid combo row ({len(parts)} cols): {raw}")
    (
        combo_id, dataset, wave, family, layout_id, execution, train_bs, eval_bs,
        lr_min, lr_max, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale,
        moe_top_k, moe_top_k_policy, feature_encoder_mode, balance_loss_lambda,
        router_impl, router_impl_by_stage, rule_bias_scale, moe_block_variant,
        router_group_feature_mode, router_use_hidden, router_use_feature,
        router_feature_proj_dim, router_feature_proj_layers, router_feature_scale,
        router_hidden_scale, router_group_feature_scale, z_loss_lambda,
        gate_entropy_lambda, gate_entropy_until, rule_agreement_lambda,
        group_coverage_lambda, inter_style, lr_scheduler_type,
        lr_scheduler_warmup_ratio, lr_scheduler_min_lr_ratio,
        lr_scheduler_plateau_factor, lr_scheduler_plateau_patience, combo_desc,
    ) = parts
    rows.append(
        {
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
            "d_router_hidden": int(d_router_hidden),
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
            "router_use_hidden": router_use_hidden,
            "router_use_feature": router_use_feature,
            "router_feature_proj_dim": int(router_feature_proj_dim),
            "router_feature_proj_layers": int(router_feature_proj_layers),
            "router_feature_scale": float(router_feature_scale),
            "router_hidden_scale": float(router_hidden_scale),
            "router_group_feature_scale": float(router_group_feature_scale),
            "z_loss_lambda": float(z_loss_lambda),
            "gate_entropy_lambda": float(gate_entropy_lambda),
            "gate_entropy_until": float(gate_entropy_until),
            "rule_agreement_lambda": float(rule_agreement_lambda),
            "group_coverage_lambda": float(group_coverage_lambda),
            "stage_inter_layer_style": inter_style,
            "lr_scheduler_type": lr_scheduler_type,
            "lr_scheduler_warmup_ratio": float(lr_scheduler_warmup_ratio),
            "lr_scheduler_min_lr_ratio": float(lr_scheduler_min_lr_ratio),
            "lr_scheduler_plateau_factor": float(lr_scheduler_plateau_factor),
            "lr_scheduler_plateau_patience": int(lr_scheduler_plateau_patience),
            "combo_desc": combo_desc,
            "assigned_gpu_slot": None,
            "assigned_gpu": None,
        }
    )

waves = sorted({r["wave"] for r in rows})
for wave_i in waves:
    wave_rows = [r for r in rows if r["wave"] == wave_i]
    for idx, row in enumerate(wave_rows):
        row["assigned_gpu_slot"] = idx
        row["assigned_gpu"] = gpus[idx]

payload = {
    "track": "fmoe_n2",
    "axis": state_tag.lower(),
    "phase": "ARCH3",
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
from pathlib import Path

status_path, combo_id, dataset, wave, gpu_slot, gpu_id, phase, status, return_code, result_path, log_path = sys.argv[1:]
payload = {
    "combo_id": combo_id,
    "dataset": dataset,
    "wave": int(wave),
    "assigned_gpu_slot": int(gpu_slot),
    "assigned_gpu": str(gpu_id),
    "phase": phase,
    "status": status,
    "return_code": int(return_code),
    "result_path": result_path or None,
    "log_path": log_path or None,
}
Path(status_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

merge_manifest() {
  local plan_path="$1"
  local status_dir="$2"
  local manifest_path="$3"
  python3 - <<'PY' "$plan_path" "$status_dir" "$manifest_path"
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
status_dir = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
status_map = {}
for status_file in status_dir.glob("*.json"):
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    status_map[payload["combo_id"]] = payload

enriched = []
for row in plan["combos"]:
    combo_id = row["combo_id"]
    merged = dict(row)
    merged.update(status_map.get(combo_id, {}))
    enriched.append(merged)

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

  IFS='|' read -r combo_id dataset wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden d_router_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl router_impl_by_stage rule_bias_scale moe_block_variant router_group_feature_mode router_use_hidden router_use_feature router_feature_proj_dim router_feature_proj_layers router_feature_scale router_hidden_scale router_group_feature_scale z_loss_lambda gate_entropy_lambda gate_entropy_until rule_agreement_lambda group_coverage_lambda inter_style lr_scheduler_type lr_scheduler_warmup_ratio lr_scheduler_min_lr_ratio lr_scheduler_plateau_factor lr_scheduler_plateau_patience combo_desc <<< "$row"

  local phase="${PHASE_PREFIX}_${combo_id}"
  local result_path_file="${status_dir}/${combo_id}.result.txt"
  local log_path_file="${status_dir}/${combo_id}.log.txt"
  local status_path="${status_dir}/${combo_id}.json"
  local rc=0
  local status="success"
  local result_path=""
  local log_path=""

  local cmd=(
    bash "${TRACK_DIR}/tune_hparam.sh"
    --dataset "$dataset"
    --gpu "$gpu_id"
    --seed "$seed"
    --phase "$phase"
    --run-axis "arch3"
    --state-tag "$STATE_TAG"
    --max-evals "$MAX_EVALS"
    --tune-epochs "$TUNE_EPOCHS"
    --tune-patience "$TUNE_PATIENCE"
    --layout-id "$layout_id"
    --execution "$execution"
    --router-family "$family"
    --router-impl "$router_impl"
    --router-impl-by-stage "$router_impl_by_stage"
    --router-use-hidden "$router_use_hidden"
    --router-use-feature "$router_use_feature"
    --rule-bias-scale "$rule_bias_scale"
    --feature-encoder-mode "$feature_encoder_mode"
    --stage-inter-layer-style "$inter_style"
    --moe-block-variant "$moe_block_variant"
    --router-group-feature-mode "$router_group_feature_mode"
    --router-feature-proj-dim "$router_feature_proj_dim"
    --router-feature-proj-layers "$router_feature_proj_layers"
    --router-feature-scale "$router_feature_scale"
    --router-hidden-scale "$router_hidden_scale"
    --router-group-feature-scale "$router_group_feature_scale"
    --train-batch-size "$train_bs"
    --eval-batch-size "$eval_bs"
    --embedding-size "128"
    --num-heads "8"
    --d-feat-emb "$d_feat_emb"
    --d-expert-hidden "$d_expert_hidden"
    --d-router-hidden "$d_router_hidden"
    --expert-scale "$expert_scale"
    --hidden-dropout "0.10"
    --weight-decay "5e-5"
    --balance-loss-lambda "$balance_loss_lambda"
    --z-loss-lambda "$z_loss_lambda"
    --gate-entropy-lambda "$gate_entropy_lambda"
    --gate-entropy-until "$gate_entropy_until"
    --rule-agreement-lambda "$rule_agreement_lambda"
    --group-coverage-lambda "$group_coverage_lambda"
    --mid-router-temperature "1.2"
    --micro-router-temperature "1.2"
    --fmoe-schedule-enable "false"
    --moe-top-k "$moe_top_k"
    --moe-top-k-policy "$moe_top_k_policy"
    --lr-space "${lr_min},${lr_max}"
    --lr-scheduler-type "$lr_scheduler_type"
    --lr-scheduler-warmup-ratio "$lr_scheduler_warmup_ratio"
    --lr-scheduler-min-lr-ratio "$lr_scheduler_min_lr_ratio"
    --lr-scheduler-plateau-factor "$lr_scheduler_plateau_factor"
    --lr-scheduler-plateau-patience "$lr_scheduler_plateau_patience"
    --combo-desc "$combo_desc"
    --result-path-file "$result_path_file"
    --log-path-file "$log_path_file"
    --exp-name "fmoe_n2_${STATE_TAG}_arch3"
    --exp-desc "FeaturedMoE_N2 ARCH3 feature-heavy probe."
    --exp-focus "combo_id,fmoe_v2_layout_id,router_family,router_impl,router_use_hidden,router_use_feature,router_feature_proj_dim,router_feature_scale,router_hidden_scale,router_group_feature_mode,stage_inter_layer_style,moe_block_variant,rule_agreement_lambda,group_coverage_lambda,lr_scheduler_type,learning_rate"
  )
  if [ "$LOG_WANDB" = "true" ]; then
    cmd+=(--log-wandb)
  fi
  if [ "$DRY_RUN" = "true" ]; then
    cmd+=(--dry-run)
  fi

  printf '[ARCH3][%s][GPU %s] ' "$combo_id" "$gpu_id"
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
INV_DIR="$(run_inventory_dir)/fmoe_n2"
run_ensure_dir "$INV_DIR"
WORK_DIR="${INV_DIR}/arch3_${STATE_TAG}_${RUN_TAG}"
STATUS_DIR="${WORK_DIR}/status"
run_ensure_dir "$WORK_DIR"
run_ensure_dir "$STATUS_DIR"

PLAN_PATH="${WORK_DIR}/combo_plan.json"
write_plan_json "$PLAN_PATH" "$GPU_LIST"

if [ -z "$MANIFEST_OUT" ]; then
  MANIFEST_OUT="${INV_DIR}/arch3_manifest_${STATE_TAG}_${RUN_TAG}.json"
fi
LATEST_MANIFEST="${INV_DIR}/arch3_manifest_${STATE_TAG}_latest.json"

echo "[ARCH3] dataset=${DATASET}"
echo "[ARCH3] state=${STATE_TAG}"
echo "[ARCH3] plan=${PLAN_PATH}"
echo "[ARCH3] manifest=${MANIFEST_OUT}"

FAIL_COUNT=0
mapfile -t ALL_WAVES < <(combo_table | awk -F'|' '{print $3}' | sort -n | uniq)

for wave in "${ALL_WAVES[@]}"; do
  selected_wave "$wave" || continue
  echo "=== [ARCH3] wave ${wave} ==="
  mapfile -t wave_rows < <(combo_table | awk -F'|' -v want="$wave" '$3 == want { print }')
  [ "${#wave_rows[@]}" -gt 0 ] || continue

  wave_pids=()
  for row in "${wave_rows[@]}"; do
    IFS='|' read -r combo_id dataset _wave family layout_id execution train_bs eval_bs lr_min lr_max d_feat_emb d_expert_hidden d_router_hidden expert_scale moe_top_k moe_top_k_policy feature_encoder_mode balance_loss_lambda router_impl router_impl_by_stage rule_bias_scale moe_block_variant router_group_feature_mode router_use_hidden router_use_feature router_feature_proj_dim router_feature_proj_layers router_feature_scale router_hidden_scale router_group_feature_scale z_loss_lambda gate_entropy_lambda gate_entropy_until rule_agreement_lambda group_coverage_lambda inter_style lr_scheduler_type lr_scheduler_warmup_ratio lr_scheduler_min_lr_ratio lr_scheduler_plateau_factor lr_scheduler_plateau_patience combo_desc <<< "$row"
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
    order_num="${combo_id#A}"
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

echo "[ARCH3] manifest written: $MANIFEST_OUT"
echo "[ARCH3] latest manifest: ${LATEST_MANIFEST}"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "[ARCH3] completed with failures: $FAIL_COUNT" >&2
  exit 1
fi
exit 0
