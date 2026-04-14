# FMoE A12 Ablation Experiment Guide

## Scope
This document explains what each launcher script runs and what each setting means.

- Dataset focus: KuaiRecLargeStrictPosV2_0.2
- Model family: FeaturedMoE_N3 / A12 variants
- All launchers use phase-wide queue execution with one row per (setting, seed)

## Launcher Scripts and Coverage

### 1) run_ablation_12.sh
Path: experiments/run/fmoe_n3/ablation/run_ablation_12.sh
Entrypoint: experiments/run/fmoe_n3/ablation/run_ablation_12.py
Axis/Phase: ablation_feature_add_v3_a12_kuairec_v1 / P17A

What it tests (12 settings):
- baseline (1): fixed A12 baseline
- layout (4): stage/layout structural variants
- feature_drop (3): category/timestamp-derived feature removals
- no_moe_bias (2): dense replacement with gated bias injection
- rule_router (2): rule-soft router variants

Settings:
1. ABL-01_BASELINE_A12: fixed baseline selected from P16A best criterion
2. ABL-02_LAYOUT_NO_MACRO: replace macro_ffn with dense_plain FFN (no skip)
3. ABL-03_LAYOUT_NO_MID: replace mid_ffn with dense_plain FFN (no skip)
4. ABL-04_LAYOUT_NO_MICRO: replace micro_ffn with dense_plain FFN (no skip)
5. ABL-05_LAYOUT_ATTN_BEFORE_MID: insert extra attention before mid_ffn
6. ABL-06_FEATURE_DROP_CATEGORY: drop category-derived feature slices
7. ABL-07_FEATURE_DROP_TIMESTAMP_DERIVED: drop timestamp-derived feature slices
8. ABL-08_FEATURE_DROP_CATEGORY_TIMESTAMP: drop category + timestamp-derived slices
9. ABL-09_DENSE_GATED_BIAS_MACRO_MID: dense macro/mid + gated_bias
10. ABL-10_DENSE_GATED_BIAS_FULL: dense macro/mid/micro + gated_bias
11. ABL-11_RULE_ROUTER_ALL: rule-soft router on all stages
12. ABL-12_RULE_ROUTER_MACRO_ONLY: macro rule-soft, mid/micro learned

### 2) ablation_2_12.sh
Path: experiments/run/fmoe_n3/ablation/ablation_2_12.sh
Entrypoint: experiments/run/fmoe_n3/ablation/ablation_2_12.py
Axis/Phase: ablation_feature_add_v3_a12_kuairec_v2 / P17B

What it tests (12 settings):
- feature_group_subset (6): 1-group, 2-group, 3-group subsets from Tempo/Focus/Memory/Exposure
- aux_loss (6): knn/balance/z-loss decomposition and lambda-strength variants

Settings:
1. ABL2-01_FG_ONLY_MEMORY
2. ABL2-02_FG_ONLY_EXPOSURE
3. ABL2-03_FG_MEMORY_EXPOSURE
4. ABL2-04_FG_FOCUS_MEMORY
5. ABL2-05_FG_NO_TEMPO
6. ABL2-06_FG_NO_EXPOSURE
7. ABL2-07_AUX_BALANCE_ONLY
8. ABL2-08_AUX_NONE
9. ABL2-09_AUX_KNN_ONLY
10. ABL2-10_AUX_KNN_BALANCE
11. ABL2-11_AUX_KNN_ONLY_LAMBDA_UP
12. ABL2-12_AUX_KNN_BALANCE_LAMBDA_UP

### 3) ablation_3_12.sh
Path: experiments/run/fmoe_n3/ablation/ablation_3_12.sh
Entrypoint: experiments/run/fmoe_n3/ablation/ablation_3_12.py
Axis/Phase: ablation_feature_add_v3_a12_kuairec_v3 / P17C

What it tests (12 settings):
- router_source (4): hidden/feature/both source decomposition
- router_granularity (4): token/session granularity decomposition
- feature_injection (4): gated_bias vs group_gated_bias decomposition

Settings:
1. ABL3-01_RS_ALL_FEATURE
2. ABL3-02_RS_ALL_HIDDEN
3. ABL3-03_RS_MACRO_FEATURE_OTHERS_BOTH
4. ABL3-04_RS_MICRO_HIDDEN_OTHERS_BOTH
5. ABL3-05_RG_ALL_TOKEN
6. ABL3-06_RG_ALL_SESSION
7. ABL3-07_RG_MACRO_TOKEN_MID_SESSION_MICRO_TOKEN
8. ABL3-08_RG_MACRO_SESSION_MID_TOKEN_MICRO_TOKEN
9. ABL3-09_FI_ALL_GATED_BIAS
10. ABL3-10_FI_MACRO_GATED_ONLY
11. ABL3-11_FI_ALL_GROUP_GATED_BIAS
12. ABL3-12_FI_MACRO_GROUP_GATED_ONLY

### 4) run_ablation_12_specify.sh
Path: experiments/run/fmoe_n3/ablation/run_ablation_12_specify.sh
Entrypoint: experiments/run/fmoe_n3/ablation/run_ablation_12_specify.py
Axis/Phase: ablation_feature_add_v3_a12_kuairec_v1_specify / P17AS

What it tests (up to 36 settings per seed):
- baseline (1): fixed A12 baseline
- layout deep-dive (11): dense replacement vs skip-removal vs dense-only variants
- feature_drop deep-dive (9): drop-only and keep-only variants (including category-only/time-only)
- dense_bias deep-dive (8): dense + gated/group-gated/no-injection families
- rule_router deep-dive (7): stage-targeted and source/granularity variants

Runtime defaults (specify launcher):
- seeds: 1,2,3
- max_evals: 10
- learning_rate choices: [0.0001, 0.00015, 0.00025, 0.00035, 0.0005, 0.0008, 0.0012]
- weight_decay choices: [1e-07, 5e-07, 1e-06, 5e-06, 1e-05]
- hidden_dropout_prob choices: [0.03, 0.05, 0.08, 0.1, 0.12]
- attn_dropout_prob choices: [0.05, 0.1, 0.15]
- max_settings_per_seed: 36

## Shared Runtime Defaults

Across current ablation launchers:
- search_algo: random
- max_evals: 5
- tune_epochs: 30
- tune_patience: 4
- train_batch_size: 4096
- eval_batch_size: 4096
- learning_rate choice set: [0.00015, 0.0002679394909038652, 0.00035, 0.0005, 0.0008]

## Typical Commands

Run all settings on one launcher:
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12.sh --gpus 0,1,2,3,4,5
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_2_12.sh --gpus 0,1,2,3,4,5
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_3_12.sh --gpus 0,1,2,3,4,5
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12_specify.sh --gpus 0,1,2,3,4,5 --seeds 1,2,3 --max-evals 10

Dry-run:
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12.sh --dry-run --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_2_12.sh --dry-run --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_3_12.sh --dry-run --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12_specify.sh --dry-run --gpus 0 --seeds 1,2,3

Subset run:
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12.sh --only-setting ABL-01_BASELINE_A12,ABL-07_FEATURE_DROP_TIMESTAMP_DERIVED --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_2_12.sh --only-setting ABL2-07_AUX_BALANCE_ONLY,ABL2-10_AUX_KNN_BALANCE --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/ablation_3_12.sh --only-setting ABL3-01_RS_ALL_FEATURE,ABL3-09_FI_ALL_GATED_BIAS --gpus 0
- /workspace/FeaturedMoE/experiments/run/fmoe_n3/ablation/run_ablation_12_specify.sh --only-setting ABLS-01_BASELINE_A12,ABLS-03_LAYOUT_NO_MACRO_SKIP,ABLS-10_LAYOUT_DENSE_FULL_ONLY --gpus 0

## Output Locations

- P17A logs summary: experiments/run/artifacts/logs/fmoe_n3/ablation_feature_add_v3_a12_kuairec_v1/KuaiRecLargeStrictPosV2_0.2/summary.csv
- P17B logs summary: experiments/run/artifacts/logs/fmoe_n3/ablation_feature_add_v3_a12_kuairec_v2/KuaiRecLargeStrictPosV2_0.2/summary.csv
- P17C logs summary: experiments/run/artifacts/logs/fmoe_n3/ablation_feature_add_v3_a12_kuairec_v3/KuaiRecLargeStrictPosV2_0.2/summary.csv
- P17AS logs summary: experiments/run/artifacts/logs/fmoe_n3/ablation_feature_add_v3_a12_kuairec_v1_specify/KuaiRecLargeStrictPosV2_0.2/summary.csv

Each completed row in summary.csv contains result_path, which points to the result json used for final metric extraction.
