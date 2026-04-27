# fmoe_n2 Experiment Overview

- generated_at_utc: 2026-03-13T08:32:28.306200+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 37
- included_runs: 37
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 2

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| KuaiRecLargeStrictPosV2_0.2 | fmoe_n2_S00_router_feature_heavy_v1_arch3 | s00_router_feature_heavy_v1 | 27 | 0 | 0.077800 | 0.0778/0.0769/0.0769 | ARCH3_A22 | combo_id, fmoe_v2_layout_id, router_family, router_impl, router_use_hidden, router_use_feature, router_feature_proj_dim, router_feature_scale, router_hidden_scale, router_group_feature_mode, stage_inter_layer_style, moe_block_variant, rule_agreement_lambda, group_coverage_lambda, lr_scheduler_type, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/ARCH3/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260313_034629_711_s00_router_feature_heavy_v1_ARCH3_A22.log |
| KuaiRecLargeStrictPosV2_0.2 | fmoe_n2_S00_router_feature_heavy_v1_SMOKE | s00_router_feature_heavy_v1 | 10 | 0 | 0.064300 | 0.0643/0.0640/0.0640 | SMOKE_A13 | fmoe_v2_layout_id, fmoe_stage_execution_mode, router_family, router_impl, stage_inter_layer_style, moe_block_variant, router_group_feature_mode, router_use_hidden, router_use_feature, router_feature_proj_dim, router_feature_scale, router_hidden_scale, feature_encoder_mode, expert_scale, moe_top_k, learning_rate, weight_decay, balance_loss_lambda, z_loss_lambda, gate_entropy_lambda, rule_agreement_lambda, group_coverage_lambda, lr_scheduler_type | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/SMOKE/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260312_170004_453_s00_router_feature_heavy_v1_SMOKE_A13.log |

## Experiment Notes

### KuaiRecLargeStrictPosV2_0.2 / fmoe_n2_S00_router_feature_heavy_v1_arch3

- 실험 설명: FeaturedMoE_N2 ARCH3 feature-heavy probe.
- 실행 규모: runs=27, oom=0, 기간=2026-03-12T17:17:11.758685+00:00 ~ 2026-03-13T08:26:18.890380+00:00
- 비교 변수: combo_id, fmoe_v2_layout_id, router_family, router_impl, router_use_hidden, router_use_feature, router_feature_proj_dim, router_feature_scale, router_hidden_scale, router_group_feature_mode, stage_inter_layer_style, moe_block_variant, rule_agreement_lambda, group_coverage_lambda, lr_scheduler_type, learning_rate
- 최고 성능: MRR@20=0.077800 (ARCH3_A22, FeaturedMoE_N2_serial_plain_S00_router_feature_heavy_v1)
- 최고 설정: fmoe_v2_layout_id=8, router_impl=learned, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=1.5, router_hidden_scale=1.0, router_group_feature_mode=mean_std, stage_inter_layer_style=attn, moe_block_variant=moe, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none, learning_rate=0.0002932133197314927
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/ARCH3/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260313_034629_711_s00_router_feature_heavy_v1_ARCH3_A22.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n2/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N2_arch3_a22_20260313_034633_039510_pid133070.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.077800 | ARCH3_A22 | fmoe_v2_layout_id=8, router_impl=learned, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=1.5, router_hidden_scale=1.0, router_group_feature_mode=mean_std, stage_inter_layer_style=attn, moe_block_variant=moe, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none, learning_rate=0.0002932133197314927 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/ARCH3/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260313_034629_711_s00_router_feature_heavy_v1_ARCH3_A22.log |
| 2 | 0.076900 | ARCH3_A21 | fmoe_v2_layout_id=8, router_impl=learned, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=1.5, router_hidden_scale=1.0, router_group_feature_mode=mean_std, stage_inter_layer_style=attn, moe_block_variant=moe, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none, learning_rate=0.00030652805103765823 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/ARCH3/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260313_034629_678_s00_router_feature_heavy_v1_ARCH3_A21.log |
| 3 | 0.076900 | ARCH3_A19 | fmoe_v2_layout_id=8, router_impl=learned, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=256, router_feature_scale=2.5, router_hidden_scale=0.5, router_group_feature_mode=mean_std, stage_inter_layer_style=attn, moe_block_variant=moe, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none, learning_rate=0.0002748605737749342 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/ARCH3/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260313_010148_785_s00_router_feature_heavy_v1_ARCH3_A19.log |

### KuaiRecLargeStrictPosV2_0.2 / fmoe_n2_S00_router_feature_heavy_v1_SMOKE

- 실험 설명: FeaturedMoE_N2 state=S00_router_feature_heavy_v1 hyperopt run with fixed combo and LR-first search.
- 실행 규모: runs=10, oom=0, 기간=2026-03-12T16:58:38.974384+00:00 ~ 2026-03-12T17:03:45.935790+00:00
- 비교 변수: fmoe_v2_layout_id, fmoe_stage_execution_mode, router_family, router_impl, stage_inter_layer_style, moe_block_variant, router_group_feature_mode, router_use_hidden, router_use_feature, router_feature_proj_dim, router_feature_scale, router_hidden_scale, feature_encoder_mode, expert_scale, moe_top_k, learning_rate, weight_decay, balance_loss_lambda, z_loss_lambda, gate_entropy_lambda, rule_agreement_lambda, group_coverage_lambda, lr_scheduler_type
- 최고 성능: MRR@20=0.064300 (SMOKE_A13, FeaturedMoE_N2_serial_plain_S00_router_feature_heavy_v1)
- 최고 설정: fmoe_v2_layout_id=8, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=none, router_use_hidden=False, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=1.5, router_hidden_scale=1.0, feature_encoder_mode=linear, expert_scale=1, moe_top_k=0, learning_rate=7.607699787069922e-05, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0, gate_entropy_lambda=0.0, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/SMOKE/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260312_170004_453_s00_router_feature_heavy_v1_SMOKE_A13.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n2/KuaiRecLargeStrictPosV2_0.2_FeaturedMoE_N2_smoke_a13_20260312_170007_563828_pid108806.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.064300 | SMOKE_A13 | fmoe_v2_layout_id=8, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=none, router_use_hidden=False, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=1.5, router_hidden_scale=1.0, feature_encoder_mode=linear, expert_scale=1, moe_top_k=0, learning_rate=7.607699787069922e-05, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0, gate_entropy_lambda=0.0, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/SMOKE/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260312_170004_453_s00_router_feature_heavy_v1_SMOKE_A13.log |
| 2 | 0.064000 | SMOKE_A29 | fmoe_v2_layout_id=8, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=mean_std, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=128, router_feature_scale=2.0, router_hidden_scale=0.75, feature_encoder_mode=linear, expert_scale=1, moe_top_k=0, learning_rate=7.012043905115853e-05, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0001, gate_entropy_lambda=0.0, rule_agreement_lambda=0.0, group_coverage_lambda=0.0, lr_scheduler_type=none | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/SMOKE/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260312_170202_895_s00_router_feature_heavy_v1_SMOKE_A29.log |
| 3 | 0.064000 | SMOKE_A27 | fmoe_v2_layout_id=8, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=mean_std, router_use_hidden=True, router_use_feature=True, router_feature_proj_dim=0, router_feature_scale=1.0, router_hidden_scale=1.0, feature_encoder_mode=linear, expert_scale=1, moe_top_k=0, learning_rate=7.607699787069922e-05, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0, gate_entropy_lambda=0.0, rule_agreement_lambda=0.01, group_coverage_lambda=0.002, lr_scheduler_type=none | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n2/s00_router_feature_heavy_v1/SMOKE/KuaiRecLargeStrictPosV2_0.2/FMoEN2/20260312_170133_286_s00_router_feature_heavy_v1_SMOKE_A27.log |

