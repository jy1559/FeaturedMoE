# fmoe_n Experiment Overview

- generated_at_utc: 2026-03-12T16:10:05.585803+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 84
- included_runs: 64
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 20
- summarized_experiments: 4

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| KuaiRecSmall0.1 | fmoe_n_S01_layout_lite_v1_arch_probe | s01_layout_lite_v1 | 36 | 0 | 0.019400 | 0.0194/0.0189/0.0184 | ARCH2_A04 | combo_id, fmoe_v2_layout_id, router_family, stage_inter_layer_style, feature_encoder_mode, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/ARCH2/KuaiRecSmall0.1/FMoE/20260312_134439_579_s01_layout_lite_v1_ARCH2_A04.log |
| KuaiRecSmall0.1 | fmoe_n_p0_anchor | hparam | 21 | 0 | 0.019300 | 0.0193/0.0179/0.0178 | P0_Q04 | combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_205828_746_hparam_P0_Q04.log |
| lastfm0.03 | fmoe_n_p0_anchor | hparam | 4 | 0 | 0.404900 | 0.4049/0.4044/0.1096 | P0_F01 | combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_172733_957_hparam_P0_F01.log |
| lastfm0.03 | fmoe_n_S01_layout_lite_v1_SMOKE | s01_layout_lite_v1 | 3 | 0 | 0.121200 | 0.1212/0.0234/0.0148 | SMOKE_ARCH3_A09 | fmoe_v2_layout_id, fmoe_stage_execution_mode, router_family, router_impl, stage_inter_layer_style, moe_block_variant, router_group_feature_mode, feature_encoder_mode, expert_scale, moe_top_k, learning_rate, weight_decay, balance_loss_lambda, z_loss_lambda, gate_entropy_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/SMOKE/lastfm0.03/FMoE/20260312_153928_003_s01_layout_lite_v1_SMOKE_ARCH3_A09.log |

## Experiment Notes

### KuaiRecSmall0.1 / fmoe_n_S01_layout_lite_v1_arch_probe

- 실험 설명: FeaturedMoE_N architecture probe for S01_layout_lite_v1.
- 실행 규모: runs=36, oom=0, 기간=2026-03-12T07:49:44.178250+00:00 ~ 2026-03-12T16:10:05.273293+00:00
- 비교 변수: combo_id, fmoe_v2_layout_id, router_family, stage_inter_layer_style, feature_encoder_mode, learning_rate
- 최고 성능: MRR@20=0.019400 (ARCH2_A04, FeaturedMoE_N_serial_plain_S01_layout_lite_v1)
- 최고 설정: fmoe_v2_layout_id=8, stage_inter_layer_style=attn, feature_encoder_mode=linear, learning_rate=0.0029173029436353353
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/ARCH2/KuaiRecSmall0.1/FMoE/20260312_134439_579_s01_layout_lite_v1_ARCH2_A04.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/KuaiRecSmall0.1_FeaturedMoE_N_arch2_a04_20260312_134442_905665_pid68924.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.019400 | ARCH2_A04 | fmoe_v2_layout_id=8, stage_inter_layer_style=attn, feature_encoder_mode=linear, learning_rate=0.0029173029436353353 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/ARCH2/KuaiRecSmall0.1/FMoE/20260312_134439_579_s01_layout_lite_v1_ARCH2_A04.log |
| 2 | 0.018900 | ARCH2_A08 | fmoe_v2_layout_id=30, stage_inter_layer_style=ffn, feature_encoder_mode=linear, learning_rate=0.0002216528005435058 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/ARCH2/KuaiRecSmall0.1/FMoE/20260312_143409_000_s01_layout_lite_v1_ARCH2_A08.log |
| 3 | 0.018400 | ARCH2_A05 | fmoe_v2_layout_id=30, stage_inter_layer_style=attn, feature_encoder_mode=linear, learning_rate=0.0003470028731718129 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/ARCH2/KuaiRecSmall0.1/FMoE/20260312_143408_943_s01_layout_lite_v1_ARCH2_A05.log |

### KuaiRecSmall0.1 / fmoe_n_p0_anchor

- 실험 설명: FeaturedMoE_N P0 anchor wave sweep.
- 실행 규모: runs=21, oom=0, 기간=2026-03-11T16:31:43.900418+00:00 ~ 2026-03-12T06:26:29.468820+00:00
- 비교 변수: combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate
- 최고 성능: MRR@20=0.019300 (P0_Q04, FeaturedMoE_N_serial_plain)
- 최고 설정: fmoe_v2_layout_id=16, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.007912891743533833
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_205828_746_hparam_P0_Q04.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/KuaiRecSmall0.1_FeaturedMoE_N_p0_q04_20260311_205831_905162_pid183119.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.019300 | P0_Q04 | fmoe_v2_layout_id=16, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.007912891743533833 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_205828_746_hparam_P0_Q04.log |
| 2 | 0.017900 | P0_Q09 | fmoe_v2_layout_id=19, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.00047449245768770243 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260312_000841_916_hparam_P0_Q09.log |
| 3 | 0.017800 | P0_Q01 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.00043488830562920437 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260312_054754_328_hparam_P0_Q01.log |

### lastfm0.03 / fmoe_n_p0_anchor

- 실험 설명: FeaturedMoE_N P0 anchor wave sweep.
- 실행 규모: runs=4, oom=0, 기간=2026-03-11T16:31:43.959483+00:00 ~ 2026-03-12T00:08:41.411396+00:00
- 비교 변수: combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate
- 최고 성능: MRR@20=0.404900 (P0_F01, FeaturedMoE_N_serial_plain)
- 최고 설정: fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.00021761017124097406
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_172733_957_hparam_P0_F01.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/lastfm0.03_FeaturedMoE_N_p0_f01_20260311_172737_378396_pid179938.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.404900 | P0_F01 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.00021761017124097406 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_172733_957_hparam_P0_F01.log |
| 2 | 0.404400 | P0_F02 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.00024912392301725146 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_205828_817_hparam_P0_F02.log |
| 3 | 0.109600 | P0_F01 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0034993361142320097 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_163143_910_hparam_P0_F01.log |

### lastfm0.03 / fmoe_n_S01_layout_lite_v1_SMOKE

- 실험 설명: FeaturedMoE_N state=S01_layout_lite_v1 hyperopt run with fixed combo and LR-first search.
- 실행 규모: runs=3, oom=0, 기간=2026-03-12T15:39:28.049484+00:00 ~ 2026-03-12T15:41:27.023061+00:00
- 비교 변수: fmoe_v2_layout_id, fmoe_stage_execution_mode, router_family, router_impl, stage_inter_layer_style, moe_block_variant, router_group_feature_mode, feature_encoder_mode, expert_scale, moe_top_k, learning_rate, weight_decay, balance_loss_lambda, z_loss_lambda, gate_entropy_lambda
- 최고 성능: MRR@20=0.121200 (SMOKE_ARCH3_A09, FeaturedMoE_N_serial_plain_S01_layout_lite_v1)
- 최고 설정: fmoe_v2_layout_id=30, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=identity, router_group_feature_mode=none, feature_encoder_mode=linear, expert_scale=3, moe_top_k=0, learning_rate=0.0017438986221667654, weight_decay=5e-05, balance_loss_lambda=0.0, z_loss_lambda=0.0, gate_entropy_lambda=0.0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/SMOKE/lastfm0.03/FMoE/20260312_153928_003_s01_layout_lite_v1_SMOKE_ARCH3_A09.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/lastfm0.03_FeaturedMoE_N_smoke_arch3_a09_20260312_153931_118131_pid85724.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.121200 | SMOKE_ARCH3_A09 | fmoe_v2_layout_id=30, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=identity, router_group_feature_mode=none, feature_encoder_mode=linear, expert_scale=3, moe_top_k=0, learning_rate=0.0017438986221667654, weight_decay=5e-05, balance_loss_lambda=0.0, z_loss_lambda=0.0, gate_entropy_lambda=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/SMOKE/lastfm0.03/FMoE/20260312_153928_003_s01_layout_lite_v1_SMOKE_ARCH3_A09.log |
| 2 | 0.023400 | SMOKE_ARCH3_A13 | fmoe_v2_layout_id=30, fmoe_stage_execution_mode=serial, router_impl=rule_soft, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=none, feature_encoder_mode=linear, expert_scale=3, moe_top_k=0, learning_rate=0.0005456489408178492, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0, gate_entropy_lambda=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/SMOKE/lastfm0.03/FMoE/20260312_154002_206_s01_layout_lite_v1_SMOKE_ARCH3_A13.log |
| 3 | 0.014800 | SMOKE_ARCH3_A25_FIX | fmoe_v2_layout_id=30, fmoe_stage_execution_mode=serial, router_impl=learned, stage_inter_layer_style=attn, moe_block_variant=moe, router_group_feature_mode=mean, feature_encoder_mode=linear, expert_scale=3, moe_top_k=0, learning_rate=0.00016212569874927477, weight_decay=5e-05, balance_loss_lambda=0.002, z_loss_lambda=0.0001, gate_entropy_lambda=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/s01_layout_lite_v1/SMOKE/lastfm0.03/FMoE/20260312_154104_431_s01_layout_lite_v1_SMOKE_ARCH3_A25_FIX.log |

