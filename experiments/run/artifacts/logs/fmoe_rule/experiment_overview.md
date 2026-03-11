# fmoe_rule Experiment Overview

- generated_at_utc: 2026-03-10T02:34:09.957345+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 28
- included_runs: 15
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 13
- summarized_experiments: 5

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | rule_split_P2DB_R1_movielens1m | hparam | 6 | 0 | 0.098800 | 0.0988/0.0978/0.0956 | RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096 | ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_112910_286_hparam_RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096.log |
| movielens1m | rule_split_P2DB_R0_movielens1m | hparam | 6 | 0 | 0.075200 | 0.0752/0.0744/0.0742 | RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192 | ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_113342_923_hparam_RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192.log |
| retail_rocket | rr_rule_quick_r1_l16f24 | hparam | 1 | 0 | 0.262000 | 0.2620 | RRRULE_R1_G4_C00_L16F24 | ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011924_825_hparam_RRRULE_R1_G4_C00_L16F24.log |
| retail_rocket | rr_rule_quick_r1_l15med | hparam | 1 | 0 | 0.261000 | 0.2610 | RRRULE_R1_G7_C03_L15MED | ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011927_820_hparam_RRRULE_R1_G7_C03_L15MED.log |
| retail_rocket | rr_rule_quick_r1_l16big | hparam | 1 | 0 | 0.259900 | 0.2599 | RRRULE_R1_G6_C02_L16BIG | ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011926_824_hparam_RRRULE_R1_G6_C02_L16BIG.log |

## Experiment Notes

### movielens1m / rule_split_P2DB_R1_movielens1m

- 실험 설명: Split run: rule=R1, fixed combo(dim/router/batch), tune lr/wd(max_evals=10).
- 실행 규모: runs=6, oom=0, 기간=2026-03-06T08:45:22.016565+00:00 ~ 2026-03-06T16:45:42.463319+00:00
- 비교 변수: ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay
- 최고 성능: MRR@20=0.098800 (RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096, FeaturedMoE_v2_serial_R1)
- 최고 설정: router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=192, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, learning_rate=0.0011522285965359267, weight_decay=0.00014082672264067198
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_112910_286_hparam_RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_rule/movielens1m_FeaturedMoE_v2_20260306_112913.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.098800 | RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096 | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=192, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, learning_rate=0.0011522285965359267, weight_decay=0.00014082672264067198 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_112910_286_hparam_RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096.log |
| 2 | 0.097800 | RULE_R1_P2DB_G7_C2_movielens1m_E160_R80_B4096 | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, learning_rate=0.0003013335862289979, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_113927_017_hparam_RULE_R1_P2DB_G7_C2_movielens1m_E160_R80_B4096.log |
| 3 | 0.095600 | RULE_R1_P2DB_G7_C3_movielens1m_E160_R96_B6144 | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=160, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, learning_rate=0.00085298788056459, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_141701_721_hparam_RULE_R1_P2DB_G7_C3_movielens1m_E160_R96_B6144.log |

### movielens1m / rule_split_P2DB_R0_movielens1m

- 실험 설명: Split run: rule=R0, fixed combo(dim/router/batch), tune lr/wd(max_evals=10).
- 실행 규모: runs=6, oom=0, 기간=2026-03-06T08:45:22.025397+00:00 ~ 2026-03-06T17:21:19.436087+00:00
- 비교 변수: ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay
- 최고 성능: MRR@20=0.075200 (RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192, FeaturedMoE_v2_serial_R0)
- 최고 설정: router_impl=rule_soft, router_impl_by_stage={}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, learning_rate=0.00995202319485767, weight_decay=1.650729197309882e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_113342_923_hparam_RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_rule/movielens1m_FeaturedMoE_v2_20260306_113345.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.075200 | RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192 | router_impl=rule_soft, router_impl_by_stage={}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, learning_rate=0.00995202319485767, weight_decay=1.650729197309882e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_113342_923_hparam_RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192.log |
| 2 | 0.074400 | RULE_R0_P2DB_G4_C2_movielens1m_E128_R64_B6144 | router_impl=rule_soft, router_impl_by_stage={}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, learning_rate=0.004502716695123429, weight_decay=4.408146121720182e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_114200_390_hparam_RULE_R0_P2DB_G4_C2_movielens1m_E128_R64_B6144.log |
| 3 | 0.074200 | RULE_R0_P2DB_G6_C3_movielens1m_E128_R64_B4096 | router_impl=rule_soft, router_impl_by_stage={}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, learning_rate=0.006913561277648127, weight_decay=1.3355830399517142e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RULE/ML1/FMoEv2/20260306_142754_882_hparam_RULE_R0_P2DB_G6_C3_movielens1m_E128_R64_B4096.log |

### retail_rocket / rr_rule_quick_r1_l16f24

- 실험 설명: RetailRocket quick rule probe: ML1 hybrid-rule winner + RR v2 winning layouts/dims, mostly R1 with R0 sentinels.
- 실행 규모: runs=1, oom=0, 기간=2026-03-10T01:19:24.871433+00:00 ~ 2026-03-10T02:29:18.118688+00:00
- 비교 변수: ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.262000 (RRRULE_R1_G4_C00_L16F24, FeaturedMoE_v2_serial_R1)
- 최고 설정: router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.0003598471606608668, weight_decay=5.478962838879961e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011924_825_hparam_RRRULE_R1_G4_C00_L16F24.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_rule/retail_rocket_FeaturedMoE_v2_rrrule_r1_g4_c00_l16f24_20260310_011927_548427_pid542044.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.262000 | RRRULE_R1_G4_C00_L16F24 | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.0003598471606608668, weight_decay=5.478962838879961e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011924_825_hparam_RRRULE_R1_G4_C00_L16F24.log |

### retail_rocket / rr_rule_quick_r1_l15med

- 실험 설명: RetailRocket quick rule probe: ML1 hybrid-rule winner + RR v2 winning layouts/dims, mostly R1 with R0 sentinels.
- 실행 규모: runs=1, oom=0, 기간=2026-03-10T01:19:27.866702+00:00 ~ 2026-03-10T02:34:09.755442+00:00
- 비교 변수: ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.261000 (RRRULE_R1_G7_C03_L15MED, FeaturedMoE_v2_serial_R1)
- 최고 설정: router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.0004201548471253923, weight_decay=6.534533777192384e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011927_820_hparam_RRRULE_R1_G7_C03_L15MED.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_rule/retail_rocket_FeaturedMoE_v2_rrrule_r1_g7_c03_l15med_20260310_011930_724789_pid542385.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261000 | RRRULE_R1_G7_C03_L15MED | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.0004201548471253923, weight_decay=6.534533777192384e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011927_820_hparam_RRRULE_R1_G7_C03_L15MED.log |

### retail_rocket / rr_rule_quick_r1_l16big

- 실험 설명: RetailRocket quick rule probe: ML1 hybrid-rule winner + RR v2 winning layouts/dims, mostly R1 with R0 sentinels.
- 실행 규모: runs=1, oom=0, 기간=2026-03-10T01:19:26.871182+00:00 ~ 2026-03-10T02:27:59.179005+00:00
- 비교 변수: ablation, router_impl, router_impl_by_stage, rule_router.n_bins, rule_router.feature_per_expert, fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.259900 (RRRULE_R1_G6_C02_L16BIG, FeaturedMoE_v2_serial_R1)
- 최고 설정: router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.000427733547621967, weight_decay=5.3504874313683086e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011926_824_hparam_RRRULE_R1_G6_C02_L16BIG.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_rule/retail_rocket_FeaturedMoE_v2_rrrule_r1_g6_c02_l16big_20260310_011929_693445_pid542281.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.259900 | RRRULE_R1_G6_C02_L16BIG | router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.n_bins=5, rule_router.feature_per_expert=4, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, hidden_dropout_prob=0.1, balance_loss_lambda=0.01, learning_rate=0.000427733547621967, weight_decay=5.3504874313683086e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_rule/hparam/RRRULE/ReR/FMoEv2/20260310_011926_824_hparam_RRRULE_R1_G6_C02_L16BIG.log |

