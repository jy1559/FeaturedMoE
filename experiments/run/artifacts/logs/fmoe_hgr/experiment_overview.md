# fmoe_hgr Experiment Overview

- generated_at_utc: 2026-03-10T01:14:11.353980+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 58
- included_runs: 58
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 5

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | P3_hgr_router_teach | hparam | 11 | 0 | 0.095800 | 0.0958/0.0956/0.0954 | P3HGR_router16_C14_A1_M6_serial_hybrid | arch_layout_id, stage_merge_mode, group_router_mode, group_top_k, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, router_distill_enable, router_distill_lambda, router_distill_temperature, router_distill_until, group_feature_spec_aux_lambda, group_feature_spec_stages, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P3HGR/ML1/FMoEHGR/20260309_213415_665_hparam_P3HGR_router16_C14_A1_M6_serial_hybrid.log |
| movielens1m | P2_hgr_dim_focus | hparam | 1 | 0 | 0.095600 | 0.0956 | P2HGR_dim8_C00_L15H_D0_serial_hybrid | arch_layout_id, stage_merge_mode, group_router_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P2HGR/ML1/FMoEHGR/20260309_123510_607_hparam_P2HGR_dim8_C00_L15H_D0_serial_hybrid.log |
| movielens1m | P1_hgr_wide_shallow | hparam | 28 | 0 | 0.094600 | 0.0946/0.0941/0.0941 | P1HGR_widewide_C64_serial_per_group | stage_merge_mode, group_router_mode, arch_layout_id, group_top_k, moe_top_k, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260308_224610_035_hparam_P1HGR_widewide_C64_serial_per_group.log |
| movielens1m | P15_hgr_layout_focus | hparam | 13 | 0 | 0.093700 | 0.0937/0.0933/0.0923 | P15HGR_layout24_C14_L15A_R2_serial_hybrid | arch_layout_id, num_layers, stage_merge_mode, group_router_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P15HGR/ML1/FMoEHGR/20260309_100000_904_hparam_P15HGR_layout24_C14_L15A_R2_serial_hybrid.log |
| movielens1m | P1_hgr_joint_fast32 | hparam | 5 | 0 | 0.092000 | 0.0920/0.0913/0.0879 | P1HGR_joint32_C16_A4_R0_serial_per_group | arch_layout_id, num_layers, stage_merge_mode, group_router_mode, group_top_k, expert_top_k, expert_use_feature, macro_routing_scope, parallel_stage_gate_temperature, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260309_041811_506_hparam_P1HGR_joint32_C16_A4_R0_serial_per_group.log |

## Experiment Notes

### movielens1m / P3_hgr_router_teach

- 실험 설명: Post-P2 HGR router-teaching phase. Structure stays fixed around the best L15-hybrid anchors, and router supervision/distillation variants are compared. combo=A0_M4 layout=15 merge=serial group=hybrid dims=128/16/160/64 scale=3 bs=4096/8192 distill=true:5e-3@tau1.5/until0.3 spec=[mid]:1e-4 group_top_k=0
- 실행 규모: runs=11, oom=0, 기간=2026-03-09T15:04:36.182997+00:00 ~ 2026-03-10T01:05:58.407968+00:00
- 비교 변수: arch_layout_id, stage_merge_mode, group_router_mode, group_top_k, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, router_distill_enable, router_distill_lambda, router_distill_temperature, router_distill_until, group_feature_spec_aux_lambda, group_feature_spec_stages, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda
- 최고 성능: MRR@20=0.095800 (P3HGR_router16_C14_A1_M6_serial_hybrid, FeaturedMoE_HGR_serial_hybrid)
- 최고 설정: arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, group_top_k=0, embedding_size=160, d_feat_emb=16, d_expert_hidden=256, d_router_hidden=112, expert_scale=3, router_distill_enable=True, router_distill_lambda=0.005, router_distill_temperature=1.5, router_distill_until=0.2, group_feature_spec_aux_lambda=0.0003, group_feature_spec_stages=['macro', 'mid'], learning_rate=0.00038613928476409276, weight_decay=4.897470001571935e-05, hidden_dropout_prob=0.10803012915104979, balance_loss_lambda=0.0030773871258551943
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P3HGR/ML1/FMoEHGR/20260309_213415_665_hparam_P3HGR_router16_C14_A1_M6_serial_hybrid.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p3hgr_router16_c14_a1_m6_serial_hybrid_20260309_213418_478049_pid529617.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.095800 | P3HGR_router16_C14_A1_M6_serial_hybrid | arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, group_top_k=0, embedding_size=160, d_feat_emb=16, d_expert_hidden=256, d_router_hidden=112, expert_scale=3, router_distill_enable=True, router_distill_lambda=0.005, router_distill_temperature=1.5, router_distill_until=0.2, group_feature_spec_aux_lambda=0.0003, group_feature_spec_stages=['macro', 'mid'], learning_rate=0.00038613928476409276, weight_decay=4.897470001571935e-05, hidden_dropout_prob=0.10803012915104979, balance_loss_lambda=0.0030773871258551943 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P3HGR/ML1/FMoEHGR/20260309_213415_665_hparam_P3HGR_router16_C14_A1_M6_serial_hybrid.log |
| 2 | 0.095600 | P3HGR_router16_C08_A1_M0_serial_hybrid | arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, group_top_k=0, embedding_size=160, d_feat_emb=16, d_expert_hidden=256, d_router_hidden=112, expert_scale=3, router_distill_enable=False, router_distill_lambda=0.0, router_distill_temperature=1.5, router_distill_until=0.2, group_feature_spec_aux_lambda=0.0001, group_feature_spec_stages=['mid'], learning_rate=0.00035503195583268986, weight_decay=3.572633224168534e-05, hidden_dropout_prob=0.10743433142281522, balance_loss_lambda=0.0030619363423038653 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P3HGR/ML1/FMoEHGR/20260309_150436_126_hparam_P3HGR_router16_C08_A1_M0_serial_hybrid.log |
| 3 | 0.095400 | P3HGR_router16_C00_A0_M0_serial_hybrid | arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, group_top_k=0, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, router_distill_enable=False, router_distill_lambda=0.0, router_distill_temperature=1.5, router_distill_until=0.2, group_feature_spec_aux_lambda=0.0001, group_feature_spec_stages=['mid'], learning_rate=0.002301208903089608, weight_decay=1.2178096156225857e-05, hidden_dropout_prob=0.10355873698454011, balance_loss_lambda=0.0026304550714151796 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P3HGR/ML1/FMoEHGR/20260309_150436_126_hparam_P3HGR_router16_C00_A0_M0_serial_hybrid.log |

### movielens1m / P2_hgr_dim_focus

- 실험 설명: Post-P15 HGR P2 dim focus. Layout/route are nearly fixed from P15, and only dim/batch/LR coupling is probed. combo=L15H_D0 layout=15 merge=serial group=hybrid dims=128/16/160/64 scale=3 bs=4096/8192
- 실행 규모: runs=1, oom=0, 기간=2026-03-09T12:35:10.656438+00:00 ~ 2026-03-09T14:58:44.453750+00:00
- 비교 변수: arch_layout_id, stage_merge_mode, group_router_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda
- 최고 성능: MRR@20=0.095600 (P2HGR_dim8_C00_L15H_D0_serial_hybrid, FeaturedMoE_HGR_serial_hybrid)
- 최고 설정: arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.002631627138677997, weight_decay=3.0981640626929434e-06, hidden_dropout_prob=0.11700993951905542, balance_loss_lambda=0.002611703960022245
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P2HGR/ML1/FMoEHGR/20260309_123510_607_hparam_P2HGR_dim8_C00_L15H_D0_serial_hybrid.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p2hgr_dim8_c00_l15h_d0_serial_hybrid_20260309_123513_415957_pid508125.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.095600 | P2HGR_dim8_C00_L15H_D0_serial_hybrid | arch_layout_id=15, stage_merge_mode=serial, group_router_mode=hybrid, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.002631627138677997, weight_decay=3.0981640626929434e-06, hidden_dropout_prob=0.11700993951905542, balance_loss_lambda=0.002611703960022245 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P2HGR/ML1/FMoEHGR/20260309_123510_607_hparam_P2HGR_dim8_C00_L15H_D0_serial_hybrid.log |

### movielens1m / P1_hgr_wide_shallow

- 실험 설명: Wide-shallow HGR screen for combo pruning. Half of the budget stays on base layout [1,1,1,1,0] to isolate routing/capacity effects; the other half stresses layout diversity and outliers. combo=C60 layout=0 merge=parallel group=hybrid group_topk=2 moe_topk=0
- 실행 규모: runs=28, oom=0, 기간=2026-03-08T14:47:02.501295+00:00 ~ 2026-03-09T02:08:38.878201+00:00
- 비교 변수: stage_merge_mode, group_router_mode, arch_layout_id, group_top_k, moe_top_k, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay
- 최고 성능: MRR@20=0.094600 (P1HGR_widewide_C64_serial_per_group, FeaturedMoE_HGR_serial_per_group)
- 최고 설정: stage_merge_mode=serial, group_router_mode=per_group, arch_layout_id=0, group_top_k=0, moe_top_k=0, embedding_size=160, d_feat_emb=16, d_expert_hidden=224, d_router_hidden=96, expert_scale=3, learning_rate=0.0019442623664061261, weight_decay=2.799101613038829e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260308_224610_035_hparam_P1HGR_widewide_C64_serial_per_group.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p1hgr_widewide_c64_serial_per_group_20260308_224612_789841_pid392990.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.094600 | P1HGR_widewide_C64_serial_per_group | stage_merge_mode=serial, group_router_mode=per_group, arch_layout_id=0, group_top_k=0, moe_top_k=0, embedding_size=160, d_feat_emb=16, d_expert_hidden=224, d_router_hidden=96, expert_scale=3, learning_rate=0.0019442623664061261, weight_decay=2.799101613038829e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260308_224610_035_hparam_P1HGR_widewide_C64_serial_per_group.log |
| 2 | 0.094100 | P1HGR_widewide_C61_serial_per_group | stage_merge_mode=serial, group_router_mode=per_group, arch_layout_id=15, group_top_k=0, moe_top_k=0, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.001146810727426218, weight_decay=1.1348910214913785e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260308_180458_516_hparam_P1HGR_widewide_C61_serial_per_group.log |
| 3 | 0.094100 | P1HGR_widewide_C21_serial_per_group | stage_merge_mode=serial, group_router_mode=per_group, arch_layout_id=5, group_top_k=0, moe_top_k=0, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.002043397479526177, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260308_180107_255_hparam_P1HGR_widewide_C21_serial_per_group.log |

### movielens1m / P15_hgr_layout_focus

- 실험 설명: Layout-focused HGR P1.5 screen. Uses widewide top layouts as anchors, keeps aux routing regularizers fixed, and only allows small dim variation so structure ranking is cleaner before P2. combo=C18 arch=L10A layout=10 total_layers=3 merge=serial group=per_group sched=off
- 실행 규모: runs=13, oom=0, 기간=2026-03-09T06:50:40.978352+00:00 ~ 2026-03-09T11:47:45.063815+00:00
- 비교 변수: arch_layout_id, num_layers, stage_merge_mode, group_router_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda
- 최고 성능: MRR@20=0.093700 (P15HGR_layout24_C14_L15A_R2_serial_hybrid, FeaturedMoE_HGR_serial_hybrid)
- 최고 설정: arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=hybrid, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.002276428743515581, weight_decay=2.8891746164538358e-05, hidden_dropout_prob=0.08354329495272685, balance_loss_lambda=0.0033777112278553915
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P15HGR/ML1/FMoEHGR/20260309_100000_904_hparam_P15HGR_layout24_C14_L15A_R2_serial_hybrid.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p15hgr_layout24_c14_l15a_r2_serial_hybrid_20260309_100003_791102_pid490439.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.093700 | P15HGR_layout24_C14_L15A_R2_serial_hybrid | arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=hybrid, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.002276428743515581, weight_decay=2.8891746164538358e-05, hidden_dropout_prob=0.08354329495272685, balance_loss_lambda=0.0033777112278553915 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P15HGR/ML1/FMoEHGR/20260309_100000_904_hparam_P15HGR_layout24_C14_L15A_R2_serial_hybrid.log |
| 2 | 0.093300 | P15HGR_layout24_C12_L15A_R0_serial_per_group | arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.0030687135628300593, weight_decay=3.6093815404044244e-06, hidden_dropout_prob=0.10864900728114563, balance_loss_lambda=0.0023022322717874133 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P15HGR/ML1/FMoEHGR/20260309_065040_930_hparam_P15HGR_layout24_C12_L15A_R0_serial_per_group.log |
| 3 | 0.092300 | P15HGR_layout24_C13_L15A_R1_serial_per_group | arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.001654444245735315, weight_decay=1.1912275651921694e-05, hidden_dropout_prob=0.11503125845511669, balance_loss_lambda=0.0026508586224976787 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P15HGR/ML1/FMoEHGR/20260309_082517_563_hparam_P15HGR_layout24_C13_L15A_R1_serial_per_group.log |

### movielens1m / P1_hgr_joint_fast32

- 실험 설명: Fast ML1M HGR vNext screen after router redesign. Jointly sweeps layout depth and model capacity with 8 layout-capacity anchors x 4 routing/schedule profiles so structure and dimension effects are visible immediately. combo=C8 arch=A2 route=R0 layout=5 total_layers=4 merge=serial group=per_group expert_feat=false sched=off
- 실행 규모: runs=5, oom=0, 기간=2026-03-09T04:18:11.555528+00:00 ~ 2026-03-09T06:26:36.358156+00:00
- 비교 변수: arch_layout_id, num_layers, stage_merge_mode, group_router_mode, group_top_k, expert_top_k, expert_use_feature, macro_routing_scope, parallel_stage_gate_temperature, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda
- 최고 성능: MRR@20=0.092000 (P1HGR_joint32_C16_A4_R0_serial_per_group, FeaturedMoE_HGR_serial_per_group)
- 최고 설정: arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, group_top_k=0, expert_top_k=1, expert_use_feature=False, macro_routing_scope=session, parallel_stage_gate_temperature=1.0, embedding_size=160, d_feat_emb=16, d_expert_hidden=224, d_router_hidden=96, expert_scale=3, learning_rate=0.0007300305349331221, weight_decay=1.3926780596377315e-06, hidden_dropout_prob=0.08588891365938163, balance_loss_lambda=0.004246119572442363
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260309_041811_506_hparam_P1HGR_joint32_C16_A4_R0_serial_per_group.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr/movielens1m_FeaturedMoE_HGR_p1hgr_joint32_c16_a4_r0_serial_per_group_20260309_041814_173363_pid440497.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.092000 | P1HGR_joint32_C16_A4_R0_serial_per_group | arch_layout_id=15, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, group_top_k=0, expert_top_k=1, expert_use_feature=False, macro_routing_scope=session, parallel_stage_gate_temperature=1.0, embedding_size=160, d_feat_emb=16, d_expert_hidden=224, d_router_hidden=96, expert_scale=3, learning_rate=0.0007300305349331221, weight_decay=1.3926780596377315e-06, hidden_dropout_prob=0.08588891365938163, balance_loss_lambda=0.004246119572442363 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260309_041811_506_hparam_P1HGR_joint32_C16_A4_R0_serial_per_group.log |
| 2 | 0.091300 | P1HGR_joint32_C24_A6_R0_serial_per_group | arch_layout_id=20, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, group_top_k=0, expert_top_k=1, expert_use_feature=True, macro_routing_scope=session, parallel_stage_gate_temperature=1.0, embedding_size=128, d_feat_emb=24, d_expert_hidden=224, d_router_hidden=96, expert_scale=3, learning_rate=0.0007870661397440257, weight_decay=3.386000463308652e-05, hidden_dropout_prob=0.12543549365939755, balance_loss_lambda=0.0023491894151101617 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260309_041811_503_hparam_P1HGR_joint32_C24_A6_R0_serial_per_group.log |
| 3 | 0.087900 | P1HGR_joint32_C00_A0_R0_serial_per_group | arch_layout_id=0, num_layers=-1, stage_merge_mode=serial, group_router_mode=per_group, group_top_k=0, expert_top_k=1, expert_use_feature=False, macro_routing_scope=session, parallel_stage_gate_temperature=1.0, embedding_size=128, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.0010333786168533372, weight_decay=1.0060252066896978e-05, hidden_dropout_prob=0.10496438251349965, balance_loss_lambda=0.0037897390086198864 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr/hparam/P1HGR/ML1/FMoEHGR/20260309_041811_515_hparam_P1HGR_joint32_C00_A0_R0_serial_per_group.log |

