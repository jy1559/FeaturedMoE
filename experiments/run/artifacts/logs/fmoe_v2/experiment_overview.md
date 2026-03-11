# fmoe_v2 Experiment Overview

- generated_at_utc: 2026-03-10T01:14:39.563028+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 255
- included_runs: 198
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 57
- summarized_experiments: 20

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | P2DB_movielens1m_serial_dimbatch | hparam | 32 | 0 | 0.098200 | 0.0982/0.0982/0.0982 | P2DB_G5_C1_serial_L7_E128_R64_B3072 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260305_162951_587_hparam_P2DB_G5_C1_serial_L7_E128_R64_B3072.log |
| movielens1m | hparam_P1S | hparam | 56 | 16 | 0.097500 | 0.0975/0.0973/0.0972 | P1S_G3_C2_serial_L7 | d_expert_hidden, fmoe_stage_execution_mode, fmoe_v2_layout_id, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ML1/FMoEv2/20260305_104036_759_hparam_P1S_G3_C2_serial_L7.log |
| movielens1m | P2DB_movielens1m_parallel_dimbatch | hparam | 19 | 6 | 0.096900 | 0.0969/0.0966/0.0965 | P2DB_G0_C2_parallel_L13_E128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260306_130000_231_hparam_P2DB_G0_C2_parallel_L13_E128_R64_B4096.log |
| movielens1m | hparam_P1 | hparam | 3 | 0 | 0.024300 | 0.0243/0.0222/0.0148 | P1_log_check | balance_loss_lambda, fmoe_v2_layout_id, hidden_dropout_prob, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_084825_719_hparam_P1_log_check.log |
| movielens1m | hparam_P0 | hparam | 3 | 1 | 0.015100 | 0.0151/0.0151 | P0_report_check | balance_loss_lambda, hidden_dropout_prob, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P0/ML1/FMoEv2/20260305_082851_910_hparam_P0_report_check.log |
| retail_rocket | P2RRF_retail_rocket_main_serial_layout16 | hparam | 12 | 0 | 0.272000 | 0.2720/0.2713/0.2702 | P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096 | fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_191442_338_hparam_P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096.log |
| retail_rocket | P1_wide_shallow_retail_rocket | hparam | 32 | 0 | 0.269900 | 0.2699/0.2695/0.2690 | P1S_G1_C7_serial_L16 | fmoe_stage_execution_mode, fmoe_v2_layout_id, learning_rate, weight_decay, train_batch_size, eval_batch_size, d_feat_emb, d_expert_hidden, d_router_hidden | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_060500_257_hparam_P1S_G1_C7_serial_L16.log |
| retail_rocket | P2DB_retail_rocket_serial_dimbatch | hparam | 4 | 2 | 0.266100 | 0.2661/0.2635 | P2DB_G1_C1_serial_L7_E160_R96_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ReR/FMoEv2/20260308_101133_510_hparam_P2DB_G1_C1_serial_L7_E160_R96_B4096.log |
| retail_rocket | P3RRT_ROUTER_retail_rocket_l16_f24_router_teach | hparam | 12 | 0 | 0.264400 | 0.2644/0.2641/0.2638 | P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20 | fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_top_k, router_distill_enable, router_distill_lambda, router_distill_temperature, router_distill_until, fmoe_v2_feature_spec_aux_lambda, fmoe_v2_feature_spec_stages, learning_rate, weight_decay, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_205812_574_hparam_P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20.log |
| retail_rocket | P2RGI_retail_rocket_serial_layout16 | hparam | 4 | 0 | 0.262100 | 0.2621/0.2619/0.2612 | P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_232_hparam_P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RGI2_retail_rocket_serial_layout16 | hparam | 2 | 0 | 0.261800 | 0.2618/0.2615 | P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log |
| retail_rocket | P2RGI_retail_rocket_serial_layout15 | hparam | 2 | 0 | 0.261800 | 0.2618/0.2602 | P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_134840_010_hparam_P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RFI_retail_rocket_serial_layout16 | hparam | 5 | 2 | 0.261700 | 0.2617/0.2612/0.2597 | P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_267_hparam_P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096.log |
| retail_rocket | P2RGI_retail_rocket_serial_layout18 | hparam | 2 | 0 | 0.261300 | 0.2613/0.2609 | P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_135515_618_hparam_P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RGI2_retail_rocket_serial_layout18 | hparam | 2 | 0 | 0.261100 | 0.2611/0.2602 | P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_810_hparam_P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RGI2_retail_rocket_serial_layout15 | hparam | 2 | 0 | 0.261000 | 0.2610/0.2609 | P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_084809_085_hparam_P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096.log |
| retail_rocket | P1RGI2_retail_rocket_serial_layout7 | hparam | 1 | 0 | 0.260800 | 0.2608 | P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RGI2_retail_rocket_serial_layout5 | hparam | 1 | 0 | 0.260200 | 0.2602 | P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_085756_917_hparam_P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RFI_retail_rocket_serial_layout18 | hparam | 3 | 2 | 0.258500 | 0.2585 | P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_260_hparam_P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096.log |
| retail_rocket | P1RFI_retail_rocket_serial_layout15 | hparam | 1 | 1 | - | - | - | fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay | - |

## Experiment Notes

### movielens1m / P2DB_movielens1m_serial_dimbatch

- 실험 설명: P2DB: fixed serial/L7, combo(dim/router/feat/expert/batch) sweep + LR/WD(profile=0) search.
- 실행 규모: runs=32, oom=0, 기간=2026-03-05T16:29:51.631530+00:00 ~ 2026-03-06T07:03:49.374980+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay
- 최고 성능: MRR@20=0.098200 (P2DB_G5_C1_serial_L7_E128_R64_B3072, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=512, d_router_hidden=64, learning_rate=0.006058821934145586, weight_decay=3.5888774253991526e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260305_162951_587_hparam_P2DB_G5_C1_serial_L7_E128_R64_B3072.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260305_162954.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.098200 | P2DB_G5_C1_serial_L7_E128_R64_B3072 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=512, d_router_hidden=64, learning_rate=0.006058821934145586, weight_decay=3.5888774253991526e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260305_162951_587_hparam_P2DB_G5_C1_serial_L7_E128_R64_B3072.log |
| 2 | 0.098200 | P2DB_G7_C1_serial_L7_E160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=512, d_router_hidden=64, learning_rate=0.006058821934145586, weight_decay=3.5888774253991526e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260305_162951_585_hparam_P2DB_G7_C1_serial_L7_E160_R80_B4096.log |
| 3 | 0.098200 | P2DB_G4_C1_serial_L7_E192_R96_B8192 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=512, d_router_hidden=64, learning_rate=0.006058821934145586, weight_decay=3.5888774253991526e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260305_162951_582_hparam_P2DB_G4_C1_serial_L7_E192_R96_B8192.log |

### movielens1m / hparam_P1S

- 실험 설명: (설명 미기록)
- 실행 규모: runs=56, oom=16, 기간=2026-03-05T08:09:32.583365+00:00 ~ 2026-03-05T15:04:22.045511+00:00
- 비교 변수: d_expert_hidden, fmoe_stage_execution_mode, fmoe_v2_layout_id, learning_rate, weight_decay
- 최고 성능: MRR@20=0.097500 (P1S_G3_C2_serial_L7, FeaturedMoE_v2_serial)
- 최고 설정: d_expert_hidden=128, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, learning_rate=0.0011300776571623009, weight_decay=0.00035223255649505133
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ML1/FMoEv2/20260305_104036_759_hparam_P1S_G3_C2_serial_L7.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260305_104039.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.097500 | P1S_G3_C2_serial_L7 | d_expert_hidden=128, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, learning_rate=0.0011300776571623009, weight_decay=0.00035223255649505133 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ML1/FMoEv2/20260305_104036_759_hparam_P1S_G3_C2_serial_L7.log |
| 2 | 0.097300 | P1S_G6_C2_serial_L18 | d_expert_hidden=128, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, learning_rate=0.0003072684111142095, weight_decay=5.746709893339327e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ML1/FMoEv2/20260305_104646_080_hparam_P1S_G6_C2_serial_L18.log |
| 3 | 0.097200 | P1S_G5_C2_serial_L16 | d_expert_hidden=128, fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, learning_rate=0.0016295562556483023, weight_decay=0.0004253941029170473 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ML1/FMoEv2/20260305_111006_104_hparam_P1S_G5_C2_serial_L16.log |

### movielens1m / P2DB_movielens1m_parallel_dimbatch

- 실험 설명: P2DB: fixed parallel/L13, combo(dim/router/feat/expert/batch) sweep + LR/WD(profile=0) search.
- 실행 규모: runs=19, oom=6, 기간=2026-03-06T06:54:41.331536+00:00 ~ 2026-03-06T20:24:56.180500+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, train_batch_size, eval_batch_size, learning_rate, weight_decay
- 최고 성능: MRR@20=0.096900 (P2DB_G0_C2_parallel_L13_E128_R64_B4096, FeaturedMoE_v2_parallel)
- 최고 설정: fmoe_stage_execution_mode=parallel, fmoe_v2_layout_id=13, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, learning_rate=0.0022310605501721123, weight_decay=0.0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260306_130000_231_hparam_P2DB_G0_C2_parallel_L13_E128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260306_130003.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096900 | P2DB_G0_C2_parallel_L13_E128_R64_B4096 | fmoe_stage_execution_mode=parallel, fmoe_v2_layout_id=13, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, learning_rate=0.0022310605501721123, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260306_130000_231_hparam_P2DB_G0_C2_parallel_L13_E128_R64_B4096.log |
| 2 | 0.096600 | P2DB_G0_C3_parallel_L13_E128_R64_B4096 | fmoe_stage_execution_mode=parallel, fmoe_v2_layout_id=13, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, learning_rate=0.007040242360138351, weight_decay=5.872440022044886e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260306_170647_622_hparam_P2DB_G0_C3_parallel_L13_E128_R64_B4096.log |
| 3 | 0.096500 | P2DB_G2_C3_parallel_L13_E128_R64_B6144 | fmoe_stage_execution_mode=parallel, fmoe_v2_layout_id=13, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, learning_rate=0.00900130127760122, weight_decay=1.1731237050093806e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ML1/FMoEv2/20260306_152043_862_hparam_P2DB_G2_C3_parallel_L13_E128_R64_B6144.log |

### movielens1m / hparam_P1

- 실험 설명: (설명 미기록)
- 실행 규모: runs=3, oom=0, 기간=2026-03-05T08:46:48.991479+00:00 ~ 2026-03-05T09:00:35.633911+00:00
- 비교 변수: balance_loss_lambda, fmoe_v2_layout_id, hidden_dropout_prob, learning_rate
- 최고 성능: MRR@20=0.024300 (P1_log_check, FeaturedMoE_v2_serial)
- 최고 설정: balance_loss_lambda=0.01, fmoe_v2_layout_id=9, hidden_dropout_prob=0.1, learning_rate=0.00173875341250144
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_084825_719_hparam_P1_log_check.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260305_084828.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.024300 | P1_log_check | balance_loss_lambda=0.01, fmoe_v2_layout_id=9, hidden_dropout_prob=0.1, learning_rate=0.00173875341250144 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_084825_719_hparam_P1_log_check.log |
| 2 | 0.022200 | P1_wandb_fix_check | balance_loss_lambda=0.01, fmoe_v2_layout_id=4, hidden_dropout_prob=0.1, learning_rate=0.00173875341250144 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_085833_252_hparam_P1_wandb_fix_check.log |
| 3 | 0.014800 | P1_log_format_check | balance_loss_lambda=0.015799699084161345, fmoe_v2_layout_id=9, hidden_dropout_prob=0.18258899382089233, learning_rate=0.00471718136105401 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1/ML1/FMoEv2/20260305_084648_949_hparam_P1_log_format_check.log |

### movielens1m / hparam_P0

- 실험 설명: (설명 미기록)
- 실행 규모: runs=3, oom=1, 기간=2026-03-05T08:07:57.063976+00:00 ~ 2026-03-05T08:29:29.238563+00:00
- 비교 변수: balance_loss_lambda, hidden_dropout_prob, learning_rate, weight_decay
- 최고 성능: MRR@20=0.015100 (P0_report_check, FeaturedMoE_v2_serial)
- 최고 설정: balance_loss_lambda=0.015799699084161345, hidden_dropout_prob=0.18258899382089233, learning_rate=0.00471718136105401, weight_decay=0.0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P0/ML1/FMoEv2/20260305_082851_910_hparam_P0_report_check.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/movielens1m_FeaturedMoE_v2_20260305_082854.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.015100 | P0_report_check | balance_loss_lambda=0.015799699084161345, hidden_dropout_prob=0.18258899382089233, learning_rate=0.00471718136105401, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P0/ML1/FMoEv2/20260305_082851_910_hparam_P0_report_check.log |
| 2 | 0.015100 | P0_envcheck_smallbs | balance_loss_lambda=0.015799699084161345, hidden_dropout_prob=0.18258899382089233, learning_rate=0.00471718136105401, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P0/movielens1m/FeaturedMoE_v2_serial/movielens1m_FeaturedMoE_v2_serial_hparam_P0_envcheck_smallbs_gpu0_20260305_080824_889.log |

### retail_rocket / P2RRF_retail_rocket_main_serial_layout16

- 실험 설명: P2RRF: RR-focused serial P2 from strong P1 layouts; 80-combo sweep with anchor-near, high-capacity, and outlier dim/router/batch mixes.
- 실행 규모: runs=12, oom=0, 기간=2026-03-08T16:35:25.407128+00:00 ~ 2026-03-09T01:16:57.937301+00:00
- 비교 변수: fmoe_v2_layout_id, fmoe_stage_execution_mode, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.272000 (P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.09209936386789819, balance_loss_lambda=0.011288535014800486, learning_rate=0.0003084807409494183, weight_decay=9.016522762022532e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_191442_338_hparam_P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rrf_l16_g4_c04_e128_f24_h160_r64_b4096_20260308_191445_359236_pid382700.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.272000 | P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096 | fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.09209936386789819, balance_loss_lambda=0.011288535014800486, learning_rate=0.0003084807409494183, weight_decay=9.016522762022532e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_191442_338_hparam_P2RRF_L16_G4_C04_E128_F24_H160_R64_B4096.log |
| 2 | 0.271300 | P2RRF_L16_G7_C03_E128_F16_H192_R80_B4096 | fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial, embedding_size=128, d_feat_emb=16, d_expert_hidden=192, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.11511741670831946, balance_loss_lambda=0.017354687244753957, learning_rate=0.000497310514030353, weight_decay=7.731741938764293e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_163525_365_hparam_P2RRF_L16_G7_C03_E128_F16_H192_R80_B4096.log |
| 3 | 0.270200 | P2RRF_L16_G7_C07_E160_F24_H192_R96_B4096 | fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial, embedding_size=160, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, expert_scale=3, hidden_dropout_prob=0.10882661705898201, balance_loss_lambda=0.017576487724467623, learning_rate=0.00034695767229261145, weight_decay=7.202371667079931e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RRF/ReR/FMoEv2/20260308_190949_048_hparam_P2RRF_L16_G7_C07_E160_F24_H192_R96_B4096.log |

### retail_rocket / P1_wide_shallow_retail_rocket

- 실험 설명: Wide-shallow screen over serial/parallel layouts (fixed dims), tune LR/WD with small eval budget.
- 실행 규모: runs=32, oom=0, 기간=2026-03-07T16:50:18.777747+00:00 ~ 2026-03-08T09:39:01.922212+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, learning_rate, weight_decay, train_batch_size, eval_batch_size, d_feat_emb, d_expert_hidden, d_router_hidden
- 최고 성능: MRR@20=0.269900 (P1S_G1_C7_serial_L16, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, learning_rate=0.0007123236720225437, weight_decay=8.774513944726588e-05, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_060500_257_hparam_P1S_G1_C7_serial_L16.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1s_g1_c7_serial_l16_20260308_060503_131862_pid257896.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.269900 | P1S_G1_C7_serial_L16 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, learning_rate=0.0007123236720225437, weight_decay=8.774513944726588e-05, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_060500_257_hparam_P1S_G1_C7_serial_L16.log |
| 2 | 0.269500 | P1S_G0_C8_serial_L7 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, learning_rate=0.0005652775638278702, weight_decay=5.209562216170883e-05, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_053004_989_hparam_P1S_G0_C8_serial_L7.log |
| 3 | 0.269000 | P1S_G0_C6_serial_L5 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=5, learning_rate=0.0002616193804600348, weight_decay=6.065204228547568e-05, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1S/ReR/FMoEv2/20260308_010717_755_hparam_P1S_G0_C6_serial_L5.log |

### retail_rocket / P2DB_retail_rocket_serial_dimbatch

- 실험 설명: P2DB: fixed serial/L7, combo(complexity: emb/feat/expert/router/scale/batch) sweep + LR/WD/dropout/balance(profile=2) search.
- 실행 규모: runs=4, oom=2, 기간=2026-03-08T10:07:39.050641+00:00 ~ 2026-03-08T13:50:47.699184+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.266100 (P2DB_G1_C1_serial_L7_E160_R96_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=160, d_feat_emb=16, d_expert_hidden=256, d_router_hidden=96, expert_scale=3, hidden_dropout_prob=0.1334922452962251, balance_loss_lambda=0.0030614999108664674, learning_rate=0.0003590322477719474, weight_decay=3.183536775457959e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ReR/FMoEv2/20260308_101133_510_hparam_P2DB_G1_C1_serial_L7_E160_R96_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2db_g1_c1_serial_l7_e160_r96_b4096_20260308_101136_299883_pid302220.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.266100 | P2DB_G1_C1_serial_L7_E160_R96_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=160, d_feat_emb=16, d_expert_hidden=256, d_router_hidden=96, expert_scale=3, hidden_dropout_prob=0.1334922452962251, balance_loss_lambda=0.0030614999108664674, learning_rate=0.0003590322477719474, weight_decay=3.183536775457959e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ReR/FMoEv2/20260308_101133_510_hparam_P2DB_G1_C1_serial_L7_E160_R96_B4096.log |
| 2 | 0.263500 | P2DB_G0_C1_serial_L7_E128_R64_B3072 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.13442349792617975, balance_loss_lambda=0.004755415323699389, learning_rate=0.0004202720962317471, weight_decay=1.5985677969733742e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2DB/ReR/FMoEv2/20260308_101133_512_hparam_P2DB_G0_C1_serial_L7_E128_R64_B3072.log |

### retail_rocket / P3RRT_ROUTER_retail_rocket_l16_f24_router_teach

- 실험 설명: P3RRT_ROUTER: RR router-teach P3 around l16_f24. Fixed seed L16/128/24/160/64/bs4096; compare top-k/distill/spec profiles under long-horizon tuning.
- 실행 규모: runs=12, oom=0, 기간=2026-03-09T15:32:02.339947+00:00 ~ 2026-03-09T23:58:33.985059+00:00
- 비교 변수: fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_top_k, router_distill_enable, router_distill_lambda, router_distill_temperature, router_distill_until, fmoe_v2_feature_spec_aux_lambda, fmoe_v2_feature_spec_stages, learning_rate, weight_decay, balance_loss_lambda
- 최고 성능: MRR@20=0.264400 (P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_top_k=2, router_distill_enable=True, router_distill_lambda=0.005, router_distill_temperature=1.5, router_distill_until=0.2, fmoe_v2_feature_spec_aux_lambda=0.0003, fmoe_v2_feature_spec_stages=['mid', 'micro'], learning_rate=0.0004213968759430877, weight_decay=5.588371316537863e-05, balance_loss_lambda=0.0017447732310841727
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_205812_574_hparam_P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p3rrt_router_g7_c11_l16f24_l16_e128_f24_h160_r64_b4096_k2_d5_smm_u20_20260309_205815_537261_pid528926.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.264400 | P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20 | fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_top_k=2, router_distill_enable=True, router_distill_lambda=0.005, router_distill_temperature=1.5, router_distill_until=0.2, fmoe_v2_feature_spec_aux_lambda=0.0003, fmoe_v2_feature_spec_stages=['mid', 'micro'], learning_rate=0.0004213968759430877, weight_decay=5.588371316537863e-05, balance_loss_lambda=0.0017447732310841727 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_205812_574_hparam_P3RRT_ROUTER_G7_C11_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMM_U20.log |
| 2 | 0.264100 | P3RRT_ROUTER_G6_C10_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMID_U20 | fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_top_k=2, router_distill_enable=True, router_distill_lambda=0.005, router_distill_temperature=1.5, router_distill_until=0.2, fmoe_v2_feature_spec_aux_lambda=0.0003, fmoe_v2_feature_spec_stages=['mid'], learning_rate=0.0003922832083749457, weight_decay=5.4852816254981286e-05, balance_loss_lambda=0.0012299865730605137 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_210319_092_hparam_P3RRT_ROUTER_G6_C10_L16F24_L16_E128_F24_H160_R64_B4096_K2_D5_SMID_U20.log |
| 3 | 0.263800 | P3RRT_ROUTER_G5_C09_L16F24_L16_E128_F24_H160_R64_B4096_K2_D2_SMID_U20 | fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_top_k=2, router_distill_enable=True, router_distill_lambda=0.002, router_distill_temperature=1.5, router_distill_until=0.2, fmoe_v2_feature_spec_aux_lambda=0.0003, fmoe_v2_feature_spec_stages=['mid'], learning_rate=0.000279232591725497, weight_decay=6.699710485810276e-05, balance_loss_lambda=0.0015164596919431154 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P3RRT/ReR/FMoEv2/20260309_210843_462_hparam_P3RRT_ROUTER_G5_C09_L16F24_L16_E128_F24_H160_R64_B4096_K2_D2_SMID_U20.log |

### retail_rocket / P2RGI_retail_rocket_serial_layout16

- 실험 설명: P2RGI: RR factorized-router P2. Keep layout fixed to top P1 candidates, vary dim/capacity, and add a small expert_top_k probe on selected seeds; center lr/wd lower when dim grows or batch shrinks.
- 실행 규모: runs=4, oom=0, 기간=2026-03-09T12:33:17.263124+00:00 ~ 2026-03-09T13:55:15.166917+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k
- 최고 성능: MRR@20=0.262100 (P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.0005443393221838013, weight_decay=5.181566984469375e-05, balance_loss_lambda=0.0028508277570718317, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_232_hparam_P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g4_c00_serial_l16_e128_f16_h128_r64_b4096_20260309_123320_144536_pid506475.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.262100 | P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.0005443393221838013, weight_decay=5.181566984469375e-05, balance_loss_lambda=0.0028508277570718317, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_232_hparam_P2RGI_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log |
| 2 | 0.261900 | P2RGI_G6_C02_serial_L16_E128_F24_H160_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=24, d_expert_hidden=160, d_router_hidden=64, expert_scale=3, learning_rate=0.000478423100148067, weight_decay=6.889024979066358e-05, balance_loss_lambda=0.0016929197615697061, fmoe_v2_feature_spec_aux_lambda=0.0003, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_233_hparam_P2RGI_G6_C02_serial_L16_E128_F24_H160_R64_B4096.log |
| 3 | 0.261200 | P2RGI_G5_C01_serial_L16_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, learning_rate=0.00031536977626543135, weight_decay=4.3125308558536196e-05, balance_loss_lambda=0.0013183797299669044, fmoe_v2_feature_spec_aux_lambda=0.0003, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_123317_233_hparam_P2RGI_G5_C01_serial_L16_E160_F16_H160_R80_B4096.log |

### retail_rocket / P1RGI2_retail_rocket_serial_layout16

- 실험 설명: P1RGI2: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2.
- 실행 규모: runs=2, oom=0, 기간=2026-03-09T07:00:25.855659+00:00 ~ 2026-03-09T11:21:41.040398+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.261800 (P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0012430932214344264, learning_rate=0.0005350956172544636, weight_decay=7.909593669567587e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g4_c00_serial_l16_e128_f16_h128_r64_b4096_20260309_070028_753184_pid473259.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261800 | P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0012430932214344264, learning_rate=0.0005350956172544636, weight_decay=7.909593669567587e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log |
| 2 | 0.261500 | P1RGI2_G5_C05_serial_L16_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0022255809279865643, learning_rate=0.0004953776784071795, weight_decay=4.447800539333446e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_091627_098_hparam_P1RGI2_G5_C05_serial_L16_E160_F16_H160_R80_B4096.log |

### retail_rocket / P2RGI_retail_rocket_serial_layout15

- 실험 설명: P2RGI: RR factorized-router P2. Keep layout fixed to top P1 candidates, vary dim/capacity, and add a small expert_top_k probe on selected seeds; center lr/wd lower when dim grows or batch shrinks.
- 실행 규모: runs=2, oom=0, 기간=2026-03-09T13:44:32.469213+00:00 ~ 2026-03-09T15:03:20.583427+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k
- 최고 성능: MRR@20=0.261800 (P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.0005189833498183475, weight_decay=7.696789152046396e-05, balance_loss_lambda=0.001949495498472354, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_134840_010_hparam_P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g4_c04_serial_l15_e128_f16_h128_r64_b4096_20260309_134842_932740_pid512586.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261800 | P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.0005189833498183475, weight_decay=7.696789152046396e-05, balance_loss_lambda=0.001949495498472354, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_134840_010_hparam_P2RGI_G4_C04_serial_L15_E128_F16_H128_R64_B4096.log |
| 2 | 0.260200 | P2RGI_G5_C05_serial_L15_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, learning_rate=0.0005007267403481261, weight_decay=8.034119732904351e-05, balance_loss_lambda=0.0015820658055280654, fmoe_v2_feature_spec_aux_lambda=0.0003, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_134432_425_hparam_P2RGI_G5_C05_serial_L15_E160_F16_H160_R80_B4096.log |

### retail_rocket / P1RFI_retail_rocket_serial_layout16

- 실험 설명: P1RFI: RR factorized-router blocked-joint P1. 12 shared anchors for layout comparison + 8 top-layout dim probes + 4 parallel sentinels; map LR band before narrow P2.
- 실행 규모: runs=5, oom=2, 기간=2026-03-09T04:19:51.605852+00:00 ~ 2026-03-09T06:23:58.482301+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.261700 (P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.11789353036322173, balance_loss_lambda=0.0018950869262041971, learning_rate=0.00042396749777455956, weight_decay=5.649612413789497e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_267_hparam_P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rfi_g5_c01_serial_l16_e160_f16_h160_r80_b4096_20260309_043253_951583_pid446004.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261700 | P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.11789353036322173, balance_loss_lambda=0.0018950869262041971, learning_rate=0.00042396749777455956, weight_decay=5.649612413789497e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_267_hparam_P1RFI_G5_C01_serial_L16_E160_F16_H160_R80_B4096.log |
| 2 | 0.261200 | P1RFI_G4_C00_serial_L16_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.08426811549643026, balance_loss_lambda=0.00137506036669114, learning_rate=0.0004950625054867597, weight_decay=5.749162629120568e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_265_hparam_P1RFI_G4_C00_serial_L16_E128_F16_H128_R64_B4096.log |
| 3 | 0.259700 | P1RFI_G6_C02_serial_L16_E192_F24_H192_R96_B3072 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=16, embedding_size=192, d_feat_emb=24, d_expert_hidden=192, d_router_hidden=96, expert_scale=3, hidden_dropout_prob=0.08587057630508826, balance_loss_lambda=0.0028555483747674955, learning_rate=0.00031947838796774007, weight_decay=9.587253483248331e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_264_hparam_P1RFI_G6_C02_serial_L16_E192_F24_H192_R96_B3072.log |

### retail_rocket / P2RGI_retail_rocket_serial_layout18

- 실험 설명: P2RGI: RR factorized-router P2. Keep layout fixed to top P1 candidates, vary dim/capacity, and add a small expert_top_k probe on selected seeds; center lr/wd lower when dim grows or batch shrinks.
- 실행 규모: runs=2, oom=0, 기간=2026-03-09T13:36:40.389428+00:00 ~ 2026-03-09T15:26:40.820765+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, learning_rate, weight_decay, balance_loss_lambda, fmoe_v2_feature_spec_aux_lambda, expert_top_k
- 최고 성능: MRR@20=0.261300 (P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.00038612558548164723, weight_decay=4.412297649816732e-05, balance_loss_lambda=0.0017681504654075925, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_135515_618_hparam_P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p2rgi_g6_c06_serial_l18_e128_f16_h128_r64_b4096_20260309_135518_743216_pid513140.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261300 | P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, learning_rate=0.00038612558548164723, weight_decay=4.412297649816732e-05, balance_loss_lambda=0.0017681504654075925, fmoe_v2_feature_spec_aux_lambda=0.0007, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_135515_618_hparam_P2RGI_G6_C06_serial_L18_E128_F16_H128_R64_B4096.log |
| 2 | 0.260900 | P2RGI_G7_C07_serial_L18_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, learning_rate=0.0003160564837711707, weight_decay=6.621743205386752e-05, balance_loss_lambda=0.0018177168914719603, fmoe_v2_feature_spec_aux_lambda=0.0003, expert_top_k=1 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P2RGI/ReR/FMoEv2/20260309_133640_342_hparam_P2RGI_G7_C07_serial_L18_E160_F16_H160_R80_B4096.log |

### retail_rocket / P1RGI2_retail_rocket_serial_layout18

- 실험 설명: P1RGI2: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2.
- 실행 규모: runs=2, oom=0, 기간=2026-03-09T07:00:25.861617+00:00 ~ 2026-03-09T11:24:45.250108+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.261100 (P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0017848656453575552, learning_rate=0.0004660339647242112, weight_decay=6.10060366623874e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_810_hparam_P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g5_c01_serial_l18_e128_f16_h128_r64_b4096_20260309_070028_648073_pid473261.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261100 | P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0017848656453575552, learning_rate=0.0004660339647242112, weight_decay=6.10060366623874e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_810_hparam_P1RGI2_G5_C01_serial_L18_E128_F16_H128_R64_B4096.log |
| 2 | 0.260200 | P1RGI2_G6_C06_serial_L18_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0029630743394517822, learning_rate=0.00033098317388026216, weight_decay=4.800500286874876e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_085401_545_hparam_P1RGI2_G6_C06_serial_L18_E160_F16_H160_R80_B4096.log |

### retail_rocket / P1RGI2_retail_rocket_serial_layout15

- 실험 설명: P1RGI2: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2.
- 실행 규모: runs=2, oom=0, 기간=2026-03-09T07:00:25.857812+00:00 ~ 2026-03-09T10:55:06.262447+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.261000 (P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0014032333977327196, learning_rate=0.0003800691765032546, weight_decay=6.487565746676257e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_084809_085_hparam_P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g7_c07_serial_l15_e160_f16_h160_r80_b4096_20260309_084811_867847_pid485983.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.261000 | P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=160, d_feat_emb=16, d_expert_hidden=160, d_router_hidden=80, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0014032333977327196, learning_rate=0.0003800691765032546, weight_decay=6.487565746676257e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_084809_085_hparam_P1RGI2_G7_C07_serial_L15_E160_F16_H160_R80_B4096.log |
| 2 | 0.260900 | P1RGI2_G6_C02_serial_L15_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=15, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0025950626142359234, learning_rate=0.000777516597560793, weight_decay=9.992622991830544e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_809_hparam_P1RGI2_G6_C02_serial_L15_E128_F16_H128_R64_B4096.log |

### retail_rocket / P1RGI2_retail_rocket_serial_layout7

- 실험 설명: P1RGI2: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2.
- 실행 규모: runs=1, oom=0, 기간=2026-03-09T07:00:25.854878+00:00 ~ 2026-03-09T08:48:08.662191+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.260800 (P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0011762949736063052, learning_rate=0.0006889953178145216, weight_decay=4.6595034703084655e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g7_c03_serial_l7_e128_f16_h128_r64_b4096_20260309_070028_748587_pid473258.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.260800 | P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=7, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0011762949736063052, learning_rate=0.0006889953178145216, weight_decay=4.6595034703084655e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_070025_807_hparam_P1RGI2_G7_C03_serial_L7_E128_F16_H128_R64_B4096.log |

### retail_rocket / P1RGI2_retail_rocket_serial_layout5

- 실험 설명: P1RGI2: RR factorized-router layout-first P1. Compare 5 strong serial layouts under 2 shared core dims, then add 2 light dim probes only on L16/L18 to check weak layout-dim interaction before P2.
- 실행 규모: runs=1, oom=0, 기간=2026-03-09T08:57:56.966901+00:00 ~ 2026-03-09T10:55:05.213805+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.260200 (P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=5, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0019425475270909442, learning_rate=0.0006754634383182488, weight_decay=3.401152721759458e-05
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_085756_917_hparam_P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rgi2_g4_c04_serial_l5_e128_f16_h128_r64_b4096_20260309_085759_749729_pid487410.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.260200 | P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=5, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.1, balance_loss_lambda=0.0019425475270909442, learning_rate=0.0006754634383182488, weight_decay=3.401152721759458e-05 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RGI2/ReR/FMoEv2/20260309_085756_917_hparam_P1RGI2_G4_C04_serial_L5_E128_F16_H128_R64_B4096.log |

### retail_rocket / P1RFI_retail_rocket_serial_layout18

- 실험 설명: P1RFI: RR factorized-router blocked-joint P1. 12 shared anchors for layout comparison + 8 top-layout dim probes + 4 parallel sentinels; map LR band before narrow P2.
- 실행 규모: runs=3, oom=2, 기간=2026-03-09T04:19:51.609783+00:00 ~ 2026-03-09T06:22:25.050601+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: MRR@20=0.258500 (P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096, FeaturedMoE_v2_serial)
- 최고 설정: fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.0873583286095242, balance_loss_lambda=0.002246199362717922, learning_rate=0.0006256632712565308, weight_decay=0.0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_260_hparam_P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v2/retail_rocket_FeaturedMoE_v2_p1rfi_g7_c03_serial_l18_e128_f16_h128_r64_b4096_20260309_043254_022613_pid446002.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.258500 | P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096 | fmoe_stage_execution_mode=serial, fmoe_v2_layout_id=18, embedding_size=128, d_feat_emb=16, d_expert_hidden=128, d_router_hidden=64, expert_scale=3, hidden_dropout_prob=0.0873583286095242, balance_loss_lambda=0.002246199362717922, learning_rate=0.0006256632712565308, weight_decay=0.0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v2/hparam/P1RFI/ReR/FMoEv2/20260309_043251_260_hparam_P1RFI_G7_C03_serial_L18_E128_F16_H128_R64_B4096.log |

### retail_rocket / P1RFI_retail_rocket_serial_layout15

- 실험 설명: P1RFI: RR factorized-router blocked-joint P1. 12 shared anchors for layout comparison + 8 top-layout dim probes + 4 parallel sentinels; map LR band before narrow P2.
- 실행 규모: runs=1, oom=1, 기간=2026-03-09T04:30:53.632852+00:00 ~ 2026-03-09T04:32:31.782215+00:00
- 비교 변수: fmoe_stage_execution_mode, fmoe_v2_layout_id, embedding_size, d_feat_emb, d_expert_hidden, d_router_hidden, expert_scale, train_batch_size, eval_batch_size, hidden_dropout_prob, balance_loss_lambda, learning_rate, weight_decay
- 최고 성능: -
- 최고 설정: -
- 최고 로그: -
- 최고 결과 JSON: -
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| - | - | - | - | - |

