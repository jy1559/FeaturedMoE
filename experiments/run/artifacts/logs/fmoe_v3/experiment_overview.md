# fmoe_v3 Experiment Overview

- generated_at_utc: 2026-03-10T16:11:12.345245+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 18
- included_runs: 18
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 5

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | fmoe_v3_phase_b_router_structure | hparam | 6 | 0 | 0.097300 | 0.0973/0.0954/0.0953 | P2ROUTER_C00_LEGACY | router_design, router_use_feature, router_group_bias_scale, router_clone_residual_scale, moe_top_k, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_052919_286_hparam_P2ROUTER_C00_LEGACY.log |
| movielens1m | fmoe_v3_phase_g_ml1_expert_scale_lr | hparam | 2 | 0 | 0.096900 | 0.0969/0.0913 | P2ESLR_S3 | router_design, expert_scale, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda, fmoe_v2_layout_id, fmoe_stage_execution_mode | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ESLR/ML1/FMoEv3/20260310_093607_463_hparam_P2ESLR_S3.log |
| movielens1m | fmoe_v3_phase_b_router_narrow | hparam | 2 | 0 | 0.096600 | 0.0966/0.0963 | P3ROUTER_C00_LEGACY_GCLONE | router_design, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, learning_rate, weight_decay, moe_top_k | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3ROUTER/ML1/FMoEv3/20260310_142435_977_hparam_P3ROUTER_C00_LEGACY_GCLONE.log |
| movielens1m | fmoe_v3_phase_c_distill_modes | hparam | 6 | 0 | 0.096600 | 0.0966/0.0963/0.0958 | P2DISTILL_base_gclone_cmp | router_design, router_distill_enable, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, router_distill_until, learning_rate, weight_decay, moe_top_k | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_071758_362_hparam_P2DISTILL_base_gclone_cmp.log |
| movielens1m | fmoe_v3_phase_c_distill_narrow | hparam | 2 | 0 | 0.096400 | 0.0964/0.0956 | P3DISTILL_C00_LEGACY_PLAIN | router_design, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, learning_rate, weight_decay, moe_top_k, router_impl_by_stage | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3DISTILL/ML1/FMoEv3/20260310_142435_966_hparam_P3DISTILL_C00_LEGACY_PLAIN.log |

## Experiment Notes

### movielens1m / fmoe_v3_phase_b_router_structure

- 실험 설명: ML1 flat-router structure screen over hidden/global/group-aware router variants.
- 실행 규모: runs=6, oom=0, 기간=2026-03-10T05:29:19.330272+00:00 ~ 2026-03-10T11:41:59.628174+00:00
- 비교 변수: router_design, router_use_feature, router_group_bias_scale, router_clone_residual_scale, moe_top_k, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda
- 최고 성능: MRR@20=0.097300 (P2ROUTER_C00_LEGACY, FeaturedMoE_v3_serial)
- 최고 설정: router_design=flat_legacy, router_group_bias_scale=0.5, router_clone_residual_scale=0.5, moe_top_k=0, learning_rate=0.003457647151765668, weight_decay=7.057864066167489e-05, hidden_dropout_prob=0.1, balance_loss_lambda=0.005
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_052919_286_hparam_P2ROUTER_C00_LEGACY.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2router_c00_legacy_20260310_052922_110009_pid584683.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.097300 | P2ROUTER_C00_LEGACY | router_design=flat_legacy, router_group_bias_scale=0.5, router_clone_residual_scale=0.5, moe_top_k=0, learning_rate=0.003457647151765668, weight_decay=7.057864066167489e-05, hidden_dropout_prob=0.1, balance_loss_lambda=0.005 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_052919_286_hparam_P2ROUTER_C00_LEGACY.log |
| 2 | 0.095400 | P2ROUTER_C03_HGCLONE | router_design=flat_hidden_group_clone12, router_group_bias_scale=0.5, router_clone_residual_scale=0.5, moe_top_k=0, learning_rate=0.0012, weight_decay=0.0001, hidden_dropout_prob=0.1, balance_loss_lambda=0.005 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_072437_930_hparam_P2ROUTER_C03_HGCLONE.log |
| 3 | 0.095300 | P2ROUTER_C05_GCOMBO | router_design=flat_group_clone_combo, router_group_bias_scale=0.5, router_clone_residual_scale=0.5, moe_top_k=0, learning_rate=0.0006, weight_decay=0.0001, hidden_dropout_prob=0.1, balance_loss_lambda=0.005 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ROUTER/ML1/FMoEv3/20260310_091531_937_hparam_P2ROUTER_C05_GCOMBO.log |

### movielens1m / fmoe_v3_phase_g_ml1_expert_scale_lr

- 실험 설명: ML1 legacy flat-router anchor with narrow LR search while varying expert_scale only.
- 실행 규모: runs=2, oom=0, 기간=2026-03-10T09:36:07.514711+00:00 ~ 2026-03-10T11:18:06.279537+00:00
- 비교 변수: router_design, expert_scale, learning_rate, weight_decay, hidden_dropout_prob, balance_loss_lambda, fmoe_v2_layout_id, fmoe_stage_execution_mode
- 최고 성능: MRR@20=0.096900 (P2ESLR_S3, FeaturedMoE_v3_serial)
- 최고 설정: router_design=flat_legacy, expert_scale=3, learning_rate=0.004, weight_decay=7.057864066167489e-05, hidden_dropout_prob=0.1, balance_loss_lambda=0.005, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ESLR/ML1/FMoEv3/20260310_093607_463_hparam_P2ESLR_S3.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2eslr_s3_20260310_093610_784853_pid638153.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096900 | P2ESLR_S3 | router_design=flat_legacy, expert_scale=3, learning_rate=0.004, weight_decay=7.057864066167489e-05, hidden_dropout_prob=0.1, balance_loss_lambda=0.005, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ESLR/ML1/FMoEv3/20260310_093607_463_hparam_P2ESLR_S3.log |
| 2 | 0.091300 | P2ESLR_S1 | router_design=flat_legacy, expert_scale=1, learning_rate=0.0032, weight_decay=7.057864066167489e-05, hidden_dropout_prob=0.1, balance_loss_lambda=0.005, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2ESLR/ML1/FMoEv3/20260310_093607_467_hparam_P2ESLR_S1.log |

### movielens1m / fmoe_v3_phase_b_router_narrow

- 실험 설명: ML1 narrow router screen with group_plus_clone distill fixed; drop weak router arms.
- 실행 규모: runs=2, oom=0, 기간=2026-03-10T14:24:36.033262+00:00 ~ 2026-03-10T16:11:12.137396+00:00
- 비교 변수: router_design, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, learning_rate, weight_decay, moe_top_k
- 최고 성능: MRR@20=0.096600 (P3ROUTER_C00_LEGACY_GCLONE, FeaturedMoE_v3_serial)
- 최고 설정: router_design=flat_legacy, router_distill_mode=group_plus_clone, router_distill_lambda_group=0.001, router_distill_lambda_clone=0.0025, learning_rate=0.0038, weight_decay=5e-05, moe_top_k=0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3ROUTER/ML1/FMoEv3/20260310_142435_977_hparam_P3ROUTER_C00_LEGACY_GCLONE.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p3router_c00_legacy_gclone_20260310_142440_208664_pid3395.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096600 | P3ROUTER_C00_LEGACY_GCLONE | router_design=flat_legacy, router_distill_mode=group_plus_clone, router_distill_lambda_group=0.001, router_distill_lambda_clone=0.0025, learning_rate=0.0038, weight_decay=5e-05, moe_top_k=0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3ROUTER/ML1/FMoEv3/20260310_142435_977_hparam_P3ROUTER_C00_LEGACY_GCLONE.log |
| 2 | 0.096300 | P3ROUTER_C01_HGCLONE_GCLONE | router_design=flat_hidden_group_clone12, router_distill_mode=group_plus_clone, router_distill_lambda_group=0.001, router_distill_lambda_clone=0.0025, learning_rate=0.0004, weight_decay=5e-05, moe_top_k=0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3ROUTER/ML1/FMoEv3/20260310_142435_967_hparam_P3ROUTER_C01_HGCLONE_GCLONE.log |

### movielens1m / fmoe_v3_phase_c_distill_modes

- 실험 설명: ML1 distill-mode screen for winning flat-router candidates.
- 실행 규모: runs=6, oom=0, 기간=2026-03-10T05:29:58.779695+00:00 ~ 2026-03-10T11:06:51.985651+00:00
- 비교 변수: router_design, router_distill_enable, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, router_distill_until, learning_rate, weight_decay, moe_top_k
- 최고 성능: MRR@20=0.096600 (P2DISTILL_base_gclone_cmp, FeaturedMoE_v3_serial)
- 최고 설정: router_design=flat_hidden_group_clone12, router_distill_enable=True, router_distill_mode=group_plus_clone, router_distill_lambda_group=0.001, router_distill_lambda_clone=0.0025, router_distill_until=0.2, learning_rate=0.0006, weight_decay=0.0001, moe_top_k=0
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_071758_362_hparam_P2DISTILL_base_gclone_cmp.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p2distill_base_gclone_cmp_20260310_071801_377812_pid606190.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096600 | P2DISTILL_base_gclone_cmp | router_design=flat_hidden_group_clone12, router_distill_enable=True, router_distill_mode=group_plus_clone, router_distill_lambda_group=0.001, router_distill_lambda_clone=0.0025, router_distill_until=0.2, learning_rate=0.0006, weight_decay=0.0001, moe_top_k=0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_071758_362_hparam_P2DISTILL_base_gclone_cmp.log |
| 2 | 0.096300 | P2DISTILL_base_clone_cmp | router_design=flat_hidden_group_clone12, router_distill_enable=True, router_distill_mode=clone_only, router_distill_lambda_group=0.0, router_distill_lambda_clone=0.0025, router_distill_until=0.2, learning_rate=0.0008, weight_decay=0.0001, moe_top_k=0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_070250_655_hparam_P2DISTILL_base_clone_cmp.log |
| 3 | 0.095800 | P2DISTILL_legacy_hybrid_cmp | router_design=flat_legacy, router_distill_enable=False, router_distill_mode=none, router_distill_lambda_group=0.0, router_distill_lambda_clone=0.0, router_distill_until=0.2, learning_rate=0.0012, weight_decay=0.0001, moe_top_k=0 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P2DISTILL/ML1/FMoEv3/20260310_091045_167_hparam_P2DISTILL_legacy_hybrid_cmp.log |

### movielens1m / fmoe_v3_phase_c_distill_narrow

- 실험 설명: ML1 narrow distill screen with flat_legacy router fixed; drop weak distill arms.
- 실행 규모: runs=2, oom=0, 기간=2026-03-10T14:24:36.033366+00:00 ~ 2026-03-10T16:04:36.311933+00:00
- 비교 변수: router_design, router_distill_mode, router_distill_lambda_group, router_distill_lambda_clone, learning_rate, weight_decay, moe_top_k, router_impl_by_stage
- 최고 성능: MRR@20=0.096400 (P3DISTILL_C00_LEGACY_PLAIN, FeaturedMoE_v3_serial)
- 최고 설정: router_design=flat_legacy, router_distill_mode=none, router_distill_lambda_group=0.0, router_distill_lambda_clone=0.0, learning_rate=0.0032, weight_decay=5e-05, moe_top_k=0, router_impl_by_stage={}
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3DISTILL/ML1/FMoEv3/20260310_142435_966_hparam_P3DISTILL_C00_LEGACY_PLAIN.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v3/movielens1m_FeaturedMoE_v3_p3distill_c00_legacy_plain_20260310_142440_267755_pid3396.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096400 | P3DISTILL_C00_LEGACY_PLAIN | router_design=flat_legacy, router_distill_mode=none, router_distill_lambda_group=0.0, router_distill_lambda_clone=0.0, learning_rate=0.0032, weight_decay=5e-05, moe_top_k=0, router_impl_by_stage={} | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3DISTILL/ML1/FMoEv3/20260310_142435_966_hparam_P3DISTILL_C00_LEGACY_PLAIN.log |
| 2 | 0.095600 | P3DISTILL_C01_LEGACY_CLONE | router_design=flat_legacy, router_distill_mode=clone_only, router_distill_lambda_group=0.0, router_distill_lambda_clone=0.0025, learning_rate=0.0026, weight_decay=5e-05, moe_top_k=0, router_impl_by_stage={} | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v3/hparam/P3DISTILL/ML1/FMoEv3/20260310_142435_977_hparam_P3DISTILL_C01_LEGACY_CLONE.log |

