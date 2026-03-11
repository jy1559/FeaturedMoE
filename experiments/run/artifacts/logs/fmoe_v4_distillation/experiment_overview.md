# fmoe_v4_distillation Experiment Overview

- generated_at_utc: 2026-03-11T05:32:47.136246+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 40
- included_runs: 40
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 2

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | fmoe_v4d_full_distill32 | hparam | 24 | 0 | 0.097500 | 0.0975/0.0966/0.0963 | PFULLV4D_C01_LEGACY_HYBRID | teacher_design, teacher_delivery, teacher_stage_mask, teacher_kl_lambda, teacher_bias_scale, teacher_until, router_impl_by_stage, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/014_C01_LEGACY_HYBRID.log |
| movielens1m | fmoe_v4d_p2_compare16 | hparam | 16 | 0 | 0.096300 | 0.0963/0.0959/0.0952 | P2CMP16_C08_RULE_HYBRID_SOFT | fmoe_v2_layout_id, fmoe_stage_execution_mode, router_impl, router_impl_by_stage, rule_router.variant, rule_router.n_bins, teacher_design, teacher_delivery, teacher_stage_mask, teacher_kl_lambda, teacher_bias_scale, teacher_until, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/P2CMP16/ML1/FMoEv4D/128_C08_RULE_HYBRID_SOFT.log |

## Experiment Notes

### movielens1m / fmoe_v4d_full_distill32

- 실험 설명: v4_distillation weighted 32-combo distillation sweep emphasizing group_comp teachers, fused bias, and mid/micro-only delivery with larger ML1 batch.
- 실행 규모: runs=24, oom=0, 기간=2026-03-10T16:42:22.056100+00:00 ~ 2026-03-11T01:29:00.659651+00:00
- 비교 변수: teacher_design, teacher_delivery, teacher_stage_mask, teacher_kl_lambda, teacher_bias_scale, teacher_until, router_impl_by_stage, learning_rate
- 최고 성능: MRR@20=0.097500 (PFULLV4D_C01_LEGACY_HYBRID, FeaturedMoE_v4_Distillation_serial)
- 최고 설정: teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, learning_rate=0.0035680107372466404
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/014_C01_LEGACY_HYBRID.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v4_distillation/movielens1m_FeaturedMoE_v4_Distillation_pfullv4d_c01_legacy_hybrid_20260310_164225_049360_pid30536.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.097500 | PFULLV4D_C01_LEGACY_HYBRID | teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, learning_rate=0.0035680107372466404 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/014_C01_LEGACY_HYBRID.log |
| 2 | 0.096600 | PFULLV4D_C00_PLAIN | teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, router_impl_by_stage={}, learning_rate=0.005080284237324735 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/016_C00_PLAIN.log |
| 3 | 0.096300 | PFULLV4D_C05_GLS_DB_MM_MAIN | teacher_design=group_local_stat12, teacher_delivery=distill_and_fused_bias, teacher_stage_mask=mid_micro_only, teacher_kl_lambda=0.002, teacher_bias_scale=0.2, teacher_until=0.25, router_impl_by_stage={}, learning_rate=0.0001907007188571642 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/018_C05_GLS_DB_MM_MAIN.log |

### movielens1m / fmoe_v4d_p2_compare16

- 실험 설명: v4_distillation Phase 2 compare-16: GLS distill vs rule_soft teacher vs direct rule-hybrid and layout controls on flat_legacy.
- 실행 규모: runs=16, oom=0, 기간=2026-03-11T02:24:29.765037+00:00 ~ 2026-03-11T05:32:46.899137+00:00
- 비교 변수: fmoe_v2_layout_id, fmoe_stage_execution_mode, router_impl, router_impl_by_stage, rule_router.variant, rule_router.n_bins, teacher_design, teacher_delivery, teacher_stage_mask, teacher_kl_lambda, teacher_bias_scale, teacher_until, learning_rate
- 최고 성능: MRR@20=0.096300 (P2CMP16_C08_RULE_HYBRID_SOFT, FeaturedMoE_v4_Distillation_serial)
- 최고 설정: fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.variant=ratio_bins, rule_router.n_bins=5, teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, learning_rate=0.004179456411798079
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/P2CMP16/ML1/FMoEv4D/128_C08_RULE_HYBRID_SOFT.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_v4_distillation/movielens1m_FeaturedMoE_v4_Distillation_p2cmp16_c08_rule_hybrid_soft_20260311_034813_544693_pid68460.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.096300 | P2CMP16_C08_RULE_HYBRID_SOFT | fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, router_impl=learned, router_impl_by_stage={'mid': 'rule_soft', 'micro': 'rule_soft'}, rule_router.variant=ratio_bins, rule_router.n_bins=5, teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, learning_rate=0.004179456411798079 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/P2CMP16/ML1/FMoEv4D/128_C08_RULE_HYBRID_SOFT.log |
| 2 | 0.095900 | P2CMP16_C07_PLAIN | fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial, router_impl=learned, router_impl_by_stage={}, rule_router.variant=ratio_bins, rule_router.n_bins=5, teacher_design=none, teacher_delivery=none, teacher_stage_mask=all, teacher_kl_lambda=0.0, teacher_bias_scale=0.0, teacher_until=0.25, learning_rate=0.005782166641387214 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/P2CMP16/ML1/FMoEv4D/127_C07_PLAIN.log |
| 3 | 0.095200 | P2CMP16_C14_L16_MM | fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial, router_impl=learned, router_impl_by_stage={}, rule_router.variant=ratio_bins, rule_router.n_bins=5, teacher_design=group_local_stat12, teacher_delivery=distill_and_fused_bias, teacher_stage_mask=mid_micro_only, teacher_kl_lambda=0.002, teacher_bias_scale=0.2, teacher_until=0.25, learning_rate=0.0007057916299349509 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/P2CMP16/ML1/FMoEv4D/134_C14_L16_MM.log |

