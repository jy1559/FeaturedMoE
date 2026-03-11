# fmoe_v4_distillation Experiment Overview

- generated_at_utc: 2026-03-11T01:32:41.128099+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 24
- included_runs: 24
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 1

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | fmoe_v4d_full_distill32 | hparam | 24 | 0 | 0.097500 | 0.0975/0.0966/0.0963 | PFULLV4D_C01_LEGACY_HYBRID | teacher_design, teacher_delivery, teacher_stage_mask, teacher_kl_lambda, teacher_bias_scale, teacher_until, router_impl_by_stage, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_v4_distillation/hparam/PFULLV4D/ML1/FMoEv4D/014_C01_LEGACY_HYBRID.log |

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

