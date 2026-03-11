# fmoe_hgr_v4 Experiment Overview

- generated_at_utc: 2026-03-10T16:31:45.870262+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 1
- included_runs: 1
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0
- summarized_experiments: 1

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| movielens1m | R0_hgr_v4_distill4_L15_D0_off | hparam | 1 | 0 | 0.094200 | 0.0942 | R0HGRv4_distill4_C00_L15_D0_off | arch_layout_id, group_router_mode, embedding_size, d_expert_hidden, d_router_hidden, expert_scale, inner_rule_mode, inner_rule_lambda, inner_rule_temperature, inner_rule_until, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr_v4/hparam/R0HGRv4/ML1/FMoEHGRv4/000_distill4_C00_L15_D0_off_r3.log |

## Experiment Notes

### movielens1m / R0_hgr_v4_distill4_L15_D0_off

- 실험 설명: HGRv4 R0: feature-aware outer restored, group-stat inner teacher, 4-way distill level comparison. layout=15 merge=serial outer=hybrid(hidden=true,feature=true) inner_mode=off lambda=0.0 tau=1.5 until=0.2 dims=128/16/160/64 bs=4096/8192
- 실행 규모: runs=1, oom=0, 기간=2026-03-10T14:29:04.469826+00:00 ~ 2026-03-10T16:29:29.713984+00:00
- 비교 변수: arch_layout_id, group_router_mode, embedding_size, d_expert_hidden, d_router_hidden, expert_scale, inner_rule_mode, inner_rule_lambda, inner_rule_temperature, inner_rule_until, learning_rate
- 최고 성능: MRR@20=0.094200 (R0HGRv4_distill4_C00_L15_D0_off, FeaturedMoE_HGRv4)
- 최고 설정: arch_layout_id=15, group_router_mode=hybrid, embedding_size=128, d_expert_hidden=160, d_router_hidden=64, expert_scale=4, inner_rule_mode=off, inner_rule_lambda=0.0, inner_rule_temperature=1.5, inner_rule_until=0.2, learning_rate=0.0007
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr_v4/hparam/R0HGRv4/ML1/FMoEHGRv4/000_distill4_C00_L15_D0_off_r3.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_hgr_v4/movielens1m_FeaturedMoE_HGRv4_r0hgrv4_distill4_c00_l15_d0_off_20260310_142907_583764_pid5615.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.094200 | R0HGRv4_distill4_C00_L15_D0_off | arch_layout_id=15, group_router_mode=hybrid, embedding_size=128, d_expert_hidden=160, d_router_hidden=64, expert_scale=4, inner_rule_mode=off, inner_rule_lambda=0.0, inner_rule_temperature=1.5, inner_rule_until=0.2, learning_rate=0.0007 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_hgr_v4/hparam/R0HGRv4/ML1/FMoEHGRv4/000_distill4_C00_L15_D0_off_r3.log |

