# FeaturedMoE_N Experiment Report

- generated_at_utc: 2026-03-11T16:32:30.379322+00:00
- track: fmoe_n
- include_rule: keep OOM runs, keep successful runs with valid MRR@20, exclude non-OOM failures and no-metric runs
- matched_end_events: 22
- included_runs: 8
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 14

## Best By Dataset (MRR@20)

| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |
|---|---:|---|---|---:|---|---|
| KuaiRecSmall0.1 | 0.001800 | hparam | P0_Q02 | 0.21 | 2026-03-11T16:31:56.396183+00:00 | 20260311_163 |
| lastfm0.03 | 0.109600 | hparam | P0_F01 | 0.38 | 2026-03-11T16:32:06.620166+00:00 | 20260311_163 |

## Included Runs

| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |
|---|---|---|---|---|---|---:|---:|---|---|
| 2026-03-11T16:32:30.250615+00:00 | lastfm0.03 | hparam | P0_F02 | success | no | 0.039600 | 0.39 | hparam/P0_F02 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=lastfm0.03, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:32:20.099319+00:00 | KuaiRecSmall0.1 | hparam | P0_Q06 | success | no | 0.001600 | 0.22 | hparam/P0_Q06 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:32:19.664466+00:00 | KuaiRecSmall0.1 | hparam | P0_Q05 | success | no | 0.001700 | 0.21 | hparam/P0_Q05 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:32:19.628494+00:00 | KuaiRecSmall0.1 | hparam | P0_Q04 | success | no | 0.001500 | 0.21 | hparam/P0_Q04 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:32:06.620166+00:00 | lastfm0.03 | hparam | P0_F01 | success | no | 0.109600 | 0.38 | hparam/P0_F01 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=lastfm0.03, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:31:56.565421+00:00 | KuaiRecSmall0.1 | hparam | P0_Q03 | success | no | 0.001600 | 0.21 | hparam/P0_Q03 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:31:56.396183+00:00 | KuaiRecSmall0.1 | hparam | P0_Q02 | success | no | 0.001800 | 0.21 | hparam/P0_Q02 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260311_163 |
| 2026-03-11T16:31:55.944046+00:00 | KuaiRecSmall0.1 | hparam | P0_Q01 | success | no | 0.001500 | 0.20 | hparam/P0_Q01 ; --max-evals=1, --tune-epochs=1, --tune-patience=3, dataset=KuaiRecSmall0.1, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260311_163 |
