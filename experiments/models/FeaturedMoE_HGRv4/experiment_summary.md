# FeaturedMoE_HGRv4 Experiment Report

- generated_at_utc: 2026-03-10T16:31:45.779358+00:00
- track: fmoe_hgr_v4
- include_rule: keep OOM runs, keep successful runs with valid MRR@20, exclude non-OOM failures and no-metric runs
- matched_end_events: 1
- included_runs: 1
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0

## Best By Dataset (MRR@20)

| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |
|---|---:|---|---|---:|---|---|
| movielens1m | 0.094200 | hparam | R0HGRv4_distill4_C00_L15_D0_off | 120.42 | 2026-03-10T16:29:29.713984+00:00 | 20260310_142 |

## Included Runs

| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |
|---|---|---|---|---|---|---:|---:|---|---|
| 2026-03-10T16:29:29.713984+00:00 | movielens1m | hparam | R0HGRv4_distill4_C00_L15_D0_off | success | no | 0.094200 | 120.42 | hparam/R0HGRv4_distill4_C00_L15_D0_off ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m | 20260310_142 |
