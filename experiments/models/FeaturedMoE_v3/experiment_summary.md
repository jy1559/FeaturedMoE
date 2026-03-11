# FeaturedMoE_v3 Experiment Report

- generated_at_utc: 2026-03-10T16:11:12.257910+00:00
- track: fmoe_v3
- include_rule: keep OOM runs, keep successful runs with valid MRR@20, exclude non-OOM failures and no-metric runs
- matched_end_events: 18
- included_runs: 18
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0

## Best By Dataset (MRR@20)

| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |
|---|---:|---|---|---:|---|---|
| movielens1m | 0.097300 | hparam | P2ROUTER_C00_LEGACY | 115.31 | 2026-03-10T07:24:37.636331+00:00 | 20260310_052 |

## Included Runs

| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |
|---|---|---|---|---|---|---:|---:|---|---|
| 2026-03-10T16:11:12.137396+00:00 | movielens1m | hparam | P3ROUTER_C01_HGCLONE_GCLONE | success | no | 0.096300 | 106.60 | hparam/P3ROUTER_C01_HGCLONE_GCLONE ; --max-evals=8, --tune-epochs=35, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_142 |
| 2026-03-10T16:04:36.311933+00:00 | movielens1m | hparam | P3DISTILL_C01_LEGACY_CLONE | success | no | 0.095600 | 100.00 | hparam/P3DISTILL_C01_LEGACY_CLONE ; --max-evals=8, --tune-epochs=35, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_142 |
| 2026-03-10T16:02:52.072151+00:00 | movielens1m | hparam | P3ROUTER_C00_LEGACY_GCLONE | success | no | 0.096600 | 98.27 | hparam/P3ROUTER_C00_LEGACY_GCLONE ; --max-evals=8, --tune-epochs=35, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_142 |
| 2026-03-10T16:01:23.668692+00:00 | movielens1m | hparam | P3DISTILL_C00_LEGACY_PLAIN | success | no | 0.096400 | 96.79 | hparam/P3DISTILL_C00_LEGACY_PLAIN ; --max-evals=8, --tune-epochs=35, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_142 |
| 2026-03-10T11:41:59.628174+00:00 | movielens1m | hparam | P2ROUTER_C05_GCOMBO | success | no | 0.095300 | 146.46 | hparam/P2ROUTER_C05_GCOMBO ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_091 |
| 2026-03-10T11:18:06.279537+00:00 | movielens1m | hparam | P2ESLR_S3 | success | no | 0.096900 | 101.98 | hparam/P2ESLR_S3 ; --max-evals=8, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_093 |
| 2026-03-10T11:06:51.985651+00:00 | movielens1m | hparam | P2DISTILL_clone_router_cmp | success | no | 0.095600 | 96.24 | hparam/P2DISTILL_clone_router_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_093 |
| 2026-03-10T10:46:25.432069+00:00 | movielens1m | hparam | P2ROUTER_C04_CRES | success | no | 0.094900 | 119.20 | hparam/P2ROUTER_C04_CRES ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_084 |
| 2026-03-10T10:43:47.959343+00:00 | movielens1m | hparam | P2DISTILL_legacy_hybrid_cmp | success | no | 0.095800 | 93.05 | hparam/P2DISTILL_legacy_hybrid_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_091 |
| 2026-03-10T10:42:39.353194+00:00 | movielens1m | hparam | P2ESLR_S1 | success | no | 0.091300 | 66.53 | hparam/P2ESLR_S1 ; --max-evals=8, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_093 |
| 2026-03-10T09:30:37.506556+00:00 | movielens1m | hparam | P2DISTILL_base_clone_cmp | success | no | 0.096300 | 147.78 | hparam/P2DISTILL_base_clone_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_070 |
| 2026-03-10T09:15:31.630805+00:00 | movielens1m | hparam | P2ROUTER_C03_HGCLONE | success | no | 0.095400 | 110.89 | hparam/P2ROUTER_C03_HGCLONE ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_072 |
| 2026-03-10T09:10:44.868045+00:00 | movielens1m | hparam | P2DISTILL_base_gclone_cmp | success | no | 0.096600 | 112.77 | hparam/P2DISTILL_base_gclone_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_071 |
| 2026-03-10T08:47:13.039991+00:00 | movielens1m | hparam | P2ROUTER_C02_GINT | success | no | 0.090800 | 85.47 | hparam/P2ROUTER_C02_GINT ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_072 |
| 2026-03-10T07:24:37.636331+00:00 | movielens1m | hparam | P2ROUTER_C00_LEGACY | success | no | 0.097300 | 115.31 | hparam/P2ROUTER_C00_LEGACY ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_052 |
| 2026-03-10T07:21:44.247866+00:00 | movielens1m | hparam | P2ROUTER_C01_HONLY | success | no | 0.092600 | 112.42 | hparam/P2ROUTER_C01_HONLY ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_052 |
| 2026-03-10T07:17:58.054079+00:00 | movielens1m | hparam | P2DISTILL_base_plain_cmp | success | no | 0.095300 | 107.99 | hparam/P2DISTILL_base_plain_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_052 |
| 2026-03-10T07:02:50.357112+00:00 | movielens1m | hparam | P2DISTILL_base_group_cmp | success | no | 0.095600 | 92.86 | hparam/P2DISTILL_base_group_cmp ; --max-evals=10, --tune-epochs=40, --tune-patience=5, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260310_052 |
