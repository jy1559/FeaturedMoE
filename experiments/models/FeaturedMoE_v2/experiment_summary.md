# FeaturedMoE_v2 Experiment Report

- generated_at_utc: 2026-03-10T02:34:09.872762+00:00
- track: fmoe_rule
- include_rule: keep OOM runs, keep successful runs with valid MRR@20, exclude non-OOM failures and no-metric runs
- matched_end_events: 28
- included_runs: 15
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 13

## Best By Dataset (MRR@20)

| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |
|---|---:|---|---|---:|---|---|
| movielens1m | 0.098800 | hparam | RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096 | 145.71 | 2026-03-06T13:54:53.029118+00:00 | 20260306_112 |
| retail_rocket | 0.262000 | hparam | RRRULE_R1_G4_C00_L16F24 | 69.89 | 2026-03-10T02:29:18.118688+00:00 | 20260310_011 |

## Included Runs

| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |
|---|---|---|---|---|---|---:|---:|---|---|
| 2026-03-10T02:34:09.755442+00:00 | retail_rocket | hparam | RRRULE_R1_G7_C03_L15MED | success | no | 0.261000 | 74.70 | hparam/RRRULE_R1_G7_C03_L15MED ; --max-evals=8, --tune-epochs=60, --tune-patience=8, dataset=retail_rocket, fmoe_v2_layout_id=15, fmoe_stage_execution_mode=serial | 20260310_011 |
| 2026-03-10T02:29:18.118688+00:00 | retail_rocket | hparam | RRRULE_R1_G4_C00_L16F24 | success | no | 0.262000 | 69.89 | hparam/RRRULE_R1_G4_C00_L16F24 ; --max-evals=8, --tune-epochs=60, --tune-patience=8, dataset=retail_rocket, fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial | 20260310_011 |
| 2026-03-10T02:27:59.179005+00:00 | retail_rocket | hparam | RRRULE_R1_G6_C02_L16BIG | success | no | 0.259900 | 68.54 | hparam/RRRULE_R1_G6_C02_L16BIG ; --max-evals=8, --tune-epochs=60, --tune-patience=8, dataset=retail_rocket, fmoe_v2_layout_id=16, fmoe_stage_execution_mode=serial | 20260310_011 |
| 2026-03-06T17:21:19.436087+00:00 | movielens1m | hparam | RULE_R0_P2DB_G4_C3_movielens1m_E160_R80_B4096 | success | no | 0.073500 | 171.85 | hparam/RULE_R0_P2DB_G4_C3_movielens1m_E160_R80_B4096 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_142 |
| 2026-03-06T17:19:23.701316+00:00 | movielens1m | hparam | RULE_R0_P2DB_G6_C3_movielens1m_E128_R64_B4096 | success | no | 0.074200 | 171.48 | hparam/RULE_R0_P2DB_G6_C3_movielens1m_E128_R64_B4096 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_142 |
| 2026-03-06T16:45:42.463319+00:00 | movielens1m | hparam | RULE_R1_P2DB_G7_C3_movielens1m_E160_R96_B6144 | success | no | 0.095600 | 148.68 | hparam/RULE_R1_P2DB_G7_C3_movielens1m_E160_R96_B6144 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_141 |
| 2026-03-06T16:23:23.795890+00:00 | movielens1m | hparam | RULE_R1_P2DB_G5_C3_movielens1m_E192_R96_B6144 | success | no | 0.095500 | 148.51 | hparam/RULE_R1_P2DB_G5_C3_movielens1m_E192_R96_B6144 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_135 |
| 2026-03-06T14:29:28.232784+00:00 | movielens1m | hparam | RULE_R0_P2DB_G4_C2_movielens1m_E128_R64_B6144 | success | no | 0.074400 | 167.46 | hparam/RULE_R0_P2DB_G4_C2_movielens1m_E128_R64_B6144 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_114 |
| 2026-03-06T14:27:54.664072+00:00 | movielens1m | hparam | RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192 | success | no | 0.075200 | 174.19 | hparam/RULE_R0_P2DB_G6_C2_movielens1m_E128_R64_B8192 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_113 |
| 2026-03-06T14:17:01.495424+00:00 | movielens1m | hparam | RULE_R1_P2DB_G7_C2_movielens1m_E160_R80_B4096 | success | no | 0.097800 | 157.57 | hparam/RULE_R1_P2DB_G7_C2_movielens1m_E160_R80_B4096 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_113 |
| 2026-03-06T13:54:53.029118+00:00 | movielens1m | hparam | RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096 | success | no | 0.098800 | 145.71 | hparam/RULE_R1_P2DB_G5_C2_movielens1m_E192_R96_B4096 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_112 |
| 2026-03-06T11:42:00.173554+00:00 | movielens1m | hparam | RULE_R0_P2DB_G4_C1_movielens1m_E128_R64_B4096 | success | no | 0.074100 | 176.64 | hparam/RULE_R0_P2DB_G4_C1_movielens1m_E128_R64_B4096 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_084 |
| 2026-03-06T11:39:26.792036+00:00 | movielens1m | hparam | RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144 | success | no | 0.074100 | 174.08 | hparam/RULE_R1_P2DB_G7_C1_movielens1m_E128_R64_B6144 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_084 |
| 2026-03-06T11:33:42.700449+00:00 | movielens1m | hparam | RULE_R0_P2DB_G6_C1_movielens1m_E160_R80_B8192 | success | no | 0.074100 | 168.34 | hparam/RULE_R0_P2DB_G6_C1_movielens1m_E160_R80_B8192 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_084 |
| 2026-03-06T11:29:10.067319+00:00 | movielens1m | hparam | RULE_R1_P2DB_G5_C1_movielens1m_E160_R96_B6144 | success | no | 0.074100 | 163.80 | hparam/RULE_R1_P2DB_G5_C1_movielens1m_E160_R96_B6144 ; --max-evals=10, --tune-epochs=100, --tune-patience=10, dataset=movielens1m, fmoe_v2_layout_id=7, fmoe_stage_execution_mode=serial | 20260306_084 |
