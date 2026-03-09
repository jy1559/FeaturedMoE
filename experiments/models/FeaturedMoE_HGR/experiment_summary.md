# FeaturedMoE_HGR Experiment Report

- generated_at_utc: 2026-03-09T02:05:24.331944+00:00
- track: fmoe_hgr
- include_rule: keep OOM runs, keep successful runs with valid MRR@20, exclude non-OOM failures and no-metric runs
- matched_end_events: 26
- included_runs: 26
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 0

## Best By Dataset (MRR@20)

| dataset | best_mrr@20 | axis | phase | duration_min | ended_at_utc | run_id |
|---|---:|---|---|---:|---|---|
| movielens1m | 0.094600 | hparam | P1HGR_widewide_C64_serial_per_group | 108.37 | 2026-03-09T00:34:32.283442+00:00 | 20260308_224 |

## Included Runs

| end_utc | dataset | axis | phase | status | oom | mrr@20 | duration_min | how | run_id |
|---|---|---|---|---|---|---:|---:|---|---|
| 2026-03-09T02:05:24.228201+00:00 | movielens1m | hparam | P1HGR_widewide_C65_serial_per_group | success | no | 0.093800 | 90.86 | hparam/P1HGR_widewide_C65_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260309_003 |
| 2026-03-09T01:02:40.301427+00:00 | movielens1m | hparam | P1HGR_widewide_C03_parallel_hybrid | success | no | 0.090700 | 200.24 | hparam/P1HGR_widewide_C03_parallel_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_214 |
| 2026-03-09T00:58:19.884274+00:00 | movielens1m | hparam | P1HGR_widewide_C45_serial_per_group | success | no | 0.094000 | 69.16 | hparam/P1HGR_widewide_C45_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_234 |
| 2026-03-09T00:54:57.933438+00:00 | movielens1m | hparam | P1HGR_widewide_C25_serial_per_group | success | no | 0.092200 | 72.79 | hparam/P1HGR_widewide_C25_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_234 |
| 2026-03-09T00:34:32.283442+00:00 | movielens1m | hparam | P1HGR_widewide_C64_serial_per_group | success | no | 0.094600 | 108.37 | hparam/P1HGR_widewide_C64_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_224 |
| 2026-03-08T23:49:10.211410+00:00 | movielens1m | hparam | P1HGR_widewide_C44_parallel_per_group | success | no | 0.091700 | 108.61 | hparam/P1HGR_widewide_C44_parallel_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_220 |
| 2026-03-08T23:42:10.087956+00:00 | movielens1m | hparam | P1HGR_widewide_C24_serial_hybrid | success | no | 0.093700 | 95.02 | hparam/P1HGR_widewide_C24_serial_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_220 |
| 2026-03-08T22:46:09.749100+00:00 | movielens1m | hparam | P1HGR_widewide_C63_parallel_hybrid | success | no | 0.091300 | 97.41 | hparam/P1HGR_widewide_C63_parallel_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_210 |
| 2026-03-08T22:07:08.757526+00:00 | movielens1m | hparam | P1HGR_widewide_C23_parallel_hybrid | success | no | 0.086500 | 75.55 | hparam/P1HGR_widewide_C23_parallel_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_205 |
| 2026-03-08T22:00:33.299906+00:00 | movielens1m | hparam | P1HGR_widewide_C43_parallel_hybrid | success | no | 0.086600 | 74.97 | hparam/P1HGR_widewide_C43_parallel_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_204 |
| 2026-03-08T21:42:25.540903+00:00 | movielens1m | hparam | P1HGR_widewide_C02_serial_per_group | success | no | 0.093700 | 91.03 | hparam/P1HGR_widewide_C02_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_201 |
| 2026-03-08T21:08:44.557649+00:00 | movielens1m | hparam | P1HGR_widewide_C62_serial_per_group | success | no | 0.094000 | 97.19 | hparam/P1HGR_widewide_C62_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_193 |
| 2026-03-08T20:51:35.377002+00:00 | movielens1m | hparam | P1HGR_widewide_C22_serial_hybrid | success | no | 0.093700 | 98.14 | hparam/P1HGR_widewide_C22_serial_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_191 |
| 2026-03-08T20:45:35.073811+00:00 | movielens1m | hparam | P1HGR_widewide_C42_parallel_per_group | success | no | 0.093300 | 92.81 | hparam/P1HGR_widewide_C42_parallel_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_191 |
| 2026-03-08T20:11:23.642682+00:00 | movielens1m | hparam | P1HGR_widewide_C01_serial_hybrid | success | no | 0.093300 | 132.72 | hparam/P1HGR_widewide_C01_serial_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_175 |
| 2026-03-08T19:31:32.858126+00:00 | movielens1m | hparam | P1HGR_widewide_C61_serial_per_group | success | no | 0.094100 | 86.57 | hparam/P1HGR_widewide_C61_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_180 |
| 2026-03-08T19:13:26.582237+00:00 | movielens1m | hparam | P1HGR_widewide_C21_serial_per_group | success | no | 0.094100 | 72.32 | hparam/P1HGR_widewide_C21_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_180 |
| 2026-03-08T19:12:46.243218+00:00 | movielens1m | hparam | P1HGR_widewide_C41_serial_per_group | success | no | 0.093300 | 70.01 | hparam/P1HGR_widewide_C41_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_180 |
| 2026-03-08T18:04:58.239843+00:00 | movielens1m | hparam | P1HGR_widewide_C60_serial_per_group | success | no | 0.092400 | 89.56 | hparam/P1HGR_widewide_C60_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_163 |
| 2026-03-08T18:02:45.149511+00:00 | movielens1m | hparam | P1HGR_widewide_C40_parallel_per_group | success | no | 0.093700 | 87.34 | hparam/P1HGR_widewide_C40_parallel_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_163 |
| 2026-03-08T18:01:06.977927+00:00 | movielens1m | hparam | P1HGR_widewide_C20_serial_hybrid | success | no | 0.093300 | 85.71 | hparam/P1HGR_widewide_C20_serial_hybrid ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_163 |
| 2026-03-08T17:58:40.330737+00:00 | movielens1m | hparam | P1HGR_widewide_C00_serial_per_group | success | no | 0.093500 | 83.26 | hparam/P1HGR_widewide_C00_serial_per_group ; --max-evals=15, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_163 |
| 2026-03-08T15:48:23.508887+00:00 | movielens1m | hparam | P1HGR_C00_serial_per_group | success | no | 0.068400 | 61.35 | hparam/P1HGR_C00_serial_per_group ; --max-evals=10, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_144 |
| 2026-03-08T15:48:15.127221+00:00 | movielens1m | hparam | P1HGR_C20_serial_hybrid | success | no | 0.067700 | 61.21 | hparam/P1HGR_C20_serial_hybrid ; --max-evals=10, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_144 |
| 2026-03-08T15:48:10.160053+00:00 | movielens1m | hparam | P1HGR_C40_parallel_per_group | success | no | 0.067300 | 61.13 | hparam/P1HGR_C40_parallel_per_group ; --max-evals=10, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_144 |
| 2026-03-08T15:47:01.862164+00:00 | movielens1m | hparam | P1HGR_C60_parallel_hybrid | success | no | 0.067100 | 59.99 | hparam/P1HGR_C60_parallel_hybrid ; --max-evals=10, --tune-epochs=20, --tune-patience=5, dataset=movielens1m | 20260308_144 |
