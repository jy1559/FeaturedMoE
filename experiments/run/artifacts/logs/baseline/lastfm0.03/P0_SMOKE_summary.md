# lastfm0.03 P0_SMOKE Summary

- updated_at_utc: 2026-03-11T16:47:41.041985+00:00
- dataset: lastfm0.03
- phase_folder: P0_SMOKE
- logs_present: 40
- rows_shown: 40
- models_shown: 8

## Metric Layout

- Each metric block is shown as `cur | run | folder`.
- `folder` means the best `run` value inside the same model table below.
- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.
- A row is shown only if the current trial still matches the run-best on at least one metric.

## Models

<details>
<summary><strong>sasrec</strong> (shown 8/8, best_mrr@20=0.1098, best_hr@10=0.2386)</summary>

- shown_rows: 8 / total_logs_for_model: 8
- folder_best: best_mrr@20=0.1098, best_hr@10=0.2386, test_mrr@20=0.1098, test_hr@10=0.2330

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sasrec_000_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_155421_626281_pid146259.json |
| sasrec_002_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_155713_874966_pid146897.json |
| sasrec_010_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_160059_635970_pid149114.json |
| sasrec_018_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_160956_786453_pid151905.json |
| sasrec_021_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_161445_100041_pid153156.json |
| sasrec_022_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_161756_123992_pid155439.json |
| sasrec_032_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_164423_312846_pid167627.json |
| sasrec_034_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.1098 | 0.1098 | 0.1098 | 0.2386 | 0.2386 | 0.2386 | 0.1098 | 0.1098 | 0.1098 | 0.2330 | 0.2330 | 0.2330 | lastfm0.03_SASRec_p0_smoke_a_pair01_c1_20260311_164539_103563_pid168080.json |

</details>

<details>
<summary><strong>gru4rec</strong> (shown 8/8, best_mrr@20=0.0014, best_hr@10=0.0034)</summary>

- shown_rows: 8 / total_logs_for_model: 8
- folder_best: best_mrr@20=0.0014, best_hr@10=0.0034, test_mrr@20=0.0012, test_hr@10=0.0032

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gru4rec_001_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_155421_580374_pid146261.json |
| gru4rec_003_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_155713_894078_pid146899.json |
| gru4rec_011_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_160059_710693_pid149117.json |
| gru4rec_019_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_160957_162287_pid151907.json |
| gru4rec_020_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_161445_209012_pid153153.json |
| gru4rec_023_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_161756_281407_pid155441.json |
| gru4rec_033_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_164423_349680_pid167628.json |
| gru4rec_035_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0014 | 0.0014 | 0.0014 | 0.0034 | 0.0034 | 0.0034 | 0.0012 | 0.0012 | 0.0012 | 0.0032 | 0.0032 | 0.0032 | lastfm0.03_GRU4Rec_p0_smoke_a_pair01_c1_20260311_164539_075198_pid168081.json |

</details>

<details>
<summary><strong>bsarec</strong> (shown 4/4, best_mrr@20=0.0001, best_hr@10=0.0002)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0001, best_hr@10=0.0002, test_mrr@20=0.0001, test_hr@10=0.0002

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bsarec_004_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_BSARec_p0_smoke_a_pair02_c1_20260311_155728_019669_pid147218.json |
| bsarec_012_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_BSARec_p0_smoke_a_pair02_c1_20260311_160113_205645_pid149468.json |
| bsarec_025_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_BSARec_p0_smoke_a_pair02_c1_20260311_161810_427798_pid155797.json |
| bsarec_036_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_BSARec_p0_smoke_a_pair02_c1_20260311_164552_841545_pid168431.json |

</details>

<details>
<summary><strong>fame</strong> (shown 4/4, best_mrr@20=0.0002, best_hr@10=0.0007)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0002, best_hr@10=0.0007, test_mrr@20=0.0001, test_hr@10=0.0005

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fame_005_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0002 | 0.0002 | 0.0002 | 0.0007 | 0.0007 | 0.0007 | 0.0001 | 0.0001 | 0.0001 | 0.0005 | 0.0005 | 0.0005 | lastfm0.03_FAME_p0_smoke_a_pair02_c1_20260311_155727_875238_pid147220.json |
| fame_013_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0002 | 0.0002 | 0.0002 | 0.0007 | 0.0007 | 0.0007 | 0.0001 | 0.0001 | 0.0001 | 0.0005 | 0.0005 | 0.0005 | lastfm0.03_FAME_p0_smoke_a_pair02_c1_20260311_160113_336396_pid149471.json |
| fame_024_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0002 | 0.0002 | 0.0002 | 0.0007 | 0.0007 | 0.0007 | 0.0001 | 0.0001 | 0.0001 | 0.0005 | 0.0005 | 0.0005 | lastfm0.03_FAME_p0_smoke_a_pair02_c1_20260311_161810_359443_pid155796.json |
| fame_037_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0002 | 0.0002 | 0.0002 | 0.0007 | 0.0007 | 0.0007 | 0.0001 | 0.0001 | 0.0001 | 0.0005 | 0.0005 | 0.0005 | lastfm0.03_FAME_p0_smoke_a_pair02_c1_20260311_164552_958943_pid168433.json |

</details>

<details>
<summary><strong>patt</strong> (shown 4/4, best_mrr@20=0.0001, best_hr@10=0.0002)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0001, best_hr@10=0.0002, test_mrr@20=0.0001, test_hr@10=0.0002

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| patt_006_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_PAtt_p0_smoke_a_pair03_c1_20260311_155804_271648_pid147621.json |
| patt_015_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_PAtt_p0_smoke_a_pair03_c1_20260311_160150_030177_pid149923.json |
| patt_026_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_PAtt_p0_smoke_a_pair03_c1_20260311_161847_119903_pid156252.json |
| patt_039_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | lastfm0.03_PAtt_p0_smoke_a_pair03_c1_20260311_164629_351335_pid168891.json |

</details>

<details>
<summary><strong>fenrec</strong> (shown 4/4, best_mrr@20=0.0001, best_hr@10=0.0003)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0001, best_hr@10=0.0003, test_mrr@20=0.0001, test_hr@10=0.0001

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fenrec_007_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | lastfm0.03_FENRec_p0_smoke_a_pair03_c1_20260311_155804_092982_pid147623.json |
| fenrec_014_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | lastfm0.03_FENRec_p0_smoke_a_pair03_c1_20260311_160149_728921_pid149920.json |
| fenrec_027_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | lastfm0.03_FENRec_p0_smoke_a_pair03_c1_20260311_161847_119577_pid156255.json |
| fenrec_038_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | lastfm0.03_FENRec_p0_smoke_a_pair03_c1_20260311_164629_354747_pid168890.json |

</details>

<details>
<summary><strong>sigma</strong> (shown 4/4, best_mrr@20=0.0001, best_hr@10=0.0002)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0001, best_hr@10=0.0002, test_mrr@20=0.0001, test_hr@10=0.0003

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sigma_008_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | lastfm0.03_SIGMA_p0_smoke_a_pair04_c1_20260311_155824_161557_pid147980.json |
| sigma_016_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | lastfm0.03_SIGMA_p0_smoke_a_pair04_c1_20260311_160210_433172_pid150268.json |
| sigma_028_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | lastfm0.03_SIGMA_p0_smoke_a_pair04_c1_20260311_161907_732275_pid156679.json |
| sigma_040_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0003 | 0.0003 | lastfm0.03_SIGMA_p0_smoke_a_pair04_c1_20260311_164650_069842_pid169256.json |

</details>

<details>
<summary><strong>srgnn</strong> (shown 4/4, best_mrr@20=0.0095, best_hr@10=0.0144)</summary>

- shown_rows: 4 / total_logs_for_model: 4
- folder_best: best_mrr@20=0.0095, best_hr@10=0.0144, test_mrr@20=0.0089, test_hr@10=0.0136

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| srgnn_009_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0095 | 0.0095 | 0.0095 | 0.0144 | 0.0144 | 0.0144 | 0.0089 | 0.0089 | 0.0089 | 0.0136 | 0.0136 | 0.0136 | lastfm0.03_SRGNN_p0_smoke_a_pair04_c1_20260311_155824_128853_pid147982.json |
| srgnn_017_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0095 | 0.0095 | 0.0095 | 0.0144 | 0.0144 | 0.0144 | 0.0089 | 0.0089 | 0.0089 | 0.0136 | 0.0136 | 0.0136 | lastfm0.03_SRGNN_p0_smoke_a_pair04_c1_20260311_160210_549881_pid150271.json |
| srgnn_029_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0095 | 0.0095 | 0.0095 | 0.0144 | 0.0144 | 0.0144 | 0.0089 | 0.0089 | 0.0089 | 0.0136 | 0.0136 | 0.0136 | lastfm0.03_SRGNN_p0_smoke_a_pair04_c1_20260311_161907_710320_pid156682.json |
| srgnn_041_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0095 | 0.0095 | 0.0095 | 0.0144 | 0.0144 | 0.0144 | 0.0089 | 0.0089 | 0.0089 | 0.0136 | 0.0136 | 0.0136 | lastfm0.03_SRGNN_p0_smoke_a_pair04_c1_20260311_164650_049207_pid169258.json |

</details>

