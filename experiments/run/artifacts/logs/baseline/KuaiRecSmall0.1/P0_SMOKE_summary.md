# KuaiRecSmall0.1 P0_SMOKE Summary

- updated_at_utc: 2026-03-11T16:47:08.936308+00:00
- dataset: KuaiRecSmall0.1
- phase_folder: P0_SMOKE
- logs_present: 10
- rows_shown: 10
- models_shown: 8

## Metric Layout

- Each metric block is shown as `cur | run | folder`.
- `folder` means the best `run` value inside the same model table below.
- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.
- A row is shown only if the current trial still matches the run-best on at least one metric.

## Models

<details>
<summary><strong>sasrec</strong> (shown 2/2, best_mrr@20=0.0013, best_hr@10=0.0037)</summary>

- shown_rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0013, best_hr@10=0.0037, test_mrr@20=0.0014, test_hr@10=0.0033

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sasrec_032_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0013 | 0.0013 | 0.0013 | 0.0037 | 0.0037 | 0.0037 | 0.0014 | 0.0014 | 0.0014 | 0.0033 | 0.0033 | 0.0033 | KuaiRecSmall0.1_SASRec_p0_smoke_a_pair01_c1_20260311_164423_450734_pid167626.json |
| sasrec_034_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0013 | 0.0013 | 0.0013 | 0.0037 | 0.0037 | 0.0037 | 0.0014 | 0.0014 | 0.0014 | 0.0033 | 0.0033 | 0.0033 | KuaiRecSmall0.1_SASRec_p0_smoke_a_pair01_c1_20260311_164538_945593_pid168079.json |

</details>

<details>
<summary><strong>gru4rec</strong> (shown 2/2, best_mrr@20=0.0012, best_hr@10=0.0040)</summary>

- shown_rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0012, best_hr@10=0.0040, test_mrr@20=0.0012, test_hr@10=0.0036

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gru4rec_033_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0012 | 0.0012 | 0.0012 | 0.0040 | 0.0040 | 0.0040 | 0.0012 | 0.0012 | 0.0012 | 0.0036 | 0.0036 | 0.0036 | KuaiRecSmall0.1_GRU4Rec_p0_smoke_a_pair01_c1_20260311_164423_423999_pid167629.json |
| gru4rec_035_a_c1_hi_bs_short.log | P0_SMOKE_A_pair01_C1 | success | 1/1 | 0.0012 | 0.0012 | 0.0012 | 0.0040 | 0.0040 | 0.0040 | 0.0012 | 0.0012 | 0.0012 | 0.0036 | 0.0036 | 0.0036 | KuaiRecSmall0.1_GRU4Rec_p0_smoke_a_pair01_c1_20260311_164539_130482_pid168082.json |

</details>

<details>
<summary><strong>bsarec</strong> (shown 1/1, best_mrr@20=0.0009, best_hr@10=0.0030)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0009, best_hr@10=0.0030, test_mrr@20=0.0007, test_hr@10=0.0026

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bsarec_036_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0009 | 0.0009 | 0.0009 | 0.0030 | 0.0030 | 0.0030 | 0.0007 | 0.0007 | 0.0007 | 0.0026 | 0.0026 | 0.0026 | KuaiRecSmall0.1_BSARec_p0_smoke_a_pair02_c1_20260311_164552_796712_pid168430.json |

</details>

<details>
<summary><strong>fame</strong> (shown 1/1, best_mrr@20=0.0012, best_hr@10=0.0036)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0012, best_hr@10=0.0036, test_mrr@20=0.0011, test_hr@10=0.0030

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fame_037_a_c1_hi_bs_short.log | P0_SMOKE_A_pair02_C1 | success | 1/1 | 0.0012 | 0.0012 | 0.0012 | 0.0036 | 0.0036 | 0.0036 | 0.0011 | 0.0011 | 0.0011 | 0.0030 | 0.0030 | 0.0030 | KuaiRecSmall0.1_FAME_p0_smoke_a_pair02_c1_20260311_164553_001912_pid168432.json |

</details>

<details>
<summary><strong>fenrec</strong> (shown 1/1, best_mrr@20=0.0012, best_hr@10=0.0035)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0012, best_hr@10=0.0035, test_mrr@20=0.0014, test_hr@10=0.0042

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fenrec_038_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0012 | 0.0012 | 0.0012 | 0.0035 | 0.0035 | 0.0035 | 0.0014 | 0.0014 | 0.0014 | 0.0042 | 0.0042 | 0.0042 | KuaiRecSmall0.1_FENRec_p0_smoke_a_pair03_c1_20260311_164629_352283_pid168889.json |

</details>

<details>
<summary><strong>patt</strong> (shown 1/1, best_mrr@20=0.0015, best_hr@10=0.0039)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0015, best_hr@10=0.0039, test_mrr@20=0.0009, test_hr@10=0.0027

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| patt_039_a_c1_hi_bs_short.log | P0_SMOKE_A_pair03_C1 | success | 1/1 | 0.0015 | 0.0015 | 0.0015 | 0.0039 | 0.0039 | 0.0039 | 0.0009 | 0.0009 | 0.0009 | 0.0027 | 0.0027 | 0.0027 | KuaiRecSmall0.1_PAtt_p0_smoke_a_pair03_c1_20260311_164629_716253_pid168892.json |

</details>

<details>
<summary><strong>sigma</strong> (shown 1/1, best_mrr@20=0.0010, best_hr@10=0.0030)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0010, best_hr@10=0.0030, test_mrr@20=0.0010, test_hr@10=0.0028

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sigma_040_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0010 | 0.0010 | 0.0010 | 0.0030 | 0.0030 | 0.0030 | 0.0010 | 0.0010 | 0.0010 | 0.0028 | 0.0028 | 0.0028 | KuaiRecSmall0.1_SIGMA_p0_smoke_a_pair04_c1_20260311_164650_269727_pid169255.json |

</details>

<details>
<summary><strong>srgnn</strong> (shown 1/1, best_mrr@20=0.0019, best_hr@10=0.0043)</summary>

- shown_rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.0019, best_hr@10=0.0043, test_mrr@20=0.0012, test_hr@10=0.0035

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder | result_json |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| srgnn_041_a_c1_hi_bs_short.log | P0_SMOKE_A_pair04_C1 | success | 1/1 | 0.0019 | 0.0019 | 0.0019 | 0.0043 | 0.0043 | 0.0043 | 0.0012 | 0.0012 | 0.0012 | 0.0035 | 0.0035 | 0.0035 | KuaiRecSmall0.1_SRGNN_p0_smoke_a_pair04_c1_20260311_164650_092010_pid169257.json |

</details>

