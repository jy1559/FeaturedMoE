# lastfm0.03 P0 Summary

- updated_at_utc: 2026-03-12T02:23:48.841241+00:00
- dataset: lastfm0.03
- phase_folder: P0
- logs_present: 7
- rows_shown: 7
- models_shown: 4

## Metric Layout

- Each metric block is shown as `cur | run | folder`.
- Each row is one hyperopt log/run, not one trial.
- Summary is built from existing log files only.
- `folder` means the best `run` value inside the same model table below.
- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.

## Models

<details>
<summary><strong>gru4rec</strong> (shown 2/2, best_mrr@20=0.3811, best_hr@10=0.4367)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.3811, best_hr@10=0.4367, test_mrr@20=0.3514, test_hr@10=0.4033

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru4rec_008_a_c1_hi_bs_short.log | P0_A_pair01_C1 | success | 20/20 | 0.3782 | 0.3782 | 0.3811 | 0.4311 | 0.4367 | 0.4367 | 0.3437 | 0.3514 | 0.3514 | 0.3942 | 0.4033 | 0.4033 |
| gru4rec_011_a_c2_std_bs_mid.log | P0_A_pair01_C2 | success | 20/20 | 0.3739 | 0.3811 | 0.3811 | 0.4270 | 0.4343 | 0.4367 | 0.3387 | 0.3473 | 0.3514 | 0.3891 | 0.3977 | 0.4033 |

</details>

<details>
<summary><strong>sasrec</strong> (shown 2/2, best_mrr@20=0.4004, best_hr@10=0.4782)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.4004, best_hr@10=0.4782, test_mrr@20=0.3801, test_hr@10=0.4576

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sasrec_009_a_c1_hi_bs_short.log | P0_A_pair01_C1 | success | 20/20 | 0.3972 | 0.3972 | 0.4004 | 0.4680 | 0.4782 | 0.4782 | 0.3759 | 0.3801 | 0.3801 | 0.4443 | 0.4576 | 0.4576 |
| sasrec_010_a_c2_std_bs_mid.log | P0_A_pair01_C2 | success | 20/20 | 0.3978 | 0.4004 | 0.4004 | 0.4685 | 0.4696 | 0.4782 | 0.3751 | 0.3761 | 0.3801 | 0.4434 | 0.4439 | 0.4576 |

</details>

<details>
<summary><strong>bsarec</strong> (shown 1/1, best_mrr@20=0.3904, best_hr@10=0.4433)</summary>

- rows: 1 / total_logs_for_model: 1
- folder_best: best_mrr@20=0.3904, best_hr@10=0.4433, test_mrr@20=0.3665, test_hr@10=0.4193

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bsarec_012_a_c1_hi_bs_short.log | P0_A_pair02_C1 | success | 20/20 | 0.3398 | 0.3904 | 0.3904 | 0.3872 | 0.4433 | 0.4433 | 0.3206 | 0.3665 | 0.3665 | 0.3659 | 0.4193 | 0.4193 |

</details>

<details>
<summary><strong>fame</strong> (shown 2/2, best_mrr@20=0.3936, best_hr@10=0.4533)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.3936, best_hr@10=0.4533, test_mrr@20=0.3724, test_hr@10=0.4302

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fame_013_a_c1_hi_bs_short.log | P0_A_pair02_C1 | running | 7/20 | 0.3749 | 0.3879 | 0.3936 | 0.4402 | 0.4394 | 0.4533 | 0.3533 | 0.3660 | 0.3724 | 0.4162 | 0.4158 | 0.4302 |
| fame_014_a_c2_std_bs_mid.log | P0_A_pair02_C2 | running | 3/20 | 0.3936 | 0.3936 | 0.3936 | 0.4533 | 0.4533 | 0.4533 | 0.3724 | 0.3724 | 0.3724 | 0.4302 | 0.4302 | 0.4302 |

</details>

