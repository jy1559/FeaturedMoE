# KuaiRecSmall0.1 P0 Summary

- updated_at_utc: 2026-03-11T22:31:39.990581+00:00
- dataset: KuaiRecSmall0.1
- phase_folder: P0
- logs_present: 8
- rows_shown: 8
- models_shown: 4

## Metric Layout

- Each metric block is shown as `cur | run | folder`.
- Each row is one hyperopt log/run, not one trial.
- Summary is built from existing log files only.
- `folder` means the best `run` value inside the same model table below.
- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.

## Models

<details>
<summary><strong>gru4rec</strong> (shown 2/2, best_mrr@20=0.0235, best_hr@10=0.0645)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0235, best_hr@10=0.0645, test_mrr@20=0.0223, test_hr@10=0.0613

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gru4rec_008_a_c1_hi_bs_short.log | P0_A_pair01_C1 | success | 20/20 | 0.0231 | 0.0231 | 0.0235 | 0.0629 | 0.0641 | 0.0645 | 0.0211 | 0.0223 | 0.0223 | 0.0590 | 0.0607 | 0.0613 |
| gru4rec_011_a_c2_std_bs_mid.log | P0_A_pair01_C2 | success | 20/20 | 0.0235 | 0.0235 | 0.0235 | 0.0622 | 0.0645 | 0.0645 | 0.0212 | 0.0220 | 0.0223 | 0.0585 | 0.0613 | 0.0613 |

</details>

<details>
<summary><strong>sasrec</strong> (shown 2/2, best_mrr@20=0.0188, best_hr@10=0.0646)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0188, best_hr@10=0.0646, test_mrr@20=0.0182, test_hr@10=0.0605

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sasrec_009_a_c1_hi_bs_short.log | P0_A_pair01_C1 | success | 20/20 | 0.0178 | 0.0178 | 0.0188 | 0.0616 | 0.0646 | 0.0646 | 0.0175 | 0.0182 | 0.0182 | 0.0597 | 0.0605 | 0.0605 |
| sasrec_010_a_c2_std_bs_mid.log | P0_A_pair01_C2 | success | 20/20 | 0.0188 | 0.0188 | 0.0188 | 0.0621 | 0.0643 | 0.0646 | 0.0169 | 0.0182 | 0.0182 | 0.0567 | 0.0587 | 0.0605 |

</details>

<details>
<summary><strong>bsarec</strong> (shown 2/2, best_mrr@20=0.0226, best_hr@10=0.0617)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0226, best_hr@10=0.0617, test_mrr@20=0.0208, test_hr@10=0.0554

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bsarec_012_a_c1_hi_bs_short.log | P0_A_pair02_C1 | success | 20/20 | 0.0079 | 0.0226 | 0.0226 | 0.0202 | 0.0617 | 0.0617 | 0.0071 | 0.0208 | 0.0208 | 0.0196 | 0.0554 | 0.0554 |
| bsarec_015_a_c2_std_bs_mid.log | P0_A_pair02_C2 | success | 20/20 | 0.0012 | 0.0209 | 0.0226 | 0.0033 | 0.0548 | 0.0617 | 0.0013 | 0.0189 | 0.0208 | 0.0041 | 0.0508 | 0.0554 |

</details>

<details>
<summary><strong>fame</strong> (shown 2/2, best_mrr@20=0.0221, best_hr@10=0.0564)</summary>

- rows: 2 / total_logs_for_model: 2
- folder_best: best_mrr@20=0.0221, best_hr@10=0.0564, test_mrr@20=0.0195, test_hr@10=0.0531

| experiment | run_phase | status | trials | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fame_013_a_c1_hi_bs_short.log | P0_A_pair02_C1 | success | 20/20 | 0.0018 | 0.0200 | 0.0221 | 0.0045 | 0.0518 | 0.0564 | 0.0013 | 0.0180 | 0.0195 | 0.0034 | 0.0484 | 0.0531 |
| fame_014_a_c2_std_bs_mid.log | P0_A_pair02_C2 | success | 20/20 | 0.0015 | 0.0221 | 0.0221 | 0.0042 | 0.0564 | 0.0564 | 0.0011 | 0.0195 | 0.0195 | 0.0029 | 0.0531 | 0.0531 |

</details>

