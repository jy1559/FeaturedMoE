# fmoe_n P0 Summary

- updated_at_utc: 2026-03-12T02:21:07.535305+00:00
- axis: hparam
- phase_folder: P0
- logs_root: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0
- logs_present: 12
- datasets_shown: 2

## Metric Layout

- Each metric block is shown as `cur | run | folder`.
- Each row is one hyperopt log/run, not one trial.
- Summary is built from existing `.log` files only.
- `cur` is the latest trial metrics seen in the log.
- `run` is final `[RUN_METRICS]` if present, else latest running-best metrics from the log.
- `folder` is the best `run` MRR@20 row inside the same dataset table below.
- Metric blocks: `best_mrr@20`, `best_hr@10`, `test_mrr@20`, `test_hr@10`.

## Datasets

<details>
<summary><strong>KuaiRecSmall0.1</strong> (shown 9/9, best_combo=Q04, best_mrr@20=0.0193, best_hr@10=0.0528)</summary>

- rows: 9
- folder_best: combo=Q04, best_mrr@20=0.0193, best_hr@10=0.0528, test_mrr@20=0.0186, test_hr@10=0.0526

| combo | experiment | status | trials | run_phase | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q01 | KU01/FMoE/20260311_172733_895_hparam_P0_Q01.log | success | 20/20 | P0_Q01 | 0.0174 | 0.0178 | 0.0193 | 0.0527 | 0.0542 | 0.0528 | 0.0159 | 0.0162 | 0.0186 | 0.0474 | 0.0487 | 0.0526 |
| Q02 | KU01/FMoE/20260311_172733_923_hparam_P0_Q02.log | success | 20/20 | P0_Q02 | 0.0145 | 0.0168 | 0.0193 | 0.0486 | 0.0531 | 0.0528 | 0.0136 | 0.0152 | 0.0186 | 0.0431 | 0.0481 | 0.0526 |
| Q03 | KU01/FMoE/20260311_172733_938_hparam_P0_Q03.log | success | 20/20 | P0_Q03 | 0.0176 | 0.0176 | 0.0193 | 0.0555 | 0.0555 | 0.0528 | 0.0158 | 0.0158 | 0.0186 | 0.0468 | 0.0468 | 0.0526 |
| Q04 | KU01/FMoE/20260311_205828_746_hparam_P0_Q04.log | success | 20/20 | P0_Q04 | 0.0147 | 0.0193 | 0.0193 | 0.0403 | 0.0528 | 0.0528 | 0.0130 | 0.0186 | 0.0186 | 0.0354 | 0.0526 | 0.0526 |
| Q05 | KU01/FMoE/20260311_205828_776_hparam_P0_Q05.log | success | 20/20 | P0_Q05 | 0.0158 | 0.0161 | 0.0193 | 0.0502 | 0.0485 | 0.0528 | 0.0139 | 0.0151 | 0.0186 | 0.0438 | 0.0447 | 0.0526 |
| Q06 | KU01/FMoE/20260311_205828_800_hparam_P0_Q06.log | success | 20/20 | P0_Q06 | 0.0056 | 0.0171 | 0.0193 | 0.0145 | 0.0537 | 0.0528 | 0.0049 | 0.0156 | 0.0186 | 0.0128 | 0.0495 | 0.0526 |
| Q07 | KU01/FMoE/20260312_000841_872_hparam_P0_Q07.log | success | 20/20 | P0_Q07 | 0.0055 | 0.0176 | 0.0193 | 0.0145 | 0.0487 | 0.0528 | 0.0045 | 0.0156 | 0.0186 | 0.0124 | 0.0447 | 0.0526 |
| Q08 | KU01/FMoE/20260312_000841_895_hparam_P0_Q08.log | success | 20/20 | P0_Q08 | 0.0046 | 0.0167 | 0.0193 | 0.0116 | 0.0446 | 0.0528 | 0.0040 | 0.0155 | 0.0186 | 0.0112 | 0.0421 | 0.0526 |
| Q09 | KU01/FMoE/20260312_000841_916_hparam_P0_Q09.log | success | 20/20 | P0_Q09 | 0.0048 | 0.0179 | 0.0193 | 0.0131 | 0.0521 | 0.0528 | 0.0046 | 0.0163 | 0.0186 | 0.0122 | 0.0472 | 0.0526 |

</details>

<details>
<summary><strong>lastfm0.03</strong> (shown 3/3, best_combo=F01, best_mrr@20=0.4049, best_hr@10=0.4715)</summary>

- rows: 3
- folder_best: combo=F01, best_mrr@20=0.4049, best_hr@10=0.4715, test_mrr@20=0.3832, test_hr@10=0.4508

| combo | experiment | status | trials | run_phase | best_mrr@20 cur | best_mrr@20 run | best_mrr@20 folder | best_hr@10 cur | best_hr@10 run | best_hr@10 folder | test_mrr@20 cur | test_mrr@20 run | test_mrr@20 folder | test_hr@10 cur | test_hr@10 run | test_hr@10 folder |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | LF03/FMoE/20260311_172733_957_hparam_P0_F01.log | success | 20/20 | P0_F01 | 0.4046 | 0.4049 | 0.4049 | 0.4720 | 0.4715 | 0.4715 | 0.3830 | 0.3832 | 0.3832 | 0.4510 | 0.4508 | 0.4508 |
| F02 | LF03/FMoE/20260311_205828_817_hparam_P0_F02.log | success | 20/20 | P0_F02 | 0.4032 | 0.4044 | 0.4049 | 0.4712 | 0.4703 | 0.4715 | 0.3814 | 0.3820 | 0.3832 | 0.4512 | 0.4491 | 0.4508 |
| F03 | LF03/FMoE/20260312_000841_932_hparam_P0_F03.log | running | 15/20 | P0_F03 | 0.3966 | 0.4047 | 0.4049 | 0.4707 | 0.4724 | 0.4715 | 0.3754 | 0.3829 | 0.3832 | 0.4489 | 0.4515 | 0.4508 |

</details>

