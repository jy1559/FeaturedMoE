# fmoe_n Experiment Overview

- generated_at_utc: 2026-03-11T16:32:30.476348+00:00
- include_rule: keep OOM runs and successful runs with valid MRR@20; exclude non-OOM errors and no-metric runs
- matched_end_events: 22
- included_runs: 8
- excluded_non_oom_error_runs: 0
- excluded_no_metric_runs: 14
- summarized_experiments: 2

## Experiment Summary Table

| dataset | experiment | axis | runs | oom | best_mrr@20 | top3_mrr@20 | best_phase | focus_vars | best_log |
|---|---|---|---:|---:|---:|---|---|---|---|
| KuaiRecSmall0.1 | fmoe_n_p0_anchor | hparam | 6 | 0 | 0.001800 | 0.0018/0.0017/0.0016 | P0_Q02 | combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_163143_868_hparam_P0_Q02.log |
| lastfm0.03 | fmoe_n_p0_anchor | hparam | 2 | 0 | 0.109600 | 0.1096/0.0396 | P0_F01 | combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_163143_910_hparam_P0_F01.log |

## Experiment Notes

### KuaiRecSmall0.1 / fmoe_n_p0_anchor

- 실험 설명: FeaturedMoE_N P0 anchor wave sweep.
- 실행 규모: runs=6, oom=0, 기간=2026-03-11T16:31:43.900418+00:00 ~ 2026-03-11T16:32:20.099319+00:00
- 비교 변수: combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate
- 최고 성능: MRR@20=0.001800 (P0_Q02, FeaturedMoE_N_serial_hybrid)
- 최고 설정: fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0017458399536675255
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_163143_868_hparam_P0_Q02.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/KuaiRecSmall0.1_FeaturedMoE_N_p0_q02_20260311_163147_141523_pid163374.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.001800 | P0_Q02 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0017458399536675255 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_163143_868_hparam_P0_Q02.log |
| 2 | 0.001700 | P0_Q05 | fmoe_v2_layout_id=16, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0036225681439254875 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_163207_039_hparam_P0_Q05.log |
| 3 | 0.001600 | P0_Q06 | fmoe_v2_layout_id=16, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0009851411907870486 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/KU01/FMoE/20260311_163207_058_hparam_P0_Q06.log |

### lastfm0.03 / fmoe_n_p0_anchor

- 실험 설명: FeaturedMoE_N P0 anchor wave sweep.
- 실행 규모: runs=2, oom=0, 기간=2026-03-11T16:31:43.959483+00:00 ~ 2026-03-11T16:32:30.250615+00:00
- 비교 변수: combo_id, dataset, family, fmoe_v2_layout_id, expert_scale, feature_encoder_mode, moe_top_k, learning_rate
- 최고 성능: MRR@20=0.109600 (P0_F01, FeaturedMoE_N_serial_plain)
- 최고 설정: fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0034993361142320097
- 최고 로그: /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_163143_910_hparam_P0_F01.log
- 최고 결과 JSON: /workspace/jy1559/FMoE/experiments/run/artifacts/results/fmoe_n/lastfm0.03_FeaturedMoE_N_p0_f01_20260311_163147_027498_pid163376.json
- Top-3 결과:
| rank | mrr@20 | phase | settings | log_file |
|---:|---:|---|---|---|
| 1 | 0.109600 | P0_F01 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0034993361142320097 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_163143_910_hparam_P0_F01.log |
| 2 | 0.039600 | P0_F02 | fmoe_v2_layout_id=7, expert_scale=3, feature_encoder_mode=linear, moe_top_k=0, learning_rate=0.0008729199768337619 | /workspace/jy1559/FMoE/experiments/run/artifacts/logs/fmoe_n/hparam/P0/LF03/FMoE/20260311_163207_077_hparam_P0_F02.log |

