# 260318 Layout + Feature Aggregation (KuaiRecLargeStrictPosV2_0.2)

Primary metric: best valid MRR@20
Baseline reference: SASRec best valid MRR@20 = 0.0785

## 1) Layout-related results

### P3 structure options

| option | n_runs | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| S1_flat_standard | 6 | 0.0810 | 0.0813 | +0.0028 |
| S2_factored_group | 5 | 0.0810 | 0.0811 | +0.0026 |
| S3_feature_source | 1 | 0.0809 | 0.0809 | +0.0024 |
| S4_deep_prefix | 1 | 0.0804 | 0.0804 | +0.0019 |

### P1 trial-level family summary (A/L)

| family | n_trials | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| A_anchor_dim_layout | 35 | 0.0789 | 0.0811 | +0.0026 |
| L_layer_layout | 44 | 0.0784 | 0.0807 | +0.0022 |

### P1 top phases by valid_mean

| run_phase | n_trials | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| A05 | 4 | 0.0808 | 0.0811 | +0.0026 |
| A06 | 4 | 0.0804 | 0.0806 | +0.0021 |
| A01 | 5 | 0.0802 | 0.0803 | +0.0018 |
| L01 | 2 | 0.0802 | 0.0806 | +0.0021 |
| L06 | 10 | 0.0801 | 0.0805 | +0.0020 |
| A02 | 7 | 0.0800 | 0.0802 | +0.0017 |

## 2) Feature injection / router source results

### P4 feature-path variants (hidden-only / feature-only / both / injection-only)

| path_type | n_runs | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| hidden_plus_feature_router | 1 | 0.0813 | 0.0813 | +0.0028 |
| feature_only_router | 1 | 0.0807 | 0.0807 | +0.0022 |
| injection_only | 1 | 0.0806 | 0.0806 | +0.0021 |
| hidden_only_router | 1 | 0.0792 | 0.0792 | +0.0007 |

### P6 router x injection combo summary

| combo | n_runs | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| standard+gated | 2 | 0.0807 | 0.0812 | +0.0027 |
| factored+gated | 2 | 0.0804 | 0.0808 | +0.0023 |
| factored+group_gated | 2 | 0.0804 | 0.0808 | +0.0023 |
| standard+group_gated | 3 | 0.0634 | 0.0802 | +0.0017 |

### P6 context summary

| context | n_runs | valid_mean | valid_max | delta_vs_SASRec |
|---|---:|---:|---:|---:|
| X2 | 4 | 0.0801 | 0.0802 | +0.0017 |
| X1 | 5 | 0.0706 | 0.0812 | +0.0027 |

## 3) Rule-based usage signal

run_id keyword scan for rule/teacher/gls: 0 hits

Interpretation: no explicit rule-based variant is visible in the run_id naming of the aggregated KuaiRec table.
