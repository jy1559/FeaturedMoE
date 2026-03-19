# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5889 | 0.1883 | 0.1864 | 2.0888 | 0.0000 |
| micro@1 | 9.0690 | 0.5685 | 0.2263 | 1.7915 | 0.6520 |
| mid@1 | 9.6389 | 0.4949 | 0.2762 | 1.9053 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0115 | 0.9886 | 0.0065 | 0.9935 | 0.0048 | 0.9952 |
| micro@1 | 0.0061 | 0.9939 | 0.0038 | 0.9962 | 0.0029 | 0.9971 |
| mid@1 | 0.0137 | 0.9864 | 0.0080 | 0.9920 | 0.0041 | 0.9959 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0252 | 0.9751 | 0.0410 | 0.9598 | 0.0460 | 0.9550 | 0.0290 | 0.9714 | 0.1130 |
| micro@1 | 0.0141 | 0.9860 | 0.0211 | 0.9792 | 0.0207 | 0.9795 | 0.0181 | 0.9821 | 0.1670 |
| mid@1 | 0.0427 | 0.9582 | 0.0322 | 0.9683 | 0.0426 | 0.9583 | 0.0501 | 0.9512 | 0.1631 |
