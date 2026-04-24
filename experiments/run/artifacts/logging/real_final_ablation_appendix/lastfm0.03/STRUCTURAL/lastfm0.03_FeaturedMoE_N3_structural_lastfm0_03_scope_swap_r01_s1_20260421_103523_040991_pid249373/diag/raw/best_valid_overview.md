# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.9829 | 0.0463 | 0.1924 | 1.9896 | 0.7548 |
| mid@1 | 7.9669 | 0.0644 | 0.2036 | 1.9065 | 0.6410 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0106 | 0.9895 | 0.0049 | 0.9951 | 0.0056 | 0.9944 |
| mid@1 | 0.0172 | 0.9829 | 0.0075 | 0.9925 | 0.0097 | 0.9903 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0132 | 0.9869 | 0.0125 | 0.9876 | 0.0130 | 0.9871 | 0.0123 | 0.9878 | 0.1354 |
| mid@1 | 0.0462 | 0.9548 | 0.0212 | 0.9790 | 0.0433 | 0.9576 | 0.0377 | 0.9630 | 0.1435 |
