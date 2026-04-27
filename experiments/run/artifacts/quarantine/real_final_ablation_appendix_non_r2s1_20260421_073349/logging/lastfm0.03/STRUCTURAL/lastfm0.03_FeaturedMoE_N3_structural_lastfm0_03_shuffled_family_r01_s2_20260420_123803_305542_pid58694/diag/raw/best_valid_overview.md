# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.9817 | 0.0479 | 0.1913 | 1.9924 | 0.7533 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0101 | 0.9899 | 0.0047 | 0.9953 | 0.0054 | 0.9946 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0128 | 0.9873 | 0.0121 | 0.9880 | 0.0126 | 0.9875 | 0.0118 | 0.9882 | 0.1346 |
