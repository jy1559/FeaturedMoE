# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.8653 | 0.2168 | 0.5354 | 0.7288 | 0.0000 |
| micro@1 | 2.7270 | 0.3164 | 0.4263 | 0.6607 | 0.5584 |
| mid@1 | 2.8662 | 0.2161 | 0.4915 | 0.6873 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0321 | 0.9684 | -0.0000 | 1.0000 | 0.0321 | 0.9684 |
| micro@1 | 0.0179 | 0.9823 | -0.0000 | 1.0000 | 0.0179 | 0.9823 |
| mid@1 | 0.0292 | 0.9712 | 0.0000 | 1.0000 | 0.0292 | 0.9712 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0400 | 0.9608 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1079 |
| micro@1 | 0.0243 | 0.9760 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1063 |
| mid@1 | 0.0378 | 0.9629 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1094 |
