# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.9019 | 0.1289 | 0.3086 | 1.3340 | 0.0000 |
| micro@1 | 4.5774 | 0.5575 | 0.4507 | 1.1485 | 0.5003 |
| mid@1 | 5.3134 | 0.3595 | 0.3429 | 1.2526 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0314 | 0.9691 | 0.0088 | 0.9912 | 0.0248 | 0.9755 |
| micro@1 | 0.0183 | 0.9819 | 0.0037 | 0.9963 | 0.0137 | 0.9864 |
| mid@1 | 0.0231 | 0.9772 | 0.0105 | 0.9896 | 0.0105 | 0.9896 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0369 | 0.9638 | 0.0503 | 0.9509 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0995 |
| micro@1 | 0.0266 | 0.9737 | 0.0313 | 0.9692 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1841 |
| mid@1 | 0.0501 | 0.9512 | 0.0192 | 0.9810 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1243 |
