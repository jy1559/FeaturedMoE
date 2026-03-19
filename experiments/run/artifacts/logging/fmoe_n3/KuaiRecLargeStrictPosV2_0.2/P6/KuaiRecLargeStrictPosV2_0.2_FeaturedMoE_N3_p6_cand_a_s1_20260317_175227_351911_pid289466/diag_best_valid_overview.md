# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3916 | 0.2311 | 0.2285 | 1.9735 | 0.0000 |
| micro@1 | 11.3203 | 0.2450 | 0.1895 | 1.9494 | 0.5104 |
| mid@1 | 9.5642 | 0.5047 | 0.3170 | 1.6471 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0365 | 0.9641 | 0.0145 | 0.9856 | 0.0251 | 0.9752 |
| micro@1 | 0.0296 | 0.9708 | 0.0116 | 0.9885 | 0.0189 | 0.9813 |
| mid@1 | 0.0362 | 0.9645 | 0.0192 | 0.9810 | 0.0184 | 0.9818 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0441 | 0.9569 | 0.0490 | 0.9522 | 0.0534 | 0.9480 | 0.0474 | 0.9537 | 0.1183 |
| micro@1 | 0.0529 | 0.9484 | 0.0619 | 0.9400 | 0.0687 | 0.9336 | 0.0447 | 0.9563 | 0.1263 |
| mid@1 | 0.0567 | 0.9448 | 0.0534 | 0.9480 | 0.0629 | 0.9391 | 0.0639 | 0.9381 | 0.1958 |
