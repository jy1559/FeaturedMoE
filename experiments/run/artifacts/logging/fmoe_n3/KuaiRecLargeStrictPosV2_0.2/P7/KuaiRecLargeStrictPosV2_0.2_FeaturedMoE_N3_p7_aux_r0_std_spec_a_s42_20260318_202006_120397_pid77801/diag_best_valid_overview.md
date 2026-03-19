# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2433 | 0.2594 | 0.2044 | 1.8300 | 0.0000 |
| micro@1 | 11.3066 | 0.2476 | 0.1886 | 1.8887 | 0.4950 |
| mid@1 | 9.0059 | 0.5766 | 0.3278 | 1.5881 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0530 | 0.9484 | 0.0258 | 0.9745 | 0.0338 | 0.9668 |
| micro@1 | 0.0348 | 0.9658 | 0.0150 | 0.9851 | 0.0227 | 0.9776 |
| mid@1 | 0.0471 | 0.9540 | 0.0256 | 0.9747 | 0.0240 | 0.9763 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0659 | 0.9362 | 0.0720 | 0.9305 | 0.0781 | 0.9249 | 0.0691 | 0.9332 | 0.1292 |
| micro@1 | 0.0618 | 0.9401 | 0.0727 | 0.9299 | 0.0770 | 0.9259 | 0.0539 | 0.9475 | 0.1247 |
| mid@1 | 0.0712 | 0.9313 | 0.0681 | 0.9342 | 0.0803 | 0.9229 | 0.0825 | 0.9208 | 0.2120 |
