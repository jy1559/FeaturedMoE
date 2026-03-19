# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.4132 | 0.3904 | 0.4434 | 2.3278 | 0.0000 |
| micro@1 | 10.0785 | 0.4366 | 0.4092 | 2.2166 | 0.4893 |
| mid@1 | 9.9575 | 0.4529 | 0.6038 | 2.2755 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0165 | 0.9836 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| micro@1 | 0.0118 | 0.9882 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mid@1 | 0.0158 | 0.9843 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1508 |
| micro@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1618 |
| mid@1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1942 |
