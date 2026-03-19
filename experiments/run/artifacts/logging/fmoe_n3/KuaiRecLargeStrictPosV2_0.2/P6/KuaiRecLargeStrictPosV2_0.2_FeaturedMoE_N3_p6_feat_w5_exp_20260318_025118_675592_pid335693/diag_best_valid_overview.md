# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9770 | 0.0879 | 0.4013 | 0.5487 | 0.0000 |
| micro@1 | 2.9686 | 0.1029 | 0.4411 | 0.9195 | 0.5206 |
| mid@1 | 2.9477 | 0.1333 | 0.4377 | 0.7960 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0338 | 0.9668 | 0.0000 | 1.0000 | 0.0338 | 0.9668 |
| micro@1 | 0.0166 | 0.9835 | -0.0000 | 1.0000 | 0.0166 | 0.9835 |
| mid@1 | 0.0200 | 0.9802 | 0.0000 | 1.0000 | 0.0200 | 0.9802 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1066 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0919 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0987 |
