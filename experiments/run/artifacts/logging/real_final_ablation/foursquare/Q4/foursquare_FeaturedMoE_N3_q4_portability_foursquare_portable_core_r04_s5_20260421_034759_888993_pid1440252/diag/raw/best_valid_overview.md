# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 8.9212 | 0.0940 | 0.1543 | 1.9932 | 0.6540 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0424 | 0.9585 | 0.0146 | 0.9855 | 0.0274 | 0.9730 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0479 | 0.9532 | 0.0000 | 1.0000 | 0.0583 | 0.9434 | 0.0000 | 0.0000 | 0.0955 |
