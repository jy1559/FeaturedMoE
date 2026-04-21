# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 15.7871 | 0.1161 | 0.1126 | 2.5211 | 0.6241 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0597 | 0.9420 | 0.0140 | 0.9861 | 0.0456 | 0.9555 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0761 | 0.9267 | 0.0770 | 0.9259 | 0.0756 | 0.9272 | 0.0674 | 0.9348 | 0.0816 |
