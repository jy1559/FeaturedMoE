# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.8749 | 0.2086 | 0.5532 | 0.7169 | 0.0000 |
| micro@1 | 2.7931 | 0.2722 | 0.4983 | 0.7563 | 0.5422 |
| mid@1 | 2.9188 | 0.1668 | 0.4233 | 0.7730 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0332 | 0.9673 | 0.0000 | 1.0000 | 0.0332 | 0.9673 |
| micro@1 | 0.0141 | 0.9860 | -0.0000 | 1.0000 | 0.0141 | 0.9860 |
| mid@1 | 0.0231 | 0.9772 | 0.0000 | 1.0000 | 0.0231 | 0.9772 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0404 | 0.9604 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1097 |
| micro@1 | 0.0185 | 0.9817 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1166 |
| mid@1 | 0.0288 | 0.9716 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0967 |
