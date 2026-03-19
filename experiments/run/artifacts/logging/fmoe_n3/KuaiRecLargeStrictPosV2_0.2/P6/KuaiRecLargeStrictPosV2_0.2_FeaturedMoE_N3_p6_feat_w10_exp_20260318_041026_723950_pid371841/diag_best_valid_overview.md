# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9563 | 0.1216 | 0.4066 | 0.5455 | 0.0000 |
| micro@1 | 2.9768 | 0.0882 | 0.3945 | 0.8975 | 0.5215 |
| mid@1 | 2.9547 | 0.1238 | 0.4261 | 0.8521 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0258 | 0.9745 | 0.0000 | 1.0000 | 0.0258 | 0.9745 |
| micro@1 | 0.0173 | 0.9828 | -0.0000 | 1.0000 | 0.0173 | 0.9828 |
| mid@1 | 0.0173 | 0.9829 | 0.0000 | 1.0000 | 0.0173 | 0.9829 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1113 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0901 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0953 |
