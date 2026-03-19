# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9645 | 0.1094 | 0.4597 | 0.7225 | 0.0000 |
| micro@1 | 2.9912 | 0.0542 | 0.3650 | 0.8302 | 0.4336 |
| mid@1 | 2.7976 | 0.2690 | 0.5166 | 0.4157 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0254 | 0.9749 | 0.0000 | 1.0000 | 0.0254 | 0.9749 |
| micro@1 | 0.0155 | 0.9846 | -0.0000 | 1.0000 | 0.0155 | 0.9846 |
| mid@1 | 0.0205 | 0.9797 | 0.0000 | 1.0000 | 0.0205 | 0.9797 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0898 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0892 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1196 |
