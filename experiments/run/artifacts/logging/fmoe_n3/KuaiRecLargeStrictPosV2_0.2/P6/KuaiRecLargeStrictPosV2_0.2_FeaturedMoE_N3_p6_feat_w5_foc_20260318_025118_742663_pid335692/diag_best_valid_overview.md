# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9297 | 0.1549 | 0.4683 | 0.6761 | 0.0000 |
| micro@1 | 2.8432 | 0.2349 | 0.4877 | 0.8153 | 0.5283 |
| mid@1 | 2.2317 | 0.5868 | 0.6719 | 0.4073 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0413 | 0.9596 | 0.0000 | 1.0000 | 0.0413 | 0.9596 |
| micro@1 | 0.0155 | 0.9846 | -0.0000 | 1.0000 | 0.0155 | 0.9846 |
| mid@1 | 0.0236 | 0.9766 | 0.0000 | 1.0000 | 0.0236 | 0.9766 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1002 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0957 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1588 |
