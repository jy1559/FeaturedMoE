# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.9592 | 0.1174 | 0.4926 | 0.7753 | 0.0000 |
| micro@1 | 2.9504 | 0.1296 | 0.4200 | 0.9069 | 0.4834 |
| mid@1 | 2.5607 | 0.4142 | 0.6330 | 0.5386 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0323 | 0.9683 | 0.0000 | 1.0000 | 0.0323 | 0.9683 |
| micro@1 | 0.0120 | 0.9881 | -0.0000 | 1.0000 | 0.0120 | 0.9881 |
| mid@1 | 0.0224 | 0.9779 | 0.0000 | 1.0000 | 0.0224 | 0.9779 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0975 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1018 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1389 |
