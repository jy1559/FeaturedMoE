# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 10.8251 | 0.3294 | 0.3374 | 2.2585 | 0.6704 |
| mid@1 | 6.5814 | 0.9074 | 0.5882 | 1.7857 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0015 | 0.9985 | 0.0003 | 0.9997 | 0.0012 | 0.9988 |
| mid@1 | 0.0133 | 0.9867 | 0.0073 | 0.9927 | 0.0063 | 0.9937 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0055 | 0.9945 | 0.0070 | 0.9931 | 0.0063 | 0.9937 | 0.0027 | 0.9973 | 0.1271 |
| mid@1 | 0.0449 | 0.9561 | 0.0382 | 0.9625 | 0.0552 | 0.9463 | 0.0583 | 0.9434 | 0.2983 |
