# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| mid@1 | 15.7309 | 0.1308 | 0.1136 | 2.5156 | 0.6774 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0585 | 0.9431 | 0.0159 | 0.9842 | 0.0425 | 0.9584 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0720 | 0.9305 | 0.0719 | 0.9307 | 0.0675 | 0.9347 | 0.0628 | 0.9392 | 0.0800 |
