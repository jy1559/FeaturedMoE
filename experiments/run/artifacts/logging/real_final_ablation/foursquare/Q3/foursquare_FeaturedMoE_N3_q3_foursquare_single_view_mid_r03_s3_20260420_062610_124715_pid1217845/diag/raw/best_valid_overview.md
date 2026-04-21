# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| mid@1 | 15.5861 | 0.1630 | 0.1149 | 2.4223 | 0.7126 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0727 | 0.9298 | 0.0214 | 0.9788 | 0.0513 | 0.9500 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0930 | 0.9112 | 0.0955 | 0.9089 | 0.0878 | 0.9159 | 0.0768 | 0.9260 | 0.0884 |
