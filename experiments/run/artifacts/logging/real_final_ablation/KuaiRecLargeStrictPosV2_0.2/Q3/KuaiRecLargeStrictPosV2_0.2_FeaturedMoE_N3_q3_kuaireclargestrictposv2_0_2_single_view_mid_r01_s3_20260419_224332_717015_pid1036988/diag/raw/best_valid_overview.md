# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| mid@1 | 4.7008 | 1.8041 | 0.5777 | 1.5367 | 0.5586 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.0733 | 0.9293 | 0.0262 | 0.9741 | 0.0539 | 0.9476 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.1056 | 0.8998 | 0.0994 | 0.9054 | 0.0938 | 0.9105 | 0.0746 | 0.9281 | 0.4267 |
