# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 15.5717 | 0.1658 | 0.1230 | 2.3107 | 0.6469 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.1064 | 0.8991 | 0.0392 | 0.9615 | 0.0697 | 0.9328 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.1278 | 0.8800 | 0.1232 | 0.8841 | 0.1233 | 0.8840 | 0.1211 | 0.8860 | 0.0850 |
