# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9404 | 0.0707 | 0.1919 | 2.4452 | 0.5548 |
| mid@1 | 11.3252 | 0.2441 | 0.2246 | 2.2422 | 0.5986 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0137 | 0.9864 | 0.0039 | 0.9961 | 0.0097 | 0.9903 |
| mid@1 | 0.0248 | 0.9755 | 0.0061 | 0.9939 | 0.0182 | 0.9820 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0133 | 0.9868 | 0.0134 | 0.9867 | 0.0132 | 0.9868 | 0.0132 | 0.9869 | 0.0918 |
| mid@1 | 0.0812 | 0.9220 | 0.0677 | 0.9345 | 0.0519 | 0.9494 | 0.0412 | 0.9596 | 0.1370 |
