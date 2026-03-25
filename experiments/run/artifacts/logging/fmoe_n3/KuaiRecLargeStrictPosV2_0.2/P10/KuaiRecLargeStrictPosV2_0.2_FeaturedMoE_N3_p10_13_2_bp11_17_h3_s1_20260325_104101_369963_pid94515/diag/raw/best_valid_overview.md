# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8000 | 0.1302 | 0.2115 | 2.1493 | 0.0000 |
| micro@1 | 6.8417 | 0.8683 | 0.4317 | 1.7091 | 0.6399 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0118 | 0.9883 | 0.0073 | 0.9927 | 0.0039 | 0.9961 |
| micro@1 | 0.0029 | 0.9971 | 0.0012 | 0.9988 | 0.0019 | 0.9981 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0215 | 0.9787 | 0.0370 | 0.9637 | 0.0372 | 0.9635 | 0.0284 | 0.9720 | 0.1024 |
| micro@1 | 0.0096 | 0.9905 | 0.0133 | 0.9868 | 0.0113 | 0.9888 | 0.0124 | 0.9877 | 0.2433 |
