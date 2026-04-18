# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.9084 | 0.1076 | 0.7570 | 2.0294 | 0.0000 |
| micro@1 | 7.9477 | 0.0811 | 0.2488 | 2.0171 | 0.7311 |
| mid@1 | 7.8338 | 0.1457 | 0.4081 | 1.9856 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0015 | 0.9985 | 0.0013 | 0.9987 | 0.0002 | 0.9998 |
| micro@1 | 0.0013 | 0.9987 | 0.0012 | 0.9988 | 0.0002 | 0.9998 |
| mid@1 | 0.0018 | 0.9982 | 0.0005 | 0.9995 | 0.0012 | 0.9988 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0014 | 0.9986 | 0.0038 | 0.9962 | 0.0035 | 0.9965 | 0.0031 | 0.9969 | 0.1430 |
| micro@1 | 0.0022 | 0.9978 | 0.0059 | 0.9941 | 0.0064 | 0.9936 | 0.0044 | 0.9956 | 0.1432 |
| mid@1 | 0.0151 | 0.9850 | 0.0320 | 0.9685 | 0.0381 | 0.9627 | 0.0299 | 0.9705 | 0.1599 |
