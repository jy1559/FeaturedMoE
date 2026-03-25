# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8051 | 0.1285 | 0.1559 | 2.2209 | 0.0000 |
| micro@1 | 8.9473 | 0.5841 | 0.3522 | 1.9315 | 0.6655 |
| mid@1 | 9.5942 | 0.5008 | 0.2390 | 1.8402 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0096 | 0.9904 | 0.0049 | 0.9951 | 0.0044 | 0.9956 |
| micro@1 | 0.0034 | 0.9966 | 0.0014 | 0.9986 | 0.0022 | 0.9978 |
| mid@1 | 0.0100 | 0.9900 | 0.0042 | 0.9958 | 0.0043 | 0.9957 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0171 | 0.9830 | 0.0308 | 0.9697 | 0.0312 | 0.9692 | 0.0233 | 0.9770 | 0.1040 |
| micro@1 | 0.0096 | 0.9905 | 0.0130 | 0.9871 | 0.0129 | 0.9871 | 0.0104 | 0.9896 | 0.1888 |
| mid@1 | 0.0400 | 0.9608 | 0.0279 | 0.9725 | 0.0350 | 0.9656 | 0.0604 | 0.9414 | 0.1463 |
