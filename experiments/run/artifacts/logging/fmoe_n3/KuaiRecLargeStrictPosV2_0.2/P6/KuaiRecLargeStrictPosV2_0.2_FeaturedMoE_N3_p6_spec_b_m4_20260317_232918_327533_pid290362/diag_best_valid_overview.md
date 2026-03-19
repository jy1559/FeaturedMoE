# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0531 | 0.2927 | 0.3070 | 2.0805 | 0.0000 |
| micro@1 | 8.1154 | 0.6919 | 0.3228 | 1.5232 | 0.4567 |
| mid@1 | 9.8294 | 0.4699 | 0.3244 | 2.0323 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0224 | 0.9778 | 0.0123 | 0.9878 | 0.0100 | 0.9901 |
| micro@1 | 0.0283 | 0.9721 | 0.0146 | 0.9855 | 0.0139 | 0.9862 |
| mid@1 | 0.0173 | 0.9828 | 0.0060 | 0.9941 | 0.0088 | 0.9913 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0318 | 0.9687 | 0.0401 | 0.9607 | 0.0380 | 0.9627 | 0.0334 | 0.9672 | 0.1334 |
| micro@1 | 0.0423 | 0.9586 | 0.0510 | 0.9503 | 0.0457 | 0.9553 | 0.0625 | 0.9394 | 0.1954 |
| mid@1 | 0.0298 | 0.9706 | 0.0280 | 0.9724 | 0.0300 | 0.9704 | 0.0435 | 0.9574 | 0.1691 |
