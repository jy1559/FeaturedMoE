# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.1417 | 0.5592 | 0.3100 | 1.8824 | 0.0000 |
| micro@1 | 10.2177 | 0.4177 | 0.4566 | 2.1550 | 0.5755 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0158 | 0.9844 | 0.0096 | 0.9904 | 0.0059 | 0.9941 |
| micro@1 | 0.0015 | 0.9985 | 0.0002 | 0.9998 | 0.0013 | 0.9987 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0280 | 0.9724 | 0.0468 | 0.9543 | 0.0438 | 0.9572 | 0.0388 | 0.9620 | 0.1673 |
| micro@1 | 0.0046 | 0.9954 | 0.0079 | 0.9921 | 0.0055 | 0.9945 | 0.0063 | 0.9937 | 0.1656 |
