# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.7365 | 0.3430 | 0.3239 | 2.1485 | 0.6160 |
| micro@1 | 7.9544 | 0.7132 | 0.2909 | 1.7407 | 0.7544 |
| mid@1 | 9.5488 | 0.5067 | 0.4597 | 2.0449 | 0.5750 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0208 | 0.9794 | 0.0184 | 0.9818 | 0.0028 | 0.9972 |
| micro@1 | 0.0442 | 0.9567 | 0.0370 | 0.9637 | 0.0166 | 0.9837 |
| mid@1 | 0.0236 | 0.9767 | 0.0157 | 0.9844 | 0.0083 | 0.9917 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0356 | 0.9650 | 0.0426 | 0.9583 | 0.0430 | 0.9579 | 0.0381 | 0.9626 | 0.1334 |
| micro@1 | 0.0689 | 0.9334 | 0.0666 | 0.9355 | 0.0742 | 0.9284 | 0.0469 | 0.9542 | 0.1907 |
| mid@1 | 0.0608 | 0.9411 | 0.0522 | 0.9492 | 0.0670 | 0.9352 | 0.0436 | 0.9573 | 0.1899 |
