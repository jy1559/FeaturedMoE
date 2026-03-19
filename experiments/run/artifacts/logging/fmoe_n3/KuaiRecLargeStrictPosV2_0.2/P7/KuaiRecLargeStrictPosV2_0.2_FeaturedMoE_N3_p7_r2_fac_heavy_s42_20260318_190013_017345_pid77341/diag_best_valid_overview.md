# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2812 | 0.2524 | 0.1588 | 1.6781 | 0.0000 |
| micro@1 | 9.7937 | 0.4746 | 0.2542 | 1.4683 | 0.5654 |
| mid@1 | 8.5004 | 0.6416 | 0.1921 | 1.4956 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0491 | 0.9521 | 0.0446 | 0.9564 | 0.0058 | 0.9942 |
| micro@1 | 0.0463 | 0.9547 | 0.0412 | 0.9597 | 0.0083 | 0.9918 |
| mid@1 | 0.0340 | 0.9665 | 0.0269 | 0.9735 | 0.0083 | 0.9917 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0625 | 0.9394 | 0.0744 | 0.9283 | 0.0803 | 0.9228 | 0.0671 | 0.9351 | 0.1202 |
| micro@1 | 0.0756 | 0.9272 | 0.0916 | 0.9125 | 0.0938 | 0.9105 | 0.0806 | 0.9226 | 0.1542 |
| mid@1 | 0.0758 | 0.9270 | 0.0694 | 0.9330 | 0.0772 | 0.9257 | 0.0669 | 0.9353 | 0.1782 |
