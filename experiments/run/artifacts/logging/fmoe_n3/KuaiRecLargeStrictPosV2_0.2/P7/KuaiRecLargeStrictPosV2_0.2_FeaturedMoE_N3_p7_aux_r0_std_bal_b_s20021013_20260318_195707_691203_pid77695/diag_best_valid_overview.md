# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.1668 | 0.2732 | 0.2556 | 1.8722 | 0.0000 |
| micro@1 | 11.3379 | 0.2417 | 0.1642 | 1.7042 | 0.5289 |
| mid@1 | 9.6537 | 0.4930 | 0.3045 | 1.6984 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0517 | 0.9497 | 0.0250 | 0.9753 | 0.0322 | 0.9683 |
| micro@1 | 0.0414 | 0.9595 | 0.0194 | 0.9808 | 0.0283 | 0.9721 |
| mid@1 | 0.0441 | 0.9569 | 0.0220 | 0.9782 | 0.0236 | 0.9767 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0645 | 0.9376 | 0.0697 | 0.9327 | 0.0759 | 0.9269 | 0.0671 | 0.9351 | 0.1418 |
| micro@1 | 0.0710 | 0.9315 | 0.0829 | 0.9205 | 0.0886 | 0.9152 | 0.0656 | 0.9366 | 0.1209 |
| mid@1 | 0.0659 | 0.9362 | 0.0639 | 0.9381 | 0.0767 | 0.9262 | 0.0743 | 0.9284 | 0.1852 |
