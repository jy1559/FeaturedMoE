# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| micro@1 | 7.4070 | 0.7875 | 0.2764 | 1.5776 | 0.6007 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.0886 | 0.9152 | 0.0415 | 0.9593 | 0.0729 | 0.9297 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro@1 | 0.1497 | 0.8609 | 0.1572 | 0.8545 | 0.1527 | 0.8584 | 0.0933 | 0.9109 | 0.1914 |
