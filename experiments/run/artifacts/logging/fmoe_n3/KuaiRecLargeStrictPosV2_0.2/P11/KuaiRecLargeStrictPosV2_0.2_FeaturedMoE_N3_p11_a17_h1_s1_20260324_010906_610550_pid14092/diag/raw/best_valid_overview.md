# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 4.5893 | 1.2707 | 0.8692 | 1.6630 | 0.0000 |
| micro@1 | 4.2128 | 1.3596 | 0.9693 | 1.7613 | 0.0610 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0147 | 0.9854 | 0.0052 | 0.9948 | 0.0135 | 0.9866 |
| micro@1 | 0.0027 | 0.9973 | 0.0006 | 0.9994 | 0.0027 | 0.9973 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0224 | 0.9778 | 0.0385 | 0.9623 | 0.0353 | 0.9653 | 0.0354 | 0.9653 | 0.4119 |
| micro@1 | 0.0089 | 0.9912 | 0.0114 | 0.9887 | 0.0118 | 0.9883 | 0.0030 | 0.9970 | 0.4458 |
