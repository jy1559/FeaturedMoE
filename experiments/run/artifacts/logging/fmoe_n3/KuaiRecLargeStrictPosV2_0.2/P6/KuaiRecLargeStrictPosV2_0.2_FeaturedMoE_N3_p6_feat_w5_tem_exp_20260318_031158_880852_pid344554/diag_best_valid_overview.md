# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.6078 | 0.2645 | 0.2563 | 1.1273 | 0.0000 |
| micro@1 | 4.6406 | 0.5412 | 0.3547 | 0.9783 | 0.3923 |
| mid@1 | 5.9840 | 0.0517 | 0.2562 | 1.6216 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0339 | 0.9667 | 0.0086 | 0.9914 | 0.0267 | 0.9737 |
| micro@1 | 0.0224 | 0.9779 | 0.0052 | 0.9948 | 0.0177 | 0.9825 |
| mid@1 | 0.0162 | 0.9839 | 0.0052 | 0.9948 | 0.0110 | 0.9891 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0445 | 0.9565 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1172 |
| micro@1 | 0.0325 | 0.9680 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1685 |
| mid@1 | 0.0271 | 0.9733 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0899 |
