# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8078 | 0.1276 | 0.2169 | 2.2381 | 0.0000 |
| micro@1 | 9.1495 | 0.5582 | 0.3291 | 1.9596 | 0.6725 |
| mid@1 | 9.6026 | 0.4997 | 0.2335 | 1.8354 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0091 | 0.9910 | 0.0042 | 0.9958 | 0.0046 | 0.9955 |
| micro@1 | 0.0031 | 0.9969 | 0.0013 | 0.9987 | 0.0020 | 0.9980 |
| mid@1 | 0.0097 | 0.9904 | 0.0032 | 0.9968 | 0.0046 | 0.9955 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0161 | 0.9840 | 0.0289 | 0.9715 | 0.0289 | 0.9715 | 0.0219 | 0.9783 | 0.1104 |
| micro@1 | 0.0085 | 0.9915 | 0.0118 | 0.9882 | 0.0117 | 0.9883 | 0.0102 | 0.9898 | 0.1798 |
| mid@1 | 0.0409 | 0.9599 | 0.0277 | 0.9726 | 0.0370 | 0.9637 | 0.0627 | 0.9392 | 0.1454 |
