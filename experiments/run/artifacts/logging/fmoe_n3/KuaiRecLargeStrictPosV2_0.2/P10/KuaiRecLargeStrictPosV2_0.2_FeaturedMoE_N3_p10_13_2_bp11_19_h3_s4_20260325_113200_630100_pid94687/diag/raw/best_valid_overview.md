# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 8.8392 | 0.5980 | 0.4683 | 1.9235 | 0.0000 |
| micro@1 | 9.5955 | 0.5006 | 0.4888 | 2.0806 | 0.4901 |
| mid@1 | 4.0988 | 1.3884 | 0.9409 | 1.6109 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0183 | 0.9819 | 0.0071 | 0.9930 | 0.0108 | 0.9892 |
| micro@1 | 0.0025 | 0.9976 | 0.0008 | 0.9992 | 0.0018 | 0.9982 |
| mid@1 | 0.0138 | 0.9863 | 0.0051 | 0.9949 | 0.0105 | 0.9896 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0263 | 0.9741 | 0.0524 | 0.9489 | 0.0476 | 0.9535 | 0.0502 | 0.9510 | 0.2372 |
| micro@1 | 0.0053 | 0.9947 | 0.0110 | 0.9891 | 0.0097 | 0.9904 | 0.0097 | 0.9904 | 0.1821 |
| mid@1 | 0.0448 | 0.9562 | 0.0393 | 0.9614 | 0.0547 | 0.9468 | 0.0322 | 0.9683 | 0.4594 |
