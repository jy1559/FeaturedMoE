# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.8869 | 0.4623 | 0.2519 | 1.7305 | 0.0000 |
| micro@1 | 7.9475 | 0.7141 | 0.3603 | 1.6465 | 0.6280 |
| mid@1 | 8.4178 | 0.6523 | 0.3022 | 1.6334 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0369 | 0.9638 | 0.0100 | 0.9901 | 0.0437 | 0.9573 |
| micro@1 | 0.0054 | 0.9946 | 0.0017 | 0.9983 | 0.0044 | 0.9956 |
| mid@1 | 0.0221 | 0.9781 | 0.0065 | 0.9936 | 0.0222 | 0.9781 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0582 | 0.9435 | 0.0833 | 0.9200 | 0.0948 | 0.9096 | 0.0733 | 0.9294 | 0.1481 |
| micro@1 | 0.0110 | 0.9890 | 0.0169 | 0.9833 | 0.0139 | 0.9862 | 0.0174 | 0.9827 | 0.1968 |
| mid@1 | 0.0560 | 0.9455 | 0.0556 | 0.9459 | 0.0592 | 0.9425 | 0.0797 | 0.9234 | 0.1890 |
