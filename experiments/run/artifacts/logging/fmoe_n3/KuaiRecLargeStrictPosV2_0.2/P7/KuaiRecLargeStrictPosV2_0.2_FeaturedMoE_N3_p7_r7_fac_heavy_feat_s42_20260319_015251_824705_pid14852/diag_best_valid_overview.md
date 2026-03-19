# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.7614 | 0.1424 | 0.1817 | 2.0902 | 0.0000 |
| micro@1 | 8.9761 | 0.5804 | 0.2338 | 1.7651 | 0.6549 |
| mid@1 | 9.6213 | 0.4972 | 0.2742 | 1.8884 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0116 | 0.9885 | 0.0068 | 0.9932 | 0.0046 | 0.9954 |
| micro@1 | 0.0073 | 0.9927 | 0.0045 | 0.9955 | 0.0035 | 0.9965 |
| mid@1 | 0.0142 | 0.9859 | 0.0079 | 0.9921 | 0.0046 | 0.9954 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0256 | 0.9748 | 0.0421 | 0.9588 | 0.0474 | 0.9537 | 0.0300 | 0.9705 | 0.1039 |
| micro@1 | 0.0169 | 0.9832 | 0.0237 | 0.9766 | 0.0238 | 0.9765 | 0.0182 | 0.9819 | 0.1767 |
| mid@1 | 0.0441 | 0.9568 | 0.0334 | 0.9671 | 0.0433 | 0.9576 | 0.0510 | 0.9503 | 0.1666 |
