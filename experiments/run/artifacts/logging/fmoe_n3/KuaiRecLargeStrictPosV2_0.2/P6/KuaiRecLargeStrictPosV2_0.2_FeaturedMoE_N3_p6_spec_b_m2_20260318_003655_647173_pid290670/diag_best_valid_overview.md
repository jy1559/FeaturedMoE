# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.7687 | 0.1402 | 0.2659 | 2.2443 | 0.0000 |
| micro@1 | 10.5159 | 0.3757 | 0.2609 | 1.9966 | 0.5579 |
| mid@1 | 10.6713 | 0.3529 | 0.4022 | 2.1511 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0160 | 0.9842 | 0.0090 | 0.9910 | 0.0061 | 0.9939 |
| micro@1 | 0.0174 | 0.9828 | 0.0075 | 0.9925 | 0.0092 | 0.9909 |
| mid@1 | 0.0152 | 0.9849 | 0.0057 | 0.9943 | 0.0079 | 0.9921 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0232 | 0.9770 | 0.0313 | 0.9692 | 0.0300 | 0.9704 | 0.0257 | 0.9746 | 0.1036 |
| micro@1 | 0.0271 | 0.9733 | 0.0356 | 0.9650 | 0.0348 | 0.9658 | 0.0359 | 0.9647 | 0.1521 |
| mid@1 | 0.0266 | 0.9738 | 0.0242 | 0.9761 | 0.0256 | 0.9747 | 0.0363 | 0.9644 | 0.1682 |
