# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.2932 | 0.5397 | 0.2480 | 1.3446 | 0.0000 |
| micro@1 | 7.5622 | 0.7661 | 0.2411 | 1.0885 | 0.4600 |
| mid@1 | 6.9746 | 0.8488 | 0.4042 | 1.3812 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0665 | 0.9356 | 0.0277 | 0.9726 | 0.0910 | 0.9132 |
| micro@1 | 0.0424 | 0.9585 | 0.0194 | 0.9808 | 0.0376 | 0.9631 |
| mid@1 | 0.0425 | 0.9584 | 0.0159 | 0.9842 | 0.0665 | 0.9360 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0838 | 0.9196 | 0.0961 | 0.9084 | 0.1027 | 0.9024 | 0.0856 | 0.9179 | 0.1628 |
| micro@1 | 0.0743 | 0.9284 | 0.0879 | 0.9159 | 0.0934 | 0.9109 | 0.0691 | 0.9332 | 0.2374 |
| mid@1 | 0.0687 | 0.9336 | 0.0640 | 0.9380 | 0.0657 | 0.9364 | 0.1009 | 0.9041 | 0.2869 |
