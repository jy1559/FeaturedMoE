# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9128 | 0.3156 | 0.4667 | 2.3227 | 0.5134 |
| micro@1 | 10.7604 | 0.3394 | 0.2993 | 2.2464 | 0.7529 |
| mid@1 | 9.1960 | 0.5522 | 0.6687 | 2.1935 | 0.3650 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0127 | 0.9874 | 0.0121 | 0.9880 | 0.0006 | 0.9994 |
| micro@1 | 0.0140 | 0.9861 | 0.0132 | 0.9869 | 0.0009 | 0.9991 |
| mid@1 | 0.0116 | 0.9885 | 0.0101 | 0.9899 | 0.0013 | 0.9987 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0196 | 0.9806 | 0.0275 | 0.9729 | 0.0234 | 0.9769 | 0.0174 | 0.9827 | 0.1021 |
| micro@1 | 0.0168 | 0.9834 | 0.0250 | 0.9753 | 0.0165 | 0.9837 | 0.0204 | 0.9798 | 0.0990 |
| mid@1 | 0.0277 | 0.9727 | 0.0360 | 0.9647 | 0.0194 | 0.9808 | 0.0392 | 0.9616 | 0.1696 |
