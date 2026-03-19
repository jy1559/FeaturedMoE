# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.6678 | 0.3534 | 0.2907 | 2.1009 | 0.0000 |
| micro@1 | 10.4956 | 0.3786 | 0.2034 | 2.0144 | 0.7551 |
| mid@1 | 11.5383 | 0.2000 | 0.2506 | 2.2843 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0237 | 0.9766 | 0.0129 | 0.9872 | 0.0115 | 0.9886 |
| micro@1 | 0.0214 | 0.9788 | 0.0107 | 0.9893 | 0.0099 | 0.9901 |
| mid@1 | 0.0111 | 0.9890 | 0.0043 | 0.9957 | 0.0065 | 0.9935 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0313 | 0.9692 | 0.0405 | 0.9603 | 0.0411 | 0.9597 | 0.0365 | 0.9642 | 0.1429 |
| micro@1 | 0.0270 | 0.9734 | 0.0327 | 0.9678 | 0.0308 | 0.9697 | 0.0336 | 0.9670 | 0.1472 |
| mid@1 | 0.0172 | 0.9830 | 0.0171 | 0.9831 | 0.0203 | 0.9799 | 0.0234 | 0.9769 | 0.1103 |
