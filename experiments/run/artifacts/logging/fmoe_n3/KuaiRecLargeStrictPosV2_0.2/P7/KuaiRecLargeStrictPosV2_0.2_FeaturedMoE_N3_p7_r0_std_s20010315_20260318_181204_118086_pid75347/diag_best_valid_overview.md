# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4957 | 0.2095 | 0.2153 | 1.9892 | 0.0000 |
| micro@1 | 11.4986 | 0.2088 | 0.1398 | 1.9286 | 0.5293 |
| mid@1 | 9.6403 | 0.4947 | 0.3317 | 1.7289 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0378 | 0.9629 | 0.0157 | 0.9844 | 0.0253 | 0.9751 |
| micro@1 | 0.0332 | 0.9673 | 0.0129 | 0.9872 | 0.0212 | 0.9790 |
| mid@1 | 0.0333 | 0.9672 | 0.0178 | 0.9824 | 0.0173 | 0.9829 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0459 | 0.9552 | 0.0508 | 0.9505 | 0.0553 | 0.9462 | 0.0490 | 0.9522 | 0.1122 |
| micro@1 | 0.0599 | 0.9419 | 0.0699 | 0.9325 | 0.0776 | 0.9254 | 0.0495 | 0.9517 | 0.1099 |
| mid@1 | 0.0509 | 0.9503 | 0.0492 | 0.9520 | 0.0590 | 0.9427 | 0.0569 | 0.9446 | 0.1898 |
