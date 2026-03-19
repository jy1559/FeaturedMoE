# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5765 | 0.1913 | 0.2292 | 2.0120 | 0.0000 |
| micro@1 | 11.1514 | 0.2759 | 0.1682 | 1.8102 | 0.5035 |
| mid@1 | 9.4629 | 0.5178 | 0.3538 | 1.7691 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0396 | 0.9611 | 0.0168 | 0.9833 | 0.0259 | 0.9745 |
| micro@1 | 0.0358 | 0.9649 | 0.0159 | 0.9842 | 0.0233 | 0.9770 |
| mid@1 | 0.0373 | 0.9634 | 0.0197 | 0.9805 | 0.0184 | 0.9817 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0481 | 0.9531 | 0.0525 | 0.9489 | 0.0570 | 0.9446 | 0.0504 | 0.9509 | 0.1188 |
| micro@1 | 0.0629 | 0.9391 | 0.0737 | 0.9290 | 0.0805 | 0.9227 | 0.0548 | 0.9467 | 0.1211 |
| mid@1 | 0.0564 | 0.9451 | 0.0543 | 0.9472 | 0.0655 | 0.9366 | 0.0606 | 0.9412 | 0.2026 |
