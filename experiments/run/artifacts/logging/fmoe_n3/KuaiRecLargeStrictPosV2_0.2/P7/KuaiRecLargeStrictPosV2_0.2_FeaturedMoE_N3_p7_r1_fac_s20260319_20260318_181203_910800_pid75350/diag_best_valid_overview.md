# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3929 | 0.2308 | 0.2692 | 2.0889 | 0.0000 |
| micro@1 | 10.0175 | 0.4449 | 0.3002 | 1.6332 | 0.5423 |
| mid@1 | 9.2948 | 0.5395 | 0.4193 | 1.9472 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0249 | 0.9754 | 0.0125 | 0.9876 | 0.0130 | 0.9871 |
| micro@1 | 0.0294 | 0.9710 | 0.0115 | 0.9885 | 0.0191 | 0.9811 |
| mid@1 | 0.0196 | 0.9806 | 0.0068 | 0.9933 | 0.0109 | 0.9892 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0343 | 0.9663 | 0.0425 | 0.9584 | 0.0443 | 0.9567 | 0.0363 | 0.9643 | 0.1206 |
| micro@1 | 0.0500 | 0.9513 | 0.0581 | 0.9435 | 0.0578 | 0.9439 | 0.0556 | 0.9459 | 0.1906 |
| mid@1 | 0.0332 | 0.9674 | 0.0313 | 0.9692 | 0.0374 | 0.9633 | 0.0422 | 0.9587 | 0.2104 |
