# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3210 | 0.2449 | 0.1702 | 1.7131 | 0.0000 |
| micro@1 | 9.6628 | 0.4918 | 0.2697 | 1.5110 | 0.5727 |
| mid@1 | 8.6218 | 0.6260 | 0.1910 | 1.5507 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0477 | 0.9534 | 0.0437 | 0.9572 | 0.0051 | 0.9949 |
| micro@1 | 0.0467 | 0.9544 | 0.0419 | 0.9590 | 0.0069 | 0.9931 |
| mid@1 | 0.0327 | 0.9678 | 0.0263 | 0.9740 | 0.0074 | 0.9927 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0605 | 0.9413 | 0.0710 | 0.9315 | 0.0770 | 0.9259 | 0.0643 | 0.9377 | 0.1193 |
| micro@1 | 0.0761 | 0.9267 | 0.0913 | 0.9127 | 0.0932 | 0.9110 | 0.0799 | 0.9232 | 0.1555 |
| mid@1 | 0.0727 | 0.9299 | 0.0666 | 0.9355 | 0.0748 | 0.9279 | 0.0649 | 0.9371 | 0.1732 |
