# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5995 | 0.1858 | 0.2479 | 2.0693 | 0.0000 |
| micro@1 | 10.4548 | 0.3844 | 0.2476 | 1.8075 | 0.5252 |
| mid@1 | 9.8991 | 0.4607 | 0.3535 | 1.9589 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0229 | 0.9774 | 0.0114 | 0.9886 | 0.0113 | 0.9887 |
| micro@1 | 0.0265 | 0.9738 | 0.0110 | 0.9891 | 0.0162 | 0.9840 |
| mid@1 | 0.0201 | 0.9801 | 0.0070 | 0.9930 | 0.0111 | 0.9890 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0320 | 0.9685 | 0.0398 | 0.9610 | 0.0416 | 0.9593 | 0.0333 | 0.9672 | 0.1104 |
| micro@1 | 0.0446 | 0.9564 | 0.0534 | 0.9480 | 0.0543 | 0.9472 | 0.0517 | 0.9496 | 0.1547 |
| mid@1 | 0.0340 | 0.9666 | 0.0332 | 0.9673 | 0.0414 | 0.9594 | 0.0417 | 0.9592 | 0.1879 |
