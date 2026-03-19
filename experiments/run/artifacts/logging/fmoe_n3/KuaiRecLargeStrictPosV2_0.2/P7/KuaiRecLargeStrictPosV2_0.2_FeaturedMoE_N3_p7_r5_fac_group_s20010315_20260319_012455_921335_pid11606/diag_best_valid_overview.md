# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3717 | 0.2351 | 0.2614 | 2.0811 | 0.0000 |
| micro@1 | 9.0161 | 0.5753 | 0.2882 | 1.4537 | 0.4575 |
| mid@1 | 9.5588 | 0.5054 | 0.3594 | 1.9662 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0243 | 0.9759 | 0.0124 | 0.9877 | 0.0124 | 0.9877 |
| micro@1 | 0.0353 | 0.9653 | 0.0155 | 0.9847 | 0.0206 | 0.9797 |
| mid@1 | 0.0189 | 0.9813 | 0.0064 | 0.9936 | 0.0103 | 0.9898 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0334 | 0.9672 | 0.0411 | 0.9597 | 0.0429 | 0.9580 | 0.0350 | 0.9656 | 0.1182 |
| micro@1 | 0.0596 | 0.9421 | 0.0693 | 0.9331 | 0.0704 | 0.9320 | 0.0661 | 0.9360 | 0.2010 |
| mid@1 | 0.0320 | 0.9685 | 0.0309 | 0.9696 | 0.0375 | 0.9632 | 0.0406 | 0.9602 | 0.1964 |
