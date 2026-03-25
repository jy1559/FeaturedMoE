# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8017 | 0.1296 | 0.2129 | 2.1625 | 0.0000 |
| micro@1 | 6.7975 | 0.8748 | 0.4357 | 1.7132 | 0.6380 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0113 | 0.9888 | 0.0071 | 0.9929 | 0.0036 | 0.9964 |
| micro@1 | 0.0028 | 0.9972 | 0.0012 | 0.9988 | 0.0018 | 0.9982 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0206 | 0.9796 | 0.0354 | 0.9653 | 0.0357 | 0.9649 | 0.0270 | 0.9734 | 0.1021 |
| micro@1 | 0.0092 | 0.9908 | 0.0129 | 0.9872 | 0.0108 | 0.9892 | 0.0122 | 0.9879 | 0.2445 |
