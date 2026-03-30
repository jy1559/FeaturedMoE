# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.3803 | 0.5285 | 0.4166 | 2.2181 | 0.5463 |
| micro@1 | 6.4274 | 0.9311 | 0.9777 | 1.9738 | 0.0413 |
| mid@1 | 9.2448 | 0.5459 | 0.6819 | 2.2227 | 0.3180 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0086 | 0.9914 | 0.0054 | 0.9946 | 0.0035 | 0.9965 |
| micro@1 | 0.0070 | 0.9931 | 0.0064 | 0.9937 | 0.0006 | 0.9994 |
| mid@1 | 0.0109 | 0.9891 | 0.0035 | 0.9965 | 0.0074 | 0.9926 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0149 | 0.9853 | 0.0243 | 0.9760 | 0.0242 | 0.9761 | 0.0207 | 0.9795 | 0.1535 |
| micro@1 | 0.0116 | 0.9885 | 0.0134 | 0.9867 | 0.0140 | 0.9861 | 0.0086 | 0.9915 | 0.2554 |
| mid@1 | 0.0318 | 0.9687 | 0.0344 | 0.9662 | 0.0331 | 0.9674 | 0.0202 | 0.9800 | 0.1938 |
