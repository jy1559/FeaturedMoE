# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 8.8998 | 0.5902 | 0.2248 | 1.2188 | 0.0000 |
| micro@1 | 9.0201 | 0.5748 | 0.2254 | 1.3342 | 0.5284 |
| mid@1 | 8.9960 | 0.5779 | 0.2060 | 1.3669 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1131 | 0.8931 | 0.0549 | 0.9465 | 0.1092 | 0.8967 |
| micro@1 | 0.0787 | 0.9243 | 0.0310 | 0.9695 | 0.0691 | 0.9332 |
| mid@1 | 0.0787 | 0.9243 | 0.0333 | 0.9672 | 0.0942 | 0.9104 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1351 | 0.8736 | 0.1448 | 0.8652 | 0.1578 | 0.8540 | 0.1407 | 0.8687 | 0.1773 |
| micro@1 | 0.1125 | 0.8936 | 0.1275 | 0.8803 | 0.1349 | 0.8738 | 0.0918 | 0.9123 | 0.1868 |
| mid@1 | 0.0969 | 0.9076 | 0.0987 | 0.9060 | 0.1262 | 0.8814 | 0.0921 | 0.9120 | 0.1587 |
