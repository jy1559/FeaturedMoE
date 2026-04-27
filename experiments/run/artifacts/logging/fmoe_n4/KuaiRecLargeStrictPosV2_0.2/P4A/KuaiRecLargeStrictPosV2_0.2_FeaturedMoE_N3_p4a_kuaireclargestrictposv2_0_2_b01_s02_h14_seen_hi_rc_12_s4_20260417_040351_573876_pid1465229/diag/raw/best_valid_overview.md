# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.8177 | 0.7314 | 0.9452 | 2.2192 | 0.0000 |
| micro@1 | 5.7613 | 1.0406 | 0.7051 | 1.7748 | 0.3398 |
| mid@1 | 4.7101 | 1.2441 | 0.9799 | 1.9141 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0065 | 0.9935 | 0.0008 | 0.9992 | 0.0052 | 0.9949 |
| micro@1 | 0.0060 | 0.9941 | 0.0009 | 0.9991 | 0.0048 | 0.9952 |
| mid@1 | 0.0069 | 0.9931 | 0.0007 | 0.9993 | 0.0076 | 0.9925 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0110 | 0.9891 | 0.0250 | 0.9753 | 0.0272 | 0.9732 | 0.0204 | 0.9798 | 0.2782 |
| micro@1 | 0.0206 | 0.9796 | 0.0301 | 0.9703 | 0.0261 | 0.9742 | 0.0156 | 0.9845 | 0.2998 |
| mid@1 | 0.0201 | 0.9801 | 0.0231 | 0.9772 | 0.0275 | 0.9729 | 0.0203 | 0.9799 | 0.4184 |
