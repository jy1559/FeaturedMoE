# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.8713 | 0.2117 | 0.5461 | 0.6795 | 0.0000 |
| micro@1 | 2.7332 | 0.3124 | 0.4908 | 0.6778 | 0.5249 |
| mid@1 | 2.8786 | 0.2053 | 0.4218 | 0.7249 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0356 | 0.9650 | 0.0000 | 1.0000 | 0.0356 | 0.9650 |
| micro@1 | 0.0165 | 0.9837 | -0.0000 | 1.0000 | 0.0165 | 0.9837 |
| mid@1 | 0.0260 | 0.9743 | 0.0000 | 1.0000 | 0.0260 | 0.9743 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0432 | 0.9577 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1106 |
| micro@1 | 0.0219 | 0.9783 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1184 |
| mid@1 | 0.0324 | 0.9682 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0996 |
