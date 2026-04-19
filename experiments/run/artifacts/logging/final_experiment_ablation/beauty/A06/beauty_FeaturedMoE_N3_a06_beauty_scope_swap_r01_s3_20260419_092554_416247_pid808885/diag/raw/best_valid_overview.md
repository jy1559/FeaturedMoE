# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.7442 | 0.2614 | 0.4433 | 1.2583 | 0.4897 |
| mid@1 | 3.9879 | 0.0550 | 0.3584 | 1.2521 | 0.5077 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0106 | 0.9895 | 0.0106 | 0.9895 | 0.0000 | 1.0000 |
| mid@1 | 0.0164 | 0.9837 | 0.0164 | 0.9837 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0149 | 0.9853 | 0.0211 | 0.9791 | 0.0212 | 0.9790 | 0.0198 | 0.9804 | 0.3319 |
| mid@1 | 0.0225 | 0.9778 | 0.0285 | 0.9719 | 0.0248 | 0.9755 | 0.0168 | 0.9833 | 0.2790 |
