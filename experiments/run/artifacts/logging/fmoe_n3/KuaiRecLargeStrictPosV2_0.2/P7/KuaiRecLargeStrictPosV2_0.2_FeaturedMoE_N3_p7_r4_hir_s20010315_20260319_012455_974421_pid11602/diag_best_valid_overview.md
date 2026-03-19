# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4997 | 0.2086 | 0.1986 | 1.9441 | 0.0000 |
| micro@1 | 9.3974 | 0.5263 | 0.2222 | 1.1864 | 0.4941 |
| mid@1 | 11.3024 | 0.2484 | 0.2725 | 2.0155 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0497 | 0.9515 | 0.0296 | 0.9708 | 0.0215 | 0.9787 |
| micro@1 | 0.0585 | 0.9432 | 0.0359 | 0.9648 | 0.0254 | 0.9750 |
| mid@1 | 0.0321 | 0.9684 | 0.0159 | 0.9842 | 0.0178 | 0.9824 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0608 | 0.9410 | 0.0673 | 0.9349 | 0.0721 | 0.9305 | 0.0633 | 0.9386 | 0.1128 |
| micro@1 | 0.0973 | 0.9073 | 0.1147 | 0.8917 | 0.1188 | 0.8880 | 0.0945 | 0.9099 | 0.1924 |
| mid@1 | 0.0478 | 0.9534 | 0.0475 | 0.9536 | 0.0550 | 0.9465 | 0.0527 | 0.9487 | 0.1242 |
