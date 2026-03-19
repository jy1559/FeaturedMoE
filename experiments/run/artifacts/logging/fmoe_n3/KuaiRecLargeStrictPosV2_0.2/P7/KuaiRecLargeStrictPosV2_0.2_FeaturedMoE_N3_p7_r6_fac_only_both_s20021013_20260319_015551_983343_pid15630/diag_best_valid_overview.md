# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0944 | 0.2857 | 0.3870 | 1.9018 | 0.0000 |
| micro@1 | 11.5083 | 0.2067 | 0.3224 | 1.6876 | 0.4079 |
| mid@1 | 9.5585 | 0.5054 | 0.6564 | 2.0970 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0330 | 0.9675 | 0.0330 | 0.9675 | 0.0000 | 1.0000 |
| micro@1 | 0.0366 | 0.9641 | 0.0366 | 0.9641 | 0.0000 | 1.0000 |
| mid@1 | 0.0106 | 0.9895 | 0.0106 | 0.9895 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0433 | 0.9577 | 0.0487 | 0.9524 | 0.0520 | 0.9493 | 0.0410 | 0.9599 | 0.1109 |
| micro@1 | 0.0588 | 0.9429 | 0.0702 | 0.9322 | 0.0726 | 0.9300 | 0.0642 | 0.9378 | 0.1088 |
| mid@1 | 0.0177 | 0.9824 | 0.0158 | 0.9843 | 0.0164 | 0.9837 | 0.0271 | 0.9733 | 0.1598 |
