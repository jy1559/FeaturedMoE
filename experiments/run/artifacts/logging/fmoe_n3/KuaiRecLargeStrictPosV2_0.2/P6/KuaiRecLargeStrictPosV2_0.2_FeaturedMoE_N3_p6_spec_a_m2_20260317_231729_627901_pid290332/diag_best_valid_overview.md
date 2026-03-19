# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.6137 | 0.1824 | 0.3829 | 2.2705 | 0.0000 |
| micro@1 | 11.7619 | 0.1423 | 0.1480 | 2.1921 | 0.6145 |
| mid@1 | 10.7918 | 0.3346 | 0.2720 | 1.8506 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0204 | 0.9798 | 0.0071 | 0.9929 | 0.0133 | 0.9868 |
| micro@1 | 0.0178 | 0.9824 | 0.0055 | 0.9945 | 0.0121 | 0.9880 |
| mid@1 | 0.0274 | 0.9730 | 0.0142 | 0.9859 | 0.0139 | 0.9862 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0257 | 0.9746 | 0.0285 | 0.9720 | 0.0288 | 0.9716 | 0.0276 | 0.9728 | 0.1121 |
| micro@1 | 0.0278 | 0.9726 | 0.0322 | 0.9683 | 0.0342 | 0.9663 | 0.0264 | 0.9740 | 0.1043 |
| mid@1 | 0.0451 | 0.9559 | 0.0408 | 0.9600 | 0.0428 | 0.9581 | 0.0567 | 0.9449 | 0.1549 |
