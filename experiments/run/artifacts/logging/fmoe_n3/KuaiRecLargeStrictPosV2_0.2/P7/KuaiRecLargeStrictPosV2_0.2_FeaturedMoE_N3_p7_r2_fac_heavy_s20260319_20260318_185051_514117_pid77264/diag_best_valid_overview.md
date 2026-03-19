# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.2990 | 0.2491 | 0.1674 | 1.6883 | 0.0000 |
| micro@1 | 9.8758 | 0.4638 | 0.2588 | 1.5009 | 0.5726 |
| mid@1 | 8.5431 | 0.6361 | 0.1935 | 1.5272 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0487 | 0.9525 | 0.0443 | 0.9566 | 0.0055 | 0.9945 |
| micro@1 | 0.0472 | 0.9539 | 0.0421 | 0.9587 | 0.0076 | 0.9924 |
| mid@1 | 0.0331 | 0.9674 | 0.0264 | 0.9740 | 0.0079 | 0.9922 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0619 | 0.9400 | 0.0732 | 0.9294 | 0.0791 | 0.9239 | 0.0661 | 0.9361 | 0.1199 |
| micro@1 | 0.0773 | 0.9256 | 0.0926 | 0.9115 | 0.0945 | 0.9098 | 0.0814 | 0.9218 | 0.1533 |
| mid@1 | 0.0743 | 0.9284 | 0.0680 | 0.9342 | 0.0760 | 0.9268 | 0.0659 | 0.9362 | 0.1746 |
