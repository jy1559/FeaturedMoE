# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9389 | 0.3114 | 0.3851 | 1.8921 | 0.0000 |
| micro@1 | 11.4569 | 0.2177 | 0.3173 | 1.6986 | 0.4003 |
| mid@1 | 9.5915 | 0.5011 | 0.6659 | 2.1123 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0321 | 0.9684 | 0.0321 | 0.9684 | 0.0000 | 1.0000 |
| micro@1 | 0.0358 | 0.9648 | 0.0358 | 0.9648 | 0.0000 | 1.0000 |
| mid@1 | 0.0105 | 0.9896 | 0.0105 | 0.9896 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0421 | 0.9587 | 0.0476 | 0.9535 | 0.0507 | 0.9506 | 0.0401 | 0.9607 | 0.1120 |
| micro@1 | 0.0578 | 0.9438 | 0.0690 | 0.9334 | 0.0714 | 0.9311 | 0.0631 | 0.9389 | 0.1054 |
| mid@1 | 0.0172 | 0.9829 | 0.0154 | 0.9848 | 0.0159 | 0.9842 | 0.0261 | 0.9743 | 0.1593 |
