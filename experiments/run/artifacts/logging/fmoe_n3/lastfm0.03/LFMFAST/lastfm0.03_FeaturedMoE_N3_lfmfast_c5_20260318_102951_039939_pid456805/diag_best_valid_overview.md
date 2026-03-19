# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4916 | 0.2103 | 0.1421 | 2.0230 | 0.0000 |
| micro@1 | 11.9020 | 0.0907 | 0.1538 | 2.0441 | 0.5946 |
| mid@1 | 11.2031 | 0.2667 | 0.1461 | 1.8144 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0445 | 0.9564 | 0.0175 | 0.9827 | 0.0285 | 0.9719 |
| micro@1 | 0.0241 | 0.9762 | 0.0076 | 0.9925 | 0.0169 | 0.9832 |
| mid@1 | 0.0399 | 0.9608 | 0.0169 | 0.9832 | 0.0255 | 0.9748 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0685 | 0.9338 | 0.0641 | 0.9379 | 0.0664 | 0.9358 | 0.0622 | 0.9397 | 0.1176 |
| micro@1 | 0.0702 | 0.9322 | 0.0754 | 0.9274 | 0.0919 | 0.9122 | 0.0524 | 0.9489 | 0.1048 |
| mid@1 | 0.0795 | 0.9236 | 0.0642 | 0.9378 | 0.0926 | 0.9115 | 0.0732 | 0.9294 | 0.1159 |
