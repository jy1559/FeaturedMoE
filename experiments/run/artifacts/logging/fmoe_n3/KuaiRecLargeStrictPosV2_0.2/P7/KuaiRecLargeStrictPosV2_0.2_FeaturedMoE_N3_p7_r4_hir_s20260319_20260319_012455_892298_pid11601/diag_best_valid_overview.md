# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4948 | 0.2096 | 0.1978 | 1.9389 | 0.0000 |
| micro@1 | 9.4056 | 0.5252 | 0.2219 | 1.1819 | 0.4965 |
| mid@1 | 11.2992 | 0.2490 | 0.2723 | 2.0113 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0501 | 0.9512 | 0.0298 | 0.9706 | 0.0217 | 0.9785 |
| micro@1 | 0.0587 | 0.9430 | 0.0360 | 0.9646 | 0.0255 | 0.9748 |
| mid@1 | 0.0323 | 0.9682 | 0.0160 | 0.9842 | 0.0179 | 0.9823 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0613 | 0.9406 | 0.0678 | 0.9344 | 0.0726 | 0.9299 | 0.0638 | 0.9382 | 0.1130 |
| micro@1 | 0.0975 | 0.9071 | 0.1149 | 0.8915 | 0.1191 | 0.8878 | 0.0946 | 0.9097 | 0.1923 |
| mid@1 | 0.0481 | 0.9531 | 0.0478 | 0.9533 | 0.0553 | 0.9462 | 0.0531 | 0.9483 | 0.1245 |
