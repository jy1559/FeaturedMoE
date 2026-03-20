# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.4320 | 0.3838 | 0.1432 | 1.7009 | 0.0000 |
| micro@1 | 19.3436 | 0.1842 | 0.0863 | 2.0660 | 0.5770 |
| mid@1 | 18.3806 | 0.2968 | 0.0944 | 1.6235 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0933 | 0.9109 | 0.0396 | 0.9612 | 0.0639 | 0.9381 |
| micro@1 | 0.0572 | 0.9444 | 0.0161 | 0.9841 | 0.0427 | 0.9582 |
| mid@1 | 0.0879 | 0.9158 | 0.0395 | 0.9613 | 0.0570 | 0.9446 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1462 | 0.8639 | 0.1418 | 0.8678 | 0.1458 | 0.8643 | 0.1337 | 0.8748 | 0.0988 |
| micro@1 | 0.1591 | 0.8529 | 0.1650 | 0.8479 | 0.1953 | 0.8226 | 0.1316 | 0.8767 | 0.0727 |
| mid@1 | 0.1558 | 0.8557 | 0.1536 | 0.8576 | 0.1861 | 0.8302 | 0.1407 | 0.8688 | 0.0818 |
