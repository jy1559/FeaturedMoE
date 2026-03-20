# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 16.8292 | 0.4341 | 0.1535 | 1.7118 | 0.0000 |
| micro@1 | 19.5512 | 0.1515 | 0.0736 | 2.2781 | 0.5851 |
| mid@1 | 18.2015 | 0.3143 | 0.1186 | 1.6097 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0970 | 0.9075 | 0.0408 | 0.9600 | 0.0671 | 0.9351 |
| micro@1 | 0.0507 | 0.9505 | 0.0130 | 0.9871 | 0.0388 | 0.9619 |
| mid@1 | 0.0886 | 0.9152 | 0.0392 | 0.9615 | 0.0589 | 0.9428 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1491 | 0.8615 | 0.1461 | 0.8641 | 0.1493 | 0.8613 | 0.1367 | 0.8723 | 0.1114 |
| micro@1 | 0.1345 | 0.8742 | 0.1457 | 0.8645 | 0.1634 | 0.8492 | 0.1087 | 0.8970 | 0.0678 |
| mid@1 | 0.1546 | 0.8567 | 0.1526 | 0.8584 | 0.1857 | 0.8305 | 0.1406 | 0.8689 | 0.0869 |
