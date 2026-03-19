# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.6660 | 0.4914 | 0.1842 | 1.6076 | 0.0000 |
| micro@1 | 7.9063 | 0.7196 | 0.3565 | 1.4698 | 0.5186 |
| mid@1 | 10.3816 | 0.3948 | 0.2680 | 1.7013 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0691 | 0.9333 | 0.0293 | 0.9711 | 0.0770 | 0.9259 |
| micro@1 | 0.0321 | 0.9684 | 0.0094 | 0.9907 | 0.0391 | 0.9617 |
| mid@1 | 0.0745 | 0.9282 | 0.0317 | 0.9688 | 0.0963 | 0.9087 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0880 | 0.9158 | 0.1001 | 0.9047 | 0.1076 | 0.8980 | 0.0907 | 0.9133 | 0.1508 |
| micro@1 | 0.0507 | 0.9506 | 0.0572 | 0.9444 | 0.0597 | 0.9420 | 0.0496 | 0.9516 | 0.2146 |
| mid@1 | 0.1118 | 0.8942 | 0.0982 | 0.9064 | 0.1203 | 0.8867 | 0.1240 | 0.8834 | 0.1470 |
