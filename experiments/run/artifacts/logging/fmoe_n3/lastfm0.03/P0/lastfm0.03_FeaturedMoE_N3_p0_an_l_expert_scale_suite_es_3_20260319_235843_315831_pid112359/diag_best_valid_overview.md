# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5159 | 0.2050 | 0.1408 | 1.5634 | 0.0000 |
| micro@1 | 11.6936 | 0.1619 | 0.1343 | 1.7701 | 0.4845 |
| mid@1 | 10.9938 | 0.3025 | 0.2035 | 1.3804 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0749 | 0.9278 | 0.0391 | 0.9617 | 0.0434 | 0.9576 |
| micro@1 | 0.0440 | 0.9569 | 0.0177 | 0.9824 | 0.0277 | 0.9727 |
| mid@1 | 0.0684 | 0.9338 | 0.0377 | 0.9630 | 0.0360 | 0.9647 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1163 | 0.8902 | 0.1122 | 0.8939 | 0.1153 | 0.8911 | 0.1064 | 0.8990 | 0.1143 |
| micro@1 | 0.1378 | 0.8712 | 0.1403 | 0.8691 | 0.1680 | 0.8453 | 0.1039 | 0.9014 | 0.1043 |
| mid@1 | 0.1295 | 0.8785 | 0.1157 | 0.8907 | 0.1586 | 0.8534 | 0.1220 | 0.8852 | 0.1316 |
