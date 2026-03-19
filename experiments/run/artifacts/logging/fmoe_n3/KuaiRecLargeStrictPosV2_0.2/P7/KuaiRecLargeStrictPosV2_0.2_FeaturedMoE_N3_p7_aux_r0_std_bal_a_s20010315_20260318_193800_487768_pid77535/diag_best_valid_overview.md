# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3488 | 0.2395 | 0.2073 | 1.8777 | 0.0000 |
| micro@1 | 10.8914 | 0.3190 | 0.1936 | 1.6385 | 0.5114 |
| mid@1 | 9.1459 | 0.5586 | 0.3328 | 1.6481 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0502 | 0.9510 | 0.0239 | 0.9764 | 0.0321 | 0.9684 |
| micro@1 | 0.0424 | 0.9585 | 0.0209 | 0.9793 | 0.0294 | 0.9710 |
| mid@1 | 0.0438 | 0.9571 | 0.0234 | 0.9769 | 0.0224 | 0.9779 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0621 | 0.9397 | 0.0680 | 0.9343 | 0.0737 | 0.9290 | 0.0652 | 0.9369 | 0.1273 |
| micro@1 | 0.0735 | 0.9291 | 0.0858 | 0.9178 | 0.0925 | 0.9116 | 0.0678 | 0.9345 | 0.1433 |
| mid@1 | 0.0660 | 0.9361 | 0.0629 | 0.9391 | 0.0749 | 0.9278 | 0.0756 | 0.9272 | 0.2073 |
