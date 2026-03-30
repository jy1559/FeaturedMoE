# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5174 | 0.2047 | 0.2398 | 2.3284 | 0.5086 |
| micro@1 | 11.5142 | 0.2054 | 0.2368 | 2.2796 | 0.6091 |
| mid@1 | 11.5500 | 0.1974 | 0.1942 | 2.0777 | 0.4948 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0175 | 0.9826 | 0.0153 | 0.9848 | 0.0019 | 0.9981 |
| micro@1 | 0.0259 | 0.9744 | 0.0250 | 0.9753 | 0.0010 | 0.9990 |
| mid@1 | 0.0327 | 0.9679 | 0.0267 | 0.9736 | 0.0058 | 0.9942 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0287 | 0.9717 | 0.0283 | 0.9721 | 0.0279 | 0.9725 | 0.0271 | 0.9733 | 0.1048 |
| micro@1 | 0.0426 | 0.9583 | 0.0331 | 0.9675 | 0.0463 | 0.9547 | 0.0350 | 0.9656 | 0.1124 |
| mid@1 | 0.0934 | 0.9108 | 0.0633 | 0.9387 | 0.0907 | 0.9133 | 0.0632 | 0.9387 | 0.1251 |
