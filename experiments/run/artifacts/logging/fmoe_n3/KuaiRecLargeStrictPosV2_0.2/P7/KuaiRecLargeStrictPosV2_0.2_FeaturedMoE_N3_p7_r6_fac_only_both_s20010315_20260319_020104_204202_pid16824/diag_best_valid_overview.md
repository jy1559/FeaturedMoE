# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9935 | 0.3026 | 0.3859 | 1.8953 | 0.0000 |
| micro@1 | 11.4788 | 0.2131 | 0.3147 | 1.6937 | 0.4043 |
| mid@1 | 9.5849 | 0.5020 | 0.6620 | 2.1061 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0325 | 0.9680 | 0.0325 | 0.9680 | 0.0000 | 1.0000 |
| micro@1 | 0.0362 | 0.9645 | 0.0362 | 0.9645 | 0.0000 | 1.0000 |
| mid@1 | 0.0105 | 0.9895 | 0.0105 | 0.9895 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0426 | 0.9583 | 0.0481 | 0.9530 | 0.0512 | 0.9500 | 0.0405 | 0.9603 | 0.1117 |
| micro@1 | 0.0583 | 0.9434 | 0.0695 | 0.9329 | 0.0719 | 0.9307 | 0.0636 | 0.9384 | 0.1072 |
| mid@1 | 0.0174 | 0.9827 | 0.0156 | 0.9846 | 0.0161 | 0.9840 | 0.0265 | 0.9739 | 0.1594 |
