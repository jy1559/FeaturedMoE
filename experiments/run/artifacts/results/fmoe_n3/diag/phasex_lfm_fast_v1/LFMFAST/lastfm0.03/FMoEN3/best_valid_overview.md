# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.5920 | 0.3646 | 0.1263 | 1.4859 | 0.0000 |
| micro@1 | 10.7840 | 0.3358 | 0.1296 | 1.5254 | 0.5030 |
| mid@1 | 10.8720 | 0.3221 | 0.1850 | 1.3777 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0882 | 0.9156 | 0.0386 | 0.9622 | 0.0793 | 0.9238 |
| micro@1 | 0.0546 | 0.9468 | 0.0180 | 0.9822 | 0.0495 | 0.9519 |
| mid@1 | 0.0816 | 0.9216 | 0.0348 | 0.9658 | 0.0806 | 0.9227 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1288 | 0.8791 | 0.1230 | 0.8843 | 0.1263 | 0.8813 | 0.1175 | 0.8892 | 0.1358 |
| micro@1 | 0.1531 | 0.8580 | 0.1582 | 0.8537 | 0.2002 | 0.8185 | 0.1160 | 0.8905 | 0.1357 |
| mid@1 | 0.1428 | 0.8669 | 0.1238 | 0.8836 | 0.1599 | 0.8522 | 0.1328 | 0.8756 | 0.1330 |
