# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.4316 | 0.3877 | 0.1544 | 1.4925 | 0.0000 |
| micro@1 | 10.5813 | 0.3662 | 0.1358 | 1.5265 | 0.5075 |
| mid@1 | 10.6152 | 0.3612 | 0.1733 | 1.3882 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0872 | 0.9165 | 0.0361 | 0.9645 | 0.0801 | 0.9230 |
| micro@1 | 0.0524 | 0.9490 | 0.0155 | 0.9846 | 0.0451 | 0.9559 |
| mid@1 | 0.0770 | 0.9258 | 0.0346 | 0.9660 | 0.0785 | 0.9247 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1274 | 0.8804 | 0.1225 | 0.8847 | 0.1254 | 0.8821 | 0.1164 | 0.8902 | 0.1328 |
| micro@1 | 0.1490 | 0.8616 | 0.1514 | 0.8595 | 0.1956 | 0.8224 | 0.1140 | 0.8923 | 0.1398 |
| mid@1 | 0.1389 | 0.8703 | 0.1167 | 0.8899 | 0.1547 | 0.8567 | 0.1285 | 0.8794 | 0.1302 |
