# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.4765 | 0.5160 | 0.1880 | 1.3323 | 0.0000 |
| micro@1 | 8.5130 | 0.6400 | 0.3452 | 1.1960 | 0.5766 |
| mid@1 | 10.4072 | 0.3912 | 0.1894 | 1.5358 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0718 | 0.9308 | 0.0349 | 0.9657 | 0.0778 | 0.9252 |
| micro@1 | 0.0562 | 0.9453 | 0.0184 | 0.9818 | 0.0608 | 0.9412 |
| mid@1 | 0.0756 | 0.9272 | 0.0242 | 0.9761 | 0.0828 | 0.9206 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0922 | 0.9119 | 0.1044 | 0.9009 | 0.1109 | 0.8951 | 0.0927 | 0.9115 | 0.1471 |
| micro@1 | 0.0797 | 0.9234 | 0.0966 | 0.9079 | 0.1007 | 0.9042 | 0.0753 | 0.9275 | 0.2159 |
| mid@1 | 0.1007 | 0.9042 | 0.1020 | 0.9030 | 0.1242 | 0.8832 | 0.1239 | 0.8834 | 0.1380 |
