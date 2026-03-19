# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.7293 | 0.2173 | 0.2697 | 1.2166 | 0.0000 |
| micro@1 | 4.8523 | 0.4863 | 0.3796 | 1.1928 | 0.4679 |
| mid@1 | 5.9894 | 0.0421 | 0.2732 | 1.6329 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0339 | 0.9667 | 0.0078 | 0.9923 | 0.0280 | 0.9724 |
| micro@1 | 0.0228 | 0.9775 | 0.0053 | 0.9947 | 0.0171 | 0.9830 |
| mid@1 | 0.0151 | 0.9850 | 0.0053 | 0.9947 | 0.0098 | 0.9903 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0440 | 0.9570 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1122 |
| micro@1 | 0.0335 | 0.9670 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1638 |
| mid@1 | 0.0255 | 0.9748 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0906 |
