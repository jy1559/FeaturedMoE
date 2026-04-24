# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.3177 | 0.5529 | 0.1445 | 2.2203 | 0.9032 |
| mid@1 | 8.3326 | 1.1833 | 0.3968 | 1.8117 | 0.8138 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0648 | 0.9373 | 0.0160 | 0.9842 | 0.0497 | 0.9515 |
| mid@1 | 0.0665 | 0.9357 | 0.0180 | 0.9822 | 0.0528 | 0.9486 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0821 | 0.9212 | 0.0841 | 0.9193 | 0.0854 | 0.9181 | 0.0784 | 0.9246 | 0.1109 |
| mid@1 | 0.0941 | 0.9102 | 0.0932 | 0.9110 | 0.0877 | 0.9161 | 0.0765 | 0.9263 | 0.2760 |
