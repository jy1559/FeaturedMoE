# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 15.7628 | 0.1227 | 0.0947 | 2.2140 | 0.6798 |
| mid@1 | 15.4403 | 0.1904 | 0.1451 | 2.1417 | 0.6750 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0801 | 0.9231 | 0.0232 | 0.9771 | 0.0583 | 0.9434 |
| mid@1 | 0.0759 | 0.9269 | 0.0184 | 0.9817 | 0.0572 | 0.9445 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1094 | 0.8964 | 0.1106 | 0.8953 | 0.1168 | 0.8897 | 0.1072 | 0.8983 | 0.0794 |
| mid@1 | 0.1162 | 0.8903 | 0.1233 | 0.8840 | 0.1355 | 0.8732 | 0.0860 | 0.9176 | 0.1012 |
