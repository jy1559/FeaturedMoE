# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.6049 | 0.3627 | 0.1277 | 1.4855 | 0.0000 |
| micro@1 | 10.8153 | 0.3310 | 0.1288 | 1.5226 | 0.4978 |
| mid@1 | 10.8783 | 0.3211 | 0.1706 | 1.3704 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0887 | 0.9151 | 0.0388 | 0.9619 | 0.0798 | 0.9234 |
| micro@1 | 0.0557 | 0.9458 | 0.0180 | 0.9821 | 0.0510 | 0.9504 |
| mid@1 | 0.0825 | 0.9208 | 0.0355 | 0.9652 | 0.0812 | 0.9222 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1294 | 0.8786 | 0.1234 | 0.8839 | 0.1269 | 0.8809 | 0.1181 | 0.8886 | 0.1355 |
| micro@1 | 0.1551 | 0.8563 | 0.1623 | 0.8502 | 0.2031 | 0.8162 | 0.1173 | 0.8893 | 0.1349 |
| mid@1 | 0.1432 | 0.8665 | 0.1258 | 0.8817 | 0.1606 | 0.8517 | 0.1338 | 0.8747 | 0.1296 |
