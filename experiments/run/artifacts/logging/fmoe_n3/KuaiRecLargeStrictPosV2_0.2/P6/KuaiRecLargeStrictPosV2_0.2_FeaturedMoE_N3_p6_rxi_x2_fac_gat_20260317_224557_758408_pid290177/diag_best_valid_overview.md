# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.0195 | 0.5748 | 0.3509 | 1.5486 | 0.0000 |
| micro@1 | 9.1995 | 0.5517 | 0.1875 | 1.2206 | 0.6237 |
| mid@1 | 10.8503 | 0.3255 | 0.2631 | 1.4918 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0889 | 0.9150 | 0.0329 | 0.9676 | 0.0969 | 0.9077 |
| micro@1 | 0.0602 | 0.9416 | 0.0261 | 0.9742 | 0.0566 | 0.9451 |
| mid@1 | 0.0722 | 0.9304 | 0.0193 | 0.9809 | 0.0907 | 0.9135 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1087 | 0.8970 | 0.1248 | 0.8826 | 0.1356 | 0.8732 | 0.1189 | 0.8879 | 0.1845 |
| micro@1 | 0.0835 | 0.9199 | 0.0975 | 0.9071 | 0.1016 | 0.9033 | 0.0799 | 0.9232 | 0.1704 |
| mid@1 | 0.1013 | 0.9037 | 0.0986 | 0.9061 | 0.1142 | 0.8920 | 0.1067 | 0.8988 | 0.1614 |
