# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8472 | 0.1616 | 0.2752 | 1.2324 | 0.0000 |
| micro@1 | 4.5612 | 0.5616 | 0.3915 | 1.0933 | 0.4219 |
| mid@1 | 5.9879 | 0.0449 | 0.2651 | 1.6170 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0301 | 0.9703 | 0.0071 | 0.9929 | 0.0251 | 0.9752 |
| micro@1 | 0.0230 | 0.9773 | 0.0050 | 0.9950 | 0.0170 | 0.9831 |
| mid@1 | 0.0158 | 0.9843 | 0.0054 | 0.9946 | 0.0102 | 0.9898 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0403 | 0.9605 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1042 |
| micro@1 | 0.0348 | 0.9658 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1732 |
| mid@1 | 0.0269 | 0.9735 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0907 |
