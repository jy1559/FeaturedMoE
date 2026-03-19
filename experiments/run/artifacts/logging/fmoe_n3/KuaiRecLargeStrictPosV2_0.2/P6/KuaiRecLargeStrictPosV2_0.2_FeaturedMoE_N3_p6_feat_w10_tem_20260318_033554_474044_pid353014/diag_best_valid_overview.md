# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 2.8613 | 0.2202 | 0.5420 | 0.7843 | 0.0000 |
| micro@1 | 2.7717 | 0.2870 | 0.4264 | 0.7101 | 0.5708 |
| mid@1 | 2.9019 | 0.1839 | 0.4620 | 0.7078 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0289 | 0.9716 | -0.0000 | 1.0000 | 0.0289 | 0.9716 |
| micro@1 | 0.0153 | 0.9848 | -0.0000 | 1.0000 | 0.0153 | 0.9848 |
| mid@1 | 0.0278 | 0.9726 | 0.0000 | 1.0000 | 0.0278 | 0.9726 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0359 | 0.9648 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1092 |
| micro@1 | 0.0205 | 0.9797 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1049 |
| mid@1 | 0.0360 | 0.9646 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1045 |
