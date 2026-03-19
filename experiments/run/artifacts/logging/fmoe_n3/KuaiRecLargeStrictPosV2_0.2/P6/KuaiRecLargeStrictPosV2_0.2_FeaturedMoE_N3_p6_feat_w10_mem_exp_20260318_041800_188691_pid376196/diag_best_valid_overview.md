# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8857 | 0.1394 | 0.3236 | 1.4379 | 0.0000 |
| micro@1 | 5.4944 | 0.3033 | 0.3718 | 1.2193 | 0.5404 |
| mid@1 | 5.6471 | 0.2500 | 0.5072 | 1.3347 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0225 | 0.9778 | 0.0049 | 0.9952 | 0.0169 | 0.9832 |
| micro@1 | 0.0249 | 0.9754 | 0.0065 | 0.9935 | 0.0190 | 0.9812 |
| mid@1 | 0.0153 | 0.9848 | 0.0050 | 0.9950 | 0.0096 | 0.9905 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1015 |
| micro@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1359 |
| mid@1 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1274 |
