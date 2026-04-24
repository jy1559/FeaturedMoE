# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.0533 | 0.4157 | 0.3329 | 2.8248 | 0.8152 |
| mid@1 | 11.4242 | 0.8664 | 0.3089 | 2.3942 | 0.8048 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0110 | 0.9891 | 0.0016 | 0.9984 | 0.0095 | 0.9905 |
| mid@1 | 0.0308 | 0.9696 | 0.0066 | 0.9935 | 0.0255 | 0.9749 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0149 | 0.9853 | 0.0150 | 0.9851 | 0.0164 | 0.9837 | 0.0126 | 0.9875 | 0.0940 |
| mid@1 | 0.0415 | 0.9593 | 0.0367 | 0.9640 | 0.0350 | 0.9656 | 0.0393 | 0.9614 | 0.1524 |
