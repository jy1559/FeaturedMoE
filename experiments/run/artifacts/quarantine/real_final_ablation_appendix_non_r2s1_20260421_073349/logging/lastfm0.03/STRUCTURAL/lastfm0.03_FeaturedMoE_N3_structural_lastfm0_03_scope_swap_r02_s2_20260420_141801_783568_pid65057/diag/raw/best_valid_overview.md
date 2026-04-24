# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9991 | 0.0148 | 0.3163 | 1.3471 | 0.6254 |
| mid@1 | 3.9844 | 0.0626 | 0.2910 | 1.3340 | 0.5609 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0050 | 0.9950 | 0.0050 | 0.9950 | 0.0000 | 1.0000 |
| mid@1 | 0.0067 | 0.9933 | 0.0067 | 0.9933 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0059 | 0.9942 | 0.0056 | 0.9944 | 0.0057 | 0.9943 | 0.0058 | 0.9942 | 0.2560 |
| mid@1 | 0.0104 | 0.9896 | 0.0110 | 0.9891 | 0.0100 | 0.9900 | 0.0079 | 0.9921 | 0.2622 |
