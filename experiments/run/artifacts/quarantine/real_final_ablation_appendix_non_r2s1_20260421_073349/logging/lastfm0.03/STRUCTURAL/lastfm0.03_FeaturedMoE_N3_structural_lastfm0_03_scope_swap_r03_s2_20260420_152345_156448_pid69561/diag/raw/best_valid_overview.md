# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9989 | 0.0163 | 0.3215 | 1.3482 | 0.6252 |
| mid@1 | 3.9837 | 0.0640 | 0.2884 | 1.3342 | 0.5607 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0049 | 0.9951 | 0.0049 | 0.9951 | 0.0000 | 1.0000 |
| mid@1 | 0.0066 | 0.9934 | 0.0066 | 0.9934 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0057 | 0.9943 | 0.0055 | 0.9946 | 0.0056 | 0.9944 | 0.0056 | 0.9944 | 0.2563 |
| mid@1 | 0.0105 | 0.9896 | 0.0111 | 0.9890 | 0.0101 | 0.9900 | 0.0079 | 0.9921 | 0.2625 |
