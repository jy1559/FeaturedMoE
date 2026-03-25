# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8377 | 0.1171 | 0.1677 | 2.2271 | 0.0000 |
| mid@1 | 9.2915 | 0.5399 | 0.1895 | 1.8039 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0088 | 0.9912 | 0.0054 | 0.9946 | 0.0032 | 0.9968 |
| mid@1 | 0.0090 | 0.9910 | 0.0027 | 0.9973 | 0.0044 | 0.9957 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0154 | 0.9847 | 0.0294 | 0.9710 | 0.0311 | 0.9694 | 0.0224 | 0.9779 | 0.1038 |
| mid@1 | 0.0408 | 0.9601 | 0.0261 | 0.9742 | 0.0347 | 0.9659 | 0.0631 | 0.9388 | 0.1586 |
