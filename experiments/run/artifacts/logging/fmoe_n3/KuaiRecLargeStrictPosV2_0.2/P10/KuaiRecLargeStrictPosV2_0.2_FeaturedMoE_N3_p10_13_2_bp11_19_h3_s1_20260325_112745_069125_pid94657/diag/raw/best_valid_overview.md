# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.7869 | 0.1345 | 0.1678 | 2.2510 | 0.0000 |
| micro@1 | 8.7470 | 0.6098 | 0.2220 | 1.9244 | 0.6845 |
| mid@1 | 9.2778 | 0.5417 | 0.1989 | 1.8176 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0078 | 0.9922 | 0.0052 | 0.9948 | 0.0026 | 0.9974 |
| micro@1 | 0.0033 | 0.9967 | 0.0012 | 0.9988 | 0.0022 | 0.9978 |
| mid@1 | 0.0087 | 0.9914 | 0.0027 | 0.9973 | 0.0042 | 0.9959 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0137 | 0.9864 | 0.0261 | 0.9743 | 0.0278 | 0.9726 | 0.0199 | 0.9803 | 0.1031 |
| micro@1 | 0.0092 | 0.9908 | 0.0126 | 0.9875 | 0.0125 | 0.9875 | 0.0100 | 0.9900 | 0.1789 |
| mid@1 | 0.0392 | 0.9616 | 0.0252 | 0.9751 | 0.0332 | 0.9673 | 0.0615 | 0.9404 | 0.1626 |
