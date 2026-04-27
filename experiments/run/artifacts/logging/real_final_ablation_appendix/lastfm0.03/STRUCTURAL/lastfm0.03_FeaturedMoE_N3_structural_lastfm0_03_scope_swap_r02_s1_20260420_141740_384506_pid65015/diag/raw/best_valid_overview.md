# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9976 | 0.0246 | 0.3203 | 1.3534 | 0.6177 |
| mid@1 | 3.9851 | 0.0611 | 0.2783 | 1.3372 | 0.5686 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0042 | 0.9958 | 0.0042 | 0.9958 | 0.0000 | 1.0000 |
| mid@1 | 0.0062 | 0.9938 | 0.0062 | 0.9938 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0049 | 0.9951 | 0.0047 | 0.9953 | 0.0049 | 0.9951 | 0.0049 | 0.9951 | 0.2575 |
| mid@1 | 0.0100 | 0.9901 | 0.0104 | 0.9897 | 0.0095 | 0.9905 | 0.0080 | 0.9920 | 0.2640 |
