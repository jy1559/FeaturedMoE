# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 8.3503 | 0.6611 | 0.3046 | 1.5761 | 0.0000 |
| micro@1 | 7.5737 | 0.7645 | 0.3955 | 1.2127 | 0.4289 |
| mid@1 | 7.2323 | 0.8119 | 0.2716 | 1.4834 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0661 | 0.9360 | 0.0261 | 0.9743 | 0.0895 | 0.9148 |
| micro@1 | 0.0381 | 0.9626 | 0.0167 | 0.9835 | 0.0515 | 0.9500 |
| mid@1 | 0.0531 | 0.9483 | 0.0165 | 0.9837 | 0.0749 | 0.9287 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0863 | 0.9173 | 0.0941 | 0.9102 | 0.1029 | 0.9023 | 0.0832 | 0.9201 | 0.1590 |
| micro@1 | 0.0608 | 0.9410 | 0.0722 | 0.9304 | 0.0700 | 0.9324 | 0.0646 | 0.9374 | 0.2745 |
| mid@1 | 0.0769 | 0.9260 | 0.0760 | 0.9268 | 0.0886 | 0.9152 | 0.0960 | 0.9085 | 0.2117 |
