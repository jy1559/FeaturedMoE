# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.6725 | 0.7510 | 0.2650 | 1.6949 | 0.5351 |
| mid@1 | 8.5603 | 0.6339 | 0.2367 | 1.6370 | 0.4969 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0999 | 0.9050 | 0.0347 | 0.9659 | 0.1043 | 0.9021 |
| mid@1 | 0.0702 | 0.9322 | 0.0222 | 0.9780 | 0.0678 | 0.9345 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1496 | 0.8611 | 0.1490 | 0.8615 | 0.1488 | 0.8618 | 0.1463 | 0.8639 | 0.1966 |
| mid@1 | 0.1987 | 0.8198 | 0.1783 | 0.8367 | 0.1290 | 0.8790 | 0.1181 | 0.8886 | 0.1697 |
