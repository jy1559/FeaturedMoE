# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.7766 | 0.2432 | 0.3901 | 1.1800 | 0.5585 |
| mid@1 | 3.8908 | 0.1675 | 0.4672 | 1.2675 | 0.5663 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0192 | 0.9809 | 0.0192 | 0.9809 | 0.0000 | 1.0000 |
| mid@1 | 0.0135 | 0.9866 | 0.0135 | 0.9866 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0277 | 0.9727 | 0.0311 | 0.9693 | 0.0322 | 0.9683 | 0.0282 | 0.9722 | 0.3341 |
| mid@1 | 0.0174 | 0.9828 | 0.0197 | 0.9805 | 0.0265 | 0.9738 | 0.0181 | 0.9820 | 0.3169 |
