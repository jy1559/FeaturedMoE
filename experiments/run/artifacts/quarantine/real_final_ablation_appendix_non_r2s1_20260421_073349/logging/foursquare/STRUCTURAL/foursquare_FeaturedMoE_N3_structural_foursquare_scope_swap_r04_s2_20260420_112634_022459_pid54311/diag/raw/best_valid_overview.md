# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9304 | 0.0764 | 0.1447 | 2.3313 | 0.6981 |
| mid@1 | 11.9582 | 0.0591 | 0.1303 | 2.3440 | 0.6881 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0360 | 0.9646 | 0.0118 | 0.9883 | 0.0245 | 0.9758 |
| mid@1 | 0.0355 | 0.9652 | 0.0150 | 0.9851 | 0.0208 | 0.9794 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0416 | 0.9592 | 0.0412 | 0.9597 | 0.0407 | 0.9601 | 0.0390 | 0.9617 | 0.0944 |
| mid@1 | 0.0409 | 0.9599 | 0.0388 | 0.9619 | 0.0422 | 0.9587 | 0.0398 | 0.9610 | 0.0903 |
