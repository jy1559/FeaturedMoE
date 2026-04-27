# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.9931 | 0.0416 | 0.2937 | 1.1791 | 0.5294 |
| mid@1 | 3.9778 | 0.0748 | 0.3180 | 1.1777 | 0.4887 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0262 | 0.9742 | 0.0262 | 0.9742 | 0.0000 | 1.0000 |
| mid@1 | 0.0251 | 0.9752 | 0.0251 | 0.9752 | 0.0000 | 1.0000 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0385 | 0.9622 | 0.0398 | 0.9610 | 0.0412 | 0.9597 | 0.0376 | 0.9631 | 0.2658 |
| mid@1 | 0.0395 | 0.9612 | 0.0409 | 0.9600 | 0.0455 | 0.9556 | 0.0267 | 0.9737 | 0.2793 |
