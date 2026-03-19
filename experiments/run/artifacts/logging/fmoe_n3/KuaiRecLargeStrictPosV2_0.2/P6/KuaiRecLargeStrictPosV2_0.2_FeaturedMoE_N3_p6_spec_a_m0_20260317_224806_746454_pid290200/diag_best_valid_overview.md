# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0366 | 0.2954 | 0.2357 | 1.9604 | 0.0000 |
| micro@1 | 11.0509 | 0.2931 | 0.1888 | 1.8007 | 0.4935 |
| mid@1 | 10.1926 | 0.4211 | 0.2644 | 1.6489 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0429 | 0.9580 | 0.0190 | 0.9812 | 0.0286 | 0.9718 |
| micro@1 | 0.0364 | 0.9642 | 0.0161 | 0.9840 | 0.0248 | 0.9755 |
| mid@1 | 0.0390 | 0.9618 | 0.0173 | 0.9829 | 0.0212 | 0.9791 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0530 | 0.9484 | 0.0583 | 0.9434 | 0.0636 | 0.9383 | 0.0566 | 0.9450 | 0.1320 |
| micro@1 | 0.0652 | 0.9368 | 0.0773 | 0.9256 | 0.0828 | 0.9206 | 0.0595 | 0.9422 | 0.1410 |
| mid@1 | 0.0610 | 0.9409 | 0.0606 | 0.9412 | 0.0728 | 0.9298 | 0.0670 | 0.9352 | 0.1823 |
