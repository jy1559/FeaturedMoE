# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3948 | 0.2305 | 0.2087 | 1.9362 | 0.0000 |
| micro@1 | 11.0706 | 0.2897 | 0.1581 | 1.6834 | 0.4952 |
| mid@1 | 8.9144 | 0.5883 | 0.3588 | 1.6489 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0452 | 0.9558 | 0.0205 | 0.9797 | 0.0290 | 0.9714 |
| micro@1 | 0.0407 | 0.9601 | 0.0189 | 0.9813 | 0.0274 | 0.9730 |
| mid@1 | 0.0422 | 0.9587 | 0.0233 | 0.9769 | 0.0203 | 0.9799 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0555 | 0.9460 | 0.0606 | 0.9412 | 0.0657 | 0.9364 | 0.0580 | 0.9437 | 0.1224 |
| micro@1 | 0.0717 | 0.9308 | 0.0838 | 0.9196 | 0.0913 | 0.9127 | 0.0643 | 0.9377 | 0.1238 |
| mid@1 | 0.0639 | 0.9381 | 0.0612 | 0.9406 | 0.0732 | 0.9294 | 0.0700 | 0.9324 | 0.2238 |
