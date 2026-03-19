# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.6820 | 0.1650 | 0.2485 | 2.0873 | 0.0000 |
| micro@1 | 8.6234 | 0.6258 | 0.3456 | 1.5896 | 0.4840 |
| mid@1 | 9.6789 | 0.4897 | 0.3295 | 1.9411 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0228 | 0.9774 | 0.0115 | 0.9885 | 0.0111 | 0.9890 |
| micro@1 | 0.0278 | 0.9726 | 0.0110 | 0.9891 | 0.0161 | 0.9840 |
| mid@1 | 0.0198 | 0.9804 | 0.0065 | 0.9935 | 0.0108 | 0.9893 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0315 | 0.9690 | 0.0383 | 0.9625 | 0.0399 | 0.9609 | 0.0322 | 0.9683 | 0.1098 |
| micro@1 | 0.0461 | 0.9550 | 0.0544 | 0.9470 | 0.0546 | 0.9468 | 0.0557 | 0.9458 | 0.2110 |
| mid@1 | 0.0332 | 0.9673 | 0.0325 | 0.9680 | 0.0409 | 0.9599 | 0.0415 | 0.9594 | 0.1867 |
