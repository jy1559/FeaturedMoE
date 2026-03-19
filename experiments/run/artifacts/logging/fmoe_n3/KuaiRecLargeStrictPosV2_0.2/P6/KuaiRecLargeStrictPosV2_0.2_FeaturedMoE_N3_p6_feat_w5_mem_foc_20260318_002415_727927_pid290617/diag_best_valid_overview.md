# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.8732 | 0.1469 | 0.3172 | 1.5054 | 0.0000 |
| micro@1 | 5.7692 | 0.2000 | 0.2712 | 1.3439 | 0.5305 |
| mid@1 | 4.9154 | 0.4697 | 0.4383 | 1.1647 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0284 | 0.9720 | 0.0071 | 0.9930 | 0.0218 | 0.9785 |
| micro@1 | 0.0263 | 0.9741 | 0.0073 | 0.9928 | 0.0194 | 0.9808 |
| mid@1 | 0.0231 | 0.9772 | 0.0042 | 0.9958 | 0.0137 | 0.9864 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0000 | 1.0000 | 0.0363 | 0.9643 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1002 |
| micro@1 | 0.0000 | 1.0000 | 0.0456 | 0.9554 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1105 |
| mid@1 | 0.0000 | 1.0000 | 0.0335 | 0.9671 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1595 |
