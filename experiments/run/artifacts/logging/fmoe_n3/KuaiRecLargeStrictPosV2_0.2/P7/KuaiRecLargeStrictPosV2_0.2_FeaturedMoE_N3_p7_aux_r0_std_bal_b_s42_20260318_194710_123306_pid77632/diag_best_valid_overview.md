# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9529 | 0.3092 | 0.2671 | 1.8322 | 0.0000 |
| micro@1 | 11.4769 | 0.2135 | 0.1615 | 1.6595 | 0.5416 |
| mid@1 | 9.5774 | 0.5029 | 0.2914 | 1.6629 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0551 | 0.9464 | 0.0272 | 0.9732 | 0.0341 | 0.9665 |
| micro@1 | 0.0437 | 0.9572 | 0.0214 | 0.9788 | 0.0298 | 0.9707 |
| mid@1 | 0.0476 | 0.9535 | 0.0243 | 0.9760 | 0.0247 | 0.9756 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0693 | 0.9330 | 0.0748 | 0.9279 | 0.0815 | 0.9218 | 0.0720 | 0.9305 | 0.1492 |
| micro@1 | 0.0744 | 0.9283 | 0.0866 | 0.9170 | 0.0927 | 0.9114 | 0.0688 | 0.9335 | 0.1214 |
| mid@1 | 0.0715 | 0.9310 | 0.0694 | 0.9329 | 0.0825 | 0.9208 | 0.0810 | 0.9222 | 0.1873 |
