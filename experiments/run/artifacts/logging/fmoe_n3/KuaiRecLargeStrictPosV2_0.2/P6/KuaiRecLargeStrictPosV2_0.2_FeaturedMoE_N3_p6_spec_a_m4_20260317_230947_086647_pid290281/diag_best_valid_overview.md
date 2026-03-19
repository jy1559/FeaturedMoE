# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3231 | 0.2445 | 0.1985 | 2.1281 | 0.0000 |
| micro@1 | 11.0286 | 0.2968 | 0.1519 | 1.8808 | 0.5852 |
| mid@1 | 7.4122 | 0.7867 | 0.3212 | 1.3663 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0280 | 0.9724 | 0.0123 | 0.9878 | 0.0163 | 0.9839 |
| micro@1 | 0.0301 | 0.9704 | 0.0109 | 0.9891 | 0.0195 | 0.9807 |
| mid@1 | 0.0306 | 0.9698 | 0.0196 | 0.9806 | 0.0164 | 0.9837 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0380 | 0.9627 | 0.0432 | 0.9577 | 0.0442 | 0.9568 | 0.0397 | 0.9611 | 0.1090 |
| micro@1 | 0.0477 | 0.9535 | 0.0557 | 0.9458 | 0.0576 | 0.9441 | 0.0475 | 0.9536 | 0.1319 |
| mid@1 | 0.0538 | 0.9476 | 0.0466 | 0.9545 | 0.0466 | 0.9544 | 0.0775 | 0.9254 | 0.2202 |
