# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.0612 | 0.2913 | 0.2411 | 1.9145 | 0.0000 |
| micro@1 | 10.9772 | 0.3052 | 0.2034 | 1.6631 | 0.4946 |
| mid@1 | 8.4995 | 0.6418 | 0.3882 | 1.6856 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0467 | 0.9544 | 0.0220 | 0.9782 | 0.0300 | 0.9705 |
| micro@1 | 0.0433 | 0.9576 | 0.0215 | 0.9787 | 0.0294 | 0.9710 |
| mid@1 | 0.0445 | 0.9564 | 0.0226 | 0.9776 | 0.0225 | 0.9777 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0581 | 0.9436 | 0.0635 | 0.9385 | 0.0684 | 0.9338 | 0.0609 | 0.9409 | 0.1371 |
| micro@1 | 0.0748 | 0.9280 | 0.0849 | 0.9186 | 0.0926 | 0.9115 | 0.0677 | 0.9346 | 0.1418 |
| mid@1 | 0.0670 | 0.9352 | 0.0645 | 0.9376 | 0.0769 | 0.9260 | 0.0761 | 0.9267 | 0.2404 |
