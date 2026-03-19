# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.5459 | 0.1983 | 0.2230 | 1.9909 | 0.0000 |
| micro@1 | 11.1507 | 0.2760 | 0.1657 | 1.8479 | 0.4950 |
| mid@1 | 9.3988 | 0.5261 | 0.3518 | 1.7386 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0413 | 0.9596 | 0.0178 | 0.9823 | 0.0269 | 0.9735 |
| micro@1 | 0.0351 | 0.9655 | 0.0151 | 0.9850 | 0.0230 | 0.9773 |
| mid@1 | 0.0387 | 0.9620 | 0.0206 | 0.9796 | 0.0191 | 0.9811 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0502 | 0.9510 | 0.0549 | 0.9466 | 0.0596 | 0.9422 | 0.0526 | 0.9487 | 0.1195 |
| micro@1 | 0.0620 | 0.9399 | 0.0729 | 0.9297 | 0.0790 | 0.9240 | 0.0539 | 0.9475 | 0.1226 |
| mid@1 | 0.0585 | 0.9432 | 0.0562 | 0.9454 | 0.0676 | 0.9346 | 0.0631 | 0.9388 | 0.2051 |
