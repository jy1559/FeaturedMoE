# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3331 | 0.2426 | 0.2661 | 2.0567 | 0.0000 |
| micro@1 | 9.0986 | 0.5647 | 0.2867 | 1.4562 | 0.4651 |
| mid@1 | 9.4580 | 0.5184 | 0.3618 | 1.9479 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0254 | 0.9750 | 0.0129 | 0.9872 | 0.0131 | 0.9870 |
| micro@1 | 0.0362 | 0.9644 | 0.0158 | 0.9843 | 0.0213 | 0.9790 |
| mid@1 | 0.0193 | 0.9809 | 0.0066 | 0.9935 | 0.0105 | 0.9895 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0348 | 0.9658 | 0.0429 | 0.9580 | 0.0446 | 0.9564 | 0.0365 | 0.9641 | 0.1209 |
| micro@1 | 0.0611 | 0.9407 | 0.0708 | 0.9317 | 0.0717 | 0.9308 | 0.0679 | 0.9344 | 0.2004 |
| mid@1 | 0.0328 | 0.9678 | 0.0316 | 0.9689 | 0.0383 | 0.9624 | 0.0419 | 0.9589 | 0.1997 |
