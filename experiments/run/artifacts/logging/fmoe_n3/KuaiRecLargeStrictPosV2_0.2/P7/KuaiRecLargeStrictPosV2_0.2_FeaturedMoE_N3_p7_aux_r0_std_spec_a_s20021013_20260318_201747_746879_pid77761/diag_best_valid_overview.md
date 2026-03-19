# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4255 | 0.2242 | 0.2045 | 1.9122 | 0.0000 |
| micro@1 | 11.3240 | 0.2443 | 0.1668 | 1.9981 | 0.5101 |
| mid@1 | 9.0865 | 0.5663 | 0.3441 | 1.6377 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0473 | 0.9538 | 0.0217 | 0.9785 | 0.0306 | 0.9699 |
| micro@1 | 0.0305 | 0.9699 | 0.0120 | 0.9880 | 0.0199 | 0.9803 |
| mid@1 | 0.0422 | 0.9587 | 0.0228 | 0.9775 | 0.0215 | 0.9788 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0582 | 0.9434 | 0.0637 | 0.9383 | 0.0691 | 0.9332 | 0.0612 | 0.9407 | 0.1230 |
| micro@1 | 0.0548 | 0.9467 | 0.0655 | 0.9366 | 0.0705 | 0.9320 | 0.0476 | 0.9535 | 0.1191 |
| mid@1 | 0.0638 | 0.9382 | 0.0609 | 0.9409 | 0.0729 | 0.9297 | 0.0714 | 0.9311 | 0.2133 |
