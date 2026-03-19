# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.6885 | 0.1632 | 0.2212 | 2.0623 | 0.0000 |
| micro@1 | 9.9586 | 0.4528 | 0.2474 | 1.7149 | 0.4898 |
| mid@1 | 10.1256 | 0.4302 | 0.3032 | 1.9713 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0235 | 0.9767 | 0.0115 | 0.9885 | 0.0121 | 0.9880 |
| micro@1 | 0.0208 | 0.9794 | 0.0086 | 0.9914 | 0.0126 | 0.9875 |
| mid@1 | 0.0188 | 0.9814 | 0.0064 | 0.9936 | 0.0104 | 0.9897 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0328 | 0.9677 | 0.0401 | 0.9607 | 0.0414 | 0.9594 | 0.0338 | 0.9667 | 0.1090 |
| micro@1 | 0.0338 | 0.9667 | 0.0419 | 0.9590 | 0.0389 | 0.9618 | 0.0439 | 0.9570 | 0.1682 |
| mid@1 | 0.0321 | 0.9684 | 0.0313 | 0.9692 | 0.0390 | 0.9618 | 0.0404 | 0.9604 | 0.1711 |
