# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.7852 | 0.3529 | 0.1237 | 1.4654 | 0.0000 |
| micro@1 | 19.6065 | 0.1417 | 0.0792 | 1.8460 | 0.5307 |
| mid@1 | 18.5680 | 0.2777 | 0.0834 | 1.3726 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1199 | 0.8870 | 0.0573 | 0.9443 | 0.0819 | 0.9213 |
| micro@1 | 0.0800 | 0.9231 | 0.0259 | 0.9745 | 0.0589 | 0.9428 |
| mid@1 | 0.1134 | 0.8928 | 0.0559 | 0.9456 | 0.0743 | 0.9284 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1783 | 0.8367 | 0.1759 | 0.8387 | 0.1785 | 0.8366 | 0.1642 | 0.8485 | 0.1009 |
| micro@1 | 0.2056 | 0.8142 | 0.2254 | 0.7982 | 0.2531 | 0.7764 | 0.1721 | 0.8419 | 0.0708 |
| mid@1 | 0.1796 | 0.8356 | 0.1888 | 0.8279 | 0.2151 | 0.8065 | 0.1664 | 0.8467 | 0.0759 |
