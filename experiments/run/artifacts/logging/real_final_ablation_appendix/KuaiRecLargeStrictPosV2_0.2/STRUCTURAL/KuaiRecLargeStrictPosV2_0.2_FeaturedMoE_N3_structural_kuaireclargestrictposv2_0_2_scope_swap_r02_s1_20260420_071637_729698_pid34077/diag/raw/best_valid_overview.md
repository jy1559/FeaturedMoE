# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 14.0936 | 0.3678 | 0.2707 | 2.5356 | 0.8653 |
| mid@1 | 6.0482 | 1.2827 | 0.3563 | 1.8283 | 0.7638 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0218 | 0.9784 | 0.0069 | 0.9932 | 0.0156 | 0.9845 |
| mid@1 | 0.0292 | 0.9712 | 0.0111 | 0.9890 | 0.0189 | 0.9813 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0263 | 0.9740 | 0.0261 | 0.9742 | 0.0273 | 0.9730 | 0.0252 | 0.9751 | 0.1189 |
| mid@1 | 0.0409 | 0.9599 | 0.0364 | 0.9643 | 0.0345 | 0.9661 | 0.0356 | 0.9651 | 0.2548 |
