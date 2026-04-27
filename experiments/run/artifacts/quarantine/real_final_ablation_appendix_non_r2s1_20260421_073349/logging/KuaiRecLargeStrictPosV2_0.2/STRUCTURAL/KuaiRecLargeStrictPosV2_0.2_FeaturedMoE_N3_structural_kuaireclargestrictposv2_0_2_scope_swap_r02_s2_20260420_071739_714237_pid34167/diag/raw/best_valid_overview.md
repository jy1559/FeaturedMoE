# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 13.7971 | 0.3996 | 0.2551 | 2.4973 | 0.8680 |
| mid@1 | 6.1589 | 1.2641 | 0.3597 | 1.8213 | 0.7154 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0259 | 0.9745 | 0.0077 | 0.9923 | 0.0188 | 0.9814 |
| mid@1 | 0.0319 | 0.9686 | 0.0104 | 0.9896 | 0.0223 | 0.9779 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0314 | 0.9691 | 0.0314 | 0.9690 | 0.0326 | 0.9679 | 0.0303 | 0.9702 | 0.1093 |
| mid@1 | 0.0484 | 0.9527 | 0.0433 | 0.9577 | 0.0417 | 0.9592 | 0.0380 | 0.9627 | 0.2325 |
