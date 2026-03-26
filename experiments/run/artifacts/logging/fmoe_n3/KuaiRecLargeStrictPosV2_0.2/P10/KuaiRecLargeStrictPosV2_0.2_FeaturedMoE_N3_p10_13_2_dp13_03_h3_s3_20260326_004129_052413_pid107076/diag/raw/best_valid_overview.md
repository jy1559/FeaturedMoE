# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:shuffle@eval#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9684 | 0.0514 | 0.1661 | 2.3976 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0064 | 0.9936 | 0.0046 | 0.9954 | 0.0019 | 0.9981 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0104 | 0.9896 | 0.0136 | 0.9865 | 0.0148 | 0.9853 | 0.0113 | 0.9887 | 0.0977 |
