# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@eval:exposure#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.1385 | 0.4285 | 0.6382 | 2.3395 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0014 | 0.9986 | 0.0011 | 0.9989 | 0.0003 | 0.9997 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0012 | 0.9988 | 0.0075 | 0.9925 | 0.0071 | 0.9929 | 0.0188 | 0.9814 | 0.1092 |
