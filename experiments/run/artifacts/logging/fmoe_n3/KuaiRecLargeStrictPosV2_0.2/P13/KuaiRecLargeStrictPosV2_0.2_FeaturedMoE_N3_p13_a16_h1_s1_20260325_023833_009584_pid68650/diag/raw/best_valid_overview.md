# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:position_shift@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3354 | 0.2421 | 0.1970 | 2.2091 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0068 | 0.9932 | 0.0038 | 0.9962 | 0.0027 | 0.9973 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0160 | 0.9841 | 0.0227 | 0.9776 | 0.0242 | 0.9761 | 0.0138 | 0.9863 | 0.1151 |
