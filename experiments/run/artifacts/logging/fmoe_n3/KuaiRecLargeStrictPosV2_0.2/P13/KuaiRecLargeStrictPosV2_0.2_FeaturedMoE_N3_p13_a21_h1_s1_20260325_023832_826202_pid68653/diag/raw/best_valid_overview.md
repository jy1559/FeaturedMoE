# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:position_shift@train#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3970 | 0.2300 | 0.1759 | 2.2215 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0091 | 0.9909 | 0.0050 | 0.9950 | 0.0036 | 0.9965 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0168 | 0.9834 | 0.0272 | 0.9732 | 0.0286 | 0.9718 | 0.0178 | 0.9824 | 0.1074 |
