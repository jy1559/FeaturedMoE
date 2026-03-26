# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:role_swap@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8557 | 0.1103 | 0.2125 | 2.2890 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0061 | 0.9939 | 0.0049 | 0.9951 | 0.0012 | 0.9988 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0146 | 0.9855 | 0.0184 | 0.9817 | 0.0170 | 0.9831 | 0.0119 | 0.9881 | 0.1070 |
