# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9137 | 0.0851 | 0.1543 | 2.4375 | 0.5777 |
| mid@1 | 11.2252 | 0.2627 | 0.1923 | 2.1265 | 0.6118 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0157 | 0.9844 | 0.0046 | 0.9954 | 0.0111 | 0.9890 |
| mid@1 | 0.0347 | 0.9659 | 0.0097 | 0.9904 | 0.0245 | 0.9758 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0157 | 0.9844 | 0.0158 | 0.9843 | 0.0157 | 0.9844 | 0.0156 | 0.9845 | 0.0933 |
| mid@1 | 0.1178 | 0.8889 | 0.0990 | 0.9058 | 0.0726 | 0.9299 | 0.0632 | 0.9388 | 0.1361 |
