# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8034 | 0.3328 | 0.3290 | 2.3057 | 0.8142 |
| mid@1 | 9.9645 | 0.4520 | 0.2700 | 2.2409 | 0.8042 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0169 | 0.9832 | 0.0055 | 0.9945 | 0.0114 | 0.9886 |
| mid@1 | 0.0160 | 0.9842 | 0.0034 | 0.9966 | 0.0124 | 0.9876 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0205 | 0.9797 | 0.0207 | 0.9795 | 0.0219 | 0.9783 | 0.0197 | 0.9805 | 0.1469 |
| mid@1 | 0.0206 | 0.9796 | 0.0184 | 0.9818 | 0.0170 | 0.9831 | 0.0202 | 0.9800 | 0.1481 |
