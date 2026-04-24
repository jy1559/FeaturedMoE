# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 17.6549 | 0.3645 | 0.4066 | 2.8577 | 0.7679 |
| mid@1 | 11.3355 | 0.8743 | 0.3792 | 2.4203 | 0.8172 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0099 | 0.9902 | 0.0013 | 0.9987 | 0.0087 | 0.9914 |
| mid@1 | 0.0290 | 0.9715 | 0.0071 | 0.9930 | 0.0225 | 0.9778 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0129 | 0.9872 | 0.0131 | 0.9870 | 0.0141 | 0.9860 | 0.0116 | 0.9885 | 0.0933 |
| mid@1 | 0.0378 | 0.9629 | 0.0335 | 0.9671 | 0.0319 | 0.9686 | 0.0359 | 0.9648 | 0.1724 |
