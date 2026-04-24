# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.9718 | 0.0485 | 0.1400 | 2.4500 | 0.5966 |
| mid@1 | 11.3586 | 0.2376 | 0.2192 | 2.2186 | 0.5957 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0126 | 0.9874 | 0.0034 | 0.9966 | 0.0092 | 0.9909 |
| mid@1 | 0.0278 | 0.9726 | 0.0072 | 0.9928 | 0.0200 | 0.9802 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0118 | 0.9883 | 0.0119 | 0.9882 | 0.0117 | 0.9884 | 0.0117 | 0.9884 | 0.0906 |
| mid@1 | 0.0890 | 0.9148 | 0.0757 | 0.9271 | 0.0556 | 0.9459 | 0.0464 | 0.9546 | 0.1391 |
