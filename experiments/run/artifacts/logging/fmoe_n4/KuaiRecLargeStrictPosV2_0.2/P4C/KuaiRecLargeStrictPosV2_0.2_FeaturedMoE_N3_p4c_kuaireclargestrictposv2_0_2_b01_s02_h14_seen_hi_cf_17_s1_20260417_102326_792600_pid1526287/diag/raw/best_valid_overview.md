# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:role_swap@both#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 5.7703 | 1.0390 | 0.8574 | 1.8460 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0318 | 0.9687 | 0.0028 | 0.9972 | 0.0317 | 0.9688 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0843 | 0.9191 | 0.0867 | 0.9169 | 0.0732 | 0.9294 | 0.0598 | 0.9420 | 0.3489 |
