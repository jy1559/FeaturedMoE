# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| mid@1 | 15.7266 | 0.1318 | 0.1007 | 2.2789 | 0.7350 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.1037 | 0.9015 | 0.0302 | 0.9703 | 0.0751 | 0.9277 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid@1 | 0.1267 | 0.8810 | 0.1258 | 0.8817 | 0.1246 | 0.8829 | 0.1136 | 0.8926 | 0.0780 |
