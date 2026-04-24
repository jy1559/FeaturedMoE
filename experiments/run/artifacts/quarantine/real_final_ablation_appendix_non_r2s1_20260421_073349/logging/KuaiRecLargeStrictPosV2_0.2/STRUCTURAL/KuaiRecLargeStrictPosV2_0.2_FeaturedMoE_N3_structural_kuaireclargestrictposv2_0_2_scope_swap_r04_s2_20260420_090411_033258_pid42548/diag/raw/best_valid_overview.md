# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.9193 | 0.3146 | 0.3225 | 2.3185 | 0.8258 |
| mid@1 | 10.0485 | 0.4407 | 0.2817 | 2.2584 | 0.7841 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0161 | 0.9840 | 0.0052 | 0.9949 | 0.0110 | 0.9891 |
| mid@1 | 0.0149 | 0.9852 | 0.0032 | 0.9969 | 0.0117 | 0.9884 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0192 | 0.9810 | 0.0193 | 0.9809 | 0.0203 | 0.9799 | 0.0186 | 0.9816 | 0.1434 |
| mid@1 | 0.0190 | 0.9812 | 0.0170 | 0.9832 | 0.0158 | 0.9843 | 0.0185 | 0.9816 | 0.1478 |
