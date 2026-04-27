# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 7.9846 | 0.0439 | 0.1895 | 1.9797 | 0.7570 |
| mid@1 | 7.9761 | 0.0548 | 0.2035 | 1.9152 | 0.6580 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0118 | 0.9883 | 0.0057 | 0.9943 | 0.0061 | 0.9940 |
| mid@1 | 0.0173 | 0.9828 | 0.0073 | 0.9927 | 0.0100 | 0.9901 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0147 | 0.9854 | 0.0140 | 0.9861 | 0.0144 | 0.9857 | 0.0136 | 0.9864 | 0.1361 |
| mid@1 | 0.0413 | 0.9596 | 0.0204 | 0.9798 | 0.0388 | 0.9620 | 0.0347 | 0.9659 | 0.1427 |
