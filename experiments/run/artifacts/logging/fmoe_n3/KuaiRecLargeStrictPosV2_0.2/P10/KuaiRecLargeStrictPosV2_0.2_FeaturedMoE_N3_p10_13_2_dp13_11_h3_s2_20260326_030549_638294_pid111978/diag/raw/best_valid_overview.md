# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:family_permute@train:focus#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.8484 | 0.1131 | 0.2486 | 2.2754 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0071 | 0.9929 | 0.0031 | 0.9969 | 0.0036 | 0.9964 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0135 | 0.9866 | 0.0228 | 0.9774 | 0.0227 | 0.9776 | 0.0160 | 0.9842 | 0.1043 |
