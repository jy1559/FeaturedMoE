# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:shuffle@eval:focus#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.7811 | 0.3362 | 0.2851 | 2.1899 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0170 | 0.9831 | 0.0089 | 0.9911 | 0.0080 | 0.9920 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0268 | 0.9736 | 0.0678 | 0.9345 | 0.0361 | 0.9646 | 0.0290 | 0.9714 | 0.1305 |
