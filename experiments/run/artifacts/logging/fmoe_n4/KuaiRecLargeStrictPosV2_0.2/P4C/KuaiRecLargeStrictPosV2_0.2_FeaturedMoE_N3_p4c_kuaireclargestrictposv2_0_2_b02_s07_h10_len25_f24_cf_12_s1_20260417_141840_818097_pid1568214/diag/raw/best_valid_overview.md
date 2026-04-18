# Diagnostic Overview

- split: best_valid
- feature_mode: runtime:unknown|ablation:perturb:shuffle@eval#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4168 | 0.2260 | 0.3551 | 2.3940 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0151 | 0.9850 | 0.0050 | 0.9950 | 0.0091 | 0.9909 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0262 | 0.9741 | 0.0288 | 0.9716 | 0.0304 | 0.9700 | 0.0270 | 0.9733 | 0.1579 |
