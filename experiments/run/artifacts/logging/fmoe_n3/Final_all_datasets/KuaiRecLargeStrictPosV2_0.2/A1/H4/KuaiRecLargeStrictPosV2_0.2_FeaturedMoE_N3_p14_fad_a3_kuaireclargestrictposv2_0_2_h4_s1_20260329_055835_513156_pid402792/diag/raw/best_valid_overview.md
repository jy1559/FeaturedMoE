# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.1438 | 0.4278 | 0.6250 | 2.2960 | 0.6367 |
| micro@1 | 11.1397 | 0.2779 | 0.5809 | 2.3086 | 0.5516 |
| mid@1 | 7.5782 | 0.7639 | 0.7800 | 2.1151 | 0.1369 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0065 | 0.9935 | 0.0060 | 0.9940 | 0.0004 | 0.9996 |
| micro@1 | 0.0121 | 0.9880 | 0.0119 | 0.9882 | 0.0001 | 0.9999 |
| mid@1 | 0.0127 | 0.9873 | 0.0115 | 0.9886 | 0.0010 | 0.9990 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0104 | 0.9896 | 0.0130 | 0.9871 | 0.0117 | 0.9884 | 0.0087 | 0.9913 | 0.1265 |
| micro@1 | 0.0150 | 0.9851 | 0.0205 | 0.9797 | 0.0138 | 0.9863 | 0.0183 | 0.9819 | 0.1026 |
| mid@1 | 0.0302 | 0.9703 | 0.0382 | 0.9625 | 0.0192 | 0.9810 | 0.0422 | 0.9587 | 0.2138 |
