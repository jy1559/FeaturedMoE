# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 3.1898 | 1.6619 | 0.5706 | 0.7105 | 0.0000 |
| micro@1 | 2.6391 | 1.8833 | 0.6191 | 0.4956 | 0.3033 |
| mid@1 | 1.7374 | 2.4304 | 0.7597 | 0.4884 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0593 | 0.9425 | 0.0351 | 0.9655 | 0.0281 | 0.9723 |
| micro@1 | 0.0440 | 0.9570 | 0.0286 | 0.9718 | 0.0415 | 0.9594 |
| mid@1 | 0.0112 | 0.9888 | 0.0064 | 0.9936 | 0.0153 | 0.9849 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0748 | 0.9279 | 0.0798 | 0.9233 | 0.0854 | 0.9181 | 0.0759 | 0.9269 | 0.4856 |
| micro@1 | 0.0776 | 0.9253 | 0.0899 | 0.9141 | 0.0870 | 0.9166 | 0.0681 | 0.9342 | 0.6241 |
| mid@1 | 0.0261 | 0.9742 | 0.0229 | 0.9773 | 0.0207 | 0.9795 | 0.0622 | 0.9397 | 0.7770 |
