# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 8.5337 | 0.6373 | 0.2248 | 1.7268 | 0.0000 |
| micro@1 | 9.2109 | 0.5503 | 0.3946 | 1.7018 | 0.5771 |
| mid@1 | 8.4528 | 0.6478 | 0.3658 | 1.6149 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0290 | 0.9714 | 0.0111 | 0.9890 | 0.0353 | 0.9654 |
| micro@1 | 0.0065 | 0.9936 | 0.0035 | 0.9965 | 0.0150 | 0.9853 |
| mid@1 | 0.0394 | 0.9613 | 0.0176 | 0.9825 | 0.0501 | 0.9514 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0483 | 0.9528 | 0.0626 | 0.9394 | 0.0702 | 0.9322 | 0.0482 | 0.9529 | 0.1776 |
| micro@1 | 0.0122 | 0.9879 | 0.0210 | 0.9793 | 0.0174 | 0.9827 | 0.0232 | 0.9771 | 0.1771 |
| mid@1 | 0.0816 | 0.9216 | 0.0746 | 0.9281 | 0.0888 | 0.9151 | 0.0793 | 0.9238 | 0.2013 |
