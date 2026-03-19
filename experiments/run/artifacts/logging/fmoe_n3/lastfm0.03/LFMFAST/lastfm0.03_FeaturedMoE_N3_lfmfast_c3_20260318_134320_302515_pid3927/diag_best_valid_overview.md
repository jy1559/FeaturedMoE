# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.6173 | 0.3609 | 0.1289 | 1.4862 | 0.0000 |
| micro@1 | 10.8185 | 0.3305 | 0.1319 | 1.5228 | 0.4973 |
| mid@1 | 10.8663 | 0.3230 | 0.1727 | 1.3691 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0887 | 0.9151 | 0.0390 | 0.9617 | 0.0799 | 0.9233 |
| micro@1 | 0.0558 | 0.9457 | 0.0180 | 0.9821 | 0.0510 | 0.9504 |
| mid@1 | 0.0824 | 0.9209 | 0.0354 | 0.9653 | 0.0813 | 0.9221 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.1294 | 0.8786 | 0.1235 | 0.8838 | 0.1269 | 0.8808 | 0.1181 | 0.8886 | 0.1357 |
| micro@1 | 0.1549 | 0.8565 | 0.1622 | 0.8503 | 0.2030 | 0.8163 | 0.1175 | 0.8892 | 0.1350 |
| mid@1 | 0.1431 | 0.8666 | 0.1255 | 0.8821 | 0.1605 | 0.8517 | 0.1339 | 0.8746 | 0.1294 |
