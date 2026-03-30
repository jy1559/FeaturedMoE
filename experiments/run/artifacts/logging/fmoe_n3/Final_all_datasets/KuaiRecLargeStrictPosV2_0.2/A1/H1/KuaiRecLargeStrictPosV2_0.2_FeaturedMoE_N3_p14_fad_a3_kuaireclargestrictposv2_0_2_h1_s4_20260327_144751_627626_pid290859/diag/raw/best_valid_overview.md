# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.0947 | 0.4344 | 0.5480 | 2.2483 | 0.4273 |
| micro@1 | 10.3761 | 0.3956 | 0.3371 | 2.0349 | 0.5922 |
| mid@1 | 9.0430 | 0.5718 | 0.5224 | 2.0665 | 0.4309 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0167 | 0.9834 | 0.0055 | 0.9945 | 0.0103 | 0.9898 |
| micro@1 | 0.0235 | 0.9768 | 0.0217 | 0.9785 | 0.0023 | 0.9977 |
| mid@1 | 0.0181 | 0.9821 | 0.0119 | 0.9881 | 0.0061 | 0.9939 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0281 | 0.9722 | 0.0419 | 0.9590 | 0.0278 | 0.9725 | 0.0380 | 0.9627 | 0.1310 |
| micro@1 | 0.0514 | 0.9499 | 0.0453 | 0.9557 | 0.0547 | 0.9468 | 0.0277 | 0.9727 | 0.0863 |
| mid@1 | 0.0489 | 0.9523 | 0.0490 | 0.9522 | 0.0616 | 0.9402 | 0.0345 | 0.9661 | 0.1286 |
