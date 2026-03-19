# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 4.3912 | 1.3163 | 0.3435 | 0.6901 | 0.0000 |
| micro@1 | 4.0712 | 1.3955 | 0.4434 | 0.5614 | 0.3928 |
| mid@1 | 1.7403 | 2.4280 | 0.7591 | 0.4561 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0673 | 0.9349 | 0.0407 | 0.9601 | 0.0366 | 0.9641 |
| micro@1 | 0.0494 | 0.9518 | 0.0247 | 0.9756 | 0.0424 | 0.9585 |
| mid@1 | 0.0133 | 0.9868 | 0.0072 | 0.9928 | 0.0197 | 0.9805 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0861 | 0.9175 | 0.0943 | 0.9100 | 0.1011 | 0.9038 | 0.0849 | 0.9186 | 0.3301 |
| micro@1 | 0.0828 | 0.9206 | 0.0970 | 0.9076 | 0.0987 | 0.9060 | 0.0745 | 0.9282 | 0.4417 |
| mid@1 | 0.0287 | 0.9717 | 0.0251 | 0.9752 | 0.0214 | 0.9788 | 0.0662 | 0.9360 | 0.7770 |
