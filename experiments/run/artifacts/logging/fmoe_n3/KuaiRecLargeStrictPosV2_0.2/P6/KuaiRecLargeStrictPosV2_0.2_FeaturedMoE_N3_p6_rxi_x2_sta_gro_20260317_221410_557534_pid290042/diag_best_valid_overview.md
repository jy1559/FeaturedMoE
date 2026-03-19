# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 9.8382 | 0.4688 | 0.1811 | 1.4615 | 0.0000 |
| micro@1 | 5.7467 | 1.0431 | 0.3398 | 0.8062 | 0.5543 |
| mid@1 | 6.2819 | 0.9541 | 0.2562 | 1.2021 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0647 | 0.9373 | 0.0264 | 0.9740 | 0.0561 | 0.9454 |
| micro@1 | 0.0608 | 0.9410 | 0.0239 | 0.9763 | 0.0544 | 0.9470 |
| mid@1 | 0.0389 | 0.9618 | 0.0171 | 0.9831 | 0.0616 | 0.9408 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0834 | 0.9200 | 0.0910 | 0.9130 | 0.0986 | 0.9061 | 0.0793 | 0.9237 | 0.1641 |
| micro@1 | 0.0875 | 0.9162 | 0.1054 | 0.9000 | 0.1112 | 0.8948 | 0.0792 | 0.9238 | 0.2989 |
| mid@1 | 0.0659 | 0.9362 | 0.0559 | 0.9456 | 0.0651 | 0.9370 | 0.0927 | 0.9115 | 0.2259 |
