# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.4915 | 0.2104 | 0.1500 | 1.6731 | 0.0000 |
| micro@1 | 9.9212 | 0.4577 | 0.2281 | 1.4596 | 0.5415 |
| mid@1 | 8.8619 | 0.5951 | 0.2099 | 1.5830 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0494 | 0.9518 | 0.0453 | 0.9557 | 0.0053 | 0.9947 |
| micro@1 | 0.0459 | 0.9551 | 0.0410 | 0.9598 | 0.0078 | 0.9922 |
| mid@1 | 0.0342 | 0.9664 | 0.0273 | 0.9731 | 0.0081 | 0.9919 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0627 | 0.9392 | 0.0737 | 0.9289 | 0.0801 | 0.9231 | 0.0670 | 0.9352 | 0.1122 |
| micro@1 | 0.0742 | 0.9285 | 0.0888 | 0.9150 | 0.0937 | 0.9106 | 0.0783 | 0.9247 | 0.1517 |
| mid@1 | 0.0745 | 0.9282 | 0.0689 | 0.9335 | 0.0759 | 0.9269 | 0.0661 | 0.9361 | 0.1814 |
