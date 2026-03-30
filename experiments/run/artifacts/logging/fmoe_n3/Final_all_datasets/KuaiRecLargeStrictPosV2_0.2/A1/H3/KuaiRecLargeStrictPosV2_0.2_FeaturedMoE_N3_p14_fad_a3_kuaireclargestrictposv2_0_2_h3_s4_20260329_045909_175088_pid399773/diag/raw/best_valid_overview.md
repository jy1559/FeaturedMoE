# Diagnostic Overview

- split: best_valid
- feature_mode: perturb:zero@both:kw=cat+theme#shift1

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 10.8423 | 0.3268 | 0.4658 | 2.3210 | 0.4946 |
| micro@1 | 10.8012 | 0.3332 | 0.3029 | 2.2393 | 0.7445 |
| mid@1 | 9.3580 | 0.5313 | 0.6009 | 2.1908 | 0.4624 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0125 | 0.9876 | 0.0118 | 0.9882 | 0.0007 | 0.9993 |
| micro@1 | 0.0142 | 0.9859 | 0.0133 | 0.9868 | 0.0009 | 0.9991 |
| mid@1 | 0.0119 | 0.9881 | 0.0103 | 0.9897 | 0.0015 | 0.9985 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0194 | 0.9808 | 0.0281 | 0.9723 | 0.0237 | 0.9766 | 0.0172 | 0.9829 | 0.1018 |
| micro@1 | 0.0168 | 0.9833 | 0.0274 | 0.9730 | 0.0165 | 0.9836 | 0.0207 | 0.9795 | 0.0976 |
| mid@1 | 0.0280 | 0.9724 | 0.0371 | 0.9636 | 0.0201 | 0.9801 | 0.0393 | 0.9615 | 0.1592 |
