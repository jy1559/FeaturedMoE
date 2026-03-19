# Diagnostic Overview

- split: best_valid
- feature_mode: none

## Base Metrics

| stage | n_eff | cv_usage | top1_max | entropy | jitter_adj |
|---|---:|---:|---:|---:|---:|
| macro@1 | 11.3983 | 0.2298 | 0.2693 | 2.0913 | 0.0000 |
| micro@1 | 10.0222 | 0.4442 | 0.3009 | 1.6345 | 0.5398 |
| mid@1 | 9.3075 | 0.5379 | 0.4186 | 1.9490 | 0.0000 |

## KNN Core

| stage | knn_js | knn_score | group_knn_js | group_knn_score | intra_group_knn_mean_js | intra_group_knn_mean_score |
|---|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0247 | 0.9756 | 0.0124 | 0.9877 | 0.0129 | 0.9872 |
| micro@1 | 0.0292 | 0.9712 | 0.0115 | 0.9886 | 0.0190 | 0.9812 |
| mid@1 | 0.0196 | 0.9806 | 0.0068 | 0.9933 | 0.0109 | 0.9892 |

## Feature-Group KNN

| stage | tempo_knn_js | tempo_knn_score | focus_knn_js | focus_knn_score | memory_knn_js | memory_knn_score | exposure_knn_js | exposure_knn_score | family_top_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| macro@1 | 0.0341 | 0.9665 | 0.0422 | 0.9587 | 0.0440 | 0.9570 | 0.0361 | 0.9646 | 0.1203 |
| micro@1 | 0.0497 | 0.9516 | 0.0579 | 0.9438 | 0.0575 | 0.9441 | 0.0553 | 0.9462 | 0.1904 |
| mid@1 | 0.0331 | 0.9674 | 0.0313 | 0.9692 | 0.0374 | 0.9633 | 0.0421 | 0.9588 | 0.2099 |
