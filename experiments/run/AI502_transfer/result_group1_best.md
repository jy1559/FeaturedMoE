# AI502 group1 best-hparam 결과 요약

- 기준: target native scratch와 같은 `setting_id`, 같은 seed의 test MRR@20 비교
- full CSV: `artifacts_group1_best/analysis/group1_best_full.csv`
- 집계 CSV: `artifacts_group1_best/analysis/group1_best_summary.csv`

## 상위 gain 조합

| pair | setting | mode | policy | n | mean gain | win-rate |
|---|---|---|---|---:|---:|---:|
| retail_rocket_to_beauty | beauty_ab_h13_low_feat_dropout | feature_encoder_a12_router_init | loaded_lr_0.05 | 5 | 0.005700 | 0.80 |
| retail_rocket_to_beauty | beauty_ab_h13_low_feat_dropout | feature_encoder_a12_router_init | loaded_lr_0.35 | 5 | 0.004020 | 0.60 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_a12_router_init | loaded_lr_0.05 | 5 | 0.002920 | 0.80 |
| lastfm_to_KuaiRec | kuairec_h10_long_context | feature_encoder_group_router_init | std | 5 | 0.002800 | 1.00 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_init | loaded_lr_0.05 | 5 | 0.002500 | 0.80 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | full_except_feature_router_init | std | 5 | 0.002480 | 1.00 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_a12_router_init | loaded_lr_0.35 | 5 | 0.002440 | 0.80 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_group_router_init | loaded_lr_0.05 | 5 | 0.002380 | 1.00 |
| foursquare_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_init | loaded_lr_0.35 | 5 | 0.002360 | 0.60 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | group_router_init | loaded_lr_0.35 | 5 | 0.002260 | 0.80 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_group_router_init | loaded_lr_0.35 | 5 | 0.002240 | 1.00 |
| foursquare_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_init | loaded_lr_0.05 | 5 | 0.002200 | 0.60 |
| lastfm_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_init | loaded_lr_0.35 | 5 | 0.002080 | 0.80 |
| lastfm_to_KuaiRec | kuairec_h10_long_context | feature_encoder_group_router_init | loaded_lr_0.35 | 5 | 0.001960 | 1.00 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_init | loaded_lr_0.35 | 5 | 0.001960 | 0.80 |
| foursquare_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_group_router_init | std | 5 | 0.001900 | 1.00 |
| foursquare_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_init | std | 5 | 0.001840 | 0.60 |
| lastfm_to_KuaiRec | kuairec_h14_feature_strong | feature_encoder_init | std | 5 | 0.001780 | 0.80 |
| beauty_to_retail_rocket | retail_r10_h13_width_refine | feature_encoder_init | loaded_lr_0.05 | 5 | 0.001740 | 0.60 |
| beauty_to_retail_rocket | retail_r15_h13_width_lr_validate | feature_encoder_init | std | 5 | 0.001700 | 0.80 |
