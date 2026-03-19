# 260318 layout + feature summary (best valid MRR@20)

- deduped run rows: 371
- SASRec best valid MRR@20: 0.0785
- rule keyword hits in run_id (rule/teacher/gls): 0

## P3 layout options
| layout_option     |   n_runs |   valid_mean |   valid_max |   delta_vs_sasrec_valid | run_id                                   |
|:------------------|---------:|-------------:|------------:|------------------------:|:-----------------------------------------|
| S1_flat_standard  |        6 |      0.08105 |      0.0813 |                  0.0028 | p3s1_05_20260316_040932_600847_pid489221 |
| S2_factored_group |        5 |      0.08102 |      0.0811 |                  0.0026 | p3s2_02_20260315_145121_507105_pid408315 |
| S3_feature_source |        1 |      0.0809  |      0.0809 |                  0.0024 | p3s3_01_20260315_145121_618681_pid408317 |
| S4_deep_prefix    |        1 |      0.0804  |      0.0804 |                  0.0019 | p3s4_01_20260315_173010_132186_pid408819 |

## P1 family summary
| family              |   n_trials |   valid_mean |   valid_max |   delta_vs_sasrec_valid |
|:--------------------|-----------:|-------------:|------------:|------------------------:|
| A_anchor_dim_layout |         35 |    0.0788886 |      0.0811 |                  0.0026 |
| L_layer_layout      |         44 |    0.0784364 |      0.0807 |                  0.0022 |

## P1 top-10 run_phase
| run_phase   | family              |   n_trials |   valid_mean |   valid_max |   delta_vs_sasrec_valid |
|:------------|:--------------------|-----------:|-------------:|------------:|------------------------:|
| A05         | A_anchor_dim_layout |          4 |    0.0808    |      0.0811 |                  0.0026 |
| A06         | A_anchor_dim_layout |          4 |    0.080425  |      0.0806 |                  0.0021 |
| A01         | A_anchor_dim_layout |          5 |    0.0802    |      0.0803 |                  0.0018 |
| L01         | L_layer_layout      |          2 |    0.0802    |      0.0806 |                  0.0021 |
| L06         | L_layer_layout      |         10 |    0.0801    |      0.0805 |                  0.002  |
| A02         | A_anchor_dim_layout |          7 |    0.0800143 |      0.0802 |                  0.0017 |
| L05         | L_layer_layout      |          6 |    0.07995   |      0.0802 |                  0.0017 |
| L07         | L_layer_layout      |          5 |    0.07994   |      0.0804 |                  0.0019 |
| L08         | L_layer_layout      |          7 |    0.0799143 |      0.0807 |                  0.0022 |
| L04         | L_layer_layout      |          6 |    0.0790667 |      0.0795 |                  0.001  |

## feature path (hidden/feature/both/injection)
| path_type                  |   n_runs |   valid_mean |   valid_max |   delta_vs_sasrec_valid | run_id                                                   |
|:---------------------------|---------:|-------------:|------------:|------------------------:|:---------------------------------------------------------|
| hidden_plus_feature_router |        1 |       0.0813 |      0.0813 |                  0.0028 | f_feat_full_c1_20260317_024302_639648_pid55543           |
| feature_only_router        |        1 |       0.0807 |      0.0807 |                  0.0022 | f_feat_feature_only_c4_20260317_032511_638577_pid78050   |
| injection_only             |        1 |       0.0806 |      0.0806 |                  0.0021 | f_feat_injection_only_c3_20260317_035417_654766_pid89434 |
| hidden_only_router         |        1 |       0.0792 |      0.0792 |                  0.0007 | f_feat_hidden_only_c2_20260317_024343_242750_pid55988    |

## P6 router x injection combo
| combo                |   n_runs |   valid_mean |   valid_max |   delta_vs_sasrec_valid |
|:---------------------|---------:|-------------:|------------:|------------------------:|
| standard+gated       |        2 |    0.0807    |      0.0812 |                  0.0027 |
| factored+gated       |        2 |    0.08045   |      0.0808 |                  0.0023 |
| factored+group_gated |        2 |    0.08035   |      0.0808 |                  0.0023 |
| standard+group_gated |        3 |    0.0634333 |      0.0802 |                  0.0017 |

## P6 context (X1/X2)
| context   |   n_runs |   valid_mean |   valid_max |   delta_vs_sasrec_valid |
|:----------|---------:|-------------:|------------:|------------------------:|
| X2        |        4 |     0.080075 |      0.0802 |                  0.0017 |
| X1        |        5 |     0.0706   |      0.0812 |                  0.0027 |