# Stage H Prep: Baseline A-G + FeaturedMoE_N3 A6 Best-of Summary

Scope: baseline `StageA`~`StageG` only for 9 baseline models, plus `fmoe_n3/Final_all_datasets` architecture `A6` for `FeaturedMoE_N3`.

Generated CSV: `/workspace/jy1559/FMoE/experiments/run/baseline/docs/stageH_AtoG_plus_fmoeA6_best_table_20260409.csv`

## 6x10 Best Table

| Dataset | Model | Best Valid MRR@20 | Test MRR@20 | MRR Run | Best Valid HR@10 | Test HR@10 | HR Run |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| KuaiRecLargeStrictPosV2_0.2 | SASRec | 0.1188 | 0.1107 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s3 | 0.1441 | 0.1388 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s3 |
| KuaiRecLargeStrictPosV2_0.2 | GRU4Rec | 0.0277 | 0.0250 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 | 0.0399 | 0.0390 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | TiSASRec | 0.0455 | 0.0435 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s3 | 0.0519 | 0.0507 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | DuoRec | 0.0505 | 0.0473 | StageG_Cross6x9_anchor2_core5 / FAD_H1_R2 / s2 | 0.0693 | 0.0666 | StageG_Cross6x9_anchor2_core5 / FAD_H1_R2 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | SIGMA | 0.1667 | 0.1661 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R2 / s1 | 0.1693 | 0.1672 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s3 |
| KuaiRecLargeStrictPosV2_0.2 | BSARec | 0.1651 | 0.1641 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R2 / s3 | 0.1673 | 0.1678 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R1 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | FEARec | 0.0794 | 0.0761 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 | 0.0863 | 0.0832 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | DIF-SR | 0.1475 | 0.1449 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R1 / s1 | 0.1583 | 0.1571 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R2 / s1 |
| KuaiRecLargeStrictPosV2_0.2 | FAME | 0.1591 | 0.1601 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s2 | 0.1660 | 0.1661 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s2 |
| KuaiRecLargeStrictPosV2_0.2 | FeaturedMoE_N3 | 0.1707 | 0.1690 | P14_A6_Final_all_datasets / H14 / s1 | 0.1777 | 0.1748 | P14_A6_Final_all_datasets / H3 / s3 |
| lastfm0.03 | SASRec | 0.2321 | 0.2298 | StageC_Focus_anchor2_core5 / B6_C1 / s1 | 0.3019 | 0.2991 | StageG_Cross6x9_anchor2_core5 / OLD_B6_C1 / s1 |
| lastfm0.03 | GRU4Rec | 0.2048 | 0.2012 | StageD_MicroWide_anchor2_core5 / C7_D5 / s1 | 0.2490 | 0.2505 | StageD_MicroWide_anchor2_core5 / C7_D5 / s1 |
| lastfm0.03 | TiSASRec | 0.2260 | 0.2249 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s2 | 0.2774 | 0.2782 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 |
| lastfm0.03 | DuoRec | 0.2324 | 0.2319 | StageE_ReLRSeed_anchor2_core5 / D1_E4 / s1 | 0.3035 | 0.2946 | StageB_Structure_anchor2_core5 / B2 / s1 |
| lastfm0.03 | SIGMA | 0.1883 | 0.1907 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s2 | 0.2283 | 0.2253 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 |
| lastfm0.03 | BSARec | 0.2198 | 0.2212 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s3 | 0.2586 | 0.2657 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R2 / s3 |
| lastfm0.03 | FEARec | 0.2269 | 0.2259 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 | 0.2976 | 0.2946 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 |
| lastfm0.03 | DIF-SR | 0.2410 | 0.2426 | StageB_Structure_anchor2_core5 / B7 / s1 | 0.2998 | 0.2981 | StageD_MicroWide_anchor2_core5 / C8_D5 / s1 |
| lastfm0.03 | FAME | 0.2242 | 0.2272 | StageE_ReLRSeed_anchor2_core5 / D6_E5 / s1 | 0.2700 | 0.2726 | StageF_TailBoost_anchor2_core5 / D2>E4_F2 / s1 |
| lastfm0.03 | FeaturedMoE_N3 | 0.2389 | 0.2352 | P14_A6_Final_all_datasets / H11 / s3 | 0.2990 | 0.2917 | P14_A6_Final_all_datasets / H6 / s2 |
| amazon_beauty | SASRec | 0.1264 | 0.0882 | StageC_Focus_anchor2_core5 / B2_C6 / s1 | 0.1770 | 0.1053 | StageC_Focus_anchor2_core5 / B2_C2 / s1 |
| amazon_beauty | GRU4Rec | 0.0197 | 0.0018 | StageD_MicroWide_anchor2_core5 / C7_D7 / s1 | 0.0265 | 0.0263 | StageD_MicroWide_anchor2_core5 / C8_D8 / s1 |
| amazon_beauty | TiSASRec | 0.0470 | 0.0135 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 | 0.0708 | 0.0263 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R2 / s2 |
| amazon_beauty | DuoRec | 0.1245 | 0.0863 | StageF_TailBoost_anchor2_core5 / MANUAL_AB_DUO_ALT_F1 / s1 | 0.1681 | 0.0965 | StageF_TailBoost_anchor2_core5 / MANUAL_AB_DUO_ALT_F6 / s2 |
| amazon_beauty | SIGMA | 0.0980 | 0.0673 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R2 / s2 | 0.0973 | 0.0789 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 |
| amazon_beauty | BSARec | 0.0162 | 0.0029 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 | 0.0265 | 0.0088 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 |
| amazon_beauty | FEARec | 0.1156 | 0.0899 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s1 | 0.1504 | 0.1053 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 |
| amazon_beauty | DIF-SR | 0.1062 | 0.0702 | StageD_MicroWide_anchor2_core5 / C8_D1 / s1 | 0.1239 | 0.0789 | StageC_Focus_anchor2_core5 / B7_C8 / s1 |
| amazon_beauty | FAME | 0.0265 | 0.0000 | StageD_MicroWide_anchor2_core5 / C7_D3 / s1 | 0.0442 | 0.0000 | StageC_Focus_anchor2_core5 / B1_C4 / s1 |
| amazon_beauty | FeaturedMoE_N3 | 0.1101 | 0.0726 | P14_A6_Final_all_datasets / H14 / s2 | 0.1327 | 0.0877 | P14_A6_Final_all_datasets / H14 / s1 |
| foursquare | SASRec | 0.1214 | 0.1166 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 | 0.2152 | 0.2115 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 |
| foursquare | GRU4Rec | 0.0846 | 0.0755 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 | 0.1343 | 0.1335 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s2 |
| foursquare | TiSASRec | 0.1131 | 0.1070 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 | 0.1932 | 0.1910 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 |
| foursquare | DuoRec | 0.1153 | 0.1074 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s3 | 0.2087 | 0.2063 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s3 |
| foursquare | SIGMA | 0.0813 | 0.0729 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R1 / s1 | 0.1330 | 0.1256 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R2 / s2 |
| foursquare | BSARec | 0.0858 | 0.0758 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R1 / s2 | 0.1445 | 0.1327 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R1 / s2 |
| foursquare | FEARec | 0.1058 | 0.0997 | StageG_Cross6x9_anchor2_core5 / FAD_H1_R1 / s1 | 0.1961 | 0.1910 | StageG_Cross6x9_anchor2_core5 / FAD_H2_R2 / s1 |
| foursquare | DIF-SR | 0.0987 | 0.0917 | StageG_Cross6x9_anchor2_core5 / FAD_H3_R1 / s1 | 0.1742 | 0.1763 | StageG_Cross6x9_anchor2_core5 / FAD_H4_R2 / s2 |
| foursquare | FAME | 0.0988 | 0.0915 | StageG_Cross6x9_anchor2_core5 / FAD_H1_R1 / s3 | 0.1598 | 0.1561 | StageG_Cross6x9_anchor2_core5 / FAD_H1_R1 / s2 |
| foursquare | FeaturedMoE_N3 | 0.1169 | 0.1160 | P14_A6_Final_all_datasets / H15 / s1 | 0.2142 | 0.2126 | P14_A6_Final_all_datasets / H15 / s2 |
| movielens1m | SASRec | 0.0778 | 0.0668 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s2 | 0.1898 | 0.1850 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s1 |
| movielens1m | GRU4Rec | 0.0761 | 0.0739 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s2 | 0.1768 | 0.1694 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s2 |
| movielens1m | TiSASRec | 0.0870 | 0.0743 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H2_R1_C1 / s2 | 0.1939 | 0.1874 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H2_R1_C1 / s3 |
| movielens1m | DuoRec | 0.0747 | 0.0661 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H2_R1_C2 / s3 | 0.1828 | 0.1786 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H2_R1_C1 / s2 |
| movielens1m | SIGMA | 0.0847 | 0.0740 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H2_R1_C2 / s2 | 0.1805 | 0.1855 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H2_R1_C2 / s3 |
| movielens1m | BSARec | 0.0876 | 0.0782 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H3_R1_C1 / s1 | 0.1828 | 0.1832 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H3_R1_C1 / s3 |
| movielens1m | FEARec | 0.0679 | 0.0623 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H1_R1_C2 / s1 | 0.1754 | 0.1689 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H2_R1_C1 / s2 |
| movielens1m | DIF-SR | 0.0850 | 0.0791 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s3 | 0.1944 | 0.1786 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s1 |
| movielens1m | FAME | 0.0831 | 0.0766 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H3_R1_C1 / s2 | 0.1791 | 0.1781 | StageG_Cross6x9_anchor2_core5 / XFER_LASTFM0_03_FAD_H3_R1_C1 / s1 |
| movielens1m | FeaturedMoE_N3 | 0.0835 | 0.0750 | P14_A6_Final_all_datasets / H9 / s3 | 0.1934 | 0.1864 | P14_A6_Final_all_datasets / H9 / s3 |
| retail_rocket | SASRec | 0.2629 | 0.2635 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s3 | 0.4201 | 0.4249 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H3_R1_C1 / s3 |
| retail_rocket | GRU4Rec | 0.2378 | 0.2373 | StageG_Cross6x9_anchor2_core5 / AGGR_SPARSE_GRU_G2 / s2 | 0.3795 | 0.3786 | StageG_Cross6x9_anchor2_core5 / AGGR_SPARSE_GRU_G2 / s1 |
| retail_rocket | TiSASRec | 0.2601 | 0.2627 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s1 | 0.4211 | 0.4234 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H2_R1_C1 / s3 |
| retail_rocket | DuoRec | 0.2730 | 0.2729 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H2_R1_C1 / s1 | 0.4313 | 0.4337 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H2_R1_C1 / s3 |
| retail_rocket | SIGMA | 0.3548 | 0.3534 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H2_R1_C1 / s1 | 0.4168 | 0.4158 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H2_R1_C1 / s2 |
| retail_rocket | BSARec | 0.3652 | 0.3640 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H3_R1_C1 / s1 | 0.4343 | 0.4327 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H3_R1_C1 / s1 |
| retail_rocket | FEARec | 0.2797 | 0.2799 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H1_R1_C2 / s2 | 0.4367 | 0.4375 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H1_R1_C2 / s1 |
| retail_rocket | DIF-SR | 0.3726 | 0.3709 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s2 | 0.4422 | 0.4400 | StageG_Cross6x9_anchor2_core5 / XFER_FOURSQUARE_FAD_H3_R1_C2 / s2 |
| retail_rocket | FAME | 0.3547 | 0.3530 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H1_R1_C1 / s1 | 0.4084 | 0.4070 | StageG_Cross6x9_anchor2_core5 / XFER_AMAZON_BEAUTY_FAD_H1_R1_C1 / s2 |
| retail_rocket | FeaturedMoE_N3 | 0.2987 | 0.2994 | P14_A6_Final_all_datasets / H2 / s2 | 0.4604 | 0.4608 | P14_A6_Final_all_datasets / H2 / s2 |

## Quick Read

- `KuaiRecLargeStrictPosV2_0.2` top MRR: FeaturedMoE_N3 0.1707, SIGMA 0.1667, BSARec 0.1651
- `KuaiRecLargeStrictPosV2_0.2` low MRR: DuoRec 0.0505, TiSASRec 0.0455, GRU4Rec 0.0277
- `lastfm0.03` top MRR: DIF-SR 0.2410, FeaturedMoE_N3 0.2389, DuoRec 0.2324
- `lastfm0.03` low MRR: BSARec 0.2198, GRU4Rec 0.2048, SIGMA 0.1883
- `amazon_beauty` top MRR: SASRec 0.1264, DuoRec 0.1245, FEARec 0.1156
- `amazon_beauty` low MRR: FAME 0.0265, GRU4Rec 0.0197, BSARec 0.0162
- `foursquare` top MRR: SASRec 0.1214, FeaturedMoE_N3 0.1169, DuoRec 0.1153
- `foursquare` low MRR: BSARec 0.0858, GRU4Rec 0.0846, SIGMA 0.0813
- `movielens1m` top MRR: BSARec 0.0876, TiSASRec 0.0870, DIF-SR 0.0850
- `movielens1m` low MRR: GRU4Rec 0.0761, DuoRec 0.0747, FEARec 0.0679
- `retail_rocket` top MRR: DIF-SR 0.3726, BSARec 0.3652, SIGMA 0.3548
- `retail_rocket` low MRR: SASRec 0.2629, TiSASRec 0.2601, GRU4Rec 0.2378

## Low-Performance Combos For Extra Search

Below threshold: best valid MRR@20 < 45% of the dataset-best among the 10 models.

- `amazon_beauty` / `BSARec`: best_valid_mrr20=0.0162 (12.8% of dataset-best 0.1264), best from `StageG_Cross6x9_anchor2_core5` `FAD_H3_R1` lr=0.000373
- `amazon_beauty` / `GRU4Rec`: best_valid_mrr20=0.0197 (15.6% of dataset-best 0.1264), best from `StageD_MicroWide_anchor2_core5` `C7_D7` lr=0.001167
- `KuaiRecLargeStrictPosV2_0.2` / `GRU4Rec`: best_valid_mrr20=0.0277 (16.2% of dataset-best 0.1707), best from `StageG_Cross6x9_anchor2_core5` `FAD_H2_R2` lr=0.003501
- `amazon_beauty` / `FAME`: best_valid_mrr20=0.0265 (21.0% of dataset-best 0.1264), best from `StageD_MicroWide_anchor2_core5` `C7_D3` lr=0.005019
- `KuaiRecLargeStrictPosV2_0.2` / `TiSASRec`: best_valid_mrr20=0.0455 (26.7% of dataset-best 0.1707), best from `StageG_Cross6x9_anchor2_core5` `FAD_H2_R2` lr=0.000587
- `KuaiRecLargeStrictPosV2_0.2` / `DuoRec`: best_valid_mrr20=0.0505 (29.6% of dataset-best 0.1707), best from `StageG_Cross6x9_anchor2_core5` `FAD_H1_R2` lr=0.000613
- `amazon_beauty` / `TiSASRec`: best_valid_mrr20=0.0470 (37.2% of dataset-best 0.1264), best from `StageG_Cross6x9_anchor2_core5` `FAD_H2_R1` lr=0.000388

## LR / Hparam Tendencies

- `SASRec`: winning lr range 0.000477 .. 0.001910; strongest picks retail_rocket:XFER_FOURSQUARE_FAD_H3_R1_C2@0.000701, lastfm0.03:B6_C1@0.000911, amazon_beauty:B2_C6@0.001910
- `GRU4Rec`: winning lr range 0.001167 .. 0.009830; strongest picks retail_rocket:AGGR_SPARSE_GRU_G2@0.009830, lastfm0.03:C7_D5@0.008015, foursquare:FAD_H2_R2@0.003815
- `TiSASRec`: winning lr range 0.000297 .. 0.000792; strongest picks retail_rocket:XFER_FOURSQUARE_FAD_H3_R1_C2@0.000560, lastfm0.03:FAD_H2_R1@0.000297, foursquare:FAD_H3_R1@0.000792
- `DuoRec`: winning lr range 0.000246 .. 0.000613; strongest picks retail_rocket:XFER_AMAZON_BEAUTY_FAD_H2_R1_C1@0.000431, lastfm0.03:D1_E4@0.000246, amazon_beauty:MANUAL_AB_DUO_ALT_F1@0.000556
- `SIGMA`: winning lr range 0.000159 .. 0.000801; strongest picks retail_rocket:XFER_AMAZON_BEAUTY_FAD_H2_R1_C1@0.000192, lastfm0.03:FAD_H2_R1@0.000261, KuaiRecLargeStrictPosV2_0.2:FAD_H4_R2@0.000801
- `BSARec`: winning lr range 0.000373 .. 0.001032; strongest picks retail_rocket:XFER_AMAZON_BEAUTY_FAD_H3_R1_C1@0.000392, lastfm0.03:FAD_H3_R1@0.000457, KuaiRecLargeStrictPosV2_0.2:FAD_H3_R2@0.001032
- `FEARec`: winning lr range 0.000176 .. 0.000705; strongest picks retail_rocket:XFER_FOURSQUARE_FAD_H1_R1_C2@0.000705, lastfm0.03:FAD_H2_R1@0.000176, amazon_beauty:FAD_H2_R2@0.000352
- `DIF-SR`: winning lr range 0.000656 .. 0.003166; strongest picks retail_rocket:XFER_FOURSQUARE_FAD_H3_R1_C2@0.000658, lastfm0.03:B7@0.000980, KuaiRecLargeStrictPosV2_0.2:FAD_H4_R1@0.000656
- `FAME`: winning lr range 0.000253 .. 0.005019; strongest picks retail_rocket:XFER_AMAZON_BEAUTY_FAD_H1_R1_C1@0.000559, lastfm0.03:D6_E5@0.000487, KuaiRecLargeStrictPosV2_0.2:FAD_H3_R1@0.000788
- `FeaturedMoE_N3`: winning lr range 0.000407 .. 0.002561; strongest picks lastfm0.03:H11@0.000407, KuaiRecLargeStrictPosV2_0.2:H14@0.000490, foursquare:H15@0.000469

## Stage H Search Suggestions

- Keep `StageG` winners as anchor for strong combos; do not expand search on combos already near dataset-best.
- Focus Stage H budget on combos in the low-performance list, especially sparse datasets (`amazon_beauty`, `retail_rocket`) and transfer-sensitive models (`GRU4Rec`, `FAME`, `FEARec`, sometimes `DuoRec`).
- For sparse weak combos, widen LR upward first: use one conservative band around current best lr and one aggressive band about `1.8x~3.0x` higher if current best lr is below about `1e-3`.
- For sequential Transformer families (`SASRec`, `TiSASRec`, `BSARec`, `DIF-SR`), bad results usually came less from LR collapse and more from transferred structure mismatch. Search `max_len`, dropout, and hidden size jointly with a still-narrow LR band.
- For `GRU4Rec`, test shorter `max_len`, lower dropout, and moderately larger hidden size on sparse datasets. Recovery-style lrs around low-`1e-3` to several-`1e-3` looked more viable than sub-`5e-4` settings.
- For `FAME`, keep the sparse recovery path separate from dense datasets. Search `num_experts`, `hidden/inner size`, and a fairly high lr band; the underperforming runs are unlikely to recover from LR-only tweaks.
- For `FEARec` and `DuoRec`, add one Stage H candidate with stronger regularization relief (`dropout` down, semantic/contrastive weights down slightly) because some transferred settings appear over-regularized outside their source dataset.
- For `FeaturedMoE_N3` A6, the logged winners often sat around roughly `1e-3` on transfer datasets. If H includes FMoE comparison refresh, search near current A6 winners rather than reopening architecture search.

## Recommended Stage H Priority

- Priority 1: `amazon_beauty / BSARec`, `amazon_beauty / GRU4Rec`, `amazon_beauty / FAME`, `amazon_beauty / TiSASRec`
- Priority 1 rationale: these are far below the dataset-best and are the clearest missed-search cases rather than simple model ceiling.
- Priority 2: `KuaiRecLargeStrictPosV2_0.2 / GRU4Rec`, `KuaiRecLargeStrictPosV2_0.2 / TiSASRec`, `KuaiRecLargeStrictPosV2_0.2 / DuoRec`
- Priority 2 rationale: the gap is too large relative to nearby model families, so Stage G transfer-derived candidates likely did not cover the right local basin.
- Priority 3: `movielens1m` broad refresh for `FEARec`, `DuoRec`, `GRU4Rec`, `SASRec`
- Priority 3 rationale: absolute spread is narrow on ML1M, but almost every model is clustered in a low band, which suggests the transfer template was serviceable but not yet truly dataset-native.

## Concrete Stage H Search Shape

- `amazon_beauty / GRU4Rec`: 3 candidates. Keep one around current recovery lr (`~1e-3`), add one higher-lr recovery candidate (`~3e-3` to `1e-2`), and add one lower-dropout + shorter-`max_len` variant.
- `amazon_beauty / FAME`: 3 candidates. Search `num_experts` and hidden/inner size with a relatively high lr band instead of only nudging lr.
- `amazon_beauty / BSARec` and `TiSASRec`: 2 candidates each. Rebuild from Amazon-native settings rather than transferred winners; keep lr around `3e-4` to `8e-4`, and vary `max_len` plus dropout.
- `KuaiRecLargeStrictPosV2_0.2 / GRU4Rec`: 2 candidates. One high-lr recovery branch and one capacity-up branch with shorter sequence length.
- `KuaiRecLargeStrictPosV2_0.2 / TiSASRec` and `DuoRec`: 2 candidates each. Preserve moderate lr, but widen structure search more than lr search.
- `movielens1m` weak combos: 2 candidates each. Stop relying only on `lastfm`/`foursquare` transfer; add at least one ML1M-local candidate built from the better `BSARec`/`TiSASRec` scale regime.

## All-Architecture FMoE_N3 Comparison

Below, the `FeaturedMoE_N3` column is recomputed from **all** `fmoe_n3/Final_all_datasets` architectures, not only `A6`.
Each `FeaturedMoE_N3` cell is written as `metric [A#/H#]`. For MRR tables, the tag is the run selected by best valid MRR@20; for HR tables, it is the run selected by best valid HR@10.

### Valid MRR@20 (6x10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1188 | 0.0277 | 0.0455 | 0.0505 | 0.1667 | 0.1651 | 0.0794 | 0.1475 | 0.1591 | `0.1707` |
| lastfm0.03 | 0.2321 | 0.2048 | 0.2260 | 0.2324 | 0.1883 | 0.2198 | 0.2269 | 0.2410 | 0.2242 | `0.2586` |
| amazon_beauty | `0.1264` | 0.0197 | 0.0470 | 0.1245 | 0.0980 | 0.0162 | 0.1156 | 0.1062 | 0.0265 | 0.1101 |
| foursquare | 0.1214 | 0.0846 | 0.1131 | 0.1153 | 0.0813 | 0.0858 | 0.1058 | 0.0987 | 0.0988 | `0.1282` |
| movielens1m | 0.0778 | 0.0761 | 0.0870 | 0.0747 | 0.0847 | 0.0876 | 0.0679 | 0.0850 | 0.0831 | `0.0977` |
| retail_rocket | 0.2629 | 0.2378 | 0.2601 | 0.2730 | 0.3548 | 0.3652 | 0.2797 | `0.3726` | 0.3547 | 0.2997 |

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1188 | 0.0277 | 0.0455 | 0.0505 | 0.1667 | 0.1651 | 0.0794 | 0.1475 | 0.1591 | 0.1707 [A6/H14] |
| lastfm0.03 | 0.2321 | 0.2048 | 0.2260 | 0.2324 | 0.1883 | 0.2198 | 0.2269 | 0.2410 | 0.2242 | 0.2586 [A1/H3] |
| amazon_beauty | 0.1264 | 0.0197 | 0.0470 | 0.1245 | 0.0980 | 0.0162 | 0.1156 | 0.1062 | 0.0265 | 0.1101 [A6/H14] |
| foursquare | 0.1214 | 0.0846 | 0.1131 | 0.1153 | 0.0813 | 0.0858 | 0.1058 | 0.0987 | 0.0988 | 0.1282 [A1/H2] |
| movielens1m | 0.0778 | 0.0761 | 0.0870 | 0.0747 | 0.0847 | 0.0876 | 0.0679 | 0.0850 | 0.0831 | 0.0977 [A1/H3] |
| retail_rocket | 0.2629 | 0.2378 | 0.2601 | 0.2730 | 0.3548 | 0.3652 | 0.2797 | 0.3726 | 0.3547 | 0.2997 [A2/H2] |

### Test MRR@20 At Best Valid MRR@20 (6x10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1107 | 0.0250 | 0.0435 | 0.0473 | 0.1661 | 0.1641 | 0.0761 | 0.1449 | 0.1601 | `0.1690` |
| lastfm0.03 | `0.2298` | 0.2012 | 0.2249 | 0.2319 | 0.1907 | 0.2212 | 0.2259 | 0.2426 | 0.2272 | 0.2241 |
| amazon_beauty | `0.0882` | 0.0018 | 0.0135 | 0.0863 | 0.0673 | 0.0029 | 0.0899 | 0.0702 | 0.0000 | 0.0726 |
| foursquare | `0.1166` | 0.0755 | 0.1070 | 0.1074 | 0.0729 | 0.0758 | 0.0997 | 0.0917 | 0.0915 | 0.0960 |
| movielens1m | `0.0668` | 0.0739 | 0.0743 | 0.0661 | 0.0740 | 0.0782 | 0.0623 | 0.0791 | 0.0766 | 0.0559 |
| retail_rocket | 0.2635 | 0.2373 | 0.2627 | 0.2729 | 0.3534 | 0.3640 | 0.2799 | `0.3709` | 0.3530 | 0.2984 |

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1107 | 0.0250 | 0.0435 | 0.0473 | 0.1661 | 0.1641 | 0.0761 | 0.1449 | 0.1601 | 0.1690 [A6/H14] |
| lastfm0.03 | 0.2298 | 0.2012 | 0.2249 | 0.2319 | 0.1907 | 0.2212 | 0.2259 | 0.2426 | 0.2272 | 0.2241 [A1/H3] |
| amazon_beauty | 0.0882 | 0.0018 | 0.0135 | 0.0863 | 0.0673 | 0.0029 | 0.0899 | 0.0702 | 0.0000 | 0.0726 [A6/H14] |
| foursquare | 0.1166 | 0.0755 | 0.1070 | 0.1074 | 0.0729 | 0.0758 | 0.0997 | 0.0917 | 0.0915 | 0.0960 [A1/H2] |
| movielens1m | 0.0668 | 0.0739 | 0.0743 | 0.0661 | 0.0740 | 0.0782 | 0.0623 | 0.0791 | 0.0766 | 0.0559 [A1/H3] |
| retail_rocket | 0.2635 | 0.2373 | 0.2627 | 0.2729 | 0.3534 | 0.3640 | 0.2799 | 0.3709 | 0.3530 | 0.2984 [A2/H2] |



### Valid HR@10 (6x10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1441 | 0.0399 | 0.0519 | 0.0693 | 0.1693 | 0.1673 | 0.0863 | 0.1583 | 0.1660 | 0.1780 [A3/H3] |
| lastfm0.03 | 0.3019 | 0.2490 | 0.2774 | 0.3035 | 0.2283 | 0.2586 | 0.2976 | 0.2998 | 0.2700 | 0.3162 [A1/H3] |
| amazon_beauty | 0.1770 | 0.0265 | 0.0708 | 0.1681 | 0.0973 | 0.0265 | 0.1504 | 0.1239 | 0.0442 | 0.1416 [A3/H2] |
| foursquare | 0.2152 | 0.1343 | 0.1932 | 0.2087 | 0.1330 | 0.1445 | 0.1961 | 0.1742 | 0.1598 | 0.2439 [A1/H3] |
| movielens1m | 0.1898 | 0.1768 | 0.1939 | 0.1828 | 0.1805 | 0.1828 | 0.1754 | 0.1944 | 0.1791 | 0.2270 [A1/H3] |
| retail_rocket | 0.4201 | 0.3795 | 0.4211 | 0.4313 | 0.4168 | 0.4343 | 0.4367 | 0.4422 | 0.4084 | 0.4618 [A2/H2] |

### Test HR@10 At Best Valid HR@10 (6x10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FeaturedMoE_N3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1388 | 0.0390 | 0.0507 | 0.0666 | 0.1672 | 0.1678 | 0.0832 | 0.1571 | 0.1661 | 0.1740 [A3/H3] |
| lastfm0.03 | 0.2991 | 0.2505 | 0.2782 | 0.2946 | 0.2253 | 0.2657 | 0.2946 | 0.2981 | 0.2726 | 0.2800 [A1/H3] |
| amazon_beauty | 0.1053 | 0.0263 | 0.0263 | 0.0965 | 0.0789 | 0.0088 | 0.1053 | 0.0789 | 0.0000 | 0.1053 [A3/H2] |
| foursquare | 0.2115 | 0.1335 | 0.1910 | 0.2063 | 0.1256 | 0.1327 | 0.1910 | 0.1763 | 0.1561 | 0.1973 [A1/H3] |
| movielens1m | 0.1850 | 0.1694 | 0.1874 | 0.1786 | 0.1855 | 0.1832 | 0.1689 | 0.1786 | 0.1781 | 0.1439 [A1/H3] |
| retail_rocket | 0.4249 | 0.3786 | 0.4234 | 0.4337 | 0.4158 | 0.4327 | 0.4375 | 0.4400 | 0.4070 | 0.4647 [A2/H2] |

### FeaturedMoE_N3 Winner Detail

| Dataset | Best Valid MRR@20 Run | Best Valid HR@10 Run |
| --- | --- | --- |
| KuaiRecLargeStrictPosV2_0.2 | A6/H14 / s1 / P14_FAD_A6_KUAIRECLARGESTRICTPOSV2_0_2_H14_S1 | A3/H3 / s4 / P14_FAD_A3_KUAIRECLARGESTRICTPOSV2_0_2_H3_S4 |
| lastfm0.03 | A1/H3 / s1 / P14_FAD_LASTFM0_03_H3_S1 | A1/H3 / s2 / P14_FAD_LASTFM0_03_H3_S2 |
| amazon_beauty | A6/H14 / s2 / P14_FAD_A6_AMAZON_BEAUTY_H14_S2 | A3/H2 / s2 / P14_FAD_A3_AMAZON_BEAUTY_H2_S2 |
| foursquare | A1/H2 / s3 / P14_FAD_FOURSQUARE_H2_S3 | A1/H3 / s2 / P14_FAD_FOURSQUARE_H3_S2 |
| movielens1m | A1/H3 / s2 / P14_FAD_MOVIELENS1M_H3_S2 | A1/H3 / s1 / P14_FAD_MOVIELENS1M_H3_S1 |
| retail_rocket | A2/H2 / s1 / P14_FAD_A2_RETAIL_ROCKET_H2_S1 | A2/H2 / s1 / P14_FAD_A2_RETAIL_ROCKET_H2_S1 |

## A1 vs A6 Reading

`A6` is not a different family from scratch. It is the `A1` layout with bias removed.
- `A1`: main architecture with the original cue/bias path kept on
- `A6`: `A1 + no_bias`, meaning `bias_mode=none`, `rule_bias_scale=0`, `feature_group_bias_lambda=0`

From the logged final runs, the pattern is mixed rather than one-sided.
- `A1`-leaning datasets: `lastfm0.03`, `foursquare`, `movielens1m`
- `A6`-leaning datasets: `amazon_beauty`, `KuaiRecLargeStrictPosV2_0.2` on valid MRR
- neither says "bias is always good" or "bias is always bad"

Practical interpretation:
- When cue/bias priors align with dataset structure, `A1` is better because the router can use that bias as useful inductive guidance.
- When those priors are noisy or over-strong for the dataset, `A6` can win because removing bias reduces over-commitment and lets the shared representation behave more neutrally.
- `A3` doing well on some HR slices suggests that targeted feature removal can be a cleaner fix than full bias removal.

## Architecture Notes For FMoE

The current metadata still treats `A1` as the architectural protagonist, and the results still support that reading more than they support replacing it wholesale with `A6`.
- `A1` remains the best default mainline candidate
- `A6` should be kept as a bias-ablation / bias-sensitivity control
- `A3` should be kept as a structural feature-drop control
- `A2` is still useful as a robustness variant, especially where router consistency is strong

About `A1` itself:
- the final `A1` name is `A1_MAIN_ATTN_MICRO_BEFORE`
- in the current launcher metadata it uses the main layout and keeps the bias path
- compared with the later `A6`, the important difference is not just dropout or minor tuning but whether the bias path is active at all
- the current implementation is better described as `macro -> mid -> micro` with wrappers `w4_bxd / w6_bxd_plus_a / w1_flat`
- so `A1_MAIN_ATTN_MICRO_BEFORE` is effectively a legacy run name; for explanation in the paper, the `macro/mid/micro main architecture` wording is clearer and more accurate

For paper usage, dataset-specific architecture selection is acceptable if it is framed honestly as validation-based model selection within a fixed candidate family.
- good framing: "we compare a fixed architecture family (`A1/A2/A3/A6`) and report the validation-selected variant per dataset"
- weaker framing: pretending one identical architecture was chosen a priori for every dataset when that is not what happened

If you want the cleanest paper story, the safest setup is:
- main results: `A1`
- control/ablation table: `A2`, `A3`, `A6`
- optional per-dataset oracle table: best-of-family (`A1/A2/A3/A6`)

## Stage H Baseline Plan

The biggest baseline issue now is not just low mean performance. It is variance: some models land near dataset-best while others clearly missed the right local basin.

Recommended Stage H target list:
- `amazon_beauty`: `BSARec`, `GRU4Rec`, `FAME`, `TiSASRec`
- `KuaiRecLargeStrictPosV2_0.2`: `GRU4Rec`, `TiSASRec`, `DuoRec`
- `movielens1m`: `SASRec`, `GRU4Rec`, `DuoRec`, `FEARec`

Stage H design:
- use the Stage G winner as one anchor candidate
- add at least one new manual candidate per weak combo
- for the worst sparse failures (`amazon_beauty` GRU/FAME), use 3 candidates instead of 2
- search structure as well as LR; do not treat this as LR-only rescue

Implemented launcher:
- [run_stageH_targeted_recovery.py](/workspace/jy1559/FMoE/experiments/run/baseline/run_stageH_targeted_recovery.py)
- [stageH_targeted_recovery.sh](/workspace/jy1559/FMoE/experiments/run/baseline/stageH_targeted_recovery.sh)

Current Stage H recovery shape:
- total target combos: 11
- expected candidates: 24
- expected runs with 3 seeds: 72
- sparse worst cases use wider and higher LR branches
- Kuai weak combos get structure-relief candidates
- ML1M weak combos get one dataset-native candidate instead of transfer-only reuse

Suggested execution:
```bash
bash /workspace/jy1559/FMoE/experiments/run/baseline/stageH_targeted_recovery.sh --dry-run
```

Fast triage mode:
- use `--fast-screen` first to quickly identify obviously bad candidates with seed `1` and reduced budget
- then keep only the promising combos/candidates for the full Stage H rerun

Suggested fast screen:
```bash
bash /workspace/jy1559/FMoE/experiments/run/baseline/stageH_targeted_recovery.sh --fast-screen
```

## FMoE A1 Refresh Plan

Given the full-family comparison, the most useful next refresh is not another A6-only sweep. It is a clean `A1` rerun across all 6 datasets with slightly better dataset-aware hparam coverage.

Implemented wrapper:
- [phase_14_final_all_datasets_A1_refresh.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_n3/phase_14_final_all_datasets_A1_refresh.sh)

Why this shape:
- `A1` already wins or is competitive on several datasets
- `A1` deserves a clean all-dataset rerun under one consistent budget
- we should test whether some of the previous `A6` wins were true architecture wins or partly hparam-coverage wins

Proposed A1 refresh defaults:
- architecture: `A1`
- datasets: all 6
- seeds: `1,2,3`
- common hparams: `H2,H3`
- dataset outliers: `KuaiRecLargeStrictPosV2_0.2:H14`, `amazon_beauty:H14`, `foursquare:H2`, `retail_rocket:H2`
- budget: `max_evals=16`, `tune_epochs=90`, `tune_patience=9`
- LR band: `2e-4 .. 4e-3`
- slightly lower family/attention dropout than the recent A6 sweep

Dropout recommendation for A1:
- keep `family_dropout_prob` modest rather than high; `0.08` is a good default refresh point
- keep `attn_dropout_prob` aligned and slightly conservative; `0.08` is a sensible first rerun setting
- if a dataset looks clearly regularization-hungry, move one notch up rather than changing the architecture label itself

Suggested execution:
```bash
bash /workspace/jy1559/FMoE/experiments/run/fmoe_n3/phase_14_final_all_datasets_A1_refresh.sh --dry-run
```

## FMoE Final Validation Direction

If the goal is a paper-ready final story, I would validate FMoE in three layers.
- Layer 1: `A1` full rerun across all datasets as the mainline model
- Layer 2: `A6` and `A3` limited refresh on the datasets where they looked especially informative
- Layer 3: best-of-family report table for transparency

Concrete recommendation:
- rerun `A1` on all 6 datasets
- rerun `A6` only on `amazon_beauty` and `KuaiRecLargeStrictPosV2_0.2`
- rerun `A3` only on datasets where HR looked notably better or where category-derived features may be noisy
- keep the final paper claim conservative: "`A1` is the primary model; `A3/A6` help interpret when feature- or bias-derived inductive cues stop being reliable"

## FMoE A7 A8 A9 Quick Probe

For the next fast probe, it is useful to separate two questions that were still mixed together in the `A1` vs `A6` comparison:
- does the gain come from restoring bias?
- or does it come from restoring the older `ATTN_MICRO_BEFORE` layout?

Recommended quick probe family:
- `A7`: `A6`-style macro/mid/micro core plus strict NN + z-loss, but with bias restored
- `A8`: old `ATTN_MICRO_BEFORE` layout plus strict NN + z-loss, with `no_bias`
- `A9`: `A8` plus bias restored

Interpretation guide:
- if `A7 > A6`, bias matters more than layout
- if `A8 > A6`, old layout matters even without bias
- if `A9` is best, old layout and bias are helping together
- if all three lose to the refreshed `A1`, then current plain macro/mid/micro is still the cleanest mainline architecture

Implementation:
- [run_final_all_datasets.py](/workspace/jy1559/FMoE/experiments/run/fmoe_n3/run_final_all_datasets.py)
- [phase_14_final_all_datasets_A7_A9_probe.sh](/workspace/jy1559/FMoE/experiments/run/fmoe_n3/phase_14_final_all_datasets_A7_A9_probe.sh)

Probe budget:
- seeds: `1`
- architectures: `A7,A8,A9`
- hparams per dataset: `2`
- budget: `max_evals=10`, `tune_epochs=60`, `tune_patience=6`
- fixed dropout for clean comparison: `family_dropout_prob=0.08`, `attn_dropout_prob=0.08`

Dataset-specific narrow hparam and LR bands:
- `KuaiRecLargeStrictPosV2_0.2`: `H3 + H14`, `2.5e-4 ~ 1.2e-3`
  - anchor: `A6/H14` best LR was about `4.90e-4`
- `lastfm0.03`: `H3 + H11`, `2.0e-4 ~ 1.2e-3`
  - anchor: `A6/H11` best LR was about `4.07e-4`
- `amazon_beauty`: `H3 + H14`, `4.5e-4 ~ 2.2e-3`
  - anchor: `A6/H14` best LR was about `9.77e-4`
- `foursquare`: `H3 + H2`, `1.5e-3 ~ 6.0e-3`
  - anchor: `A1/H2` best LR was about `4.17e-3`
- `movielens1m`: `H3 + H1`, `1.2e-3 ~ 4.5e-3`
  - anchor: `A1/H3` best LR was about `3.07e-3`
- `retail_rocket`: `H3 + H2`, `7.0e-4 ~ 3.0e-3`
  - anchor: `A2/H2` best LR was about `1.84e-3`

Why dropout is fixed for this probe:
- the current question is architecture and bias, not regularization rescue
- changing dropout together with layout would blur the interpretation
- if `A8` or `A9` looks promising, only then is it worth doing a second small sweep on dropout

Suggested execution:
```bash
bash /workspace/jy1559/FMoE/experiments/run/fmoe_n3/phase_14_final_all_datasets_A7_A9_probe.sh --dry-run
```

## Expanded 6x12 Valid/Test Comparison

Columns: 9 baseline models + `FMoE_A3` (`A3_NO_CATEGORY`) + `FMoE_A5` (`A5_NO_CATEGORY_NO_TIMESTAMP`) + `FMoE_Best` (best architecture among all `A1~A9`).

Highlight rule: row-best is **bold**. Values marked with `*` are either in the row top-3 or within about 3% of the row-best.

### Valid-Selected 6x12

### Valid MRR@20 (selected by best valid MRR@20)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1188 | 0.0277 | 0.0455 | 0.1663 | 0.1667* | 0.1651 | 0.0794 | 0.1475 | 0.1591 | 0.1693* [A3/H3] | 0.1662 [A5/H3] | **0.1718 [A8/H14]** |
| lastfm0.03 | 0.2431 | 0.2048 | 0.2472* | 0.2361 | 0.1929 | 0.2365 | 0.2389 | 0.2513* | 0.2430 | 0.2365 [A3/H5] | 0.2366 [A5/H3] | **0.2586 [A1/H3]** |
| amazon_beauty | **0.1264** | 0.0197 | 0.0744 | 0.1245* | 0.0980 | 0.0162 | 0.1156* | 0.1062 | 0.0265 | 0.1015 [A3/H2] | 0.1003 [A5/H2] | 0.1101 [A6/H14] |
| foursquare | **0.1312** | 0.0898 | 0.1273* | 0.1200 | 0.0813 | 0.1087 | 0.1118 | 0.1229 | 0.1030 | 0.1187 [A3/H2] | 0.1182 [A5/H2] | 0.1282* [A1/H2] |
| movielens1m | 0.0778 | 0.0761 | 0.0870* | 0.0747 | 0.0847 | 0.0876* | 0.0679 | 0.0850 | 0.0831 | 0.0868 [A3/H3] | 0.0833 [A5/H5] | **0.0977 [A1/H3]** |
| retail_rocket | 0.2629 | 0.2378 | 0.2601 | 0.2730 | 0.3548* | 0.3652* | 0.2797 | **0.3726** | 0.3547 | - | - | 0.2997 [A2/H2] |

### Test MRR@20 At Best Valid MRR@20

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1107 | 0.0250 | 0.0435 | 0.1656* | 0.1661* | 0.1641* | 0.0761 | 0.1449 | 0.1601 | 0.1682* [A3/H3] | 0.1679* [A5/H3] | **0.1690 [A8/H14]** |
| lastfm0.03 | 0.2118 | 0.2012 | 0.2075 | 0.2025 | 0.1664 | 0.2003 | 0.2058 | 0.2160 | 0.2066 | **0.2353 [A3/H5]** | 0.2327* [A5/H3] | 0.2241* [A1/H3] |
| amazon_beauty | 0.0882* | 0.0018 | 0.0025 | 0.0863* | 0.0673 | 0.0029 | **0.0899** | 0.0702 | 0.0000 | 0.0730 [A3/H2] | 0.0620 [A5/H2] | 0.0726 [A6/H14] |
| foursquare | 0.1043* | 0.0601 | 0.0962 | 0.0941 | 0.0634 | 0.0854 | 0.0904 | 0.0907 | 0.0811 | **0.1156 [A3/H2]** | 0.1109* [A5/H2] | 0.0960 [A1/H2] |
| movielens1m | 0.0668 | 0.0739 | 0.0743 | 0.0661 | 0.0740 | 0.0782* | 0.0623 | **0.0791** | 0.0766* | 0.0738 [A3/H3] | 0.0732 [A5/H5] | 0.0559 [A1/H3] |
| retail_rocket | 0.2635 | 0.2373 | 0.2627 | 0.2729 | 0.3534* | 0.3640* | 0.2799 | **0.3709** | 0.3530 | - | - | 0.2984 [A2/H2] |

### Valid HR@10 (selected by best valid HR@10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1441 | 0.0399 | 0.0519 | 0.1693 | 0.1693 | 0.1673 | 0.0863 | 0.1583 | 0.1660 | 0.1780* [A3/H3] | 0.1750* [A5/H1] | **0.1783 [A9/H14]** |
| lastfm0.03 | 0.3067 | 0.2490 | 0.2966 | 0.3115* | 0.2283 | 0.2801 | 0.3133* | 0.2998 | 0.2825 | 0.2926 [A3/H1] | 0.2982 [A5/H3] | **0.3162 [A1/H3]** |
| amazon_beauty | **0.1770** | 0.0265 | 0.1062 | 0.1681* | 0.0973 | 0.0265 | 0.1504* | 0.1239 | 0.0442 | 0.1416 [A3/H2] | 0.1239 [A5/H3] | 0.1416 [A3/H2] |
| foursquare | 0.2339* | 0.1527 | 0.2200 | 0.2255* | 0.1330 | 0.1800 | 0.2184 | 0.2003 | 0.1629 | 0.2208 [A3/H2] | 0.2118 [A5/H3] | **0.2439 [A1/H3]** |
| movielens1m | 0.1898 | 0.1768 | 0.1939* | 0.1828 | 0.1805 | 0.1828 | 0.1754 | 0.1944* | 0.1791 | 0.1884 [A3/H3] | 0.1893 [A5/H3] | **0.2270 [A1/H3]** |
| retail_rocket | 0.4201 | 0.3795 | 0.4211 | 0.4313 | 0.4168 | 0.4343 | 0.4367* | 0.4422* | 0.4084 | - | - | **0.4620 [A6/H2]** |

### Test HR@10 At Best Valid HR@10

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1388 | 0.0390 | 0.0507 | 0.1691 | 0.1672 | 0.1672 | 0.0832 | 0.1571 | 0.1661 | 0.1740* [A3/H3] | 0.1718* [A5/H1] | **0.1746 [A9/H14]** |
| lastfm0.03 | 0.2744 | 0.2505 | 0.2564 | 0.2723 | 0.2253 | 0.2452 | 0.2774 | **0.2981** | 0.2455 | 0.2864* [A3/H1] | 0.2952* [A5/H3] | 0.2800 [A1/H3] |
| amazon_beauty | **0.1053** | 0.0088 | 0.0088 | 0.0965* | 0.0789* | 0.0088 | **0.1053** | 0.0789* | 0.0000 | **0.1053 [A3/H2]** | 0.0789* [A5/H3] | **0.1053 [A3/H2]** |
| foursquare | 0.1868 | 0.1098 | 0.1716 | 0.1839 | 0.1256 | 0.1414 | 0.1729 | 0.1582 | 0.1311 | **0.2128 [A3/H2]** | 0.2063* [A5/H3] | 0.1973* [A1/H3] |
| movielens1m | 0.1850* | 0.1694 | **0.1874** | 0.1786 | 0.1790 | 0.1832* | 0.1689 | 0.1786 | 0.1781 | 0.1841* [A3/H3] | 0.1814 [A5/H3] | 0.1439 [A1/H3] |
| retail_rocket | 0.4249 | 0.3786 | 0.4234 | 0.4337 | 0.4158 | 0.4327 | 0.4375* | 0.4400* | 0.4070 | - | - | **0.4651 [A6/H2]** |

### Test-Selected 6x12

### Test MRR@20 (selected by best test MRR@20)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1425 | 0.0533 | 0.0767 | 0.1656 | **0.3267** | 0.2255 | 0.0797 | 0.2606 | 0.3264* | 0.3028 [A3/H3] | 0.1679 [A5/H3] | 0.3261* [A1/H3] |
| lastfm0.03 | 0.2311 | 0.2013 | 0.2262 | 0.2319 | 0.1907 | 0.2233 | 0.2259 | **0.2457** | 0.2311 | 0.2388* [A3/H2] | 0.2371 [A5/H5] | 0.2401* [A6/H2] |
| amazon_beauty | **0.1273** | 0.0110 | 0.0533 | 0.1265* | 0.1228 | 0.0029 | 0.1252* | 0.1268* | 0.0139 | 0.0738 [A3/H2] | 0.0620 [A5/H2] | 0.1146 [A1/H3] |
| foursquare | **0.1175** | 0.0755 | 0.1070 | 0.1079 | 0.0745 | 0.0854 | 0.0997 | 0.0962 | 0.0939 | 0.1156* [A3/H2] | 0.1109 [A5/H2] | 0.1160* [A6/H15] |
| movielens1m | 0.0715 | 0.0739 | **0.0854** | 0.0720 | 0.0753 | 0.0782 | 0.0641 | 0.0791* | 0.0779 | 0.0774 [A3/H1] | 0.0816* [A5/H1] | 0.0816* [A5/H1] |
| retail_rocket | 0.2635 | 0.2373 | 0.2630 | 0.2744 | 0.3553* | 0.3640* | 0.2801 | **0.3709** | 0.3530 | - | - | 0.2994 [A6/H2] |

### Valid MRR@20 At Best Test MRR@20

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.0132 | 0.0126 | 0.0115 | **0.1663** | 0.0101 | 0.0115 | 0.0138* | 0.0111 | 0.0108 | 0.0131 [A3/H3] | 0.1662* [A5/H3] | 0.0129 [A1/H3] |
| lastfm0.03 | 0.2307 | 0.1981 | 0.2254 | 0.2324 | 0.1883 | 0.2194 | 0.2269 | **0.2400** | 0.2226 | 0.2336* [A3/H2] | 0.2334* [A5/H5] | 0.2376* [A6/H2] |
| amazon_beauty | 0.0840* | 0.0088 | 0.0168 | 0.0745 | 0.0575 | 0.0162 | 0.0765 | 0.0578 | 0.0133 | **0.1008 [A3/H2]** | 0.1003* [A5/H2] | 0.0751 [A1/H3] |
| foursquare | 0.1184* | 0.0846 | 0.1131 | 0.1114 | 0.0734 | 0.1087 | 0.1058 | 0.0985 | 0.0982 | **0.1187 [A3/H2]** | 0.1182* [A5/H2] | 0.1169* [A6/H15] |
| movielens1m | 0.0742 | 0.0761 | 0.0842* | 0.0736 | 0.0800 | **0.0876** | 0.0667 | 0.0850* | 0.0822 | 0.0783 [A3/H1] | 0.0815 [A5/H1] | 0.0815 [A5/H1] |
| retail_rocket | 0.2629 | 0.2378 | 0.2595 | 0.2725 | 0.3536 | 0.3652* | 0.2789 | **0.3726** | 0.3547* | - | - | 0.2987 [A6/H2] |

### Test HR@10 (selected by best test HR@10)

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.1844 | 0.0671 | 0.0805 | 0.1691 | 0.3280* | 0.2696 | 0.0838 | 0.2906 | 0.3288* | 0.3244* [A3/H3] | 0.1759 [A5/H3] | **0.3312 [A1/H3]** |
| lastfm0.03 | **0.2991** | 0.2511 | 0.2792 | 0.2989* | 0.2290 | 0.2657 | 0.2946* | 0.2981* | 0.2768 | 0.2917* [A3/H3] | 0.2952* [A5/H3] | 0.2984* [A6/H9] |
| amazon_beauty | **0.1491** | 0.0263 | 0.0614 | 0.1404* | 0.1228 | 0.0088 | 0.1316* | 0.1404* | 0.0175 | 0.1053 [A3/H2] | 0.0877 [A5/H2] | 0.1316* [A1/H3] |
| foursquare | **0.2178** | 0.1335 | 0.1910 | 0.2063 | 0.1293 | 0.1414 | 0.1950 | 0.1763 | 0.1590 | 0.2128* [A3/H2] | 0.2084 [A5/H3] | 0.2131* [A2/H1] |
| movielens1m | 0.1850 | 0.1694 | **0.2026** | 0.1837 | 0.1855 | 0.1841 | 0.1707 | 0.1850 | 0.1850 | 0.1864 [A3/H1] | 0.1929* [A5/H1] | 0.1966* [A6/H5] |
| retail_rocket | 0.4249 | 0.3786 | 0.4234 | 0.4345 | 0.4162 | 0.4327 | 0.4390* | 0.4400* | 0.4070 | - | - | **0.4651 [A6/H2]** |

### Valid HR@10 At Best Test HR@10

| Dataset | SASRec | GRU4Rec | TiSASRec | DuoRec | SIGMA | BSARec | FEARec | DIF-SR | FAME | FMoE_A3 | FMoE_A5 | FMoE_Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KuaiRecLargeStrictPosV2_0.2 | 0.0248* | 0.0232 | 0.0202 | 0.1693* | 0.0112 | 0.0194 | 0.0216 | 0.0147 | 0.0123 | 0.0243 [A3/H3] | **0.1712 [A5/H3]** | 0.0232 [A1/H3] |
| lastfm0.03 | **0.3019** | 0.2471 | 0.2750 | 0.2998* | 0.2275 | 0.2586 | 0.2976* | 0.2998* | 0.2679 | 0.2902 [A3/H3] | 0.2982* [A5/H3] | 0.2960* [A6/H9] |
| amazon_beauty | 0.1062* | 0.0265 | 0.0442 | 0.1150* | 0.0619 | 0.0177 | 0.1062* | 0.0708 | 0.0177 | **0.1416 [A3/H2]** | 0.1150* [A5/H2] | 0.0973 [A1/H3] |
| foursquare | 0.2145* | 0.1343 | 0.1932 | 0.2087 | 0.1296 | 0.1800 | 0.1932 | 0.1742 | 0.1582 | **0.2208 [A3/H2]** | 0.2116 [A5/H3] | 0.2121* [A2/H1] |
| movielens1m | 0.1898* | 0.1768 | **0.1925** | 0.1801 | 0.1805 | 0.1801 | 0.1662 | 0.1861* | 0.1768 | 0.1824 [A3/H1] | 0.1722 [A5/H1] | 0.1796 [A6/H5] |
| retail_rocket | 0.4201 | 0.3795 | 0.4211 | 0.4308 | 0.4162 | 0.4343 | 0.4366* | 0.4422* | 0.4084 | - | - | **0.4620 [A6/H2]** |
