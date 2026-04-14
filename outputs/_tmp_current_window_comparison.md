# Current Window (>=2026-04-13) Best MRR@20 by Dataset/Model
- scanned_json_files: 20204
- selected_json_files: 4974
- cutoff_date: 2026-04-13

| dataset | GRU4Rec | SASRec | TiSASRec | FMoE_n3 | FMoE_n4 | winner | winner_mrr@20 |
|---|---:|---:|---:|---:|---:|---|---:|
| beauty | 0.0278 | 0.0697 | 0.0630 | - | 0.0570 | SASRec | 0.0697 |
| retail_rocket | 0.3125 | 0.3696 | 0.3546 | 0.3074 | - | SASRec | 0.3696 |
| movielens1m | 0.0591 | 0.0578 | 0.0605 | 0.0772 | - | FMoE_n3 | 0.0772 |
| lastfm0.03 | 0.2561 | 0.3054 | 0.3045 | 0.2421 | - | SASRec | 0.3054 |
| foursquare | 0.1297 | 0.1710 | 0.1682 | 0.1158 | 0.1628 | SASRec | 0.1710 |

## Overall Mean MRR@20 (available datasets only)
- GRU4Rec: mean=0.157040 (n=5)
- SASRec: mean=0.194700 (n=5)
- TiSASRec: mean=0.190160 (n=5)
- FMoE_n3: mean=0.185625 (n=4)
- FMoE_n4: mean=0.109900 (n=2)

## Best Run Path (current window)
### beauty
- GRU4Rec: 0.027800, ts=2026-04-14T13:11:46.400239, dir=baseline_2, raw_model=GRU4Rec, path=experiments/run/artifacts/results/baseline_2/beauty_GRU4Rec_abcd_v2_lean_b_dbeauty_mgru4rec_a008_l04_s1_20260414_131055_604174_pid1022340.json
- SASRec: 0.069700, ts=2026-04-14T08:10:55.998377, dir=baseline_2, raw_model=SASRec, path=experiments/run/artifacts/results/baseline_2/beauty_SASRec_abcd_v2_lean_a_dbeauty_msasrec_a017_l04_s1_20260414_081046_003270_pid962784.json
- TiSASRec: 0.063000, ts=2026-04-14T14:01:39.069658, dir=baseline_2, raw_model=TiSASRec, path=experiments/run/artifacts/results/baseline_2/beauty_TiSASRec_abcd_v2_lean_b_dbeauty_mtisasrec_a007_l01_s1_20260414_133659_012190_pid1036972.json
- FMoE_n4: 0.057000, ts=2026-04-14T14:32:05.726287, dir=fmoe_n4, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n4/beauty_FeaturedMoE_N3_abcd_a12_hparam_v1_a_dbeauty_mfeatured_moe_n3_a012_l04_s1_20260414_143117_452105_pid1055014.json

### retail_rocket
- GRU4Rec: 0.312500, ts=2026-04-14T08:37:26.665837, dir=baseline_2, raw_model=GRU4Rec, path=experiments/run/artifacts/results/baseline_2/retail_rocket_GRU4Rec_abcd_v2_lean_a_dretail_rocket_mgru4rec_a009_l04_s1_20260414_083623_695601_pid978399.json
- SASRec: 0.369600, ts=2026-04-14T09:17:11.573794, dir=baseline_2, raw_model=SASRec, path=experiments/run/artifacts/results/baseline_2/retail_rocket_SASRec_abcd_v2_lean_a_dretail_rocket_msasrec_a013_l02_s1_20260414_091243_382206_pid979979.json
- TiSASRec: 0.354600, ts=2026-04-13T21:30:46.054800, dir=baseline_2, raw_model=TiSASRec, path=experiments/run/artifacts/results/baseline_2/retail_rocket_TiSASRec_abcd_v1_a_dretail_rocket_mtisasrec_a022_l04_s1_20260413_212429_899599_pid832010.json
- FMoE_n3: 0.307400, ts=2026-04-13T01:23:36.961567, dir=fmoe_n3, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n3/retail_rocket_FeaturedMoE_N3_p14_fad_a8_retail_rocket_h2_s1_20260413_003926_440664_pid277890.json

### movielens1m
- GRU4Rec: 0.059100, ts=2026-04-14T12:11:42.755433, dir=baseline_2, raw_model=GRU4Rec, path=experiments/run/artifacts/results/baseline_2/movielens1m_GRU4Rec_abcd_v2_lean_a_dmovielens1m_mgru4rec_a009_l04_s1_20260414_121112_594132_pid996375.json
- SASRec: 0.057800, ts=2026-04-14T12:24:37.366840, dir=baseline_2, raw_model=SASRec, path=experiments/run/artifacts/results/baseline_2/movielens1m_SASRec_abcd_v2_lean_a_dmovielens1m_msasrec_a003_l04_s1_20260414_122222_905809_pid999535.json
- TiSASRec: 0.060500, ts=2026-04-14T12:50:26.270412, dir=baseline_2, raw_model=TiSASRec, path=experiments/run/artifacts/results/baseline_2/movielens1m_TiSASRec_abcd_v2_lean_a_dmovielens1m_mtisasrec_a007_l03_s1_20260414_124706_231283_pid1009282.json
- FMoE_n3: 0.077200, ts=2026-04-14T10:24:21.605658, dir=fmoe_n3, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n3/movielens1m_FeaturedMoE_N3_p15a_tgt_foursquare_to_movielens_a12_all_stage_full_router_h3_h5_s1_20260414_092228_262254_pid980302.json

### lastfm0.03
- GRU4Rec: 0.256100, ts=2026-04-14T10:11:06.480321, dir=baseline_2, raw_model=GRU4Rec, path=experiments/run/artifacts/results/baseline_2/lastfm0.03_GRU4Rec_abcd_v2_lean_a_dlastfm0.03_mgru4rec_a006_l04_s1_20260414_100915_331545_pid986025.json
- SASRec: 0.305400, ts=2026-04-14T01:25:16.567616, dir=baseline_2, raw_model=SASRec, path=experiments/run/artifacts/results/baseline_2/lastfm0.03_SASRec_abcd_v1_a_dlastfm0.03_msasrec_a021_l04_s1_20260414_012017_736141_pid860256.json
- TiSASRec: 0.304500, ts=2026-04-14T11:05:09.997855, dir=baseline_2, raw_model=TiSASRec, path=experiments/run/artifacts/results/baseline_2/lastfm0.03_TiSASRec_abcd_v2_lean_a_dlastfm0.03_mtisasrec_a001_l02_s1_20260414_105205_602815_pid991376.json
- FMoE_n3: 0.242100, ts=2026-04-13T11:04:27.724615, dir=fmoe_n3, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n3/lastfm0.03_FeaturedMoE_N3_p16a_stage1_lastfm0_03_lfm_h11_balanced_s1_20260413_084650_595542_pid575336.json

### foursquare
- GRU4Rec: 0.129700, ts=2026-04-14T13:58:28.053102, dir=baseline_2, raw_model=GRU4Rec, path=experiments/run/artifacts/results/baseline_2/foursquare_GRU4Rec_abcd_v2_lean_b_dfoursquare_mgru4rec_a009_l04_s1_20260414_134639_694062_pid1042437.json
- SASRec: 0.171000, ts=2026-04-13T17:38:38.452855, dir=baseline_2, raw_model=SASRec, path=experiments/run/artifacts/results/baseline_2/foursquare_SASRec_abcd_v1_a_dfoursquare_msasrec_a020_l06_s1_20260413_173740_129152_pid782557.json
- TiSASRec: 0.168200, ts=2026-04-13T17:50:13.520652, dir=baseline_2, raw_model=TiSASRec, path=experiments/run/artifacts/results/baseline_2/foursquare_TiSASRec_abcd_v1_a_dfoursquare_mtisasrec_a020_l06_s1_20260413_174909_358244_pid789945.json
- FMoE_n3: 0.115800, ts=2026-04-13T13:03:11.511244, dir=fmoe_n3, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n3/foursquare_FeaturedMoE_N3_p16a_stage1_foursquare_fsq_h15_feature_strong_s1_20260413_122456_973443_pid618093.json
- FMoE_n4: 0.162800, ts=2026-04-14T14:51:40.512319, dir=fmoe_n4, raw_model=FeaturedMoE_N3, path=experiments/run/artifacts/results/fmoe_n4/foursquare_FeaturedMoE_N3_abcd_a12_hparam_v1_a_dfoursquare_mfeatured_moe_n3_a002_l04_s1_20260414_144828_861928_pid1059571.json

