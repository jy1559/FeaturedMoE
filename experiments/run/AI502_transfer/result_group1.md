# AI502 Transfer Group1 결과 정리

## 요약

Group1은 실험 완결성과 transfer QC 기준으로 신뢰 가능하다. 다만 현재 전체 bash는 group2 init을 계속 수행 중이므로, 전역 `artifacts/analysis/*.csv`에는 group2 일부가 섞인다. 이 문서는 group1 pair/triplet만 manifest와 result JSON에서 다시 필터링해 계산했다.

| phase | expected rows | matched results |
| --- | --- | --- |
| native | 72 | 72 |
| init | 384 | 384 |
| freeze | 144 | 144 |
| multihop_bridge | 24 | 24 |
| multihop | 72 | 72 |


- 실패 로그: 없음
- group1 누락 result: 0
- transfer no-op 의심 row: 0
- hparam consistency 위반: 0


## Hparam 일치성 검증

모든 transfer row에서 source checkpoint, target native baseline, export checkpoint가 같은 `hparam_id` 경로를 공유했다. 즉 `shared_3` source는 `shared_3` target에만 transfer되고, `shared_4`/`shared_5`/`shared_6`도 동일한 방식으로 비교된다. 따라서 아래 gain은 같은 hparam/seed target scratch와 직접 비교한 값이다.


## Native Baseline: Seed 평균, Hparam 나열

| dataset | hparam | seeds | native test MRR@20 mean | min | max |
| --- | --- | --- | --- | --- | --- |
| beauty | shared_3 | 3 | 0.07157 | 0.06970 | 0.07300 |
| beauty | shared_4 | 3 | 0.06383 | 0.05640 | 0.07480 |
| beauty | shared_5 | 3 | 0.06467 | 0.06040 | 0.06690 |
| beauty | shared_6 | 3 | 0.05877 | 0.05350 | 0.06680 |
| retail_rocket | shared_3 | 3 | 0.37070 | 0.36870 | 0.37270 |
| retail_rocket | shared_4 | 3 | 0.37043 | 0.36810 | 0.37200 |
| retail_rocket | shared_5 | 3 | 0.37030 | 0.36880 | 0.37140 |
| retail_rocket | shared_6 | 3 | 0.37080 | 0.36890 | 0.37270 |
| foursquare | shared_3 | 3 | 0.16417 | 0.16290 | 0.16640 |
| foursquare | shared_4 | 3 | 0.16677 | 0.16600 | 0.16820 |
| foursquare | shared_5 | 3 | 0.16757 | 0.16470 | 0.17120 |
| foursquare | shared_6 | 3 | 0.16790 | 0.16260 | 0.17210 |
| KuaiRec | shared_3 | 3 | 0.33733 | 0.33520 | 0.33940 |
| KuaiRec | shared_4 | 3 | 0.33647 | 0.33500 | 0.33840 |
| KuaiRec | shared_5 | 3 | 0.33810 | 0.33730 | 0.33900 |
| KuaiRec | shared_6 | 3 | 0.33670 | 0.33510 | 0.33980 |
| lastfm | shared_3 | 3 | 0.30867 | 0.30750 | 0.31040 |
| lastfm | shared_4 | 3 | 0.30903 | 0.30800 | 0.31020 |
| lastfm | shared_5 | 3 | 0.30810 | 0.30510 | 0.31120 |
| lastfm | shared_6 | 3 | 0.30713 | 0.30540 | 0.30860 |
| movielens1m | shared_3 | 3 | 0.06240 | 0.05830 | 0.06620 |
| movielens1m | shared_4 | 3 | 0.05880 | 0.05590 | 0.06060 |
| movielens1m | shared_5 | 3 | 0.05720 | 0.05510 | 0.05940 |
| movielens1m | shared_6 | 3 | 0.05850 | 0.05710 | 0.06020 |


## Transfer QC: 로드/변화량

| phase | mode | rows | loaded | init changed | train changed |
| --- | --- | --- | --- | --- | --- |
| init | all_router_init | 48 | 72 | 48 | 24 |
| init | feature_encoder_a12_router_init | 48 | 102 | 75 | 48 |
| init | feature_encoder_group_router_init | 48 | 54 | 51 | 48 |
| init | feature_encoder_init | 48 | 30 | 27 | 24 |
| init | feature_encoder_router_init | 48 | 102 | 75 | 48 |
| init | full_except_feature_router_init | 48 | 96 | 83 | 77 |
| init | full_model_init | 48 | 198 | 158 | 125 |
| init | group_router_init | 48 | 24 | 24 | 24 |
| freeze | feature_encoder_a12_router_init | 48 | 102 | 75 | 0 |
| freeze | feature_encoder_group_router_init | 48 | 54 | 51 | 0 |
| freeze | full_model_init | 48 | 198 | 158 | 0 |
| multihop | feature_encoder_a12_router_init | 36 | 102 | 75 | 48 |
| multihop | full_model_init | 36 | 198 | 158 | 125 |
| multihop_bridge | feature_encoder_a12_router_init | 12 | 102 | 75 | 48 |
| multihop_bridge | full_model_init | 12 | 198 | 158 | 125 |


Freeze에서 `train changed=0`은 정상이다. load된 parameter만 freeze했기 때문에 해당 tensor가 학습 중 변하지 않아야 한다.


## Init Transfer: 논문식 축 비교

아래 표는 mode를 논문 주장에 맞게 cue, router, cue+router, full, full-except-cue/router 계열로 묶은 것이다. `family mean`은 해당 family 내 모든 row 평균이고, `best mode mean`은 family 안에서 가장 좋은 mode 평균이다.

| pair | family | best mode in family | rows | family mean gain | best mode mean gain | wins |
| --- | --- | --- | --- | --- | --- | --- |
| lastfm -> KuaiRec | Cue only | feature_encoder_init | 12 | -0.00173 | -0.00173 | 2/12 |
| lastfm -> KuaiRec | Router only | group_router_init | 24 | -0.00078 | -0.00078 | 16/24 |
| lastfm -> KuaiRec | Cue+router | feature_encoder_group_router_init | 24 | -0.00126 | -0.00126 | 8/24 |
| lastfm -> KuaiRec | Full | full_model_init | 12 | -0.00183 | -0.00183 | 6/12 |
| lastfm -> KuaiRec | Full except cue/router | full_except_feature_router_init | 12 | -0.00025 | -0.00025 | 7/12 |
| foursquare -> KuaiRec | Cue only | feature_encoder_init | 12 | +0.00077 | +0.00077 | 8/12 |
| foursquare -> KuaiRec | Router only | group_router_init | 24 | +0.00075 | +0.00075 | 12/24 |
| foursquare -> KuaiRec | Cue+router | feature_encoder_group_router_init | 24 | -0.00167 | -0.00167 | 6/24 |
| foursquare -> KuaiRec | Full | full_model_init | 12 | -0.00173 | -0.00173 | 6/12 |
| foursquare -> KuaiRec | Full except cue/router | full_except_feature_router_init | 12 | -0.00126 | -0.00126 | 4/12 |
| beauty -> retail_rocket | Cue only | feature_encoder_init | 12 | +0.00088 | +0.00088 | 8/12 |
| beauty -> retail_rocket | Router only | group_router_init | 24 | +0.00036 | +0.00036 | 10/24 |
| beauty -> retail_rocket | Cue+router | feature_encoder_group_router_init | 24 | -0.00050 | -0.00050 | 6/24 |
| beauty -> retail_rocket | Full | full_model_init | 12 | +0.00248 | +0.00248 | 10/12 |
| beauty -> retail_rocket | Full except cue/router | full_except_feature_router_init | 12 | +0.00249 | +0.00249 | 12/12 |
| retail_rocket -> beauty | Cue only | feature_encoder_init | 12 | +0.00089 | +0.00089 | 6/12 |
| retail_rocket -> beauty | Router only | group_router_init | 24 | +0.00041 | +0.00041 | 18/24 |
| retail_rocket -> beauty | Cue+router | feature_encoder_group_router_init | 24 | +0.00217 | +0.00217 | 14/24 |
| retail_rocket -> beauty | Full | full_model_init | 12 | -0.00256 | -0.00256 | 4/12 |
| retail_rocket -> beauty | Full except cue/router | full_except_feature_router_init | 12 | -0.00996 | -0.00996 | 1/12 |


해석 포인트는 다음과 같다.

- `retail_rocket -> beauty`는 cue+router 계열이 full보다 좋다. 이 축은 feature/router transfer 주장을 가장 직접적으로 지지한다.
- `beauty -> retail_rocket`은 full 계열이 가장 강하지만, cue-only도 양수다. commerce 사이 transfer에서는 일반 representation도 꽤 먹힌다.
- `foursquare -> KuaiRec`은 cue-only와 router-only가 full보다 낫다. rich-context target에서 full transfer보다 작은 부분 transfer가 안전하다는 신호다.
- `lastfm -> KuaiRec`은 평균적으로 약하다. rich-context끼리라도 source domain이 항상 좋은 것은 아니다.


## Init Transfer Full Table: Seed만 평균, Hparam 나열

| pair | hparam | mode | seeds | native mean | transfer mean | gain | wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lastfm -> KuaiRec | shared_3 | all_router_init | 3 | 0.33733 | 0.33980 | +0.00247 | 3/3 |
| lastfm -> KuaiRec | shared_3 | feature_encoder_a12_router_init | 3 | 0.33733 | 0.33773 | +0.00040 | 1/3 |
| lastfm -> KuaiRec | shared_3 | feature_encoder_group_router_init | 3 | 0.33733 | 0.33773 | +0.00040 | 1/3 |
| lastfm -> KuaiRec | shared_3 | feature_encoder_init | 3 | 0.33733 | 0.33630 | -0.00103 | 1/3 |
| lastfm -> KuaiRec | shared_3 | feature_encoder_router_init | 3 | 0.33733 | 0.33773 | +0.00040 | 1/3 |
| lastfm -> KuaiRec | shared_3 | full_except_feature_router_init | 3 | 0.33733 | 0.33623 | -0.00110 | 1/3 |
| lastfm -> KuaiRec | shared_3 | full_model_init | 3 | 0.33733 | 0.33957 | +0.00223 | 2/3 |
| lastfm -> KuaiRec | shared_3 | group_router_init | 3 | 0.33733 | 0.33980 | +0.00247 | 3/3 |
| lastfm -> KuaiRec | shared_4 | all_router_init | 3 | 0.33647 | 0.33703 | +0.00057 | 2/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_a12_router_init | 3 | 0.33647 | 0.33840 | +0.00193 | 2/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_group_router_init | 3 | 0.33647 | 0.33840 | +0.00193 | 2/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_init | 3 | 0.33647 | 0.33563 | -0.00083 | 0/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_router_init | 3 | 0.33647 | 0.33840 | +0.00193 | 2/3 |
| lastfm -> KuaiRec | shared_4 | full_except_feature_router_init | 3 | 0.33647 | 0.33720 | +0.00073 | 3/3 |
| lastfm -> KuaiRec | shared_4 | full_model_init | 3 | 0.33647 | 0.33487 | -0.00160 | 2/3 |
| lastfm -> KuaiRec | shared_4 | group_router_init | 3 | 0.33647 | 0.33703 | +0.00057 | 2/3 |
| lastfm -> KuaiRec | shared_5 | all_router_init | 3 | 0.33810 | 0.33297 | -0.00513 | 2/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_a12_router_init | 3 | 0.33810 | 0.33313 | -0.00497 | 1/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_group_router_init | 3 | 0.33810 | 0.33313 | -0.00497 | 1/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_init | 3 | 0.33810 | 0.33623 | -0.00187 | 1/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_router_init | 3 | 0.33810 | 0.33313 | -0.00497 | 1/3 |
| lastfm -> KuaiRec | shared_5 | full_except_feature_router_init | 3 | 0.33810 | 0.33773 | -0.00037 | 2/3 |
| lastfm -> KuaiRec | shared_5 | full_model_init | 3 | 0.33810 | 0.33593 | -0.00217 | 2/3 |
| lastfm -> KuaiRec | shared_5 | group_router_init | 3 | 0.33810 | 0.33297 | -0.00513 | 2/3 |
| lastfm -> KuaiRec | shared_6 | all_router_init | 3 | 0.33670 | 0.33570 | -0.00100 | 1/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_a12_router_init | 3 | 0.33670 | 0.33430 | -0.00240 | 0/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_group_router_init | 3 | 0.33670 | 0.33430 | -0.00240 | 0/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_init | 3 | 0.33670 | 0.33350 | -0.00320 | 0/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_router_init | 3 | 0.33670 | 0.33430 | -0.00240 | 0/3 |
| lastfm -> KuaiRec | shared_6 | full_except_feature_router_init | 3 | 0.33670 | 0.33643 | -0.00027 | 1/3 |
| lastfm -> KuaiRec | shared_6 | full_model_init | 3 | 0.33670 | 0.33093 | -0.00577 | 0/3 |
| lastfm -> KuaiRec | shared_6 | group_router_init | 3 | 0.33670 | 0.33570 | -0.00100 | 1/3 |
| foursquare -> KuaiRec | shared_3 | all_router_init | 3 | 0.33733 | 0.33680 | -0.00053 | 1/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_a12_router_init | 3 | 0.33733 | 0.33457 | -0.00277 | 1/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_group_router_init | 3 | 0.33733 | 0.33457 | -0.00277 | 1/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_init | 3 | 0.33733 | 0.33807 | +0.00073 | 2/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_router_init | 3 | 0.33733 | 0.33457 | -0.00277 | 1/3 |
| foursquare -> KuaiRec | shared_3 | full_except_feature_router_init | 3 | 0.33733 | 0.33847 | +0.00113 | 2/3 |
| foursquare -> KuaiRec | shared_3 | full_model_init | 3 | 0.33733 | 0.33437 | -0.00297 | 1/3 |
| foursquare -> KuaiRec | shared_3 | group_router_init | 3 | 0.33733 | 0.33680 | -0.00053 | 1/3 |
| foursquare -> KuaiRec | shared_4 | all_router_init | 3 | 0.33647 | 0.33817 | +0.00170 | 2/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_a12_router_init | 3 | 0.33647 | 0.33860 | +0.00213 | 2/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_group_router_init | 3 | 0.33647 | 0.33860 | +0.00213 | 2/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_init | 3 | 0.33647 | 0.33790 | +0.00143 | 2/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_router_init | 3 | 0.33647 | 0.33860 | +0.00213 | 2/3 |
| foursquare -> KuaiRec | shared_4 | full_except_feature_router_init | 3 | 0.33647 | 0.33530 | -0.00117 | 1/3 |
| foursquare -> KuaiRec | shared_4 | full_model_init | 3 | 0.33647 | 0.33817 | +0.00170 | 2/3 |
| foursquare -> KuaiRec | shared_4 | group_router_init | 3 | 0.33647 | 0.33817 | +0.00170 | 2/3 |
| foursquare -> KuaiRec | shared_5 | all_router_init | 3 | 0.33810 | 0.33930 | +0.00120 | 2/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_a12_router_init | 3 | 0.33810 | 0.33533 | -0.00277 | 0/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_group_router_init | 3 | 0.33810 | 0.33533 | -0.00277 | 0/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_init | 3 | 0.33810 | 0.33787 | -0.00023 | 2/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_router_init | 3 | 0.33810 | 0.33533 | -0.00277 | 0/3 |
| foursquare -> KuaiRec | shared_5 | full_except_feature_router_init | 3 | 0.33810 | 0.33397 | -0.00413 | 0/3 |
| foursquare -> KuaiRec | shared_5 | full_model_init | 3 | 0.33810 | 0.33710 | -0.00100 | 2/3 |
| foursquare -> KuaiRec | shared_5 | group_router_init | 3 | 0.33810 | 0.33930 | +0.00120 | 2/3 |
| foursquare -> KuaiRec | shared_6 | all_router_init | 3 | 0.33670 | 0.33733 | +0.00063 | 1/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_a12_router_init | 3 | 0.33670 | 0.33343 | -0.00327 | 0/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_group_router_init | 3 | 0.33670 | 0.33343 | -0.00327 | 0/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_init | 3 | 0.33670 | 0.33787 | +0.00117 | 2/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_router_init | 3 | 0.33670 | 0.33343 | -0.00327 | 0/3 |
| foursquare -> KuaiRec | shared_6 | full_except_feature_router_init | 3 | 0.33670 | 0.33583 | -0.00087 | 1/3 |
| foursquare -> KuaiRec | shared_6 | full_model_init | 3 | 0.33670 | 0.33207 | -0.00463 | 1/3 |
| foursquare -> KuaiRec | shared_6 | group_router_init | 3 | 0.33670 | 0.33733 | +0.00063 | 1/3 |
| beauty -> retail_rocket | shared_3 | all_router_init | 3 | 0.37070 | 0.37120 | +0.00050 | 1/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_a12_router_init | 3 | 0.37070 | 0.37080 | +0.00010 | 1/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_group_router_init | 3 | 0.37070 | 0.37080 | +0.00010 | 1/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_init | 3 | 0.37070 | 0.37203 | +0.00133 | 2/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_router_init | 3 | 0.37070 | 0.37080 | +0.00010 | 1/3 |
| beauty -> retail_rocket | shared_3 | full_except_feature_router_init | 3 | 0.37070 | 0.37400 | +0.00330 | 3/3 |
| beauty -> retail_rocket | shared_3 | full_model_init | 3 | 0.37070 | 0.37457 | +0.00387 | 3/3 |
| beauty -> retail_rocket | shared_3 | group_router_init | 3 | 0.37070 | 0.37120 | +0.00050 | 1/3 |
| beauty -> retail_rocket | shared_4 | all_router_init | 3 | 0.37043 | 0.37093 | +0.00050 | 2/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_a12_router_init | 3 | 0.37043 | 0.36977 | -0.00067 | 1/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_group_router_init | 3 | 0.37043 | 0.36977 | -0.00067 | 1/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_init | 3 | 0.37043 | 0.37147 | +0.00103 | 3/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_router_init | 3 | 0.37043 | 0.36977 | -0.00067 | 1/3 |
| beauty -> retail_rocket | shared_4 | full_except_feature_router_init | 3 | 0.37043 | 0.37300 | +0.00257 | 3/3 |
| beauty -> retail_rocket | shared_4 | full_model_init | 3 | 0.37043 | 0.37257 | +0.00213 | 3/3 |
| beauty -> retail_rocket | shared_4 | group_router_init | 3 | 0.37043 | 0.37093 | +0.00050 | 2/3 |
| beauty -> retail_rocket | shared_5 | all_router_init | 3 | 0.37030 | 0.37107 | +0.00077 | 1/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_a12_router_init | 3 | 0.37030 | 0.37073 | +0.00043 | 1/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_group_router_init | 3 | 0.37030 | 0.37073 | +0.00043 | 1/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_init | 3 | 0.37030 | 0.37253 | +0.00223 | 3/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_router_init | 3 | 0.37030 | 0.37073 | +0.00043 | 1/3 |
| beauty -> retail_rocket | shared_5 | full_except_feature_router_init | 3 | 0.37030 | 0.37317 | +0.00287 | 3/3 |
| beauty -> retail_rocket | shared_5 | full_model_init | 3 | 0.37030 | 0.37290 | +0.00260 | 3/3 |
| beauty -> retail_rocket | shared_5 | group_router_init | 3 | 0.37030 | 0.37107 | +0.00077 | 1/3 |
| beauty -> retail_rocket | shared_6 | all_router_init | 3 | 0.37080 | 0.37047 | -0.00033 | 1/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_a12_router_init | 3 | 0.37080 | 0.36893 | -0.00187 | 0/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_group_router_init | 3 | 0.37080 | 0.36893 | -0.00187 | 0/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_init | 3 | 0.37080 | 0.36970 | -0.00110 | 0/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_router_init | 3 | 0.37080 | 0.36893 | -0.00187 | 0/3 |
| beauty -> retail_rocket | shared_6 | full_except_feature_router_init | 3 | 0.37080 | 0.37203 | +0.00123 | 3/3 |
| beauty -> retail_rocket | shared_6 | full_model_init | 3 | 0.37080 | 0.37210 | +0.00130 | 1/3 |
| beauty -> retail_rocket | shared_6 | group_router_init | 3 | 0.37080 | 0.37047 | -0.00033 | 1/3 |
| retail_rocket -> beauty | shared_3 | all_router_init | 3 | 0.07157 | 0.06713 | -0.00443 | 2/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_a12_router_init | 3 | 0.07157 | 0.07270 | +0.00113 | 1/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_group_router_init | 3 | 0.07157 | 0.07270 | +0.00113 | 1/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_init | 3 | 0.07157 | 0.07937 | +0.00780 | 3/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_router_init | 3 | 0.07157 | 0.07270 | +0.00113 | 1/3 |
| retail_rocket -> beauty | shared_3 | full_except_feature_router_init | 3 | 0.07157 | 0.05553 | -0.01603 | 0/3 |
| retail_rocket -> beauty | shared_3 | full_model_init | 3 | 0.07157 | 0.06000 | -0.01157 | 0/3 |
| retail_rocket -> beauty | shared_3 | group_router_init | 3 | 0.07157 | 0.06713 | -0.00443 | 2/3 |
| retail_rocket -> beauty | shared_4 | all_router_init | 3 | 0.06383 | 0.06570 | +0.00187 | 2/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_a12_router_init | 3 | 0.06383 | 0.06363 | -0.00020 | 1/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_group_router_init | 3 | 0.06383 | 0.06363 | -0.00020 | 1/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_init | 3 | 0.06383 | 0.05980 | -0.00403 | 0/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_router_init | 3 | 0.06383 | 0.06363 | -0.00020 | 1/3 |
| retail_rocket -> beauty | shared_4 | full_except_feature_router_init | 3 | 0.06383 | 0.05977 | -0.00407 | 1/3 |
| retail_rocket -> beauty | shared_4 | full_model_init | 3 | 0.06383 | 0.06190 | -0.00193 | 1/3 |
| retail_rocket -> beauty | shared_4 | group_router_init | 3 | 0.06383 | 0.06570 | +0.00187 | 2/3 |
| retail_rocket -> beauty | shared_5 | all_router_init | 3 | 0.06467 | 0.06597 | +0.00130 | 2/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_a12_router_init | 3 | 0.06467 | 0.06500 | +0.00033 | 2/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_group_router_init | 3 | 0.06467 | 0.06500 | +0.00033 | 2/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_init | 3 | 0.06467 | 0.06357 | -0.00110 | 1/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_router_init | 3 | 0.06467 | 0.06500 | +0.00033 | 2/3 |
| retail_rocket -> beauty | shared_5 | full_except_feature_router_init | 3 | 0.06467 | 0.05320 | -0.01147 | 0/3 |
| retail_rocket -> beauty | shared_5 | full_model_init | 3 | 0.06467 | 0.06113 | -0.00353 | 1/3 |
| retail_rocket -> beauty | shared_5 | group_router_init | 3 | 0.06467 | 0.06597 | +0.00130 | 2/3 |
| retail_rocket -> beauty | shared_6 | all_router_init | 3 | 0.05877 | 0.06167 | +0.00290 | 3/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_a12_router_init | 3 | 0.05877 | 0.06617 | +0.00740 | 3/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_group_router_init | 3 | 0.05877 | 0.06617 | +0.00740 | 3/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_init | 3 | 0.05877 | 0.05967 | +0.00090 | 2/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_router_init | 3 | 0.05877 | 0.06617 | +0.00740 | 3/3 |
| retail_rocket -> beauty | shared_6 | full_except_feature_router_init | 3 | 0.05877 | 0.05050 | -0.00827 | 0/3 |
| retail_rocket -> beauty | shared_6 | full_model_init | 3 | 0.05877 | 0.06557 | +0.00680 | 2/3 |
| retail_rocket -> beauty | shared_6 | group_router_init | 3 | 0.05877 | 0.06167 | +0.00290 | 3/3 |


## Hparam별 Top-3 Init Mode

| pair | hparam | top-3 modes by mean gain |
| --- | --- | --- |
| lastfm -> KuaiRec | shared_3 | group_router_init (+0.00247, 3/3); all_router_init (+0.00247, 3/3); full_model_init (+0.00223, 2/3) |
| lastfm -> KuaiRec | shared_4 | feature_encoder_router_init (+0.00193, 2/3); feature_encoder_group_router_init (+0.00193, 2/3); feature_encoder_a12_router_init (+0.00193, 2/3) |
| lastfm -> KuaiRec | shared_5 | full_except_feature_router_init (-0.00037, 2/3); feature_encoder_init (-0.00187, 1/3); full_model_init (-0.00217, 2/3) |
| lastfm -> KuaiRec | shared_6 | full_except_feature_router_init (-0.00027, 1/3); group_router_init (-0.00100, 1/3); all_router_init (-0.00100, 1/3) |
| foursquare -> KuaiRec | shared_3 | full_except_feature_router_init (+0.00113, 2/3); feature_encoder_init (+0.00073, 2/3); group_router_init (-0.00053, 1/3) |
| foursquare -> KuaiRec | shared_4 | feature_encoder_router_init (+0.00213, 2/3); feature_encoder_group_router_init (+0.00213, 2/3); feature_encoder_a12_router_init (+0.00213, 2/3) |
| foursquare -> KuaiRec | shared_5 | group_router_init (+0.00120, 2/3); all_router_init (+0.00120, 2/3); feature_encoder_init (-0.00023, 2/3) |
| foursquare -> KuaiRec | shared_6 | feature_encoder_init (+0.00117, 2/3); group_router_init (+0.00063, 1/3); all_router_init (+0.00063, 1/3) |
| beauty -> retail_rocket | shared_3 | full_model_init (+0.00387, 3/3); full_except_feature_router_init (+0.00330, 3/3); feature_encoder_init (+0.00133, 2/3) |
| beauty -> retail_rocket | shared_4 | full_except_feature_router_init (+0.00257, 3/3); full_model_init (+0.00213, 3/3); feature_encoder_init (+0.00103, 3/3) |
| beauty -> retail_rocket | shared_5 | full_except_feature_router_init (+0.00287, 3/3); full_model_init (+0.00260, 3/3); feature_encoder_init (+0.00223, 3/3) |
| beauty -> retail_rocket | shared_6 | full_model_init (+0.00130, 1/3); full_except_feature_router_init (+0.00123, 3/3); group_router_init (-0.00033, 1/3) |
| retail_rocket -> beauty | shared_3 | feature_encoder_init (+0.00780, 3/3); feature_encoder_router_init (+0.00113, 1/3); feature_encoder_group_router_init (+0.00113, 1/3) |
| retail_rocket -> beauty | shared_4 | group_router_init (+0.00187, 2/3); all_router_init (+0.00187, 2/3); feature_encoder_router_init (-0.00020, 1/3) |
| retail_rocket -> beauty | shared_5 | group_router_init (+0.00130, 2/3); all_router_init (+0.00130, 2/3); feature_encoder_router_init (+0.00033, 2/3) |
| retail_rocket -> beauty | shared_6 | feature_encoder_router_init (+0.00740, 3/3); feature_encoder_group_router_init (+0.00740, 3/3); feature_encoder_a12_router_init (+0.00740, 3/3) |


## Freeze Full Table: Seed만 평균, Hparam 나열

`freeze effect`는 같은 pair/hparam/mode의 init-only 평균 gain 대비 freeze 평균 gain 차이다.

| pair | hparam | mode | seeds | freeze gain | freeze effect | wins |
| --- | --- | --- | --- | --- | --- | --- |
| lastfm -> KuaiRec | shared_3 | feature_encoder_a12_router_init | 3 | -0.00603 | -0.00643 | 0/3 |
| lastfm -> KuaiRec | shared_3 | feature_encoder_group_router_init | 3 | -0.00603 | -0.00643 | 0/3 |
| lastfm -> KuaiRec | shared_3 | full_model_init | 3 | -0.04890 | -0.05113 | 0/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_a12_router_init | 3 | -0.00520 | -0.00713 | 1/3 |
| lastfm -> KuaiRec | shared_4 | feature_encoder_group_router_init | 3 | -0.00520 | -0.00713 | 1/3 |
| lastfm -> KuaiRec | shared_4 | full_model_init | 3 | -0.05243 | -0.05083 | 0/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_a12_router_init | 3 | -0.00837 | -0.00340 | 0/3 |
| lastfm -> KuaiRec | shared_5 | feature_encoder_group_router_init | 3 | -0.00837 | -0.00340 | 0/3 |
| lastfm -> KuaiRec | shared_5 | full_model_init | 3 | -0.05910 | -0.05693 | 0/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_a12_router_init | 3 | -0.00537 | -0.00297 | 0/3 |
| lastfm -> KuaiRec | shared_6 | feature_encoder_group_router_init | 3 | -0.00537 | -0.00297 | 0/3 |
| lastfm -> KuaiRec | shared_6 | full_model_init | 3 | -0.07557 | -0.06980 | 0/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_a12_router_init | 3 | -0.01040 | -0.00763 | 0/3 |
| foursquare -> KuaiRec | shared_3 | feature_encoder_group_router_init | 3 | -0.01040 | -0.00763 | 0/3 |
| foursquare -> KuaiRec | shared_3 | full_model_init | 3 | -0.02363 | -0.02067 | 0/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_a12_router_init | 3 | -0.00197 | -0.00410 | 1/3 |
| foursquare -> KuaiRec | shared_4 | feature_encoder_group_router_init | 3 | -0.00197 | -0.00410 | 1/3 |
| foursquare -> KuaiRec | shared_4 | full_model_init | 3 | -0.03523 | -0.03693 | 0/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_a12_router_init | 3 | -0.00483 | -0.00207 | 0/3 |
| foursquare -> KuaiRec | shared_5 | feature_encoder_group_router_init | 3 | -0.00483 | -0.00207 | 0/3 |
| foursquare -> KuaiRec | shared_5 | full_model_init | 3 | -0.04067 | -0.03967 | 0/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_a12_router_init | 3 | -0.00963 | -0.00637 | 0/3 |
| foursquare -> KuaiRec | shared_6 | feature_encoder_group_router_init | 3 | -0.00963 | -0.00637 | 0/3 |
| foursquare -> KuaiRec | shared_6 | full_model_init | 3 | -0.03120 | -0.02657 | 0/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_a12_router_init | 3 | +0.00120 | +0.00110 | 3/3 |
| beauty -> retail_rocket | shared_3 | feature_encoder_group_router_init | 3 | +0.00120 | +0.00110 | 3/3 |
| beauty -> retail_rocket | shared_3 | full_model_init | 3 | -0.08573 | -0.08960 | 0/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_a12_router_init | 3 | +0.00173 | +0.00240 | 3/3 |
| beauty -> retail_rocket | shared_4 | feature_encoder_group_router_init | 3 | +0.00173 | +0.00240 | 3/3 |
| beauty -> retail_rocket | shared_4 | full_model_init | 3 | -0.07920 | -0.08133 | 0/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_a12_router_init | 3 | +0.00130 | +0.00087 | 2/3 |
| beauty -> retail_rocket | shared_5 | feature_encoder_group_router_init | 3 | +0.00130 | +0.00087 | 2/3 |
| beauty -> retail_rocket | shared_5 | full_model_init | 3 | -0.08520 | -0.08780 | 0/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_a12_router_init | 3 | -0.00043 | +0.00143 | 1/3 |
| beauty -> retail_rocket | shared_6 | feature_encoder_group_router_init | 3 | -0.00043 | +0.00143 | 1/3 |
| beauty -> retail_rocket | shared_6 | full_model_init | 3 | -0.08520 | -0.08650 | 0/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_a12_router_init | 3 | -0.00433 | -0.00547 | 0/3 |
| retail_rocket -> beauty | shared_3 | feature_encoder_group_router_init | 3 | -0.00433 | -0.00547 | 0/3 |
| retail_rocket -> beauty | shared_3 | full_model_init | 3 | -0.02923 | -0.01767 | 0/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_a12_router_init | 3 | -0.00300 | -0.00280 | 1/3 |
| retail_rocket -> beauty | shared_4 | feature_encoder_group_router_init | 3 | -0.00300 | -0.00280 | 1/3 |
| retail_rocket -> beauty | shared_4 | full_model_init | 3 | -0.01367 | -0.01173 | 1/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_a12_router_init | 3 | -0.00320 | -0.00353 | 1/3 |
| retail_rocket -> beauty | shared_5 | feature_encoder_group_router_init | 3 | -0.00320 | -0.00353 | 1/3 |
| retail_rocket -> beauty | shared_5 | full_model_init | 3 | -0.00753 | -0.00400 | 1/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_a12_router_init | 3 | +0.00633 | -0.00107 | 3/3 |
| retail_rocket -> beauty | shared_6 | feature_encoder_group_router_init | 3 | +0.00633 | -0.00107 | 3/3 |
| retail_rocket -> beauty | shared_6 | full_model_init | 3 | -0.00603 | -0.01283 | 0/3 |


Freeze 결론은 group1에서 명확하다. full model freeze는 모든 pair에서 큰 손실이고, feature/router freeze는 `beauty -> retail_rocket`에서만 안정적으로 양수다. KuaiRec target에서는 freeze보다 init 후 fine-tune이 낫다.


## Multihop Full Table: Seed만 평균, Hparam 나열

| role | hparam | mode | seeds | native C mean | transfer C mean | gain | wins |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a_to_c_direct | shared_3 | feature_encoder_a12_router_init | 3 | 0.33733 | 0.33507 | -0.00227 | 1/3 |
| a_to_c_direct | shared_3 | full_model_init | 3 | 0.33733 | 0.33110 | -0.00623 | 0/3 |
| b_to_c_direct | shared_3 | feature_encoder_a12_router_init | 3 | 0.33733 | 0.33570 | -0.00163 | 1/3 |
| b_to_c_direct | shared_3 | full_model_init | 3 | 0.33733 | 0.33523 | -0.00210 | 1/3 |
| a_to_b_to_c | shared_3 | feature_encoder_a12_router_init | 3 | 0.33733 | 0.33740 | +0.00007 | 1/3 |
| a_to_b_to_c | shared_3 | full_model_init | 3 | 0.33733 | 0.33900 | +0.00167 | 3/3 |
| a_to_c_direct | shared_4 | feature_encoder_a12_router_init | 3 | 0.33647 | 0.33540 | -0.00107 | 0/3 |
| a_to_c_direct | shared_4 | full_model_init | 3 | 0.33647 | 0.33180 | -0.00467 | 0/3 |
| b_to_c_direct | shared_4 | feature_encoder_a12_router_init | 3 | 0.33647 | 0.33643 | -0.00003 | 1/3 |
| b_to_c_direct | shared_4 | full_model_init | 3 | 0.33647 | 0.33553 | -0.00093 | 1/3 |
| a_to_b_to_c | shared_4 | feature_encoder_a12_router_init | 3 | 0.33647 | 0.33817 | +0.00170 | 2/3 |
| a_to_b_to_c | shared_4 | full_model_init | 3 | 0.33647 | 0.33773 | +0.00127 | 2/3 |
| a_to_c_direct | shared_5 | feature_encoder_a12_router_init | 3 | 0.33810 | 0.33733 | -0.00077 | 1/3 |
| a_to_c_direct | shared_5 | full_model_init | 3 | 0.33810 | 0.33767 | -0.00043 | 2/3 |
| b_to_c_direct | shared_5 | feature_encoder_a12_router_init | 3 | 0.33810 | 0.33860 | +0.00050 | 1/3 |
| b_to_c_direct | shared_5 | full_model_init | 3 | 0.33810 | 0.34007 | +0.00197 | 2/3 |
| a_to_b_to_c | shared_5 | feature_encoder_a12_router_init | 3 | 0.33810 | 0.33903 | +0.00093 | 2/3 |
| a_to_b_to_c | shared_5 | full_model_init | 3 | 0.33810 | 0.33707 | -0.00103 | 1/3 |
| a_to_c_direct | shared_6 | feature_encoder_a12_router_init | 3 | 0.33670 | 0.33620 | -0.00050 | 1/3 |
| a_to_c_direct | shared_6 | full_model_init | 3 | 0.33670 | 0.33040 | -0.00630 | 1/3 |
| b_to_c_direct | shared_6 | feature_encoder_a12_router_init | 3 | 0.33670 | 0.33760 | +0.00090 | 2/3 |
| b_to_c_direct | shared_6 | full_model_init | 3 | 0.33670 | 0.33327 | -0.00343 | 1/3 |
| a_to_b_to_c | shared_6 | feature_encoder_a12_router_init | 3 | 0.33670 | 0.33990 | +0.00320 | 3/3 |
| a_to_b_to_c | shared_6 | full_model_init | 3 | 0.33670 | 0.33690 | +0.00020 | 2/3 |


Multihop은 `beauty -> retail_rocket -> KuaiRec`의 feature A12 router가 가장 좋은 신호다. 직접 `beauty -> KuaiRec`는 음수인데, commerce bridge인 `retail_rocket`을 거치면 평균 gain이 양수로 바뀐다. 논문식으로는 “domain bridge가 cue/router portability를 개선한다”는 보조 실험으로 쓸 수 있다.


## 논문에서 비교할 수 있는 방식

1. **Module transfer 비교**: `scratch`, `cue only`, `router_e only`, `cue+router_e`, `cue+router_d/e`, `full`, `full except cue/router`를 같은 pair/hparam/seed 안에서 비교한다. 핵심 주장은 full transfer가 아니라 cue/router subset이 더 안정적일 수 있다는 점이다.

2. **Target domain별 비교**: commerce target(`beauty`, `retail_rocket`)과 rich-context target(`KuaiRec`)을 나눠서, 어느 target에서 freeze가 손해인지 보여준다. Group1에서는 KuaiRec freeze가 명확히 나쁘다.

3. **Sequential vs direct transfer**: `A -> C`, `B -> C`, `A -> B -> C`를 같은 C baseline에 대해 비교한다. Group1에서는 sequential feature/router transfer가 direct보다 좋다.

4. **Robustness 표기**: 평균 gain만 쓰지 말고 `wins/3 seeds` 또는 `wins/12 hparam-seed`를 같이 둔다. 작은 gain이라도 win-rate가 높으면 안정적, 평균은 높지만 win-rate가 낮으면 hparam 의존적이라고 해석한다.

5. **Negative transfer 분석**: full model freeze처럼 큰 음수인 설정을 failure case로 보여주면, 왜 부분 transfer가 필요한지 설명하기 쉽다.
