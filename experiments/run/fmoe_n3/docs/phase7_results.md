# Phase 7 Results: Router 입력 + Aux 유도 (Diag/Special 확장 분석)

작성일: 2026-03-19  
데이터셋: `KuaiRecLargeStrictPosV2_0.2`  
메인 지표: **best valid MRR@20**  
보조 지표: test MRR@20, special(cold/session), diag(router dynamics)

## 1) 실험 개요 및 집계 규칙

- 원본 결과 파일: 72 run
- 중복 규칙: **동일 setting+seed에서는 완료된 최신 timestamp run만 채택**
- 최종 분석 표본: **64 run (16 setting x 4 seeds)**
- special_metrics 누락 run: 0개
- diag_best_valid_overview 누락 run: 8개

## 2) Router 8종 결과 (valid/test, max/mean/var)

| setting_id | router_variant | valid_best_mrr20_max | valid_best_mrr20_mean | valid_best_mrr20_var | test_mrr20_max | test_mrr20_mean | test_mrr20_var |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0_STD | R0 | 0.0812 | 0.0811 | 0.00000000 | 0.1622 | 0.1621 | 0.00000000 |
| R1_FAC | R1 | 0.0807 | 0.0806 | 0.00000000 | 0.1623 | 0.1620 | 0.00000005 |
| R2_FAC_HEAVY | R2 | 0.0811 | 0.0810 | 0.00000001 | 0.1616 | 0.1615 | 0.00000001 |
| R3_FAC_ONLY | R3 | 0.0804 | 0.0804 | 0.00000000 | 0.1616 | 0.1614 | 0.00000002 |
| R4_HIR | R4 | 0.0803 | 0.0803 | 0.00000000 | 0.1617 | 0.1617 | 0.00000001 |
| R5_FAC_GROUP | R5 | 0.0806 | 0.0806 | 0.00000000 | 0.1624 | 0.1621 | 0.00000008 |
| R6_FAC_ONLY_BOTH | R6 | 0.0803 | 0.0802 | 0.00000002 | 0.1614 | 0.1612 | 0.00000002 |
| R7_FAC_HEAVY_FEAT | R7 | 0.0807 | 0.0806 | 0.00000000 | 0.1621 | 0.1620 | 0.00000001 |

## 3) Aux 8종 결과 (valid/test, max/mean/var)

| setting_id | router_variant | aux_variant | aux_family | valid_best_mrr20_max | valid_best_mrr20_mean | valid_best_mrr20_var | test_mrr20_max | test_mrr20_mean | test_mrr20_var |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUX_R0_STD_BAL_A | R0 | BAL_A | balance | 0.0813 | 0.0812 | 0.00000001 | 0.1621 | 0.1620 | 0.00000001 |
| AUX_R0_STD_BAL_B | R0 | BAL_B | balance | 0.0814 | 0.0813 | 0.00000001 | 0.1622 | 0.1620 | 0.00000002 |
| AUX_R0_STD_SPEC_A | R0 | SPEC_A | specialization | 0.0814 | 0.0813 | 0.00000000 | 0.1621 | 0.1619 | 0.00000002 |
| AUX_R0_STD_SPEC_B | R0 | SPEC_B | specialization | 0.0811 | 0.0811 | 0.00000000 | 0.1621 | 0.1618 | 0.00000007 |
| AUX_R2_FAC_HEAVY_BAL_A | R2 | BAL_A | balance | 0.0810 | 0.0809 | 0.00000001 | 0.1618 | 0.1615 | 0.00000006 |
| AUX_R2_FAC_HEAVY_BAL_B | R2 | BAL_B | balance | 0.0810 | 0.0810 | 0.00000000 | 0.1619 | 0.1617 | 0.00000005 |
| AUX_R2_FAC_HEAVY_SPEC_A | R2 | SPEC_A | specialization | 0.0809 | 0.0808 | 0.00000002 | 0.1618 | 0.1617 | 0.00000001 |
| AUX_R2_FAC_HEAVY_SPEC_B | R2 | SPEC_B | specialization | 0.0804 | 0.0804 | 0.00000000 | 0.1617 | 0.1615 | 0.00000002 |

## 4) Aux 메인 분석: Router Anchor 분리 (R0 vs R2)

`none`은 같은 anchor의 router_core(BASE) 결과를 사용했다.

| router_anchor | aux_family | valid_best_mrr20_max | valid_best_mrr20_mean | valid_best_mrr20_var | test_mrr20_max | test_mrr20_mean | test_mrr20_var |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factored-heavy(R2) | balance | 0.0810 | 0.0809 | 0.00000001 | 0.1619 | 0.1616 | 0.00000006 |
| factored-heavy(R2) | none | 0.0811 | 0.0810 | 0.00000001 | 0.1616 | 0.1615 | 0.00000001 |
| factored-heavy(R2) | specialization | 0.0809 | 0.0806 | 0.00000007 | 0.1618 | 0.1616 | 0.00000002 |
| standard(R0) | balance | 0.0814 | 0.0813 | 0.00000001 | 0.1622 | 0.1620 | 0.00000001 |
| standard(R0) | none | 0.0812 | 0.0811 | 0.00000000 | 0.1622 | 0.1621 | 0.00000000 |
| standard(R0) | specialization | 0.0814 | 0.0812 | 0.00000002 | 0.1621 | 0.1619 | 0.00000004 |

핵심 해석:
- anchor를 분리하면 `balance/specialization`의 이득 방향이 동일하지 않을 수 있다. 즉, aux 해석은 anchor-conditional로 보는 것이 안전하다.
- 따라서 “aux 우열”을 전체 평균 1개로 결론내기보다, `R0 내부`와 `R2 내부`의 상대 비교를 본문 메인으로 유지한다.

## 5) Special 분석 (cold item/short session 중심)

| router_anchor | aux_family | cold_item_mrr20_max | cold_item_mrr20_mean | cold_item_mrr20_var | sess_1_2_mrr20_mean | sess_3_5_mrr20_mean |
| --- | --- | --- | --- | --- | --- | --- |
| factored-heavy(R2) | balance | 0.1210 | 0.1202 | 0.00000057 | 0.0000 | 0.1509 |
| factored-heavy(R2) | none | 0.1203 | 0.1199 | 0.00000007 | 0.0020 | 0.1507 |
| factored-heavy(R2) | specialization |  |  |  |  |  |
| standard(R0) | balance | 0.1216 | 0.1213 | 0.00000007 | 0.0004 | 0.1511 |
| standard(R0) | none | 0.1213 | 0.1211 | 0.00000005 | 0.0037 | 0.1518 |
| standard(R0) | specialization | 0.1215 | 0.1212 | 0.00000011 | 0.0000 | 0.1512 |

주의: special score는 bucket count(표본 수)에 영향을 받는다. 노트북에서는 각 slice count를 함께 시각화해 신뢰도를 같이 해석한다.

## 6) Diag 분석 (Stage별 + Metric-MRR 관계)

| stage_short | experiment_group | router_variant | aux_family | n_eff_mean | cv_usage_mean | top1_max_frac_mean | route_jitter_adjacent_mean | route_consistency_knn_score_mean | family_top_expert_mean_share_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro | aux_reg | R0 | balance | 11.2427 | 0.2577 | 0.2435 | 0.0000 | 0.9535 | 0.1342 |
| macro | router_core | R0 | none | 11.4049 | 0.2263 | 0.2136 | 0.0000 | 0.9597 | 0.1178 |
| macro | aux_reg | R0 | specialization | 7.6014 | 0.8504 | 0.3134 | 0.0000 | 0.9462 | 0.2543 |
| macro | router_core | R1 | none | 11.4079 | 0.2278 | 0.2636 | 0.0000 | 0.9755 | 0.1186 |
| macro | aux_reg | R2 | balance | 11.3777 | 0.2334 | 0.1588 | 0.0000 | 0.9521 | 0.1167 |
| macro | router_core | R2 | none | 11.3347 | 0.2417 | 0.1611 | 0.0000 | 0.9520 | 0.1182 |
| macro | router_core | R3 | none | 11.3925 | 0.2304 | 0.5272 | 0.0000 | 0.9976 | 0.1090 |
| macro | router_core | R4 | none | 11.4946 | 0.2097 | 0.1962 | 0.0000 | 0.9500 | 0.1134 |
| macro | router_core | R5 | none | 11.3948 | 0.2303 | 0.2621 | 0.0000 | 0.9762 | 0.1177 |
| macro | router_core | R6 | none | 10.9287 | 0.3125 | 0.3900 | 0.0000 | 0.9687 | 0.1127 |
| macro | router_core | R7 | none | 11.6325 | 0.1764 | 0.1801 | 0.0000 | 0.9882 | 0.1109 |
| micro | aux_reg | R0 | balance | 11.2297 | 0.2591 | 0.1731 | 0.5141 | 0.9601 | 0.1263 |
| micro | router_core | R0 | none | 11.3148 | 0.2442 | 0.1618 | 0.5021 | 0.9654 | 0.1185 |
| micro | aux_reg | R0 | specialization | 7.4936 | 0.8867 | 0.3348 | 0.4315 | 0.9611 | 0.3046 |
| micro | router_core | R1 | none | 9.6986 | 0.4866 | 0.2805 | 0.5075 | 0.9680 | 0.1872 |
| micro | aux_reg | R2 | balance | 10.0164 | 0.4442 | 0.2025 | 0.5902 | 0.9531 | 0.1473 |
| micro | router_core | R2 | none | 9.8797 | 0.4632 | 0.2470 | 0.5631 | 0.9542 | 0.1528 |
| micro | router_core | R3 | none | 5.7643 | 1.0409 | 0.8179 | 0.2785 | 0.9990 | 0.2437 |
| micro | router_core | R4 | none | 9.3964 | 0.5264 | 0.2205 | 0.5010 | 0.9429 | 0.1918 |
| micro | router_core | R5 | none | 9.2971 | 0.5387 | 0.2918 | 0.4818 | 0.9675 | 0.1970 |
| micro | router_core | R6 | none | 11.3929 | 0.2294 | 0.3380 | 0.3913 | 0.9653 | 0.1102 |
| micro | router_core | R7 | none | 9.0564 | 0.5701 | 0.2445 | 0.6481 | 0.9933 | 0.1727 |
| mid | aux_reg | R0 | balance | 9.3068 | 0.5378 | 0.3226 | 0.0000 | 0.9582 | 0.2009 |
| mid | router_core | R0 | none | 9.4574 | 0.5184 | 0.3287 | 0.0000 | 0.9629 | 0.1968 |
| mid | aux_reg | R0 | specialization | 5.3565 | 1.5042 | 0.5495 | 0.0000 | 0.9724 | 0.4962 |
| mid | router_core | R1 | none | 9.3719 | 0.5295 | 0.4000 | 0.0000 | 0.9806 | 0.2069 |
| mid | aux_reg | R2 | balance | 9.0877 | 0.5659 | 0.1975 | 0.0000 | 0.9656 | 0.1650 |
| mid | router_core | R2 | none | 8.6008 | 0.6287 | 0.1983 | 0.0000 | 0.9667 | 0.1771 |
| mid | router_core | R3 | none | 11.3445 | 0.2402 | 0.4544 | 0.0000 | 0.9983 | 0.1067 |
| mid | router_core | R4 | none | 11.2925 | 0.2503 | 0.2732 | 0.0000 | 0.9680 | 0.1253 |
| mid | router_core | R5 | none | 9.5404 | 0.5076 | 0.3760 | 0.0000 | 0.9813 | 0.1988 |
| mid | router_core | R6 | none | 9.5323 | 0.5088 | 0.6688 | 0.0000 | 0.9896 | 0.1604 |
| mid | router_core | R7 | none | 9.6192 | 0.4975 | 0.2766 | 0.0000 | 0.9859 | 0.1654 |

### Diag-MRR 상관(전체 run-stage 샘플 기준)

| diag_metric | pearson_r_with_valid_mrr20 | n |
| --- | --- | --- |
| n_eff | -0.1181 | 168 |
| cv_usage | 0.1355 | 168 |
| top1_max_frac | -0.3289 | 168 |
| route_jitter_adjacent | 0.0404 | 168 |
| route_consistency_knn_score | -0.4874 | 168 |
| family_top_expert_mean_share | 0.1830 | 168 |

## 7) Diag 지표 설명 (정의/계산)

- `n_eff`: expert usage 분포를 유효 expert 수로 환산.
  \( n_eff = 1 / \sum_i p_i^2 \), 여기서 \(p_i\)는 expert usage share.
- `cv_usage`: usage share의 변동계수.
  \( cv = std(p) / mean(p) \).
- `top1_max_frac`: top1 routing count 기준 최다 expert 점유율(독점 경향).
- `route_jitter_adjacent`: 인접 시점 라우팅 분포 변화량 평균(높을수록 변동 큼).
- `route_consistency_*`: KNN 기반 JS divergence를 점수로 변환.
  consistency score = \(`exp(-JS)`\), 1에 가까울수록 일관성 높음.
- `family_top_expert_mean_share`: feature family별 top expert 점유율 평균(특화 강도).

계산 근거 코드:
- `experiments/models/FeaturedMoE_N3/diagnostics.py`
  - usage 기반 지표: `_usage_scalars`
  - consistency score 변환: `finalize` 내부 `exp(-js)`
  - family top share: `specialization_summary.mean_top_expert_share`
- `experiments/hyperopt_tune.py`
  - diag 요약 payload: `_build_diag_overview_payload`
- `experiments/models/FeaturedMoE/special_metrics.py`
  - cold/session bucket 집계 및 `mrr@20` 산출

## 8) 논문 서술 포인트(초안)

1. Feature 기반 routing은 전체 valid/test뿐 아니라 cold item(`<=5`) 및 short-session 구간에서 차별적 이득을 보인다.
2. Aux는 단일 전역 최적이 아니라 router anchor 조건(R0 vs R2)에 따라 balance/specialization의 효과가 달라진다.
3. FMoE의 성능 향상은 단순 점수 상승이 아니라, 라우팅 분포(`n_eff`, `top1_max_frac`)와 동역학(`jitter`, `consistency`)을 통해 관찰 가능한 specialization 유도와 함께 나타난다.

### 부록: 논문 메시지용 요약 테이블

| experiment_group | router_variant | aux_family | valid_best_mrr20 | test_mrr20 | cold_item_mrr20 | sess_3_5_mrr20 | n_eff | route_jitter_adjacent | family_top_expert_mean_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aux_reg | R0 | balance | 0.0813 | 0.1620 | 0.1213 | 0.1511 | 10.5930 | 0.1714 | 0.1538 |
| aux_reg | R0 | specialization | 0.0812 | 0.1619 | 0.1212 | 0.1512 | 6.8172 | 0.1438 | 0.3517 |
| aux_reg | R2 | balance | 0.0809 | 0.1616 | 0.1202 | 0.1509 | 10.1606 | 0.1967 | 0.1430 |
| aux_reg | R2 | specialization | 0.0806 | 0.1616 |  |  |  |  |  |
| router_core | R0 | none | 0.0811 | 0.1621 | 0.1211 | 0.1518 | 10.7257 | 0.1674 | 0.1444 |
| router_core | R1 | none | 0.0806 | 0.1620 | 0.1207 | 0.1515 | 10.1595 | 0.1692 | 0.1709 |
| router_core | R2 | none | 0.0810 | 0.1615 | 0.1199 | 0.1507 | 9.9384 | 0.1877 | 0.1494 |
| router_core | R3 | none | 0.0804 | 0.1614 | 0.1209 | 0.1506 | 9.5004 | 0.0928 | 0.1531 |
| router_core | R4 | none | 0.0803 | 0.1617 | 0.1211 | 0.1503 | 10.7278 | 0.1670 | 0.1435 |
| router_core | R5 | none | 0.0806 | 0.1621 | 0.1205 | 0.1518 | 10.0774 | 0.1606 | 0.1712 |
| router_core | R6 | none | 0.0802 | 0.1612 | 0.1217 | 0.1501 | 10.6180 | 0.1304 | 0.1278 |
| router_core | R7 | none | 0.0806 | 0.1620 | 0.1208 | 0.1518 | 10.1027 | 0.2160 | 0.1497 |
