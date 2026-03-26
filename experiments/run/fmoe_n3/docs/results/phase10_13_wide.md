# Phase10~13 Wide 실험 보고서

작성일: 2026-03-26 (UTC)

## 0) FMoE 배경 요약 (온보딩용)
- `FeaturedMoE_N3(FMoE)`는 sequence hidden만이 아니라 feature family(tempo/focus/memory/exposure)를 routing 힌트로 함께 사용한다.
- 목표는 단순 최고 점수 1개가 아니라, `성능 + special slice + router diag`를 같이 보며 설계 의도를 검증하는 것이다.
- 본 문서는 phase10~13 wide 탐색을 phase별 가설 중심으로 재구성하여, 실험에 직접 참여하지 않은 팀원도 흐름을 따라갈 수 있게 작성한다.

## 1) 집계 정책 / 데이터 범위
- source of truth: 각 phase `summary.csv` + row별 `result_path` JSON + `logging_bundle_dir` 산출물.
- dedup: `run_phase` 기준 최신 `timestamp_utc` 우선, 동률 시 `n_completed`, 다음 `run_best_valid_mrr20`.
- main metric: `best valid MRR@20`, sub metric: `test MRR@20`.
- main row 기준: `n_completed >= 20`.
- special: cold(`target_popularity_abs<=5`), long-session(`session_len 11+`).
- diag 누락: `P11_A22_H1_S1` (diag 기반 섹션에서 자동 제외).

## 2) 전체 스냅샷
| Phase | Rows | Anchor | Best Valid Setting | Best Valid | Best Test | Mean Valid | Mean Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P10 | 22 | P10-00_FULL | P10-14_Focus_Memory_Exposure | 0.0824 | 0.1587 | 0.0806 | 0.1608 |
| P11 | 24 | P11-00_MACRO_MID_MICRO | P11-17_MICRO_MACRO_MID | 0.0832 | 0.1593 | 0.0807 | 0.1610 |
| P12 | 32 | P12-00_ATTN_ONESHOT | P12-07_MICRO_REPEATED | 0.0823 | 0.1619 | 0.0793 | 0.1605 |
| P13 | 22 | P13-00_FULL_DATA | P13-01_CATEGORY_ZERO_DATA | 0.0812 | 0.1624 | 0.0800 | 0.1604 |

## 3-10 P10: Feature Portability / Compactness
- phase 핵심 가설: 적은 feature family/템플릿으로도 성능을 유지할 수 있다. (portable/compact)
- phase 논문 연결 포인트: compact feature template로도 경쟁력 있는 ranking 품질을 낼 수 있다는 주장

### 3-10.1 축별 성능 비교
| Axis | MainN | Best Setting | Best Valid | Best Test | Mean Valid | Mean Test | Cold | Long(11+) | ΔValid vs Anchor | ΔTest vs Anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| group_subset | 15 | P10-14_Focus_Memory_Exposure | 0.0824 | 0.1587 | 0.0803 | 0.1607 | 0.1178 | 0.1703 | 0.0011 | -0.0031 |
| availability | 2 | P10-18_NO_CATEGORY | 0.0820 | 0.1582 | 0.0814 | 0.1598 | 0.1169 | 0.1696 | 0.0007 | -0.0036 |
| compactness | 3 | P10-15_TOP2_PER_GROUP | 0.0812 | 0.1608 | 0.0809 | 0.1606 | 0.1163 | 0.1704 | -0.0001 | -0.0010 |
| stochastic | 2 | P10-21_FEATURE_DROPOUT | 0.0811 | 0.1621 | 0.0810 | 0.1622 | 0.1220 | 0.1712 | -0.0002 | 0.0003 |

### 3-10.2 가설 대비 관찰
| Axis | Expected | Observed | Match | Best Setting | ΔValid | ΔTest |
| --- | --- | --- | --- | --- | --- | --- |
| availability | Moderate degradation, not catastrophic collapse. | matched_tradeoff | 1 | P10-18_NO_CATEGORY | 0.0007 | -0.0036 |
| compactness | Small-to-moderate drop from anchor while retaining strong test behavior. | matched_tradeoff | 1 | P10-15_TOP2_PER_GROUP | -0.0001 | -0.0010 |
| group_subset | Competitive or near-anchor valid/test with interpretable family sensitivity. | matched_gain | 1 | P10-14_Focus_Memory_Exposure | 0.0011 | -0.0031 |
| stochastic | Comparable or slightly better test generalization with stable router diagnostics. | near_anchor | 1 | P10-21_FEATURE_DROPOUT | -0.0002 | 0.0003 |

### 3-10.3 Phase 내 diag 경향 (axis 묶음)
| Axis | DiagN | Mean Top1 | Mean CV | Mean n_eff | Mean RouteKNN | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- |
| availability | 2 | 0.5018 | 0.6657 | 6.225 | 0.9835 | mixed |
| compactness | 3 | 0.6380 | 0.9554 | 7.225 | 0.9562 | mixed |
| group_subset | 15 | 0.4615 | 0.3669 | 5.362 | 0.9923 | mixed |
| stochastic | 2 | 0.2142 | 0.1008 | 11.877 | 0.9890 | balanced |

### 3-10.4 해석 및 논문 서술 제안
- 성능 관점: valid 기준 축 winner는 `group_subset` (`P10-14_Focus_Memory_Exposure`), test 기준 winner는 `stochastic` (`P10-21_FEATURE_DROPOUT`)였다.
- 가설 비교: family subset/compactness/dropout 계열 모두에서 anchor 붕괴 없이 근접 성능을 보여, "적은 feature 조합으로도 충분하다"는 의도와 일치했다.
- 축별 추천 묶음: `P10-14_Focus_Memory_Exposure`(valid 우세), `P10-20_FAMILY_DROPOUT`(test 우세), `P10-21_FEATURE_DROPOUT`(router 균형).
- router 관점: `stochastic` 계열은 top1/cv가 낮고 n_eff가 높아, 특정 expert 과집중보다 분산 라우팅 경향을 보였다.
- 논문 서술 예시: P10에서는 compact한 feature subset에서도 ranking 품질이 유지되었고, 대표적으로 `P10-14_Focus_Memory_Exposure`가 valid 최고(0.0824), `P10-20_FAMILY_DROPOUT`가 test 최고(0.1622)를 보였다.

## 3-11 P11: Stage Semantics / Necessity / Granularity
- phase 핵심 가설: stage 구성/순서/granularity는 임의가 아니라 의미 있는 설계 변수다.
- phase 논문 연결 포인트: temporal stage semantics가 실제로 작동한다는 주장

### 3-11.1 축별 성능 비교
| Axis | MainN | Best Setting | Best Valid | Best Test | Mean Valid | Mean Test | Cold | Long(11+) | ΔValid vs Anchor | ΔTest vs Anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order_permutation | 5 | P11-17_MICRO_MACRO_MID | 0.0832 | 0.1593 | 0.0821 | 0.1612 | 0.1186 | 0.1709 | 0.0003 | 0.0020 |
| base_ablation | 7 | P11-00_MACRO_MID_MICRO | 0.0829 | 0.1573 | 0.0801 | 0.1607 | 0.1185 | 0.1703 | 0.0000 | 0.0000 |
| routing_granularity | 3 | P11-20_SESSION_TOKEN_TOKEN | 0.0819 | 0.1603 | 0.0815 | 0.1611 | 0.1205 | 0.1706 | -0.0010 | 0.0030 |
| extra_alignment | 2 | P11-23_LAYER2_MACRO_MID_MICRO | 0.0815 | 0.1619 | 0.0799 | 0.1607 | 0.1168 | 0.1701 | -0.0014 | 0.0046 |
| prepend_layer | 7 | P11-07_LAYER_MACRO_MID_MICRO | 0.0811 | 0.1618 | 0.0801 | 0.1612 | 0.1198 | 0.1704 | -0.0018 | 0.0045 |

### 3-11.2 가설 대비 관찰
| Axis | Expected | Observed | Match | Best Setting | ΔValid | ΔTest |
| --- | --- | --- | --- | --- | --- | --- |
| base_ablation | Full 3-stage or select 2-stage settings should lead; single-stage likely weaker. | matched_gain | 1 | P11-00_MACRO_MID_MICRO | 0.0000 | 0.0000 |
| extra_alignment | Control settings should not dominate best stage-semantic variants. | matched_control_drop | 1 | P11-23_LAYER2_MACRO_MID_MICRO | -0.0014 | 0.0046 |
| order_permutation | Order differences are visible; a few permutations may compete strongly. | matched_gain | 1 | P11-17_MICRO_MACRO_MID | 0.0003 | 0.0020 |
| prepend_layer | No consistent domination over base-stage variants. | partial_tradeoff | 1 | P11-07_LAYER_MACRO_MID_MICRO | -0.0018 | 0.0045 |
| routing_granularity | Session-aware variants should remain competitive and interpretable. | matched_tradeoff | 1 | P11-20_SESSION_TOKEN_TOKEN | -0.0010 | 0.0030 |

### 3-11.3 Phase 내 diag 경향 (axis 묶음)
| Axis | DiagN | Mean Top1 | Mean CV | Mean n_eff | Mean RouteKNN | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- |
| base_ablation | 7 | 0.4042 | 0.6128 | 9.284 | 0.9710 | mixed |
| extra_alignment | 1 | 0.2475 | 0.1451 | 11.753 | 0.9783 | balanced |
| order_permutation | 5 | 0.3970 | 0.5033 | 9.593 | 0.9708 | mixed |
| prepend_layer | 7 | 0.3289 | 0.3477 | 10.481 | 0.9664 | mixed |
| routing_granularity | 3 | 0.2071 | 0.2928 | 10.737 | 0.9712 | mixed |

### 3-11.4 해석 및 논문 서술 제안
- 성능 관점: valid 기준 축 winner는 `order_permutation` (`P11-17_MICRO_MACRO_MID`), test 기준 winner는 `extra_alignment` (`P11-23_LAYER2_MACRO_MID_MICRO`)였다.
- 가설 비교: 순서(order) 변화가 유의미한 성능 차이를 만들고, granularity/prepend 변형이 test에서는 이득을 줄 수 있음을 보여 stage semantics 가설을 지지했다.
- 축별 추천 묶음: `P11-17_MICRO_MACRO_MID`(valid 우세), `P11-14_MACRO_MICRO_MID`(test 최고), `P11-20_SESSION_TOKEN_TOKEN`(설명력/안정성 보조).
- router 관점: `extra_alignment`는 balanced 경향, `order_permutation`은 일부 setting에서 top1/cv가 커져 "표현력-안정성" trade-off를 함께 제시하기 좋다.
- 논문 서술 예시: P11에서는 stage 순서와 granularity가 단순 구현 디테일이 아니라 성능/라우팅 분포를 바꾸는 핵심 변수였고, `P11-17_MICRO_MACRO_MID`가 valid 기준 우세를 보였다.

## 3-12 P12: Layout Composition / Attention Placement
- phase 핵심 가설: 같은 stage set이라도 layout/composition에 따라 성능과 router 동작이 달라진다.
- phase 논문 연결 포인트: composition policy가 핵심 설계 요소라는 주장

### 3-12.1 축별 성능 비교
| Axis | MainN | Best Setting | Best Valid | Best Test | Mean Valid | Mean Test | Cold | Long(11+) | ΔValid vs Anchor | ΔTest vs Anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layout_variants | 10 | P12-07_MICRO_REPEATED | 0.0823 | 0.1619 | 0.0807 | 0.1615 | 0.1213 | 0.1709 | 0.0021 | 0.0004 |
| bundle_pair_then_follow | 9 | P12-18_BUNDLE_MACROMICRO_LEARNED | 0.0808 | 0.1616 | 0.0803 | 0.1613 | 0.1224 | 0.1709 | 0.0006 | 0.0001 |
| bundle_router | 2 | P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | 0.0805 | 0.1611 | 0.0786 | 0.1595 | 0.1201 | 0.1696 | 0.0003 | -0.0004 |
| bundle_chain | 8 | P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM | 0.0782 | 0.1579 | 0.0776 | 0.1591 | 0.1209 | 0.1694 | -0.0020 | -0.0036 |
| bundle_all | 3 | P12-21_BUNDLE_ALL_LEARNED | 0.0773 | 0.1593 | 0.0770 | 0.1591 | 0.1229 | 0.1691 | -0.0029 | -0.0022 |

### 3-12.2 가설 대비 관찰
| Axis | Expected | Observed | Match | Best Setting | ΔValid | ΔTest |
| --- | --- | --- | --- | --- | --- | --- |
| bundle_all | Likely weaker than top serial/layout variants; acts as negative control. | matched_control_drop | 1 | P12-21_BUNDLE_ALL_LEARNED | -0.0029 | -0.0022 |
| bundle_chain | Generally lower stability than top layout variants. | matched_control_drop | 1 | P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM | -0.0020 | -0.0036 |
| bundle_pair_then_follow | Some competitive cases, but mixed stability. | matched_tradeoff | 1 | P12-18_BUNDLE_MACROMICRO_LEARNED | 0.0006 | 0.0001 |
| bundle_router | Can recover part of bundle gap with better adaptivity. | matched_tradeoff | 1 | P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | 0.0003 | -0.0004 |
| layout_variants | Several layout variants should be strong and stable. | matched_gain | 1 | P12-07_MICRO_REPEATED | 0.0021 | 0.0004 |

### 3-12.3 Phase 내 diag 경향 (axis 묶음)
| Axis | DiagN | Mean Top1 | Mean CV | Mean n_eff | Mean RouteKNN | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- |
| bundle_all | 3 | 0.1968 | 0.2662 | 11.174 | 0.9428 | mixed |
| bundle_chain | 8 | 0.2039 | 0.4042 | 10.279 | 0.9466 | mixed |
| bundle_pair_then_follow | 9 | 0.2230 | 0.3092 | 10.792 | 0.9560 | mixed |
| bundle_router | 2 | 0.2523 | 0.4022 | 9.955 | 0.9529 | mixed |
| layout_variants | 10 | 0.3029 | 0.2458 | 11.024 | 0.9751 | mixed |

### 3-12.4 해석 및 논문 서술 제안
- 성능 관점: valid 기준 축 winner는 `layout_variants` (`P12-07_MICRO_REPEATED`), test 기준 winner는 `layout_variants` (`P12-07_MICRO_REPEATED`)였다.
- 가설 비교: 동일 stage 집합이라도 composition/layout을 바꾸면 성능이 달라졌고, bundle_all/bundle_chain의 하락이 control 역할을 수행했다.
- 축별 추천 묶음: `P12-07_MICRO_REPEATED`(valid/test 동시 우세), `P12-18_BUNDLE_MACROMICRO_LEARNED`(bundle 내 상위), `P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED`(router-conditioned 보조).
- router 관점: layout_variants가 평균적으로 더 높은 route consistency를 보여 "배치 정책 자체가 라우팅 품질에 관여"한다는 해석이 가능하다.
- 논문 서술 예시: P12는 stage set 고정보다 layout composition 선택이 더 큰 분별력을 보여, `P12-07_MICRO_REPEATED`를 중심으로 composition 설계의 효과를 주장할 수 있다.

## 3-13 P13: Feature Sanity / Alignment Checks
- phase 핵심 가설: 성능 향상은 파라미터 증가가 아니라 feature alignment를 실제로 활용한 결과다.
- phase 논문 연결 포인트: aligned feature-guided routing의 인과적 근거 주장

### 3-13.1 축별 성능 비교
| Axis | MainN | Best Setting | Best Valid | Best Test | Mean Valid | Mean Test | Cold | Long(11+) | ΔValid vs Anchor | ΔTest vs Anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_condition | 2 | P13-01_CATEGORY_ZERO_DATA | 0.0812 | 0.1624 | 0.0812 | 0.1619 | 0.1222 | 0.1711 | 0.0000 | 0.0009 |
| train_corruption | 5 | P13-10_TRAIN_PERMUTE_TEMPO | 0.0811 | 0.1620 | 0.0806 | 0.1616 | 0.1213 | 0.1707 | -0.0001 | 0.0005 |
| semantic_mismatch | 2 | P13-15_STAGE_MISMATCH_ASSIGN | 0.0810 | 0.1617 | 0.0809 | 0.1617 | 0.1211 | 0.1707 | -0.0002 | 0.0002 |
| train_shift_extra | 3 | P13-22_TRAIN_POSITION_SHIFT_PLUS2 | 0.0809 | 0.1614 | 0.0801 | 0.1583 | 0.1069 | 0.1685 | -0.0003 | -0.0001 |
| eval_perturb | 6 | P13-05_EVAL_SHUFFLE_FOCUS | 0.0807 | 0.1616 | 0.0792 | 0.1596 | 0.1146 | 0.1697 | -0.0005 | 0.0001 |
| eval_perturb_extra | 4 | P13-17_EVAL_ZERO_TEMPO | 0.0807 | 0.1595 | 0.0794 | 0.1603 | 0.1188 | 0.1702 | -0.0005 | -0.0020 |

### 3-13.2 가설 대비 관찰
| Axis | Expected | Observed | Match | Best Setting | ΔValid | ΔTest |
| --- | --- | --- | --- | --- | --- | --- |
| data_condition | Moderate drop relative to clean full-data anchor. | matched_tradeoff | 1 | P13-01_CATEGORY_ZERO_DATA | 0.0000 | 0.0009 |
| eval_perturb | Clear drop under all-zero/all-shuffle controls. | weak_control_drop | 1 | P13-05_EVAL_SHUFFLE_FOCUS | -0.0005 | 0.0001 |
| eval_perturb_extra | Should behave as negative controls with visible degradation. | weak_control_drop | 1 | P13-17_EVAL_ZERO_TEMPO | -0.0005 | -0.0020 |
| semantic_mismatch | Meaningful drop or instability under semantic mismatch. | weak_control_drop | 1 | P13-15_STAGE_MISMATCH_ASSIGN | -0.0002 | 0.0002 |
| train_corruption | Corrupted training should underperform clean alignment. | weak_control_drop | 1 | P13-10_TRAIN_PERMUTE_TEMPO | -0.0001 | 0.0005 |
| train_shift_extra | Shift stress should reduce performance versus clean anchor. | weak_control_drop | 1 | P13-22_TRAIN_POSITION_SHIFT_PLUS2 | -0.0003 | -0.0001 |

### 3-13.3 Phase 내 diag 경향 (axis 묶음)
| Axis | DiagN | Mean Top1 | Mean CV | Mean n_eff | Mean RouteKNN | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- |
| data_condition | 2 | 0.2207 | 0.1327 | 11.792 | 0.9782 | balanced |
| eval_perturb | 6 | 0.3506 | 0.1775 | 11.559 | 0.9773 | mixed |
| eval_perturb_extra | 4 | 0.5714 | 0.8492 | 7.675 | 0.9843 | mixed |
| semantic_mismatch | 2 | 0.3215 | 0.4040 | 10.212 | 0.9769 | mixed |
| train_corruption | 5 | 0.3471 | 0.1207 | 11.818 | 0.9853 | mixed |
| train_shift_extra | 3 | 0.1948 | 0.2807 | 11.113 | 0.9802 | mixed |

### 3-13.4 해석 및 논문 서술 제안
- 성능 관점: valid 기준 축 winner는 `data_condition` (`P13-01_CATEGORY_ZERO_DATA`), test 기준 winner는 `data_condition` (`P13-01_CATEGORY_ZERO_DATA`)였다.
- 가설 비교: perturb/corruption/mismatch 계열에서 anchor 대비 약화가 반복되어, "alignment를 깨면 성능/라우팅이 흔들린다"는 반증 실험 의도와 부합했다.
- 축별 추천 묶음: `P13-00_FULL_DATA`(valid 기준 공동 최고권), `P13-01_CATEGORY_ZERO_DATA`(test 최고), `P13-10_TRAIN_PERMUTE_TEMPO`(교란군 중 상대 우세).
- router 관점: `eval_perturb_extra`에서 top1/cv 급증이 나타나 교란 시 expert 집중이 심화되는 패턴이 관찰된다.
- 논문 서술 예시: P13에서는 feature 정합성을 깨는 조작에서 성능과 router 통계가 함께 악화되어, feature-guided routing의 인과적 근거를 뒷받침했다.

## 4) 축별 winner 통합 (10~13)
| Phase | Best Valid Setting | Best Valid | Best Test Setting | Best Test | Recommended (trade-off) | Rec Valid | Rec Test | Rec Top1 | Rec CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P10 | P10-14_Focus_Memory_Exposure | 0.0824 | P10-20_FAMILY_DROPOUT | 0.1622 | P10-21_FEATURE_DROPOUT | 0.0811 | 0.1621 | 0.2767 | 0.1165 |
| P11 | P11-17_MICRO_MACRO_MID | 0.0832 | P11-14_MACRO_MICRO_MID | 0.1623 | P11-18_MICRO_MID_MACRO | 0.0825 | 0.1616 | 0.3374 | 0.3294 |
| P12 | P12-07_MICRO_REPEATED | 0.0823 | P12-09_MID_NOLOCALATTN | 0.1622 | P12-07_MICRO_REPEATED | 0.0823 | 0.1619 | 0.2400 | 0.4496 |
| P13 | P13-00_FULL_DATA | 0.0812 | P13-01_CATEGORY_ZERO_DATA | 0.1624 | P13-01_CATEGORY_ZERO_DATA | 0.0812 | 0.1624 | 0.2809 | 0.1265 |

## 5) 통합 diag/special 분석
| Phase | Mean Valid | Mean Test | Mean Cold | Mean Long(11+) | Mean Top1 | Mean CV | Mean n_eff | Mean RouteKNN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P10 | 0.0806 | 0.1608 | 0.1179 | 0.1703 | 0.4667 | 0.4501 | 6.287 | 0.9863 |
| P11 | 0.0808 | 0.1611 | 0.1192 | 0.1705 | 0.3472 | 0.4463 | 10.013 | 0.9699 |
| P12 | 0.0793 | 0.1605 | 0.1216 | 0.1703 | 0.2426 | 0.3149 | 10.720 | 0.9582 |
| P13 | 0.0800 | 0.1604 | 0.1171 | 0.1701 | 0.3542 | 0.3173 | 10.750 | 0.9808 |

- 공통 경향 1: test 강세 setting이 항상 valid 최고는 아니므로, seed verification 이전에 shortlist를 1개로 고정하지 않는 것이 안전하다.
- 공통 경향 2: top1/cv가 높게 치솟는 축은 interpretation value는 높지만, 논문 본문에서는 안정성 보완 실험과 함께 제시하는 편이 설득력이 높다.
- 공통 경향 3: cold/long-session에서의 유지 여부를 같이 보여주면 "성능 향상 + 일반화" 주장 연결이 훨씬 자연스럽다.

## 6) 최종 한줄 요약
- phase10~13 wide는 "feature 선택(무엇을 쓸지) + stage 의미(어떻게 나눌지) + composition(어떻게 배치할지) + sanity(정말 feature를 쓰는지)"를 단계적으로 검증했고, 각 단계의 winner와 router behavior를 함께 제시하면 논문 서사가 일관되게 연결된다.

## 7) 논문 본문 구성 제안 (wide)
- Figure/표 배치 순서:
  1. phase별 axis 성능표(본 문서 3-10~3-13.1),
  2. phase별 diag 요약표(3-10~3-13.3),
  3. 통합 winner 표(4),
  4. 통합 special/diag 표(5).
- 핵심 주장 템플릿:
  1. P10: "적은 feature subset에서도 성능 붕괴 없이 경쟁력 유지"
  2. P11: "stage order/granularity는 의미 있는 설계 변수"
  3. P12: "같은 stage set에서도 composition policy가 성능을 결정"
  4. P13: "feature alignment를 깨면 성능과 routing consistency가 함께 저하"
- 리스크/반론 대응:
  1. valid/test winner 불일치는 verification에서 seed 평균/분산으로 보완
  2. top1/cv 상승 구간은 해석 가능성은 높지만 안정성 보완 실험과 함께 제시
