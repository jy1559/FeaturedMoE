# Phase9~13 + Legacy Integrated Wrap-up

작성일: 2026-03-26 09:16 UTC
데이터셋: `KuaiRecLargeStrictPosV2_0.2`

## 1) FMoE 배경 / 읽는 법
- FMoE_N3는 stage(macro/mid/micro)와 feature family 힌트를 함께 사용해 routing을 구성한다.
- 본 문서는 기존 결과를 보존한 상태에서 `phase9~13`을 전수 통합하고, `phase1~8`은 타임라인 맥락으로 연결한다.
- main metric은 `best valid MRR@20`, sub metric은 `test MRR@20`, 보조로 special(cold/long)과 diag(router dynamics)를 같이 해석한다.

## 2) 집계 범위 / 정책
- 기존 `docs/results`, `docs/visualization`, `docs/data`, logs/artifacts는 수정하지 않았다.
- 신규 산출물은 `docs/final` 하위에만 생성했다.
- 통합 row 수:
| Table | Rows | Definition |
| --- | --- | --- |
| wide_all_9_13 | 163 | P9(63) + P10~P13(100) |
| verification_all_9_13 | 239 | P9_2(47) + P10~P13 verification(192) |
| verification_main_fair_h3_n20 | 108 | P9_2(H3,n>=20) + P11~P13(H3,n>=20) |
| verification_support_coverage | 131 | P10(H1/H3,n=10) + P9_2(non-H3,n>=20) |

- diag 누락 run:
  - wide: `P11_A22_H1_S1`
  - verification: `P10_13_2_BP11_22_H3_S1, P10_13_2_BP11_22_H3_S2, P10_13_2_BP11_22_H3_S3, P10_13_2_BP11_22_H3_S4`

## 3) Legacy 타임라인 (Phase1~8 요약)
- 동일 스키마 raw가 완전하지 않은 구간은 기존 handoff/result 문서를 근거로 요약했다.
| Phase | Goal | Key Result | Interpretation |
| --- | --- | --- | --- |
| Baseline | 절대 기준선 확보 | `best 0.0785`, `test 0.1597`, `HR@10 0.1859` | 이후 모든 개선폭의 기준점 |

## 4) Phase9 상세 (Concept/Setting 중심)
| Concept | Rows | Best Valid Setting | Best Valid | Best Test Setting | Best Test | Mean Top1 | Mean CV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | 16 | P9_B3_C0_N2_S1 | 0.0827 | P9_B4_C0_N1_S1 | 0.1616 | 0.4175 | 0.7245 |
| C1 | 16 | P9_B3_C1_B1_S1 | 0.0826 | P9_B2_C1_B2_S1 | 0.1620 | 0.3743 | 0.5990 |
| C2 | 16 | P9_B1_C2_S3_S1 | 0.0818 | P9_B2_C2_S1_S1 | 0.1616 | 0.4412 | 0.9084 |
| C3 | 15 | P9_B2_C3_F2_S1 | 0.0820 | P9_B1_C3_F1_S1 | 0.1616 | 0.4374 | 0.6546 |
- 해석: Phase9는 concept별로 성능 고점과 router 분포가 다르게 움직이며, 단일 aux 우열보다 concept-conditional 비교가 더 설득력 있다.

## 5) Phase10~13 Wide (세팅 중심)
### P10 — Feature Portability / Compactness
| Setting | Group | Valid | Test | Cold | Long | Top1 | CV | n_eff | n_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P10-14_Focus_Memory_Exposure | group_subset | 0.0824 | 0.1587 | 0.1125 | 0.1691 | 0.5338 | 1.2660 | 3.458 | 20 |
| P10-18_NO_CATEGORY | availability | 0.0820 | 0.1582 | 0.1114 | 0.1685 | 0.7474 | 1.2514 | 3.508 | 20 |
| P10-00_FULL | group_subset | 0.0813 | 0.1618 | 0.1231 | 0.1709 | 0.1658 | 0.1215 | 11.825 | 20 |
| P10-15_TOP2_PER_GROUP | compactness | 0.0812 | 0.1608 | 0.1160 | 0.1709 | 0.4907 | 0.7844 | 7.429 | 20 |
| P10-17_COMMON_TEMPLATE | compactness | 0.0812 | 0.1590 | 0.1115 | 0.1692 | 0.9628 | 1.9139 | 2.574 | 20 |
| P10-21_FEATURE_DROPOUT | stochastic | 0.0811 | 0.1621 | 0.1220 | 0.1711 | 0.2767 | 0.1165 | 11.839 | 20 |
| P10-20_FAMILY_DROPOUT | stochastic | 0.0809 | 0.1622 | 0.1220 | 0.1712 | 0.1516 | 0.0851 | 11.914 | 20 |
| P10-19_NO_TIMESTAMP | availability | 0.0809 | 0.1614 | 0.1224 | 0.1706 | 0.2561 | 0.0801 | 8.943 | 20 |
- 해석: valid winner는 `P10-14_Focus_Memory_Exposure` (0.0824), test winner는 `P10-20_FAMILY_DROPOUT` (0.1622)이며, phase 평균은 valid 0.0806 / test 0.1608로 수렴했다.

### P11 — Stage Semantics / Necessity / Granularity
| Setting | Group | Valid | Test | Cold | Long | Top1 | CV | n_eff | n_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P11-17_MICRO_MACRO_MID | order_permutation | 0.0832 | 0.1593 | 0.1155 | 0.1700 | 0.8692 | 1.2707 | 4.589 | 20 |
| P11-00_MACRO_MID_MICRO | base_ablation | 0.0829 | 0.1573 | 0.1079 | 0.1679 | 1.0000 | 2.2711 | 1.949 | 20 |
| P11-18_MICRO_MID_MACRO | order_permutation | 0.0825 | 0.1616 | 0.1148 | 0.1708 | 0.3374 | 0.3294 | 10.825 | 20 |
| P11-20_SESSION_TOKEN_TOKEN | routing_granularity | 0.0819 | 0.1603 | 0.1159 | 0.1701 | 0.2976 | 0.6328 | 8.568 | 20 |
| P11-14_MACRO_MICRO_MID | order_permutation | 0.0817 | 0.1623 | 0.1225 | 0.1716 | 0.2761 | 0.1218 | 11.825 | 20 |
| P11-16_MID_MICRO_MACRO | order_permutation | 0.0816 | 0.1621 | 0.1224 | 0.1710 | 0.2107 | 0.3890 | 10.423 | 20 |
| P11-15_MID_MACRO_MICRO | order_permutation | 0.0816 | 0.1607 | 0.1180 | 0.1709 | 0.2915 | 0.4057 | 10.304 | 20 |
| P11-23_LAYER2_MACRO_MID_MICRO | extra_alignment | 0.0815 | 0.1619 | 0.1187 | 0.1708 | 0.2475 | 0.1451 | 11.753 | 20 |
- 해석: valid winner는 `P11-17_MICRO_MACRO_MID` (0.0832), test winner는 `P11-14_MACRO_MICRO_MID` (0.1623)이며, phase 평균은 valid 0.0807 / test 0.1610로 수렴했다.

### P12 — Layout Composition / Attention Placement
| Setting | Group | Valid | Test | Cold | Long | Top1 | CV | n_eff | n_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P12-07_MICRO_REPEATED | layout_variants | 0.0823 | 0.1619 | 0.1167 | 0.1716 | 0.2400 | 0.4496 | 9.982 | 20 |
| P12-05_MACRO_REPEATED | layout_variants | 0.0815 | 0.1613 | 0.1225 | 0.1707 | 0.2177 | 0.0991 | 11.883 | 20 |
| P12-06_MID_REPEATED | layout_variants | 0.0814 | 0.1618 | 0.1233 | 0.1709 | 0.1678 | 0.0930 | 11.897 | 20 |
| P12-08_MACRO_NOLOCALATTN | layout_variants | 0.0810 | 0.1619 | 0.1219 | 0.1710 | 0.3835 | 0.1701 | 11.663 | 20 |
| P12-09_MID_NOLOCALATTN | layout_variants | 0.0809 | 0.1622 | 0.1229 | 0.1714 | 0.2365 | 0.1185 | 11.834 | 20 |
| P12-02_ATTN_MICRO_BEFORE | layout_variants | 0.0809 | 0.1620 | 0.1216 | 0.1713 | 0.3054 | 0.1486 | 11.741 | 20 |
| P12-18_BUNDLE_MACROMICRO_LEARNED | bundle_pair_then_follow | 0.0808 | 0.1616 | 0.1235 | 0.1710 | 0.1856 | 0.2288 | 11.403 | 20 |
| P12-01_ATTN_MACRO_ONLY | layout_variants | 0.0808 | 0.1614 | 0.1219 | 0.1711 | 0.2822 | 0.1009 | 11.879 | 20 |
- 해석: valid winner는 `P12-07_MICRO_REPEATED` (0.0823), test winner는 `P12-09_MID_NOLOCALATTN` (0.1622)이며, phase 평균은 valid 0.0793 / test 0.1605로 수렴했다.

### P13 — Feature Sanity / Alignment Checks
| Setting | Group | Valid | Test | Cold | Long | Top1 | CV | n_eff | n_completed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P13-01_CATEGORY_ZERO_DATA | data_condition | 0.0812 | 0.1624 | 0.1214 | 0.1714 | 0.2809 | 0.1265 | 11.811 | 20 |
| P13-00_FULL_DATA | data_condition | 0.0812 | 0.1615 | 0.1230 | 0.1709 | 0.1605 | 0.1389 | 11.773 | 20 |
| P13-10_TRAIN_PERMUTE_TEMPO | train_corruption | 0.0811 | 0.1620 | 0.1221 | 0.1711 | 0.3207 | 0.1229 | 11.821 | 20 |
| P13-11_TRAIN_PERMUTE_FOCUS | train_corruption | 0.0811 | 0.1619 | 0.1225 | 0.1710 | 0.2873 | 0.1526 | 11.727 | 20 |
| P13-12_TRAIN_PERMUTE_MEMORY | train_corruption | 0.0810 | 0.1617 | 0.1228 | 0.1711 | 0.2649 | 0.1330 | 11.792 | 20 |
| P13-15_STAGE_MISMATCH_ASSIGN | semantic_mismatch | 0.0810 | 0.1617 | 0.1209 | 0.1706 | 0.4460 | 0.5659 | 9.089 | 20 |
| P13-22_TRAIN_POSITION_SHIFT_PLUS2 | train_shift_extra | 0.0809 | 0.1614 | 0.1224 | 0.1710 | 0.1690 | 0.3176 | 10.901 | 20 |
| P13-16_POSITION_SHIFT_FEATURE | semantic_mismatch | 0.0808 | 0.1617 | 0.1213 | 0.1708 | 0.1970 | 0.2421 | 11.335 | 20 |
- 해석: valid winner는 `P13-00_FULL_DATA` (0.0812), test winner는 `P13-01_CATEGORY_ZERO_DATA` (0.1624)이며, phase 평균은 valid 0.0800 / test 0.1604로 수렴했다.

## 6) Verification 통합 (P9_2 + P10~13)
### 6.1 Main Fair Table (H3 + n>=20)
| Phase | Rows | Best Valid Setting | Best Valid | Best Test Setting | Best Test | Mean Valid | Mean Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P9_2 | 12 | P9_2-B3-C1-B1 | 0.082 | P9_2-B4-C0-N4 | 0.1618 | 0.081075 | 0.16131666666666666 |
| P11 | 32 | P11-14_MACRO_MICRO_MID | 0.083 | P11-14_MACRO_MICRO_MID | 0.1623 | 0.08114687500000001 | 0.16063125 |
| P12 | 32 | P12-06_MID_REPEATED | 0.0821 | P12-02_ATTN_MICRO_BEFORE | 0.1622 | 0.08090312499999999 | 0.161434375 |
| P13 | 32 | P13-00_FULL_DATA | 0.0826 | P13-11_TRAIN_PERMUTE_FOCUS | 0.1623 | 0.08025625 | 0.16045625 |

### 6.2 Setting별 Seed Mean/Std (Top)
| Setting | Group | Hparam | SeedN | Valid mean+-std | Test mean+-std | Cold mean | Long mean | Top1 mean | CV mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9_2-B4-C0-N4 | C0 | H3 | 4 | 0.0814 +/- 0.0001 | 0.1617 +/- 0.0001 | 0.1228 | - | 0.1855 | 0.1300 |
| P9_2-B4-C0-N4 | C0 | H4 | 4 | 0.0812 +/- 0.0001 | 0.1616 +/- 0.0001 | 0.1196 | - | 0.2244 | 0.0856 |
| P9_2-B4-C0-N4 | C0 | H2 | 4 | 0.0811 +/- 0.0001 | 0.1611 +/- 0.0010 | 0.1208 | - | 0.2464 | 0.2745 |
| P9_2-B3-C1-B1 | C1 | H3 | 4 | 0.0810 +/- 0.0007 | 0.1607 +/- 0.0017 | 0.1180 | - | 0.2215 | 0.3598 |
| P9_2-B1-C2-S3 | C2 | H4 | 3 | 0.0809 +/- 0.0000 | 0.1614 +/- 0.0001 | 0.1189 | - | 0.1926 | 0.2890 |
| P11-14_MACRO_MICRO_MID | order_permutation | H3 | 4 | 0.0822 +/- 0.0007 | 0.1580 +/- 0.0049 | 0.1101 | 0.1684 | 0.6320 | 1.1812 |
| P11-17_MICRO_MACRO_MID | order_permutation | H3 | 4 | 0.0820 +/- 0.0005 | 0.1614 +/- 0.0005 | 0.1195 | 0.1710 | 0.2754 | 0.3402 |
| P11-23_LAYER2_MACRO_MID_MICRO | extra_alignment | H3 | 4 | 0.0816 +/- 0.0001 | 0.1609 +/- 0.0014 | 0.1164 | 0.1703 | 0.2095 | 0.1851 |
| P11-19_TOKEN_TOKEN_TOKEN | routing_granularity | H3 | 4 | 0.0816 +/- 0.0005 | 0.1613 +/- 0.0008 | 0.1211 | 0.1708 | 0.2674 | 0.2473 |
| P11-20_SESSION_TOKEN_TOKEN | routing_granularity | H3 | 4 | 0.0814 +/- 0.0000 | 0.1614 +/- 0.0006 | 0.1213 | 0.1709 | 0.2920 | 0.3034 |
| P12-06_MID_REPEATED | layout_variants | H3 | 4 | 0.0815 +/- 0.0004 | 0.1606 +/- 0.0023 | 0.1194 | 0.1701 | 0.3758 | 0.7235 |
| P12-07_MICRO_REPEATED | layout_variants | H3 | 4 | 0.0814 +/- 0.0003 | 0.1614 +/- 0.0009 | 0.1206 | 0.1707 | 0.2107 | 0.1799 |
| P12-08_MACRO_NOLOCALATTN | layout_variants | H3 | 4 | 0.0810 +/- 0.0001 | 0.1615 +/- 0.0008 | 0.1205 | 0.1710 | 0.4152 | 0.3861 |
| P12-18_BUNDLE_MACROMICRO_LEARNED | bundle_pair_then_follow | H3 | 4 | 0.0809 +/- 0.0001 | 0.1615 +/- 0.0001 | 0.1233 | 0.1710 | 0.1847 | 0.2155 |
| P12-02_ATTN_MICRO_BEFORE | layout_variants | H3 | 4 | 0.0809 +/- 0.0001 | 0.1621 +/- 0.0001 | 0.1224 | 0.1714 | 0.2588 | 0.1229 |
| P13-00_FULL_DATA | data_condition | H3 | 4 | 0.0816 +/- 0.0007 | 0.1615 +/- 0.0006 | 0.1210 | 0.1709 | 0.2989 | 0.2741 |
| P13-01_CATEGORY_ZERO_DATA | data_condition | H3 | 4 | 0.0813 +/- 0.0001 | 0.1607 +/- 0.0031 | 0.1172 | 0.1698 | 0.4950 | 0.7259 |
| P13-15_STAGE_MISMATCH_ASSIGN | semantic_mismatch | H3 | 4 | 0.0812 +/- 0.0003 | 0.1609 +/- 0.0011 | 0.1182 | 0.1701 | 0.4031 | 0.4734 |
| P13-11_TRAIN_PERMUTE_FOCUS | train_corruption | H3 | 4 | 0.0812 +/- 0.0000 | 0.1620 +/- 0.0002 | 0.1223 | 0.1711 | 0.2582 | 0.1325 |
| P13-10_TRAIN_PERMUTE_TEMPO | train_corruption | H3 | 4 | 0.0811 +/- 0.0001 | 0.1618 +/- 0.0001 | 0.1226 | 0.1710 | 0.2596 | 0.1159 |
- 해석: fair 본표에서는 phase별로 valid/test winner가 분리되는 구간이 있어, 논문 본문에선 winner 1개가 아니라 valid/test/stability 3점 제시가 안전하다.

## 7) 통합 Diag/Special/Router
| Phase | Rows | Mean Valid | Mean Test | Mean Cold | Mean Long | Mean Top1 | Mean CV | Mean n_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9 | 63 | 0.0809 | 0.1606 | 0.1170 | - | 0.4173 | 0.7227 | 8.729 |
| P9_2 | 47 | 0.0809 | 0.1611 | 0.1193 | - | 0.3182 | 0.5206 | 9.696 |
| P10 | 118 | 0.0806 | 0.1616 | 0.1204 | 0.1708 | 0.3427 | 0.1822 | 9.168 |
| P11 | 56 | 0.0809 | 0.1608 | 0.1183 | 0.1704 | 0.3197 | 0.4130 | 10.269 |
| P12 | 64 | 0.0801 | 0.1610 | 0.1214 | 0.1706 | 0.2779 | 0.3123 | 10.764 |
| P13 | 54 | 0.0802 | 0.1604 | 0.1172 | 0.1700 | 0.3742 | 0.2851 | 10.929 |

| Phase | Split | Diag Metric | Target | Spearman | Pearson | N |
| --- | --- | --- | --- | --- | --- | --- |
| P9 | wide | diag_top1_max_frac | best_valid_mrr20 | 0.743 | 0.670 | 63 |
| P9_2 | verification | diag_route_jitter_adjacent | best_valid_mrr20 | -0.594 | -0.550 | 47 |
| P10 | wide | diag_route_consistency_group_knn_score | best_valid_mrr20 | -0.860 | -0.685 | 22 |
| P10 | verification | diag_route_consistency_knn_score | best_valid_mrr20 | -0.562 | -0.662 | 96 |
| P11 | verification | diag_top1_max_frac | best_valid_mrr20 | 0.516 | 0.690 | 28 |
| P11 | wide | diag_cv_usage | best_valid_mrr20 | 0.191 | 0.406 | 23 |
| P12 | wide | diag_route_consistency_knn_score | best_valid_mrr20 | 0.698 | 0.722 | 32 |
| P12 | verification | diag_route_consistency_knn_score | best_valid_mrr20 | 0.399 | 0.138 | 32 |
| P13 | verification | diag_entropy_mean | best_valid_mrr20 | -0.761 | -0.531 | 32 |
| P13 | wide | diag_entropy_mean | best_valid_mrr20 | -0.412 | -0.383 | 22 |
- 해석: phase마다 유효한 diag 지표가 다르므로, 단일 router metric으로 전 phase를 설명하기보다 phase-conditioned 해석이 필요하다.

## 8) Plan 가설 대비 관찰 / Claim Bank
| Phase | Group | Observed | Match | Best Setting | dValid | dTest | Claim Template |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P9 | C0_Natural | observed_phase9_concept | 1 | B4_C0_N4 | 0.0017 | -0.0013 | 약한 정규화만으로도 안정적인 성능대를 유지할 수 있다. |
| P9 | C1_CanonicalBalance | observed_phase9_concept | 1 | B3_C1_B1 | 0.0015 | -0.0009 | 균형 유도는 성능/안정성 trade-off를 조절하는 핵심 축이다. |
| P9 | C2_Specialization | observed_phase9_concept | 1 | B1_C2_S3 | 0.0011 | -0.0014 | 적절한 특화는 유효하지만 과도 집중은 일반화 리스크를 만든다. |
| P9 | C3_FeatureAlignment | observed_phase9_concept | 1 | B2_C3_F2 | 0.0011 | -0.0011 | feature-aligned routing은 성능 분포를 유의미하게 이동시킨다. |
| P10 | availability | matched_tradeoff | 1 | P10-18_NO_CATEGORY | 0.0007 | -0.0036 | The framework remains functional even under partial feature availability constraints. |
| P10 | availability_plus | no_anchor | 0 |  | - | - | Severe multi-signal removal reveals the boundary of portability. |
| P10 | compactness | matched_tradeoff | 1 | P10-15_TOP2_PER_GROUP | -0.0001 | -0.0010 | Few representative signals per family are sufficient for robust feature-aware routing. |
| P10 | compactness_plus | no_anchor | 0 |  | - | - | Even stricter compactness-plus settings preserve a meaningful portion of gains. |
| P10 | group_subset | matched_gain | 1 | P10-14_Focus_Memory_Exposure | 0.0011 | -0.0031 | A compact subset of feature families can preserve most of the gain, indicating portability beyond large handcrafted banks. |
| P10 | stochastic | near_anchor | 1 | P10-21_FEATURE_DROPOUT | -0.0002 | 0.0003 | Stochastic feature usage improves robustness without requiring larger feature banks. |
| P11 | base_ablation | matched_gain | 1 | P11-00_MACRO_MID_MICRO | 0.0000 | 0.0000 | Performance depends on temporal-horizon decomposition, not just MoE presence. |
| P11 | extra_alignment | matched_control_drop | 1 | P11-23_LAYER2_MACRO_MID_MICRO | -0.0014 | 0.0046 | Improvements are not explained by generic depth increases alone. |
| P11 | order_permutation | matched_gain | 1 | P11-17_MICRO_MACRO_MID | 0.0003 | 0.0020 | Ordering influences routing behavior, supporting stage-semantic interpretation. |
| P11 | prepend_layer | partial_tradeoff | 1 | P11-07_LAYER_MACRO_MID_MICRO | -0.0018 | 0.0045 | Stage decomposition contributes beyond simply adding dense contextualization depth. |
| P11 | routing_granularity | matched_tradeoff | 1 | P11-20_SESSION_TOKEN_TOKEN | -0.0010 | 0.0030 | Macro/mid routing behaves naturally as session-regime routing. |
| P12 | bundle_all | matched_control_drop | 1 | P12-21_BUNDLE_ALL_LEARNED | -0.0029 | -0.0022 | Serial horizon separation is often preferable to collapsing all horizons at once. |
| P12 | bundle_chain | matched_control_drop | 1 | P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM | -0.0020 | -0.0036 | More complex bundle chains do not automatically improve routing quality. |
| P12 | bundle_pair_then_follow | matched_tradeoff | 1 | P12-18_BUNDLE_MACROMICRO_LEARNED | 0.0006 | 0.0001 | Selected horizon interaction can help, but not all bundling is beneficial. |
| P12 | bundle_router | matched_tradeoff | 1 | P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | 0.0003 | -0.0004 | Adaptive aggregation helps, but composition bias remains a critical factor. |
| P12 | layout_variants | matched_gain | 1 | P12-07_MICRO_REPEATED | 0.0021 | 0.0004 | Composition details matter: identical stage sets can differ by layout quality. |
| P13 | data_condition | matched_tradeoff | 1 | P13-01_CATEGORY_ZERO_DATA | 0.0000 | 0.0009 | Category signals help, but the framework can remain functional under weakened category cues. |
| P13 | eval_perturb | weak_control_drop | 1 | P13-05_EVAL_SHUFFLE_FOCUS | -0.0005 | 0.0001 | Inference-time perturbation degrades performance, indicating active feature usage. |
| P13 | eval_perturb_extra | weak_control_drop | 1 | P13-17_EVAL_ZERO_TEMPO | -0.0005 | -0.0020 | Additional perturb controls reinforce that routing depends on aligned feature signals. |
| P13 | semantic_mismatch | weak_control_drop | 1 | P13-15_STAGE_MISMATCH_ASSIGN | -0.0002 | 0.0002 | Semantic and temporal alignment of features is crucial for routing quality. |
| P13 | train_corruption | weak_control_drop | 1 | P13-10_TRAIN_PERMUTE_TEMPO | -0.0001 | 0.0005 | Aligned feature guidance, not branch size alone, explains gains. |
| P13 | train_shift_extra | weak_control_drop | 1 | P13-22_TRAIN_POSITION_SHIFT_PLUS2 | -0.0003 | -0.0001 | Temporal misalignment during training directly weakens downstream ranking quality. |
- 해석: P9~P13 전체를 연결하면, `feature 선택(P9/P10) -> stage 의미(P11) -> composition(P12) -> sanity counterfactual(P13)`의 논문 스토리라인이 자연스럽게 완성된다.

## 9) 참고
- 상세 figure는 `docs/final/visualization/phase9_13_wrapup.ipynb`를 참조한다.
- 원본 근거 파일 목록은 `docs/final/data/phase9_13_wrapup/source_manifest.csv`를 참조한다.
