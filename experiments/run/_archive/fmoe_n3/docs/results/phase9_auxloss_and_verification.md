# Phase9 결과 보고서: 축별(Concept) 분석 중심

## 1. 왜 이 phase를 했는가
- 목적: aux를 많이 넣는 탐색이 아니라, 개념 축(concept)을 바꿨을 때 어떤 성능/진단 패턴이 나오는지 설명하기 위함.
- 본 문서는 완료도가 높은 Phase9 본실험 중심으로 작성하고, Phase9_2는 진행 요약만 짧게 포함.

## 2. Concept 축 정의
| concept_id | concept_name | 의도 |
| --- | --- | --- |
| C0 | 자연형(Natural) | aux 최소/약한 안정화로 baseline 대비 변화 확인 |
| C1 | 균형형(CanonicalBalance) | expert/group usage 균형 유도 |
| C2 | 특화형(Specialization) | route 집중(특화)과 안정성 trade-off 확인 |
| C3 | 정렬형(FeatureAlignment) | feature prior와 routing 정렬 효과 검증 |

## 3.C0 축 상세: 자연형(Natural)
실험 이유
- aux 최소/약한 안정화로 baseline 대비 변화 확인

run 단위 결과(축 내부 직접 비교)
| run_phase | base_id | combo_id | main_aux | support_aux | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9_B4_C0_N4_S1 | B4 | N4 | z | none | 0.0827 | 0.1591 | 0.1153 | 0.1456 | 2.3613 | 2.0204 | 0.9931 |
| P9_B3_C0_N2_S1 | B3 | N2 | route_smoothness | none | 0.0827 | 0.1581 | 0.1087 | 0.1458 | 1.2108 | 2.9851 | 1.0000 |
| P9_B4_C0_N3_S1 | B4 | N3 | balance | z | 0.0824 | 0.1585 | 0.1145 | 0.1465 | 4.3634 | 1.3229 | 0.8775 |
| P9_B2_C0_N4_S1 | B2 | N4 | z | none | 0.0816 | 0.1603 | 0.1134 | 0.1484 | 8.1452 | 0.6879 | 0.5910 |
| P9_B4_C0_N2_S1 | B4 | N2 | route_smoothness | none | 0.0810 | 0.1603 | 0.1171 | 0.1487 | 9.1358 | 0.5599 | 0.3028 |
| P9_B1_C0_N4_S1 | B1 | N4 | z | none | 0.0809 | 0.1605 | 0.1171 | 0.1489 | 9.5189 | 0.5105 | 0.3268 |
| P9_B3_C0_N4_S1 | B3 | N4 | z | none | 0.0809 | 0.1597 | 0.1140 | 0.1481 | 7.6780 | 0.7503 | 0.3055 |
| P9_B1_C0_N2_S1 | B1 | N2 | route_smoothness | none | 0.0807 | 0.1612 | 0.1192 | 0.1504 | 11.7179 | 0.1552 | 0.2213 |
| P9_B4_C0_N1_S1 | B4 | N1 | none | none | 0.0807 | 0.1616 | 0.1203 | 0.1510 | 11.7037 | 0.1591 | 0.4077 |
| P9_B1_C0_N1_S1 | B1 | N1 | none | none | 0.0806 | 0.1611 | 0.1191 | 0.1500 | 11.7626 | 0.1421 | 0.1630 |
| P9_B2_C0_N3_S1 | B2 | N3 | balance | z | 0.0806 | 0.1607 | 0.1172 | 0.1476 | 8.3839 | 0.6567 | 0.2333 |
| P9_B1_C0_N3_S1 | B1 | N3 | balance | z | 0.0806 | 0.1612 | 0.1190 | 0.1502 | 11.8159 | 0.1248 | 0.1904 |
| P9_B2_C0_N1_S1 | B2 | N1 | none | none | 0.0805 | 0.1605 | 0.1172 | 0.1482 | 6.3487 | 0.9435 | 0.3644 |
| P9_B2_C0_N2_S1 | B2 | N2 | route_smoothness | none | 0.0802 | 0.1614 | 0.1191 | 0.1505 | 11.7657 | 0.1411 | 0.2425 |
| P9_B3_C0_N3_S1 | B3 | N3 | balance | z | 0.0802 | 0.1602 | 0.1171 | 0.1490 | 11.0596 | 0.2916 | 0.2500 |
| P9_B3_C0_N1_S1 | B3 | N1 | none | none | 0.0801 | 0.1614 | 0.1190 | 0.1501 | 11.7649 | 0.1414 | 0.2102 |

축 해석
- best(valid): `P9_B4_C0_N4_S1` = 0.0827
- valid range: 0.0026, test range: 0.0035

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_top1_max_frac | 16 | 0.9572 | 0.7352 |
| best_valid_mrr20 | diag_family_top_expert_mean_share | 16 | 0.8732 | 0.7293 |
| test_mrr20 | cold_item_mrr20 | 16 | 0.8896 | 0.9100 |
| test_mrr20 | diag_entropy_mean | 16 | 0.8875 | 0.8923 |

## 3.C1 축 상세: 균형형(CanonicalBalance)
실험 이유
- expert/group usage 균형 유도

run 단위 결과(축 내부 직접 비교)
| run_phase | base_id | combo_id | main_aux | support_aux | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9_B3_C1_B1_S1 | B3 | B1 | balance | z | 0.0826 | 0.1595 | 0.1142 | 0.1478 | 6.2645 | 0.9568 | 0.4408 |
| P9_B2_C1_B3_S1 | B2 | B3 | factored_group_balance | z | 0.0824 | 0.1587 | 0.1089 | 0.1472 | 5.4800 | 1.0908 | 0.4764 |
| P9_B1_C1_B4_S1 | B1 | B4 | primitive_balance | z | 0.0821 | 0.1594 | 0.1142 | 0.1464 | 5.0734 | 1.1685 | 0.5508 |
| P9_B3_C1_B4_S1 | B3 | B4 | primitive_balance | z | 0.0817 | 0.1601 | 0.1136 | 0.1484 | 9.8057 | 0.4731 | 0.2252 |
| P9_B4_C1_B1_S1 | B4 | B1 | balance | z | 0.0814 | 0.1557 | 0.1021 | 0.1427 | 2.3213 | 2.0419 | 0.9999 |
| P9_B4_C1_B2_S1 | B4 | B2 | balance_strong | z | 0.0811 | 0.1614 | 0.1165 | 0.1490 | 9.5230 | 0.5100 | 0.3160 |
| P9_B1_C1_B1_S1 | B1 | B1 | balance | z | 0.0811 | 0.1616 | 0.1162 | 0.1497 | 8.7671 | 0.6073 | 0.3102 |
| P9_B1_C1_B3_S1 | B1 | B3 | factored_group_balance | z | 0.0809 | 0.1615 | 0.1171 | 0.1494 | 9.5694 | 0.5040 | 0.3014 |
| P9_B4_C1_B3_S1 | B4 | B3 | factored_group_balance | z | 0.0809 | 0.1614 | 0.1203 | 0.1496 | 11.5159 | 0.2050 | 0.4412 |
| P9_B2_C1_B4_S1 | B2 | B4 | primitive_balance | z | 0.0809 | 0.1607 | 0.1165 | 0.1481 | 8.9522 | 0.5835 | 0.3423 |
| P9_B4_C1_B4_S1 | B4 | B4 | primitive_balance | z | 0.0808 | 0.1613 | 0.1202 | 0.1496 | 11.4688 | 0.2152 | 0.4401 |
| P9_B1_C1_B2_S1 | B1 | B2 | balance_strong | z | 0.0807 | 0.1613 | 0.1190 | 0.1497 | 11.7776 | 0.1374 | 0.2543 |
| P9_B3_C1_B3_S1 | B3 | B3 | factored_group_balance | z | 0.0803 | 0.1601 | 0.1181 | 0.1482 | 9.7143 | 0.4851 | 0.2756 |
| P9_B3_C1_B2_S1 | B3 | B2 | balance_strong | z | 0.0803 | 0.1603 | 0.1174 | 0.1469 | 10.8660 | 0.3231 | 0.1949 |
| P9_B2_C1_B2_S1 | B2 | B2 | balance_strong | z | 0.0802 | 0.1620 | 0.1205 | 0.1514 | 11.7595 | 0.1430 | 0.2425 |
| P9_B2_C1_B1_S1 | B2 | B1 | balance | z | 0.0801 | 0.1615 | 0.1200 | 0.1506 | 11.7696 | 0.1399 | 0.1775 |

축 해석
- best(valid): `P9_B3_C1_B1_S1` = 0.0826
- valid range: 0.0025, test range: 0.0063

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | cold_item_mrr20 | 16 | -0.6107 | -0.8316 |
| best_valid_mrr20 | diag_family_top_expert_mean_share | 16 | 0.5258 | 0.8301 |
| test_mrr20 | sess_3_5_mrr20 | 16 | 0.9462 | 0.8953 |
| test_mrr20 | cold_item_mrr20 | 16 | 0.9298 | 0.7035 |

## 3.C2 축 상세: 특화형(Specialization)
실험 이유
- route 집중(특화)과 안정성 trade-off 확인

run 단위 결과(축 내부 직접 비교)
| run_phase | base_id | combo_id | main_aux | support_aux | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9_B1_C2_S3_S1 | B1 | S3 | route_sharpness | route_monopoly | 0.0818 | 0.1596 | 0.1160 | 0.1463 | 1.0711 | 3.1942 | 0.9842 |
| P9_B1_C2_S4_S1 | B1 | S4 | route_smoothness | route_sharpness | 0.0812 | 0.1602 | 0.1173 | 0.1491 | 1.1199 | 3.1169 | 0.9831 |
| P9_B4_C2_S1_S1 | B4 | S1 | route_sharpness | none | 0.0810 | 0.1616 | 0.1168 | 0.1498 | 7.0684 | 0.8353 | 0.6313 |
| P9_B4_C2_S4_S1 | B4 | S4 | route_smoothness | route_sharpness | 0.0809 | 0.1607 | 0.1173 | 0.1484 | 9.3584 | 0.5313 | 0.3920 |
| P9_B4_C2_S2_S1 | B4 | S2 | route_sharpness_strong | none | 0.0808 | 0.1614 | 0.1200 | 0.1498 | 11.1948 | 0.2682 | 0.4442 |
| P9_B4_C2_S3_S1 | B4 | S3 | route_sharpness | route_monopoly | 0.0807 | 0.1614 | 0.1193 | 0.1496 | 11.2125 | 0.2650 | 0.4280 |
| P9_B1_C2_S1_S1 | B1 | S1 | route_sharpness | none | 0.0806 | 0.1610 | 0.1188 | 0.1500 | 11.2865 | 0.2514 | 0.2536 |
| P9_B2_C2_S3_S1 | B2 | S3 | route_sharpness | route_monopoly | 0.0806 | 0.1600 | 0.1162 | 0.1480 | 1.2416 | 2.9436 | 0.9796 |
| P9_B2_C2_S2_S1 | B2 | S2 | route_sharpness_strong | none | 0.0805 | 0.1611 | 0.1192 | 0.1498 | 9.6516 | 0.4933 | 0.1871 |
| P9_B3_C2_S4_S1 | B3 | S4 | route_smoothness | route_sharpness | 0.0804 | 0.1616 | 0.1189 | 0.1507 | 11.7710 | 0.1395 | 0.3603 |
| P9_B1_C2_S2_S1 | B1 | S2 | route_sharpness_strong | none | 0.0804 | 0.1611 | 0.1205 | 0.1495 | 6.7066 | 0.8884 | 0.3608 |
| P9_B2_C2_S4_S1 | B2 | S4 | route_smoothness | route_sharpness | 0.0804 | 0.1615 | 0.1199 | 0.1506 | 11.7648 | 0.1414 | 0.3251 |
| P9_B3_C2_S1_S1 | B3 | S1 | route_sharpness | none | 0.0803 | 0.1616 | 0.1201 | 0.1512 | 11.4662 | 0.2158 | 0.1843 |
| P9_B3_C2_S3_S1 | B3 | S3 | route_sharpness | route_monopoly | 0.0803 | 0.1611 | 0.1198 | 0.1498 | 9.5465 | 0.5070 | 0.1807 |
| P9_B3_C2_S2_S1 | B3 | S2 | route_sharpness_strong | none | 0.0803 | 0.1612 | 0.1197 | 0.1498 | 9.4428 | 0.5204 | 0.1859 |
| P9_B2_C2_S1_S1 | B2 | S1 | route_sharpness | none | 0.0802 | 0.1616 | 0.1199 | 0.1510 | 11.4322 | 0.2229 | 0.1786 |

축 해석
- best(valid): `P9_B1_C2_S3_S1` = 0.0818
- valid range: 0.0016, test range: 0.0020

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_top1_max_frac | 16 | 0.7850 | 0.9001 |
| best_valid_mrr20 | diag_route_jitter_adjacent | 16 | -0.6507 | -0.7711 |
| test_mrr20 | sess_3_5_mrr20 | 16 | 0.9044 | 0.7837 |
| test_mrr20 | diag_n_eff | 16 | 0.8756 | 0.7361 |

## 3.C3 축 상세: 정렬형(FeatureAlignment)
실험 이유
- feature prior와 routing 정렬 효과 검증

run 단위 결과(축 내부 직접 비교)
| run_phase | base_id | combo_id | main_aux | support_aux | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P9_B2_C3_F2_S1 | B2 | F2 | route_prior_strong | z | 0.0820 | 0.1593 | 0.1145 | 0.1473 | 4.4720 | 1.2975 | 0.7829 |
| P9_B2_C3_F4_S1 | B2 | F4 | wrapper_group_feature_align | group_prior_align | 0.0817 | 0.1601 | 0.1167 | 0.1474 | 8.7025 | 0.6156 | 0.4288 |
| P9_B4_C3_F1_S1 | B4 | F1 | route_prior | none | 0.0815 | 0.1563 | 0.1007 | 0.1436 | 1.9108 | 2.2979 | 1.0000 |
| P9_B3_C3_F2_S1 | B3 | F2 | route_prior_strong | z | 0.0814 | 0.1593 | 0.1167 | 0.1458 | 6.2928 | 0.9523 | 0.4816 |
| P9_B1_C3_F1_S1 | B1 | F1 | route_prior | none | 0.0811 | 0.1616 | 0.1174 | 0.1492 | 6.1344 | 0.9778 | 0.5831 |
| P9_B3_C3_F3_S1 | B3 | F3 | group_prior_align | none | 0.0810 | 0.1608 | 0.1149 | 0.1488 | 9.1288 | 0.5608 | 0.4040 |
| P9_B2_C3_F1_S1 | B2 | F1 | route_prior | none | 0.0809 | 0.1598 | 0.1180 | 0.1479 | 6.5366 | 0.9142 | 0.7616 |
| P9_B4_C3_F4_S1 | B4 | F4 | wrapper_group_feature_align | group_prior_align | 0.0808 | 0.1613 | 0.1202 | 0.1497 | 11.4703 | 0.2149 | 0.4403 |
| P9_B1_C3_F2_S1 | B1 | F2 | route_prior_strong | z | 0.0808 | 0.1610 | 0.1172 | 0.1490 | 10.5279 | 0.3739 | 0.2265 |
| P9_B1_C3_F4_S1 | B1 | F4 | wrapper_group_feature_align | group_prior_align | 0.0807 | 0.1612 | 0.1193 | 0.1499 | 11.7883 | 0.1340 | 0.2010 |
| P9_B1_C3_F3_S1 | B1 | F3 | group_prior_align | none | 0.0806 | 0.1611 | 0.1190 | 0.1501 | 11.7719 | 0.1392 | 0.1654 |
| P9_B4_C3_F2_S1 | B4 | F2 | route_prior_strong | z | 0.0806 | 0.1613 | 0.1195 | 0.1496 | 11.5276 | 0.2024 | 0.4148 |
| P9_B3_C3_F4_S1 | B3 | F4 | wrapper_group_feature_align | group_prior_align | 0.0803 | 0.1607 | 0.1182 | 0.1488 | 9.1176 | 0.5623 | 0.2713 |
| P9_B2_C3_F3_S1 | B2 | F3 | group_prior_align | none | 0.0802 | 0.1607 | 0.1170 | 0.1479 | 10.0503 | 0.4404 | 0.2121 |
| P9_B3_C3_F1_S1 | B3 | F1 | route_prior | none | 0.0801 | 0.1615 | 0.1201 | 0.1503 | 11.7830 | 0.1357 | 0.1873 |

축 해석
- best(valid): `P9_B2_C3_F2_S1` = 0.0820
- valid range: 0.0019, test range: 0.0053

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_family_top_expert_mean_share | 15 | 0.6602 | 0.8211 |
| best_valid_mrr20 | diag_top1_max_frac | 15 | 0.7103 | 0.7907 |
| test_mrr20 | sess_3_5_mrr20 | 15 | 0.9483 | 0.9024 |
| test_mrr20 | cold_item_mrr20 | 15 | 0.9044 | 0.7646 |

## 4. 축별 best와 수정 효과
축별 best run
| concept_id | concept_name | run_phase | base_id | combo_id | best_valid_mrr20 | test_mrr20 |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 자연형(Natural) | P9_B3_C0_N2_S1 | B3 | N2 | 0.0827 | 0.1581 |
| C1 | 균형형(CanonicalBalance) | P9_B3_C1_B1_S1 | B3 | B1 | 0.0826 | 0.1595 |
| C2 | 특화형(Specialization) | P9_B1_C2_S3_S1 | B1 | S3 | 0.0818 | 0.1596 |
| C3 | 정렬형(FeatureAlignment) | P9_B2_C3_F2_S1 | B2 | F2 | 0.0820 | 0.1593 |

축별 효과 비교
| concept_id | concept_name | n | valid_range | test_range | valid_best_minus_median | test_best_minus_median |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 자연형(Natural) | 16 | 0.002600 | 0.003500 | 0.002000 | 0.001100 |
| C1 | 균형형(CanonicalBalance) | 16 | 0.002500 | 0.006300 | 0.001700 | 0.001000 |
| C2 | 특화형(Specialization) | 16 | 0.001600 | 0.002000 | 0.001250 | 0.000450 |
| C3 | 정렬형(FeatureAlignment) | 15 | 0.001900 | 0.005300 | 0.001200 | 0.000800 |

해석
- valid 기준으론 C1 계열이 강하고, test 평균은 C2가 상대적으로 높아 목적별 선택 분리가 필요.
- C1/C3의 test range가 커서 강한 aux 조합은 변동성 관리가 중요.

## 5. 전체 special/diag와 최종 요약
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_top1_max_frac | 63 | 0.6701 | 0.7428 |
| best_valid_mrr20 | diag_family_top_expert_mean_share | 63 | 0.5600 | 0.7218 |
| best_valid_mrr20 | cold_item_mrr20 | 63 | -0.6218 | -0.7067 |
| best_valid_mrr20 | diag_n_eff | 63 | -0.6878 | -0.6770 |
| best_valid_mrr20 | diag_cv_usage | 63 | 0.5902 | 0.6770 |
| best_valid_mrr20 | sess_3_5_mrr20 | 63 | -0.6570 | -0.6557 |
| test_mrr20 | sess_3_5_mrr20 | 63 | 0.9296 | 0.8778 |
| test_mrr20 | cold_item_mrr20 | 63 | 0.9017 | 0.7384 |
| test_mrr20 | diag_entropy_mean | 63 | 0.6142 | 0.7105 |
| test_mrr20 | diag_n_eff | 63 | 0.7585 | 0.7016 |
| test_mrr20 | diag_cv_usage | 63 | -0.6820 | -0.7016 |
| test_mrr20 | diag_family_top_expert_mean_share | 63 | -0.6508 | -0.6579 |

최종 해석
- 좋은 후보: C0/C1 상위 valid + C2의 test 안정 후보를 함께 보고 shortlist 구성 권장.
- 보통 후보: 변동폭은 큰데 test 이득이 제한적인 조합은 부록/참고 결과로 분리 가능.
- 논문 주장 후보: "aux concept 변경이 진단 패턴과 함께 성능 분포를 이동시킨다".

## 6. Phase9_2는 짧게 (진행 상태만)
| planned_total | main_completed20 | partial_lt20 | pending |
| --- | --- | --- | --- |
| 64 | 47 | 8 | 9 |
- 현재 문서의 해석 중심은 Phase9 본실험이며, Phase9_2는 완료 후 별도 확정 해석 권장.