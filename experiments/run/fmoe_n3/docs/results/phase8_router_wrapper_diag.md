# Phase8 결과 보고서: 축별 흐름 중심 분석

## 1. 이 phase를 왜 했는가
- 목적: 라우팅 구성요소를 A→B→C→D 순으로 바꾸며 "어떤 수정이 실제 효과를 만들었는지"를 분리 설명하기 위함.
- 메인 지표(Main metric): `best valid MRR@20`, 보조 지표(Sub metric): `test MRR@20`
- 본문 통계는 `n_completed>=20`만 사용, partial/pending은 별도 취급.

## 2. 축 정의와 후보
| axis_group | 축 이름 | 변경 요소 | 후보 예시 | 실험 의도 |
| --- | --- | --- | --- | --- |
| A | 래퍼 코어(Wrapper core) | wrapper 조합 | all_w1~all_w6, mixed_1~3 | bias 제거 상태에서 routing 구조 효과만 분리 |
| B | 바이어스 증강(Bias augmentation) | bias mode | bias_off/feat/rule/both/group_feat | feature/rule bias가 성능/붕괴 지표에 미치는 영향 확인 |
| C | 소스 프로파일(Source profile) | primitive source | src_base/src_all_both/src_abc_feature | hidden/feature/both 소스 조합의 일반화 영향 파악 |
| D | 탑케이 정련(Top-k refinement) | primitive/final top-k | tk_dense/tk_d1/tk_d1_final4 | sparse 제약이 성능과 안정성에 미치는 영향 확인 |
| CFM | 확인 패스(Confirmation) | 핵심 설정 재실행 | 핵심 setting + 다른 seed | 재현성 확인 및 진행률 점검 |

## 3.A 축 상세: 래퍼 코어(Wrapper core)
실험 이유
- bias 제거 상태에서 routing 구조 효과만 분리

run 단위 결과(평균/분산 아님, 각 run 직접 비교)
| run_phase | setting_id | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_1_2_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8_SCR_A_ALL_W2_S1 | ALL_W2 | 0.0819 | 0.1603 | 0.1162 | 0.003096 | 0.1489 | 8.9370 | 0.5854 | 0.3995 |
| P8_SCR_A_ALL_W5_S1 | ALL_W5 | 0.0819 | 0.1602 | 0.1168 | 0.0294 | 0.1475 | 9.0545 | 0.5704 | 0.3444 |
| P8_SCR_A_MIXED_1_S1 | MIXED_1 | 0.0814 | 0.1587 | 0.1119 | 0.000000 | 0.1469 | 7.3806 | 0.7911 | 0.5013 |
| P8_SCR_A_MIXED_2_S1 | MIXED_2 | 0.0813 | 0.1604 | 0.1158 | 0.000000 | 0.1471 | 7.3927 | 0.7894 | 0.7096 |
| P8_SCR_A_ALL_W4_S1 | ALL_W4 | 0.0810 | 0.1613 | 0.1213 | 0.007598 | 0.1500 | 11.8859 | 0.0980 | 0.1821 |
| P8_SCR_A_ALL_W6_S1 | ALL_W6 | 0.0809 | 0.1599 | 0.1147 | 0.003460 | 0.1489 | 2.6246 | 1.8900 | 0.7200 |
| P8_SCR_A_ALL_W3_S1 | ALL_W3 | 0.0805 | 0.1613 | 0.1192 | 0.000000 | 0.1507 | 11.9262 | 0.0787 | 0.2398 |
| P8_SCR_A_MIXED_3_S1 | MIXED_3 | 0.0803 | 0.1617 | 0.1191 | 0.000000 | 0.1506 | 11.9022 | 0.0906 | 0.1870 |
| P8_SCR_A_ALL_W1_S1 | ALL_W1 | 0.0801 | 0.1615 | 0.1201 | 0.000000 | 0.1511 | 11.8691 | 0.1050 | 0.2503 |

축 해석
- best(valid): `P8_SCR_A_ALL_W2_S1` = 0.0819, best(test): 0.1617
- valid range: 0.0018, test range: 0.0030
- lowest(valid): `P8_SCR_A_ALL_W1_S1` = 0.0801

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | sess_3_5_mrr20 | 9 | -0.7920 | -0.8285 |
| best_valid_mrr20 | diag_route_jitter_adjacent | 9 | -0.8906 | -0.8285 |
| test_mrr20 | diag_route_consistency_intra_group_knn_mean_score | 9 | 0.7646 | 0.8703 |
| test_mrr20 | diag_entropy_mean | 9 | 0.6543 | 0.8536 |

## 3.B 축 상세: 바이어스 증강(Bias augmentation)
실험 이유
- feature/rule bias가 성능/붕괴 지표에 미치는 영향 확인

run 단위 결과(평균/분산 아님, 각 run 직접 비교)
| run_phase | setting_id | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_1_2_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8_SCR_B_ALL_W2_BIAS_RULE_S1 | ALL_W2_BIAS_RULE | 0.0826 | 0.1596 | 0.1163 | 0.000000 | 0.1473 | 10.5142 | 0.3759 | 0.1722 |
| P8_SCR_B_MIXED_2_BIAS_GROUP_FEAT_S1 | MIXED_2_BIAS_GROUP_FEAT | 0.0823 | 0.1570 | 0.1103 | 0.000000 | 0.1447 | 2.6545 | 1.8764 | 1.0000 |
| P8_SCR_B_ALL_W5_BIAS_RULE_S1 | ALL_W5_BIAS_RULE | 0.0821 | 0.1605 | 0.1169 | 0.003922 | 0.1484 | 8.9805 | 0.5799 | 0.4856 |
| P8_SCR_B_MIXED_2_BIAS_BOTH_S1 | MIXED_2_BIAS_BOTH | 0.0821 | 0.1594 | 0.1143 | 0.000000 | 0.1470 | 5.3546 | 1.1140 | 0.9846 |
| P8_SCR_B_ALL_W2_BIAS_GROUP_FEAT_RULE_S1 | ALL_W2_BIAS_GROUP_FEAT_RULE | 0.0817 | 0.1605 | 0.1164 | 0.003460 | 0.1488 | 9.9532 | 0.4535 | 0.2436 |
| P8_SCR_B_ALL_W5_BIAS_FEAT_S1 | ALL_W5_BIAS_FEAT | 0.0816 | 0.1604 | 0.1144 | 0.000000 | 0.1477 | 9.7737 | 0.4773 | 0.3374 |
| P8_SCR_B_MIXED_1_BIAS_GROUP_FEAT_RULE_S1 | MIXED_1_BIAS_GROUP_FEAT_RULE | 0.0815 | 0.1593 | 0.1144 | 0.000000 | 0.1471 | 10.3278 | 0.4024 | 0.1858 |
| P8_SCR_B_ALL_W2_BIAS_FEAT_S1 | ALL_W2_BIAS_FEAT | 0.0813 | 0.1614 | 0.1168 | 0.000000 | 0.1501 | 8.5732 | 0.6322 | 0.3718 |
| P8_SCR_B_ALL_W5_BIAS_OFF_S1 | ALL_W5_BIAS_OFF | 0.0813 | 0.1606 | 0.1153 | 0.0588 | 0.1484 | 9.6484 | 0.4937 | 0.2542 |
| P8_SCR_B_MIXED_2_BIAS_GROUP_FEAT_RULE_S1 | MIXED_2_BIAS_GROUP_FEAT_RULE | 0.0810 | 0.1608 | 0.1143 | 0.000000 | 0.1487 | 7.9387 | 0.7153 | 0.3196 |
| P8_SCR_B_MIXED_1_BIAS_RULE_S1 | MIXED_1_BIAS_RULE | 0.0809 | 0.1601 | 0.1151 | 0.000000 | 0.1485 | 9.9610 | 0.4524 | 0.2564 |
| P8_SCR_B_ALL_W5_BIAS_BOTH_S1 | ALL_W5_BIAS_BOTH | 0.0809 | 0.1601 | 0.1154 | 0.000000 | 0.1468 | 8.8210 | 0.6003 | 0.6444 |
| P8_SCR_B_MIXED_1_BIAS_GROUP_FEAT_S1 | MIXED_1_BIAS_GROUP_FEAT | 0.0808 | 0.1618 | 0.1211 | 0.000000 | 0.1516 | 11.6153 | 0.1820 | 0.2119 |
| P8_SCR_B_MIXED_1_BIAS_BOTH_S1 | MIXED_1_BIAS_BOTH | 0.0807 | 0.1617 | 0.1211 | 0.000000 | 0.1514 | 11.6568 | 0.1716 | 0.2039 |
| P8_SCR_B_ALL_W2_BIAS_BOTH_S1 | ALL_W2_BIAS_BOTH | 0.0807 | 0.1611 | 0.1192 | 0.009804 | 0.1501 | 11.7447 | 0.1474 | 0.2405 |
| P8_SCR_B_ALL_W2_BIAS_GROUP_FEAT_S1 | ALL_W2_BIAS_GROUP_FEAT | 0.0807 | 0.1612 | 0.1195 | 0.0147 | 0.1505 | 11.7472 | 0.1467 | 0.2870 |
| P8_SCR_B_MIXED_1_BIAS_OFF_S1 | MIXED_1_BIAS_OFF | 0.0807 | 0.1614 | 0.1206 | 0.000000 | 0.1509 | 11.6616 | 0.1703 | 0.2401 |
| P8_SCR_B_MIXED_1_BIAS_FEAT_S1 | MIXED_1_BIAS_FEAT | 0.0807 | 0.1614 | 0.1209 | 0.000000 | 0.1508 | 11.6773 | 0.1662 | 0.2231 |
| P8_SCR_B_ALL_W5_BIAS_GROUP_FEAT_RULE_S1 | ALL_W5_BIAS_GROUP_FEAT_RULE | 0.0807 | 0.1617 | 0.1194 | 0.0196 | 0.1508 | 11.6013 | 0.1854 | 0.4772 |
| P8_SCR_B_ALL_W5_BIAS_GROUP_FEAT_S1 | ALL_W5_BIAS_GROUP_FEAT | 0.0807 | 0.1616 | 0.1194 | 0.004902 | 0.1506 | 11.5918 | 0.1877 | 0.4575 |
| P8_SCR_B_ALL_W2_BIAS_OFF_S1 | ALL_W2_BIAS_OFF | 0.0806 | 0.1610 | 0.1176 | 0.000000 | 0.1482 | 10.5760 | 0.3669 | 0.2344 |
| P8_SCR_B_MIXED_2_BIAS_RULE_S1 | MIXED_2_BIAS_RULE | 0.0804 | 0.1601 | 0.1171 | 0.004202 | 0.1482 | 5.1053 | 1.1621 | 0.3493 |
| P8_SCR_B_MIXED_2_BIAS_FEAT_S1 | MIXED_2_BIAS_FEAT | 0.0802 | 0.1614 | 0.1184 | 0.000000 | 0.1504 | 11.7979 | 0.1309 | 0.2499 |
| P8_SCR_B_MIXED_2_BIAS_OFF_S1 | MIXED_2_BIAS_OFF | 0.0802 | 0.1619 | 0.1206 | 0.000000 | 0.1514 | 11.7093 | 0.1576 | 0.2988 |

축 해석
- best(valid): `P8_SCR_B_ALL_W2_BIAS_RULE_S1` = 0.0826, best(test): 0.1619
- valid range: 0.0024, test range: 0.0049
- lowest(valid): `P8_SCR_B_MIXED_2_BIAS_OFF_S1` = 0.0802

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_route_consistency_intra_group_knn_mean_score | 24 | -0.7497 | -0.7177 |
| best_valid_mrr20 | cold_item_mrr20 | 24 | -0.6631 | -0.6935 |
| test_mrr20 | sess_3_5_mrr20 | 24 | 0.9289 | 0.9433 |
| test_mrr20 | cold_item_mrr20 | 24 | 0.8699 | 0.8381 |

## 3.C 축 상세: 소스 프로파일(Source profile)
실험 이유
- hidden/feature/both 소스 조합의 일반화 영향 파악

run 단위 결과(평균/분산 아님, 각 run 직접 비교)
| run_phase | setting_id | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_1_2_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE | 0.0822 | 0.1596 | 0.1158 | 0.000000 | 0.1482 | 5.9311 | 1.0115 | 0.6432 |
| P8_SCR_C_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE | 0.0816 | 0.1597 | 0.1119 | 0.000000 | 0.1477 | 7.2819 | 0.8049 | 0.7844 |
| P8_SCR_C_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_S1 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE | 0.0812 | 0.1602 | 0.1163 | 0.000000 | 0.1477 | 9.6330 | 0.4957 | 0.4134 |
| P8_SCR_C_ALL_W2_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_S1 | ALL_W2_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE | 0.0812 | 0.1601 | 0.1173 | 0.003460 | 0.1476 | 9.1211 | 0.5618 | 0.3028 |
| P8_SCR_C_ALL_W5_BIAS_RULE_SRC_BASE_S1 | ALL_W5_BIAS_RULE_SRC_BASE | 0.0811 | 0.1602 | 0.1162 | 0.000000 | 0.1481 | 9.5481 | 0.5067 | 0.3239 |
| P8_SCR_C_ALL_W2_BIAS_RULE_SRC_ABC_FEATURE_S1 | ALL_W2_BIAS_RULE_SRC_ABC_FEATURE | 0.0810 | 0.1605 | 0.1176 | 0.008403 | 0.1482 | 8.9449 | 0.5844 | 0.5088 |
| P8_SCR_C_ALL_W5_BIAS_RULE_SRC_ALL_BOTH_S1 | ALL_W5_BIAS_RULE_SRC_ALL_BOTH | 0.0810 | 0.1600 | 0.1164 | 0.008403 | 0.1482 | 1.3581 | 2.7993 | 0.9643 |
| P8_SCR_C_ALL_W2_BIAS_RULE_SRC_BASE_S1 | ALL_W2_BIAS_RULE_SRC_BASE | 0.0810 | 0.1610 | 0.1161 | 0.000000 | 0.1484 | 10.3390 | 0.4008 | 0.2048 |
| P8_SCR_C_MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH_S1 | MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH | 0.0809 | 0.1617 | 0.1219 | 0.000000 | 0.1509 | 11.8386 | 0.1167 | 0.2127 |
| P8_SCR_C_MIXED_2_BIAS_GROUP_FEAT_SRC_ABC_FEATURE_S1 | MIXED_2_BIAS_GROUP_FEAT_SRC_ABC_FEATURE | 0.0808 | 0.1614 | 0.1202 | 0.000000 | 0.1498 | 11.5262 | 0.2027 | 0.4323 |
| P8_SCR_C_ALL_W2_BIAS_RULE_SRC_ALL_BOTH_S1 | ALL_W2_BIAS_RULE_SRC_ALL_BOTH | 0.0807 | 0.1611 | 0.1193 | 0.000000 | 0.1503 | 11.6711 | 0.1679 | 0.1653 |
| P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ALL_BOTH_S1 | MIXED_2_BIAS_BOTH_SRC_ALL_BOTH | 0.0807 | 0.1618 | 0.1211 | 0.000000 | 0.1510 | 11.9136 | 0.0851 | 0.2645 |
| P8_SCR_C_MIXED_2_BIAS_GROUP_FEAT_SRC_A_HIDDEN_B_D_FEATURE_S1 | MIXED_2_BIAS_GROUP_FEAT_SRC_A_HIDDEN_B_D_FEATURE | 0.0805 | 0.1614 | 0.1200 | 0.004525 | 0.1509 | 11.7522 | 0.1452 | 0.2051 |
| P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_A_HIDDEN_B_D_FEATURE_S1 | MIXED_2_BIAS_BOTH_SRC_A_HIDDEN_B_D_FEATURE | 0.0805 | 0.1614 | 0.1201 | 0.003922 | 0.1508 | 11.7682 | 0.1403 | 0.2030 |
| P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_BASE_S1 | MIXED_2_BIAS_BOTH_SRC_BASE | 0.0803 | 0.1620 | 0.1202 | 0.000000 | 0.1513 | 11.7578 | 0.1435 | 0.3070 |
| P8_SCR_C_MIXED_2_BIAS_GROUP_FEAT_SRC_BASE_S1 | MIXED_2_BIAS_GROUP_FEAT_SRC_BASE | 0.0803 | 0.1608 | 0.1173 | 0.000000 | 0.1477 | 9.3558 | 0.5316 | 0.2831 |

축 해석
- best(valid): `P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_S1` = 0.0822, best(test): 0.1620
- valid range: 0.0019, test range: 0.0024
- lowest(valid): `P8_SCR_C_MIXED_2_BIAS_GROUP_FEAT_SRC_BASE_S1` = 0.0803

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_family_top_expert_mean_share | 16 | 0.2909 | 0.7323 |
| best_valid_mrr20 | cold_item_mrr20 | 16 | -0.6542 | -0.7204 |
| test_mrr20 | diag_family_top_expert_mean_share | 16 | -0.5328 | -0.9550 |
| test_mrr20 | diag_entropy_mean | 16 | 0.5568 | 0.9402 |

## 3.D 축 상세: 탑케이 정련(Top-k refinement)
실험 이유
- sparse 제약이 성능과 안정성에 미치는 영향 확인

run 단위 결과(평균/분산 아님, 각 run 직접 비교)
| run_phase | setting_id | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_1_2_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE | 0.0812 | 0.1600 | 0.1153 | 0.000000 | 0.1478 | 10.1290 | 0.4298 | 0.2812 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE_S1 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE | 0.0807 | 0.1619 | 0.1200 | 0.0294 | 0.1514 | 11.5801 | 0.1904 | 0.4706 |
| P8_SCR_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE | 0.0807 | 0.1616 | 0.1203 | 0.000000 | 0.1511 | 11.7451 | 0.1473 | 0.3950 |
| P8_SCR_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1 | 0.0806 | 0.1611 | 0.1186 | 0.000000 | 0.1499 | 4.2291 | 1.3555 | 0.3748 |
| P8_SCR_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1_FINAL4_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1_FINAL4 | 0.0806 | 0.1613 | 0.1197 | 0.000000 | 0.1496 | 4.2217 | 1.3574 | 0.2767 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_D1_FINAL4_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_D1_FINAL4 | 0.0798 | 0.1604 | 0.1160 | 0.000000 | 0.1478 | 4.0528 | 1.4003 | 0.5888 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_A3_D1_FINAL4_S1 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_A3_D1_FINAL4 | 0.0798 | 0.1602 | 0.1150 | 0.000000 | 0.1475 | 4.1379 | 1.3784 | 0.5481 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_D1_FINAL4_S1 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_D1_FINAL4 | 0.0798 | 0.1603 | 0.1151 | 0.000000 | 0.1475 | 4.1283 | 1.3809 | 0.5536 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_A3_D1_FINAL4_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_A3_D1_FINAL4 | 0.0798 | 0.1602 | 0.1159 | 0.000000 | 0.1479 | 4.1340 | 1.3794 | 0.5667 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_D1_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_D1 | 0.0798 | 0.1604 | 0.1162 | 0.000000 | 0.1480 | 4.0577 | 1.3991 | 0.5967 |
| P8_SCR_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_D1_S1 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_D1 | 0.0797 | 0.1603 | 0.1132 | 0.000000 | 0.1479 | 4.2002 | 1.3627 | 0.5443 |
| P8_SCR_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_A3_D1_FINAL4_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_A3_D1_FINAL4 | 0.0793 | 0.1611 | 0.1175 | 0.000000 | 0.1494 | 4.1310 | 1.3802 | 0.3155 |

축 해석
- best(valid): `P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S1` = 0.0812, best(test): 0.1619
- valid range: 0.0019, test range: 0.0019
- lowest(valid): `P8_SCR_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_A3_D1_FINAL4_S1` = 0.0793

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_n_eff | 12 | 0.7260 | 0.7424 |
| best_valid_mrr20 | diag_cv_usage | 12 | -0.7273 | -0.7424 |
| test_mrr20 | cold_item_mrr20 | 12 | 0.9174 | 0.8838 |
| test_mrr20 | sess_3_5_mrr20 | 12 | 0.9743 | 0.8380 |

## 3.CFM 축 상세: 확인 패스(Confirmation)
실험 이유
- 재현성 확인 및 진행률 점검

run 단위 결과(평균/분산 아님, 각 run 직접 비교)
| run_phase | setting_id | best_valid_mrr20 | test_mrr20 | cold_item_mrr20 | sess_1_2_mrr20 | sess_3_5_mrr20 | diag_n_eff | diag_cv_usage | diag_top1_max_frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P8_CFM_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE_S2 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE | 0.0813 | 0.1606 | 0.1163 | 0.0147 | 0.1482 | 9.9991 | 0.4473 | 0.3136 |
| P8_CFM_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S2 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE | 0.0809 | 0.1597 | 0.1167 | 0.000000 | 0.1460 | 8.9157 | 0.5882 | 0.3233 |
| P8_CFM_D_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE | 0.0808 | 0.1614 | 0.1203 | 0.000000 | 0.1497 | 11.5342 | 0.2010 | 0.4345 |
| P8_CFM_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE | 0.0807 | 0.1616 | 0.1195 | 0.004902 | 0.1506 | 11.5938 | 0.1872 | 0.4581 |
| P8_CFM_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S3 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE | 0.0807 | 0.1616 | 0.1187 | 0.0118 | 0.1504 | 11.6213 | 0.1805 | 0.4883 |

축 해석
- best(valid): `P8_CFM_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE_S2` = 0.0813, best(test): 0.1616
- valid range: 0.0006, test range: 0.0019
- lowest(valid): `P8_CFM_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S3` = 0.0807

축 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_top1_max_frac | 5 | -0.8391 | -0.9747 |
| best_valid_mrr20 | diag_n_eff | 5 | -0.5925 | -0.8721 |
| test_mrr20 | sess_3_5_mrr20 | 5 | 0.9958 | 0.9747 |
| test_mrr20 | diag_cv_usage | 5 | -0.9900 | -0.9747 |

## 4. 축별 best와 효과 비교
축별 best run
| axis_group | run_phase | setting_id | best_valid_mrr20 | test_mrr20 |
| --- | --- | --- | --- | --- |
| A | P8_SCR_A_ALL_W2_S1 | ALL_W2 | 0.0819 | 0.1603 |
| B | P8_SCR_B_ALL_W2_BIAS_RULE_S1 | ALL_W2_BIAS_RULE | 0.0826 | 0.1596 |
| C | P8_SCR_C_MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_S1 | MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE | 0.0822 | 0.1596 |
| CFM | P8_CFM_D_ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE_S2 | ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE | 0.0813 | 0.1606 |
| D | P8_SCR_D_ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE_S1 | ALL_W5_BIAS_RULE_SRC_A_HIDDEN_B_D_FEATURE_TK_DENSE | 0.0812 | 0.1600 |

축 수정 효과(변동폭/피크-중앙값)
| axis_group | n | valid_range | test_range | valid_best_minus_median | test_best_minus_median |
| --- | --- | --- | --- | --- | --- |
| B | 24 | 0.002400 | 0.004900 | 0.001750 | 0.001000 |
| D | 12 | 0.001900 | 0.001900 | 0.001400 | 0.001500 |
| C | 16 | 0.001900 | 0.002400 | 0.001250 | 0.001100 |
| A | 9 | 0.001800 | 0.003000 | 0.000900 | 0.001300 |
| CFM | 5 | 0.000600 | 0.001900 | 0.000500 | 0.000200 |

해석
- B 축이 valid/test 변동폭 모두 가장 커, 이번 phase에서 가장 영향이 큰 수정 축으로 해석 가능.
- D 축은 test 변동폭은 제한적이나 valid 평균대가 낮아 강한 top-k 제약은 보수적으로 다루는 것이 타당.

## 5. 전체 special/diag 정리와 최종 요약
전체 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | sess_3_5_mrr20 | 66 | -0.3845 | -0.4127 |
| best_valid_mrr20 | cold_item_mrr20 | 66 | -0.3495 | -0.3772 |
| best_valid_mrr20 | diag_route_consistency_group_knn_score | 66 | -0.2879 | -0.3039 |
| best_valid_mrr20 | diag_n_eff | 66 | 0.0718 | -0.1611 |
| best_valid_mrr20 | diag_cv_usage | 66 | -0.0511 | 0.1611 |
| best_valid_mrr20 | sess_1_2_mrr20 | 66 | 0.1437 | 0.1566 |
| test_mrr20 | sess_3_5_mrr20 | 66 | 0.9037 | 0.9009 |
| test_mrr20 | cold_item_mrr20 | 66 | 0.8728 | 0.8405 |
| test_mrr20 | diag_entropy_mean | 66 | 0.5333 | 0.7607 |
| test_mrr20 | diag_n_eff | 66 | 0.6101 | 0.7121 |
| test_mrr20 | diag_cv_usage | 66 | -0.5902 | -0.7121 |
| test_mrr20 | diag_family_top_expert_mean_share | 66 | -0.5603 | -0.7102 |

최종 해석
- 좋았던 계열: B축 상위 설정(특히 `ALL_W2_BIAS_RULE`)은 valid 피크 관점에서 강함.
- 보통/보수 계열: D축 일부 설정은 안정성 대비 성능 이득이 제한적.
- 논문 주장 후보: "bias augmentation이 가장 큰 성능 레버", "n_eff/cv_usage 계열 진단과 test 동행".