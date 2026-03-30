# Phase8~13 Special/Diag 통계 및 Correlation 정리 (v7: special 복구 + metric-centric)

## 수정 요약
- `axis별 1개 추천` 대신 **축별 top3 추천**으로 복구
- `special(cold/short)` 표를 다시 확장하고, valid/test slice를 함께 표기
- 표 렌더 깨짐 방지를 위해 표 앞뒤 공백을 명시하고 섹션 단위를 분리
- diag는 기존 요청대로 **metric 가로축 중심** 구성 유지

## 집계 범위
- run 수: `532`
- raw rho(valid,test): `0.0094` (n=532)
- axis-centered rho(valid,test): `-0.2008` (n=532)

## Special: Test 고려 축별 Top3 세팅
점수는 `0.25*valid + 0.40*test + 0.20*test_cold(<=5) + 0.15*test_short(1-2)` (축 내 z-score)입니다.
각 값은 동일 setting의 평균이며, `+-`는 std입니다.

| family | axis | rank | setting | n | valid MRR@20 | test MRR@20 | valid cold<=5 | test cold<=5 | valid short 1-2 | test short 1-2 | score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P8 Router Core` | `P8 Stage A: wrapper_combo` | 1 | `ALL_W4` | 1 | 0.0810 | 0.1613 | 0.0353 | 0.1213 | 0.0101 | 0.0076 | 0.600 |
| `P8 Router Core` | `P8 Stage A: wrapper_combo` | 2 | `ALL_W5` | 1 | 0.0819 | 0.1602 | 0.0355 | 0.1168 | 0.0076 | 0.0294 | 0.527 |
| `P8 Router Core` | `P8 Stage A: wrapper_combo` | 3 | `MIXED_3` | 1 | 0.0803 | 0.1617 | 0.0337 | 0.1191 | 0.0116 | 0.0000 | 0.234 |
| `P8 Router Core` | `P8 Stage B: bias_mode` | 1 | `ALL_W5_BIAS_GROUP_FEAT_RULE` | 1 | 0.0807 | 0.1617 | 0.0343 | 0.1194 | 0.0121 | 0.0196 | 0.550 |
| `P8 Router Core` | `P8 Stage B: bias_mode` | 2 | `ALL_W5_BIAS_OFF` | 1 | 0.0813 | 0.1606 | 0.0351 | 0.1153 | 0.0303 | 0.0588 | 0.549 |
| `P8 Router Core` | `P8 Stage B: bias_mode` | 3 | `MIXED_1_BIAS_GROUP_FEAT` | 1 | 0.0808 | 0.1618 | 0.0355 | 0.1211 | 0.0101 | 0.0000 | 0.513 |
| `P8 Router Core` | `P8 Stage C: source_profile` | 1 | `MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH` | 1 | 0.0809 | 0.1617 | 0.0355 | 0.1219 | 0.0195 | 0.0000 | 0.654 |
| `P8 Router Core` | `P8 Stage C: source_profile` | 2 | `MIXED_2_BIAS_BOTH_SRC_ALL_BOTH` | 1 | 0.0807 | 0.1618 | 0.0353 | 0.1211 | 0.0152 | 0.0000 | 0.540 |
| `P8 Router Core` | `P8 Stage C: source_profile` | 3 | `MIXED_2_BIAS_GROUP_FEAT_SRC_A_HIDDEN_B_D_FEATURE` | 1 | 0.0805 | 0.1614 | 0.0351 | 0.1200 | 0.0111 | 0.0045 | 0.376 |
| `P8 Router Core` | `P8 Stage D: topk_profile` | 1 | `ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE` | 1 | 0.0807 | 0.1619 | 0.0345 | 0.1200 | 0.0096 | 0.0294 | 1.724 |
| `P8 Router Core` | `P8 Stage D: topk_profile` | 2 | `MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE` | 1 | 0.0807 | 0.1616 | 0.0345 | 0.1203 | 0.0099 | 0.0000 | 1.039 |
| `P8 Router Core` | `P8 Stage D: topk_profile` | 3 | `MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1_FINAL4` | 1 | 0.0806 | 0.1613 | 0.0341 | 0.1197 | 0.0152 | 0.0000 | 0.754 |
| `P8 Verification` | `P8 Verification Base-A` | 1 | `A_H3` | 4 | 0.0812 +- 0.0001 | 0.1612 +- 0.0006 | 0.0357 +- 0.0010 | 0.1208 +- 0.0033 | 0.0088 +- 0.0064 | 0.0000 +- 0.0000 | 0.254 |
| `P8 Verification` | `P8 Verification Base-A` | 2 | `A_H1` | 4 | 0.0810 +- 0.0004 | 0.1609 +- 0.0010 | 0.0347 +- 0.0003 | 0.1177 +- 0.0022 | 0.0088 +- 0.0044 | 0.0025 +- 0.0028 | 0.112 |
| `P8 Verification` | `P8 Verification Base-A` | 3 | `A_H4` | 4 | 0.0810 +- 0.0000 | 0.1617 +- 0.0001 | 0.0346 +- 0.0001 | 0.1185 +- 0.0001 | 0.0088 +- 0.0015 | 0.0000 +- 0.0000 | 0.086 |
| `P8 Verification` | `P8 Verification Base-B` | 1 | `B_H3` | 4 | 0.0811 +- 0.0006 | 0.1610 +- 0.0009 | 0.0355 +- 0.0012 | 0.1194 +- 0.0035 | 0.0069 +- 0.0023 | 0.0009 +- 0.0018 | 0.158 |
| `P8 Verification` | `P8 Verification Base-B` | 2 | `B_H4` | 4 | 0.0806 +- 0.0002 | 0.1613 +- 0.0008 | 0.0347 +- 0.0004 | 0.1185 +- 0.0006 | 0.0095 +- 0.0038 | 0.0033 +- 0.0004 | 0.100 |
| `P8 Verification` | `P8 Verification Base-B` | 3 | `B_H2` | 4 | 0.0805 +- 0.0003 | 0.1610 +- 0.0014 | 0.0341 +- 0.0004 | 0.1195 +- 0.0033 | 0.0044 +- 0.0022 | 0.0042 +- 0.0049 | 0.062 |
| `P8 Verification` | `P8 Verification Base-C` | 1 | `C_H3` | 4 | 0.0814 +- 0.0003 | 0.1610 +- 0.0008 | 0.0346 +- 0.0013 | 0.1190 +- 0.0042 | 0.0112 +- 0.0052 | 0.0009 +- 0.0017 | 0.256 |
| `P8 Verification` | `P8 Verification Base-C` | 2 | `C_H4` | 4 | 0.0810 +- 0.0001 | 0.1616 +- 0.0001 | 0.0349 +- 0.0002 | 0.1190 +- 0.0003 | 0.0088 +- 0.0015 | 0.0000 +- 0.0000 | 0.157 |
| `P8 Verification` | `P8 Verification Base-C` | 3 | `C_H1` | 4 | 0.0810 +- 0.0004 | 0.1609 +- 0.0010 | 0.0348 +- 0.0008 | 0.1176 +- 0.0023 | 0.0154 +- 0.0123 | 0.0021 +- 0.0026 | -0.164 |
| `P8 Verification` | `P8 Verification Base-D` | 1 | `D_H4` | 4 | 0.0813 +- 0.0004 | 0.1613 +- 0.0003 | 0.0355 +- 0.0004 | 0.1182 +- 0.0017 | 0.0220 +- 0.0157 | 0.0029 +- 0.0034 | 0.237 |
| `P8 Verification` | `P8 Verification Base-D` | 2 | `D_H3` | 4 | 0.0812 +- 0.0005 | 0.1610 +- 0.0006 | 0.0338 +- 0.0005 | 0.1179 +- 0.0021 | 0.0066 +- 0.0042 | 0.0048 +- 0.0039 | 0.144 |
| `P8 Verification` | `P8 Verification Base-D` | 3 | `D_H2` | 4 | 0.0806 +- 0.0001 | 0.1615 +- 0.0007 | 0.0334 +- 0.0011 | 0.1200 +- 0.0017 | 0.0101 +- 0.0071 | 0.0016 +- 0.0019 | 0.066 |
| `P9 Aux Concept` | `P9 C0: Natural` | 1 | `N1` | 4 | 0.0805 +- 0.0003 | 0.1611 +- 0.0005 | 0.0343 +- 0.0003 | 0.1189 +- 0.0013 | 0.0059 +- 0.0036 | 0.0074 +- 0.0147 | 0.287 |
| `P9 Aux Concept` | `P9 C0: Natural` | 2 | `N3` | 4 | 0.0809 +- 0.0010 | 0.1602 +- 0.0012 | 0.0348 +- 0.0005 | 0.1169 +- 0.0019 | 0.0146 +- 0.0148 | 0.0196 +- 0.0277 | 0.023 |
| `P9 Aux Concept` | `P9 C0: Natural` | 3 | `N2` | 4 | 0.0811 +- 0.0011 | 0.1603 +- 0.0015 | 0.0348 +- 0.0008 | 0.1160 +- 0.0050 | 0.0093 +- 0.0022 | 0.0037 +- 0.0074 | -0.089 |
| `P9 Aux Concept` | `P9 C1: CanonicalBalance` | 1 | `B2` | 4 | 0.0806 +- 0.0004 | 0.1613 +- 0.0007 | 0.0345 +- 0.0003 | 0.1183 +- 0.0017 | 0.0057 +- 0.0051 | 0.0033 +- 0.0039 | 0.262 |
| `P9 Aux Concept` | `P9 C1: CanonicalBalance` | 2 | `B4` | 4 | 0.0814 +- 0.0006 | 0.1604 +- 0.0008 | 0.0345 +- 0.0009 | 0.1161 +- 0.0030 | 0.0078 +- 0.0069 | 0.0011 +- 0.0021 | 0.072 |
| `P9 Aux Concept` | `P9 C1: CanonicalBalance` | 3 | `B3` | 4 | 0.0811 +- 0.0009 | 0.1604 +- 0.0013 | 0.0343 +- 0.0011 | 0.1161 +- 0.0050 | 0.0075 +- 0.0066 | 0.0011 +- 0.0023 | 0.008 |
| `P9 Aux Concept` | `P9 C2: Specialization` | 1 | `S1` | 4 | 0.0805 +- 0.0004 | 0.1614 +- 0.0003 | 0.0347 +- 0.0004 | 0.1189 +- 0.0015 | 0.0052 +- 0.0025 | 0.0037 +- 0.0074 | 0.314 |
| `P9 Aux Concept` | `P9 C2: Specialization` | 2 | `S2` | 4 | 0.0805 +- 0.0002 | 0.1612 +- 0.0001 | 0.0353 +- 0.0003 | 0.1199 +- 0.0005 | 0.0120 +- 0.0050 | 0.0000 +- 0.0000 | 0.116 |
| `P9 Aux Concept` | `P9 C2: Specialization` | 3 | `S4` | 4 | 0.0807 +- 0.0004 | 0.1610 +- 0.0007 | 0.0350 +- 0.0009 | 0.1184 +- 0.0013 | 0.0076 +- 0.0029 | 0.0008 +- 0.0015 | -0.047 |
| `P9 Aux Concept` | `P9 C3: FeatureAlignment` | 1 | `F4` | 4 | 0.0809 +- 0.0006 | 0.1608 +- 0.0006 | 0.0349 +- 0.0004 | 0.1186 +- 0.0015 | 0.0101 +- 0.0062 | 0.0033 +- 0.0041 | 0.191 |
| `P9 Aux Concept` | `P9 C3: FeatureAlignment` | 2 | `F3` | 3 | 0.0806 +- 0.0004 | 0.1609 +- 0.0002 | 0.0342 +- 0.0004 | 0.1170 +- 0.0021 | 0.0040 +- 0.0039 | 0.0098 +- 0.0170 | 0.135 |
| `P9 Aux Concept` | `P9 C3: FeatureAlignment` | 3 | `F2` | 4 | 0.0812 +- 0.0006 | 0.1602 +- 0.0011 | 0.0348 +- 0.0010 | 0.1170 +- 0.0020 | 0.0136 +- 0.0094 | 0.0000 +- 0.0000 | 0.030 |
| `P9 Verification` | `P9_2 K1` | 1 | `N4` | 16 | 0.0811 +- 0.0002 | 0.1614 +- 0.0006 | 0.0353 +- 0.0009 | 0.1207 +- 0.0019 | 0.0143 +- 0.0039 | 0.0002 +- 0.0009 | 0.000 |
| `P9 Verification` | `P9_2 K2` | 1 | `B1` | 16 | 0.0806 +- 0.0004 | 0.1609 +- 0.0009 | 0.0346 +- 0.0009 | 0.1182 +- 0.0037 | 0.0088 +- 0.0049 | 0.0029 +- 0.0057 | 0.000 |
| `P9 Verification` | `P9_2 K3` | 1 | `S3` | 15 | 0.0808 +- 0.0002 | 0.1611 +- 0.0013 | 0.0337 +- 0.0010 | 0.1190 +- 0.0048 | 0.0113 +- 0.0038 | 0.0003 +- 0.0010 | 0.000 |
| `P10 Feature Portability` | `P10 availability` | 1 | `NO_TIMESTAMP` | 9 | 0.0807 +- 0.0003 | 0.1617 +- 0.0002 | 0.0349 +- 0.0006 | 0.1212 +- 0.0014 | 0.0134 +- 0.0029 | 0.0000 +- 0.0000 | 0.009 |
| `P10 Feature Portability` | `P10 availability` | 2 | `NO_CATEGORY` | 9 | 0.0809 +- 0.0004 | 0.1614 +- 0.0012 | 0.0342 +- 0.0008 | 0.1192 +- 0.0031 | 0.0089 +- 0.0044 | 0.0028 +- 0.0035 | -0.009 |
| `P10 Feature Portability` | `P10 availability_plus` | 1 | `NO_CATEGORY_NO_TIMESTAMP` | 8 | 0.0803 +- 0.0001 | 0.1614 +- 0.0001 | 0.0353 +- 0.0005 | 0.1212 +- 0.0011 | 0.0146 +- 0.0074 | 0.0000 +- 0.0000 | 0.000 |
| `P10 Feature Portability` | `P10 compactness` | 1 | `TOP2_PER_GROUP` | 9 | 0.0808 +- 0.0002 | 0.1618 +- 0.0005 | 0.0352 +- 0.0010 | 0.1210 +- 0.0020 | 0.0115 +- 0.0036 | 0.0044 +- 0.0058 | 0.222 |
| `P10 Feature Portability` | `P10 compactness` | 2 | `COMMON_TEMPLATE` | 9 | 0.0807 +- 0.0003 | 0.1616 +- 0.0010 | 0.0345 +- 0.0009 | 0.1204 +- 0.0034 | 0.0087 +- 0.0033 | 0.0041 +- 0.0044 | -0.018 |
| `P10 Feature Portability` | `P10 compactness` | 3 | `TOP1_PER_GROUP` | 9 | 0.0803 +- 0.0001 | 0.1619 +- 0.0003 | 0.0341 +- 0.0006 | 0.1209 +- 0.0009 | 0.0180 +- 0.0087 | 0.0008 +- 0.0016 | -0.204 |
| `P10 Feature Portability` | `P10 compactness_plus` | 1 | `COMMON_TEMPLATE_NO_CATEGORY` | 8 | 0.0806 +- 0.0002 | 0.1619 +- 0.0001 | 0.0341 +- 0.0006 | 0.1209 +- 0.0009 | 0.0113 +- 0.0052 | 0.0000 +- 0.0000 | 0.000 |
| `P10 Feature Portability` | `P10 feature_subset` | 1 | `FULL` | 9 | 0.0811 +- 0.0003 | 0.1615 +- 0.0002 | 0.0355 +- 0.0008 | 0.1216 +- 0.0015 | 0.0164 +- 0.0013 | 0.0000 +- 0.0000 | 0.430 |
| `P10 Feature Portability` | `P10 feature_subset` | 2 | `Tempo_Focus_Exposure` | 1 | 0.0806 | 0.1618 | 0.0353 | 0.1220 | 0.0101 | 0.0000 | 0.405 |
| `P10 Feature Portability` | `P10 feature_subset` | 3 | `Tempo_Memory` | 9 | 0.0799 +- 0.0001 | 0.1619 +- 0.0001 | 0.0340 +- 0.0006 | 0.1206 +- 0.0007 | 0.0024 +- 0.0012 | 0.0008 +- 0.0016 | 0.203 |
| `P10 Feature Portability` | `P10 stochastic` | 1 | `FEATURE_DROPOUT` | 9 | 0.0807 +- 0.0004 | 0.1619 +- 0.0003 | 0.0341 +- 0.0005 | 0.1211 +- 0.0010 | 0.0130 +- 0.0025 | 0.0004 +- 0.0011 | 0.005 |
| `P10 Feature Portability` | `P10 stochastic` | 2 | `FAMILY_DROPOUT` | 9 | 0.0806 +- 0.0003 | 0.1619 +- 0.0006 | 0.0339 +- 0.0008 | 0.1211 +- 0.0014 | 0.0111 +- 0.0023 | 0.0028 +- 0.0040 | -0.005 |
| `P11 Stage Semantics` | `P11 base_ablation` | 1 | `MID_MICRO` | 1 | 0.0804 | 0.1617 | 0.0344 | 0.1218 | 0.0051 | 0.0000 | 0.235 |
| `P11 Stage Semantics` | `P11 base_ablation` | 2 | `MACRO_MID_MICRO` | 5 | 0.0816 +- 0.0007 | 0.1607 +- 0.0019 | 0.0358 +- 0.0008 | 0.1199 +- 0.0067 | 0.0156 +- 0.0010 | 0.0000 +- 0.0000 | 0.087 |
| `P11 Stage Semantics` | `P11 base_ablation` | 3 | `MACRO_MID` | 5 | 0.0807 +- 0.0002 | 0.1612 +- 0.0005 | 0.0345 +- 0.0003 | 0.1188 +- 0.0024 | 0.0108 +- 0.0055 | 0.0000 +- 0.0000 | 0.002 |
| `P11 Stage Semantics` | `P11 extra_alignment` | 1 | `LAYER2_MACRO_MID_MICRO` | 5 | 0.0816 +- 0.0001 | 0.1611 +- 0.0013 | 0.0347 +- 0.0007 | 0.1168 +- 0.0033 | 0.0189 +- 0.0145 | 0.0000 +- 0.0000 | 0.638 |
| `P11 Stage Semantics` | `P11 extra_alignment` | 2 | `LAYER_ONLY_BASELINE` | 5 | 0.0783 +- 0.0001 | 0.1594 +- 0.0001 | 0.0328 +- 0.0007 | 0.1128 +- 0.0013 | 0.0012 +- 0.0027 | 0.0000 +- 0.0000 | -0.638 |
| `P11 Stage Semantics` | `P11 order_permutation` | 1 | `MICRO_MACRO_MID` | 5 | 0.0822 +- 0.0007 | 0.1610 +- 0.0010 | 0.0356 +- 0.0012 | 0.1187 +- 0.0035 | 0.0096 +- 0.0070 | 0.0015 +- 0.0021 | 0.297 |
| `P11 Stage Semantics` | `P11 order_permutation` | 2 | `MICRO_MID_MACRO` | 1 | 0.0825 | 0.1616 | 0.0330 | 0.1148 | 0.0000 | 0.0000 | 0.237 |
| `P11 Stage Semantics` | `P11 order_permutation` | 3 | `MID_MICRO_MACRO` | 1 | 0.0816 | 0.1621 | 0.0344 | 0.1224 | 0.0152 | 0.0000 | 0.105 |
| `P11 Stage Semantics` | `P11 routing_granularity` | 1 | `TOKEN_TOKEN_TOKEN` | 5 | 0.0816 +- 0.0005 | 0.1614 +- 0.0007 | 0.0359 +- 0.0006 | 0.1215 +- 0.0027 | 0.0130 +- 0.0048 | 0.0000 +- 0.0000 | 0.126 |
| `P11 Stage Semantics` | `P11 routing_granularity` | 2 | `TOKEN_SESSION_TOKEN` | 1 | 0.0812 | 0.1614 | 0.0357 | 0.1225 | 0.0152 | 0.0000 | -0.057 |
| `P11 Stage Semantics` | `P11 routing_granularity` | 3 | `SESSION_TOKEN_TOKEN` | 5 | 0.0815 +- 0.0002 | 0.1612 +- 0.0007 | 0.0356 +- 0.0011 | 0.1202 +- 0.0038 | 0.0105 +- 0.0078 | 0.0000 +- 0.0000 | -0.114 |
| `P12 Layout Composition` | `P12 bundle_chain` | 1 | `BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED` | 1 | 0.0775 | 0.1596 | 0.0345 | 0.1215 | 0.0051 | 0.0294 | 0.515 |
| `P12 Layout Composition` | `P12 bundle_chain` | 2 | `BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED` | 1 | 0.0780 | 0.1596 | 0.0346 | 0.1218 | 0.0034 | 0.0000 | 0.505 |
| `P12 Layout Composition` | `P12 bundle_chain` | 3 | `BUNDLE_MACROMID_THEN_MIDMICRO_MEAN` | 1 | 0.0777 | 0.1595 | 0.0351 | 0.1225 | 0.0061 | 0.0084 | 0.406 |
| `P12 Layout Composition` | `P12 bundle_pair_then_follow` | 1 | `BUNDLE_MACROMICRO_LEARNED` | 5 | 0.0808 +- 0.0001 | 0.1616 +- 0.0001 | 0.0362 +- 0.0002 | 0.1233 +- 0.0001 | 0.0053 +- 0.0016 | 0.0091 +- 0.0014 | 0.618 |
| `P12 Layout Composition` | `P12 bundle_pair_then_follow` | 2 | `BUNDLE_MACROMICRO_MEAN` | 1 | 0.0807 | 0.1617 | 0.0362 | 0.1235 | 0.0092 | 0.0059 | 0.538 |
| `P12 Layout Composition` | `P12 bundle_pair_then_follow` | 3 | `BUNDLE_MACROMICRO_SUM` | 1 | 0.0803 | 0.1615 | 0.0363 | 0.1232 | 0.0175 | 0.0084 | 0.133 |
| `P12 Layout Composition` | `P12 layout_variants` | 1 | `MACRO_REPEATED` | 1 | 0.0815 | 0.1613 | 0.0358 | 0.1225 | 0.0152 | 0.0065 | 0.504 |
| `P12 Layout Composition` | `P12 layout_variants` | 2 | `ATTN_MICRO_BEFORE` | 5 | 0.0809 +- 0.0001 | 0.1621 +- 0.0001 | 0.0353 +- 0.0004 | 0.1223 +- 0.0007 | 0.0141 +- 0.0023 | 0.0014 +- 0.0019 | 0.340 |
| `P12 Layout Composition` | `P12 layout_variants` | 3 | `MID_NOLOCALATTN` | 5 | 0.0808 +- 0.0001 | 0.1621 +- 0.0001 | 0.0353 +- 0.0006 | 0.1225 +- 0.0006 | 0.0141 +- 0.0023 | 0.0000 +- 0.0000 | 0.285 |
| `P13 Feature Sanity` | `P13 data_condition` | 1 | `FULL_DATA` | 5 | 0.0815 +- 0.0006 | 0.1615 +- 0.0005 | 0.0357 +- 0.0002 | 0.1214 +- 0.0031 | 0.0156 +- 0.0009 | 0.0000 +- 0.0000 | 0.191 |
| `P13 Feature Sanity` | `P13 data_condition` | 2 | `CATEGORY_ZERO_DATA` | 5 | 0.0812 +- 0.0001 | 0.1610 +- 0.0028 | 0.0341 +- 0.0016 | 0.1180 +- 0.0090 | 0.0136 +- 0.0034 | 0.0000 +- 0.0000 | -0.191 |
| `P13 Feature Sanity` | `P13 eval_perturb` | 1 | `EVAL_SHUFFLE_FOCUS` | 1 | 0.0807 | 0.1616 | 0.0343 | 0.1213 | 0.0173 | 0.0000 | 1.144 |
| `P13 Feature Sanity` | `P13 eval_perturb` | 2 | `EVAL_SHUFFLE_TEMPO` | 1 | 0.0805 | 0.1614 | 0.0335 | 0.1213 | 0.0169 | 0.0000 | 1.075 |
| `P13 Feature Sanity` | `P13 eval_perturb` | 3 | `EVAL_SHUFFLE_EXPOSURE` | 1 | 0.0795 | 0.1610 | 0.0310 | 0.1194 | 0.0303 | 0.0000 | 0.792 |
| `P13 Feature Sanity` | `P13 semantic_mismatch` | 1 | `FEATURE_ROLE_SWAP` | 4 | 0.0810 +- 0.0001 | 0.1617 +- 0.0010 | 0.0343 +- 0.0013 | 0.1202 +- 0.0040 | 0.0112 +- 0.0051 | 0.0000 +- 0.0000 | 0.092 |
| `P13 Feature Sanity` | `P13 semantic_mismatch` | 2 | `STAGE_MISMATCH_ASSIGN` | 5 | 0.0812 +- 0.0003 | 0.1611 +- 0.0010 | 0.0344 +- 0.0011 | 0.1187 +- 0.0041 | 0.0076 +- 0.0076 | 0.0000 +- 0.0000 | -0.060 |
| `P13 Feature Sanity` | `P13 semantic_mismatch` | 3 | `POSITION_SHIFT_FEATURE` | 1 | 0.0808 | 0.1617 | 0.0344 | 0.1213 | 0.0170 | 0.0000 | -0.071 |
| `P13 Feature Sanity` | `P13 train_corruption` | 1 | `TRAIN_PERMUTE_FOCUS` | 5 | 0.0812 +- 0.0001 | 0.1620 +- 0.0002 | 0.0353 +- 0.0005 | 0.1223 +- 0.0003 | 0.0152 +- 0.0000 | 0.0000 +- 0.0000 | 0.301 |
| `P13 Feature Sanity` | `P13 train_corruption` | 2 | `TRAIN_PERMUTE_TEMPO` | 5 | 0.0811 +- 0.0001 | 0.1619 +- 0.0002 | 0.0352 +- 0.0009 | 0.1225 +- 0.0004 | 0.0152 +- 0.0087 | 0.0000 +- 0.0000 | 0.161 |
| `P13 Feature Sanity` | `P13 train_corruption` | 3 | `TRAIN_PERMUTE_MEMORY` | 1 | 0.0810 | 0.1617 | 0.0353 | 0.1228 | 0.0101 | 0.0000 | -0.014 |

### 축별 요약(Top1 vs Top2, test/cold/short 차이)

| axis | top1 | top2 | Δtest(top1-top2) | Δcold<=5 | Δshort1-2 |
| --- | --- | --- | ---: | ---: | ---: |
| `P8 Stage A: wrapper_combo` | `ALL_W4` | `ALL_W5` | 0.0011 | 0.0045 | -0.0218 |
| `P8 Stage B: bias_mode` | `ALL_W5_BIAS_GROUP_FEAT_RULE` | `ALL_W5_BIAS_OFF` | 0.0011 | 0.0041 | -0.0392 |
| `P8 Stage C: source_profile` | `MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH` | `MIXED_2_BIAS_BOTH_SRC_ALL_BOTH` | -0.0001 | 0.0008 | 0.0000 |
| `P8 Stage D: topk_profile` | `ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE` | `MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE` | 0.0003 | -0.0003 | 0.0294 |
| `P8 Verification Base-A` | `A_H3` | `A_H1` | 0.0004 | 0.0031 | -0.0025 |
| `P8 Verification Base-B` | `B_H3` | `B_H4` | -0.0003 | 0.0009 | -0.0024 |
| `P8 Verification Base-C` | `C_H3` | `C_H4` | -0.0006 | -0.0000 | 0.0009 |
| `P8 Verification Base-D` | `D_H4` | `D_H3` | 0.0003 | 0.0003 | -0.0018 |
| `P9 C0: Natural` | `N1` | `N3` | 0.0010 | 0.0020 | -0.0123 |
| `P9 C1: CanonicalBalance` | `B2` | `B4` | 0.0009 | 0.0022 | 0.0023 |
| `P9 C2: Specialization` | `S1` | `S2` | 0.0002 | -0.0010 | 0.0037 |
| `P9 C3: FeatureAlignment` | `F4` | `F3` | -0.0000 | 0.0016 | -0.0065 |
| `P9_2 K1` | `N4` | - | - | - | - |
| `P9_2 K2` | `B1` | - | - | - | - |
| `P9_2 K3` | `S3` | - | - | - | - |
| `P10 availability` | `NO_TIMESTAMP` | `NO_CATEGORY` | 0.0003 | 0.0021 | -0.0028 |
| `P10 availability_plus` | `NO_CATEGORY_NO_TIMESTAMP` | - | - | - | - |
| `P10 compactness` | `TOP2_PER_GROUP` | `COMMON_TEMPLATE` | 0.0002 | 0.0006 | 0.0003 |
| `P10 compactness_plus` | `COMMON_TEMPLATE_NO_CATEGORY` | - | - | - | - |
| `P10 feature_subset` | `FULL` | `Tempo_Focus_Exposure` | -0.0003 | -0.0004 | 0.0000 |
| `P10 stochastic` | `FEATURE_DROPOUT` | `FAMILY_DROPOUT` | 0.0001 | -0.0000 | -0.0024 |
| `P11 base_ablation` | `MID_MICRO` | `MACRO_MID_MICRO` | 0.0010 | 0.0019 | 0.0000 |
| `P11 extra_alignment` | `LAYER2_MACRO_MID_MICRO` | `LAYER_ONLY_BASELINE` | 0.0017 | 0.0040 | 0.0000 |
| `P11 order_permutation` | `MICRO_MACRO_MID` | `MICRO_MID_MACRO` | -0.0006 | 0.0039 | 0.0015 |
| `P11 routing_granularity` | `TOKEN_TOKEN_TOKEN` | `TOKEN_SESSION_TOKEN` | -0.0000 | -0.0010 | 0.0000 |
| `P12 bundle_chain` | `BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED` | `BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED` | 0.0000 | -0.0003 | 0.0294 |
| `P12 bundle_pair_then_follow` | `BUNDLE_MACROMICRO_LEARNED` | `BUNDLE_MACROMICRO_MEAN` | -0.0001 | -0.0002 | 0.0032 |
| `P12 layout_variants` | `MACRO_REPEATED` | `ATTN_MICRO_BEFORE` | -0.0008 | 0.0002 | 0.0051 |
| `P13 data_condition` | `FULL_DATA` | `CATEGORY_ZERO_DATA` | 0.0005 | 0.0034 | 0.0000 |
| `P13 eval_perturb` | `EVAL_SHUFFLE_FOCUS` | `EVAL_SHUFFLE_TEMPO` | 0.0002 | 0.0000 | 0.0000 |
| `P13 semantic_mismatch` | `FEATURE_ROLE_SWAP` | `STAGE_MISMATCH_ASSIGN` | 0.0006 | 0.0015 | 0.0000 |
| `P13 train_corruption` | `TRAIN_PERMUTE_FOCUS` | `TRAIN_PERMUTE_TEMPO` | 0.0002 | -0.0002 | 0.0000 |

## Diag를 Metric 축으로 보기
### 1) 전체/국소 correlation

| metric | raw rho(test) | raw rho(valid) | centered rho(test) | centered rho(valid) | local median rho(test) | local median rho(valid) | opposite sign(|rho|>=0.3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cv` | -0.6100 | 0.2171 | -0.6576 | 0.3519 | -0.6412 | 0.4332 | 15/18 |
| `ent` | 0.4657 | -0.0649 | 0.6373 | -0.2610 | 0.6924 | -0.5820 | 15/18 |
| `intra_nn` | 0.5599 | -0.2680 | 0.6244 | -0.3973 | 0.6103 | -0.5202 | 17/19 |
| `n_eff` | 0.4897 | -0.0361 | 0.6193 | -0.2419 | 0.6412 | -0.4332 | 15/19 |
| `top1` | -0.2969 | 0.0703 | -0.4274 | 0.1363 | -0.4756 | 0.2073 | 11/16 |
| `expert_nn` | 0.4026 | -0.1683 | 0.4093 | -0.2842 | 0.4636 | -0.4514 | 13/14 |
| `group_nn` | 0.2658 | -0.1597 | 0.2801 | -0.2886 | 0.2589 | -0.3015 | 6/7 |
| `feat_nn` | 0.1458 | -0.1632 | 0.2098 | -0.2753 | 0.2390 | -0.3446 | 3/3 |
| `focus_nn` | 0.1987 | -0.1014 | 0.2012 | -0.0949 | 0.2857 | -0.1908 | 3/4 |
| `exposure_nn` | 0.1523 | 0.1906 | 0.1870 | 0.0635 | 0.3035 | -0.1689 | 4/5 |
| `tempo_nn` | 0.1310 | -0.1201 | 0.1639 | -0.1955 | 0.1913 | -0.2347 | 4/4 |
| `memory_nn` | 0.1959 | 0.1536 | 0.1475 | 0.0601 | 0.1652 | -0.3164 | 3/3 |

### 2) Phase별 Diag 프로파일 (metric 가로축)

| phase | n_eff | ent | top1 | intra_nn | expert_nn | group_nn | feat_nn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `P8` | 9.6441 | 2.0679 | 0.3823 | 0.9913 | 0.9906 | 0.9972 | - |
| `P9` | 8.7288 | 1.8460 | 0.4173 | 0.9924 | 0.9882 | 0.9938 | - |
| `P9_2` | 9.6962 | 1.9378 | 0.3182 | 0.9930 | 0.9887 | 0.9945 | - |
| `P10` | 9.1683 | 2.0101 | 0.3427 | 0.9976 | 0.9958 | 0.9981 | 0.9845 |
| `P11` | 10.2694 | 2.0413 | 0.3197 | 0.9936 | 0.9895 | 0.9954 | 0.9713 |
| `P12` | 10.7641 | 2.0332 | 0.2779 | 0.9924 | 0.9859 | 0.9937 | 0.9648 |
| `P13` | 10.9290 | 2.2154 | 0.3742 | 0.9966 | 0.9929 | 0.9962 | 0.9799 |

### 3) 세팅 변화가 Diag에 주는 영향 (축내 test 상위25% vs 하위25%)

| metric | mean Δ(top-bottom) | positive axes | negative axes | n_axes |
| --- | ---: | ---: | ---: | ---: |
| `n_eff` | 3.0434 | 22 | 2 | 24 |
| `cv` | -0.5982 | 2 | 22 | 24 |
| `ent` | 0.3964 | 20 | 4 | 24 |
| `top1` | -0.1920 | 1 | 23 | 24 |
| `focus_nn` | 0.0328 | 6 | 4 | 10 |
| `memory_nn` | 0.0074 | 7 | 3 | 10 |
| `intra_nn` | 0.0050 | 22 | 2 | 24 |
| `expert_nn` | 0.0033 | 20 | 4 | 24 |
| `feat_nn` | 0.0025 | 7 | 3 | 10 |
| `tempo_nn` | 0.0020 | 8 | 2 | 10 |
| `exposure_nn` | 0.0015 | 4 | 5 | 10 |
| `group_nn` | 0.0013 | 18 | 6 | 24 |

### 4) 동일 축/동일 세팅(주로 hparam/seed 차이) 통제 correlation

통제 그룹: `55` groups

| metric | rho_test(controlled) | rho_valid(controlled) | n |
| --- | ---: | ---: | ---: |
| `n_eff` | 0.7719 | -0.3337 | 299 |
| `top1` | -0.6337 | 0.0982 | 299 |
| `ent` | 0.7607 | -0.4722 | 299 |
| `intra_nn` | 0.7379 | -0.3820 | 299 |
| `group_nn` | 0.1793 | -0.3771 | 299 |
| `feat_nn` | 0.3500 | -0.2649 | 188 |
| `expert_nn` | 0.4312 | -0.3685 | 299 |

## Valid/Test 반대 신호 해석 (왜 자주 뒤집히는가)
- centered corr(test_cold_special, valid): `-0.1845`
- centered corr(test_cold_special, test): `0.8078`
- centered corr(test_short_special, valid): `0.0542`
- centered corr(test_short_special, test): `0.0553`
- 관찰상 `cold`는 test 일반화와 강하게 연결되고, valid와는 반대 또는 약한 경우가 많음
- 따라서 selection을 valid 단일 지표로 고정하면, test/cold에서의 이득을 놓치거나 반대로 갈 수 있음
- 결론은 인과 단정이 아니라, `같은 축/유사 세팅 내에서 일관된 경향`으로 서술하는 것이 안전

## 최종 실무 가이드 (축별)
- 각 축에서 top1만 고정하지 말고 top2~3을 같이 검증하는 게 안전
- 특히 Δtest가 작고 Δcold/Δshort가 엇갈리는 축은 top2까지 유지

| axis | 추천 후보(top3) | 코멘트 |
| --- | --- | --- |
| `P8 Stage A: wrapper_combo` | `ALL_W4`, `ALL_W5`, `MIXED_3` | test=0.1613, cold=0.1213, short=0.0076 |
| `P8 Stage B: bias_mode` | `ALL_W5_BIAS_GROUP_FEAT_RULE`, `ALL_W5_BIAS_OFF`, `MIXED_1_BIAS_GROUP_FEAT` | test=0.1617, cold=0.1194, short=0.0196 |
| `P8 Stage C: source_profile` | `MIXED_2_BIAS_GROUP_FEAT_SRC_ALL_BOTH`, `MIXED_2_BIAS_BOTH_SRC_ALL_BOTH`, `MIXED_2_BIAS_GROUP_FEAT_SRC_A_HIDDEN_B_D_FEATURE` | test=0.1617, cold=0.1219, short=0.0000 |
| `P8 Stage D: topk_profile` | `ALL_W5_BIAS_RULE_SRC_ABC_FEATURE_TK_DENSE`, `MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_DENSE`, `MIXED_2_BIAS_BOTH_SRC_ABC_FEATURE_TK_D1_FINAL4` | test=0.1619, cold=0.1200, short=0.0294 |
| `P8 Verification Base-A` | `A_H3`, `A_H1`, `A_H4` | test=0.1612, cold=0.1208, short=0.0000 |
| `P8 Verification Base-B` | `B_H3`, `B_H4`, `B_H2` | test=0.1610, cold=0.1194, short=0.0009 |
| `P8 Verification Base-C` | `C_H3`, `C_H4`, `C_H1` | test=0.1610, cold=0.1190, short=0.0009 |
| `P8 Verification Base-D` | `D_H4`, `D_H3`, `D_H2` | test=0.1613, cold=0.1182, short=0.0029 |
| `P9 C0: Natural` | `N1`, `N3`, `N2` | test=0.1611, cold=0.1189, short=0.0074 |
| `P9 C1: CanonicalBalance` | `B2`, `B4`, `B3` | test=0.1613, cold=0.1183, short=0.0033 |
| `P9 C2: Specialization` | `S1`, `S2`, `S4` | test=0.1614, cold=0.1189, short=0.0037 |
| `P9 C3: FeatureAlignment` | `F4`, `F3`, `F2` | test=0.1608, cold=0.1186, short=0.0033 |
| `P9_2 K1` | `N4` | test=0.1614, cold=0.1207, short=0.0002 |
| `P9_2 K2` | `B1` | test=0.1609, cold=0.1182, short=0.0029 |
| `P9_2 K3` | `S3` | test=0.1611, cold=0.1190, short=0.0003 |
| `P10 availability` | `NO_TIMESTAMP`, `NO_CATEGORY` | test=0.1617, cold=0.1212, short=0.0000 |
| `P10 availability_plus` | `NO_CATEGORY_NO_TIMESTAMP` | test=0.1614, cold=0.1212, short=0.0000 |
| `P10 compactness` | `TOP2_PER_GROUP`, `COMMON_TEMPLATE`, `TOP1_PER_GROUP` | test=0.1618, cold=0.1210, short=0.0044 |
| `P10 compactness_plus` | `COMMON_TEMPLATE_NO_CATEGORY` | test=0.1619, cold=0.1209, short=0.0000 |
| `P10 feature_subset` | `FULL`, `Tempo_Focus_Exposure`, `Tempo_Memory` | test=0.1615, cold=0.1216, short=0.0000 |
| `P10 stochastic` | `FEATURE_DROPOUT`, `FAMILY_DROPOUT` | test=0.1619, cold=0.1211, short=0.0004 |
| `P11 base_ablation` | `MID_MICRO`, `MACRO_MID_MICRO`, `MACRO_MID` | test=0.1617, cold=0.1218, short=0.0000 |
| `P11 extra_alignment` | `LAYER2_MACRO_MID_MICRO`, `LAYER_ONLY_BASELINE` | test=0.1611, cold=0.1168, short=0.0000 |
| `P11 order_permutation` | `MICRO_MACRO_MID`, `MICRO_MID_MACRO`, `MID_MICRO_MACRO` | test=0.1610, cold=0.1187, short=0.0015 |
| `P11 routing_granularity` | `TOKEN_TOKEN_TOKEN`, `TOKEN_SESSION_TOKEN`, `SESSION_TOKEN_TOKEN` | test=0.1614, cold=0.1215, short=0.0000 |
| `P12 bundle_chain` | `BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED`, `BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED`, `BUNDLE_MACROMID_THEN_MIDMICRO_MEAN` | test=0.1596, cold=0.1215, short=0.0294 |
| `P12 bundle_pair_then_follow` | `BUNDLE_MACROMICRO_LEARNED`, `BUNDLE_MACROMICRO_MEAN`, `BUNDLE_MACROMICRO_SUM` | test=0.1616, cold=0.1233, short=0.0091 |
| `P12 layout_variants` | `MACRO_REPEATED`, `ATTN_MICRO_BEFORE`, `MID_NOLOCALATTN` | test=0.1613, cold=0.1225, short=0.0065 |
| `P13 data_condition` | `FULL_DATA`, `CATEGORY_ZERO_DATA` | test=0.1615, cold=0.1214, short=0.0000 |
| `P13 eval_perturb` | `EVAL_SHUFFLE_FOCUS`, `EVAL_SHUFFLE_TEMPO`, `EVAL_SHUFFLE_EXPOSURE` | test=0.1616, cold=0.1213, short=0.0000 |
| `P13 semantic_mismatch` | `FEATURE_ROLE_SWAP`, `STAGE_MISMATCH_ASSIGN`, `POSITION_SHIFT_FEATURE` | test=0.1617, cold=0.1202, short=0.0000 |
| `P13 train_corruption` | `TRAIN_PERMUTE_FOCUS`, `TRAIN_PERMUTE_TEMPO`, `TRAIN_PERMUTE_MEMORY` | test=0.1620, cold=0.1223, short=0.0000 |

## 구조 변경 추천 (1~3개로 압축)
### 추천 1) `일반화/Cold 우선` (기본 메인라인)
- 변경 포인트:
`P10 availability=NO_TIMESTAMP`, `P10 compactness=TOP2_PER_GROUP`, `P10 stochastic=FEATURE_DROPOUT`, `P11 routing_granularity=TOKEN_TOKEN_TOKEN`, `P12 bundle_pair_then_follow=BUNDLE_MACROMICRO_LEARNED`
- 왜 이 조합이 1순위인가:
`test/cold`가 안정적으로 높고, 특이 slice(특히 cold<=5)에서 손실이 작음.
- 근거:
`NO_TIMESTAMP(test=0.1617, cold=0.1212)`, `TOP2_PER_GROUP(test=0.1618)`, `FEATURE_DROPOUT(test=0.1619)`, `TOKEN_TOKEN_TOKEN(test=0.1614, cold=0.1215)`, `BUNDLE_MACROMICRO_LEARNED(test=0.1616, cold=0.1233)`.
- diag 근거:
test와 양(+)인 `n_eff/ent/intra_nn`가 높아지는 방향이 유리 (`centered rho(test): n_eff=0.6193, ent=0.6373, intra_nn=0.6244`), 과집중(`top1`)은 불리(`-0.4274`).

### 추천 2) `Valid 상단 유지 + Test 타협형` (논문 main 결과형)
- 변경 포인트:
`P10 feature_subset=FULL`, `P10 compactness=COMMON_TEMPLATE`, `P11 order_permutation=MICRO_MACRO_MID`, `P11 base_ablation=MID_MICRO`, `P12 layout_variants=MACRO_REPEATED`
- 왜 이 조합이 2순위인가:
valid 상단 후보를 유지하면서 test를 크게 깨지 않는 균형형.
- 근거:
`FULL(valid=0.0811, test=0.1615)`, `MICRO_MACRO_MID(valid=0.0822, test=0.1610)`, `MID_MICRO(test=0.1617, cold=0.1218)`, `MACRO_REPEATED(valid=0.0815, test=0.1613)`.
- special/diag 관점:
valid만 극대화하면 test/cold와 충돌 가능성이 커서, `MID_MICRO + MACRO_REPEATED` 같이 test 안정 축을 함께 묶는 것이 안전.

### 추천 3) `Short session 특화` (보조 트랙)
- 변경 포인트:
`P12 bundle_chain=BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED`를 중심으로, 나머지는 추천 1의 안정 축을 결합.
- 왜 3순위인가:
전체 test는 낮지만 short 1-2에서 이득이 가장 크다.
- 근거:
`BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED(test_short_1-2=0.0294)`로 short slice 강점이 명확.
- 주의:
같은 setting의 전체 test는 `0.1596`으로 낮아 mainline 단독 채택은 비권장, short 특화 실험용으로 유지.

## 바로 가져다 쓸 최종 조합 (2~3개)
아래는 축별 setting을 그대로 붙여서 실행 가능한 형태의 추천 recipe입니다.

| 최종안 | 목적 | 축별 setting recipe | 채택 근거(valid/test/special/diag) |
| --- | --- | --- | --- |
| `FINAL-A (Main Default)` | 전체 일반화 + cold 안정 | `P10: NO_TIMESTAMP + TOP2_PER_GROUP + FEATURE_DROPOUT + FULL` ; `P11: TOKEN_TOKEN_TOKEN + MID_MICRO` ; `P12: BUNDLE_MACROMICRO_LEARNED + MACRO_REPEATED` | test 상단(0.1614~0.1619 구간) + cold 강점(0.1212~0.1233). diag도 `n_eff/ent/intra_nn` 양의 방향과 정합. |
| `FINAL-B (Valid-High Balanced)` | valid 상단 유지 + test 손실 최소화 | `P10: FULL + COMMON_TEMPLATE` ; `P11: MICRO_MACRO_MID + MID_MICRO` ; `P12: MACRO_REPEATED` ; `P13 sanity check: TRAIN_PERMUTE_FOCUS` | valid 상위 축을 보존(`MICRO_MACRO_MID valid=0.0822`)하면서 `MID_MICRO/P12`로 test 안전장치. train_corruption 축에서 `TRAIN_PERMUTE_FOCUS(test=0.1620)`도 우호적. |
| `FINAL-C (Short Specialist)` | short 1-2 특화 | `P10: NO_TIMESTAMP + FEATURE_DROPOUT` ; `P11: TOKEN_TOKEN_TOKEN` ; `P12: BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED` | short slice 최우선일 때 사용(`test_short_1-2=0.0294`). 다만 전체 test 하락 리스크(`0.1596`)가 있어 보조 트랙 권장. |

### 선택 가이드
- 논문 메인 1개만 고르면: `FINAL-A`
- valid 강조가 필요하면: `FINAL-B`를 같이 보고, test/cold 하락 여부를 반드시 동시 체크
- short 서비스 시나리오가 명확하면: `FINAL-C`를 분리 배치
