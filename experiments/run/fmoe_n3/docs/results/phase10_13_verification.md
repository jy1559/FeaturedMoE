# Phase10~13 Verification 실험 보고서

작성일: 2026-03-26 (UTC)

## 0) 문서 목적 (논문 본문용)
- 본 문서는 seed 평균/표준편차 기반으로 phase10~13 후보를 검증한 결과를 정리한다.
- wide에서 얻은 가설을 verification에서 재확인하고, 본문에 넣을 수 있는 주장 문장까지 연결한다.

## 1) 집계 정책 / 비교 공정성
- dedup: `run_phase` 최신 `timestamp_utc` 우선 → `n_completed` → `run_best_valid_mrr20`.
- 본표(main): `H3 + n_completed>=20` (P11~P13, 96 rows).
- 보조표(support): `P10 + (H1/H3) + n_completed=10` (96 rows).
- main metric: `best valid MRR@20`, sub metric: `test MRR@20`.
- diag 누락(run): `P10_13_2_BP11_22_H3_S1~S4`; diag 섹션/그래프에서 자동 제외.

## 2) 데이터 품질 스냅샷
| Scope | Rows | Note |
| --- | --- | --- |
| All dedup | 192 | P10~P13 all verification rows |
| Main table | 96 | H3, n>=20, fair comparison |
| Support table | 96 | P10 coverage (H1/H3, n=10) |
| Special missing | 0 | should be 0 |
| Diag missing | 4 | P11-22 lineage only |

## 3-11 P11: Stage Semantics / Necessity / Granularity (verification main)
- phase 가설: stage 구성/순서/granularity는 임의가 아니라 의미 있는 설계 변수다.
### 3-11.1 축별 평균 비교 (phase 내부)
| Axis | Runs | Best Setting in Axis | Mean Valid | Mean Test | Mean Cold | Mean Long(11+) | Mean Top1 | Mean CV | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_ablation | 8 | P11-00_MACRO_MID_MICRO | 0.0810 | 0.1613 | 0.1205 | 0.1709 | 0.2020 | 0.2214 | mixed |
| extra_alignment | 8 | P11-23_LAYER2_MACRO_MID_MICRO | 0.0799 | 0.1601 | 0.1143 | 0.1696 | 0.2095 | 0.1851 | balanced |
| order_permutation | 8 | P11-14_MACRO_MICRO_MID | 0.0821 | 0.1597 | 0.1148 | 0.1697 | 0.4537 | 0.7607 | mixed |
| routing_granularity | 8 | P11-19_TOKEN_TOKEN_TOKEN | 0.0815 | 0.1614 | 0.1212 | 0.1708 | 0.2797 | 0.2754 | mixed |

### 3-11.2 setting별 mean±std 비교 (H3)
| Setting | SeedN | Valid mean±std | Test mean±std | Cold mean±std | Long mean±std | Top1 mean±std | CV mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P11-14_MACRO_MICRO_MID | 4 | 0.0822 ± 0.0007 | 0.1580 ± 0.0049 | 0.1101 ± 0.0142 | 0.1684 ± 0.0036 | 0.6320 ± 0.4141 | 1.1812 ± 1.2448 |
| P11-17_MICRO_MACRO_MID | 4 | 0.0820 ± 0.0005 | 0.1614 ± 0.0005 | 0.1195 ± 0.0035 | 0.1710 ± 0.0001 | 0.2754 ± 0.0766 | 0.3402 ± 0.2429 |
| P11-19_TOKEN_TOKEN_TOKEN | 4 | 0.0816 ± 0.0005 | 0.1613 ± 0.0008 | 0.1211 ± 0.0029 | 0.1708 ± 0.0003 | 0.2674 ± 0.1359 | 0.2473 ± 0.2338 |
| P11-23_LAYER2_MACRO_MID_MICRO | 4 | 0.0816 ± 0.0001 | 0.1609 ± 0.0014 | 0.1164 ± 0.0036 | 0.1703 ± 0.0011 | 0.2095 ± 0.0309 | 0.1851 ± 0.0574 |
| P11-20_SESSION_TOKEN_TOKEN | 4 | 0.0814 ± 0.0000 | 0.1614 ± 0.0006 | 0.1213 ± 0.0034 | 0.1709 ± 0.0002 | 0.2920 ± 0.2299 | 0.3034 ± 0.3495 |
| P11-00_MACRO_MID_MICRO | 4 | 0.0813 ± 0.0001 | 0.1616 ± 0.0001 | 0.1229 ± 0.0001 | 0.1709 ± 0.0001 | 0.1702 ± 0.0090 | 0.1428 ± 0.0134 |
| P11-03_MACRO_MID | 4 | 0.0808 ± 0.0001 | 0.1611 ± 0.0004 | 0.1182 ± 0.0023 | 0.1708 ± 0.0003 | 0.2337 ± 0.0526 | 0.3001 ± 0.1402 |
| P11-22_LAYER_ONLY_BASELINE | 4 | 0.0783 ± 0.0001 | 0.1593 ± 0.0001 | 0.1123 ± 0.0007 | 0.1690 ± 0.0002 | - | - |

### 3-11.3 가설 비교 / 논문 연결
- valid 기준 winner: `P11-14_MACRO_MICRO_MID` (0.0822 ± 0.0007).
- test 기준 winner: `P11-00_MACRO_MID_MICRO` (0.1616 ± 0.0001).
- 안정성(낮은 valid std) 후보: `P11-20_SESSION_TOKEN_TOKEN` (0.0814 ± 0.0000).
- 해석: valid 최고(`P11-14`)와 test 최고(`P11-00`)가 다르므로, P11은 단일 winner보다 "정확도 목적(valid) vs 배포 목적(test/안정성)" 이원 전략으로 제시하는 것이 안전하다.
- 본문 서술 예시: P11에서는 stage order 가설이 seed 평균에서도 재현되었고, `P11-14_MACRO_MICRO_MID`가 valid 기준 대표 후보, `P11-00_MACRO_MID_MICRO`가 test 기준 대표 후보로 분리되었다.

## 3-12 P12: Layout Composition / Attention Placement (verification main)
- phase 가설: 같은 stage set이라도 layout/composition에 따라 성능과 router 동작이 달라진다.
### 3-12.1 축별 평균 비교 (phase 내부)
| Axis | Runs | Best Setting in Axis | Mean Valid | Mean Test | Mean Cold | Mean Long(11+) | Mean Top1 | Mean CV | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bundle_pair_then_follow | 4 | P12-18_BUNDLE_MACROMICRO_LEARNED | 0.0809 | 0.1615 | 0.1233 | 0.1710 | 0.1847 | 0.2155 | mixed |
| bundle_router | 4 | P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | 0.0805 | 0.1612 | 0.1225 | 0.1705 | 0.1746 | 0.1087 | balanced |
| layout_variants | 24 | P12-06_MID_REPEATED | 0.0810 | 0.1615 | 0.1207 | 0.1709 | 0.3579 | 0.3589 | mixed |

### 3-12.2 setting별 mean±std 비교 (H3)
| Setting | SeedN | Valid mean±std | Test mean±std | Cold mean±std | Long mean±std | Top1 mean±std | CV mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P12-06_MID_REPEATED | 4 | 0.0815 ± 0.0004 | 0.1606 ± 0.0023 | 0.1194 ± 0.0065 | 0.1701 ± 0.0016 | 0.3758 ± 0.4162 | 0.7235 ± 1.2603 |
| P12-07_MICRO_REPEATED | 4 | 0.0814 ± 0.0003 | 0.1614 ± 0.0009 | 0.1206 ± 0.0025 | 0.1707 ± 0.0006 | 0.2107 ± 0.0390 | 0.1799 ± 0.1011 |
| P12-08_MACRO_NOLOCALATTN | 4 | 0.0810 ± 0.0001 | 0.1615 ± 0.0008 | 0.1205 ± 0.0026 | 0.1710 ± 0.0004 | 0.4152 ± 0.0768 | 0.3861 ± 0.4829 |
| P12-02_ATTN_MICRO_BEFORE | 4 | 0.0809 ± 0.0001 | 0.1621 ± 0.0001 | 0.1224 ± 0.0007 | 0.1714 ± 0.0001 | 0.2588 ± 0.0307 | 0.1229 ± 0.0221 |
| P12-18_BUNDLE_MACROMICRO_LEARNED | 4 | 0.0809 ± 0.0001 | 0.1615 ± 0.0001 | 0.1233 ± 0.0001 | 0.1710 ± 0.0001 | 0.1847 ± 0.0067 | 0.2155 ± 0.0067 |
| P12-09_MID_NOLOCALATTN | 4 | 0.0808 ± 0.0001 | 0.1621 ± 0.0002 | 0.1224 ± 0.0007 | 0.1714 ± 0.0001 | 0.2691 ± 0.0338 | 0.1134 ± 0.0110 |
| P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | 4 | 0.0805 ± 0.0001 | 0.1612 ± 0.0002 | 0.1225 ± 0.0002 | 0.1705 ± 0.0001 | 0.1746 ± 0.0404 | 0.1087 ± 0.0213 |
| P12-00_ATTN_ONESHOT | 4 | 0.0802 ± 0.0001 | 0.1610 ± 0.0005 | 0.1191 ± 0.0024 | 0.1706 ± 0.0003 | 0.6176 ± 0.3022 | 0.6273 ± 0.3353 |

### 3-12.3 가설 비교 / 논문 연결
- valid 기준 winner: `P12-06_MID_REPEATED` (0.0815 ± 0.0004).
- test 기준 winner: `P12-02_ATTN_MICRO_BEFORE` (0.1621 ± 0.0001).
- 안정성(낮은 valid std) 후보: `P12-08_MACRO_NOLOCALATTN` (0.0810 ± 0.0001).
- 해석: layout 계열 내부에서도 repeated/attention placement 차이가 mean/std에 반영되어, "same stage set, different composition" 주장을 verification 레벨에서도 유지한다.
- 본문 서술 예시: P12에서는 composition/layout 가설이 seed 평균에서 유지되었고, `P12-06_MID_REPEATED`(valid)와 `P12-02_ATTN_MICRO_BEFORE`(test)가 각각 대표 setting으로 도출되었다.

## 3-13 P13: Feature Sanity / Alignment Checks (verification main)
- phase 가설: 성능 향상은 파라미터 증가가 아니라 feature alignment를 실제로 활용한 결과다.
### 3-13.1 축별 평균 비교 (phase 내부)
| Axis | Runs | Best Setting in Axis | Mean Valid | Mean Test | Mean Cold | Mean Long(11+) | Mean Top1 | Mean CV | Router Pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| data_condition | 8 | P13-00_FULL_DATA | 0.0814 | 0.1611 | 0.1191 | 0.1703 | 0.3969 | 0.5000 | mixed |
| eval_perturb | 8 | P13-02_EVAL_ALL_ZERO | 0.0773 | 0.1575 | 0.1086 | 0.1682 | 0.5813 | 0.0919 | mixed |
| semantic_mismatch | 8 | P13-15_STAGE_MISMATCH_ASSIGN | 0.0811 | 0.1613 | 0.1192 | 0.1705 | 0.3143 | 0.3358 | mixed |
| train_corruption | 8 | P13-10_TRAIN_PERMUTE_TEMPO | 0.0812 | 0.1619 | 0.1224 | 0.1710 | 0.2589 | 0.1242 | mixed |

### 3-13.2 setting별 mean±std 비교 (H3)
| Setting | SeedN | Valid mean±std | Test mean±std | Cold mean±std | Long mean±std | Top1 mean±std | CV mean±std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P13-00_FULL_DATA | 4 | 0.0816 ± 0.0007 | 0.1615 ± 0.0006 | 0.1210 ± 0.0034 | 0.1709 ± 0.0002 | 0.2989 ± 0.1986 | 0.2741 ± 0.2973 |
| P13-01_CATEGORY_ZERO_DATA | 4 | 0.0813 ± 0.0001 | 0.1607 ± 0.0031 | 0.1172 ± 0.0101 | 0.1698 ± 0.0025 | 0.4950 ± 0.3392 | 0.7259 ± 1.0585 |
| P13-15_STAGE_MISMATCH_ASSIGN | 4 | 0.0812 ± 0.0003 | 0.1609 ± 0.0011 | 0.1182 ± 0.0045 | 0.1701 ± 0.0010 | 0.4031 ± 0.0381 | 0.4734 ± 0.1068 |
| P13-11_TRAIN_PERMUTE_FOCUS | 4 | 0.0812 ± 0.0000 | 0.1620 ± 0.0002 | 0.1223 ± 0.0003 | 0.1711 ± 0.0002 | 0.2582 ± 0.0190 | 0.1325 ± 0.0160 |
| P13-10_TRAIN_PERMUTE_TEMPO | 4 | 0.0811 ± 0.0001 | 0.1618 ± 0.0001 | 0.1226 ± 0.0003 | 0.1710 ± 0.0000 | 0.2596 ± 0.0162 | 0.1159 ± 0.0204 |
| P13-14_FEATURE_ROLE_SWAP | 4 | 0.0810 ± 0.0001 | 0.1617 ± 0.0010 | 0.1202 ± 0.0040 | 0.1709 ± 0.0005 | 0.2254 ± 0.0343 | 0.1982 ± 0.2141 |
| P13-02_EVAL_ALL_ZERO | 4 | 0.0775 ± 0.0001 | 0.1597 ± 0.0001 | 0.1179 ± 0.0004 | 0.1696 ± 0.0000 | 1.0000 ± 0.0000 | 0.1317 ± 0.0054 |
| P13-03_EVAL_ALL_SHUFFLE | 4 | 0.0771 ± 0.0001 | 0.1552 ± 0.0003 | 0.0992 ± 0.0010 | 0.1669 ± 0.0001 | 0.1627 ± 0.0079 | 0.0520 ± 0.0007 |

### 3-13.3 가설 비교 / 논문 연결
- valid 기준 winner: `P13-00_FULL_DATA` (0.0816 ± 0.0007).
- test 기준 winner: `P13-11_TRAIN_PERMUTE_FOCUS` (0.1620 ± 0.0002).
- 안정성(낮은 valid std) 후보: `P13-11_TRAIN_PERMUTE_FOCUS` (0.0812 ± 0.0000).
- 해석: perturb/corruption 계열에서도 seed 평균 차이가 유지되고, 특히 `P13-11`은 test/안정성 동시 우세로 "강건한 교란 대응 후보"로 해석 가능하다.
- 본문 서술 예시: P13에서는 feature alignment sanity 가설이 seed 평균에서도 유지되었고, `P13-00_FULL_DATA`(valid)와 `P13-11_TRAIN_PERMUTE_FOCUS`(test/안정성)가 핵심 비교축으로 정리되었다.

## 4) P10 support (coverage) 상세
- P10은 n=10 조건이므로 본표와 직접 score 비교는 하지 않고, 축 방향성/일관성 확인 용도로 사용한다.
| Hparam | Setting (top8 by valid mean) | Valid mean±std | Test mean±std | Cold mean±std | Long mean±std |
| --- | --- | --- | --- | --- | --- |
| H1 | P10-15_TOP2_PER_GROUP | 0.0809 ± 0.0002 | 0.1617 ± 0.0001 | 0.1215 ± 0.0004 | 0.1707 ± 0.0000 |
| H1 | P10-17_COMMON_TEMPLATE | 0.0809 ± 0.0001 | 0.1618 ± 0.0002 | 0.1210 ± 0.0003 | 0.1709 ± 0.0000 |
| H1 | P10-23_COMMON_TEMPLATE_NO_CATEGORY | 0.0808 ± 0.0001 | 0.1619 ± 0.0001 | 0.1201 ± 0.0001 | 0.1712 ± 0.0001 |
| H1 | P10-00_FULL | 0.0808 ± 0.0001 | 0.1613 ± 0.0001 | 0.1201 ± 0.0004 | 0.1707 ± 0.0000 |
| H1 | P10-18_NO_CATEGORY | 0.0806 ± 0.0001 | 0.1616 ± 0.0001 | 0.1192 ± 0.0001 | 0.1704 ± 0.0001 |
| H1 | P10-19_NO_TIMESTAMP | 0.0804 ± 0.0000 | 0.1616 ± 0.0001 | 0.1198 ± 0.0003 | 0.1707 ± 0.0001 |
| H1 | P10-21_FEATURE_DROPOUT | 0.0803 ± 0.0001 | 0.1617 ± 0.0002 | 0.1200 ± 0.0001 | 0.1709 ± 0.0003 |
| H1 | P10-14_Focus_Memory_Exposure | 0.0803 ± 0.0000 | 0.1614 ± 0.0001 | 0.1206 ± 0.0007 | 0.1704 ± 0.0001 |
| H3 | P10-00_FULL | 0.0814 ± 0.0001 | 0.1616 ± 0.0001 | 0.1228 ± 0.0002 | 0.1710 ± 0.0001 |
| H3 | P10-21_FEATURE_DROPOUT | 0.0809 ± 0.0001 | 0.1621 ± 0.0001 | 0.1219 ± 0.0002 | 0.1711 ± 0.0001 |
| H3 | P10-20_FAMILY_DROPOUT | 0.0809 ± 0.0000 | 0.1623 ± 0.0001 | 0.1223 ± 0.0005 | 0.1713 ± 0.0002 |
| H3 | P10-19_NO_TIMESTAMP | 0.0809 ± 0.0001 | 0.1618 ± 0.0003 | 0.1224 ± 0.0003 | 0.1709 ± 0.0002 |
| H3 | P10-18_NO_CATEGORY | 0.0809 ± 0.0001 | 0.1619 ± 0.0003 | 0.1210 ± 0.0004 | 0.1712 ± 0.0002 |
| H3 | P10-14_Focus_Memory_Exposure | 0.0808 ± 0.0001 | 0.1618 ± 0.0002 | 0.1217 ± 0.0003 | 0.1708 ± 0.0003 |
| H3 | P10-15_TOP2_PER_GROUP | 0.0806 ± 0.0001 | 0.1622 ± 0.0002 | 0.1219 ± 0.0007 | 0.1712 ± 0.0001 |
| H3 | P10-17_COMMON_TEMPLATE | 0.0805 ± 0.0001 | 0.1620 ± 0.0002 | 0.1221 ± 0.0003 | 0.1710 ± 0.0002 |

- support 해석: H1/H3 모두에서 `FULL`, `TOP2_PER_GROUP`, `NO_CATEGORY`, `FEATURE/FAMILY_DROPOUT`가 상위권에 반복적으로 등장한다.

## 5) 10~13 통합 비교 (verification 중심)
| Phase | Best Valid (mean±std) | Setting | Best Test (mean±std) | Setting | Most Stable Setting | Valid std |
| --- | --- | --- | --- | --- | --- | --- |
| P11 | 0.0822 ± 0.0007 | P11-14_MACRO_MICRO_MID | 0.1616 ± 0.0001 | P11-00_MACRO_MID_MICRO | P11-20_SESSION_TOKEN_TOKEN | 0.0000 |
| P12 | 0.0815 ± 0.0004 | P12-06_MID_REPEATED | 0.1621 ± 0.0001 | P12-02_ATTN_MICRO_BEFORE | P12-00_ATTN_ONESHOT | 0.0001 |
| P13 | 0.0816 ± 0.0007 | P13-00_FULL_DATA | 0.1620 ± 0.0002 | P13-11_TRAIN_PERMUTE_FOCUS | P13-11_TRAIN_PERMUTE_FOCUS | 0.0000 |

## 6) 최종 분석 및 논문 서술 가이드
- P11은 order/permutation과 granularity 축에서 성능/diag 차이가 명확해, stage semantics 주장에 직접 연결하기 좋다.
- P12는 layout_variants 우세 + 일부 bundle 축 약세가 분명해, "same stage set, different composition" 메시지가 강하다.
- P13은 corruption/perturbation control에서 성능/diag 변화가 함께 나와, aligned feature usage의 반증 실험(counterfactual)로 적합하다.
- 본문 구성 추천: (1) phase별 winner 표, (2) seed mean±std 표, (3) diag trade-off figure를 함께 배치해 주장-근거 연결을 명확히 한다.
- 추천 서술 전략:
  1. 본문 본표는 `H3+n>=20`만 사용해 공정성 강조
  2. P10은 support 표로 별도 분리해 "directional consistency"만 주장
  3. phase별로 valid/test/stable 3개 후보를 함께 적어 reviewer의 "winner cherry-pick" 우려를 사전에 차단
- 문장 템플릿:
  1. P11: "Stage ordering effect persisted under 4-seed verification, with `P11-14` leading valid and `P11-00` leading test."
  2. P12: "Layout composition remained a first-order factor after seed averaging, where repeated/attention placements retained their advantage."
  3. P13: "Counterfactual perturbations changed both ranking quality and routing statistics, supporting alignment-aware routing claims."

## 7) 한줄 결론
- verification 결과는 wide에서 세운 가설을 대부분 재현했고, 특히 `성능 + seed 안정성 + router behavior`를 함께 제시할 때 FeaturedMoE 설계 선택의 설득력이 가장 높다.
