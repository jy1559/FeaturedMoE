# Phase8~13 축별 결과 정리 (요청 반영 재가공)

## 공통 표 형식
- n: 집계에 포함된 run 수
- valid/test: `mean +- std` 형식 (`MRR@20`)
- verification 요약은 본문 축 표 아래에 별도 배치

---

## Phase 8
### wrapper
| name | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | ---: | ---: | ---: |
| all_w2 | a+d residual wrapper (all stage) | 1 | 0.0819 +- 0.0000 | 0.1603 +- 0.0000 |
| all_w5 | e*d product wrapper (all stage) | 2 | 0.0819 +- 0.0000 | 0.1602 +- 0.0000 |
| mixed_2 | macro=w4, mid=w6, micro=w1 | 1 | 0.0813 +- 0.0000 | 0.1604 +- 0.0000 |
| mixed_1 | macro/mid=w4, micro=w1 | 2 | 0.0810 +- 0.0006 | 0.1601 +- 0.0020 |
| all_w4 | b*d product wrapper (all stage) | 2 | 0.0809 +- 0.0001 | 0.1614 +- 0.0001 |
| all_w6 | b*d+a residual wrapper (all stage) | 2 | 0.0806 +- 0.0004 | 0.1609 +- 0.0015 |
| all_w3 | b*c product wrapper (all stage) | 1 | 0.0805 +- 0.0000 | 0.1613 +- 0.0000 |
| mixed_3 | macro=w6, mid/micro=w1 | 1 | 0.0803 +- 0.0000 | 0.1617 +- 0.0000 |
| all_w1 | flat wrapper (all stage) | 1 | 0.0801 +- 0.0000 | 0.1615 +- 0.0000 |

### bias
| name | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | ---: | ---: | ---: |
| bias_rule | rule target bias only | 5 | 0.0817 +- 0.0010 | 0.1600 +- 0.0004 |
| bias_group_feat_rule | group feature + rule bias | 4 | 0.0812 +- 0.0005 | 0.1606 +- 0.0010 |
| bias_both | feature/group + rule bias | 4 | 0.0812 +- 0.0006 | 0.1605 +- 0.0010 |
| bias_group_feat | group feature prior bias | 4 | 0.0811 +- 0.0008 | 0.1604 +- 0.0023 |
| bias_feat | feature_group_bias only | 4 | 0.0809 +- 0.0006 | 0.1611 +- 0.0005 |
| bias_off | bias off | 4 | 0.0807 +- 0.0004 | 0.1612 +- 0.0006 |

### source
| name | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | ---: | ---: | ---: |
| src_abc_feature | a/b/c=feature | 3 | 0.0813 +- 0.0008 | 0.1605 +- 0.0009 |
| src_all_both | all primitive=both | 2 | 0.0809 +- 0.0001 | 0.1608 +- 0.0012 |
| src_a_hidden_b_d_feature | a=hidden, b/d=feature | 4 | 0.0809 +- 0.0005 | 0.1606 +- 0.0009 |
| src_base | base source(a/b/c=both, d/e=feature) | 3 | 0.0808 +- 0.0004 | 0.1611 +- 0.0009 |

### top-k
| name | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | ---: | ---: | ---: |
| tk_dense | primitive/final top-k dense | 8 | 0.0809 +- 0.0002 | 0.1610 +- 0.0008 |
| tk_d1_final4 | d_cond=1 + final=4 | 3 | 0.0801 +- 0.0005 | 0.1607 +- 0.0006 |
| tk_d1 | d_cond top-k=1, final dense | 3 | 0.0800 +- 0.0005 | 0.1606 +- 0.0004 |
| tk_a3_d1_final4 | a=3, d=1, final=4 | 3 | 0.0796 +- 0.0003 | 0.1605 +- 0.0005 |

---

## Phase 9
### P9 wide 완료 run 전체 (run-level)
| run_phase | base | concept | combo | main_aux | support_aux | valid | test |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| P9_B2_C3_F4_S1 | B2 | C3 | F4 | wrapper_group_feature_align | group_prior_align | 0.0817 | 0.1601 |
| P9_B3_C3_F3_S1 | B3 | C3 | F3 | group_prior_align | none | 0.0810 | 0.1608 |
| P9_B4_C3_F4_S1 | B4 | C3 | F4 | wrapper_group_feature_align | group_prior_align | 0.0808 | 0.1613 |
| P9_B1_C3_F4_S1 | B1 | C3 | F4 | wrapper_group_feature_align | group_prior_align | 0.0807 | 0.1612 |
| P9_B1_C3_F3_S1 | B1 | C3 | F3 | group_prior_align | none | 0.0806 | 0.1611 |
| P9_B3_C3_F4_S1 | B3 | C3 | F4 | wrapper_group_feature_align | group_prior_align | 0.0803 | 0.1607 |
| P9_B2_C3_F3_S1 | B2 | C3 | F3 | group_prior_align | none | 0.0802 | 0.1607 |
| P9_B4_C3_F3_S1 | B4 | C3 | F3 | group_prior_align | none | 0.0601 | 0.1228 |

### P9_2 verification (base x combo x hparam, 16 settings)
| setting | base | combo | hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| B2::F2::H3 | B2 | F2 | H3 | 4 | 0.0820 +- 0.0005 | 0.1598 +- 0.0007 |
| B4::N4::H3 | B4 | N4 | H3 | 4 | 0.0814 +- 0.0001 | 0.1617 +- 0.0001 |
| B4::N4::H4 | B4 | N4 | H4 | 4 | 0.0812 +- 0.0001 | 0.1616 +- 0.0001 |
| B4::N4::H2 | B4 | N4 | H2 | 4 | 0.0811 +- 0.0001 | 0.1611 +- 0.0010 |
| B3::B1::H3 | B3 | B1 | H3 | 4 | 0.0810 +- 0.0007 | 0.1607 +- 0.0017 |
| B2::F2::H1 | B2 | F2 | H1 | 4 | 0.0809 +- 0.0011 | 0.1611 +- 0.0010 |
| B1::S3::H4 | B1 | S3 | H4 | 4 | 0.0809 +- 0.0001 | 0.1614 +- 0.0001 |
| B1::S3::H3 | B1 | S3 | H3 | 4 | 0.0809 +- 0.0001 | 0.1616 +- 0.0001 |
| B2::F2::H4 | B2 | F2 | H4 | 4 | 0.0808 +- 0.0004 | 0.1613 +- 0.0006 |
| B4::N4::H1 | B4 | N4 | H1 | 4 | 0.0808 +- 0.0001 | 0.1611 +- 0.0004 |
| B3::B1::H4 | B3 | B1 | H4 | 4 | 0.0808 +- 0.0003 | 0.1611 +- 0.0007 |
| B1::S3::H1 | B1 | S3 | H1 | 4 | 0.0808 +- 0.0004 | 0.1597 +- 0.0021 |
| B1::S3::H2 | B1 | S3 | H2 | 4 | 0.0807 +- 0.0001 | 0.1617 +- 0.0001 |
| B3::B1::H1 | B3 | B1 | H1 | 4 | 0.0804 +- 0.0002 | 0.1608 +- 0.0004 |
| B3::B1::H2 | B3 | B1 | H2 | 4 | 0.0803 +- 0.0001 | 0.1613 +- 0.0006 |
| B2::F2::H2 | B2 | F2 | H2 | 4 | 0.0803 +- 0.0000 | 0.1616 +- 0.0000 |

### P9_2 verification 요약 (hparam)
| hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | ---: | ---: | ---: |
| H3 | 16 | 0.0813 +- 0.0006 | 0.1610 +- 0.0011 |
| H4 | 16 | 0.0809 +- 0.0003 | 0.1613 +- 0.0005 |
| H1 | 16 | 0.0807 +- 0.0006 | 0.1607 +- 0.0012 |
| H2 | 16 | 0.0806 +- 0.0003 | 0.1614 +- 0.0006 |

---

## Phase 10 (wide + verification 재정리)
완료 run만 집계(`status=run_complete`).

### A) Feature subset + compactness (wide)
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P10-00_FULL | group_subset | All 4 families | 1 | 0.0813 +- 0.0000 | 0.1618 +- 0.0000 |
| P10-01_TEMPO | group_subset | Tempo only | 1 | 0.0795 +- 0.0000 | 0.1612 +- 0.0000 |
| P10-02_FOCUS | group_subset | Focus only | 1 | 0.0794 +- 0.0000 | 0.1613 +- 0.0000 |
| P10-03_MEMORY | group_subset | Memory only | 1 | 0.0800 +- 0.0000 | 0.1592 +- 0.0000 |
| P10-04_EXPOSURE | group_subset | Exposure only | 1 | 0.0792 +- 0.0000 | 0.1592 +- 0.0000 |
| P10-05_TEMPO_FOCUS | group_subset | Tempo + Focus | 1 | 0.0798 +- 0.0000 | 0.1617 +- 0.0000 |
| P10-06_TEMPO_MEMORY | group_subset | Tempo + Memory | 1 | 0.0801 +- 0.0000 | 0.1621 +- 0.0000 |
| P10-07_TEMPO_EXPOSURE | group_subset | Tempo + Exposure | 1 | 0.0804 +- 0.0000 | 0.1594 +- 0.0000 |
| P10-08_FOCUS_MEMORY | group_subset | Focus + Memory | 1 | 0.0802 +- 0.0000 | 0.1604 +- 0.0000 |
| P10-09_FOCUS_EXPOSURE | group_subset | Focus + Exposure | 1 | 0.0799 +- 0.0000 | 0.1613 +- 0.0000 |
| P10-10_MEMORY_EXPOSURE | group_subset | Memory + Exposure | 1 | 0.0805 +- 0.0000 | 0.1609 +- 0.0000 |
| P10-11_TEMPO_FOCUS_MEMORY | group_subset | Tempo + Focus + Memory | 1 | 0.0804 +- 0.0000 | 0.1610 +- 0.0000 |
| P10-12_TEMPO_FOCUS_EXPOSURE | group_subset | Tempo + Focus + Exposure | 1 | 0.0806 +- 0.0000 | 0.1618 +- 0.0000 |
| P10-13_TEMPO_MEMORY_EXPOSURE | group_subset | Tempo + Memory + Exposure | 1 | 0.0808 +- 0.0000 | 0.1612 +- 0.0000 |
| P10-14_FOCUS_MEMORY_EXPOSURE | group_subset | Focus + Memory + Exposure | 1 | 0.0824 +- 0.0000 | 0.1587 +- 0.0000 |
| P10-15_TOP2_PER_GROUP | compactness | Keep top-2 representative features per family/stage | 1 | 0.0812 +- 0.0000 | 0.1608 +- 0.0000 |
| P10-16_TOP1_PER_GROUP | compactness | Keep top-1 representative feature per family/stage | 1 | 0.0804 +- 0.0000 | 0.1621 +- 0.0000 |
| P10-17_COMMON_TEMPLATE | compactness | Fixed reusable common template across stages | 1 | 0.0812 +- 0.0000 | 0.1590 +- 0.0000 |
| P10-18_NO_CATEGORY | availability | Drop category/theme-derived columns | 1 | 0.0820 +- 0.0000 | 0.1582 +- 0.0000 |
| P10-19_NO_TIMESTAMP | availability | Drop timestamp/pace/interval-derived columns | 1 | 0.0809 +- 0.0000 | 0.1614 +- 0.0000 |

### A) Feature subset + compactness (verification, hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P10-00_FULL | H1 | group_subset | All 4 families | 4 | 0.0808 +- 0.0001 | 0.1613 +- 0.0000 |
| P10-00_FULL | H3 | group_subset | All 4 families | 4 | 0.0814 +- 0.0001 | 0.1616 +- 0.0001 |
| P10-06_Tempo_Memory | H3 | group_subset | Tempo + Memory | 4 | 0.0798 +- 0.0001 | 0.1619 +- 0.0002 |
| P10-06_Tempo_Memory | H1 | group_subset | Tempo + Memory | 4 | 0.0800 +- 0.0000 | 0.1618 +- 0.0001 |
| P10-14_Focus_Memory_Exposure | H1 | group_subset | Focus + Memory + Exposure | 4 | 0.0803 +- 0.0000 | 0.1614 +- 0.0000 |
| P10-14_Focus_Memory_Exposure | H3 | group_subset | Focus + Memory + Exposure | 4 | 0.0808 +- 0.0001 | 0.1618 +- 0.0001 |
| P10-15_TOP2_PER_GROUP | H1 | compactness | Keep top-2 representative features per family/stage | 4 | 0.0809 +- 0.0001 | 0.1617 +- 0.0001 |
| P10-15_TOP2_PER_GROUP | H3 | compactness | Keep top-2 representative features per family/stage | 4 | 0.0806 +- 0.0000 | 0.1622 +- 0.0001 |
| P10-16_TOP1_PER_GROUP | H3 | compactness | Keep top-1 representative feature per family/stage | 4 | 0.0804 +- 0.0001 | 0.1618 +- 0.0003 |
| P10-16_TOP1_PER_GROUP | H1 | compactness | Keep top-1 representative feature per family/stage | 4 | 0.0803 +- 0.0000 | 0.1620 +- 0.0002 |
| P10-17_COMMON_TEMPLATE | H3 | compactness | Fixed reusable common template across stages | 4 | 0.0805 +- 0.0000 | 0.1620 +- 0.0002 |
| P10-17_COMMON_TEMPLATE | H1 | compactness | Fixed reusable common template across stages | 4 | 0.0809 +- 0.0001 | 0.1618 +- 0.0002 |
| P10-18_NO_CATEGORY | H3 | availability | Drop category/theme-derived columns | 4 | 0.0809 +- 0.0001 | 0.1619 +- 0.0002 |
| P10-18_NO_CATEGORY | H1 | availability | Drop category/theme-derived columns | 4 | 0.0806 +- 0.0000 | 0.1616 +- 0.0001 |
| P10-19_NO_TIMESTAMP | H3 | availability | Drop timestamp/pace/interval-derived columns | 4 | 0.0809 +- 0.0000 | 0.1618 +- 0.0002 |
| P10-19_NO_TIMESTAMP | H1 | availability | Drop timestamp/pace/interval-derived columns | 4 | 0.0804 +- 0.0000 | 0.1616 +- 0.0001 |
| P10-22_NO_CATEGORY_NO_TIMESTAMP | H1 | availability_plus | Drop both category/theme and timestamp-derived columns | 4 | 0.0803 +- 0.0002 | 0.1613 +- 0.0000 |
| P10-22_NO_CATEGORY_NO_TIMESTAMP | H3 | availability_plus | Drop both category/theme and timestamp-derived columns | 4 | 0.0804 +- 0.0000 | 0.1615 +- 0.0001 |
| P10-23_COMMON_TEMPLATE_NO_CATEGORY | H1 | compactness_plus | Common template while removing category/theme columns | 4 | 0.0808 +- 0.0000 | 0.1619 +- 0.0000 |
| P10-23_COMMON_TEMPLATE_NO_CATEGORY | H3 | compactness_plus | Common template while removing category/theme columns | 4 | 0.0804 +- 0.0001 | 0.1619 +- 0.0002 |

### B) Portability / stochastic claim set (wide)
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P10-00_FULL | group_subset | All 4 families | 1 | 0.0813 +- 0.0000 | 0.1618 +- 0.0000 |
| P10-18_NO_CATEGORY | availability | Drop category/theme-derived columns | 1 | 0.0820 +- 0.0000 | 0.1582 +- 0.0000 |
| P10-19_NO_TIMESTAMP | availability | Drop timestamp/pace/interval-derived columns | 1 | 0.0809 +- 0.0000 | 0.1614 +- 0.0000 |
| P10-20_FAMILY_DROPOUT | stochastic | Train-time family dropout (session scope) | 1 | 0.0809 +- 0.0000 | 0.1622 +- 0.0000 |
| P10-21_FEATURE_DROPOUT | stochastic | Train-time element-wise feature dropout | 1 | 0.0811 +- 0.0000 | 0.1621 +- 0.0000 |

### B) Portability / stochastic claim set (verification, hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P10-00_FULL | H1 | group_subset | All 4 families | 4 | 0.0808 +- 0.0001 | 0.1613 +- 0.0000 |
| P10-00_FULL | H3 | group_subset | All 4 families | 4 | 0.0814 +- 0.0001 | 0.1616 +- 0.0001 |
| P10-18_NO_CATEGORY | H3 | availability | Drop category/theme-derived columns | 4 | 0.0809 +- 0.0001 | 0.1619 +- 0.0002 |
| P10-18_NO_CATEGORY | H1 | availability | Drop category/theme-derived columns | 4 | 0.0806 +- 0.0000 | 0.1616 +- 0.0001 |
| P10-19_NO_TIMESTAMP | H3 | availability | Drop timestamp/pace/interval-derived columns | 4 | 0.0809 +- 0.0000 | 0.1618 +- 0.0002 |
| P10-19_NO_TIMESTAMP | H1 | availability | Drop timestamp/pace/interval-derived columns | 4 | 0.0804 +- 0.0000 | 0.1616 +- 0.0001 |
| P10-20_FAMILY_DROPOUT | H3 | stochastic | Train-time family dropout (session scope) | 4 | 0.0809 +- 0.0000 | 0.1623 +- 0.0001 |
| P10-20_FAMILY_DROPOUT | H1 | stochastic | Train-time family dropout (session scope) | 4 | 0.0802 +- 0.0001 | 0.1613 +- 0.0002 |
| P10-21_FEATURE_DROPOUT | H1 | stochastic | Train-time element-wise feature dropout | 4 | 0.0803 +- 0.0000 | 0.1617 +- 0.0002 |
| P10-21_FEATURE_DROPOUT | H3 | stochastic | Train-time element-wise feature dropout | 4 | 0.0809 +- 0.0001 | 0.1621 +- 0.0001 |
| P10-22_NO_CATEGORY_NO_TIMESTAMP | H1 | availability_plus | Drop both category/theme and timestamp-derived columns | 4 | 0.0803 +- 0.0002 | 0.1613 +- 0.0000 |
| P10-22_NO_CATEGORY_NO_TIMESTAMP | H3 | availability_plus | Drop both category/theme and timestamp-derived columns | 4 | 0.0804 +- 0.0000 | 0.1615 +- 0.0001 |
| P10-23_COMMON_TEMPLATE_NO_CATEGORY | H1 | compactness_plus | Common template while removing category/theme columns | 4 | 0.0808 +- 0.0000 | 0.1619 +- 0.0000 |
| P10-23_COMMON_TEMPLATE_NO_CATEGORY | H3 | compactness_plus | Common template while removing category/theme columns | 4 | 0.0804 +- 0.0001 | 0.1619 +- 0.0002 |

### P10 verification 요약 (hparam 분리, seed는 hparam 내부 집계)
hparam:
| hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| H1 | 48 | 0.0805 +- 0.0003 | 0.1616 +- 0.0003 |
| H3 | 48 | 0.0806 +- 0.0004 | 0.1619 +- 0.0003 |

H1 seed:
| seed | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| 1 | 12 | 0.0805 +- 0.0003 | 0.1616 +- 0.0003 |
| 2 | 12 | 0.0805 +- 0.0003 | 0.1616 +- 0.0002 |
| 3 | 12 | 0.0805 +- 0.0003 | 0.1616 +- 0.0002 |
| 4 | 12 | 0.0805 +- 0.0003 | 0.1616 +- 0.0003 |

H3 seed:
| seed | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| 1 | 12 | 0.0807 +- 0.0004 | 0.1620 +- 0.0003 |
| 2 | 12 | 0.0806 +- 0.0004 | 0.1619 +- 0.0002 |
| 3 | 12 | 0.0806 +- 0.0004 | 0.1618 +- 0.0003 |
| 4 | 12 | 0.0806 +- 0.0004 | 0.1619 +- 0.0003 |

---

## Phase 11 (wide + verification 재정리)
완료 run만 집계(`status=run_complete`).

### A) Stage semantics core (ablation + prepend + permutation + extra) - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P11-00_MACRO_MID_MICRO | base_ablation | Full 3-stage baseline | 1 | 0.0829 +- 0.0000 | 0.1573 +- 0.0000 |
| P11-01_MID_MICRO | base_ablation | Drop macro stage | 1 | 0.0804 +- 0.0000 | 0.1617 +- 0.0000 |
| P11-02_MACRO_MICRO | base_ablation | Drop mid stage | 1 | 0.0794 +- 0.0000 | 0.1615 +- 0.0000 |
| P11-03_MACRO_MID | base_ablation | Drop micro stage | 1 | 0.0805 +- 0.0000 | 0.1619 +- 0.0000 |
| P11-04_MACRO_ONLY | base_ablation | Macro-only horizon | 1 | 0.0789 +- 0.0000 | 0.1609 +- 0.0000 |
| P11-05_MID_ONLY | base_ablation | Mid-only horizon | 1 | 0.0797 +- 0.0000 | 0.1605 +- 0.0000 |
| P11-06_MICRO_ONLY | base_ablation | Micro-only horizon | 1 | 0.0789 +- 0.0000 | 0.1614 +- 0.0000 |
| P11-07_LAYER_MACRO_MID_MICRO | prepend_layer | Dense layer before full 3-stage | 1 | 0.0811 +- 0.0000 | 0.1618 +- 0.0000 |
| P11-08_LAYER_MID_MICRO | prepend_layer | Dense layer before mid/micro | 1 | 0.0806 +- 0.0000 | 0.1614 +- 0.0000 |
| P11-09_LAYER_MACRO_MICRO | prepend_layer | Dense layer before macro/micro | 1 | 0.0796 +- 0.0000 | 0.1615 +- 0.0000 |
| P11-10_LAYER_MACRO_MID | prepend_layer | Dense layer before macro/mid | 1 | 0.0809 +- 0.0000 | 0.1601 +- 0.0000 |
| P11-11_LAYER_MACRO | prepend_layer | Dense layer before macro | 1 | 0.0791 +- 0.0000 | 0.1604 +- 0.0000 |
| P11-12_LAYER_MID | prepend_layer | Dense layer before mid | 1 | 0.0804 +- 0.0000 | 0.1616 +- 0.0000 |
| P11-13_LAYER_MICRO | prepend_layer | Dense layer before micro | 1 | 0.0790 +- 0.0000 | 0.1614 +- 0.0000 |
| P11-14_MACRO_MICRO_MID | order_permutation | Permutation macro->micro->mid | 1 | 0.0817 +- 0.0000 | 0.1623 +- 0.0000 |
| P11-15_MID_MACRO_MICRO | order_permutation | Permutation mid->macro->micro | 1 | 0.0816 +- 0.0000 | 0.1607 +- 0.0000 |
| P11-16_MID_MICRO_MACRO | order_permutation | Permutation mid->micro->macro | 1 | 0.0816 +- 0.0000 | 0.1621 +- 0.0000 |
| P11-17_MICRO_MACRO_MID | order_permutation | Permutation micro->macro->mid | 1 | 0.0832 +- 0.0000 | 0.1593 +- 0.0000 |
| P11-18_MICRO_MID_MACRO | order_permutation | Permutation micro->mid->macro | 1 | 0.0825 +- 0.0000 | 0.1616 +- 0.0000 |
| P11-22_LAYER_ONLY_BASELINE | extra_alignment | Pure dense layer-only baseline | 1 | 0.0783 +- 0.0000 | 0.1595 +- 0.0000 |
| P11-23_LAYER2_MACRO_MID_MICRO | extra_alignment | Two dense prep layers before full stages | 1 | 0.0815 +- 0.0000 | 0.1619 +- 0.0000 |

### B) Routing granularity - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P11-19_TOKEN_TOKEN_TOKEN | routing_granularity | Token routing for all stages | 1 | 0.0814 +- 0.0000 | 0.1617 +- 0.0000 |
| P11-20_SESSION_TOKEN_TOKEN | routing_granularity | Macro session, mid/micro token | 1 | 0.0819 +- 0.0000 | 0.1603 +- 0.0000 |
| P11-21_TOKEN_SESSION_TOKEN | routing_granularity | Macro token, mid session, micro token | 1 | 0.0812 +- 0.0000 | 0.1614 +- 0.0000 |

### A) Stage semantics core - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P11-00_MACRO_MID_MICRO | H3 | base_ablation | Full 3-stage baseline | 4 | 0.0813 +- 0.0001 | 0.1616 +- 0.0001 |
| P11-03_MACRO_MID | H3 | base_ablation | Drop micro stage | 4 | 0.0808 +- 0.0001 | 0.1611 +- 0.0004 |
| P11-14_MACRO_MICRO_MID | H3 | order_permutation | Permutation macro->micro->mid | 4 | 0.0822 +- 0.0006 | 0.1580 +- 0.0043 |
| P11-17_MICRO_MACRO_MID | H3 | order_permutation | Permutation micro->macro->mid | 4 | 0.0820 +- 0.0005 | 0.1614 +- 0.0005 |
| P11-22_LAYER_ONLY_BASELINE | H3 | extra_alignment | Pure dense layer-only baseline | 4 | 0.0783 +- 0.0000 | 0.1593 +- 0.0001 |
| P11-23_LAYER2_MACRO_MID_MICRO | H3 | extra_alignment | Two dense prep layers before full stages | 4 | 0.0816 +- 0.0001 | 0.1609 +- 0.0012 |

### B) Routing granularity - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P11-19_TOKEN_TOKEN_TOKEN | H3 | routing_granularity | Token routing for all stages | 4 | 0.0816 +- 0.0005 | 0.1613 +- 0.0007 |
| P11-20_SESSION_TOKEN_TOKEN | H3 | routing_granularity | Macro session, mid/micro token | 4 | 0.0814 +- 0.0000 | 0.1614 +- 0.0005 |

### P11 verification 요약 (hparam 분리, seed는 hparam 내부 집계)
hparam:
| hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| H3 | 32 | 0.0811 +- 0.0012 | 0.1606 +- 0.0020 |

H3 seed:
| seed | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| 1 | 8 | 0.0811 +- 0.0013 | 0.1600 +- 0.0024 |
| 2 | 8 | 0.0813 +- 0.0013 | 0.1603 +- 0.0027 |
| 3 | 8 | 0.0810 +- 0.0011 | 0.1614 +- 0.0008 |
| 4 | 8 | 0.0812 +- 0.0012 | 0.1609 +- 0.0009 |

---

## Phase 12 (wide + verification 재정리)
완료 run만 집계(`status=run_complete`).

### A) Layout - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P12-00_ATTN_ONESHOT | layout_variants | [attn, macro_ffn, mid_ffn, micro_ffn] | 1 | 0.0802 +- 0.0000 | 0.1615 +- 0.0000 |
| P12-01_ATTN_MACRO_ONLY | layout_variants | [attn, macro_ffn, attn, mid_ffn, micro_ffn] | 1 | 0.0808 +- 0.0000 | 0.1614 +- 0.0000 |
| P12-02_ATTN_MICRO_BEFORE | layout_variants | [attn, macro_ffn, mid_ffn, attn, micro_ffn] | 1 | 0.0809 +- 0.0000 | 0.1620 +- 0.0000 |
| P12-03_NO_ATTN_ONLY_MOEFFN | layout_variants | [macro_ffn, mid_ffn, micro_ffn] | 1 | 0.0774 +- 0.0000 | 0.1593 +- 0.0000 |
| P12-04_LAYER_PLUS_MOEFFN | layout_variants | [layer, macro_ffn, mid_ffn, micro_ffn] | 1 | 0.0802 +- 0.0000 | 0.1612 +- 0.0000 |
| P12-05_MACRO_REPEATED | layout_variants | [macro, macro_ffn, mid, micro] | 1 | 0.0815 +- 0.0000 | 0.1613 +- 0.0000 |
| P12-06_MID_REPEATED | layout_variants | [macro, mid, mid_ffn, micro] | 1 | 0.0814 +- 0.0000 | 0.1618 +- 0.0000 |
| P12-07_MICRO_REPEATED | layout_variants | [macro, mid, micro, micro_ffn] | 1 | 0.0823 +- 0.0000 | 0.1619 +- 0.0000 |
| P12-08_MACRO_NOLOCALATTN | layout_variants | [macro_ffn, mid, micro] | 1 | 0.0810 +- 0.0000 | 0.1619 +- 0.0000 |
| P12-09_MID_NOLOCALATTN | layout_variants | [macro, mid_ffn, micro] | 1 | 0.0809 +- 0.0000 | 0.1622 +- 0.0000 |

### B) Bundle - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P12-10_BUNDLE_MACROMID_SUM | bundle_pair_then_follow | bundle_macro_mid_sum -> micro | 1 | 0.0802 +- 0.0000 | 0.1616 +- 0.0000 |
| P12-11_BUNDLE_MACROMID_MEAN | bundle_pair_then_follow | bundle_macro_mid_mean -> micro | 1 | 0.0802 +- 0.0000 | 0.1613 +- 0.0000 |
| P12-12_BUNDLE_MACROMID_LEARNED | bundle_pair_then_follow | bundle_macro_mid_learned -> micro | 1 | 0.0803 +- 0.0000 | 0.1615 +- 0.0000 |
| P12-13_BUNDLE_MIDMICRO_SUM | bundle_pair_then_follow | bundle_mid_micro_sum -> macro | 1 | 0.0800 +- 0.0000 | 0.1597 +- 0.0000 |
| P12-14_BUNDLE_MIDMICRO_MEAN | bundle_pair_then_follow | bundle_mid_micro_mean -> macro | 1 | 0.0801 +- 0.0000 | 0.1615 +- 0.0000 |
| P12-15_BUNDLE_MIDMICRO_LEARNED | bundle_pair_then_follow | bundle_mid_micro_learned -> macro | 1 | 0.0802 +- 0.0000 | 0.1617 +- 0.0000 |
| P12-16_BUNDLE_MACROMICRO_SUM | bundle_pair_then_follow | bundle_macro_micro_sum -> mid | 1 | 0.0803 +- 0.0000 | 0.1615 +- 0.0000 |
| P12-17_BUNDLE_MACROMICRO_MEAN | bundle_pair_then_follow | bundle_macro_micro_mean -> mid | 1 | 0.0807 +- 0.0000 | 0.1617 +- 0.0000 |
| P12-18_BUNDLE_MACROMICRO_LEARNED | bundle_pair_then_follow | bundle_macro_micro_learned -> mid | 1 | 0.0808 +- 0.0000 | 0.1616 +- 0.0000 |
| P12-19_BUNDLE_ALL_SUM | bundle_all | bundle_macro_mid_micro_sum | 1 | 0.0766 +- 0.0000 | 0.1589 +- 0.0000 |
| P12-20_BUNDLE_ALL_MEAN | bundle_all | bundle_macro_mid_micro_mean | 1 | 0.0772 +- 0.0000 | 0.1592 +- 0.0000 |
| P12-21_BUNDLE_ALL_LEARNED | bundle_all | bundle_macro_mid_micro_learned | 1 | 0.0773 +- 0.0000 | 0.1593 +- 0.0000 |
| P12-22_BUNDLE_MACROMID_THEN_MIDMICRO_LEARNED | bundle_chain | bundle_macro_mid_learned -> bundle_mid_micro_learned | 1 | 0.0776 +- 0.0000 | 0.1595 +- 0.0000 |
| P12-23_BUNDLE_MACROMICRO_THEN_MIDMICRO_LEARNED | bundle_chain | bundle_macro_micro_learned -> bundle_mid_micro_learned | 1 | 0.0775 +- 0.0000 | 0.1596 +- 0.0000 |
| P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | bundle_router | bundle_macro_mid_router -> micro | 1 | 0.0805 +- 0.0000 | 0.1611 +- 0.0000 |
| P12-25_BUNDLE_ALL_ROUTER_CONDITIONED | bundle_router | bundle_macro_mid_micro_router | 1 | 0.0767 +- 0.0000 | 0.1580 +- 0.0000 |
| P12-26_BUNDLE_MACROMID_THEN_MIDMICRO_SUM | bundle_chain | bundle_macro_mid_sum -> bundle_mid_micro_sum | 1 | 0.0772 +- 0.0000 | 0.1575 +- 0.0000 |
| P12-27_BUNDLE_MACROMID_THEN_MIDMICRO_MEAN | bundle_chain | bundle_macro_mid_mean -> bundle_mid_micro_mean | 1 | 0.0777 +- 0.0000 | 0.1595 +- 0.0000 |
| P12-28_BUNDLE_MACROMID_THEN_MIDMICRO_ROUTER_CONDITIONED | bundle_chain | bundle_macro_mid_router -> bundle_mid_micro_router | 1 | 0.0775 +- 0.0000 | 0.1594 +- 0.0000 |
| P12-29_BUNDLE_MACROMICRO_THEN_MIDMICRO_SUM | bundle_chain | bundle_macro_micro_sum -> bundle_mid_micro_sum | 1 | 0.0782 +- 0.0000 | 0.1579 +- 0.0000 |
| P12-30_BUNDLE_MACROMICRO_THEN_MIDMICRO_MEAN | bundle_chain | bundle_macro_micro_mean -> bundle_mid_micro_mean | 1 | 0.0774 +- 0.0000 | 0.1595 +- 0.0000 |
| P12-31_BUNDLE_MACROMICRO_THEN_MIDMICRO_ROUTER_CONDITIONED | bundle_chain | bundle_macro_micro_router -> bundle_mid_micro_router | 1 | 0.0780 +- 0.0000 | 0.1596 +- 0.0000 |

### A) Layout - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P12-00_ATTN_ONESHOT | H3 | layout_variants | [attn, macro_ffn, mid_ffn, micro_ffn] | 4 | 0.0802 +- 0.0000 | 0.1610 +- 0.0004 |
| P12-02_ATTN_MICRO_BEFORE | H3 | layout_variants | [attn, macro_ffn, mid_ffn, attn, micro_ffn] | 4 | 0.0809 +- 0.0001 | 0.1621 +- 0.0001 |
| P12-06_MID_REPEATED | H3 | layout_variants | [macro, mid, mid_ffn, micro] | 4 | 0.0815 +- 0.0004 | 0.1606 +- 0.0020 |
| P12-07_MICRO_REPEATED | H3 | layout_variants | [macro, mid, micro, micro_ffn] | 4 | 0.0814 +- 0.0003 | 0.1614 +- 0.0008 |
| P12-08_MACRO_NOLOCALATTN | H3 | layout_variants | [macro_ffn, mid, micro] | 4 | 0.0810 +- 0.0000 | 0.1615 +- 0.0007 |
| P12-09_MID_NOLOCALATTN | H3 | layout_variants | [macro, mid_ffn, micro] | 4 | 0.0808 +- 0.0000 | 0.1621 +- 0.0001 |

### B) Bundle - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P12-18_BUNDLE_MACROMICRO_LEARNED | H3 | bundle_pair_then_follow | bundle_macro_micro_learned -> mid | 4 | 0.0809 +- 0.0001 | 0.1615 +- 0.0001 |
| P12-24_BUNDLE_MACROMID_ROUTER_CONDITIONED | H3 | bundle_router | bundle_macro_mid_router -> micro | 4 | 0.0805 +- 0.0000 | 0.1612 +- 0.0002 |

### P12 verification 요약 (hparam 분리, seed는 hparam 내부 집계)
hparam:
| hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| H3 | 32 | 0.0809 +- 0.0004 | 0.1614 +- 0.0009 |

H3 seed:
| seed | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| 1 | 8 | 0.0809 +- 0.0003 | 0.1613 +- 0.0006 |
| 2 | 8 | 0.0810 +- 0.0005 | 0.1615 +- 0.0007 |
| 3 | 8 | 0.0809 +- 0.0003 | 0.1618 +- 0.0003 |
| 4 | 8 | 0.0809 +- 0.0005 | 0.1611 +- 0.0015 |

---

## Phase 13 (wide + verification 재정리)
완료 run만 집계(`status=run_complete`).

### A) Data condition - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P13-00_FULL_DATA | data_condition | Clean full feature setup | 1 | 0.0812 +- 0.0000 | 0.1615 +- 0.0000 |
| P13-01_CATEGORY_ZERO_DATA | data_condition | Category/theme columns zeroed while feature shape is preserved | 1 | 0.0812 +- 0.0000 | 0.1624 +- 0.0000 |

### B) Eval perturb - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P13-02_EVAL_ALL_ZERO | eval_perturb | Eval-only zero all families | 1 | 0.0776 +- 0.0000 | 0.1598 +- 0.0000 |
| P13-03_EVAL_ALL_SHUFFLE | eval_perturb | Eval-only shuffle all families | 1 | 0.0770 +- 0.0000 | 0.1556 +- 0.0000 |
| P13-04_EVAL_SHUFFLE_TEMPO | eval_perturb | Eval-only shuffle Tempo family | 1 | 0.0805 +- 0.0000 | 0.1614 +- 0.0000 |
| P13-05_EVAL_SHUFFLE_FOCUS | eval_perturb | Eval-only shuffle Focus family | 1 | 0.0807 +- 0.0000 | 0.1616 +- 0.0000 |
| P13-06_EVAL_SHUFFLE_MEMORY | eval_perturb | Eval-only shuffle Memory family | 1 | 0.0799 +- 0.0000 | 0.1585 +- 0.0000 |
| P13-07_EVAL_SHUFFLE_EXPOSURE | eval_perturb | Eval-only shuffle Exposure family | 1 | 0.0795 +- 0.0000 | 0.1610 +- 0.0000 |
| P13-17_EVAL_ZERO_TEMPO | eval_perturb_extra | Eval-only zero Tempo family | 1 | 0.0807 +- 0.0000 | 0.1595 +- 0.0000 |
| P13-18_EVAL_ZERO_FOCUS | eval_perturb_extra | Eval-only zero Focus family | 1 | 0.0802 +- 0.0000 | 0.1618 +- 0.0000 |
| P13-19_EVAL_ZERO_MEMORY | eval_perturb_extra | Eval-only zero Memory family | 1 | 0.0783 +- 0.0000 | 0.1597 +- 0.0000 |
| P13-20_EVAL_ZERO_EXPOSURE | eval_perturb_extra | Eval-only zero Exposure family | 1 | 0.0786 +- 0.0000 | 0.1602 +- 0.0000 |

### C) Train corruption - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P13-08_TRAIN_GLOBAL_PERMUTE_ALL | train_corruption | Train-only global permutation | 1 | 0.0790 +- 0.0000 | 0.1605 +- 0.0000 |
| P13-10_TRAIN_PERMUTE_TEMPO | train_corruption | Train-only family permutation Tempo | 1 | 0.0811 +- 0.0000 | 0.1620 +- 0.0000 |
| P13-11_TRAIN_PERMUTE_FOCUS | train_corruption | Train-only family permutation Focus | 1 | 0.0811 +- 0.0000 | 0.1619 +- 0.0000 |
| P13-12_TRAIN_PERMUTE_MEMORY | train_corruption | Train-only family permutation Memory | 1 | 0.0810 +- 0.0000 | 0.1617 +- 0.0000 |
| P13-13_TRAIN_PERMUTE_EXPOSURE | train_corruption | Train-only family permutation Exposure | 1 | 0.0806 +- 0.0000 | 0.1618 +- 0.0000 |
| P13-21_TRAIN_POSITION_SHIFT_PLUS1 | train_shift_extra | Train-only position shift +1 | 1 | 0.0802 +- 0.0000 | 0.1576 +- 0.0000 |
| P13-22_TRAIN_POSITION_SHIFT_PLUS2 | train_shift_extra | Train-only position shift +2 | 1 | 0.0809 +- 0.0000 | 0.1614 +- 0.0000 |
| P13-23_TRAIN_POSITION_SHIFT_PLUS3 | train_shift_extra | Train-only position shift +3 | 1 | 0.0792 +- 0.0000 | 0.1558 +- 0.0000 |

### D) Semantic mismatch - wide
| name | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- |
| P13-15_STAGE_MISMATCH_ASSIGN | semantic_mismatch | Stage mismatch assignment | 1 | 0.0810 +- 0.0000 | 0.1617 +- 0.0000 |
| P13-16_POSITION_SHIFT_FEATURE | semantic_mismatch | Position shift on all families (shift=1) | 1 | 0.0808 +- 0.0000 | 0.1617 +- 0.0000 |

### A) Data condition - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P13-00_FULL_DATA | H3 | data_condition | Clean full feature setup | 4 | 0.0816 +- 0.0006 | 0.1615 +- 0.0005 |
| P13-01_CATEGORY_ZERO_DATA | H3 | data_condition | Category/theme columns zeroed while feature shape is preserved | 4 | 0.0813 +- 0.0001 | 0.1607 +- 0.0026 |

### B) Eval perturb - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P13-02_EVAL_ALL_ZERO | H3 | eval_perturb | Eval-only zero all families | 4 | 0.0775 +- 0.0000 | 0.1597 +- 0.0001 |
| P13-03_EVAL_ALL_SHUFFLE | H3 | eval_perturb | Eval-only shuffle all families | 4 | 0.0771 +- 0.0000 | 0.1552 +- 0.0003 |

### C) Train corruption - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P13-10_TRAIN_PERMUTE_TEMPO | H3 | train_corruption | Train-only family permutation Tempo | 4 | 0.0811 +- 0.0000 | 0.1618 +- 0.0001 |
| P13-11_TRAIN_PERMUTE_FOCUS | H3 | train_corruption | Train-only family permutation Focus | 4 | 0.0812 +- 0.0000 | 0.1620 +- 0.0002 |

### D) Semantic mismatch - verification (hparam 분리)
| name | hparam | setting_group | 설명 | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- | --- | --- | --- |
| P13-14_FEATURE_ROLE_SWAP | H3 | semantic_mismatch | Role swap across families | 4 | 0.0810 +- 0.0001 | 0.1617 +- 0.0008 |
| P13-15_STAGE_MISMATCH_ASSIGN | H3 | semantic_mismatch | Stage mismatch assignment | 4 | 0.0812 +- 0.0002 | 0.1609 +- 0.0009 |

### P13 verification 요약 (hparam 분리, seed는 hparam 내부 집계)
hparam:
| hparam | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| H3 | 32 | 0.0803 +- 0.0017 | 0.1605 +- 0.0023 |

H3 seed:
| seed | n | valid (mean +- std) | test (mean +- std) |
| --- | --- | --- | --- |
| 1 | 8 | 0.0802 +- 0.0017 | 0.1605 +- 0.0022 |
| 2 | 8 | 0.0803 +- 0.0019 | 0.1607 +- 0.0023 |
| 3 | 8 | 0.0802 +- 0.0017 | 0.1608 +- 0.0021 |
| 4 | 8 | 0.0803 +- 0.0017 | 0.1598 +- 0.0025 |
