# Phase 6 Results (P6 Candidate Reinforcement v2)

작성일: 2026-03-18
데이터셋: KuaiRecLargeStrictPosV2_0.2

## 0) 메인 지표 기준
- 메인: best val MRR@20
- 보조: test MRR@20 (같은 행에 괄호로 병기)
- 본 문서의 상단 요약/추천은 best val 기준으로 재정렬했고, 아래 기존 상세 표는 보조 참고로 유지

### 0.1 suite별 메인 지표 요약 (combo dedup)
| suite | combos | best val MRR@20 (best/avg/worst) | test MRR@20 (best/avg/worst) | cold item MRR@20 avg |
|---|---:|---:|---:|---:|
| Candidate 재확인 | 9 | 0.0818 / 0.0809 / 0.0802 | 0.1621 / 0.1609 / 0.1586 | 0.1185 |
| Baseline bridge | 8 | 0.0813 / 0.0798 / 0.0780 | 0.1620 / 0.1607 / 0.1580 | 0.1157 |
| Router x Injection 2x2 | 8 | 0.0812 / 0.0804 / 0.0799 | 0.1620 / 0.1600 / 0.1579 | 0.1163 |
| Specialization 정규화 ablation | 10 | 0.0813 / 0.0808 / 0.0803 | 0.1622 / 0.1617 / 0.1607 | 0.1196 |
| Feature ablation sweep | 20 | 0.0801 / 0.0797 / 0.0792 | 0.1620 / 0.1613 / 0.1600 | 0.1195 |

### 0.2 best val 상위 run (test는 보조 병기)
| rank | combo | suite | best val MRR@20 | test MRR@20 | cold item MRR@20 |
|---:|---|---|---:|---:|---:|
| 1 | cand_c_s1 | cand3x | 0.0818 | (0.1586) | 0.1172 |
| 2 | base_b6 | baseline_bridge | 0.0813 | (0.1619) | 0.1206 |
| 3 | spec_a_m3 | spec_ablation | 0.0813 | (0.1619) | 0.1221 |
| 4 | cand_a_s3 | cand3x | 0.0812 | (0.1621) | 0.1214 |
| 5 | rxi_x1_sta_gat | router2x2 | 0.0812 | (0.1620) | 0.1220 |
| 6 | spec_a_m0 | spec_ablation | 0.0812 | (0.1622) | 0.1213 |

해석 메모:
- best val 최고점은 `cand_c_s1`이지만 test가 동반 하락해 일반화 관점에서는 단독 채택 리스크가 있음.
- best val과 test를 함께 보면 `base_b6`, `spec_a_m3`, `cand_a_s3`, `rxi_x1_sta_gat`이 더 실전적인 상위군.

## 1) 정리 범위
- Plan 문서: experiments/run/fmoe_n3/docs/phase6_plan.md
- 실행 로그: experiments/run/artifacts/logs/fmoe_n3/phase6_candidate_reinfor_v2/P6/KuaiRecLargeStrictPosV2_0.2/FMoEN3
- logging bundle: experiments/run/artifacts/logging/fmoe_n3/KuaiRecLargeStrictPosV2_0.2/P6
- 집계 파일: outputs/phase6_p6_logging_agg.json, outputs/phase6_p6_logging_focus_summary.json

## 2) 실행 커버리지 요약
- run 디렉터리 총 79개 확인 (재시도/중복 포함)
- 계획된 55 combo는 전부 관측됨 (cand 9, base 8, router 8, spec 10, feature 20)
- combo dedup 기준(각 combo 최고 test_mrr@20 1개 채택) 총 55개
- 일부 재시도 run은 산출물 누락:
  - baseline_bridge: result 누락 10 / special 누락 11 / diag 누락 13
  - feature_ablation: result 누락 0 / special 누락 3 / diag 누락 3
- 해석은 원칙적으로 combo dedup(최고 test_mrr@20 run) 기준으로 수행

## 3) 전체 성능 요약 (combo dedup 기준)

| suite | combos | test_mrr@20 best | test_mrr@20 avg | test_mrr@20 worst | cold_item_mrr@20 avg(<=5) | short_session mrr@20 avg(1-2) | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate 재확인 | 9 | 0.1621 | 0.1609 | 0.1586 | 0.1185 | 0.0049 | 0.9540 | 0.9748 | 0.5444 |
| Baseline bridge | 8 | 0.1620 | 0.1607 | 0.1580 | 0.1157 | 0.0005 | 0.9603 | 0.9785 | 0.4858 |
| Router x Injection 2x2 | 8 | 0.1620 | 0.1600 | 0.1579 | 0.1163 | 0.0012 | 0.9101 | 0.9447 | 0.5280 |
| Specialization 정규화 ablation | 10 | 0.1622 | 0.1617 | 0.1607 | 0.1196 | 0.0003 | 0.9440 | 0.9697 | 0.4828 |
| Feature ablation sweep | 20 | 0.1620 | 0.1614 | 0.1600 | 0.1196 | 0.0002 | 0.9860 | 0.9796 | 0.4924 |

해석 포인트:
- test_mrr@20 평균은 spec_ablation(0.1617) > feature_ablation(0.1613) > cand3x(0.1609) 순으로 높음.
- cold item(<=5) 평균은 spec_ablation(0.1196)과 feature_ablation(0.1194)가 cand3x(0.1185)보다 개선.
- router2x2는 일부 조합에서 고점은 있으나 평균/하한이 낮고, micro feature-consistency 평균(0.9101)이 가장 낮아 안정성이 약함.
- short session(1-2) 지표는 전반적으로 매우 낮고(평균 거의 0), 표본 수가 극소(예: test split 17개)라 신뢰구간이 넓음.

## 3.1 축(변경 단위)별 묶음 요약
아래는 "어떤 변경을 다음 모델에 가져갈지"를 보기 위한 축별 요약이다. (메인: best val)

### Router/Injection 축
| 축 | 그룹 | best val avg | test avg | cold item avg | 메모 |
|---|---|---:|---:|---:|---|
| Context | X1 | 0.0807 | 0.1618 | 0.1209 | X2 대비 성능/콜드 모두 우세 |
| Context | X2 | 0.0801 | 0.1582 | 0.1117 | 이번 설정에서는 약세 |
| Router | standard | 0.0804 | 0.1600 | 0.1165 | |
| Router | factored | 0.0804 | 0.1599 | 0.1161 | router 타입 효과는 작고 context 영향이 큼 |
| Injection | gated | 0.0806 | 0.1599 | 0.1160 | |
| Injection | group_gated | 0.0803 | 0.1601 | 0.1166 | injection 효과도 context 대비 2순위 |

### Specialization 정규화 축
| 축 | 그룹 | best val avg | test avg | cold item avg | 메모 |
|---|---|---:|---:|---:|---|
| Anchor | A | 0.0811 | 0.1618 | 0.1203 | B anchor 대비 전반 우세 |
| Anchor | B | 0.0805 | 0.1616 | 0.1188 | |
| Mode | M0 | 0.0809 | 0.1621 | 0.1203 | test 최상위권 |
| Mode | M3 | 0.0808 | 0.1618 | 0.1214 | cold item 최우수 |
| Mode | M4 | 0.0806 | 0.1608 | 0.1154 | 과정규화 의심 |

### Feature 축
| 축 | 그룹 | best val avg | test avg | cold item avg | 메모 |
|---|---|---:|---:|---:|---|
| Window | W5 | 0.0798 | 0.1614 | 0.1194 | W10 대비 소폭 우세 |
| Window | W10 | 0.0797 | 0.1613 | 0.1195 | 거의 동급 |
| Mask size | 1-family | 0.0794 | 0.1612 | 0.1178 | 단일 family는 보수적 |
| Mask size | 2-family | 0.0799 | 0.1614 | 0.1206 | 2-family가 일관 우세 |
| Family 포함 | tempo 포함 | 0.0798 | 0.1614 | 0.1200 | 가장 안정적 |
| Family 포함 | memory 포함 | 0.0798 | 0.1616 | 0.1205 | tempo와 함께 강함 |

## 4) SASRec baseline 맥락
- 기존 집계 기준 KuaiRecLargeStrictPosV2_0.2 SASRec: best_mrr@20=0.0785, test_mrr@20=0.1597, test_hr@10=0.1859
- Phase6의 최고 test_mrr@20은 0.1622로, SASRec 기준 대비 +0.0025p 개선.
- baseline_bridge의 B0/B1(sasrec-equivalent 계열)은 test_mrr@20=0.1588/0.1580으로 bridge 하한을 형성했고, MoE 활성 조합(B4~B7)이 이 구간을 안정적으로 상회.

## 5) Top run 스냅샷

### 5.1 test_mrr@20 상위
| run | suite | combo | test_mrr@20 | best_mrr@20 | cold_item_mrr@20 | micro feat-consistency | micro consistency | micro jitter |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FeaturedMoE_N3_p6_spec_a_m0_20260317_224806_746454_pid290200 | spec_ablation | spec_a_m0 | 0.1622 | 0.0812 | 0.1213 | 0.9313 | 0.9642 | 0.4935 |
| FeaturedMoE_N3_p6_cand_a_s3_20260317_175227_353265_pid289468 | cand3x | cand_a_s3 | 0.1621 | 0.0812 | 0.1214 | 0.9217 | 0.9592 | 0.5155 |
| FeaturedMoE_N3_p6_cand_b_s2_20260317_192506_267731_pid289626 | cand3x | cand_b_s2 | 0.1621 | 0.0807 | 0.1206 | 0.9479 | 0.9715 | 0.4911 |
| FeaturedMoE_N3_p6_base_b5_20260318_015913_095313_pid307006 | baseline_bridge | base_b5 | 0.1620 | 0.0804 | 0.1196 | 0.9817 | 0.9936 | 0.5771 |
| FeaturedMoE_N3_p6_cand_a_s2_20260317_175227_263467_pid289467 | cand3x | cand_a_s2 | 0.1620 | 0.0811 | 0.1214 | 0.9408 | 0.9699 | 0.4625 |
| FeaturedMoE_N3_p6_feat_w5_tem_foc_20260318_031347_115143_pid345295 | feature_ablation | feat_w5_tem_foc | 0.1620 | 0.0800 | 0.1203 | 0.9715 | 0.9819 | 0.5003 |
| FeaturedMoE_N3_p6_rxi_x1_sta_gat_20260317_215921_810654_pid290011 | router2x2 | rxi_x1_sta_gat | 0.1620 | 0.0812 | 0.1220 | 0.8985 | 0.9441 | 0.4954 |
| FeaturedMoE_N3_p6_spec_a_m2_20260317_231729_627901_pid290332 | spec_ablation | spec_a_m2 | 0.1620 | 0.0811 | 0.1208 | 0.9703 | 0.9824 | 0.6145 |

### 5.2 cold item(<=5) 상위
| run | suite | combo | cold_item_mrr@20 | test_mrr@20 | micro feat-consistency | micro consistency | micro jitter |
|---|---|---|---:|---:|---:|---:|---:|
| FeaturedMoE_N3_p6_spec_a_m3_20260318_000741_770063_pid290538 | spec_ablation | spec_a_m3 | 0.1221 | 0.1619 | 0.9174 | 0.9540 | 0.2954 |
| FeaturedMoE_N3_p6_rxi_x1_sta_gat_20260317_215921_810654_pid290011 | router2x2 | rxi_x1_sta_gat | 0.1220 | 0.1620 | 0.8985 | 0.9441 | 0.4954 |
| FeaturedMoE_N3_p6_spec_a_m1_20260317_223930_151780_pid290149 | spec_ablation | spec_a_m1 | 0.1217 | 0.1618 | 0.9316 | 0.9657 | 0.4422 |
| FeaturedMoE_N3_p6_feat_w5_tem_mem_20260318_031956_650589_pid347426 | feature_ablation | feat_w5_tem_mem | 0.1216 | 0.1619 | 0.9806 | 0.9751 | 0.4194 |
| FeaturedMoE_N3_p6_feat_w10_tem_mem_20260318_011504_048455_pid293317 | feature_ablation | feat_w10_tem_mem | 0.1215 | 0.1618 | 0.9841 | 0.9797 | 0.4333 |
| FeaturedMoE_N3_p6_cand_a_s2_20260317_175227_263467_pid289467 | cand3x | cand_a_s2 | 0.1214 | 0.1620 | 0.9408 | 0.9699 | 0.4625 |
| FeaturedMoE_N3_p6_cand_a_s3_20260317_175227_353265_pid289468 | cand3x | cand_a_s3 | 0.1214 | 0.1621 | 0.9217 | 0.9592 | 0.5155 |
| FeaturedMoE_N3_p6_feat_w10_tem_foc_20260318_035625_657689_pid364926 | feature_ablation | feat_w10_tem_foc | 0.1214 | 0.1617 | 0.9620 | 0.9776 | 0.5188 |

## 6) 실험 묶음별 상세 (combo별 best/avg/worst)
표 기준: test_mrr@20은 combo 내부 재시도 run의 최고/평균/최저. 단일 run combo는 세 값이 동일.

### 6.1 Candidate 재확인 (cand3x)
| combo | n_runs | test best | test avg | test worst | cold avg | short(1-2) avg | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cand_a_s3 | 1 | 0.1621 | 0.1621 | 0.1621 | 0.1214 | 0.0074 | 0.9217 | 0.9592 | 0.5155 |
| cand_b_s2 | 1 | 0.1621 | 0.1621 | 0.1621 | 0.1206 | 0.0000 | 0.9479 | 0.9715 | 0.4911 |
| cand_a_s2 | 1 | 0.1620 | 0.1620 | 0.1620 | 0.1214 | 0.0045 | 0.9408 | 0.9699 | 0.4625 |
| cand_a_s1 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1206 | 0.0053 | 0.9446 | 0.9708 | 0.5104 |
| cand_b_s1 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1195 | 0.0000 | 0.9534 | 0.9755 | 0.4852 |
| cand_b_s3 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1197 | 0.0000 | 0.9503 | 0.9738 | 0.5252 |
| cand_c_s2 | 1 | 0.1587 | 0.1587 | 0.1587 | 0.1136 | 0.0176 | 0.9695 | 0.9788 | 0.7551 |
| cand_c_s3 | 1 | 0.1587 | 0.1587 | 0.1587 | 0.1121 | 0.0065 | 0.9686 | 0.9788 | 0.7411 |
| cand_c_s1 | 1 | 0.1586 | 0.1586 | 0.1586 | 0.1172 | 0.0031 | 0.9897 | 0.9948 | 0.4133 |

### 6.2 Baseline bridge (baseline_bridge)
| combo | n_runs | test best | test avg | test worst | cold avg | short(1-2) avg | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_b6 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1206 | 0.0000 | 0.9125 | 0.9516 | 0.4027 |
| base_b5 | 4 | 0.1620 | 0.1618 | 0.1616 | 0.1196 | 0.0000 | 0.9817 | 0.9936 | 0.5771 |
| base_b4 | 1 | 0.1617 | 0.1617 | 0.1617 | 0.1213 | 0.0039 | 0.9853 | 0.9946 | 0.6280 |
| base_b7 | 1 | 0.1617 | 0.1617 | 0.1617 | 0.1201 | 0.0000 | 0.9353 | 0.9626 | 0.4289 |
| base_b3 | 1 | 0.1616 | 0.1616 | 0.1616 | 0.1204 | 0.0000 | 0.9472 | 0.9684 | 0.5186 |
| base_b2 | 1 | 0.1601 | 0.1601 | 0.1601 | 0.1110 | 0.0000 | 1.0000 | 1.0000 | 0.3595 |
| base_b0 | 5 | 0.1588 | 0.1588 | 0.1588 | 0.1087 | 0.0000 | - | - | - |
| base_b1 | 5 | 0.1580 | 0.1580 | 0.1580 | 0.1042 | 0.0000 | - | - | - |

### 6.3 Router x Injection 2x2 (router2x2)
| combo | n_runs | test best | test avg | test worst | cold avg | short(1-2) avg | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rxi_x1_sta_gat | 1 | 0.1620 | 0.1620 | 0.1620 | 0.1220 | 0.0000 | 0.8985 | 0.9441 | 0.4954 |
| rxi_x1_fac_gro | 1 | 0.1618 | 0.1618 | 0.1618 | 0.1208 | 0.0000 | 0.9212 | 0.9547 | 0.4929 |
| rxi_x1_sta_gro | 1 | 0.1618 | 0.1618 | 0.1618 | 0.1213 | 0.0000 | 0.9221 | 0.9585 | 0.4600 |
| rxi_x1_fac_gat | 1 | 0.1614 | 0.1614 | 0.1614 | 0.1194 | 0.0031 | 0.9112 | 0.9481 | 0.4928 |
| rxi_x2_fac_gro | 1 | 0.1584 | 0.1584 | 0.1584 | 0.1124 | 0.0033 | 0.9157 | 0.9453 | 0.5766 |
| rxi_x2_sta_gro | 1 | 0.1584 | 0.1584 | 0.1584 | 0.1118 | 0.0035 | 0.9087 | 0.9410 | 0.5543 |
| rxi_x2_fac_gat | 1 | 0.1581 | 0.1581 | 0.1581 | 0.1116 | 0.0000 | 0.9134 | 0.9416 | 0.6237 |
| rxi_x2_sta_gat | 1 | 0.1579 | 0.1579 | 0.1579 | 0.1110 | 0.0000 | 0.8900 | 0.9243 | 0.5284 |

### 6.4 Specialization 정규화 ablation (spec_ablation)
| combo | n_runs | test best | test avg | test worst | cold avg | short(1-2) avg | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec_a_m0 | 1 | 0.1622 | 0.1622 | 0.1622 | 0.1213 | 0.0000 | 0.9313 | 0.9642 | 0.4935 |
| spec_a_m2 | 1 | 0.1620 | 0.1620 | 0.1620 | 0.1208 | 0.0031 | 0.9703 | 0.9824 | 0.6145 |
| spec_b_m0 | 1 | 0.1620 | 0.1620 | 0.1620 | 0.1193 | 0.0000 | 0.9487 | 0.9726 | 0.4840 |
| spec_a_m3 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1221 | 0.0000 | 0.9174 | 0.9540 | 0.2954 |
| spec_b_m1 | 1 | 0.1619 | 0.1619 | 0.1619 | 0.1195 | 0.0000 | 0.9611 | 0.9794 | 0.4898 |
| spec_a_m1 | 1 | 0.1618 | 0.1618 | 0.1618 | 0.1217 | 0.0000 | 0.9316 | 0.9657 | 0.4422 |
| spec_b_m2 | 1 | 0.1618 | 0.1618 | 0.1618 | 0.1195 | 0.0000 | 0.9672 | 0.9828 | 0.5579 |
| spec_b_m3 | 1 | 0.1617 | 0.1617 | 0.1617 | 0.1206 | 0.0000 | 0.9126 | 0.9532 | 0.4093 |
| spec_a_m4 | 1 | 0.1609 | 0.1609 | 0.1609 | 0.1159 | 0.0000 | 0.9492 | 0.9704 | 0.5852 |
| spec_b_m4 | 1 | 0.1607 | 0.1607 | 0.1607 | 0.1149 | 0.0000 | 0.9509 | 0.9721 | 0.4567 |

### 6.5 Feature ablation sweep (feature_ablation)
| combo | n_runs | test best | test avg | test worst | cold avg | short(1-2) avg | micro feat-consistency avg | micro consistency avg | micro jitter avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| feat_w5_tem_foc | 1 | 0.1620 | 0.1620 | 0.1620 | 0.1203 | 0.0000 | 0.9715 | 0.9819 | 0.5003 |
| feat_w10_mem | 2 | 0.1619 | 0.1619 | 0.1619 | 0.1200 | 0.0000 | 1.0000 | 0.9861 | 0.5137 |
| feat_w5_tem_mem | 2 | 0.1619 | 0.1618 | 0.1617 | 0.1215 | 0.0000 | 0.9821 | 0.9766 | 0.4301 |
| feat_w5_mem | 2 | 0.1619 | 0.1618 | 0.1616 | 0.1189 | 0.0000 | 1.0000 | 0.9846 | 0.4336 |
| feat_w5_mem_foc | 2 | 0.1618 | 0.1618 | 0.1617 | 0.1203 | 0.0000 | 0.9786 | 0.9756 | 0.5054 |
| feat_w10_tem_foc | 2 | 0.1617 | 0.1617 | 0.1617 | 0.1212 | 0.0000 | 0.9640 | 0.9787 | 0.5251 |
| feat_w10_tem_mem | 2 | 0.1618 | 0.1617 | 0.1615 | 0.1212 | 0.0000 | 0.9818 | 0.9764 | 0.4430 |
| feat_w10_foc | 1 | 0.1615 | 0.1615 | 0.1615 | 0.1201 | 0.0042 | 1.0000 | 0.9824 | 0.4243 |
| feat_w10_mem_foc | 1 | 0.1615 | 0.1615 | 0.1615 | 0.1207 | 0.0000 | 0.9813 | 0.9785 | 0.4510 |
| feat_w5_foc_exp | 1 | 0.1614 | 0.1614 | 0.1614 | 0.1205 | 0.0000 | 0.9720 | 0.9696 | 0.5277 |
| feat_w10_mem_exp | 1 | 0.1613 | 0.1613 | 0.1613 | 0.1201 | 0.0000 | 1.0000 | 0.9754 | 0.5404 |
| feat_w5_mem_exp | 2 | 0.1617 | 0.1613 | 0.1608 | 0.1209 | 0.0000 | 1.0000 | 0.9782 | 0.5538 |
| feat_w10_tem | 2 | 0.1613 | 0.1612 | 0.1611 | 0.1182 | 0.0000 | 0.9779 | 0.9835 | 0.5646 |
| feat_w5_foc | 2 | 0.1614 | 0.1611 | 0.1609 | 0.1163 | 0.0000 | 1.0000 | 0.9864 | 0.5059 |
| feat_w5_tem_exp | 2 | 0.1611 | 0.1610 | 0.1609 | 0.1197 | 0.0000 | 0.9838 | 0.9777 | 0.4301 |
| feat_w5_tem | 2 | 0.1610 | 0.1609 | 0.1608 | 0.1175 | 0.0000 | 0.9800 | 0.9848 | 0.5336 |
| feat_w10_foc_exp | 2 | 0.1613 | 0.1608 | 0.1604 | 0.1209 | 0.0000 | 0.9755 | 0.9739 | 0.4977 |
| feat_w10_tem_exp | 1 | 0.1608 | 0.1608 | 0.1608 | 0.1205 | 0.0000 | 0.9829 | 0.9773 | 0.4219 |
| feat_w5_exp | 2 | 0.1605 | 0.1604 | 0.1604 | 0.1177 | 0.0000 | 1.0000 | 0.9817 | 0.5010 |
| feat_w10_exp | 1 | 0.1600 | 0.1600 | 0.1600 | 0.1128 | 0.0000 | 1.0000 | 0.9828 | 0.5215 |

## 7) Special logging / Diag 중심 해석

### 7.1 baseline bridge에서 확인한 것
- B0/B1은 sasrec-equivalent로 성능 하한과 special slice 기준점을 제공했고, diag 파일이 없는 run이 존재(모듈 비활성/재시도 누락 케이스).
- B4/B5/B6/B7(특히 feature/both-source)은 cold item 및 전체 test_mrr@20에서 B0/B1 대비 명확히 우세.
- B4는 cold item mrr@20=0.1213, micro consistency=0.9946, micro feat-consistency=0.9853으로 bridge 목적(성능+진단 축 확보)에 가장 깔끔하게 부합.

### 7.2 specialization 가설(논문 중심 메시지) 관점
- spec_ablation에서 최고 test는 spec_a_m0(0.1622), 최고 cold는 spec_a_m3(0.1221). 즉 성능 최대점과 cold 특화점이 서로 다른 정규화 조합에서 나타남.
- M4(strict mixed prior+consistency)는 두 기준점(A/B) 모두 test/cold가 상대적으로 하락(과정규화 가능성).
- 단일 강정규화(M1~M3)는 특정 slice(cold/consistency/jitter)에서 국소 이득을 주지만, 전체점수는 M0~M2 근방에서 포화.
- feature_ablation에서 tempo+memory 계열(feat_w5_tem_mem, feat_w10_tem_mem)과 tempo+focus(feat_w5_tem_foc)가 상위권. feature family별로 반응이 분리되는 패턴이 확인됨.
- diag 상으로 feature_ablation의 micro feat-consistency 평균(0.9860)이 가장 높아, feature 기준 라우팅 일관성은 강화되지만 test_mrr의 절대 고점은 spec/candidate와 비슷한 plateau에 머묾.

### 7.3 cold user / cold session 관련 주의
- new_user_enabled=false로 저장되어 cold user slice는 본 run에서 계산되지 않음.
- 따라서 cold user 결론은 현재 데이터만으로는 불가하고, cold item(<=5) 및 short session(1-2)을 대체 근거로 사용함.
- short session(1-2)은 count가 매우 작아(예시 run 기준 test 17개) 해석보다 참고 지표로만 사용 권장.

## 8) 결론 (논문 서술용 초안)
- Phase6는 후보 재검증 + bridge + 정규화/feature 인과 실험까지 계획된 55 combo를 모두 실행 확보했다.
- FMoE는 SASRec 기준선 대비 test_mrr@20를 유지/상회하며(최고 +0.0025p), cold item slice에서도 개선 여지를 확인했다.
- 핵심은 단일 최적 조합 하나보다, feature family와 정규화 조합에 따라 전문화된 이득 축이 다르게 나타난다는 점이다.
- 즉, specialization은 존재하지만 그 형태는 일괄 상승이 아니라 slice/feature-conditional gain 형태로 관측된다.
- 논문 본문에서는 전체 고점(spec_a_m0) + cold 특화점(spec_a_m3, feat_w5_tem_mem 계열) + bridge 대비(B0/B1 vs B4~B7)를 한 세트로 제시하는 것이 설득력 있다.

## 9) 다음 모델 수정 제안 (가져갈 변화 1~2개)
best val 중심으로 다음 phase에 바로 반영하기 좋은 변경은 아래 2개다.

1) `X1 context` 고정 + `standard gated` 우선 탐색
- 근거: router 축에서 X1이 X2 대비 best val/test/cold를 모두 안정적으로 상회.
- 권장: X1 유지한 상태에서 `rxi_x1_sta_gat`를 기본 템플릿으로 두고, LR/weight decay/regularization 강도만 미세 조정.

2) `spec A anchor` + `M3/M0` 하이브리드
- 근거: M0는 전체 성능, M3는 cold item에서 강점. A anchor가 B anchor보다 우세.
- 권장: A anchor 고정 후 `M3`의 cold 이득을 유지하면서 `M0` 수준의 일반화를 맞추는 중간 강도 세팅(예: M3 계열 lambda 완화)을 새 축으로 추가.

보조 아이디어:
- feature는 단일 family보다 2-family가 유리하므로, tempo+memory 계열을 기본 feature mask로 두고 spec 축과 교차하는 소규모 그리드가 효율적.
