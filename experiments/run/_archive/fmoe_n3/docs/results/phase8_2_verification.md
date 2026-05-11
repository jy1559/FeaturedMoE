# Phase8_2 결과 보고서: 검증 흐름 중심 분석

## 1. 왜 이 실험을 했는가
- 목적: Phase8 후보를 `4(base) x 4(hparam) x 4(seed)`로 재현성 검증.
- 요구 포인트: base별 통합 평균/분산 + 동일 hparam(4 seed) 평균/분산을 분리 보고.

## 2. ID 설명
Base ID
| base_id | base_name | 설정 |
| --- | --- | --- |
| A | B1_phasewise_combo | all_w5 + bias_rule + src_abc_feature + tk_dense |
| B | B2_best_learned_mixed2 | mixed_2 + bias_group_feat + src_base + tk_dense |
| C | B3_clean_all_w5 | all_w5 + bias_off + src_base + tk_dense |
| D | B4_clean_all_w2 | all_w2 + bias_off + src_base + tk_dense |

Hparam ID
| hvar_id | h_name | 설정 |
| --- | --- | --- |
| H1 | baseline | emb128/ff256/expert128/router64/wd1e-6/drop0.15 |
| H2 | capacity_up_light_reg | emb160/ff320/expert160/router80/wd5e-7/drop0.12 |
| H3 | capacity_up_bal_reg | emb160/ff320/expert160/router80/wd2e-6/drop0.18 |
| H4 | compact_strong_reg | emb112/ff224/expert112/router56/wd3e-6/drop0.20 |

## 3.A base 상세 (B1_phasewise_combo)
base 통합(4 hparam x 4 seed = 16 runs)
| base_id | n_runs | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- |
| A | 16 | 0.0811 | 0.000000 | 0.1608 | 0.000002 |

동일 hparam에서 4 seed 평균/분산
| hvar_id | h_name | n | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | baseline | 4 | 0.0810 | 0.000000 | 0.1609 | 0.000001 |
| H2 | capacity_up_light_reg | 4 | 0.0811 | 0.000000 | 0.1596 | 0.000007 |
| H3 | capacity_up_bal_reg | 4 | 0.0812 | 0.000000 | 0.1612 | 0.000000 |
| H4 | compact_strong_reg | 4 | 0.0810 | 0.000000 | 0.1617 | 0.000000 |

base 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_route_consistency_group_knn_score | 16 | -0.7904 | -0.8896 |
| best_valid_mrr20 | diag_route_consistency_knn_score | 16 | -0.8128 | -0.7511 |
| test_mrr20 | sess_3_5_mrr20 | 16 | 0.9467 | 0.9328 |
| test_mrr20 | diag_cv_usage | 16 | -0.9580 | -0.8071 |

## 3.B base 상세 (B2_best_learned_mixed2)
base 통합(4 hparam x 4 seed = 16 runs)
| base_id | n_runs | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- |
| B | 16 | 0.0808 | 0.000000 | 0.1609 | 0.000001 |

동일 hparam에서 4 seed 평균/분산
| hvar_id | h_name | n | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | baseline | 4 | 0.0811 | 0.000000 | 0.1604 | 0.000000 |
| H2 | capacity_up_light_reg | 4 | 0.0805 | 0.000000 | 0.1610 | 0.000002 |
| H3 | capacity_up_bal_reg | 4 | 0.0811 | 0.000000 | 0.1610 | 0.000001 |
| H4 | compact_strong_reg | 4 | 0.0806 | 0.000000 | 0.1613 | 0.000001 |

base 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_entropy_mean | 16 | -0.3760 | -0.6859 |
| best_valid_mrr20 | diag_route_jitter_adjacent | 16 | -0.6063 | -0.6593 |
| test_mrr20 | sess_3_5_mrr20 | 16 | 0.8976 | 0.9270 |
| test_mrr20 | diag_entropy_mean | 16 | 0.8615 | 0.9270 |

## 3.C base 상세 (B3_clean_all_w5)
base 통합(4 hparam x 4 seed = 16 runs)
| base_id | n_runs | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- |
| C | 16 | 0.0811 | 0.000000 | 0.1609 | 0.000001 |

동일 hparam에서 4 seed 평균/분산
| hvar_id | h_name | n | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | baseline | 4 | 0.0810 | 0.000000 | 0.1609 | 0.000001 |
| H2 | capacity_up_light_reg | 4 | 0.0812 | 0.000000 | 0.1601 | 0.000002 |
| H3 | capacity_up_bal_reg | 4 | 0.0814 | 0.000000 | 0.1610 | 0.000001 |
| H4 | compact_strong_reg | 4 | 0.0810 | 0.000000 | 0.1616 | 0.000000 |

base 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_route_consistency_group_knn_score | 16 | -0.6222 | -0.8601 |
| best_valid_mrr20 | diag_route_consistency_knn_score | 16 | -0.6145 | -0.7357 |
| test_mrr20 | sess_3_5_mrr20 | 16 | 0.9692 | 0.9290 |
| test_mrr20 | diag_entropy_mean | 16 | 0.9286 | 0.7545 |

## 3.D base 상세 (B4_clean_all_w2)
base 통합(4 hparam x 4 seed = 16 runs)
| base_id | n_runs | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- |
| D | 16 | 0.0811 | 0.000000 | 0.1608 | 0.000001 |

동일 hparam에서 4 seed 평균/분산
| hvar_id | h_name | n | valid_mean | valid_var | test_mean | test_var |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | baseline | 4 | 0.0814 | 0.000001 | 0.1596 | 0.000003 |
| H2 | capacity_up_light_reg | 4 | 0.0806 | 0.000000 | 0.1615 | 0.000000 |
| H3 | capacity_up_bal_reg | 4 | 0.0812 | 0.000000 | 0.1610 | 0.000000 |
| H4 | compact_strong_reg | 4 | 0.0813 | 0.000000 | 0.1613 | 0.000000 |

base 내부 special/diag 상관(Top)
| target | feature | n | pearson | spearman |
| --- | --- | --- | --- | --- |
| best_valid_mrr20 | diag_route_jitter_adjacent | 16 | -0.8390 | -0.9227 |
| best_valid_mrr20 | diag_n_eff | 16 | -0.8153 | -0.7992 |
| test_mrr20 | diag_n_eff | 16 | 0.9424 | 0.9299 |
| test_mrr20 | diag_cv_usage | 16 | -0.9620 | -0.9299 |

## 4. 최종 정리(전체 64 runs)
base별 비교
| base_id | base_name | n | valid_mean | valid_std | test_mean | test_std |
| --- | --- | --- | --- | --- | --- | --- |
| A | B1_phasewise_combo | 16 | 0.0811 | 0.000233 | 0.1608 | 0.001533 |
| B | B2_best_learned_mixed2 | 16 | 0.0808 | 0.000508 | 0.1609 | 0.000922 |
| C | B3_clean_all_w5 | 16 | 0.0811 | 0.000310 | 0.1609 | 0.001053 |
| D | B4_clean_all_w2 | 16 | 0.0811 | 0.000529 | 0.1608 | 0.001224 |

hparam별 비교
| hvar_id | h_name | n | valid_mean | valid_std | test_mean | test_std |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | baseline | 16 | 0.0811 | 0.000525 | 0.1605 | 0.001226 |
| H2 | capacity_up_light_reg | 16 | 0.0809 | 0.000386 | 0.1606 | 0.001731 |
| H3 | capacity_up_bal_reg | 16 | 0.0812 | 0.000389 | 0.1610 | 0.000661 |
| H4 | compact_strong_reg | 16 | 0.0810 | 0.000300 | 0.1615 | 0.000433 |

최종 해석
- 이 phase는 큰 성능 차를 만드는 단계보다, 후보의 재현성과 분산 프로파일을 제공하는 단계로 기능함.
- 논문 서술에서는 "고점 후보"와 "안정 후보"를 분리 제시하는 구조가 적합.