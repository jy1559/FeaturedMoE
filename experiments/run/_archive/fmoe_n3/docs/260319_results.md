# 260319 KuaiRec Results Summary (Baseline vs FMoE_N3)

## 범위
- 데이터셋: KuaiRecLargeStrictPosV2_0.2
- 사용 소스:
  - experiments/run/artifacts/results/baseline/*.json
  - experiments/run/artifacts/results/fmoe_n3/*.json
  - experiments/run/artifacts/results/fmoe_n3/normal/**.json
  - experiments/run/artifacts/results/fmoe_n3/sidecar/phase4_residual_topk_v2/**.summary.json
- 본 문서는 special/diag 상세보다 일반 실험 결과(주요 metric) 중심으로 정리했다.

## 1) Baseline 요약 (모델별 best 1개)
기준: 각 모델에서 valid MRR@20 최고 run 1개 선택.

| Model | Best Valid MRR@20 | Best Valid HR@10 | Test MRR@20 (same run) | Test HR@10 (same run) |
|---|---:|---:|---:|---:|
| SIGMA | 0.0800 | 0.1087 | 0.1590 | 0.1836 |
| FAME | 0.0792 | 0.1080 | 0.1586 | 0.1827 |
| PAtt | 0.0790 | 0.1081 | 0.1585 | 0.1828 |
| BSARec | 0.0789 | 0.1090 | 0.1560 | 0.1839 |
| SASRec | 0.0785 | 0.1099 | 0.1597 | 0.1859 |
| FENRec | 0.0778 | 0.1061 | 0.1584 | 0.1817 |
| GRU4Rec | 0.0775 | 0.1043 | 0.1553 | 0.1793 |
| SRGNN | 0.0694 | 0.0981 | 0.1411 | 0.1707 |

### Baseline metric별 winner (전체 baseline run 기준)
| Metric | Winner | Value |
|---|---|---:|
| best_valid_mrr20 | SIGMA | 0.0800 |
| best_valid_hr10 | SASRec | 0.1099 |
| best_valid_test_mrr20 | SASRec | 0.1597 |
| best_valid_test_hr10 | SASRec | 0.1860 |

해석:
- KuaiRec baseline에서 valid MRR@20은 SIGMA가 근소 우세.
- 하지만 test MRR@20, test HR@10은 SASRec run이 가장 높아, 발표에서는 SASRec을 강한 baseline anchor로 두는 구성이 자연스럽다.

## 2) FMoE_N3 핵심 체크포인트

### (A) 간단 초기 실험 (phase1 proxy)
요청 취지에 맞춰 phase1의 a01~a06 계열 중 best(valid) 선택:

| Checkpoint | Run ID | Best Valid MRR@20 | Test MRR@20 | Test HR@10 |
|---|---|---:|---:|---:|
| FMoE_N3 phase1 simple best | a05_20260314_065556_106453_pid264454 | 0.0811 | 0.1620 | 0.1889 |

### (B) 현재 phase6 candidate 중 best(valid)
요청 취지에 맞춰 phase6 candidate 영역에서 valid MRR@20 최고 run 선택:

| Checkpoint | Run ID | Best Valid MRR@20 | Test MRR@20 | Test HR@10 |
|---|---|---:|---:|---:|
| FMoE_N3 phase6 candidate best(valid) | p6_cand_c_s1_20260317_194731_518128_pid289680 | 0.0818 | 0.1586 | 0.1818 |

### (C) 참고: KuaiRec 전체 FMoE_N3 최고권
- phase4 축 실험 구간에서 valid MRR@20 0.0814 수준의 run들이 다수 존재.
- 즉 "phase6 candidate best(valid)"가 반드시 "전체 최고 test 성능"을 의미하진 않는다.
- 발표에서는 의도적으로 아래처럼 분리해서 말하는 것이 안전하다.
  - "탐색 목적 최고(valid)"
  - "최종 test 지표 최고"

## 3) Axis-wise Ablation Facets (Phase4 중심)
아래는 phase4 sidecar summary 기준 축별 집계다.
- `best_valid_mrr20_max`: 해당 axis variation에서 최고 run
- `best_valid_mrr20_mean`: 해당 axis variation 평균
- `reliability`: 반복 수(n) 기반 단순 신뢰도 표시

### R-axis (Residual/Sharing 계열)
| variation | n | max(valid MRR@20) | mean(valid MRR@20) | mean(test MRR@20) | reliability |
|---|---:|---:|---:|---:|---|
| base | 8 | 0.0814 | 0.08126 | 0.16233 | high |
| shared_only | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| shared_moe_fixed03 | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| shared_moe_fixed05 | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| shared_moe_global | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| shared_moe_stage | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| shared_moe_warmup | 4 | 0.0812 | 0.08120 | 0.16225 | high |

해석:
- R-axis에서는 base가 아주 근소하게 높고, 나머지 변형은 거의 동률이다.
- 즉, 이 축만으로 큰 성능 점프를 기대하기 어렵고, 안정성/해석성 관점으로 선택할 항목이다.

### K-axis (Top-k/Expert grouping 계열)
| variation | n | max(valid MRR@20) | mean(valid MRR@20) | mean(test MRR@20) | reliability |
|---|---:|---:|---:|---:|---|
| group_dense | 4 | 0.0814 | 0.08140 | 0.16228 | high |
| group_top1 | 4 | 0.0814 | 0.08140 | 0.16228 | high |
| group_top2 | 4 | 0.0814 | 0.08140 | 0.16228 | high |
| groupignore_global6 | 4 | 0.0814 | 0.08140 | 0.16228 | high |
| 12e_top6 | 4 | 0.0814 | 0.08135 | 0.16228 | high |
| 12e_top3 | 4 | 0.0814 | 0.08133 | 0.16228 | high |
| 12e_dense | 4 | 0.0813 | 0.08125 | 0.16225 | high |
| 4e_top2 | 4 | 0.0813 | 0.08122 | 0.16225 | high |
| 4e_dense | 4 | 0.0812 | 0.08120 | 0.16225 | high |
| 4e_top1 | 4 | 0.0812 | 0.08120 | 0.16225 | high |

해석:
- K-axis에서는 group 계열이 소폭 우세하지만 절대 차이는 작다.
- "dense/top1/top2/groupignore" 간 성능 차가 거의 없으므로, 계산량/안정성 우선으로 선택 가능하다.

### F-axis (Feature injection variants)
| variation | n | max(valid MRR@20) | mean(test MRR@20) | reliability |
|---|---:|---:|---:|---|
| feat_full | 1 | 0.0814 | 0.1622 | low |
| feat_hidden_only | 1 | 0.0814 | 0.1623 | low |
| feat_feature_only | 1 | 0.0814 | 0.1623 | low |
| feat_injection_only | 1 | 0.0814 | 0.1623 | low |

해석:
- F-axis는 반복 수가 1이라 통계적 신뢰도가 낮다.
- 슬라이드에는 "탐색적 결과 (low reliability)" 라벨을 붙이는 것을 권장.

## 4) 발표용 메시지 템플릿
- baseline anchor(SASRec) 대비 FMoE_N3는 초기(simple phase1)부터 valid MRR@20 기준 개선이 관찰됨.
- phase6 candidate는 탐색 valid 기준으로는 최고점(0.0818)을 만들었지만, test 관점에서는 일부 earlier 축 실험과 역전 가능성이 있음.
- axis-wise는 K/R 축에서 성능 격차가 작고, F 축은 현재 표본 부족.

## 5) 신뢰도 낮은 부분과 추가 실험 제안
현재 신뢰도 낮음:
- F-axis: 각 variation당 n=1
- phase6 해석: valid-best와 test-best가 다를 수 있음

추가 권장 실험(슬라이드 보강용):
1. F-axis 4개 variation에 대해 동일 seed 3회 반복 (총 12 run)로 평균±표준편차 표시
2. phase6 candidate 상위 3개(run_id 기준) 재평가: seed 고정 재실행 + 동일 eval 스크립트로 test 안정화
3. R/K 축은 차이가 작으므로, 성능 외 지표(학습시간, OOM율, GPU 사용량)를 보조축으로 같이 제시

---

원본 집계 파일:
- experiments/run/fmoe_n3/docs/data/260319_baseline_kuairec_model_best.csv
- experiments/run/fmoe_n3/docs/data/260319_baseline_kuairec_metric_winners.csv
- experiments/run/fmoe_n3/docs/data/260319_fmoe_n3_kuairec_all.csv
- experiments/run/fmoe_n3/docs/data/260319_phase4_axis_agg.csv
- experiments/run/fmoe_n3/docs/data/260319_phase4_axis_rows.csv
- experiments/run/fmoe_n3/docs/data/260319_summary.json
